from accelerate import Accelerator
import sys
sys.path.append('./')
import argparse
import os
import shutil

import torch
from torch.utils.data import DataLoader
from transformers import set_seed, LogitsProcessor, LogitsProcessorList
from tqdm.auto import tqdm
import numpy as np

from src.model.dataset import PointCloudDataset, PointCloudCollator
from src.model.tokenizer import WhitespaceTokenizer
from src.model.SurfBart import SurfaceBartv2
from src.utils.misc import load_config, save_smileslist, get_logger, save_scores


class GroupProbPreservingTempProcessor(LogitsProcessor):
    def __init__(self,
                 group_ranges,
                 group_temps,
                 device='cuda',
                 min_prob=1e-9):
        super().__init__()
        self.group_ranges = group_ranges
        self.group_temps = group_temps
        self.device = device
        self.min_prob = min_prob

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1) First convert original logits to probability distribution
        probs = torch.softmax(scores, dim=-1)  # [batch_size, vocab_size]
        new_probs = torch.zeros_like(probs)

        batch_size, vocab_size = probs.shape

        # For each group, first calculate the probability sum of that group in the original distribution, then apply temperature transformation
        for (start, end), T in zip(self.group_ranges, self.group_temps):
            if start >= vocab_size:
                continue
            local_end = min(end, vocab_size)

            group_probs = probs[:, start:local_end]  # [batch_size, group_size]
            group_sum = group_probs.sum(dim=-1, keepdim=True)  # [batch_size, 1]

            # Create a mask to judge whether each row is > 0
            # shape = [batch_size], True means this row's group_sum > 0
            nonzero_mask = (group_sum.squeeze(-1) > 0)

            # If this group's probability sum is 0, no need for temperature transformation, directly set to 0
            if nonzero_mask.any():
                # Only operate on rows where nonzero_mask==True
                group_probs_nonzero = group_probs[nonzero_mask]             # [N, group_size]
                group_sum_nonzero   = group_sum[nonzero_mask].squeeze(-1)   # [N]

                # 1) First normalize within group: p_i / sum
                #    group_probs_normalized = [N, group_size]
                group_probs_normalized = group_probs_nonzero / group_sum_nonzero.unsqueeze(-1)

                # 2) Apply temperature in probability domain: (p^(1/T)) / ∑(p^(1/T))
                group_probs_normalized = group_probs_normalized.clamp_min(self.min_prob)
                p_temp = group_probs_normalized.pow(1.0 / T)  # [N, group_size]
                denom = p_temp.sum(dim=-1, keepdim=True).clamp_min(self.min_prob)
                p_temp = p_temp / denom  # Normalize again

                # 3) Multiply back by original probability sum within group, keeping total sum between groups unchanged
                final_group_probs = p_temp * group_sum_nonzero.unsqueeze(-1)

                # Write back to new_probs
                new_probs[nonzero_mask, start:local_end] = final_group_probs

            # If this group is all 0, keep new_probs[:, start:local_end] = 0

        # Finally convert new_probs back to logits
        new_probs = new_probs.clamp_min(self.min_prob)
        new_scores = torch.log(new_probs)
        return new_scores
    

def load_model(model_path, train_config, tokenizer):
    """Load and return a SurfaceBart model based on the provided path."""
    model = SurfaceBartv2(train_config['model'], tokenizer=tokenizer)
    model.load_weights(model_path)
    return model

def prepare_dataloader(data_config, tokenizer, max_length, batch_size):
    """Prepare a DataLoader for batch processing."""
    dataset = PointCloudDataset(data_config, tokenizer, max_length=max_length)
    collator = PointCloudCollator()
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

def generate_outputs(model, dataloader, tokenizer, generation_config, upscale_temp=1.0, downscale_temp=1.0):
    """Generate outputs using the model and DataLoader."""
    group_ranges = [(0, 99), (99, 460)]
    group_temps = [upscale_temp, downscale_temp]  # Temperature can be adjusted as needed
        # Instantiate new processor
    custom_processor = GroupProbPreservingTempProcessor(
        group_ranges=group_ranges,
        group_temps=group_temps,
        device='cuda',
        min_prob=1e-9
    )
    custom_logits_processors = LogitsProcessorList([custom_processor])

    generated_smiles, generated_scores = [], []
    model.eval()  # Set the model to evaluation mode

    for batch in tqdm(dataloader):
        batch = {key: value for key, value in batch.items()}
        generated_ids, scores = model.sample(
            pointcloud=batch['pointcloud'],
            normals=batch['normals'],
            attention_mask=None,
            generation_config=generation_config,
            logits_processor=custom_logits_processors,
            prefix_ids=[tokenizer.encode('<BOS>')[1], tokenizer.encode('<std>')[1]],       
        )

        gen_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        gen_smiles = [smiles.lstrip('<aug>').lstrip('<std>').lstrip() for smiles in gen_smiles]
        generated_smiles.extend(gen_smiles)
        generated_scores.extend(scores)

    return generated_smiles, generated_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/surfbart_generation.yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Set output path and save configuration file
    os.makedirs(config['save_path'], exist_ok=True)
    shutil.copy(args.config_path, os.path.join(config['save_path'], 'config.yaml'))

    # Set up logger in the save path
    logger = get_logger("SurfaceBart_Generation", config['save_path'])
    logger.info('---------------------------------------------')
    logger.info("Generation Config loaded: \n" + str(config))

    # load train config
    train_config = load_config(os.path.join(os.path.dirname(config['model_path']), 'config.yaml'))
    logger.info(f"Train config loaded from {os.path.join(os.path.dirname(config['model_path']), 'config.yaml')}")

    # Set seed
    logger.info(f"Setting seed to: {config['seed']}")
    set_seed(config['seed'])

    # Initialize Accelerate
    logger.info("Initializing Accelerate...")
    accelerator = Accelerator()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = WhitespaceTokenizer()

    # Load model
    logger.info(f"Loading model from path: {config['model_path']}")
    model = load_model(config['model_path'], train_config, tokenizer)

    # Prepare DataLoader
    logger.info("Preparing DataLoader...")
    dataloader = prepare_dataloader(config['data'], 
                                    tokenizer, 
                                    train_config['model']['bart']['max_position_embeddings'], 
                                    config['batch_size'])

    # Prepare model and dataloader for Accelerate
    model, dataloader = accelerator.prepare(model, dataloader)

    # Generate outputs
    logger.info("Generating outputs...")
    generated_smiles, generated_scores = generate_outputs(model, 
                                                          dataloader, 
                                                          tokenizer, 
                                                          config['generation_config'], 
                                                          config['upscale_temp'],
                                                          config['downscale_temp'])

    # Save generated outputs
    logger.info(f"Saving generated outputs to {os.path.join(config['save_path'], 'generated_smiles.txt')}...")
    save_smileslist(generated_smiles, os.path.join(config['save_path'], 'generated_smiles.txt'))
    logger.info(f"Finished generating {len(generated_smiles)} outputs.")

    # Save generated scores
    logger.info(f"Saving generated scores to {os.path.join(config['save_path'], 'generated_scores.txt')}...")
    save_scores(generated_scores, os.path.join(config['save_path'], 'generated_scores.txt'))
    logger.info(f"Finished saving {len(generated_scores)} scores.")


if __name__ == '__main__':
    main()
