import sys
sys.path.append('.')
import argparse
import os
import shutil

import torch
from transformers import BartForCausalLM, GenerationConfig, set_seed, LogitsProcessor, LogitsProcessorList
from tqdm.auto import tqdm
import numpy as np

from src.model.tokenizer import WhitespaceTokenizer
from src.utils.misc import load_config, get_logger, save_smileslist


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



def load_model(model_path):
    model = BartForCausalLM.from_pretrained(model_path)
    return model


def generate_smiles(model, tokenizer, config):
    """Use model to generate SMILES strings and output scores."""
    generated_smiles = []
    generated_scores = []

    bos_token_id = model.config.bos_token_id
    scale_times = getattr(config, 'scale_times', 1)

    # In this example, vocabulary range [0..459] is divided into two groups:
    #   group A: [0..99)   -> use upscale_temp
    #   group B: [99..460) -> use downscale_temp
    group_ranges = [(0, 99), (99, 460)]
    group_temps = [
        config.get('upscale_temp', 1.2),
        config.get('downscale_temp', 0.8)
    ]

    # Instantiate new processor
    custom_processor = GroupProbPreservingTempProcessor(
        group_ranges=group_ranges,
        group_temps=group_temps,
        device='cuda',
        min_prob=1e-9
    )
    custom_logits_processors = LogitsProcessorList([custom_processor])

    print(scale_times * config['num_samples'])
    for i in tqdm(range(0, scale_times * config['num_samples'], config['batch_size'])):
        input_ids = torch.full(
            (config['batch_size'], 1),
            bos_token_id,
            dtype=torch.long
        ).to('cuda')

        outputs = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(**config['generation_config']),
            logits_processor=custom_logits_processors,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_ids = outputs.sequences
        gen_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_smiles.extend(gen_smiles)

        # Calculate sequence scores
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = GenerationConfig(**config['generation_config']).length_penalty
        true_scores = transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty)

        generated_scores.extend(true_scores)

    return generated_smiles, generated_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/unconditional_generation.yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Set output path and save configuration file
    os.makedirs(config.save_path, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(config.save_path, 'config.yaml'))

    # Set up logger
    logger = get_logger("SMILES_Generation", config.save_path)
    logger.info('---------------------------------------------')
    logger.info("config loaded: \n" + str(config))

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = WhitespaceTokenizer()

    # Load model
    logger.info(f"Loading model from path: {config.model_path}")
    model = load_model(config.model_path)
    model.eval()
    model.to('cuda')

    # Generate SMILES for each seed
    for seed_i in range(getattr(config, 'sample_times', 1)):
        logger.info('*****************************************')
        run_save_path = os.path.join(config.save_path, f'run_{seed_i}')
        os.makedirs(run_save_path, exist_ok=True)

        logger.info(f'Run {seed_i + 1}/{config.sample_times} with seed {config.seed + seed_i}')
        set_seed(config.seed + seed_i)

        # Generate SMILES
        logger.info("Generating SMILES...")
        generated_smiles, generated_scores = generate_smiles(model, tokenizer, config)

        # only save the top num_samples
        combined = list(zip(generated_smiles, generated_scores))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        top_smiles_scores = combined_sorted[:config.num_samples]

        generated_smiles = [x[0] for x in top_smiles_scores]
        generated_scores = [x[1] for x in top_smiles_scores]

        # Save generated SMILES
        smiles_file = os.path.join(run_save_path, 'generated_smiles.txt')
        logger.info(f"Saving generated SMILES to {smiles_file}...")
        save_smileslist(generated_smiles, smiles_file)

        # Save scores
        score_file = os.path.join(run_save_path, 'generated_scores.txt')
        logger.info(f"Saving generation scores to {score_file}...")
        with open(score_file, 'w', encoding='utf-8') as f:
            for s in generated_scores:
                f.write(f"{s}\n")

        logger.info(f'Finished generating {len(generated_smiles)} SMILES\n')


if __name__ == '__main__':
    main()
