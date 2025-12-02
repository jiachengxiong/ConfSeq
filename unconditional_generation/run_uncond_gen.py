# generate_and_reconstruct_cli.py
import sys
sys.path.append('.')
import argparse
import os
import shutil

import torch
from transformers import (
    BartForCausalLM,
    GenerationConfig,
    set_seed,
    LogitsProcessor,
    LogitsProcessorList,
)
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np

from rdkit.Chem import SDWriter
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')

from src.model.tokenizer import WhitespaceTokenizer
from src.utils.misc import load_config, get_logger, save_smileslist
from src.utils.reconstruct import convert_tdsmiles_to_mol


class GroupProbPreservingTempProcessor(LogitsProcessor):
    """
    A logits processor that applies temperature scaling within pre-defined
    token index groups, while preserving the total probability mass of
    each group.

    The idea:
    - Convert logits -> probs
    - For each group [start, end):
        1) Compute group total probability (per batch row)
        2) Normalize within the group
        3) Apply temperature in probability space: p^(1/T) / sum(p^(1/T))
        4) Multiply back by original group probability sum
    - Convert the modified probabilities back to logits.
    """

    def __init__(
        self,
        group_ranges,
        group_temps,
        device='cuda',
        min_prob: float = 1e-9,
    ):
        """
        Args:
            group_ranges: list of (start, end) index pairs, end is exclusive.
            group_temps: list of temperatures, same length as group_ranges.
            device: not strictly needed here, but kept for compatibility.
            min_prob: minimal probability for numerical stability.
        """
        super().__init__()
        self.group_ranges = group_ranges
        self.group_temps = group_temps
        self.device = device
        self.min_prob = min_prob

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)  # [batch_size, vocab_size]
        new_probs = torch.zeros_like(probs)

        batch_size, vocab_size = probs.shape

        # Process each group independently
        for (start, end), T in zip(self.group_ranges, self.group_temps):
            if start >= vocab_size:
                # If group start is beyond the vocab, skip
                continue
            local_end = min(end, vocab_size)

            group_probs = probs[:, start:local_end]  # [batch_size, group_size]
            group_sum = group_probs.sum(dim=-1, keepdim=True)  # [batch_size, 1]

            # A mask indicating which rows have non-zero mass in this group
            nonzero_mask = (group_sum.squeeze(-1) > 0)

            if nonzero_mask.any():
                # Only operate on rows with non-zero group probability
                group_probs_nonzero = group_probs[nonzero_mask]          # [N, group_size]
                group_sum_nonzero = group_sum[nonzero_mask].squeeze(-1)  # [N]

                # Step 1: normalize within the group
                group_probs_normalized = group_probs_nonzero / group_sum_nonzero.unsqueeze(-1)

                # Step 2: apply temperature in probability space
                group_probs_normalized = group_probs_normalized.clamp_min(self.min_prob)
                p_temp = group_probs_normalized.pow(1.0 / T)
                denom = p_temp.sum(dim=-1, keepdim=True).clamp_min(self.min_prob)
                p_temp = p_temp / denom  # renormalize

                # Step 3: multiply back by the original group sum
                final_group_probs = p_temp * group_sum_nonzero.unsqueeze(-1)

                # Write back to new_probs
                new_probs[nonzero_mask, start:local_end] = final_group_probs

            # Rows where group_sum == 0 remain zero in new_probs

        # Convert probabilities back to logits
        new_probs = new_probs.clamp_min(self.min_prob)
        new_scores = torch.log(new_probs)
        return new_scores


def load_model(model_path: str) -> BartForCausalLM:
    """
    Load a BartForCausalLM model from the given path or Hugging Face repo.

    model_path 可以是：
      - 本地路径（包含 config.json / pytorch_model.bin 等）
      - Hugging Face Hub 上的 repo 名称（例如 'Oopstom/confseq-conf-gen'）
    """
    model = BartForCausalLM.from_pretrained(model_path)
    return model


def build_generation_config(config):
    """
    Build a GenerationConfig object from the nested config field.
    Supports both dict-like and attribute-style access.
    """
    if isinstance(config, dict):
        gen_cfg_dict = config['generation_config']
    else:
        gen_cfg_dict = config.generation_config
    return GenerationConfig(**gen_cfg_dict)


def generate_smiles(
    model: BartForCausalLM,
    tokenizer: WhitespaceTokenizer,
    config,
    device: str = 'cuda',
    group_split: int = 99,
):
    """
    Use the model to generate SMILES strings with group-wise temperature scaling.

    The function:
    - Builds a logits processor with two token index groups:
        [0, group_split)      -> upscale_temp
        [group_split, vocab)  -> downscale_temp
    - Generates scale_times * num_samples sequences
    - Scores each sequence using transition scores
    - Returns the list of generated SMILES and their scores
    """
    generated_smiles = []
    generated_scores = []

    # BOS token id; fall back to decoder_start_token_id if BOS is not set
    bos_token_id = model.config.bos_token_id
    if bos_token_id is None:
        bos_token_id = model.config.decoder_start_token_id

    # Number of "over-generation" times before ranking and truncation
    scale_times = getattr(config, 'scale_times', 1)

    # Determine vocab size; either from tokenizer or model config
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        vocab_size = model.config.vocab_size

    # Build group ranges: two groups, split at group_split
    group_ranges = [
        (0, group_split),
        (group_split, vocab_size),
    ]

    # Read temperatures from config (or use defaults)
    upscale_temp = getattr(config, 'upscale_temp', None)
    if hasattr(config, 'get'):
        upscale_temp = config.get('upscale_temp', 1.2) if upscale_temp is None else upscale_temp
    if upscale_temp is None:
        upscale_temp = 1.2

    downscale_temp = getattr(config, 'downscale_temp', None)
    if hasattr(config, 'get'):
        downscale_temp = config.get('downscale_temp', 0.8) if downscale_temp is None else downscale_temp
    if downscale_temp is None:
        downscale_temp = 0.8

    group_temps = [upscale_temp, downscale_temp]

    # Instantiate custom logits processor
    custom_processor = GroupProbPreservingTempProcessor(
        group_ranges=group_ranges,
        group_temps=group_temps,
        device=device,
        min_prob=1e-9,
    )
    custom_logits_processors = LogitsProcessorList([custom_processor])

    # Build GenerationConfig and get length penalty
    gen_config = build_generation_config(config)
    length_penalty = gen_config.length_penalty

    total_to_generate = scale_times * config.num_samples
    print(f"Total sequences to generate (before ranking): {total_to_generate}")

    # Loop over mini-batches
    for i in tqdm(range(0, total_to_generate, config.batch_size)):
        batch_size = config.batch_size

        # Prepare BOS-only input batch
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
        ).to(device)

        # Generate sequences with custom logits processor
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=gen_config,
            logits_processor=custom_logits_processors,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences
        gen_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_smiles.extend(gen_smiles)

        # Compute transition scores for each generated token
        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )  # shape: [batch, seq_len]

        # Count the number of valid (negative log-prob) positions per sequence
        np_scores = transition_scores.cpu().numpy()
        output_length = np.sum(np_scores < 0, axis=1)

        # Apply length penalty for final sequence score
        true_scores = transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty)
        generated_scores.extend(true_scores)

    return generated_smiles, generated_scores


def reconstruct_mols_from_smiles(
    smiles_list,
    sdf_out_path: str,
    chunksize: int = 16,
    num_workers: int = 4,
    logger=None,
):
    """
    Convert a list of ConfSeq into 3D molecules and write them to an SDF file.

    Args:
        smiles_list: list of ConfSeq strings.
        sdf_out_path: path to the output SDF file.
        chunksize: chunk size for process_map parallelization.
        num_workers: number of worker processes.
        logger: optional logger for information messages.

    Returns:
        success_count: number of successfully reconstructed molecules.
    """
    if logger is not None:
        logger.info('Converting %d ConfSeq to molecules...', len(smiles_list))

    writer = SDWriter(sdf_out_path)

    # Run conversion in parallel; process_map provides a tqdm progress bar
    results = process_map(
        convert_tdsmiles_to_mol,
        smiles_list,
        chunksize=chunksize,
        max_workers=num_workers,
    )

    success = 0
    for result in results:
        # We assume convert_tdsmiles_to_mol returns either:
        # - an RDKit Mol object, or
        # - a string starting with 'Error:' indicating failure
        if isinstance(result, str) and result.startswith('Error:'):
            continue
        elif result is not None:
            writer.write(result)
            success += 1

    writer.close()

    if logger is not None:
        logger.info(
            'Successfully reconstructed %d molecules out of %d ConfSeq.',
            success,
            len(smiles_list),
        )

    return success


def parse_args():
    """
    Parse command-line arguments for the combined script:
    1) generate ConfSeq using a Bart model
    2) immediately reconstruct them into molecules (SDF)
    """
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: generate ConfSeq and reconstruct them into 3D molecules."
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='./configs/unconditional_generation.yaml',
        help='Path to YAML config file used as defaults.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='Oopstom/confseq-uncond-gen',
        help=(
            "Model path or Hugging Face repo name. "
            "Default: 'Oopstom/confseq-uncond-gen' (will be auto-downloaded from HF Hub)."
        ),
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Override output directory for runs and SDF files (otherwise taken from config).',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of ConfSeq to keep per run after ranking.',
    )
    parser.add_argument(
        '--sample_times',
        type=int,
        default=None,
        help='Number of independent sampling runs (different seeds).',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size used during generation.',
    )
    parser.add_argument(
        '--scale_times',
        type=int,
        default=None,
        help='Over-generation factor; total generated = scale_times * num_samples.',
    )
    parser.add_argument(
        '--upscale_temp',
        type=float,
        default=None,
        help='Temperature used for the first token group [0, group_split).',
    )
    parser.add_argument(
        '--downscale_temp',
        type=float,
        default=None,
        help='Temperature used for the second token group [group_split, vocab).',
    )
    parser.add_argument(
        '--group_split',
        type=int,
        default=99,
        help='Vocabulary split index: [0, group_split) and [group_split, vocab).',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device for inference, e.g., 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=None,
        help='Chunksize passed to process_map for parallel reconstruction.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of worker processes for reconstruction.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration as a base (YAML / dict / EasyDict)
    config = load_config(args.config_path)

    config.model_path = args.model_path

    if args.save_path is not None:
        config.save_path = args.save_path
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.sample_times is not None:
        config.sample_times = args.sample_times
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.scale_times is not None:
        config.scale_times = args.scale_times
    if args.upscale_temp is not None:
        config.upscale_temp = args.upscale_temp
    if args.downscale_temp is not None:
        config.downscale_temp = args.downscale_temp

    if args.chunksize is not None:
        config.chunksize = args.chunksize
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Ensure some default fields exist
    if not hasattr(config, 'sample_times'):
        config.sample_times = 1
    if not hasattr(config, 'batch_size'):
        config.batch_size = 32
    if not hasattr(config, 'chunksize'):
        config.chunksize = 16
    if not hasattr(config, 'num_workers'):
        config.num_workers = 4
    if not hasattr(config, 'seed'):
        config.seed = 42

    # Create output directory and copy the (original) config file
    os.makedirs(config.save_path, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(config.save_path, 'config.yaml'))

    # Set up logger
    logger = get_logger("Generate_and_Reconstruct", config.save_path)
    logger.info('---------------------------------------------')
    logger.info("Loaded config (with CLI overrides):\n%s", str(config))

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = WhitespaceTokenizer()

    # Load model
    logger.info("Loading model from path or HF repo: %s", config.model_path)
    model = load_model(config.model_path)
    model.eval()
    model.to(args.device)

    # Run multiple sampling + reconstruction rounds with different seeds
    for seed_i in range(config.sample_times):
        logger.info('*****************************************')
        run_save_path = os.path.join(config.save_path, f'run_{seed_i}')
        os.makedirs(run_save_path, exist_ok=True)

        current_seed = config.seed + seed_i
        logger.info(
            'Run %d / %d with seed %d',
            seed_i + 1,
            config.sample_times,
            current_seed,
        )
        set_seed(current_seed)

        # 1) Generation step: ConfSeq
        logger.info("Generating ConfSeq...")
        generated_smiles, generated_scores = generate_smiles(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=args.device,
            group_split=args.group_split,
        )

        # Rank sequences by score and keep only top num_samples
        combined = list(zip(generated_smiles, generated_scores))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        top_smiles_scores = combined_sorted[:config.num_samples]

        generated_smiles = [x[0] for x in top_smiles_scores]
        generated_scores = [x[1] for x in top_smiles_scores]

        # Save ConfSeq
        smiles_file = os.path.join(run_save_path, 'generated_smiles.txt')
        logger.info("Saving generated ConfSeq to %s", smiles_file)
        save_smileslist(generated_smiles, smiles_file)

        # Save scores
        score_file = os.path.join(run_save_path, 'generated_scores.txt')
        logger.info("Saving generation scores to %s", score_file)
        with open(score_file, 'w', encoding='utf-8') as f:
            for s in generated_scores:
                f.write(f"{float(s)}\n")

        logger.info(
            'Finished generating %d ConfSeq for run %d. Starting reconstruction...',
            len(generated_smiles),
            seed_i,
        )

        # 2) Reconstruction step: ConfSeq -> molecules
        sdf_out_path = os.path.join(run_save_path, 'generated_mols.sdf')
        reconstruct_mols_from_smiles(
            smiles_list=generated_smiles,
            sdf_out_path=sdf_out_path,
            chunksize=config.chunksize,
            num_workers=config.num_workers,
            logger=logger,
        )

        logger.info('Run %d completed.\n', seed_i)


if __name__ == '__main__':
    main()
