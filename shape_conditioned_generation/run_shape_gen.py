#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ConfSeq / SurfBART-v2 shape-based molecule generation · Command-line interface
"""

import os
import sys
sys.path.append('../')
import gc
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import pandas as pd
from accelerate import Accelerator
from rdkit import Chem, RDLogger
from rdkit.Chem.PropertyMol import PropertyMol
from transformers import set_seed
from tqdm.contrib.concurrent import process_map

from huggingface_hub import snapshot_download  

RDLogger.DisableLog("rdApp.warning")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_ROOT_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_ROOT_DIR, exist_ok=True)


GEN_CONFIG_PATH = "./configs/surfbart_generation.yaml"
LOCAL_TRAIN_CONFIG_PATH = "checkpoint/config.yaml"

from src.utils.misc import load_config, save_pickle
from src.utils.reconstruct import convert_tdsmiles_to_mol, convert_smiles_to_mol
from src.model.tokenizer import WhitespaceTokenizer
from src.model.SurfBart import SurfaceBartv2
from src.utils.scoring_func import compute_similarity_and_artifacts
from src.model.InferDataset import PointCloudDataset, PointCloudCollator


def prepare_dataloader(
    reference_mols: List[Chem.Mol],
    num_samples: int,
    batch_size: int,
    normalize_pointcloud: bool = True,
    cache_pc: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = PointCloudDataset(
        mol_list=reference_mols,
        num_samples=num_samples,
        normalize_pc=normalize_pointcloud,
        cache=cache_pc,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=PointCloudCollator(),
        shuffle=False,
    )


def generate_tdsmiles(
    model: SurfaceBartv2,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: WhitespaceTokenizer,
    device: torch.device,
    generation_config: Dict[str, Any],
    atom_token_temp: float,
    angle_token_temp: float,
) -> Tuple[List[str], List[float]]:
    from transformers import LogitsProcessorList, LogitsProcessor

    class TemperatureScheduler(LogitsProcessor):
        def __init__(
            self,
            ranges: List[Tuple[int, int]],
            temps: List[float],
            min_prob: float = 1e-9,
        ):
            super().__init__()
            self.ranges = ranges
            self.temps = temps
            self.min_prob = min_prob

        def __call__(self, input_ids, scores):
            probs = torch.softmax(scores, dim=-1)
            new_probs = torch.zeros_like(probs)

            for (start, end), tmp in zip(self.ranges, self.temps):
                group = probs[:, start:end]
                group_sum = group.sum(-1, keepdim=True)
                mask = group_sum.squeeze(-1) > 0

                if mask.any():
                    normalized = group[mask] / group_sum[mask]
                    adjusted = normalized.clamp_min(self.min_prob).pow(1.0 / tmp)
                    adjusted /= adjusted.sum(-1, keepdim=True).clamp_min(self.min_prob)
                    new_probs[mask, start:end] = adjusted * group_sum[mask]

            new_probs = new_probs.clamp_min(self.min_prob)
            return torch.log(new_probs)

    temp_processor = TemperatureScheduler(
        ranges=[(0, 99), (99, 460)],
        temps=[atom_token_temp, angle_token_temp],
    )
    logits_processors = LogitsProcessorList([temp_processor])

    model.eval()
    all_smiles: List[str] = []
    all_scores: List[float] = []

    for batch in dataloader:
        batch_on_device = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        generated_ids, scores = model.sample(
            pointcloud=batch_on_device["pointcloud"],
            normals=batch_on_device["normals"],
            generation_config=generation_config,
            logits_processor=logits_processors,
            prefix_ids=[
                tokenizer.encode("<BOS>")[1],
                tokenizer.encode("<std>")[1],
            ],
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [s.lstrip("<aug>").lstrip("<std>").lstrip() for s in decoded]

        all_smiles.extend(decoded)
        all_scores.extend(scores)

    return all_smiles, all_scores


def convert_and_group_molecules(
    smiles_list: List[str],
    score_list: List[float],
    num_return_sequences: int,
) -> List[List[PropertyMol]]:
    num_workers = min(20, os.cpu_count() or 4)
    num_seq = int(num_return_sequences)

    mol_results = process_map(
        convert_tdsmiles_to_mol,
        smiles_list,
        max_workers=num_workers,
        chunksize=20,
    )

    grouped_results: List[List[PropertyMol]] = []
    index = 0
    total_groups = len(mol_results) // num_seq

    for _ in range(total_groups):
        bucket: List[PropertyMol] = []
        for _ in range(num_seq):
            if index >= len(mol_results):
                break
            mol_obj, score_value = mol_results[index], score_list[index]
            index += 1

            if mol_obj is None or (
                isinstance(mol_obj, str) and mol_obj.startswith("Error")
            ):
                continue

            prop_mol = PropertyMol(mol_obj)
            prop_mol.SetProp("score", f"{float(score_value):.6f}")
            bucket.append(prop_mol)
        grouped_results.append(bucket)

    return grouped_results


def generate_and_save_results_cli(
    reference_mols: List[Chem.Mol],
    generation_parameters: Dict[str, Any],
    model: SurfaceBartv2,
    tokenizer: WhitespaceTokenizer,
    accelerator: Accelerator,
    gen_cfg: Dict[str, Any],
    output_dir: Path,
    num_samples: int,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = accelerator.device
    batch_size = gen_cfg.get("batch_size", 1)

    dataloader = prepare_dataloader(
        reference_mols=reference_mols,
        num_samples=num_samples,
        batch_size=batch_size,
        normalize_pointcloud=True,
        cache_pc=True,
    )
    dataloader = accelerator.prepare(dataloader)

    smiles_list, score_list = generate_tdsmiles(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        generation_config=generation_parameters,
        atom_token_temp=generation_parameters.get("atom_token_temp", 1.0),
        angle_token_temp=generation_parameters.get("angle_token_temp", 1.0),
    )

    num_sequences = generation_parameters.get("num_return_sequences", 1)
    grouped_mols = convert_and_group_molecules(
        smiles_list=smiles_list,
        score_list=score_list,
        num_return_sequences=num_sequences,
    )

    pkl_path = output_dir / "generated_mols.pkl"
    save_pickle(grouped_mols, str(pkl_path))

    img_dir = output_dir / "img"
    mol_dir = output_dir / "mol"

    df, artifacts = compute_similarity_and_artifacts(
        ref_mols=reference_mols,
        gen_data=grouped_mols,
        img_dir=img_dir,
        mol_dir=mol_dir,
    )

    sdf_path = output_dir / "all_generated_molecules.sdf"
    with Chem.SDWriter(str(sdf_path)) as sdf_writer:
        for _, row in df.iterrows():
            try:
                ref_idx = int(row["ref_id"])
                gen_idx = int(row["gen_mol_id"])
                gen_mol = grouped_mols[ref_idx][gen_idx]

                gen_mol.SetProp("ref_id", str(ref_idx))
                gen_mol.SetProp("gen_mol_id", str(gen_idx))

                if "shape_similarity" in row and pd.notna(row["shape_similarity"]):
                    gen_mol.SetProp(
                        "shape_similarity", f"{row['shape_similarity']:.4f}"
                    )
                if "graph_similarity" in row and pd.notna(row["graph_similarity"]):
                    gen_mol.SetProp(
                        "graph_similarity", f"{row['graph_similarity']:.4f}"
                    )
                if "score" in row and pd.notna(row["score"]):
                    gen_mol.SetProp("score", f"{row['score']:.4f}")
                if "gen_smiles" in row:
                    gen_mol.SetProp("gen_smiles", str(row["gen_smiles"]))

                sdf_writer.write(gen_mol)
            except (IndexError, KeyError) as e:
                print(f"[WARN] Skipped one molecule when writing SDF: {e}")

    desired_order = [
        "ref_id",
        "gen_mol_id",
        "shape_similarity",
        "graph_similarity",
        "score",
        "ref_smiles",
        "gen_smiles",
    ]
    csv_cols = [c for c in desired_order if c in df.columns]
    df_out = df[csv_cols].copy()
    csv_path = output_dir / "similarity_report.csv"
    df_out.to_csv(csv_path, index=False)

    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] SDF saved to: {sdf_path}")
    print(f"[INFO] Pickle saved to: {pkl_path}")
    print(f"[INFO] Images (if any) saved under: {img_dir}")
    print(f"[INFO] 3D mol files saved under: {mol_dir}")

    return df, pkl_path, sdf_path


def load_reference_mols_from_args(args: argparse.Namespace) -> List[Chem.Mol]:
    if args.smiles:
        mol_or_err = convert_smiles_to_mol(args.smiles)
        if isinstance(mol_or_err, str):
            raise ValueError(f"Could not parse SMILES '{args.smiles}': {mol_or_err}")
        return [mol_or_err]

    if args.sdf:
        sdf_path = Path(args.sdf)
        if not sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            raise ValueError(f"No valid molecules parsed from SDF: {sdf_path}")
        return [mols[0]]

    raise ValueError("You must provide either --smiles or --sdf.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ConfSeq / SurfBART-v2 shape-based molecule generation (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_input = parser.add_mutually_exclusive_group(required=True)
    g_input.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Reference molecule SMILES string.",
    )
    g_input.add_argument(
        "--sdf",
        type=str,
        default=None,
        help="Path to an SDF file containing a single reference molecule.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--atom-token-temp",
        type=float,
        default=1.2,
        help="Temperature for atom tokens (LogitsProcessor group).",
    )
    parser.add_argument(
        "--angle-token-temp",
        type=float,
        default=1.2,
        help="Temperature for angle/dihedral tokens (LogitsProcessor group).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for top_k in generation_config.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional override for top_p in generation_config.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=None,
        help="Optional override for num_return_sequences.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for max_length.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/confseq_shapegen_cli",
        help="Directory to save all outputs.",
    )

    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default='Oopstom/confseq-shape-gen',
        help=(
            "Hugging Face repo id, e.g. 'Oopstom/confseq-shape-gen'. "
            "If set, snapshot_download will be used and both config.yaml "
            "and pytorch_model.bin are taken from this repo."
        ),
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    reference_mols = load_reference_mols_from_args(args)
    print(f"[INFO] Loaded {len(reference_mols)} reference molecule(s).")

    gen_cfg = load_config(GEN_CONFIG_PATH)

    if args.hf_repo_id is not None:
        print(f"[INFO] Downloading model snapshot from Hugging Face Hub: {args.hf_repo_id}")
        repo_dir = snapshot_download(repo_id=args.hf_repo_id)
        train_config_path = os.path.join(repo_dir, "config.yaml")  
        model_path = repo_dir                                     
    else:
        train_config_path = LOCAL_TRAIN_CONFIG_PATH
        model_path = gen_cfg.get(
            "model_path",
            "./checkpoints/surfbartv2-sample1024-merge-angles-0421/checkpoint-175000",
        )

    print(f"[INFO] Training config path: {train_config_path}")
    train_cfg = load_config(train_config_path)

    tokenizer = WhitespaceTokenizer()
    model = SurfaceBartv2(train_cfg["model"], tokenizer=tokenizer)

    print(f"[INFO] Loading model weights from directory: {model_path}")
    model.load_weights(model_path)
    model.eval()

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    print(f"[INFO] Model prepared on device: {accelerator.device}")

    num_samples = train_cfg.get("data", {}).get("num_samples", 1024)

    generation_parameters: Dict[str, Any] = dict(gen_cfg.get("generation_config", {}))
    generation_parameters["do_sample"] = generation_parameters.get("do_sample", True)

    if args.top_k is not None:
        generation_parameters["top_k"] = args.top_k
    if args.top_p is not None:
        generation_parameters["top_p"] = args.top_p
    if args.num_return_sequences is not None:
        generation_parameters["num_return_sequences"] = args.num_return_sequences
    if args.max_length is not None:
        generation_parameters["max_length"] = args.max_length

    generation_parameters["atom_token_temp"] = args.atom_token_temp
    generation_parameters["angle_token_temp"] = args.angle_token_temp

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        generate_and_save_results_cli(
            reference_mols=reference_mols,
            generation_parameters=generation_parameters,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            gen_cfg=gen_cfg,
            output_dir=output_dir,
            num_samples=num_samples,
        )
    finally:
        del model
        del tokenizer
        del accelerator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] Finished and cleaned up.")


if __name__ == "__main__":
    main()
