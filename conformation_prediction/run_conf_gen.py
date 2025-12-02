"""
confseq_predictor.py
====================
A consolidated utility script that bundles the notebook logic you provided into a
single, reusable Python module.  It exposes a *single* public helper
`generate_mols_from_smiles()` that accepts a SMILES string and returns a list of
RDKit `Mol` objects (each bearing a single 3-D conformer) together with the
model-estimated sequence scores.

Requirements
------------
* Python ≥ 3.10 (for | type hints)  [or adjust hints if you need 3.9]
* RDKit (2023.09 or later)
* transformers ≥ 4.38
* timeout_decorator
* indigo-python
* scipy
* tqdm
* demo.ConfSeq.py in your $PYTHONPATH, providing:
    - ``get_ConfSeq_pair_from_mol``
    - ``get_mol_from_ConfSeq_pair``
    - ``randomize_mol``

Model weights
-------------
By default, the model weights are loaded from:

    ./checkpoints/model_epoch_3_175000.pth

You can override this via the environment variable:

    CONFSEQ_WEIGHTS=/path/to/your/weights.pth

Example (Python)
----------------
>>> from confseq_predictor import generate_mols_from_smiles
>>> smiles = "CC(F)(C(F)(F)F)C1=CC=C..."  # your molecule
>>> mols, scores = generate_mols_from_smiles(
...     smiles,
...     conf_num=50,
...     device="cuda",
...     seed=123,
...     top_p=0.95,
...     top_k=256,
... )
>>> print(len(mols))
50
"""

from __future__ import annotations

import os
import sys
sys.path.append('../') 
import random
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdchem, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import mode  # kept for compatibility with ConfSeq utilities
from timeout_decorator import timeout_decorator  # noqa: F401
from tqdm.contrib.concurrent import process_map
from transformers import BartConfig, BartForConditionalGeneration, set_seed
from huggingface_hub import hf_hub_download


from demo.ConfSeq import (
    get_ConfSeq_pair_from_mol,
    get_mol_from_ConfSeq_pair,
    randomize_mol,
)


# silence noisy RDKit logging
RDLogger.DisableLog("rdApp.*")

# -----------------------------------------------------------------------------
# Global vocabulary construction (mirrors the notebook)
# -----------------------------------------------------------------------------
VOCAB: List[str] = [chr(i) for i in range(33, 127)]  # printable ASCII
VOCAB.extend(f"<{i}>" for i in range(-180, 180))
VOCAB += ["<mask>", "<unk>", "<sos>", "<eos>", "<pad>"]

pad_idx = VOCAB.index("<pad>")
vocab_id_dic = {ch: idx for idx, ch in enumerate(VOCAB)}
id_vocab_dic = {idx: ch for idx, ch in enumerate(VOCAB)}

# -----------------------------------------------------------------------------
# Default weight path (no longer exposed as function parameter)
# -----------------------------------------------------------------------------
DEFAULT_WEIGHT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "model_epoch_3_175000.pth"
)
HF_REPO_ID = os.environ.get("CONFSEQ_HF_REPO", "Oopstom/confseq-conf-gen")
HF_FILENAME = os.environ.get("CONFSEQ_HF_FILENAME", "model_epoch_3_175000.pth")


def _resolve_weight_path() -> str:
    env_path = os.environ.get("CONFSEQ_WEIGHTS")
    if env_path:
        return env_path

    if HF_REPO_ID:
        try:
            weight_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
            )
            return weight_path
        except Exception as e:
            print(
                f"[ConfSeq] WARNING: failed to download weights from HF Hub "
                f"({HF_REPO_ID}/{HF_FILENAME}): {e}",
                file=sys.stderr,
            )

    return str(DEFAULT_WEIGHT_PATH)



# -----------------------------------------------------------------------------
# RDKit helpers
# -----------------------------------------------------------------------------

def neutralize_charge(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        num_h = atom.GetNumExplicitHs()

        if charge < 0 or (charge > 0 and num_h >= charge):
            has_opposite_neighbor = False
            for neighbor in atom.GetNeighbors():
                neighbor_charge = neighbor.GetFormalCharge()
                if neighbor_charge == -charge:
                    has_opposite_neighbor = True
                    break

            if not has_opposite_neighbor:
                atom.SetFormalCharge(0)
                if charge > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - charge)
                elif charge < 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - charge)

    Chem.SanitizeMol(mol)
    return mol


def remove_double_bond_type(mol_block: str) -> str:
    lines = mol_block.split("\n")
    atom_num = int(lines[3][:3].strip())
    bond_num = int(lines[3][3:6].strip())
    for i in range(4 + atom_num, 4 + atom_num + bond_num):
        if lines[i][8] == "2":
            lines[i] = lines[i][:11] + "0" + lines[i][12:]
    return "\n".join(lines)


def generate_conformations(mol: Chem.Mol, *, num_confs: int = 1) -> List[Chem.Mol]:
    mol = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDG()
    params.enforceChirality = False
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    conformers: List[Chem.Mol] = []
    for conf_id in range(num_confs):
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        m.AddConformer(mol.GetConformer(conf_id))
        m = Chem.RemoveHs(m)
        mblock = remove_double_bond_type(Chem.MolToMolBlock(m))
        m = Chem.MolFromMolBlock(mblock, sanitize=True, removeHs=True)
        Chem.AssignAtomChiralTagsFromStructure(m)
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        conformers.append(m)
    return conformers


def rm_invalid_chirality(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    rings = rdmolops.GetSymmSSSR(mol)
    counts = {}
    for ring in rings:
        for idx in ring:
            counts[idx] = counts.get(idx, 0) + 1
    for atom_idx, ct in counts.items():
        if ct == 3:
            mol.GetAtomWithIdx(atom_idx).SetChiralTag(
                rdchem.ChiralType.CHI_UNSPECIFIED
            )
    return mol


def aug_mol(mol: Chem.Mol, mode: int = 0) -> Chem.Mol:
    mol = Chem.RemoveHs(mol)
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    mol = rm_invalid_chirality(mol)
    mol = randomize_mol(mol)
    Chem.MolToSmiles(mol)
    ran_mol = Chem.RenumberAtoms(mol, eval(mol.GetProp("_smilesAtomOutputOrder")))

    if mode == 0:
        root = 0
    elif mode == 1:
        root = random.randint(0, ran_mol.GetNumAtoms() - 1)
    else:  # mode == 2
        Chem.MolToSmiles(ran_mol, doRandom=True)
        ran_mol = Chem.RenumberAtoms(
            ran_mol, eval(ran_mol.GetProp("_smilesAtomOutputOrder"))
        )
        Chem.MolToSmiles(ran_mol, canonical=False)
        return Chem.RenumberAtoms(
            ran_mol, eval(ran_mol.GetProp("_smilesAtomOutputOrder"))
        )

    Chem.MolToSmiles(ran_mol, rootedAtAtom=root)
    ran_mol = Chem.RenumberAtoms(
        ran_mol, eval(ran_mol.GetProp("_smilesAtomOutputOrder"))
    )
    Chem.MolToSmiles(ran_mol, canonical=False)
    return Chem.RenumberAtoms(
        ran_mol, eval(ran_mol.GetProp("_smilesAtomOutputOrder"))
    )


# -----------------------------------------------------------------------------
# Sequence utilities
# -----------------------------------------------------------------------------

def pad_sequences(
    sequences: List[List[int]],
    padding_value: int = pad_idx,
) -> List[List[int]]:
    max_len = max(len(seq) for seq in sequences)
    return [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]


# -----------------------------------------------------------------------------
# Model loader (cached, no device binding)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model(weight_path: str) -> BartForConditionalGeneration:
    config = BartConfig(
        pad_token_id=vocab_id_dic["<pad>"],
        eos_token_id=vocab_id_dic["<eos>"],
        sos_token_id=vocab_id_dic["<sos>"],
        forced_eos_token_id=None,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        d_model=256,
        dropout=0.3,
        attention_dropout=0.3,
        classifier_dropout=0.3,
        num_hidden_layers=6,
        static_position_embeddings=True,
        max_position_embeddings=512,
        activation_function="relu",
        share_embeddings=True,
        vocab_size=len(VOCAB),
    )

    model = BartForConditionalGeneration(config)
    chkpt = torch.load(weight_path, map_location="cpu")
    new_state = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v) for k, v in chkpt.items()
    )
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model.generation_config.bos_token_id = None
    model.generation_config.decoder_start_token_id = vocab_id_dic["<sos>"]
    return model


# -----------------------------------------------------------------------------
# Generation wrapper
# -----------------------------------------------------------------------------

def _safe_get_mol_from_confseq(para: Tuple[str, str]) -> Chem.Mol | None:
    in_smiles, td_smiles = para
    try:
        mol = get_mol_from_ConfSeq_pair(in_smiles, td_smiles, is_op=True)
        mol = Chem.MolFromMolBlock(
            remove_double_bond_type(Chem.MolToMolBlock(mol))
        )
        return mol
    except Exception:
        return None


def _predict_conformers_confseq(
    in_smiles: str,
    *,
    temperature: float,
    conf_num: int,
    model: BartForConditionalGeneration,
    device: torch.device,
    seed: int,
    top_p: float,
    top_k: int,
    max_length: int,
) -> Tuple[List[Chem.Mol], List[float]]:
    nucleus_p = float(top_p)
    temperature = round(float(temperature), 2)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    num_return_sequences = conf_num * 2

    encoder_inputs = [
        [vocab_id_dic["<sos>"]] + [vocab_id_dic[ch] for ch in in_smiles.split(" ")]
    ]
    encoder_inputs *= num_return_sequences
    enc_padded = torch.tensor(
        pad_sequences(encoder_inputs, padding_value=pad_idx)
    ).long().to(device)

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=enc_padded,
            max_length=max_length,
            num_beams=1,
            do_sample=True,
            top_k=top_k,
            top_p=nucleus_p,
            temperature=temperature,
            eos_token_id=vocab_id_dic["<eos>"],
            pad_token_id=pad_idx,
            early_stopping=True,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    input_lens = [len(s) for s in encoder_inputs]
    seq_scores: List[float] = []
    for i, inp_len in enumerate(input_lens):
        probs = [
            F.softmax(score[i], dim=0) for score in gen_out.scores[: inp_len - 1]
        ]
        ids = gen_out.sequences[i][1:inp_len]
        tokens = torch.tensor(ids).to(device)
        selected = torch.stack(
            [prob_vec[token] for prob_vec, token in zip(probs, tokens)]
        )
        seq_scores.append(torch.exp(torch.mean(torch.log(selected))).item())

    pars = [
        (
            in_smiles,
            " ".join(
                id_vocab_dic[t.item()]
                for t in gen_out.sequences[i][1 : input_lens[i]]
            ),
        )
        for i in range(num_return_sequences)
    ]

    confs = process_map(
        _safe_get_mol_from_confseq,
        pars,
        max_workers=min(64, os.cpu_count() or 4),
        chunksize=1,
        desc="Back-mapping",
    )

    out_mols: List[Chem.Mol] = []
    out_scores: List[float] = []
    for mol, sc in zip(confs, seq_scores):
        if mol is not None:
            out_mols.append(mol)
            out_scores.append(sc)
            if len(out_mols) >= conf_num:
                break

    return out_mols, out_scores


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def generate_mols_from_smiles(
    smiles: str,
    *,
    conf_num: int = 100,
    temperature: float = 2.0,
    device: str | torch.device | None = None,
    seed: int = 42,
    top_p: float = 0.96,
    top_k: int = 360,
    max_length: int = 256,
) -> Tuple[List[Chem.Mol], List[float]]:
    try:
        base_mol = Chem.MolFromSmiles(smiles)
        if base_mol is None:
            raise ValueError("Invalid SMILES input")

        base_mol = neutralize_charge(base_mol)
        base_mol = rdMolStandardize.LargestFragmentChooser().choose(base_mol)
        base_mol = generate_conformations(base_mol, num_confs=1)[0]
        base_mol = rm_invalid_chirality(base_mol)
        base_mol = randomize_mol(base_mol)
        base_mol = aug_mol(base_mol, mode=0)

        in_smiles, _ = get_ConfSeq_pair_from_mol(base_mol)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pre-processing failed: {exc}") from exc

    dev = _resolve_device(device)
    weight_path = _resolve_weight_path()
    model = _get_model(weight_path)
    model.to(dev)

    mols, scores = _predict_conformers_confseq(
        in_smiles,
        temperature=temperature,
        conf_num=conf_num,
        model=model,
        device=dev,
        seed=seed,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length,
    )
    return mols, scores


# -----------------------------------------------------------------------------
# Utilities for I/O (SDF writing)
# -----------------------------------------------------------------------------

def write_mols_to_sdf(
    mols: List[Chem.Mol],
    scores: List[float] | None,
    sdf_path: str | Path,
) -> None:
    sdf_path = str(sdf_path)
    os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
    writer = Chem.SDWriter(sdf_path)
    try:
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            m = Chem.Mol(mol)
            m.SetProp("_Name", f"conf_{i}")
            if scores is not None and i < len(scores):
                try:
                    m.SetDoubleProp("ConfSeqScore", float(scores[i]))
                except Exception:
                    m.SetProp("ConfSeqScore", str(scores[i]))
            writer.write(m)
    finally:
        writer.close()


# -----------------------------------------------------------------------------
# CLI for quick testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 3-D conformers via ConfSeq/BART and save to an SDF file."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="Input SMILES string.",
    )
    parser.add_argument(
        "--conf_num",
        type=int,
        default=10,
        help="Number of conformers to output (default: 10).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Sampling temperature for generation (default: 2.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to run on: "auto", "cpu", "cuda", "cuda:0", etc. (default: auto).',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="Nucleus sampling top-p value (default: 0.96).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=360,
        help="Top-k sampling cutoff (default: 360).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum generated sequence length (default: 256).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="confseq_confs.sdf",
        help="Output SDF file path (default: confseq_confs.sdf).",
    )

    args = parser.parse_args()

    mols, scs = generate_mols_from_smiles(
        args.smiles,
        conf_num=args.conf_num,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
        top_p=args.top_p,
        top_k=args.top_k,
        max_length=args.max_length,
    )

    if not mols:
        print("No valid conformers were generated.")
    else:
        write_mols_to_sdf(mols, scs, args.out)
        print(
            f"Generated {len(mols)} conformers from SMILES:\n"
            f"  {args.smiles}\n"
            f"SDF written to: {args.out}"
        )
        if scs:
            print(f"Best ConfSeq score: {max(scs):.6f}")
