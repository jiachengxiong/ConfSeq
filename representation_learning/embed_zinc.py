# -*- coding: utf-8 -*-
"""
Pipeline: ZINC SMILES de-duplication -> 3D conformer generation -> save SDF -> TD-SMILES (ConfSeq) conversion

Timing: only measures two coarse-grained stages
    (1) Conformer generation
    (2) TD-SMILES (ConfSeq) conversion

Dependencies: RDKit, numpy, pandas, tqdm, timeout_decorator
Can be executed both in Jupyter or as a standalone script (via CLI).
"""

import os, time, json
import math
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from tqdm.contrib.concurrent import process_map

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import timeout_decorator
import random

from demo.ConfSeq import (
    run_aug_mol_get_ConfSeq_pair_0,
    run_aug_mol_get_ConfSeq_pair_1,
    run_aug_mol_get_ConfSeq_pair_2,
    random_adjust_numbers,
)
from tqdm import tqdm


def mol2confseq(
    mols_list,
    num_workers: int = 10,
    aug_mode: int = 0,
    aug_times: int = 1,
    do_random: bool = False,
    disable_tqdm: bool = False,
):
    """
    Convert a list of RDKit molecule objects into their corresponding Conformation Sequence (ConfSeq) strings.

    Parameters
    ----------
    mols_list : list[Mol]
        A list of RDKit Mol objects to be processed.
    num_workers : int, optional
        Number of parallel workers used for ConfSeq conversion. Default is 10.
    aug_mode : int, optional
        Conformation/SMILES augmentation mode that controls how the molecule is randomized:
            0 : Rooted SMILES augmentation at atom index 0 (canonical rooted SMILES).
            1 : Rooted SMILES augmentation at a random atom index (randomized rooted SMILES).
            2 : Fully randomized atom ordering using random SMILES generation.
    aug_times : int, optional
        How many augmented ConfSeq strings to generate per molecule (molecules are repeated aug_times times
        in the processing list). Default is 1.
    do_random : bool, optional
        If True, apply additional random numeric perturbations to the ConfSeq tokens via `random_adjust_numbers`.
        Default is False.
    disable_tqdm : bool, optional
        If True, disable tqdm progress bars. Default is False.

    Returns
    -------
    confseq_list : list[str]
        A list of ConfSeq strings. Each entry corresponds to one item of `mols_list * aug_times`.
        If processing fails or the format of the returned string is unexpected, the value is "error".
    """

    datas = []
    for mol in mols_list:
        if mol is not None:
            datas.append((mol, Chem.MolToSmiles(mol)))

    if aug_mode == 0:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_0,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000,
            disable=disable_tqdm,
        )
    elif aug_mode == 1:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_1,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000,
            disable=disable_tqdm,
        )
    elif aug_mode == 2:
        results_t0 = process_map(
            run_aug_mol_get_ConfSeq_pair_2,
            tqdm(datas * aug_times, disable=disable_tqdm),
            max_workers=num_workers,
            chunksize=1000,
            disable=disable_tqdm,
        )
    else:
        raise ValueError(f"Invalid aug_mode: {aug_mode}")

    random.seed(42)
    if do_random:
        for i in range(len(results_t0)):
            if random.random() >= 0.5:
                results_t0[i] = random_adjust_numbers(results_t0[i])
            # Normalize angle token format
            results_t0[i] = results_t0[i].replace("<180>", "<-180>")

    confseq_list = []
    for i in range(len(results_t0)):
        parts = results_t0[i].split("\t")
        if len(parts) == 3:
            confseq_list.append(parts[2])
        else:
            confseq_list.append("error")

    return confseq_list


# --------------------
# Basic utilities
# --------------------
def parse_smiles_file(path: str, max_records: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Read a 'SMILES ZINCID' text file and return a list of (smiles, zinc_id) tuples.

    Parameters
    ----------
    path : str
        Path to the SMILES text file. Each non-empty line must contain at least two columns:
        SMILES and ZINC ID separated by whitespace.
    max_records : int or None, optional
        If not None, stop reading once this many records have been collected.

    Returns
    -------
    items : list[tuple[str, str]]
        List of (smiles, zinc_id) pairs.
    """
    items: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # allow multiple whitespaces
            if len(parts) < 2:
                continue
            smi, zid = parts[0], parts[1]
            items.append((smi, zid))
            if max_records is not None and len(items) >= max_records:
                break
    return items


def mol_from_smiles(smi: str) -> Optional[Mol]:
    """
    Robust SMILES parsing and sanitization.

    Parameters
    ----------
    smi : str
        SMILES string.

    Returns
    -------
    mol : Mol or None
        Sanitized RDKit Mol object on success; None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def get_inchikey_or_cansmi(mol: Mol) -> Tuple[str, str]:
    """
    Return a structural key and a canonical SMILES for a molecule.

    The structural key is InChIKey if available; if InChI is not enabled in RDKit,
    the canonical isomeric SMILES is used as a fallback.

    Parameters
    ----------
    mol : Mol
        RDKit molecule.

    Returns
    -------
    key : str
        InChIKey or canonical isomeric SMILES (fallback).
    can : str
        Canonical isomeric SMILES.
    """
    can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    try:
        key = Chem.MolToInchiKey(mol)  # requires RDKit compiled with InChI
    except Exception:
        key = can
    return key, can


# --------------------
# 3D conformer generation (with timeout, coarse-grained timing only)
# --------------------
def _embed_and_optimize_one_conf(mol: Mol, max_iters: int = 200) -> Optional[Mol]:
    """
    Generate and optimize a single 3D conformer for a molecule.

    Pipeline:
        - Add explicit hydrogens
        - Generate 3D coordinates using ETKDGv3
        - MMFF94s geometry optimization (fallback to UFF if MMFF fails)
        - Remove hydrogens and assign stereochemistry

    Parameters
    ----------
    mol : Mol
        RDKit molecule (no hydrogens required).
    max_iters : int, optional
        Maximum number of optimization iterations for the force field.

    Returns
    -------
    mol3d : Mol or None
        Molecule with 3D coordinates (hydrogens removed) if successful; otherwise None.
    """
    try:
        molH = Chem.AddHs(mol, addCoords=False)
        params = AllChem.ETKDGv3()
        params.numThreads = 0  # use RDKit internal default thread management
        params.randomSeed = 0xC0FFEE
        params.pruneRmsThresh = 0.1

        cid = AllChem.EmbedMolecule(molH, params)
        if cid < 0:
            return None

        try:
            mp = AllChem.MMFFGetMoleculeProperties(molH, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(molH, mp)
            ff.Initialize()
            ff.Minimize(maxIts=max_iters)
        except Exception:
            ff = AllChem.UFFGetMoleculeForceField(molH)
            ff.Initialize()
            ff.Minimize(maxIts=max_iters)

        mol3d = Chem.RemoveHs(molH)
        Chem.AssignAtomChiralTagsFromStructure(mol3d)
        Chem.AssignStereochemistry(mol3d, cleanIt=True, force=True)
        return mol3d
    except Exception:
        return None


def _generate_with_timeout(mol: Mol, timeout_s: int = 10) -> Optional[Mol]:
    """
    Wrapper for 3D conformer generation with a per-molecule timeout.

    Parameters
    ----------
    mol : Mol
        RDKit molecule to process.
    timeout_s : int, optional
        Timeout in seconds for the conformer generation and optimization.

    Returns
    -------
    mol3d : Mol or None
        3D molecule on success; None if timeout or any error occurs.
    """

    try:

        @timeout_decorator.timeout(timeout_s)
        def _run():
            return _embed_and_optimize_one_conf(mol)

        return _run()
    except Exception:
        return None


def worker_generate(item: Tuple[str, str]) -> Dict[str, Any]:
    """
    Worker function for parallel 3D generation.

    Parameters
    ----------
    item : tuple[str, str]
        (smiles, ZINC_ID) pair.

    Returns
    -------
    res : dict
        Result dictionary with the following keys:
            "ZINC_ID" : str
            "input_smiles" : str
            "canonical_smiles" : str or None
            "mol3d" : Mol or None
            "status" : str, one of {"init", "smiles_parse_fail", "gen3d_fail", "ok"}
            "key" : str or None (InChIKey or canonical SMILES key)
    """
    smi, zid = item
    res: Dict[str, Any] = {
        "ZINC_ID": zid,
        "input_smiles": smi,
        "canonical_smiles": None,
        "mol3d": None,
        "status": "init",
        "key": None,
    }
    mol = mol_from_smiles(smi)
    if mol is None:
        res["status"] = "smiles_parse_fail"
        return res
    key, can = get_inchikey_or_cansmi(mol)
    res["canonical_smiles"] = can
    res["key"] = key

    mol3d = _generate_with_timeout(mol, timeout_s=10)
    res["mol3d"] = mol3d
    res["status"] = "ok" if mol3d is not None else "gen3d_fail"
    return res


# --------------------
# De-duplication + 3D generation + SDF writing
# --------------------
def deduplicate_by_key(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    De-duplicate a list of (SMILES, ZINC_ID) pairs based on structural key.

    The structural key is InChIKey if available, otherwise canonical isomeric SMILES.
    Only the first occurrence of each key is kept.

    Parameters
    ----------
    items : list[tuple[str, str]]
        List of (smiles, zinc_id) pairs.

    Returns
    -------
    out : list[tuple[str, str]]
        De-duplicated list of (smiles, zinc_id) pairs.
    """
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for smi, zid in items:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        key, _ = get_inchikey_or_cansmi(mol)
        if key in seen:
            continue
        seen.add(key)
        out.append((smi, zid))
    return out


def write_sdf(path_sdf: str, records: List[Dict[str, Any]]) -> int:
    """
    Write molecules with 3D coordinates to an SDF file.

    Only records with:
        - status == "ok"
        - non-None "mol3d"
    are written.

    For each molecule, the following properties are added:
        "_Name"          : ZINC_ID
        "ZINC_ID"        : ZINC_ID
        "input_smiles"   : original input SMILES
        "canonical_smiles": canonical isomeric SMILES (when available)

    Parameters
    ----------
    path_sdf : str
        Output SDF path.
    records : list[dict]
        List of worker_generate results.

    Returns
    -------
    n : int
        Number of molecules written to SDF.
    """
    writer = Chem.SDWriter(path_sdf)
    n = 0
    for r in records:
        if r["status"] != "ok" or r["mol3d"] is None:
            continue
        m: Mol = r["mol3d"]
        m.SetProp("_Name", r["ZINC_ID"])
        m.SetProp("ZINC_ID", r["ZINC_ID"])
        m.SetProp("input_smiles", r["input_smiles"])
        if r.get("canonical_smiles"):
            m.SetProp("canonical_smiles", r["canonical_smiles"])
        writer.write(m)
        n += 1
    writer.close()
    return n


# --------------------
# Main pipeline
# --------------------
def run_pipeline(
    smiles_path: str = "data/ZINC/all_instock_smiles.smi",
    sdf_out: str = "outputs/zinc_instock_3d.sdf",
    csv_out: str = "outputs/zinc_instock_tdsmiles.csv",
    timing_out: str = "outputs/zinc_instock_timing.json",
    num_workers: int = 16,
    max_records: Optional[int] = None,
    aug_mode: int = 0,
    aug_times: int = 1,
    do_random: bool = False,
    disable_tqdm: bool = False,
):
    """
    Full end-to-end pipeline:

        1. Load SMILES + ZINC_ID file
        2. De-duplicate by structural key (InChIKey or canonical SMILES)
        3. Generate 3D conformers (parallel, ETKDGv3 + MMFF94s/UFF)
        4. Write 3D molecules to SDF
        5. Convert valid 3D molecules to TD-SMILES (ConfSeq)
        6. Save TD-SMILES table (CSV) and timing summary (JSON)

    Parameters
    ----------
    smiles_path : str, optional
        Input SMILES file path (one line per molecule: "SMILES ZINC_ID").
    sdf_out : str, optional
        Output SDF file path for 3D molecules.
    csv_out : str, optional
        Output CSV path for TD-SMILES table.
    timing_out : str, optional
        Output JSON path for timing summary.
    num_workers : int, optional
        Number of parallel workers for both 3D generation and ConfSeq conversion.
    max_records : int or None, optional
        If not None, only process the first `max_records` lines of the SMILES file.
    aug_mode : int, optional
        ConfSeq augmentation mode (0, 1, or 2).
    aug_times : int, optional
        Number of ConfSeq augmentations per molecule.
    do_random : bool, optional
        Whether to apply extra random numeric perturbation to ConfSeq tokens.
    disable_tqdm : bool, optional
        Disable progress bars if True.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ["ZINC_ID", "input_smiles", "canonical_smiles", "confseq"].
    timing_report : dict
        Dictionary summarizing total timings and throughput for the two main stages.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(sdf_out), exist_ok=True)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(timing_out), exist_ok=True)

    # 1) Load raw items
    items = parse_smiles_file(smiles_path, max_records=max_records)
    print(f"Loaded {len(items)} lines from {smiles_path}")

    # 2) De-duplicate by structural key
    print("De-duplicating by structural key (InChIKey / canonical SMILES)...")
    dedup_items = deduplicate_by_key(items)
    print(f"After de-duplication: {len(dedup_items)} unique molecules")

    # 3) 3D conformer generation (parallel, coarse-grained timing)
    print("Generating 3D conformers (ETKDGv3 + MMFF94s/UFF)...")
    t0 = time.perf_counter()
    recs = process_map(
        worker_generate,
        dedup_items,
        max_workers=num_workers,
        chunksize=1000,
        desc="3D",
        disable=disable_tqdm,
    )
    t1 = time.perf_counter()
    confgen_sec = t1 - t0

    # 4) Write SDF (not counted into conformer generation time)
    n_written = write_sdf(sdf_out, recs)
    print(f"Wrote {n_written} molecules to {sdf_out}")

    # 5) Collect valid molecules for ConfSeq conversion
    valid_mols, ids, in_smis, can_smis = [], [], [], []
    for r in recs:
        if r["status"] == "ok" and r["mol3d"] is not None:
            valid_mols.append(r["mol3d"])
            ids.append(r["ZINC_ID"])
            in_smis.append(r["input_smiles"])
            can_smis.append(r["canonical_smiles"])
    n_valid = len(valid_mols)
    if n_valid == 0:
        print("No valid molecules for ConfSeq conversion; exiting.")
        # Save empty CSV and timing report
        pd.DataFrame(
            columns=["ZINC_ID", "input_smiles", "canonical_smiles", "confseq"]
        ).to_csv(csv_out, index=False)
        with open(timing_out, "w") as f:
            json.dump(
                {
                    "loaded_lines": len(items),
                    "unique_after_dedup": len(dedup_items),
                    "n_written_sdf": n_written,
                    "n_valid_for_confseq": 0,
                    "confgen_seconds": confgen_sec,
                    "confgen_throughput_mol_per_s": None,
                    "confseq_seconds": None,
                    "confseq_throughput_mol_per_s": None,
                },
                f,
                indent=2,
            )
        return pd.DataFrame(), {
            "confgen_seconds": confgen_sec,
            "confseq_seconds": None,
        }

    # 6) TD-SMILES (ConfSeq) conversion (coarse-grained timing)
    print("Converting to TD-SMILES (ConfSeq)...")
    t2 = time.perf_counter()
    td_list = mol2confseq(
        mols_list=valid_mols,
        num_workers=num_workers,
        aug_mode=aug_mode,
        aug_times=aug_times,
        do_random=do_random,
        disable_tqdm=True,
    )
    t3 = time.perf_counter()
    confseq_sec = t3 - t2

    # 7) Build DataFrame (aligned by ZINC_ID)
    df = pd.DataFrame(
        {
            "ZINC_ID": ids,
            "input_smiles": in_smis,
            "canonical_smiles": can_smis,
            "confseq": td_list,
        }
    )
    df.to_csv(csv_out, index=False)
    print(f"Saved TD-SMILES table: {csv_out} (rows={len(df)})")

    # 8) Print and save timing summary
    confgen_tps = n_valid / confgen_sec if confgen_sec > 0 else float("nan")
    confseq_tps = n_valid / confseq_sec if confseq_sec > 0 else float("nan")

    print(
        f"[Time] Conformer generation: {confgen_sec:.2f} s  | kept={n_valid} | "
        f"throughput={confgen_tps:.2f} mol/s"
    )
    print(
        f"[Time] ConfSeq conversion : {confseq_sec:.2f} s  | kept={n_valid} | "
        f"throughput={confseq_tps:.2f} mol/s"
    )

    timing_report = {
        "loaded_lines": len(items),
        "unique_after_dedup": len(dedup_items),
        "n_written_sdf": n_written,
        "n_valid_for_confseq": n_valid,
        "confgen_seconds": confgen_sec,
        "confgen_throughput_mol_per_s": confgen_tps,
        "confseq_seconds": confseq_sec,
        "confseq_throughput_mol_per_s": confseq_tps,
        "params": {
            "num_workers": num_workers,
            "aug_mode": aug_mode,
            "aug_times": aug_times,
            "do_random": do_random,
        },
    }
    with open(timing_out, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f"Saved timing summary: {timing_out}")

    return df, timing_report


# --------------------
# Command-line entry
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "ZINC SMILES -> 3D conformer generation -> SDF export -> TD-SMILES (ConfSeq) conversion\n"
            "The script measures overall time of (1) conformer generation and (2) ConfSeq conversion."
        )
    )

    parser.add_argument(
        "--smiles_path",
        type=str,
        default="data/ZINC/all_instock_smiles.smi",
        help="Input SMILES file path. Each line should be 'SMILES ZINC_ID'.",
    )
    parser.add_argument(
        "--sdf_out",
        type=str,
        default="outputs/zinc_instock_3d_test.sdf",
        help="Output SDF file path for 3D molecules.",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="outputs/zinc_instock_tdsmiles_test.csv",
        help="Output CSV file path for TD-SMILES table.",
    )
    parser.add_argument(
        "--timing_out",
        type=str,
        default="outputs/zinc_instock_timing_test.json",
        help="Output JSON file path for timing summary.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Number of parallel workers for both conformer generation and ConfSeq conversion.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help=(
            "Maximum number of records to read from the SMILES file. "
            "If None, all records are used."
        ),
    )
    parser.add_argument(
        "--aug_mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "ConfSeq augmentation mode: "
            "0 = rooted SMILES at atom 0; "
            "1 = rooted SMILES at random atom; "
            "2 = fully randomized SMILES."
        ),
    )
    parser.add_argument(
        "--aug_times",
        type=int,
        default=1,
        help="Number of ConfSeq augmentations per molecule (molecules will be repeated aug_times times).",
    )
    parser.add_argument(
        "--do_random",
        action="store_true",
        help="Apply additional random numeric perturbations to ConfSeq tokens.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars if set.",
    )

    args = parser.parse_args()

    df, timing = run_pipeline(
        smiles_path=args.smiles_path,
        sdf_out=args.sdf_out,
        csv_out=args.csv_out,
        timing_out=args.timing_out,
        num_workers=args.num_workers,
        max_records=args.max_records,
        aug_mode=args.aug_mode,
        aug_times=args.aug_times,
        do_random=args.do_random,
        disable_tqdm=args.disable_tqdm,
    )

    # Optional: print brief head summary to stdout
    print("\n[Summary] First few TD-SMILES rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head())
    print("\n[Summary] Timing report:")
    print(json.dumps(timing, indent=2))
