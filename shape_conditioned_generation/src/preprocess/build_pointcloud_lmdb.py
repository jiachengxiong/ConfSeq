import os
import sys
sys.path.append('.')
import gc
import math
import lmdb
import pickle
import argparse

import numpy as np
import torch
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm.auto import tqdm


from src.utils.misc import load_pickle
from src.utils.data import mol2TDsmiles
from src.utils.shape import get_mesh, get_pointcloud_from_mesh


__all__ = [
    "sample_pointcloud_from_mol",
    "write_batch",
    "process_dataset",
    "parse_args",
]

# -------------------------------------------------
# Sampling functions
# -------------------------------------------------

def sample_pointcloud_from_mol(mol, num_samples, normalize=False, return_normals=True):
    """Single molecule → (N,3)[, (N,3)] sampling

    Returns
    -------
    tuple | None
        (pc, normals) if successful; None if failed
    """
    try:
        mesh = get_mesh(mol)
        pc, normals = get_pointcloud_from_mesh(
            mesh,
            num_samples=num_samples,
            return_mesh=False,
            return_normals=return_normals,
        )
        pc = pc.squeeze(0)          # (N,3)
        normals = normals.squeeze(0) if return_normals else None

        if normalize:
            pc -= pc.mean(dim=0, keepdim=True)

        return pc, normals
    except Exception:
        # Return None for main process to filter / placeholder
        return None


# -------------------------------------------------
# Placeholder utility functions
# -------------------------------------------------

def make_empty_pc(return_normals=True):
    """Generate empty point cloud / normal tensors with shape=(1024,3)"""
    empty_pc = torch.zeros((1024, 3), dtype=torch.float32)
    empty_norm = torch.zeros((1024, 3), dtype=torch.float32) if return_normals else None
    return empty_pc, empty_norm


# -------------------------------------------------
# Single batch write function
# -------------------------------------------------

def write_batch(txn, start_key, pcs, td_smiles_list, mols, test_split=False):
    """Write a batch to LMDB

    Parameters
    -------
    pcs : list[tuple[Tensor, Tensor | None]]
        Point clouds and normals; if point cloud is empty placeholder, then ``pc.shape == (0, 3)``
    td_smiles_list : list[str]
        TD‑SMILES aligned with pcs
    mols : list[Chem.Mol]
        RDKit Mol aligned with pcs
    """
    for i, (pc_tuple, smi, mol) in enumerate(zip(pcs, td_smiles_list, mols)):
        pc, norm = pc_tuple
        key = f"{start_key + i:08d}".encode("ascii")

        datum = {
            "pointcloud": pc.numpy().astype(np.float32),
            "normals": norm.numpy().astype(np.float32) if norm is not None else None,
            "valid": pc.shape[0] > 0,  # True indicates successful sampling
        }
        if not test_split:
            datum.update({
                "TD_smiles": smi,
                "rdmol": Chem.MolToMolBlock(mol),
            })

        txn.put(key, pickle.dumps(datum, protocol=pickle.HIGHEST_PROTOCOL))


# -------------------------------------------------
# Main processing logic
# -------------------------------------------------

def process_dataset(
    data_path,
    save_dir,
    split,
    num_samples,
    num_workers,
    map_size_gb,
    batch_size,
    seed,
    aug_mode,
    aug_times,
):
    """Read → Point cloud sampling → TD‑SMILES generation → Placeholder / filtering → LMDB writing"""
    torch.manual_seed(seed)

    # ---------- 1. Load raw data ----------
    print("Loading raw molecules …")
    raw = load_pickle(data_path)

    if split == "test":
        mols = raw                         # List[Chem.Mol]
        smiles = [""] * len(mols)          # Placeholder
    else:
        mols = raw["rdkit_mol_cistrans_stereo"]
        smiles = raw["SMILES_nostereo"]

    n_mol = len(mols)
    print(f"Total molecules: {n_mol}")

    # ---------- 2. Initialize LMDB ----------
    os.makedirs(save_dir, exist_ok=True)
    lmdb_path = os.path.join(
        save_dir,
        f"pointcloud_{split}_samples{num_samples}_augmode{aug_mode}_augtimes{aug_times}_lmdb",
    )
    env = lmdb.open(
        lmdb_path,
        map_size=int(map_size_gb * 1024**3),
        subdir=True,
        lock=True,
        map_async=True,
        writemap=True,
        meminit=False,
        readahead=False,
    )
    print(f"LMDB will be saved to: {lmdb_path}")

    # ---------- 3. Batch processing ----------
    n_batches = math.ceil(n_mol / batch_size)
    with tqdm(total=n_batches, desc="Batches") as pbar:
        idx_global = 0
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n_mol)

            mol_batch = mols[s:e]
            smi_batch = smiles[s:e]

            # 3‑1. Parallel point cloud sampling
            pcs = Parallel(n_jobs=num_workers, backend="loky")(
                delayed(sample_pointcloud_from_mol)(
                    mol,
                    num_samples=num_samples,
                    normalize=True,
                    return_normals=True,
                ) for mol in mol_batch
            )

            # 3‑2. Generate TD‑SMILES (train/val only)
            if split != "test":
                td_batch = mol2TDsmiles(
                    mol_batch,
                    num_workers=num_workers,
                    aug_mode=aug_mode,
                    aug_times=aug_times,
                )
            else:
                td_batch = smi_batch  # Test split uses placeholder

            # ---------- 3‑3. Realign and placeholder / filter ----------
            pcs_final, mols_final, td_final = [], [], []

            if split != "test":
                stride = aug_times
                for idx, (pc_tuple, mol) in enumerate(zip(pcs, mol_batch)):
                    td_list = td_batch[idx * stride: (idx + 1) * stride]

                    if pc_tuple is None:
                        # Sampling failed: print warning and generate empty placeholder
                        print(f"[Warning] Point‑cloud sampling failed for molecule global‑idx {s + idx}.")
                        empty_pc_tuple = make_empty_pc(return_normals=True)
                        for td in td_list:
                            if td is None or td.lower() == "error":
                                td = "ERROR"
                            pcs_final.append(empty_pc_tuple)
                            mols_final.append(mol)
                            td_final.append(td)
                        continue

                    # Normal path: filter errors in TD‑SMILES
                    for td in td_list:
                        if td is None or td.lower() == "error":
                            continue
                        pcs_final.append(pc_tuple)
                        mols_final.append(mol)
                        td_final.append(td)
            else:
                # Test split: keep original, but placeholder or filter invalid point clouds
                for idx, (pc_tuple, mol) in enumerate(zip(pcs, mol_batch)):
                    if pc_tuple is None:
                        print(f"[Warning] Point‑cloud sampling failed for TEST molecule global‑idx {s + idx}.")
                        pcs_final.append(make_empty_pc(return_normals=True))
                        mols_final.append(mol)
                        td_final.append("")  # Placeholder
                        continue
                    pcs_final.append(pc_tuple)
                    mols_final.append(mol)
                    td_final.append("")  # Placeholder

            # ---------- 3‑4. Write to LMDB ----------
            if pcs_final:  # May be all filtered, but now includes placeholders, so almost never empty
                with env.begin(write=True) as txn:
                    write_batch(
                        txn,
                        idx_global,
                        pcs_final,
                        td_final,
                        mols_final,
                        test_split=(split == "test"),
                    )
                idx_global += len(pcs_final)

            pbar.update(1)

            # ---------- 3‑5. Memory cleanup ----------
            del pcs, pcs_final, mol_batch, smi_batch, td_batch, mols_final, td_final
            gc.collect()

    # ---------- 4. Wrap up ----------
    env.sync()
    env.close()
    print("Data successfully saved. Done.")


# -------------------------------------------------
# Command line parsing
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build LMDB point‑cloud dataset (improved)")
    parser.add_argument("--data_path", default='data/MOSES/MOSES2_training_val_dataset.pkl', help="Raw pickle/... path")
    parser.add_argument("--save_dir", default='data/', help="LMDB output directory")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--num_samples", type=int, default=1024, help="Number of sample points per molecule")
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--map_size", type=int, default=100, help="LMDB map_size (GiB)")
    parser.add_argument("--batch_size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug_mode", type=int, choices=[0, 1, 2], default=0, help="Data augmentation mode")
    parser.add_argument("--aug_times", type=int, default=1, help="Data augmentation multiplier")
    return parser.parse_args()


# -------------------------------------------------
# Main entry point
# -------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        data_path=args.data_path,
        save_dir=args.save_dir,
        split=args.split,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        map_size_gb=args.map_size,
        batch_size=args.batch_size,
        seed=args.seed,
        aug_mode=args.aug_mode,
        aug_times=args.aug_times,
    )
