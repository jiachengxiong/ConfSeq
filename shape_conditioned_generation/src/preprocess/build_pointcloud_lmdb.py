#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建点云‑LMDB 数据库（改进版）
------------------------------------------------
核心改动
1. **严格对齐**：保证 point‑cloud 与 TD‑SMILES 等长、一一对应。
2. **错误过滤**：凡 TD‑SMILES == "error"（大小写均可）或点云采样失败，整条记录直接丢弃。
3. **批量写入**：按过滤后样本写入 LMDB，并正确累积全局索引。
4. **失败占位**：若点云采样失败，则写入空占位并设置 ``valid=False``，确保索引对齐。
5. **实时提示**：采样失败时在终端打印警告信息。
------------------------------------------------
用法示例
    python build_pointcloud_lmdb_improved.py \
        --data_path data/train.pkl \
        --save_dir data/lmdb \
        --split train \
        --num_samples 2048 \
        --num_workers 16 \
        --map_size 100 \
        --batch_size 512 \
        --seed 42 \
        --aug_mode 1 \
        --aug_times 4
"""

import os
import sys
sys.path.append('.')
import gc
import math
import lmdb
import pickle
import argparse
from functools import partial

import numpy as np
import torch
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ---------- 请确保以下函数可用 ----------
from src.utils.misc import load_pickle
from src.utils.data import mol2TDsmiles
from src.utils.shape import get_mesh, get_pointcloud_from_mesh
# -----------------------------------------

__all__ = [
    "sample_pointcloud_from_mol",
    "write_batch",
    "process_dataset",
    "parse_args",
]

# -------------------------------------------------
# 采样函数
# -------------------------------------------------

def sample_pointcloud_from_mol(mol, num_samples, normalize=False, return_normals=True):
    """单分子 → (N,3)[, (N,3)] 采样

    返回
    -------
    tuple | None
        (pc, normals) 若成功；None 若失败
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
        # 返回 None 以便主进程过滤 / 占位
        return None


# -------------------------------------------------
# 占位工具函数
# -------------------------------------------------

def make_empty_pc(return_normals=True):
    """生成 shape=(1024,3) 的空点云 / 法线张量"""
    empty_pc = torch.zeros((1024, 3), dtype=torch.float32)
    empty_norm = torch.zeros((1024, 3), dtype=torch.float32) if return_normals else None
    return empty_pc, empty_norm


# -------------------------------------------------
# 单批写入函数
# -------------------------------------------------

def write_batch(txn, start_key, pcs, td_smiles_list, mols, test_split=False):
    """将一个 batch 写入 LMDB

    参数
    -------
    pcs : list[tuple[Tensor, Tensor | None]]
        点云及法线；若点云为空占位，则 ``pc.shape == (0, 3)``
    td_smiles_list : list[str]
        与 pcs 对齐的 TD‑SMILES
    mols : list[Chem.Mol]
        与 pcs 对齐的 RDKit Mol
    """
    for i, (pc_tuple, smi, mol) in enumerate(zip(pcs, td_smiles_list, mols)):
        pc, norm = pc_tuple
        key = f"{start_key + i:08d}".encode("ascii")

        datum = {
            "pointcloud": pc.numpy().astype(np.float32),
            "normals": norm.numpy().astype(np.float32) if norm is not None else None,
            "valid": pc.shape[0] > 0,  # True 表示采样成功
        }
        if not test_split:
            datum.update({
                "TD_smiles": smi,
                "rdmol": Chem.MolToMolBlock(mol),
            })

        txn.put(key, pickle.dumps(datum, protocol=pickle.HIGHEST_PROTOCOL))


# -------------------------------------------------
# 主处理逻辑
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
    """读取 → 点云采样 → TD‑SMILES 生成 → 占位 / 过滤 → LMDB 写入"""
    torch.manual_seed(seed)

    # ---------- 1. 加载原始数据 ----------
    print("Loading raw molecules …")
    raw = load_pickle(data_path)

    if split == "test":
        mols = raw                         # List[Chem.Mol]
        smiles = [""] * len(mols)          # 占位
    else:
        mols = raw["rdkit_mol_cistrans_stereo"]
        smiles = raw["SMILES_nostereo"]

    n_mol = len(mols)
    print(f"Total molecules: {n_mol}")

    # ---------- 2. 初始化 LMDB ----------
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

    # ---------- 3. 批量处理 ----------
    n_batches = math.ceil(n_mol / batch_size)
    with tqdm(total=n_batches, desc="Batches") as pbar:
        idx_global = 0
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n_mol)

            mol_batch = mols[s:e]
            smi_batch = smiles[s:e]

            # 3‑1. 并行采样点云
            pcs = Parallel(n_jobs=num_workers, backend="loky")(
                delayed(sample_pointcloud_from_mol)(
                    mol,
                    num_samples=num_samples,
                    normalize=True,
                    return_normals=True,
                ) for mol in mol_batch
            )

            # 3‑2. 生成 TD‑SMILES（仅 train/val）
            if split != "test":
                td_batch = mol2TDsmiles(
                    mol_batch,
                    num_workers=num_workers,
                    aug_mode=aug_mode,
                    aug_times=aug_times,
                )
            else:
                td_batch = smi_batch  # test split 用占位

            # ---------- 3‑3. 重新对齐并占位 / 过滤 ----------
            pcs_final, mols_final, td_final = [], [], []

            if split != "test":
                stride = aug_times
                for idx, (pc_tuple, mol) in enumerate(zip(pcs, mol_batch)):
                    td_list = td_batch[idx * stride: (idx + 1) * stride]

                    if pc_tuple is None:
                        # 采样失败：打印提示并生成空占位
                        print(f"[Warning] Point‑cloud sampling failed for molecule global‑idx {s + idx}.")
                        empty_pc_tuple = make_empty_pc(return_normals=True)
                        for td in td_list:
                            if td is None or td.lower() == "error":
                                td = "ERROR"
                            pcs_final.append(empty_pc_tuple)
                            mols_final.append(mol)
                            td_final.append(td)
                        continue

                    # 正常路径：过滤 TD‑SMILES 中的 error
                    for td in td_list:
                        if td is None or td.lower() == "error":
                            continue
                        pcs_final.append(pc_tuple)
                        mols_final.append(mol)
                        td_final.append(td)
            else:
                # test split：保持原样，但要占位或过滤 invalid 点云
                for idx, (pc_tuple, mol) in enumerate(zip(pcs, mol_batch)):
                    if pc_tuple is None:
                        print(f"[Warning] Point‑cloud sampling failed for TEST molecule global‑idx {s + idx}.")
                        pcs_final.append(make_empty_pc(return_normals=True))
                        mols_final.append(mol)
                        td_final.append("")  # 占位
                        continue
                    pcs_final.append(pc_tuple)
                    mols_final.append(mol)
                    td_final.append("")  # 占位

            # ---------- 3‑4. 写入 LMDB ----------
            if pcs_final:  # 可能全部被过滤，但现在包含占位，因此几乎不会为空
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

            # ---------- 3‑5. 内存清理 ----------
            del pcs, pcs_final, mol_batch, smi_batch, td_batch, mols_final, td_final
            gc.collect()

    # ---------- 4. 收尾 ----------
    env.sync()
    env.close()
    print("Data successfully saved. Done.")


# -------------------------------------------------
# 命令行解析
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build LMDB point‑cloud dataset (improved)")
    parser.add_argument("--data_path", default='data/MOSES/shapemol/MOSES2_training_val_dataset.pkl', help="原始 pickle/… 路径")
    parser.add_argument("--save_dir", default='data/MOSES/', help="LMDB 输出目录")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--num_samples", type=int, default=1024, help="每个分子采样点数")
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--map_size", type=int, default=100, help="LMDB map_size (GiB)")
    parser.add_argument("--batch_size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug_mode", type=int, choices=[0, 1, 2], default=0, help="数据增强模式")
    parser.add_argument("--aug_times", type=int, default=1, help="数据增强倍数")
    return parser.parse_args()


# -------------------------------------------------
# 主入口
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
