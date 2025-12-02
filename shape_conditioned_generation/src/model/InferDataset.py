# pointcloud_onthefly_dataset.py
import random, torch, gc
from typing import List, Dict, Optional, Callable
from torch.utils.data import Dataset
from rdkit import Chem

# 引入用户已实现的采样函数（确保在 PYTHONPATH）
from src.preprocess.build_pointcloud_lmdb import sample_pointcloud_from_mol

__all__ = ["PointCloudCollator", "PointCloudDataset"]


# ----------------------------------------------------------------------
# 1. Collator：与训练阶段保持一致，仅删去注释字段
# ----------------------------------------------------------------------
class PointCloudCollator:
    """将 batch 中同名项按 0 维堆叠；labels/attention_mask 若不存在则忽略。"""
    def __call__(self, batch):
        pointclouds = torch.stack([b["pointcloud"] for b in batch])   # (B,N,3)
        normals     = torch.stack([b["normals"]    for b in batch])   # (B,N,3)

        out = {"pointcloud": pointclouds, "normals": normals}
        return out


# ----------------------------------------------------------------------
# 2. Dataset：Mol → Point-Cloud（即时采样）
# ----------------------------------------------------------------------
class PointCloudDataset(Dataset):
    """
    适用于 *推断 / 采样阶段* 的轻量级数据集。

    Parameters
    ----------
    mol_list     : List[rdkit.Chem.Mol]
        输入分子对象列表。
    num_samples  : int, default 1024
        每个分子采样点数。
    normalize_pc : bool, default True
        是否对点云进行中心化。
    max_length   : int, default 512
        Tokenizer 最大长度。
    cache        : bool, default True
        是否缓存第一次采样结果，以避免多 epoch 重采样。
    """

    def __init__(
        self,
        mol_list    : List[Chem.Mol],
        num_samples : int  = 1024,
        normalize_pc: bool = True,
        cache       : bool = True,
    ):
        if not isinstance(mol_list, list) or not mol_list:
            raise ValueError("mol_list 必须为非空 List[rdkit.Chem.Mol].")
        self.mol_list     = mol_list
        self.num_samples  = num_samples
        self.normalize_pc = normalize_pc
        self.enable_cache = cache
        self._cache: Dict[int, Dict] = {}   # idx -> sample

    # -------------------------------------------------- #
    # 公共 API
    # -------------------------------------------------- #
    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, idx: int):
        # ---------------- 1. 读取 / 缓存 ---------------- #
        if self.enable_cache and idx in self._cache:
            return self._cache[idx]

        mol = self.mol_list[idx]
        pc_tuple = sample_pointcloud_from_mol(
            mol,
            num_samples=self.num_samples,
            normalize=self.normalize_pc,
            return_normals=True,
        )
        if pc_tuple is None:
            # 若采样失败，返回全零占位；推断阶段可自行过滤
            pc, normals = self._make_empty_pc()
        else:
            pc, normals = pc_tuple

        sample = {
            "pointcloud": pc,        # Tensor (N,3)
            "normals"   : normals,   # Tensor (N,3)
        }

        # ---------------- 3. 写入缓存 ------------------- #
        if self.enable_cache:
            self._cache[idx] = sample
        return sample

    # -------------------------------------------------- #
    # 内部工具
    # -------------------------------------------------- #
    @staticmethod
    def _make_empty_pc():
        empty_pc  = torch.zeros((1024, 3), dtype=torch.float32)
        empty_nrm = torch.zeros_like(empty_pc)
        return empty_pc, empty_nrm

    # -------------------------------------------------- #
    def __del__(self):
        self._cache.clear()
        gc.collect()
