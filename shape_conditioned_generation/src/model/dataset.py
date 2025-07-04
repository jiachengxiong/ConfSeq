
import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from rdkit import Chem
import gc
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# Import utility functions
from typing import List, Optional
import random
    

class PointCloudCollator():
    def __init__(self):
        pass

    def __call__(self, batch):
        pointclouds = [item['pointcloud'] for item in batch]
        normals = [item['normals'] for item in batch]
        labels = [item['labels'] for item in batch] if 'labels' in batch[0].keys() else None
        attention_masks = [item['attention_mask'] for item in batch] if 'attention_mask' in batch[0].keys() else None

        pointclouds = torch.stack(pointclouds)
        normals = torch.stack(normals)
        labels = torch.stack(labels) if labels is not None else None
        attention_masks = torch.stack(attention_masks) if attention_masks is not None else None

        if labels is not None:
            return {
                'pointcloud': pointclouds,
                'normals': normals,
                'labels': labels,
                # 'attention_mask': attention_masks
            }
        else:
            return {
                'pointcloud': pointclouds,
                'normals': normals,
                # 'attention_mask': attention_masks
            }
           

class PointCloudDataset(Dataset):
    """
    Minimal-IO Dataset – 直接从 *已处理好的* point-cloud LMDB 读取。

    Parameters
    ----------
    lmdb_path : str
        LMDB 数据库目录；必须已包含所有所需字段。
    tokenizer : transformers.PreTrainedTokenizerBase, optional
        用于编码 TD_SMILES / SMILES。
    use_smiles : bool, default False
        若 True，尝试把 rdmol 转为普通 SMILES；否则直接用存储的 TD_SMILES。
    max_length : int, default 512
        tokenizer 的最大长度。
    cache_keys : bool, default True
        是否将全部键缓存到 `keys_cache.pkl` 加速再次启动。
    """

    def __init__(
        self,
        config: dict,
        tokenizer=None,
        max_length: int = 512,
        cache_keys: bool = True,
    ):
        super().__init__()

        lmdb_path = config.get("data_dir")
        if not os.path.isdir(lmdb_path):
            raise FileNotFoundError(f"LMDB 路径不存在: {lmdb_path}")

        self.lmdb_path  = lmdb_path
        self.tokenizer  = tokenizer
        self.use_smiles = config.get("use_smiles", False)
        self.max_length = max_length

        # 延迟连接（按 worker）
        self.env: Optional[lmdb.Environment] = None
        self.txn = None

        # ---------- 读取 / 缓存全部键 ----------
        cache_file = os.path.join(lmdb_path, "keys_cache.pkl")
        if cache_keys and os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                self.keys: List[bytes] = pickle.load(f)
        else:
            self.keys = self._scan_all_keys()
            if cache_keys:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.keys, f)

        if not self.keys:
            raise RuntimeError(f"空数据集: {lmdb_path}")

        self.skip_idx: List[int] = []   # 记录坏样本索引

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _scan_all_keys(self) -> List[bytes]:
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn, txn.cursor() as cur:
            keys = [k for k, _ in cur]
        env.close()
        return keys

    def _connect_db(self):
        """按需（按 worker）建立连接。"""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

    def _make_smiles_from_mol(self, mol, mode: str) -> str:
        """
        根据 mode 生成 canonical 或 random SMILES。
        mode : 'std' | 'aug'  (大小写均可)
        """
        if mode.lower() == "std":
            return " ".join(Chem.MolToSmiles(mol, canonical=True))

        # ---- 随机 SMILES：RDKit 2023.09+ 支持 doRandom=True ----
        try:
            return " ".join(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
        except TypeError:
            # 老版本兼容：用随机根原子
            rand_root = random.randrange(mol.GetNumAtoms())
            return " ".join(Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=rand_root))

    # ------------------------------------------------------------------ #
    # PyTorch API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.keys):
            raise IndexError(f"索引 {idx} 超出范围。")

        self._connect_db()                       # 第一次真正连库

        # 若碰见坏样本，随机替换
        if idx in self.skip_idx:
            return self.__getitem__(random.randint(0, len(self.keys) - 1))

        key = self.keys[idx]
        buf = self.txn.get(key)
        if buf is None:
            self.skip_idx.append(idx)
            return self.__getitem__(random.randint(0, len(self.keys) - 1))

        try:
            datum = pickle.loads(buf)
        except Exception:
            self.skip_idx.append(idx)
            return self.__getitem__(random.randint(0, len(self.keys) - 1))

        # -------------------------------------------------------------- #
        # 处理轻量记录：若缺失 pointcloud，则用 pc_key / parent_key 取回
        # -------------------------------------------------------------- #
        if "pointcloud" not in datum:
            ref_key = datum.get("pc_key") or datum.get("parent_key")
            if ref_key is None:
                raise KeyError("轻量记录缺少 'pc_key' / 'parent_key' 字段。")
            ref_buf = self.txn.get(ref_key)
            if ref_buf is None:
                self.skip_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.keys) - 1))
            base = pickle.loads(ref_buf)
            # 把点云信息补进 datum，后续流程统一
            datum["pointcloud"] = base["pointcloud"]
            datum["normals"]    = base["normals"]

        # -------------------- Tensor 化 -------------------- #
        pc      = torch.as_tensor(datum["pointcloud"], dtype=torch.float32)
        normals = torch.as_tensor(datum["normals"],    dtype=torch.float32)

        # -------------------- 文本 ------------------------ #
        if "TD_smiles" not in datum or not datum["TD_smiles"]:
            # 测试集: 仅返回点云
            return {"pointcloud": pc, "normals": normals}

        if self.tokenizer is None:
            raise ValueError("需要 tokenizer 以处理文本字段。")

        td_text: str = datum["TD_smiles"]
        if self.use_smiles:
            mol = Chem.MolFromMolBlock(datum["rdmol"])
            if mol is None:
                raise RuntimeError("无法解析 rdmol → MolBlock")

            if td_text.lstrip().startswith("<std>"):
                prefix = "<std>"
                smiles = self._make_smiles_from_mol(mol, mode="std")
            elif td_text.lstrip().startswith("<aug>"):
                prefix = "<aug>"
                smiles = self._make_smiles_from_mol(mol, mode="aug")
            else:
                # 无前缀：保持旧行为
                prefix = ""
                smiles = self._make_smiles_from_mol(mol, mode="std")

            text = f"{prefix} {smiles}"
        else:
            text = td_text
            
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "labels"     : enc["input_ids"].squeeze(0),      #  (L,)
            "attention_mask": enc["attention_mask"].squeeze(0), #  (L,)
            "pointcloud"    : pc,                               # (N,3)
            "normals"       : normals,                          # (N,3)
        }

    # ------------------------------------------------------------------ #
    def __del__(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.txn = None
        gc.collect()
