
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
    Minimal-IO Dataset – directly read from *pre-processed* point-cloud LMDB.

    Parameters
    ----------
    lmdb_path : str
        LMDB database directory; must contain all required fields.
    tokenizer : transformers.PreTrainedTokenizerBase, optional
        Used to encode TD_SMILES / SMILES.
    use_smiles : bool, default False
        If True, try to convert rdmol to plain SMILES; otherwise use stored TD_SMILES directly.
    max_length : int, default 512
        Maximum length for tokenizer.
    cache_keys : bool, default True
        Whether to cache all keys to `keys_cache.pkl` for faster subsequent startup.
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
            raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")

        self.lmdb_path  = lmdb_path
        self.tokenizer  = tokenizer
        self.use_smiles = config.get("use_smiles", False)
        self.max_length = max_length

        # Lazy connection (per worker)
        self.env: Optional[lmdb.Environment] = None
        self.txn = None

        # ---------- Read / cache all keys ----------
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
            raise RuntimeError(f"Empty dataset: {lmdb_path}")

        self.skip_idx: List[int] = []   # Record bad sample indices

    # ------------------------------------------------------------------ #
    # Internal tools
    # ------------------------------------------------------------------ #
    def _scan_all_keys(self) -> List[bytes]:
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn, txn.cursor() as cur:
            keys = [k for k, _ in cur]
        env.close()
        return keys

    def _connect_db(self):
        """Establish connection on demand (per worker)."""
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
        Generate canonical or random SMILES based on mode.
        mode : 'std' | 'aug'  (case insensitive)
        """
        if mode.lower() == "std":
            return " ".join(Chem.MolToSmiles(mol, canonical=True))

        # ---- Random SMILES: RDKit 2023.09+ supports doRandom=True ----
        try:
            return " ".join(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
        except TypeError:
            # Compatibility with older versions: use random root atom
            rand_root = random.randrange(mol.GetNumAtoms())
            return " ".join(Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=rand_root))

    # ------------------------------------------------------------------ #
    # PyTorch API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range.")

        self._connect_db()                       

        # If encountering a bad sample, randomly replace it
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
        # Handle lightweight records: if missing pointcloud, retrieve using pc_key / parent_key
        # -------------------------------------------------------------- #
        if "pointcloud" not in datum:
            ref_key = datum.get("pc_key") or datum.get("parent_key")
            if ref_key is None:
                raise KeyError("Lightweight record missing 'pc_key' / 'parent_key' field.")
            ref_buf = self.txn.get(ref_key)
            if ref_buf is None:
                self.skip_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.keys) - 1))
            base = pickle.loads(ref_buf)
            # Add point cloud information to datum for unified processing
            datum["pointcloud"] = base["pointcloud"]
            datum["normals"]    = base["normals"]

        # -------------------- Tensorization -------------------- #
        pc      = torch.as_tensor(datum["pointcloud"], dtype=torch.float32)
        normals = torch.as_tensor(datum["normals"],    dtype=torch.float32)

        # -------------------- Text ------------------------ #
        if "TD_smiles" not in datum or not datum["TD_smiles"]:
            # Test set: return only point cloud
            return {"pointcloud": pc, "normals": normals}

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required to process text fields.")

        td_text: str = datum["TD_smiles"]
        if self.use_smiles:
            mol = Chem.MolFromMolBlock(datum["rdmol"])
            if mol is None:
                raise RuntimeError("Unable to parse rdmol → MolBlock")

            if td_text.lstrip().startswith("<std>"):
                prefix = "<std>"
                smiles = self._make_smiles_from_mol(mol, mode="std")
            elif td_text.lstrip().startswith("<aug>"):
                prefix = "<aug>"
                smiles = self._make_smiles_from_mol(mol, mode="aug")
            else:
                # No prefix: keep old behavior
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
