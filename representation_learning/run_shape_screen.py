# -*- coding: utf-8 -*-
"""
ConfSeq-based 3D shape similarity search (PubChem / ZINC) - CLI

Usage examples
--------------
1) Search similar molecules in PubChem using a SMILES string:

    python run_shape_screen.py \
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \
        --db pubchem \
        --topk 50 \
        --out-prefix results/pubchem_aspirin

2) Search in ZINC using a single-molecule SDF file:

    python run_shape_screen.py \
        --sdf ./query_mol.sdf \
        --db zinc \
        --topk 20 \
        --out-prefix results/zinc_query

3) Search in PubChem using a MOL file:

    python run_shape_screen.py \
        --mol ./query.mol \
        --db pubchem \
        --topk 30

The script will:
    * Initialize the ConfSeq encoder and FAISS indices.
    * Convert the query molecule into ConfSeq and then an embedding.
    * Perform k-NN search in the chosen database (PubChem or ZINC).
    * Save a CSV report and an SDF file containing 3D conformers of hits.
"""

from transformers import set_seed
set_seed(42)

import os
import sys
sys.path.append('../')
import json
import time
import shutil
import argparse
from typing import Optional, List, Tuple
from collections import OrderedDict
from pathlib import Path

# RDKit / ML
import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops, Draw, AllChem
from rdkit.Chem.rdDepictor import Compute2DCoords
import copy
import pandas as pd

# Faiss
import faiss

# NEW: Hugging Face Hub
from huggingface_hub import hf_hub_download

# ---------------- Env / Constants ----------------
# PubChem
FAISS_INDEX_PATH_PUBCHEM = "./data/pubchem_faiss.index"
PUBCHEM_EMBED_PATH = "./data/PubChem_embedding.npz"      # Build index from this npz if FAISS index does not exist
PUBCHEM_CID_PATH = "./data/PubChem_CID.npy"
PUBCHEM_CONFSEQ_JSON = "./data/PubChem_ID_ConfSeq_dic.json"

# ZINC
FAISS_INDEX_PATH_ZINC = "./data/zinc_faiss.index"
ZINC_EMBED_PATH = "./data/zinc_confseq_embeds.npy"            # Build index from this .npy/.npz if FAISS index does not exist
ZINC_CSV_PATH = "./data/zinc_instock_tdsmiles_test.csv"       # Must contain columns: ZINC_ID, canonical_smiles, td_smiles
ZINC_BASE_URL = "https://zinc20.docking.org/substances"       # Can be replaced with the real ZINC22 URL if needed

HF_REPO_ID = "Oopstom/confseq-shape-screen"
HF_MODEL_FILENAME = "model_epoch_1.pth"

# ---------------- ConfSeq helpers ----------------
from demo.ConfSeq import (
    get_ConfSeq_pair_from_mol,
    randomize_mol,
    replace_angle_brackets_with_line,
    get_mol_from_ConfSeq_pair,
    remove_degree_in_molblock
)

def convert_tdsmiles_to_mol(td_smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a ConfSeq / ConfSeq-like string to an RDKit Mol with 3D coordinates.
    """
    try:
        in_smiles = replace_angle_brackets_with_line(td_smiles)
        generated_mol = get_mol_from_ConfSeq_pair(in_smiles, td_smiles)
        generated_mol = Chem.MolFromMolBlock(
            remove_degree_in_molblock(Chem.MolToMolBlock(generated_mol))
        )
        if generated_mol is not None:
            return generated_mol
    except Exception as e:
        print(f"Failed to convert ConfSeq to Mol: {e}")
    return None

def rm_invalid_chirality(mol: Chem.Mol) -> Chem.Mol:
    """
    Clear chiral tags for atoms involved in three rings (often problematic).
    """
    mol = copy.deepcopy(mol)
    rings = rdmolops.GetSymmSSSR(mol)
    atom_in_rings_count = {}
    for ring in rings:
        for atom_idx in ring:
            atom_in_rings_count[atom_idx] = atom_in_rings_count.get(atom_idx, 0) + 1
    atoms_in_3_rings = [atom for atom, count in atom_in_rings_count.items() if count == 3]
    for atom_idx in atoms_in_3_rings:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)
    return mol

def neutralize_charge(mol: Chem.Mol) -> Chem.Mol:
    """
    Heuristically neutralize formal charges on atoms that do not have an
    oppositely charged neighbor.

    Note
    ----
    This is a simple heuristic and may not be chemically perfect. For complex
    cases, upstream standardization is recommended.
    """
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0:
            has_opposite_neighbor = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetFormalCharge() == -charge:
                    has_opposite_neighbor = True
                    break
            if not has_opposite_neighbor:
                atom.SetFormalCharge(0)
                # Simple correction for explicit hydrogens; complex cases should be handled upstream
                try:
                    num_h = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(max(0, num_h - charge))
                except Exception:
                    pass
    Chem.SanitizeMol(mol)
    return mol

def get_ConfSeq(query_mol: Chem.Mol) -> str:
    """
    Generate the ConfSeq / ConfSeq representation for a given molecule.
    """
    if query_mol is not None:
        try:
            query_mol = neutralize_charge(query_mol)
            query_mol = rm_invalid_chirality(query_mol)
            query_mol = randomize_mol(query_mol)
            Chem.MolToSmiles(query_mol)
            query_mol = Chem.RenumberAtoms(
                query_mol,
                eval(query_mol.GetProp('_smilesAtomOutputOrder'))
            )
            Chem.MolToSmiles(query_mol, canonical=False)
            query_mol = Chem.RenumberAtoms(
                query_mol,
                eval(query_mol.GetProp('_smilesAtomOutputOrder'))
            )
            in_smiles, TD_smiles = get_ConfSeq_pair_from_mol(query_mol)
            TD_smiles = TD_smiles.replace('<180>', '<-180>')
            return TD_smiles
        except Exception:
            return ''
    return ''

def convert_confseq_to_smiles(confseq: str) -> str:
    """
    Approximate conversion from ConfSeq to a regular SMILES string.

    This is intended for 2D depiction and text display, and does NOT guarantee
    that the resulting SMILES fully preserves the original 3D information.
    """
    if not confseq or confseq == 'N/A':
        return 'N/A'
    try:
        in_smiles = replace_angle_brackets_with_line(confseq)
        in_smiles = in_smiles.replace('^ |', '').replace(' !', '')
        in_smiles = in_smiles.replace('/ -', '/').replace('\\ -', '\\')
        in_smiles = ''.join(in_smiles.split(' '))
        mol = Chem.MolFromSmiles(in_smiles)
        return Chem.MolToSmiles(mol) if mol else "Invalid SMILES generated from ConfSeq"
    except Exception as e:
        return f"Error during conversion: {e}"

def generate_smiles_image(smiles: str, output_path: str) -> bool:
    """
    Generate a 2D image of a SMILES string and save it as an image file.
    """
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Compute2DCoords(mol)
        Draw.MolToFile(mol, output_path, size=(300, 300))
        return True
    except Exception as e:
        print(f"Error generating image for SMILES '{smiles}': {e}")
        return False

# ---------------- Model (encoder) ----------------
class CustomBartEncoder(nn.Module):
    """
    Wrapper that exposes only the BART encoder to produce sequence embeddings.
    """
    def __init__(self, bart):
        super().__init__()
        self.bart_model = bart

    def forward(self, input_ids, attention_mask=None):
        return self.bart_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

def get_embedding_for_seq(
    model: nn.Module,
    confseq: str,
    vocab_dict: dict,
    device: torch.device
) -> Optional[np.ndarray]:
    """
    Tokenize a ConfSeq string using the given vocab and obtain a single
    fixed-length embedding (mean-pooled hidden states).
    """
    if not confseq:
        return None
    token_ids = [vocab_dict.get(token, vocab_dict['<unk>']) for token in confseq.split(' ')]
    token_tensor = torch.LongTensor([token_ids]).to(device)
    with torch.no_grad():
        emb = model(token_tensor).mean(dim=1).squeeze().cpu().numpy()
    return emb

# ---------------- Globals ----------------
bart_encoder_model: Optional[nn.Module] = None
device: Optional[torch.device] = None
vocab_dict: Optional[dict] = None

# PubChem assets
faiss_index_pubchem: Optional[faiss.Index] = None
pubchem_cid: Optional[np.ndarray] = None
PubChem_ID_ConfSeq_dic: Optional[dict] = None

# ZINC assets
faiss_index_zinc: Optional[faiss.Index] = None
zinc_ids: Optional[np.ndarray] = None          # np.array of str, aligned to embeddings row order
zinc_smiles: Optional[np.ndarray] = None       # canonical_smiles
zinc_tdsmiles: Optional[np.ndarray] = None     # td_smiles

# ---------------- Loading helpers ----------------
def _build_or_load_faiss(index_path: str, embed_path: str) -> faiss.Index:
    """
    Load or build a Faiss index.

    Behavior
    --------
    * If `index_path` exists: load the index.
    * If not: build an index from `embed_path` and save it to `index_path`.

    The embedding file can be:
      - .npy : direct 2D ndarray
      - .npz : default to 'arr_0'; if there is only one array, use it; otherwise raise.
    """
    if os.path.exists(index_path):
        print(f"[FAISS] Loading index: {index_path}")
        idx = faiss.read_index(index_path)
        print(f"[FAISS] ntotal={idx.ntotal}")
        return idx

    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")

    print(f"[FAISS] Building index from: {embed_path}")
    obj = np.load(embed_path, allow_pickle=False)

    if isinstance(obj, np.lib.npyio.NpzFile):
        if "arr_0" in obj.files:
            arr = obj["arr_0"]
        elif len(obj.files) == 1:
            arr = obj[obj.files[0]]
        else:
            raise ValueError(
                f"NPZ contains multiple arrays {obj.files}; "
                f"please keep a single array or ensure the main array is under key 'arr_0'."
            )
    else:
        arr = obj

    arr = np.asarray(arr, dtype="float32")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={arr.shape}")

    d = arr.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(arr)
    faiss.write_index(idx, index_path)
    print(f"[FAISS] Built & saved: {index_path}, ntotal={idx.ntotal}")
    return idx

def _load_pubchem_assets():
    """
    Load FAISS index and mapping files for PubChem.
    """
    global faiss_index_pubchem, pubchem_cid, PubChem_ID_ConfSeq_dic
    faiss_index_pubchem = _build_or_load_faiss(
        FAISS_INDEX_PATH_PUBCHEM,
        PUBCHEM_EMBED_PATH
    )

    if not os.path.exists(PUBCHEM_CID_PATH):
        raise FileNotFoundError(f"Required file not found: {PUBCHEM_CID_PATH}")
    pubchem_cid = np.load(PUBCHEM_CID_PATH)

    if not os.path.exists(PUBCHEM_CONFSEQ_JSON):
        raise FileNotFoundError(f"Required file not found: {PUBCHEM_CONFSEQ_JSON}")
    with open(PUBCHEM_CONFSEQ_JSON, 'r', encoding='utf-8') as f:
        PubChem_ID_ConfSeq_dic = json.load(f)

def _load_zinc_assets():
    """
    Load FAISS index and CSV mappings for the ZINC database.
    """
    global faiss_index_zinc, zinc_ids, zinc_smiles, zinc_tdsmiles
    faiss_index_zinc = _build_or_load_faiss(
        FAISS_INDEX_PATH_ZINC,
        ZINC_EMBED_PATH
    )

    if not os.path.exists(ZINC_CSV_PATH):
        raise FileNotFoundError(f"ZINC mapping CSV not found: {ZINC_CSV_PATH}")

    usecols = ["ZINC_ID", "canonical_smiles", "td_smiles"]
    df = pd.read_csv(ZINC_CSV_PATH, usecols=usecols, dtype=str, low_memory=False)

    zinc_ids = df["ZINC_ID"].fillna("").astype(str).to_numpy()
    zinc_smiles = df["canonical_smiles"].fillna("").astype(str).to_numpy()
    zinc_tdsmiles = df["td_smiles"].fillna("").astype(str).to_numpy()

    print(f"[ZINC] CSV loaded: {len(zinc_ids)} rows (aligned to embeddings).")
    if faiss_index_zinc.ntotal != len(zinc_ids):
        raise ValueError(
            f"[ZINC] Row mismatch: embeddings={faiss_index_zinc.ntotal} vs CSV={len(zinc_ids)}. "
            f"Please ensure the embedding matrix row order/size matches the CSV row order."
        )

def load_model_and_data():
    """
    Initialize:
      * Device
      * ConfSeq vocabulary and BART-based encoder
      * PubChem FAISS index + mappings
      * ZINC FAISS index + mappings (best-effort; will warn on failure)
    """
    global bart_encoder_model, vocab_dict, device
    print("Initializing encoder and databases...")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build vocabulary and encoder model
    vocab = [chr(i) for i in range(33, 127)]
    [vocab.append('<' + str(i) + '>') for i in range(-180, 180)]
    vocab.extend(['<mask>', '<unk>', '<sos>', '<eos>', '<pad>'])
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}

    config = BartConfig(
        pad_token_id=vocab.index('<pad>'),
        eos_token_id=vocab.index('<eos>'),
        sos_token_id=vocab.index('<sos>'),
        encoder_layers=6,
        encoder_attention_heads=8,
        decoder_layers=0,
        decoder_attention_heads=0,
        d_model=256,
        vocab_size=len(vocab)
    )
    bart_base = BartForConditionalGeneration(config=config)
    bart_encoder_model = CustomBartEncoder(bart_base)

    # ==== 这里开始：从 Hugging Face 下载并加载权重 ====
    try:
        print(f"Downloading model weights from Hugging Face: {HF_REPO_ID}/{HF_MODEL_FILENAME}")
        state_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME
        )
        checkpoint = torch.load(state_path, map_location='cpu')
        # 如果你的 ckpt 里是 {"model_state_dict": ...} 之类，这里可以按需调整
        if isinstance(checkpoint, dict) and all(
            not k.startswith("model.") for k in checkpoint.keys()
        ):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
        bart_encoder_model.load_state_dict(new_state, strict=False)
        print("Model weights loaded from Hugging Face.")
    except Exception as e:
        print(
            f"Warning: failed to load model weights from Hugging Face "
            f"({HF_REPO_ID}/{HF_MODEL_FILENAME}): {e}"
        )
        print("The encoder will remain randomly initialized (untrained).")

    bart_encoder_model.to(device)
    bart_encoder_model.eval()

    # Preload PubChem
    _load_pubchem_assets()

    # Try to preload ZINC (optional; failure is non-fatal)
    try:
        _load_zinc_assets()
    except Exception as e:
        print(f"[ZINC] Skipped at init: {e}")

    print("Initialization complete.")

# ---------------- Core search ----------------
def _faiss_search(
    index: faiss.Index,
    query_vec: np.ndarray,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform FAISS search.

    Returns
    -------
    distances : np.ndarray
        Euclidean distances to the nearest neighbors (sqrt of L2^2 reported by FAISS).
    indices : np.ndarray
        Row indices of the nearest neighbors in the embedding matrix.
    """
    if index.ntotal == 0:
        raise RuntimeError("FAISS index is empty (ntotal=0).")

    k = min(int(top_k), index.ntotal)
    q = np.asarray([query_vec], dtype="float32")

    start = time.perf_counter()
    dist2, idx = index.search(q, k)
    elapsed = time.perf_counter() - start
    print(f"[FAISS] search completed in {elapsed:.4f} seconds (k={k}, ntotal={index.ntotal})")

    return np.sqrt(dist2[0]), idx[0]

def _handle_pubchem(
    topk_indices: np.ndarray,
    topk_distances: np.ndarray,
    sdf_writer: Chem.SDWriter,
    top_k: int
):
    """
    Build result rows and SDF entries for PubChem hits.
    """
    results = []
    for i, (row_idx, dist) in enumerate(zip(topk_indices, topk_distances)):
        cid = str(int(pubchem_cid[row_idx]))
        similarity = 1.0 / (1.0 + float(dist))
        result_confseq = PubChem_ID_ConfSeq_dic.get(cid, '')

        smiles = convert_confseq_to_smiles(result_confseq) if result_confseq else ''
        results.append({
            "Rank": i + 1,
            "PubChem_CID": cid,
            "Similarity": f"{similarity:.4f}",
            "PubChem_URL": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "SMILES": smiles or "N/A"
        })

        # If we have ConfSeq for this CID, convert back to 3D and write to SDF
        if result_confseq:
            mol_3d = convert_tdsmiles_to_mol(result_confseq)
            if mol_3d:
                mol_3d.SetProp("_Name", f"Rank_{i+1}_CID_{cid}")
                mol_3d.SetProp("Rank", str(i + 1))
                mol_3d.SetProp("PubChem_CID", cid)
                mol_3d.SetProp("Similarity", f"{similarity:.4f}")
                sdf_writer.write(mol_3d)
    return results

def _handle_zinc(
    topk_indices: np.ndarray,
    topk_distances: np.ndarray,
    sdf_writer: Chem.SDWriter,
    top_k: int
):
    """
    Build result rows and SDF entries for ZINC hits.
    """
    if any(x is None for x in [zinc_ids, zinc_smiles, zinc_tdsmiles]):
        raise RuntimeError("ZINC assets not loaded. Please ensure ZINC index/CSV are available.")

    results = []
    for i, (row_idx, dist) in enumerate(zip(topk_indices, topk_distances)):
        zid = zinc_ids[row_idx]
        cano = zinc_smiles[row_idx] or ""
        td = zinc_tdsmiles[row_idx] or ""
        similarity = 1.0 / (1.0 + float(dist))

        results.append({
            "Rank": i + 1,
            "ZINC_ID": zid,
            "Similarity": f"{similarity:.4f}",
            "ZINC_URL": f"{ZINC_BASE_URL}/{zid}",
            "SMILES": cano if cano else "N/A"
        })

        # If CSV contains td_smiles, reconstruct 3D and write to SDF
        if td:
            mol_3d = convert_tdsmiles_to_mol(td)
            if mol_3d:
                mol_3d.SetProp("_Name", f"Rank_{i+1}_{zid}")
                mol_3d.SetProp("Rank", str(i + 1))
                mol_3d.SetProp("ZINC_ID", zid)
                mol_3d.SetProp("Similarity", f"{similarity:.4f}")
                sdf_writer.write(mol_3d)
    return results

def run_similarity_search_local(
    query_mol: Chem.Mol,
    top_k: int,
    db: str,
    out_prefix: str
):
    """
    Run local (CLI) similarity search.

    Parameters
    ----------
    query_mol : Chem.Mol
        Single RDKit molecule to be used as the query.
    top_k : int
        Number of nearest neighbors to retrieve.
    db : {'pubchem', 'zinc'}
        Database to search.
    out_prefix : str
        Prefix for output CSV and SDF files.

    Returns
    -------
    rows : list[dict]
        Result rows written to CSV.
    csv_report_path : str
        Path to the CSV report.
    sdf_report_path : str
        Path to the SDF file containing hit conformers.
    """
    if query_mol is None:
        raise ValueError("query_mol is None.")

    confseq = get_ConfSeq(query_mol)
    if not confseq:
        raise ValueError("Could not generate ConfSeq for the input molecule.")

    query_embed = get_embedding_for_seq(bart_encoder_model, confseq, vocab_dict, device)
    if query_embed is None:
        raise ValueError("Could not generate embedding for the input molecule.")

    # Prepare output files
    out_prefix = Path(out_prefix)
    csv_report_path = out_prefix.with_suffix(".csv")
    sdf_report_path = out_prefix.with_suffix(".sdf")
    sdf_writer = Chem.SDWriter(str(sdf_report_path))

    try:
        if db == 'pubchem':
            if faiss_index_pubchem is None or pubchem_cid is None or PubChem_ID_ConfSeq_dic is None:
                raise RuntimeError("PubChem assets not loaded.")
            print("Searching PubChem (Faiss)...")
            topk_distances, topk_indices = _faiss_search(faiss_index_pubchem, query_embed, top_k)
            rows = _handle_pubchem(topk_indices, topk_distances, sdf_writer, top_k)

        elif db == 'zinc':
            global faiss_index_zinc
            if faiss_index_zinc is None or zinc_ids is None:
                _load_zinc_assets()
            if faiss_index_zinc is None:
                raise RuntimeError("ZINC assets not available.")
            print("Searching ZINC (Faiss)...")
            topk_distances, topk_indices = _faiss_search(faiss_index_zinc, query_embed, top_k)
            rows = _handle_zinc(topk_indices, topk_distances, sdf_writer, top_k)

        else:
            raise ValueError("Invalid 'db' value. Use 'pubchem' or 'zinc'.")

        sdf_writer.close()
        print(f"SDF report created: {sdf_report_path}")

        df = pd.DataFrame(rows)
        df.to_csv(csv_report_path, index=False, encoding='utf-8')
        print(f"CSV report created: {csv_report_path}")

        return rows, str(csv_report_path), str(sdf_report_path)

    finally:
        try:
            sdf_writer.close()
        except Exception:
            pass

# ---------------- CLI helpers ----------------
def load_single_mol_from_sdf(path: str) -> Chem.Mol:
    """
    Load a single molecule from an SDF file.

    This function enforces that exactly one valid molecule is present in the file.
    """
    suppl = Chem.SDMolSupplier(path)
    mols = [m for m in suppl if m is not None]
    if len(mols) == 0:
        raise ValueError(f"No valid molecule found in SDF file: {path}")
    if len(mols) > 1:
        raise ValueError(f"More than one molecule found in SDF file: {path}. Please provide exactly one.")
    return mols[0]

def load_single_mol_from_molfile(path: str) -> Chem.Mol:
    """
    Load a single molecule from a MOL file (V2000/V3000).
    """
    mol = Chem.MolFromMolFile(path)
    if mol is None:
        raise ValueError(f"Failed to read MOL file: {path}")
    return mol

def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfSeq-based 3D shape similarity search (PubChem / ZINC) - CLI"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smiles",
        type=str,
        help="Input SMILES string (single molecule)."
    )
    group.add_argument(
        "--sdf",
        type=str,
        help="Input SDF file path (must contain exactly one molecule)."
    )
    group.add_argument(
        "--mol",
        type=str,
        help="Input MOL file path (V2000/V3000, single molecule)."
    )

    parser.add_argument(
        "--db",
        type=str,
        choices=["pubchem", "zinc"],
        required=True,
        help="Database to search: 'pubchem' or 'zinc'."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top similar molecules (1-1000)."
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="similarity_results",
        help="Output file prefix for CSV/SDF. Default: similarity_results"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.topk < 1 or args.topk > 1000:
        raise ValueError("topk must be in [1, 1000].")

    # Initialize encoder model and databases (FAISS indices, mapping files, etc.)
    load_model_and_data()

    # Read input molecule (must be a single molecule)
    if args.smiles is not None:
        print(f"Reading molecule from SMILES: {args.smiles}")
        mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            raise ValueError("Failed to parse SMILES string.")
        # Generate a 3D conformer for potential SDF writing (simple embedding + MMFF optimization)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

    elif args.sdf is not None:
        print(f"Reading molecule from SDF file: {args.sdf}")
        mol = load_single_mol_from_sdf(args.sdf)

    elif args.mol is not None:
        print(f"Reading molecule from MOL file: {args.mol}")
        mol = load_single_mol_from_molfile(args.mol)

    else:
        raise ValueError("No input molecule provided.")

    # Run similarity search
    rows, csv_path, sdf_path = run_similarity_search_local(
        query_mol=mol,
        top_k=args.topk,
        db=args.db,
        out_prefix=args.out_prefix
    )

    # Print a concise result table to stdout
    print("\nTop hits:")
    if args.db == "pubchem":
        header = ["Rank", "PubChem_CID", "Similarity", "PubChem_URL", "SMILES"]
    else:
        header = ["Rank", "ZINC_ID", "Similarity", "ZINC_URL", "SMILES"]

    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r.get(h, "")) for h in header))

    print(f"\nCSV saved to: {csv_path}")
    print(f"SDF saved to: {sdf_path}")

if __name__ == "__main__":
    main()
