# -*- coding: utf-8 -*-
"""
ConfSeq embedding pipeline

Functionality
-------------
1. Instantiate the BART encoder model using the ConfSeq vocabulary;
2. Load encoder weights from a trained checkpoint;
3. Batch-encode ConfSeq sequences from a specified CSV column into molecular embeddings;
4. Save the resulting embeddings as a .npy file for downstream FAISS indexing or analysis.

Typical Usage
-------------
Integrated with the ZINC processing pipeline:
    1) First generate TD-SMILES (ConfSeq) using the ZINC pipeline, producing a CSV file with:
           ZINC_ID, input_smiles, canonical_smiles, td_smiles
    2) Then use this script to batch-encode the `td_smiles` column and obtain:
           ConfSeq embedding array (.npy)

Dependencies
------------
- torch
- transformers
- numpy
- pandas
- tqdm
"""

import os
import sys
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartConfig


# ============================================================
# 1. ConfSeq vocabulary and token mapping
# ============================================================
def build_confseq_vocab():
    """
    Build the ConfSeq vocabulary consistent with the training setup.
    - Visible ASCII characters: chr(33) ~ chr(126)
    - Angle tokens: "<-180>" ... "<179>"
    - Special tokens: <mask>, <unk>, <sos>, <eos>, <pad>
    """
    vocab: List[str] = [chr(i) for i in range(33, 127)]  # Visible ASCII characters

    # Angle tokens: <-180> ... <179>
    for i in range(-180, 180):
        vocab.append("<" + str(i) + ">")

    vocab.extend(["<mask>", "<unk>", "<sos>", "<eos>", "<pad>"])

    vocab_dict = {tok: idx for idx, tok in enumerate(vocab)}
    return vocab, vocab_dict


# ============================================================
# 2. BART encoder wrapper
# ============================================================
class CustomBartEncoder(nn.Module):
    """
    A lightweight wrapper that exposes only the encoder output
    for representation learning and embedding extraction.
    """

    def __init__(self, bart_model: BartForConditionalGeneration):
        super().__init__()
        self.bart_model = bart_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Shape: [batch_size, seq_len, hidden_dim]
        return outputs.last_hidden_state


def build_bart_encoder(vocab_size: int, pad_id: int, eos_id: int, sos_id: int):
    """
    Build a BART encoder model using the same configuration as the original training setup.
    The decoder is disabled since only the encoder is used for embedding extraction.
    """
    config = BartConfig()

    config.pad_token_id = pad_id
    config.eos_token_id = eos_id
    config.sos_token_id = sos_id
    config.forced_eos_token_id = None

    # Encoder architecture
    config.encoder_layers = 6
    config.encoder_attention_heads = 8

    # Decoder is not used in this project
    config.decoder_layers = 0
    config.decoder_attention_heads = 0

    # Hidden size & vocabulary size
    config.d_model = 256
    config.vocab_size = vocab_size

    bart = BartForConditionalGeneration(config=config)
    encoder = CustomBartEncoder(bart)
    return encoder


def load_checkpoint_to_encoder(encoder: nn.Module, ckpt_path: str):
    """
    Load encoder weights from a checkpoint file.
    Automatically removes the 'module.' prefix if the checkpoint was saved with DataParallel.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    encoder.load_state_dict(new_state_dict, strict=True)


# ============================================================
# 3. ConfSeq Dataset / Collate / Pooling
# ============================================================
class ConfSeqDataset(Dataset):
    """
    A lightweight Dataset for ConfSeq tokenized sequences.
    """

    def __init__(self, seq_list, vocab_dict, pad_id, unk_id):
        self.seq_list = list(seq_list)
        self.vocab_dict = vocab_dict
        self.pad_id = pad_id
        self.unk_id = unk_id

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = str(self.seq_list[idx]) if self.seq_list[idx] is not None else ""
        toks = seq.strip().split()

        if len(toks) == 0:
            # Empty sequence: use a single <pad> token so that mean pooling yields a zero vector
            ids = [self.pad_id]
        else:
            ids = [self.vocab_dict.get(tok, self.unk_id) for tok in toks]

        return torch.tensor(ids, dtype=torch.long)


def batch_collate_fn(batch_tensors, pad_id: int):
    """
    Pad a list of 1D LongTensors into a batch tensor of shape [B, L_max],
    and generate the corresponding attention mask.

    Returns:
        input_ids: [B, L_max]
        attention_mask: [B, L_max] where pad positions are 0 and valid tokens are 1
    """
    input_ids = pad_sequence(batch_tensors, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask


def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling over valid tokens using attention_mask:
        embedding = sum(h_t * mask_t) / (sum(mask_t) + eps)

    Parameters:
        last_hidden_state: [B, L, D]
        attention_mask:    [B, L] (0/1)

    Returns:
        pooled_output:     [B, D]
    """
    # [B, L, 1] -> broadcast to [B, L, D]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)  # [B, D]
    sum_mask = mask_expanded.sum(dim=1)  # [B, D], each element equals the valid token length

    pooled_output = sum_embeddings / (sum_mask + 1e-8)
    return pooled_output


# ============================================================
# 4. Batch embedding extraction
# ============================================================
def get_embeddings_batch(
    encoder_model: nn.Module,
    sequences,
    vocab_dict,
    pad_id: int,
    unk_id: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda"),
):
    """
    Encode a batch of ConfSeq sequences and return a tensor of shape [N, D].

    Parameters
    ----------
    encoder_model : CustomBartEncoder
        The initialized encoder model.
    sequences : Iterable[str]
        A list of ConfSeq strings (e.g., from a DataFrame column).
    vocab_dict : dict
        Token-to-id mapping.
    pad_id : int
        ID of the <pad> token.
    unk_id : int
        ID of the <unk> token.
    batch_size : int
        DataLoader batch size.
    device : torch.device
        Computation device.

    Returns
    -------
    all_embeds : torch.Tensor
        Tensor of shape [N, hidden_dim] stored on CPU.
    """
    ds = ConfSeqDataset(sequences, vocab_dict=vocab_dict, pad_id=pad_id, unk_id=unk_id)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: batch_collate_fn(b, pad_id=pad_id),
    )

    encoder_model.to(device)
    encoder_model.eval()

    all_embeds = []

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(
            dl, desc="Encoding ConfSeq batches", ncols=100
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # Encoder output shape: [B, L, D]
            last_hidden = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(last_hidden, attention_mask)  # [B, D]

            all_embeds.append(pooled.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)  # [N, D]
    return all_embeds


# ============================================================
# 5. One-click CSV-based encoding interface
# ============================================================
def encode_confseq_from_csv(
    input_csv: str,
    confseq_column: str,
    ckpt_path: str,
    output_npy: str,
    batch_size: int = 2048,
    device_str: str = "cuda",
):
    """
    Load ConfSeq sequences from a specified CSV column, generate embeddings
    using a BART encoder, and save them as a .npy file.

    Typical use cases:
        - Encode the `td_smiles` column produced by the ZINC pipeline;
        - Encode any custom ConfSeq library for downstream screening.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    confseq_column : str
        Column name that stores ConfSeq strings (e.g., "td_smiles").
    ckpt_path : str
        Path to the encoder checkpoint (.pth).
    output_npy : str
        Path to the output .npy file storing [N, D] float32 embeddings.
    batch_size : int
        Batch size for DataLoader.
    device_str : str
        Device string, either "cuda" or "cpu".
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1) Build vocabulary and model
    vocab, vocab_dict = build_confseq_vocab()
    pad_id = vocab.index("<pad>")
    eos_id = vocab.index("<eos>")
    sos_id = vocab.index("<sos>")
    unk_id = vocab.index("<unk>")

    encoder_model = build_bart_encoder(
        vocab_size=len(vocab),
        pad_id=pad_id,
        eos_id=eos_id,
        sos_id=sos_id,
    )
    load_checkpoint_to_encoder(encoder_model, ckpt_path)

    # 2) Load CSV & extract ConfSeq column
    df = pd.read_csv(input_csv)
    if confseq_column not in df.columns:
        raise ValueError(
            f"Column '{confseq_column}' not found in CSV '{input_csv}'. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = df[confseq_column].astype(str).fillna("").tolist()
    print(f"Loaded {len(sequences)} ConfSeq sequences from column '{confseq_column}'.")

    # 3) Batch encoding
    embeddings_tensor = get_embeddings_batch(
        encoder_model=encoder_model,
        sequences=sequences,
        vocab_dict=vocab_dict,
        pad_id=pad_id,
        unk_id=unk_id,
        batch_size=batch_size,
        device=device,
    )

    # 4) Save results
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, embeddings_tensor.numpy().astype(np.float32))
    print(
        f"Saved embeddings to '{output_npy}' with shape {tuple(embeddings_tensor.shape)}."
    )

    return embeddings_tensor


# ============================================================
# 6. Command-line entry point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Batch-encode ConfSeq sequences from a CSV file using a BART encoder checkpoint.\n"
            "Typical usage: encode 'td_smiles' column produced by the ZINC->ConfSeq pipeline."
        )
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file containing ConfSeq sequences.",
    )
    parser.add_argument(
        "--confseq_column",
        type=str,
        default="td_smiles",
        help="Name of the column that stores ConfSeq strings (default: 'td_smiles').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the encoder checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        required=True,
        help="Path to output .npy file for embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for DataLoader (default: 2048).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device, e.g. 'cuda' or 'cpu' (default: 'cuda').",
    )

    args = parser.parse_args()

    encode_confseq_from_csv(
        input_csv=args.input_csv,
        confseq_column=args.confseq_column,
        ckpt_path=args.checkpoint,
        output_npy=args.output_npy,
        batch_size=args.batch_size,
        device_str=args.device,
    )
