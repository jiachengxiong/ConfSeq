# ConfSeq - 3D molecular representation learning

This directory contains the code and configuration files for the 3D molecular representation learning module of **ConfSeq**.
![Overview](./assets/overview.png)

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `representation_learning`.

---

## 📦 Data Preparation

We use the **DUD-E** and **PCBA** datasets to evaluate the virtual screening capability of the model.  
Additionally, molecular pair similarity data for model training are generated using **RDKit** and **LSalign**.

Raw and processed datasets can be downloaded from [this link](https://1drv.ms/f/c/940c94b59e54c472/Ev9je1Q3Y2FMtL6tyvrDkgUBEMDUNuRlVFuydOPFM5mVNw?e=quBnOT).

After downloading, extract the archive and place its contents inside the `data` folder.  
The expected folder structure of this directory is as follows:

```

.
├── README.md
├── infer_for_DUDE.ipynb
├── infer_for_PCBA.ipynb
├── view_PDB_embdding.ipynb
├── screen_pubchem.ipynb
├── train.py
├── embed_zinc.py
├── get_confseq_embeddings.py
├── data
│   ├── DUDE
│   ├── PCBA
│   ├── PDB
│   ├── Pairwise_molecular_similarity
│   └── pubchem_embeddings
├── checkpoints
└── model_epoch_1.pth
└── assets
└── overview.png

```

---

## 🏋️ Model Training

To train the representation learning model, run:

```

accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 4 train.py

```

Alternatively, you may download a pre-trained model checkpoint from [this link](https://1drv.ms/f/c/940c94b59e54c472/Esl0IQNq44BIneU_K80LCmMBA02BJcstSDygUk8vJfQQjw?e=ezX4tc) and place it in the `checkpoints` directory.

---

## 🧬 ZINC → 3D → ConfSeq Pipeline (Optional)

We provide a standalone script to convert large-scale SMILES libraries (e.g., ZINC) into 3D conformers and corresponding ConfSeq strings:

**Script:** `embed_zinc.py`

This script performs:

1. SMILES parsing and de-duplication by structural key.
2. Single-conformer 3D generation using ETKDGv3 + MMFF/UFF.
3. Export of valid 3D molecules into SDF.
4. Conversion of 3D molecules into ConfSeq.
5. Coarse-grained timing of 3D generation and ConfSeq conversion.

Example usage:

```

python embed_zinc.py 
--smiles_path data/ZINC/all_instock_smiles.smi 
--sdf_out outputs/zinc_instock_3d.sdf 
--csv_out outputs/zinc_instock_confseq.csv 
--timing_out outputs/zinc_instock_timing.json 
--num_workers 128

```

The output CSV will contain a `confseq` column that can be directly used for embedding.

---

## 🔢 Batch ConfSeq Embedding from CSV

For large-scale embedding of ConfSeq strings (e.g., from ZINC or custom libraries), we provide a dedicated batch embedding script:

**Script:** `get_confseq_embeddings.py`

This script:

- Reconstructs the ConfSeq vocabulary.
- Builds a BART-based encoder.
- Loads a pretrained checkpoint.
- Encodes ConfSeq sequences from a specified CSV column.
- Exports embeddings as a `.npy` file.

Typical usage with ZINC ConfSeq output:

```

python get_confseq_embeddings.py 
--input_csv outputs/zinc_instock_confseq.csv 
--confseq_column confseq 
--checkpoint checkpoints/model_epoch_1.pth 
--output_npy outputs/zinc_instock_confseq_embeds.npy 
--batch_size 2048 
--device cuda

```

This produces a `[N, 256]` embedding matrix aligned with the input CSV rows.

---

## 📊 Evaluation

To perform the evaluation, please run the `infer_for_DUDE.ipynb` and `infer_for_PCBA.ipynb` notebooks. For shape-based virtual screening in pubchem dataset, you can use the `screen_pubchem.ipynb` notebook. A molecule with 3D conformation should be provided in order to calculate the query embeddings and top-k results along with the corresponding distance will be computed.  

> [!Note]
> It could be memory expensive to load the pubchem embeddings, which requires at least 25 GB of RAM. For convenience, you may consider using the online service at [this link](https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Screen)

---

## 🎨 Embedding Visualization

To visualize the representations of ligands in the PDB, run `view_PDB_embdding.ipynb` notebooks.
