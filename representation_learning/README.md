# ConfSeq - Unconditional Molecular Generation

This directory contains the code and configuration files for the 3D molecular representation learning module of **ConfSeq**.

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `representation_learning`.

---
## 📦 Data Preparation

We use the **DUD-E** and **PCBA** datasets to evaluate the virtual screening capability of the model.  
Additionally, molecular pair similarity data for model training are generated using **RDKit** and **LSalign**.

Raw and processed datasets can be downloaded from [this link](mylink).

After downloading, extract the archive and place its contents inside the `data` folder.  
The expected folder structure is as follows:

```
data/
├── DUDE/
├── PCBA/
├── PDB/
└── Pairwise_molecular_similarity/
```

---
## 🏋️ Model Training

To train the representation learning model, run:

```
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 4 train.py
```

Alternatively, you may download a pre-trained model checkpoint from [this link](mylink) and place it in the `checkpoints` directory.

---
