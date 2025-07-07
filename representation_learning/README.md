# ConfSeq - Unconditional Molecular Generation

This directory contains the code and configuration files for the 3D molecular representation learning module of **ConfSeq**.

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `representation_learning`.

---
## 📦 Data Preparation

We use the **DUD-E** and **PCBA** datasets to evaluate the virtual screening capability of the model.  
Additionally, molecular pair similarity data for model training are generated using **RDKit** and **LSalign**.

Raw and processed datasets can be downloaded from [your link here].

---
## 🏋️ Model Training

To train the representation learning model, run:

```
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 4 train.py
```

Alternatively, you may download a pre-trained model checkpoint from [this link](mylink) and place it in the `checkpoints` directory.

---
