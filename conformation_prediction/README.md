# ConfSeq - Molecular Conformation Prediction 

This directory contains the code and configuration files for the 3D molecular conformation prediction module of **ConfSeq**.

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `conformation_prediction`.

---
## 📦 Data Preparation

We use the **GEOM-Drugs** dataset to train and evaluate our model.  
The original data were downloaded from [this repository](https://github.com/OdinZhang/SDEGen).  
Alternatively, you can download the processed data from our cloud storage: [this link](mylink).

After downloading, place the files inside the `raw_data` folder.

The expected folder structure is as follows:

---
## 🏋️ Model Training

To train the representation learning model, run:

```
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 4 train.py
```

Alternatively, you may download a pre-trained model checkpoint from [this link](mylink) and place it in the `checkpoints` directory.

---
## 📊 Evaluation

To perform the evaluation, please run the `pcba.ipynb` and `dude.ipynb` notebooks.

---
## 🎨 Embedding Visualization
To visualize the representations of ligands in the PDB, run `V.ipynb` notebooks.


