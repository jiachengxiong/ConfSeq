# ConfSeq - Molecular Conformation Prediction 

This directory contains the code and configuration files for the 3D molecular conformation prediction module of **ConfSeq**.

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `conformation_prediction`.

---
## 📦 Data Preparation

We use the **GEOM-Drugs** dataset to train and evaluate our model.  The original data were downloaded from [this repository](https://github.com/OdinZhang/SDEGen).  Alternatively, you can download the processed data from our cloud storage: [this link](mylink).

After downloading, place the files inside the `raw_data` folder. The expected folder structure is as follows:

```
.
└── raw_data/
    ├── train_data_39k.pkl
    ├── val_data_5k.pkl
    └── test_data_200.pkl
```

To convert the raw data into ConfSeq-formatted text files, run:

```
!python process_raw_data.py
```

The generated files will be saved in the `processed_data` folder.  Alternatively, you may download the processed results from [this link](mylink) and place them in the `processed_data` folder. The expected folder structure is as follows:
```
.
└── processed_data/
    ├── train_data_39k_ConfSeq_aug_0.txt
    ├── train_data_39k_ConfSeq_aug_1.txt
    ├── train_data_39k_ConfSeq_aug_2.txt
    ├── val_data_5k_ConfSeq.txt
    ├── test_data_200_in_smiles_aug_0.json
    ├── test_data_200_in_smiles_aug_1.json
    └── test_data_200_in_smiles_aug_2.json
```

---
## 🏋️ Model Training

To train the conformation prediction model, run:

```
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 4 train_model.py
```

Alternatively, you may download a pre-trained model checkpoint from [this link](mylink) and place it in the `checkpoints` directory.

---
## 🤖 Inference

To perform inference using different sampling temperatures, run:

```
python infer_and_evaluate_temperature_series.py
```

The generated files will be saved in the `prediction_data` folder. Alternatively, you may download the inference results from [this link](mylink) and place them in the `prediction_data` folder. The expected folder structure is as follows:

---

## 📊 Evaluation

To evaluate all generated results in the `prediction_data` folder, run:

```
python eval_all.py
```

The generated files will be saved in the `prediction_data` folder.
