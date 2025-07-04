# ConfSeq - Unconditional Generation

This directory contains the code and configuration files for the unconditional molecular generation module of **ConfSeq**.

> \[!NOTE]
> Execute all commands below inside the `confseq` conda environment. Make sure your working directory is set to `unconditional_generation`.

## Data Preparation

We employ the **GEOM-Drugs** dataset for unconditional molecular generation, following the same data split strategy as in the [EDM paper](https://arxiv.org/abs/2203.17003).

To download and preprocess the dataset:

```bash
cd data/geom_raw
wget https://dataverse.harvard.edu/api/access/datafile/4360331
tar -xzvf 4360331
cd ../..
python src/preprocess/build_geom_dataset.py
```

Next, process the dataset to obtain the ConfSeq representations by executing:

```bash
bash scripts/preprocess.sh
```

Upon successful execution, the processed dataset will be available at `data/geom_confseq`. This dataset will serve as the input for training and evaluation. Alternatively, you can download the processed datasets directly from [this link](mylink).

## Model Training

> \[!CAUTION]
> We utilize the **BART** architecture for unconditional molecular generation. However, when training with `BartForCausalLM` (BART’s decoder), we encountered abnormally low training and validation losses (\~1e-6). This issue is documented in [this GitHub thread](https://github.com/huggingface/transformers/issues/27517).
>
> To address it, we modified the source code of `transformers` accordingly. If you wish to retrain our model, please apply the modifications described in the GitHub issue or refer directly to our modified [source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py).

To train the unconditional generation model, run:

```bash
bash scripts/train_bartforcausallm.sh
```

A pre-trained model checkpoint is also available for download [here](mylink).

## Molecule Generation

To generate molecules using the trained model, execute:

```bash
bash scripts/sample.sh
```

## Evaluation

To evaluate the quality of generated molecules, use the following command:

```bash
bash scripts/evaluate_confseq.sh
```
