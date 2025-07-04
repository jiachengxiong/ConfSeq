# ConfSeq - Unconditional Generation

This is the directory for the unconditional generation part of ConfSeq. It contains the code and configuration files necessary to train and evaluate the model.

## Environment

To set up the environment for running the unconditional generation model, you need to install the following dependencies:

> [!NOTE] All the following commands should be run in the `confseq` conda environment and the working directory should be `unconditional_generation`.

## Data preparation

We adopted GEOM-Drugs dataset for unconditional molecule generation. We used the same data split as in the [EDM paper](https://arxiv.org/abs/2203.17003).

First download the dataset and extract conformers following EDM:
```bash
cd data/geom_raw
wget https://dataverse.harvard.edu/api/access/datafile/4360331
tar -xzvf 4360331
cd ../..
python src/preprocess/build_geom_dataset.py
```

Then we need to process the dataset to obtain ConfSeq representations of the molecules. This can be done using the following command:
```bash
bash scripts/preprocess.sh
```

The scripts above should get us the processed dataset in `data/geom_confseq`. The processed dataset will be used for training and evaluation. The processed datasets are also available [here](mylink).


## Training the unconditional generation model

> [!CAUTION]
> We chose BART as the model architecture for unconditional molecular generation. However, when we use `BartForCausalLM`(the decoder part of BART), we found that the training and validation loss are really low (about 1e-6). We found this [issue](https://github.com/huggingface/transformers/issues/27517) on GitHub, and change the source code of `transformers`[(code)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py) to fix this issue. Those who want to retrain our model should modify the source code of `transformers` according to the issue mentioned above.

One can easily train the unconditional generation model using the following command:
```bash
bash scripts/train_bartforcausallm.sh
```
We also provide a pre-trained model checkpoint for the unconditional generation task. You can download it from [here](mylink).

## Generation

To generate molecules using the trained model, you can use the following command:
```bash
bash scripts/sample.sh
```

## Evaluation

To evaluate the generated molecules, you can use the following command:
```bash
bash scripts/evaluate_confseq.sh
```
