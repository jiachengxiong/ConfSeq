# ConfSeq - Unconditional Molecular Generation

This directory contains the code and configuration files for the unconditional molecular generation module of **ConfSeq**.

> [!Note]
> Ensure all commands below are executed inside the `confseq` Conda environment, with your working directory set to `unconditional_generation`.

---

## 📦 Data Preparation

We utilize the **GEOM-Drugs** dataset for unconditional molecular generation, adopting the same data split strategy as described in the [EDM paper](https://arxiv.org/abs/2203.17003).

To download and preprocess the dataset, execute the following commands:

```bash
cd data/geom_raw
wget https://dataverse.harvard.edu/api/access/datafile/4360331
tar -xzvf 4360331
cd ../..
python src/preprocess/build_geom_dataset.py
```

Next, generate the ConfSeq representations with:

```bash
bash scripts/preprocess.sh
```

Upon successful execution, the processed dataset will be available in `data/geom_confseq`, which serves as the input for model training and evaluation. Alternatively, you may download the preprocessed dataset directly from [this link](https://1drv.ms/f/c/940c94b59e54c472/EgOVrlM7J2JGqyXerZpYDREBerFk7jMkjWjRsptXBIjb7w?e=DLFaka).

---

## 🏋️ Model Training

> [!CAUTION]
> We employ the **BART** architecture for unconditional molecular generation. However, when using `BartForCausalLM` (BART’s decoder-only model), abnormally low training and validation losses (\~1e-6) were observed. This issue is documented in [this GitHub discussion](https://github.com/huggingface/transformers/issues/27517).
>
> To resolve this, modifications to the `transformers` library source code are required. If you intend to retrain the model, please refer to the GitHub thread and update the loss calculation section in the `BartForCausalLM` implementation, found in the [official source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py).
>
> Specifically, replace the loss calculation block:
>
> ```python
> loss = None
> if labels is not None:
>     labels = labels.to(logits.device)
>     loss_fct = CrossEntropyLoss()
>     loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
> ```
>
> with the following corrected version:
>
> ```python
> loss = None
> if labels is not None:
>     # Shift so that tokens < n predict n
>     shift_logits = logits[..., :-1, :].contiguous()
>     shift_labels = labels[..., 1:].contiguous()
>     # Flatten the tokens
>     loss_fct = CrossEntropyLoss()
>     shift_logits = shift_logits.view(-1, self.config.vocab_size)
>     shift_labels = shift_labels.view(-1)
>     # Enable model parallelism
>     shift_labels = shift_labels.to(shift_logits.device)
>     loss = loss_fct(shift_logits, shift_labels)
> ```
>
> These changes ensure that the model computes the training loss correctly.

To train the unconditional generation model, run:

```bash
bash scripts/train_bartforcausallm.sh
```

Alternatively, you may download a pre-trained model checkpoint from [this link](https://1drv.ms/f/c/940c94b59e54c472/EjZcjDariRlJjvuQ8aa4xREBt0y0_ywdUTMz3c5puc6pYQ?e=cd7IN6) and place it in the `checkpoints` directory.

---

## 🔬 Molecule Generation

To generate molecules using the trained model, execute:

```bash
bash scripts/sample.sh
```

---

## 📈 Evaluation

To evaluate the basic quality of the generated molecules, use:

```bash
bash scripts/evaluate_confseq.sh
```

