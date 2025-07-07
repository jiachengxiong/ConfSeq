# ConfSeq - Shape-conditioned Generation

This directory contains the code and configuration files for the shape-conditioned molecular generation module of **ConfSeq**.

![Overview](./assets/overview.png)

> \[!NOTE]
> Execute all commands below inside the `confseq` conda environment. Make sure your working directory is set to `shape_conditioned_generation`.

## Data Preparation

We employ the **MOSES** dataset for unconditional molecular generation, following the same data split strategy as in the [DiffSMol paper](https://www.nature.com/articles/s42256-025-01030-w).

First, download the dataset and put it in the `data/MOSES/` directory. You can download the dataset from [this link](mylink):

Next, process the dataset to obtain the molecule surface pointcloud and corresponding ConfSeq representations by executing:

```bash
bash scripts/preprocess.sh
```

> \[!CAUTION]
> While sampling pointclouds following the protocols of the DiffSMol paper, we run into some issues with the oddt library. You'd better modify the source code of oddt to avoid these issues. Specifically, you should open the defination of function `oddt.surface.generate_surface_marching_cubes` and change the following lines:
> ```python
> try:
>     from skimage.morphology import ball, binary_closing
>     from skimage import __version__ as skimage_version
>     if LooseVersion(skimage_version) >= LooseVersion('0.13'):
>         from skimage.measure import marching_cubes_lewiner as marching_cubes
>     else:
>         from skimage.measure import marching_cubes
> except ImportError as e:
>     warnings.warn('scikit-image could not be imported and is required for'
>                   'generating molecular surfaces.')
>     skimage = None
> ```
> into this:
> ```python
> try:
>     from skimage.morphology import ball, binary_closing
>     from skimage import __version__ as skimage_version
>     # if LooseVersion(skimage_version) >= LooseVersion('0.13'):
>     #     from skimage.measure import marching_cubes_lewiner as marching_cubes
>     # else:
>     from skimage.measure import marching_cubes
> except ImportError as e:
>     warnings.warn('scikit-image could not be imported and is required for'
>                   'generating molecular surfaces.')
>     skimage = None
> ```

Upon successful execution, the processed dataset will be available at `data/`. This dataset will serve as the input for training and evaluation. Alternatively, you can download the processed datasets directly from [this link](mylink).

## Model Training

To train the unconditional generation model, run:

```bash
bash scripts/train_surfbart.sh
```

A pre-trained model checkpoint is also available for download [here](mylink).

## Molecule Generation

To generate molecules using the trained model, execute:

```bash
bash scripts/sample_confseq.sh
```

## Evaluation

We utilize the ShaEP software to calculate the shape similarity between the generated molecules and the reference molecules as suggested by [SQUID](https://github.com/keiradams/SQUID). You can download the ShaEP software from [this link](https://users.abo.fi/mivainio/shaep/index.php) or use the software provided in the `software` directory.

To evaluate the quality of generated molecules, please run the following command:

```bash
bash scripts/evaluate_confseq.sh
```