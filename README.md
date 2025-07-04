# ConfSeq

This is the official repository for the paper "Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language"

![overview](./assets/overview.png)

## Environment

To set up the environment for running ConfSeq-series models, you need to install the dependencies as follows. A conda environment is recommended.
```bash
conda env create -f environment.yaml
conda activate confseq
```

## Running the Models
The ConfSeq-series models are organized into subdirectories, each corresponding to a specific task or model type. We provide the algorithm of ConfSeq in `ConfSeq_3_2.py`.

Please refer to README.md in each subdirectory for more details on how to run the models.

## Citation
If you find this code useful, please cite our paper:
```bibtex
@article {Xiong2025.05.07.652440,
	author = {Xiong, Jiacheng and Shi, Yuqi and Zhang, Wei and Zhang, Runze and Chen, Zhiyi and Zeng, Chuanlong and Jiang, Xun and Cao, Duanhua and Xiong, Zhaoping and Zheng, Mingyue},
	title = {Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language},
	elocation-id = {2025.05.07.652440},
	year = {2025},
	doi = {10.1101/2025.05.07.652440},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/05/12/2025.05.07.652440},
	eprint = {https://www.biorxiv.org/content/early/2025/05/12/2025.05.07.652440.full.pdf},
	journal = {bioRxiv}
}
```