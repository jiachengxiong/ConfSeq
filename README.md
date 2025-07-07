# ConfSeq

This is the official repository for the paper:
**“Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language.”**

![Overview](./assets/overview.png)

We also provide the ready-to-use ConfSeq-series models online at [Sciminer](https://sciminer.protonunfold.com/), including:
- **Molecular conformation generation**: [ConfSeq-Conf-Gen](https://sciminer.protonunfold.com/utility?tool=Confseq%20Conf%20Gen).
- **Shape-conditioned Generation**: [ConfSeq-Shape-Gen](https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Gen).
- **ConfSeq based Shape Screening**: [ConfSeq-Shape-Screen](https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Screen).
  
---

## 📦 Environment Setup

It is recommended to use a Conda environment to install the required dependencies. The following commands will create a new Conda environment named `test` with Python 3.11 and CUDA 12.6, and install the necessary packages for running ConfSeq models.

```bash
conda create -n test python=3.11
conda activate test

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install rdkit==2025.3.2 transformers==4.50 accelerate
pip install jupyter epam.indigo smilesPE posebusters timeout_decorator fcd_torch easydict py3dmol swanlab lmdb scikit-image matplotlib seaborn
conda install openbabel -c conda-forge
conda install ninja
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install oddt
export PATH=/usr/local/cuda-12.9/bin:$PATH  # only if you encountered CUDA version mismatch
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH  # only if you encountered CUDA version mismatch
pip install ./shape_conditioned_generation/src/model/pointops
```
We also provide a conda-packed environment in [this link](mylink) for easy setup. You can download the file and run the following command to create the environment:


## Demo

We provide a demo notebook in `demo` directory on how to use convert a 3D molecule into a ConfSeq sequence and vise versa. The detailed algorithm of ConfSeq can be found at `demo/ConfSeq.py`.

---

## 🚀 Running the Models

The ConfSeq-series models are organized into subdirectories, each corresponding to a specific task mentioned in the paper.
The core algorithm 

Please refer to the `README.md` file within each subdirectory for detailed instructions on running the respective models.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jiachengxiong/ConfSeq\&type=Date)](https://www.star-history.com/#jiachengxiong/ConfSeq&Date)

---

## 📖 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{Xiong2025.05.07.652440,
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

