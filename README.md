# ConfSeq: Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language

**ConfSeq** introduces a unified language for representing 3D molecular conformations, enabling seamless integration of geometric molecular data with modern sequence-based deep learning architectures.

<p align="center">
  <img src="./assets/Figure_1.png" width="720">
</p>

---

# 🔗 **Online Access to Pretrained ConfSeq-series Models**

All pretrained ConfSeq models can be accessed directly through **Sciminer**:

| Task                             | Model                  | Link                                                                                                                                               |
| -------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3D Conformation Generation**   | *ConfSeq-Conf-Gen*     | [https://sciminer.protonunfold.com/utility?tool=Confseq%20Conf%20Gen](https://sciminer.protonunfold.com/utility?tool=Confseq%20Conf%20Gen)         |
| **Shape-conditioned Generation** | *ConfSeq-Shape-Gen*    | [https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Gen](https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Gen)       |
| **Shape-based Screening**        | *ConfSeq-Shape-Screen* | [https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Screen](https://sciminer.protonunfold.com/utility?tool=ConfSeq%20Shape%20Screen) |

---

# ⚙️ **Environment Setup**

We recommend using a clean Conda environment. The following creates a CUDA-enabled environment with all required dependencies:

```bash
conda create -n confseq python=3.10 -y
conda activate confseq

# PyTorch + CUDA 12.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Core libraries
pip install rdkit==2024.9.3 transformers==4.50 accelerate==1.8.1
pip install jupyter epam.indigo==1.32.0 SmilesPE==0.0.3 posebusters==0.4.4 timeout_decorator==0.5.0
pip install fcd_torch easydict py3dmol swanlab lmdb scikit-image matplotlib seaborn

# Geometry-related dependencies
conda install openbabel -c conda-forge -y
conda install ninja -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install oddt==0.7

# Install pointops for SurfBART-v2
pip install ./shape_conditioned_generation/src/model/pointops
```

> [!NOTE]
> If CUDA mismatch occurs, specify your CUDA bin/lib:
>
> ```bash
> export PATH=/usr/local/cuda-12.9/bin:$PATH
> export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
> ```

For a pre-built environment with all dependencies, download the Conda environment package **[here](https://1drv.ms/u/c/940c94b59e54c472/EfXEzVb_NeNGpi6csZ0PSnAB5kSEkiqQvTMyhff1BwBmtQ?e=ZnQ8CO)**.

---

# 🧪 **Demo Notebook**

A concise tutorial demonstrating ConfSeq encoding and decoding is available in:

```
demo/ConfSeq.ipynb
```

The implementation of ConfSeq is contained in:

```
demo/ConfSeq.py
```

---

# 📂 **Repository Structure**

```
ConfSeq/
│
├── conformation_prediction/      # ConfSeq-Conf-Gen models and scripts
├── unconditional_generation/     # Unconditional generative models
├── shape_conditioned_generation/ # SurfBART-v2 for shape-conditioned generation
├── representation_learning/      # Shape-aware molecular representation learning
└── demo/                         # ConfSeq demo utilities and notebook
```

Each subdirectory contains an independent `README.md` with task-specific instructions.

---

# 🚀 **One-click Inference Scripts**

We provide clean, minimal entry points for all ConfSeq-series tasks.

---

## **1. Conformation Prediction**

```bash
cd conformation_prediction
python run_conf_gen.py \
  --smiles "CC(F)(C(F)(F)F)C1=CC=CC=C1" \
  --conf_num 50 \
  --temperature 2.0 \
  --top_p 0.96 \
  --top_k 360 \
  --device cuda \
  --seed 42 \
  --max_length 256 \
  --out outputs/confs.sdf
```

---

## **2. Unconditional Molecule Generation**

```bash
cd unconditional_generation
python run_uncond_gen.py \
  --save_path ./results/confseq_uncond_demo \
  --num_samples 1000 \
  --sample_times 3 \
  --batch_size 64 \
  --scale_times 4 \
  --upscale_temp 1.0 \
  --downscale_temp 0.6 \
  --group_split 99 \
  --device cuda \
  --chunksize 16 \
  --num_workers 4
```

---

## **3. Shape-conditioned Generation**

```bash
cd shape_conditioned_generation
python run_shape_gen.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --output-dir ./results/shape_gen \
  --seed 42 \
  --atom-token-temp 1.2 \
  --angle-token-temp 1.2
```

---

## **4. Shape-based Screening**

Large embedding files must be downloaded manually from
👉 [https://1drv.ms/f/c/940c94b59e54c472/Ev9je1Q3Y2FMtL6tyvrDkgUBEMDUNuRlVFuydOPFM5mVNw?e=quBnOT](https://1drv.ms/f/c/940c94b59e54c472/Ev9je1Q3Y2FMtL6tyvrDkgUBEMDUNuRlVFuydOPFM5mVNw?e=quBnOT)

```bash
cd representation_learning
python run_shape_screen.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --db pubchem \
  --topk 50 \
  --out-prefix results/pubchem_aspirin
```

---

# 📦 **Datasets & Checkpoints**

All datasets, pretrained weights, and FAISS indices are available at:

👉 [https://1drv.ms/f/c/940c94b59e54c472/EgN2JBqq641Mvp8zVDTM0O0Bu3wdg0YwRFZyPrYfASjBmQ?e=4YEtnZ](https://1drv.ms/f/c/940c94b59e54c472/EgN2JBqq641Mvp8zVDTM0O0Bu3wdg0YwRFZyPrYfASjBmQ?e=4YEtnZ)

Please follow the instructions in each task directory regarding placement.

---

# ⭐ **Star History**

<p align="center">
  <a href="https://www.star-history.com/#jiachengxiong/ConfSeq&Date">
    <img src="https://api.star-history.com/svg?repos=jiachengxiong/ConfSeq&type=Date" width="600">
  </a>
</p>

---

# 📝 **Citation**

If this repository contributes to your work, please cite:

```bibtex
@article{Xiong2025.05.07.652440,
  author = {Xiong, Jiacheng and Shi, Yuqi and Zhang, Wei and Zhang, Runze and Chen, Zhiyi and Zeng, Chuanlong and Jiang, Xun and Cao, Duanhua and Xiong, Zhaoping and Zheng, Mingyue},
  title = {Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language},
  elocation-id = {2025.05.07.652440},
  year = {2025},
  doi = {10.1101/2025.05.07.652440},
  journal = {bioRxiv}
}
```

---

# 📬 **Feedback & Issues**

For questions, feature requests, and practical feedback, please open an issue on GitHub.
Pull requests are welcome.

---
