# Latent Graph Neural SDEs (LGNSDE)

![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)  
![Python](https://img.shields.io/badge/python-3.8%2B-green)  
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🚀 Overview

This repository contains the code for **“Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations”**, published at ICLR 2025. We introduce **Latent Graph Neural SDEs (LGNSDEs)**, a novel framework that

- Combines **Graph Neural Networks** with **Stochastic Differential Equations** to learn uncertainty-aware node embeddings.
- Quantifies both **aleatoric** (data) and **epistemic** (model) uncertainty via a Bayesian prior–posterior SDE in latent space.
- Provides **theoretical guarantees** on well-posedness, variance bounds, and robustness to input perturbations.
- Demonstrates state-of-the-art performance on out-of-distribution detection, noise robustness, and active learning across five benchmark graph datasets.

## 📂 Repository Structure

```
├── data/                   # (Optional) scripts or instructions to download datasets
├── src/                    # Core implementation
│   ├── models/             # LGNSDE, GNODE, GCN, etc.
│   ├── utils/              # Data loaders, training routines
│   └── train.py            # Training & evaluation entry point
├── experiments/            # Jupyter notebooks or scripts for tables & figures
├── results/                # Precomputed logs, model checkpoints
├── requirements.txt        # Python dependencies
└── README.md               # (You are here)
```

## 🔧 Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/<your-username>/LGNSDE.git
   cd LGNSDE
   ```

2. Create a Python environment and install dependencies  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> **requirements.txt** should include at least:
> ```
> torch>=1.8
> torchdiffeq
> numpy
> scipy
> scikit-learn
> networkx
> pandas
> tqdm
> ```

## ▶️ Usage

Train and evaluate LGNSDE on, e.g., the Cora dataset:

```bash
python src/train.py \
  --dataset cora \
  --model LGNSDE \
  --epochs 200 \
  --lr 0.005 \
  --weight_decay 1e-4 \
  --hidden_dim 64 \
  --drift_prior -0.5 \
  --diffusion_sigma 1.0
```

Key flags:

- `--dataset` : one of {cora, citeseer, computers, photo, pubmed}  
- `--model`   : LGNSDE | GNODE | GCN | BGCN | Ensemble  
- `--drift_prior`   : prior drift coefficient (e.g., -0.5)  
- `--diffusion_sigma`: diffusion magnitude (e.g., 1.0)  

Results (AUROC, AURC, accuracy) will be printed and saved under `results/`.

## 📊 Experiments

- **Out-of-Distribution Detection**: leave-one-class-out protocol; evaluate entropy separation between in-dist and OOD.  
- **Noise Robustness**: add Gaussian noise (σ = std(X_test)×0.5) at test time.  
- **Active Learning**: iterative label acquisition (random vs. max-entropy).  

See the `experiments/` folder for notebooks that reproduce Tables 1–5 and Figures 2–3 in the paper.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{bergna2025LGNSDE,
  title     = {Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations},
  author    = {Bergna, Richard and Calvo-Ordoñéz, Sergio and Opolka, Felix L. and Liò, Pietro and Hernandez-Lobato, Jose Miguel},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2025},
}
```

## ⚖️ License

This project is licensed under the **MIT License**. See LICENSE.md for details.

---

**Contact:** Richard Bergna ― rsb63@cam.ac.uk  
**Repo:** https://github.com/<your-username>/LGNSDE

