# Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations

This repository contains the official PyTorch implementation for the paper: [**"Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations"**](https://arxiv.org/abs/2506.20362) currenly under double-blind review as a conference paper.

## Abstract

We present LaplaceGNN, a novel self-supervised graph learning framework that bypasses the need for negative sampling by leveraging spectral bootstrapping techniques. Our method integrates Laplacian-based signals into the learning process, allowing the model to effectively capture rich structural representations without relying on contrastive objectives or handcrafted augmentations. By focusing on positive alignment, LaplaceGNN achieves linear scaling while offering a simpler, more efficient, self-supervised alternative for graph neural networks, applicable across diverse domains. Our contributions are twofold: we precompute spectral augmentations through max-min centrality-guided optimization, enabling rich structural supervision without relying on handcrafted augmentations, then we integrate an adversarial bootstrapped training scheme that further strengthens feature learning and robustness. Our extensive experiments on different benchmark datasets show that LaplaceGNN achieves superior performance compared to state-of-the-art self-supervised graph methods.

---

## Key Features

-   **Spectral Bootstrapping:** A self-supervised approach that eliminates the need for negative sampling, reducing computational complexity from O(N²) to O(N).
-   **Laplacian-based Augmentations:** A principled method to generate graph views via Laplacian optimization and centrality-guided augmentations, removing the need for hand-crafted augmentation strategies.
-   **Adversarial Teacher-Student Framework:** A robust training scheme that improves representation quality and enhances resistance to adversarial attacks.
-   **State-of-the-Art Performance:** Achieves superior results on node classification, graph classification, and transfer learning benchmarks.

---

## Repository Structure

The repository is organized to support experiments for node-level and graph-level tasks.

```
.
├── config_node/                # Configuration files for node classification tasks
├── laplaceGNN/                 # Core package: models, data loaders, protocol, objectives, run logging
├── laplacian_augmentations/    # Spectral view generation (sparse and original dense), samplers, structural adversary
├── ssl_adv_node/               # Node-level SSL: run_adv_node.py (our pipeline), run_baselines.py (CCA-SSG, GraphMAE, PolyGCL)
├── ssl_adv_graph/              # Graph-level SSL (ogb/ and tudataset/)
├── scripts/                    # Tuning, retraining, ablation, dose-response, attacks, scaling, paper tables and figures
├── experiments/                # Launchers of the controlled study, with a README
├── results/                    # Aggregated results (CSV) behind every table and figure of the paper
├── tests/                      # Unit tests (eigenvalue gradients, protocol, view samplers); run as plain scripts
├── paper/                      # LaTeX sources of the controlled study (build with `make`)
├── METHOD_CHANGES.md           # Every change with respect to the original method, with all result tables
├── main_node.sh                # Example script to run node classification
├── main_ogb.sh                 # Example script to run OGB graph classification
├── main_tudata.sh              # Example script to run TU-Dataset graph classification
├── requirements.txt            # Pinned versions used for the results
└── README.md
```

-   **`config_node/`**: Contains `.cfg` files with hyperparameters for various node classification datasets.
-   **`data/`**: Stores datasets. Standard datasets from PyTorch Geometric or OGB will be downloaded and processed here automatically.
-   **`laplaceGNN/`**: Contains the core model implementation (`models.py`, `laplaceGNN.py`) and data loaders (`data.py`).
-   **`laplacian_augmentations/`**: Contains the core model augmentations strategies (`laplacian_node.py`).
-   **`ssl_adv_node/` & `ssl_adv_graph/`**: These directories contain the main training and evaluation logic for node-level and graph-level tasks, respectively.
-   **`main_*.sh`**: Shell scripts demonstrating how to launch experiments for different task types.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LorenzoBini4/laplaceGNN
    cd laplaceGNN
    ```

2.  **Create a Python environment:**
    We recommend using Conda or a Python virtual environment.
    ```bash
    conda create -n laplacegnn python=3.8
    conda activate laplacegnn
    ```

3.  **Install dependencies:**
    Install PyTorch according to your CUDA version. For example:
    ```bash
    # For CUDA 11.8
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    Then install the remaining packages with the pinned versions used for all results (PyG extensions and DGL need the wheel indices listed at the top of the file):
    ```bash
    pip install -r requirements.txt
    ```

---

## Running Experiments

The repository is structured to run three main types of experiments, demonstrated by the provided shell scripts.

### 1. Node Classification

Node classification experiments (e.g., on Cora, CiteSeer, Coauthor-CS, Amazon Computers) are configured via `.cfg` files.

-   **Configuration**: Modify hyperparameters for a specific dataset in its corresponding file inside `config_node/`.
-   **Execution**: Run the `main_node.sh` script. You can uncomment the line for the dataset you wish to run.

    ```bash
    # Example for Coauthor-CS
    bash main_node.sh
    ```
    This command executes `python -u -m ssl_adv_node.run_adv_node --flagfile=config_node/coauthor-cs.cfg`, logging the output to `run-coauthor-cs.out` and `run-coauthor-cs.err`.

### 2. Graph Classification (OGB Datasets)

Experiments on OGB graph-level datasets (e.g., `ogbg-molhiv`, `ogbg-molbbbp`) are run using the `main_ogb.sh` script.

-   **Configuration**: Modify the `dataset` variable and other command-line arguments directly within `main_ogb.sh`.
-   **Execution**:
    ```bash
    bash main_ogb.sh
    ```
    This runs the `ssl_adv_graph.ogb.run_adv_graph` module with the specified arguments.

### 3. Graph Classification (TU Datasets)

Experiments on TU Datasets (e.g., `PROTEINS`, `MUTAG`) are run using the `main_tudata.sh` script.

-   **Configuration**: Set the `dataset` variable in `main_tudata.sh`.
-   **Execution**:
    ```bash
    bash main_tudata.sh
    ```
    This script will execute the `ssl_adv_graph.tudataset.run_adv_graph` module.

---

## Controlled study (reworked pipeline)

The repository also contains a reworked pipeline used for a controlled study of what drives graph self-supervised learning: encoders, spectral structural views and adversarial training, compared with BGRL, CCA-SSG, GraphMAE and PolyGCL under one protocol (fixed splits, linear probe, validation-only selection, equal random-search budget, fresh seeds for reported numbers). `METHOD_CHANGES.md` documents every change with respect to the original method and contains all result tables; `paper/` holds the manuscript.

```bash
# Equal-budget random search for one method on one dataset (test accuracy is logged but never used for selection)
python scripts/tune_node.py --dataset cora --method laplacegnn_full --trials 60 --workers 4
# Retrain the selected configuration with fresh seeds
python scripts/run_best.py --dataset cora --methods bgrl,ccassg,laplacegnn_full --seeds 3
# Paired encoder comparison, ablation, dose-response, attacks, scaling
python scripts/paired_encoder.py
python scripts/ablate.py --dataset cora
python scripts/dose_response.py --dataset cora --method laplacegnn_full --seeds 3
python scripts/make_attacks.py && python scripts/scaling_sparse.py
# Tables and figures of the paper, then the PDF
python scripts/paper_tables.py && python scripts/paper_figures.py && make -C paper
```

The launchers that produced the reported runs are in `experiments/`, and the unit tests run with `python tests/test_spectral.py` and `python tests/test_phase0.py`. Datasets are downloaded to `data/` on first use. Runs write their full configuration, git state and per-split scores to `runs/` and `logs/`, both ignored by git; `results/` holds the aggregated CSV files.

**PolyGCL baseline.** The PolyGCL baseline runs the authors' code from [ChenJY-Count/PolyGCL](https://github.com/ChenJY-Count/PolyGCL), which has no license and is therefore not redistributed here. To reproduce it, create `third_party/__init__.py` (empty) and `third_party/polygcl/` (both are git-ignored), copy `model.py` and `ChebnetII_pro.py` from the `HeterophilousGraph/` directory of that repository, copy the `cheby` function of its `utils.py` to `third_party/polygcl/cheby.py`, write `from .model import Model` in `third_party/polygcl/__init__.py`, and change the imports in the copied files to relative ones. Our copy also changes the discriminator in `model.py` to the bilinear form $(hW)\cdot c + b$.

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{bini2025laplacegnn,
  title={Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations},
  author={Lorenzo Bini and Stéphane Marchand-Maillet},
  journal={arXiv preprint},
  year={2025}
}
