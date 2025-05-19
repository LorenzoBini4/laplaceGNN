# Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Examples](#examples)
    - [Example 1: Node Classification](#example-1-node-classification)
    - [Example 2: Graph Classification](#example-2-graph-classification)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
## Introduction

We present LaplaceGNN, a novel self-supervised graph learning framework that bypasses the need for negative sampling by leveraging spectral bootstrapping techniques. Our method integrates Laplacian-based signals into the learning process, allowing the model to effectively capture rich structural representations without relying on contrastive objectives or handcrafted augmentations. By focusing on positive alignment, LaplaceGNN achieves linear scaling while offering a simpler, more efficient, self-supervised alternative for graph neural networks, applicable across diverse domains. Our contributions are twofold: we precompute spectral augmentations through max-min centrality-guided optimization, enabling rich structural supervision without relying on handcrafted augmentations, then we integrate an adversarial bootstrapped training scheme that further strengthens feature learning and robustness. Our extensive experiments on different benchmark datasets show that LaplaceGNN achieves superior performance compared to state-of-the-art self-supervised graph methods, offering a promising direction for efficiently learning expressive graph representations. This repository contains the implementation of LaplaceGNN, along with scripts for training and evaluation.

## Features 

- **Spectral Bootstrapping**: Enhances the learning process by incorporating spectral information from the graph.
- **Adversarial Training**: Improves model robustness by training with adversarial examples.
- **Scalability**: Designed to lineary handle large-scale graphs efficiently.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the LaplaceGNN model, use the following command:

```bash
python train.py --config config.yaml
```

### Evaluation

To evaluate the trained model, use:

```bash
python evaluate.py --config config.yaml
```

## Configuration

The configuration file `config.yaml` contains various parameters for training and evaluation. Modify this file to suit your needs.

## Examples

### Example 1: Node Classification

```bash
python train.py --config configs/node_classification.yaml
```

### Example 2: Graph Classification

```bash
python train.py --config configs/graph_classification.yaml
```

## Contributing

We welcome contributions to improve LaplaceGNN. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please please don't hesitate to open an issue on GitHub or directly reach out by email.
