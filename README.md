# Transformer Model Reproduction and Experimentation Framework

Welcome to the **Transformer Model Reproduction** repository! This project implements a flexible framework for experimenting with transformer architectures using **PyTorch** and **PyTorch Lightning**, along with **Weights and Biases (wandb)** for tracking and managing experiments. 

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
  - [Running Single Experiments](#running-single-experiments)
  - [Running Sweeps](#running-sweeps)
  - [Alternative: Run Experiments in Google Colab](#run-experiment-in-google-colab)
- [Configuration](#configuration)
- [Results & Tracking](#results--tracking)
- [Contributing](#contributing)

---

## Overview

This repository aims to reproduce the transformer model architecture from scratch using **PyTorch**, making the code highly modular and customizable. It includes training loops, hyperparameter tuning, and experiment tracking features to aid machine learning engineers and researchers in quickly setting up and testing transformer-based models.

---

## Architecture

The architecture follows a standard transformer model implementation, incorporating the following core components:
- **Encoder-Decoder architecture**
- **Multi-head self-attention mechanism**
- **Position-wise feed-forward networks**
- **Positional encodings**

These components are implemented using **plain PyTorch** for a hands-on understanding of transformer internals.

---

## Features

- **End-to-End Transformer Model:** Built using plain PyTorch for flexibility and customization.
- **PyTorch Lightning for Training:** Simplified training loop and model management.
- **Transformers:** For tokenization.
- **Weights and Biases Integration:** Seamless tracking and visualization of experiments, including hyperparameter sweeps.
- **Flexible Experiment Configuration:** All hyperparameters and model settings can be adjusted via a single YAML configuration file.
- **Sweep Support:** Automate running multiple experiments via wandb sweeps with provided command templates.
- **Interactive Notebooks:** Pre-built notebooks with commands for training, validation, and tracking, reducing boilerplate code.

---

## Setup

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/transformer-reproduction.git
cd transformer-reproduction

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Weights and Biases (wandb)
- Other dependencies are listed in `requirements.txt`.

---

## Usage

This project provides two main ways to experiment with the transformer model: running **single experiments** and running **sweeps** for hyperparameter tuning.

### Running Single Experiments

You can easily start a training experiment by executing the command below. All configurations are defined in the `config.py` file.
```bash
python3 sweep.py
```
The experiment will be logged to wandb if you set up the API key correctly (explained [here](https://docs.wandb.ai/quickstart)).

### Running Sweeps

You can easily start a training experiment by executing the command below. All configurations are defined in the `sweep_config.py` file.
```bash
python3 train.py
```
Pre-built sweep configurations can be found in the `sweep_config.py` file.

### Alternative: Run Experiments in Google Colab

For those who prefer to run experiments without setting up the environment locally, you can use the `run_experiment.ipynb` notebook. This notebook provides an easy way to run your experiments directly in **Google Colab**, with the following features:

- **Automatic Environment Setup:** The notebook automatically installs all necessary dependencies, ensuring you have everything you need to run the experiments.
- **GPU Access:** Easily connect to a GPU instance for faster training and experimentation.
- **Interactive Interface:** Enjoy an interactive Jupyter environment where you can modify configurations and visualize results in real time.

To get started, simply open the notebook in Google Colab:

<a href="https://colab.research.google.com/github/ammar20112001/Attention-Is-All-You-Need--reproduced/blob/main/run_experiments.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" width="200"/>
</a>

---

## Configuration

All the hyperparameters and model settings are managed via a **separate config files for single experiments and sweeps** (`config.py` and `sweep_config.py`). You can easily adjust:

- Model architecture parameters (e.g., number of layers, attention heads)
- Optimizer settings (e.g., learning rate, weight decay)
- Training configurations (e.g., batch size, number of epochs)
- Wandb settings (e.g., project name, entity)
- And much more...

Here's a sample from the `config.py`:

```yaml
# Dataset configurations
"dataset": "kaitchup/opus-English-to-French",
"lang_src": "en",
"lang_tgt": "it",
"datasource": 'opus_books',
# Tokenizer
'tokenizer': 'bert-base-uncased',
# Data loader configurations
'batch_size': 64,
# Data split configurations
'train_frac': 0.9,
'test_val_frac': 0.5,
'train_rows': 10_000, # Set value to `False` to laod full dataset
'test_val_rows': 1_000,

# Transformer model configurations
'd_model': 256, # In paper: 512
'heads': 4, # In paper: 8
'n_stack': 3, # In paper: 6
'max_seq_len': 256, # In paper: 512
'src_vocab_size': 40_000,
'tgt_vocab_size': 40_000,
'dropout': 0.1, # In paper: 0.1
'd_fc': 1024, # In paper: 2048
'enc_max_seq_len': 256, # In paper: 512
'dec_max_seq_len': 256, # In paper: 512

# Training configurations
'epochs': 1,
# Learning rate
'lr': 0.001,
'optimizer': 'Adam', # Hard coded

# W&B configs
'log_text_len': 15
```

---

## Results & Tracking

All experiment logs, metrics, and visualizations are automatically tracked with **Weights and Biases**. Once your experiment runs, you can visit your wandb project page to explore:

- **Training/Validation curves**
- **Attention maps**
- **Hyperparameter sweeps performance**
- **Model checkpoints**
- And much more...

Make sure to set up your wandb account and API key by running:

```bash
wandb login
```
If you're not logged in, you'll be prompted to do so when you run experiment.

For more details on wandb usage, refer to the [official documentation](https://docs.wandb.ai/).

---

## Contributing

Contributions are welcome! If you'd like to improve this project or add new features, feel free to submit a pull request or open an issue.

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
4. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
6. Push to the branch:
   ```bash
   git push origin feature/my-new-feature
   ```
8. Open a pull request.

---
