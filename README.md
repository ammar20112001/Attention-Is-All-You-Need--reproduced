##  Attention Is All You Need | Transformer Model Reproduction and Experimentation Framework | Serverless Deployment Configuration with Minimal UI

Welcome to the **Transformer Model Reproduction** repository! This project implements a flexible framework for experimenting with transformer architectures using **PyTorch** and **PyTorch Lightning**, along with **Weights and Biases (wandb)** for tracking and managing experiments. The repository provides multiple deployment options, including a serverless architecture using **AWS Lambda** and **S3**, as well as a hosted version on **Hugging Face Spaces** for easy interaction with the model, making it accessible to users without requiring local setup.

To access model directly and interact with it, visit the following link:
[Visit the Hugging Face Spaces App](https://huggingface.co/spaces/ammar-20112001/Attention-Is-All-You-Need-reproduced)

**Important Notes:**
1. Since we are using AWS services Lambda and S3, please be mindful and use it only for testing purposes. Avoid making too many requests to the server, as it incurs costs and has limits on the number of requests. Your consideration will help ensure that everyone can use the service without excessive spending on my side.
2. The model was trained on Google Collab and thus is not very well trained to provide highly accurate translations due to limited resources and training time. However, it still does a noteworthy job. The purpose of this project is to offer a streamlined process that anyone can utilize and experiment with using their own resources.

---

## 🛠️ Tech Stack

Python, PyTorch, PyTorch Lightning, Weights and Biases (wandb), Hugging Face Spaces, AWS Lambda, S3, Streamlit, Flask, Git

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="40" alt="pytorch logo"  />
  <img width="12" />
  <img src="https://i.ibb.co/jb9xddb/pytorch-lightning-logo.png" height="40" alt="ligthning logo"  />
  <img width="12" />
  <img src="https://i.ibb.co/QmRmd7M/simple-icons-weightsandbiases.png" height="40" alt="wandb logo"  />
  <img width="12" />
  <img src="https://i.ibb.co/BzLrJQ2/hf-logo.png" height="40" alt="transformers logo"  />
  <img width="12" />
  <img src="https://i.ibb.co/fNqxY7R/aws-lambda-1024x1021.png" height="40" alt="aws lambda logo"  />
  <img width="12" />
  <img src="https://i.ibb.co/z55CzGz/streamlit-logo.png" alt="Streamlit" height="50"/>
  <img width="12" />
  <img src="https://i.ibb.co/hH6HpkF/file-icons-flask.png" height="40" alt="flask logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="40" alt="git logo"  />
</div>


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
- [Model Staging](#model-staging)
- [Web Application](#web-app)
- [AWS Lambda (Serverless Deployment)](#aws-lambda-serverless-deployment)
- [Hugging Face Spaces](#aws-lambda-serverless-deployment)
- [Contributing](#contributing)

---

## Overview

This repository aims to reproduce the transformer model architecture from scratch using **PyTorch**, making the code highly modular and customizable. It includes training loops, hyperparameter tuning, and experiment tracking features to aid machine learning engineers and researchers in quickly setting up and testing transformer-based models. The project also provides multiple deployment options, including a serverless architecture using **AWS Lambda** and **S3**, as well as a hosted version on **Hugging Face Spaces** for easy and direct interaction with the model (to directly interact with the model, see [Hugging Face Spaces](#aws-lambda-serverless-deployment)).

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
- **Web Applications**: Includes Flask-based applications and a Streamlit app, offering various deployment configurations for interacting with the transformer model.
- **Serverless Deployment**: Leverages **AWS Lambda** and **S3** for serverless model deployment, ensuring scalability and ease of access.
- **Hugging Face Spaces**: A hosted version of the Streamlit application allows users to interact with the model directly via a web interface without needing to set up any local environment.

---

## Setup

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ammar20112001/Attention-Is-All-You-Need--reproduced.git
cd Attention-Is-All-You-Need--reproduced

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

## Model Staging

The `stage_model.py` script provides functionality to stage a transformer model for production. It pulls a trained model artifact from **Weights and Biases (wandb)**, converts it to **TorchScript**, and prepares it for deployment. Here’s a breakdown of its key functionalities:

1. **Downloading Model Artifact**: The script downloads a specific model version from a wandb artifact repository into a local directory, `Models/`, which is created if it doesn’t already exist.

2. **TorchScript Conversion**: After downloading, the PyTorch model is loaded and converted to **TorchScript**, making it compatible with production environments.

3. **Saving for Production**: The TorchScript version of the model is saved locally as `model.pt` and is also uploaded back to wandb for other access.

To use the staging script, run the following command with appropriate arguments:

```bash
python stage_model.py --fetch --entity <wandb_entity> --from_project <wandb_project_name> --artifact <artifact_name> --version <artifact_version>
```

Replace `<wandb_entity>`, `<wandb_project_name>`, `<artifact_name>`, and `<artifact_version>` with your specific values.

---

## Web Application

This repository includes three web applications, each with different deployment configurations:

- **Flask_AWSlambda**: This app is a Flask-based interface for interacting with the transformer model, deployed as a service on **AWS Lambda** and **S3**. To run it, use `flaskapi.py`.
```bash
  flask --app flaskapi run
  ```
  
- **Flask_ModelInService**: This Flask app loads the transformer model directly in the server, without relying on external services for inference. Launch it with `flaskapi.py`.
```bash
  flask --app flaskapi run
  ```
  
- **Streamlit_AWSlambda**: This app uses Streamlit and connects to a model-as-a-service deployment on **AWS Lambda**. Run it with `app.py`.
```bash
  streamlit run app.py
  ```

---

## AWS Lambda (Serverless Deployment)

The model and its dependencies are packaged and deployed on AWS Lambda and S3 for serverless deployment. This setup uses a `.sh` file to manage dependencies and ensure a streamlined production environment. Here’s a breakdown of the deployment steps:

1. **Dependency Packaging**: The script copies all dependencies from the `.venv` directory, manually removes any unnecessary files, and minimizes the package size by excluding certain directories and files.

2. **Zipping Components**: The `torch` library and other dependencies are zipped separately (`torch.zip` and `dependencies.zip`), reducing the package size. 

3. **S3 and Lambda Deployment**: Finally, the script uploads the `dependencies.zip` and `torch.zip` files to an S3 bucket, then updates the Lambda function code directly from S3.

To run the script, use:

```bash
./deploy.sh
```

---

## Hugging Face Spaces

This repository includes a deployed version of the Streamlit application on **Hugging Face Spaces**. This allows users to interact with the transformer model effortlessly without needing to set up any environment or run code locally.

- **Interactive Streamlit Application**: The application is hosted on Hugging Face Spaces, providing a user-friendly interface to interact with the transformer model. Users can access it directly via the following link:

  [Visit the Hugging Face Spaces App](https://huggingface.co/spaces/ammar-20112001/Attention-Is-All-You-Need-reproduced)

This setup utilizes serverless deployment on **AWS Lambda** to handle inference requests, ensuring a smooth experience for anyone wanting to play around with the model.

**Disclaimer**: If the inference time is longer than expected, it may be due to the cold start of the Lambda server. This delay typically only occurs the first time the model is invoked after a period of inactivity.

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
