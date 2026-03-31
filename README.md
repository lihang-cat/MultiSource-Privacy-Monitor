# MultiSource-Privacy-Monitor (MS-SIDM)

This repository contains the official implementation of **Optimized MS-SIDM: A Context-Aware Sensitive Information Detection Framework via LLM Reasoning and Non-Linear Risk Quantification**. The framework aims to accurately detect privacy leakage risks in multi-source environments and quantify threat levels through Large Language Models (LLMs) and advanced threat scoring mechanisms.

## 📑 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Generation](#-data-generation)
- [Data Download](#-data-download)
- [Training & Evaluation](#-training--evaluation)
  - [Base Model (BERT)](#base-model-bert)
  - [LLM Fine-tuning (LoRA)](#llm-fine-tuning-lora)
- [Threat Scoring](#%EF%B8%8F-threat-scoring)
- [Visualization](#-visualization)

---

## 📂 Project Structure

```text
MultiSource-Privacy-Monitor/
├── data/                           # Data processing and generation module
│   ├── generate_diverse_data.py    # Generate diverse, multi-scenario sensitive data
│   ├── generate_Adversarial_data.py# Generate adversarial stress-test data 
│   └── Bert_dataset.py             # Dataloader for the BERT model
├── models/                         # Model definitions and fine-tuning scripts
│   ├── Bert.py                     # BERT sequence classifier architecture
│   ├── Qwen/                       # Qwen2.5 (3B/7B) LoRA fine-tuning and export scripts
│   ├── Llama 2-7B/                 # Llama 2 fine-tuning scripts
│   ├── ChatGLM3-6B/                # ChatGLM3 fine-tuning scripts
│   └── Baichuan2-7B/               # Baichuan2 fine-tuning and bug-fix scripts
├── multi-source-scenario-data/     # Multi-source scenario experimental data generation
│   ├── synthetic_multi.py          # Simulate multi-source data leakage scenarios (2-4 sources)
│   └── synthetic_single.py         # Simulate severe single-point leakage scenarios
├── total_score/                    # Core threat scoring calculation module
│   ├── inference.py                # Batch inference using LLMs for label extraction
│   └── final_TS.py                 # Calculate Total Score (TS) incorporating density and environmental factors
├── utils/                          # Evaluation and statistical tools
│   ├── plot_cm.py                  # Plot confusion matrix heatmaps for model comparison
│   └── run_Mcnemar.py              # Execute McNemar's statistical significance test
├── visualization/                  # Publication-grade (SCI/IEEE Access) chart generation scripts
│   ├── beta_tau_lambda.py          # Hyperparameter sensitivity analysis (Fig 6)
│   ├── hyperparameter_@.py         # Compensation coefficient analysis (Fig 5)
│   ├── linear vs ours.py           # Comparison between traditional linear models and MS-SIDM
│   ├── multi_source.py             # Incremental risk bar charts for multi-source scenarios (Fig 4)
│   └── wo_compensation.py          # Ablation study analysis
├── Bert_train.py                   # BERT baseline training entry
├── Bert_test.py                    # BERT baseline testing and metric calculation
└── requirements.txt                # Project dependencies
```

---

## 🛠 Installation

We recommend using Python 3.8+ and PyTorch. On AutoDL or a local server, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: LLM fine-tuning heavily relies on `ms-swift` and `modelscope`. Please ensure these libraries are properly installed.

---

## 📊 Data Generation

This project features a high-quality synthetic data generation pipeline based on LLMs (e.g., Qwen-Max, DeepSeek-V3), ensuring zero real-world data leakage through physical isolation.

1. **Generate diverse base data**:
   ```bash
   python data/generate_diverse_data.py
   ```
2. **Generate adversarial/stress-test data** (includes deep obfuscation, code injection, high-entropy traps, etc.):
   ```bash
   python data/generate_Adversarial_data.py
   ```
3. **Generate multi-source evaluation scenarios**:
   ```bash
   cd multi-source-scenario-data/test_data/
   python synthetic_multi.py
   ```

*(Please remember to configure your API Key in the scripts before running)*

---

## 📥 Data Download

Due to file size limits, datasets and model weights(qwen2.5-7b) are hosted externally. You can download the processed datasets and model checkpoints from our Google Drive: 
**[https://drive.google.com/drive/folders/1BeQ5d5-mzZR1cZAAXbgTyChQoFaHkGB7?usp=sharing]**

---

## 🚀 Training & Evaluation

The framework supports traditional classification baselines as well as LoRA fine-tuning for various open-source LLMs.

### Base Model (BERT)
The model supports multi-label classification for `Identity`, `Location`, `Credential`, and `Safe`.

- **Training**:
  ```bash
  python Bert_train.py
  ```
- **Testing & Evaluation** (Outputs Micro-F1, Accuracy, and 95% Confidence Intervals):
  ```bash
  python Bert_test.py
  ```

### LLM Fine-tuning (LoRA)
We use `ms-swift` for instruction fine-tuning. Taking Qwen 2.5 as an example:

1. **Prepare Data**: Convert generated data into the dialogue format supported by Swift:
   ```bash
   python models/Qwen/convert_data.py
   ```
2. **Execute Fine-tuning** (Supports multiple sub-experiments):
   ```bash
   cd models/Qwen/
   bash run_train_7b.sh
   ```
3. **Merge LoRA Weights**:
   ```bash
   bash run_export.sh
   ```

*(Scripts for Llama-2, ChatGLM3, and Baichuan2 are also included. To fix known code issues in Baichuan2, simply run `models/Baichuan2-7B/fix_baichuan_code.py`)*

---

## 🛡️ Threat Scoring

The core of the MS-SIDM framework lies in its ability to quantify threats in multi-source environments.

1. **Batch LLM Inference**:
   Extract sensitive labels from multi-source data streams.
   ```bash
   python total_score/inference.py
   ```
2. **Calculate Final Total Score (TS)**:
   Determine the risk level (CRITICAL, HIGH, MEDIUM, LOW, SAFE) based on **Information Type Weights**, **Density Compensation Factor ($a$)**, and **Environmental Exposure Amplification Factor**.
   ```bash
   python total_score/final_TS.py
   ```

---

## 📈 Visualization

The `visualization/` directory contains Python scripts to generate publication-grade (e.g., IEEE Access) charts.

- **Model Comparison & Ablation Studies**: `linear vs ours.py`, `wo_compensation.py`
- **Hyperparameter Sensitivity Analysis**: `beta_tau_lambda.py`, `hyperparameter_@.py`
- **Multi-source Impact Analysis**: `multi_source.py`

Simply run the corresponding script to generate high-resolution `.pdf` and `.png` charts in the current directory:

```bash
python visualization/beta_tau_lambda.py
```

To verify the statistical significance of the model's advantages, run McNemar's Test:

```bash
python utils/run_Mcnemar.py
```
```