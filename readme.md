
# 📄 Qwen Document Layout & OCR Fine-tuning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat-square&logo=python)
![Transformers](https://img.shields.io/badge/Transformers-4.5.2-orange.svg?style=flat-square&logo=huggingface)
![Swift](https://img.shields.io/badge/Swift-4.0.0-red.svg?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A100%2040G-green.svg?style=flat-square&logo=nvidia)

> An end-to-end vision-language pipeline based on the **Qwen3.5-2B** model, designed for advanced document page understanding, layout analysis, region localization, and OCR extraction.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Workflow & Usage](#-workflow--usage)
- [Recommended Use Cases](#-recommended-use-cases)
- [Roadmap & Future Improvements](#-roadmap--future-improvements)

---

## 🎯 Project Overview

This project tackles complex document understanding tasks by integrating a **custom Spatial Enhancement Module**. This module significantly improves the model's ability to capture document structures and spatial relationships.

**Main Objectives:**

- **Identify & Classify:** Recognize document elements across various page layouts.
- **Localize:** Generate precise bounding boxes for key regions.
- **Extract:** Perform highly accurate OCR text extraction.
- **Describe:** Produce concise, contextual alt-text for charts, tables, and figures.

---

## ✨ Key Features

- 🚀 **Powered by Qwen3.5-2B:** Robust baseline for vision-language tasks.
- 🧩 **Custom Spatial Enhancer:** A specialized module injected to strengthen spatial reasoning capabilities.
- 🔄 **Two-Stage Fine-Tuning Pipeline:**
    - *Stage 1:* Full-parameter training focused on learning spatial enhancement.
    - *Stage 2:* LoRA-based efficient training for specific task adaptation.
- 🛠️ **End-to-End Workflow:** Covers everything from data construction to training, inference, and evaluation.

---

## 💻 System Requirements

| Component | Specification |
| :--- | :--- |
| **Python** | `3.12` |
| **Transformers** | `4.5.2` |
| **Swift** | `4.0.0` |
| **Hardware** | NVIDIA A100 40G (or equivalent) |
| **Precision** | `bf16` (Highly Recommended) |

---

## 🛠️ Installation

It is highly recommended to use an isolated virtual environment for this project.

**1. Create and activate a virtual environment:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install "transformers==4.5.2"
pip install "swift==4.0.0"
pip install pillow tqdm numpy python-Levenshtein
```

> **Note:** If your environment requires a specific CUDA-compatible PyTorch build, please install the appropriate wheel from the [official PyTorch website](https://pytorch.org/).

---

## 📂 Project Structure

```text
.
├── qwen_training.py       # Converts raw JSON annotations to unified training format
├── stage1_dataset.py      # Builds VQA-style training samples for Stage 1
├── train_stage1.py        # Stage 1: Full-parameter training script
├── train_stage2.py        # Stage 2: LoRA-based training script
├── model_patcher.py       # Injects spatial enhancer and adds forward hooks
├── run_original_model.py  # Batch inference using the vanilla model
├── run_with_adapter.py    # Inference using Base Model + LoRA + Spatial Enhancer
├── json_transform.py      # Converts .mmd output files to structured JSON
└── DLA_OCR_RO_score.py    # Evaluation script (F1, RO, OCR accuracy, mIoU)
```

---

## 🚀 Workflow & Usage

> **⚠️ Important:** The scripts currently use **hard-coded local paths**. Please ensure you update the directory paths in the scripts to match your local data layout before executing.

### 1️⃣ Data Preparation

Generate the JSONL dataset suitable for downstream training from your raw annotations:

```bash
python qwen_training.py
```

*(Optional)* Build additional VQA-style examples for Stage 1:

```bash
python stage1_dataset.py
```

### 2️⃣ Training Pipeline

**Stage 1: Spatial Enhancement Learning**

Freezes most base parameters and uses full-parameter training for the spatial module.

```bash
python train_stage1.py
```

**Stage 2: Task Adaptation**

LoRA-based fine-tuning optimized for OCR and layout-style tasks.

```bash
python train_stage2.py
```

### 3️⃣ Inference

You can run inference using either the vanilla model or the enhanced model:

```bash
# Option A: Original Model
python run_original_model.py

# Option B: Enhanced Model (Base + LoRA + Spatial Enhancer)
python run_with_adapter.py
```

### 4️⃣ Evaluation

Compare your predictions against ground-truth annotations to get key metrics (F1, RO, mIoU):

```bash
python DLA_OCR_RO_score.py
```

---

## 💡 Recommended Use Cases

This architecture excels in scenarios requiring deep structural understanding, including:

- 📑 Automated Document OCR & Digitization
- 📐 Complex Page Layout Understanding
- 🎯 Precise Document Element Localization
- 📊 Table and Figure Contextualization
- 🧠 Vision-Language Fine-Tuning Research

---

## 🗺️ Roadmap & Future Improvements

- [ ] **Configuration Management:** Replace hard-coded paths with YAML/JSON config files.
- [ ] **CLI Support:** Add `argparse` for flexible command-line execution (e.g., overriding batch size, LR).
- [ ] **Experiment Tracking:** Integrate logging and visualization tools (e.g., Weights & Biases, TensorBoard).
- [ ] **Documentation:** Provide a sample dataset format guide and a mini-tutorial.