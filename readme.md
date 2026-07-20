# Qwen Document Layout / OCR Fine-tuning Project

This project is designed for document page understanding tasks. It uses a Qwen3.5-2B based model to perform layout analysis, region localization, OCR extraction, and alt-text generation for figures and tables. A custom spatial enhancement module is also integrated to improve the model's ability to capture document structure and spatial relationships.

## 1. Project Overview

The main objectives of this project are:

- Recognize and classify document elements on each page
- Generate bounding boxes for key regions
- Extract OCR text from the document page
- Produce concise descriptions for charts, tables, and figures
- Improve performance through a two-stage fine-tuning pipeline

## 2. Key Features

- Fine-tuning and inference based on Qwen3.5-2B model
- Custom Spatial Enhancer module to strengthen spatial reasoning
- Two-stage training pipeline:
  - Stage 1: full-parameter training to learn spatial enhancement capabilities
  - Stage 2: LoRA-based training for efficient task adaptation
- Complete workflow covering data construction, training, inference, and evaluation

## 3. Environment

This README is tailored for the following environment:

- Python: 3.12
- Transformers: 5.2.0
- Swift: 4.0.0
- GPU: NVIDIA A100 40G
- Recommended precision: bf16

## 4. Installation

It is recommended to create a dedicated virtual environment first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install "transformers==4.5.2"
pip install "swift==4.0.0"
pip install pillow tqdm numpy python-Levenshtein
```

If your environment requires a CUDA-compatible PyTorch build, please install the appropriate wheel for your system. On an A100 40G server, bf16 training is generally recommended.

## 5. Project Structure

- qwen_training.py: converts original annotation JSON files into a unified training dataset format
- train_stage1.py: Stage 1 full-parameter training script
- train_stage2.py: Stage 2 LoRA-based training script
- model_patcher.py: injects the spatial enhancement module and adds forward hooks
- stage1_dataset.py: builds VQA-style training samples
- run_original_model.py: performs batch inference with the original model
- run_with_adapter.py: performs inference using the base model + LoRA + spatial enhancer
- json_transform.py: converts inference output from .mmd files into JSON files
- DLA_OCR_RO_score.py: evaluates layout/OCR quality and reports F1, RO, OCR accuracy, and mIoU

## 6. Data Preparation

This project assumes that you already have document images and corresponding annotation files. The scripts currently use hard-coded local paths, so you should update them according to your own directory layout before running.

## 7. Training Workflow

### 7.1 Generate the training data

Run:

```bash
python qwen_training.py
```

This script scans the original annotation directory and builds a JSONL dataset suitable for downstream training.

### 7.2 Build Stage 1 VQA-style data (optional)

If you want to create additional VQA-style examples for Stage 1 training, run:

```bash
python stage1_dataset.py
```

### 7.3 Stage 1 training

Run:

```bash
python train_stage1.py
```

This stage uses full-parameter training and primarily focuses on learning the spatial enhancement capability while freezing most of the base model parameters.

### 7.4 Stage 2 training

Run:

```bash
python train_stage2.py
```

This stage uses LoRA-based fine-tuning for better adaptation to OCR and layout-style tasks.

## 8. Inference

### 8.1 Run inference with the original model

```bash
python run_original_model.py
```

### 8.2 Run inference with the base model + LoRA + spatial enhancer

```bash
python run_with_adapter.py
```

## 9. Evaluation

Run:

```bash
python DLA_OCR_RO_score.py
```

This script compares prediction results with ground-truth annotations and reports the main evaluation metrics.

## 10. Notes

- The current scripts use hard-coded absolute paths. Please update them when moving the project to another machine or environment.
- Training hyperparameters such as batch size, learning rate, and maximum sequence length should be adjusted based on GPU memory and dataset size.
- For A100 40G, bf16 is recommended for better throughput and memory efficiency.
- The model output is typically stored as .mmd files, which can later be converted to JSON via json_transform.py for downstream analysis.

## 11. Recommended Use Cases

This project is suitable for:

- Document OCR recognition
- Page layout understanding
- Document element localization
- Table and figure understanding
- Vision-language fine-tuning for document tasks

## 12. Future Improvements

Possible improvements include:

- Replacing hard-coded paths with configuration files
- Supporting command-line arguments for training and inference
- Adding logging and visualization for training results
- Providing a more complete example workflow and dataset format guide
