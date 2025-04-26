# 766_FinalProject

## Requirements

- Python 3.10+
- Conda (Miniconda or Anaconda)
- GPU with at least **24GB VRAM** (e.g., RTX 4090)
- CUDA 11.8 or 12.1+

---

## Setup Instructions
### 1. Create the Conda Environment
I rented the GPU online to do. 
contain python@3.9.0、cudatoolkit@11.8.0、cudnn@8.9.2.26、pytorch@2.0.1、opencv-python@4.9.0.80、matplotlib@3.8.3

```bash
conda activate torch251tf2170-py310-cuda124
```

## 2. download the original bert and roBERTa model. 
```bash
mkdir bert-base-uncased && cd bert-base-uncased

wget https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/config.json
wget https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/tokenizer.json
wget https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/vocab.txt
```

## 3. run those train models for each model and datasets.
BERT and reBERTa model would be utilized for each noisydata situation.
logistics regression training file would already contain the evaluation part. 

## 4. run evaluate files for comparing the results of pre-trained Transformer-Based Models.

## 5. run error_diff_recall.py to compare the relation between classes and models. 
