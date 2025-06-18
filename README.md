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
Similar steps to download roBERTa model. Utilizing this method to download is because there is some limitations with the rented remote GPU.

## 3. run those train models for each model and datasets.
BERT and reBERTa model would be utilized for each noisydata situation.
logistics regression training file would already contain the evaluation part. 

## 4. run evaluate files for comparing the results of pre-trained Transformer-Based Models.

## 5. run error_diff_recall.py to compare the relation between classes and models. 

---

## Use ComputeCanada to run
- in CCfolder, bert_CCtest4ENV.sh is about the virtual environment, can use freeze to get suitable requirements.txt file. But this file would use the local .py file to run
```bash
pip freeze | sed 's/+computecanada//' > requirements_clean.txt 
```
- the ComputeCanadaUse/bert_CCtest8Success.sh file is the final version can be used with sbatch. Put all data and codes to $SLURM_TMPDIR and get output models under $SLURM_TMPDIR/Tmp/output_model_CC, this would be cp back to local layer. 
- For details how to use, check my notion link:
[2025Summer/ComputeCanada启动！]（https://www.notion.so/ComputeCanada-1ebf7996c05280f1998ef25755510639）
