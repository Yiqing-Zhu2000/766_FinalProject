
import os
import pickle
import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from utils import load_texts_and_labels, load_val_indices

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }

def main():
    data_dir = "data/NoisyAG-NewsMid"
    val_path = f"{data_dir}/val_indices/NoisyAG-NewsMid_val_indices.pickle"

    print(f"🟢 Loading dataset from: {data_dir}")
    train_texts, train_labels, test_texts, test_labels = load_texts_and_labels(data_dir)
    val_indices = load_val_indices(val_path)

    train_texts_arr = [train_texts[i] for i in range(len(train_texts)) if i not in val_indices]
    train_labels_arr = [train_labels[i] for i in range(len(train_labels)) if i not in val_indices]
    val_texts_arr   = [train_texts[i] for i in val_indices]
    val_labels_arr  = [train_labels[i] for i in val_indices]

    model_path = "./bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4)

    train_dataset = NewsDataset(train_texts_arr, train_labels_arr, tokenizer)
    val_dataset = NewsDataset(val_texts_arr, val_labels_arr, tokenizer)

    output_path = "./output_model/bert_noisymid"

    training_args = TrainingArguments(
        output_dir=output_path,
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        logging_dir=output_path + "/logs",
        logging_steps=1000000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    print("🚀 Training BERT ...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"⏱️ Training time: {{(end_time - start_time) / 60:.2f}} minutes")

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    main()
