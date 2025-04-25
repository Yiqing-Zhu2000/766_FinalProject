
import os
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
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
    data_dir = "GtSample50000"
    val_path = f"{data_dir}/val_indices/GtSample50000_val_indices.pickle"

    train_texts, train_labels, test_texts, test_labels = load_texts_and_labels(data_dir)
    val_indices = load_val_indices(val_path)

    train_texts_arr = [train_texts[i] for i in range(len(train_texts)) if i not in val_indices]
    train_labels_arr = [train_labels[i] for i in range(len(train_labels)) if i not in val_indices]
    val_texts_arr   = [train_texts[i] for i in val_indices]
    val_labels_arr  = [train_labels[i] for i in val_indices]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

    train_dataset = NewsDataset(train_texts_arr, train_labels_arr, tokenizer)
    val_dataset = NewsDataset(val_texts_arr, val_labels_arr, tokenizer)

    training_args = TrainingArguments(
        output_dir="./bert_ckpt",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Test set prediction
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v for k, v in test_encodings.items()})
    preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()

    print("Test classification report:")
    print(classification_report(test_labels, preds))

if __name__ == "__main__":
    main()
