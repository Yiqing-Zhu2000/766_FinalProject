
import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report
from utils import load_texts_and_labels
from train_roberta_clean import NewsDataset

def evaluate(model_dir, batch_size=4):
    model_name = os.path.basename(model_dir.rstrip("/"))
    print(f" Evaluating model: {model_name}")
    print(f" Model path: {model_dir}")

    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    _, _, test_texts, test_labels = load_texts_and_labels("data/GtSample50000")
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch['labels'])

    target_names = ["World", "Sports", "Business", "Sci/Tech"]
    print(" Evaluation on clean test set complete:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    for name in ["roberta_clean", "roberta_noisybest", "roberta_noisymid", "roberta_noisyworst"]:
        evaluate(model_dir=f"./output_model/{name}", batch_size=4)
