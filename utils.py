
import os
import pickle

def load_texts_and_labels(data_dir):
    txt_dir = os.path.join(data_dir, "txt_data")
    with open(os.path.join(txt_dir, "train.txt"), "r", encoding="utf-8") as f:
        train_texts = f.readlines()
    with open(os.path.join(txt_dir, "train_labels.pickle"), "rb") as f:
        train_labels = pickle.load(f)
    with open(os.path.join(txt_dir, "test.txt"), "r", encoding="utf-8") as f:
        test_texts = f.readlines()
    with open(os.path.join(txt_dir, "test_labels.pickle"), "rb") as f:
        test_labels = pickle.load(f)
    return train_texts, train_labels, test_texts, test_labels

def load_val_indices(val_path):
    with open(val_path, "rb") as f:
        val_indices = pickle.load(f)
    return val_indices
