
# train_logreg_clean.py

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import load_texts_and_labels

def main():
    data_dir = "data/GtSample50000"
    print(f" Loading data from: {data_dir}")
    train_texts, train_labels, test_texts, test_labels = load_texts_and_labels(data_dir)

    print("Extracting TF-IDF features ")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print(" Training Logistic Regression ...")
    clf = LogisticRegression(max_iter=2000, solver='liblinear')
    clf.fit(X_train, train_labels)

    print(" Evaluating on clean test set:")
    preds = clf.predict(X_test)

    target_names = ["World", "Sports", "Business", "Sci/Tech"]
    print(classification_report(test_labels, preds, target_names=target_names))

    print("Saving model and vectorizer ...")
    os.makedirs("output_model/logreg_clean", exist_ok=True)
    with open("output_model/logreg_clean/model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("output_model/logreg_clean/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    main()
