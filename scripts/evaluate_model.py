import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(path):
    return pd.read_csv(path)

def evaluate(model_path, data_path):
    df = load_data(data_path)

    X = df.drop('species', axis=1)
    y = df['species']

    model = joblib.load(model_path)

    predictions = model.predict(X)

    acc = accuracy_score(y, predictions)
    print(f"\nAccuracy: {acc:.4f}\n")

    report = classification_report(y, predictions)
    print(report)

    # Save classification report
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")

    print("\nEvaluation results saved in /results folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .pkl model file")
    parser.add_argument("--input", required=True, help="path to iris.csv")

    args = parser.parse_args()
    evaluate(args.model, args.input)
