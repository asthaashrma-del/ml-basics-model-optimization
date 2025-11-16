import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def train_baseline(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def train_tuned(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=500))
    ])

    param_grid = {
        'model__C': [0.1, 1, 5, 10],
        'model__solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_

def main(mode, input_path, output_path):
    df = load_data(input_path)

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if mode == "baseline":
        print("Training baseline model...")
        model = train_baseline(X_train, y_train)
    elif mode == "tuned":
        print("Training tuned model...")
        model = train_tuned(X_train, y_train)
    else:
        raise ValueError("Invalid mode! Use 'baseline' or 'tuned'.")

    joblib.dump(model, output_path)
    print(f"Model saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="baseline or tuned")
    parser.add_argument("--input", required=True, help="path to iris.csv")
    parser.add_argument("--out", required=True, help="output model path")

    args = parser.parse_args()
    main(args.mode, args.input, args.out)
