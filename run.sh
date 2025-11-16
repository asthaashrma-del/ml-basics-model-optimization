#!/bin/bash

echo "Training baseline model..."
python scripts/train_model.py --mode baseline --input data/iris.csv --out models/baseline_model.pkl

echo "Training tuned model..."
python scripts/train_model.py --mode tuned --input data/iris.csv --out models/tuned_model.pkl

echo "Evaluating tuned model..."
python scripts/evaluate_model.py --model models/tuned_model.pkl --input data/iris.csv

echo "All tasks completed."
