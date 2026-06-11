# Political Tweet Classification: SLP vs. MLP vs. SVM

NLP pipeline for predicting the political party (Democrat vs. Republican) of U.S. congressional tweets, using a dataset of 86,460 tweets from elected officials.

## Overview

The pipeline covers text cleaning, TF-IDF vectorization, exploratory word frequency analysis, and model evaluation across three approaches:

- **Single-Layer Perceptron (SLP)** — shallow neural network baseline
- **Multi-Layer Perceptron (MLP)** — 3-layer network with 25 neurons per layer
- **LinearSVC (SVM)** — linear support vector machine with Platt scaling

## Results

| Model | AUC |
|---|---|
| **LinearSVC (SVM)** | **0.846** |
| MLP (3 layers, 25 neurons) | 0.805 |
| SLP (1 layer, 10 neurons) | 0.788 |

LinearSVC outperformed both neural network approaches — consistent with the known strength of linear models on high-dimensional sparse TF-IDF feature spaces. Word frequency analysis revealed a genuine partisan signal: Democrat tweets reference `trump` prominently while Republican tweets focus on policy terms (`tax`, `bill`, `act`).

## Dataset

`tweetdata1.csv` — upload to the same directory as the notebook before running.

## Tech Stack

Python · pandas · scikit-learn · NLTK · emoji · matplotlib
