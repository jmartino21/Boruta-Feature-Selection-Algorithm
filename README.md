# Feature Selection for Machine Learning Models

## Overview
This project investigates the impact of **feature selection** on the performance of machine learning classifiers. It compares model accuracy before and after applying the **Boruta feature selection algorithm** on a Parkinson’s disease dataset.

## Features
- **MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)** – Fundamental frequency features.
- **Jitter and Shimmer Variations** – Measures of voice instability.
- **NHR, HNR** – Noise-to-harmonics ratio.
- **Class Label** – Status (0: Healthy, 1: Parkinson’s disease).

## Models Implemented
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**

## Feature Selection
- **Boruta Algorithm**: Identifies the most important features by comparing actual features with shadow features using a **Random Forest-based selection**.
- **Selected Features**: `MDVP:Fo(Hz), MDVP:Flo(Hz), MDVP:APQ`

## Installation
### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn boruta
```

## Dataset
This project requires the **parkinsons.data** dataset. Ensure it is placed in the same directory as the script. If missing, you can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Usage
### Running the Script
Execute the script using:
```bash
python feature_selection_ml.py
```

### Steps Performed
1. Loads and preprocesses the dataset.
2. Trains Naive Bayes, Decision Tree, and Random Forest models on **all features**.
3. Selects the most important features using the **Boruta feature selection algorithm**.
4. Retrains and evaluates models using **only the selected features**.
5. Optimizes **Random Forest hyperparameters**, tuning **n_estimators** and **max_depth**.
6. Visualizes error rates vs. number of trees.

## Output
- Accuracy scores for each classification model **before and after feature selection**.
- Confusion matrices for each classifier.
- Graph showing **error rates vs. number of estimators** for Random Forest.

## Findings
- **Naive Bayes improved the most** (+18% accuracy) after feature selection.
- **Random Forest accuracy increased by 3%**, showing reduced overfitting.
- **Decision Tree accuracy dropped (-4%)**, likely due to reliance on removed features.

## License
This project is open-source and available for modification and use.

