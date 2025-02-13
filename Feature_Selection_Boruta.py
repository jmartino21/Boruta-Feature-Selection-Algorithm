import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}. Please make sure the dataset is in the correct directory.")
    df = pd.read_csv(filename, delimiter=',', header=0)
    return df

# Preprocess dataset
def preprocess_data(df):
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR']
    X = df[features].values
    y = df[['status']].values.ravel()
    return train_test_split(X, y, test_size=0.5, random_state=123)

# Train and evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.2%}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return accuracy

# Feature selection using Boruta
def boruta_feature_selection(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    feat_selector = BorutaPy(model, n_estimators='auto', max_iter=10, random_state=42)
    feat_selector.fit(X_train, y_train)
    
    selected_features = [feat for feat, keep in zip(features, feat_selector.support_) if keep]
    print("Selected Features:", selected_features)
    return selected_features

# Tune Random Forest hyperparameters
def tune_random_forest(X_train, y_train, X_test, y_test):
    x = list(range(1, 10))
    error_values = []
    
    for d in range(1, 6):
        for N in range(1, 10):
            rfc = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy', random_state=123)
            y_pred = rfc.fit(X_train, y_train).predict(X_test)
            error_RFC = np.mean(y_test != y_pred)
            error_values.append(error_RFC)
    
    error_rate_list = [error_values[i:i+9] for i in range(0, len(error_values), 9)]
    
    for i, y in enumerate(error_rate_list):
        plt.plot(x, y, label=f'd={i+1}')
    
    plt.title("Random Forest: Error Rate vs. Number of Trees")
    plt.xlabel("Number of Trees (N)")
    plt.ylabel("Error Rate")
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.show()

# Main script execution
if __name__ == "__main__":
    file_path = 'parkinsons.data'  # Ensure this file is present
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Evaluate models before feature selection
    print("Evaluating models with all features...")
    evaluate_model(GaussianNB(), X_train, y_train, X_test, y_test, "Naive Bayes")
    evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, "Decision Tree")
    evaluate_model(RandomForestClassifier(n_estimators=5, max_depth=3, random_state=123), X_train, y_train, X_test, y_test, "Random Forest")
    
    # Feature selection
    selected_features = boruta_feature_selection(X_train, y_train)
    X_train_selected = X_train[:, [features.index(f) for f in selected_features]]
    X_test_selected = X_test[:, [features.index(f) for f in selected_features]]
    
    # Evaluate models after feature selection
    print("Evaluating models with selected features...")
    evaluate_model(GaussianNB(), X_train_selected, y_train, X_test_selected, y_test, "Naive Bayes (Selected Features)")
    evaluate_model(DecisionTreeClassifier(), X_train_selected, y_train, X_test_selected, y_test, "Decision Tree (Selected Features)")
    evaluate_model(RandomForestClassifier(n_estimators=5, max_depth=3, random_state=123), X_train_selected, y_train, X_test_selected, y_test, "Random Forest (Selected Features)")
    
    # Hyperparameter tuning
    tune_random_forest(X_train, y_train, X_test, y_test)
