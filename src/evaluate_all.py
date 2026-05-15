import numpy as np
import pickle
from sklearn.metrics import classification_report, f1_score

def load_test_data():
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_test, y_test

def evaluate_sklearn_model(model_path, X_test, y_test, classes, name):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if name == "Naive Bayes":
        X_test = np.abs(X_test[:, :10000])
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n=== {name} ===")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    return f1

if __name__ == "__main__":
    classes = ["stress", "depression", "bipolar", "personality_disorder", "anxiety"]
    X_test, y_test = load_test_data()
    
    results = {}
    
    # Logistic Regression
    results['Logistic Regression'] = evaluate_sklearn_model(
        "models/logistic_regression.pkl", X_test, y_test, classes, "Logistic Regression"
    )
    
    # Naive Bayes
    results['Naive Bayes'] = evaluate_sklearn_model(
        "models/naive_bayes.pkl", X_test, y_test, classes, "Naive Bayes"
    )
    
    # BERT Fine-tuned (from Colab)
    results['BERT Fine-tuned'] = 0.8106  # From Colab evaluation
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        status = "✓ BEST" if f1 >= 0.80 else ""
        print(f"{name:20s}: {f1:.4f} {status}")
    
    print(f"\nTarget (80%): {'ACHIEVED' if max(results.values()) >= 0.80 else 'NOT ACHIEVED'}")