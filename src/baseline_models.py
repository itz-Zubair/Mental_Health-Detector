import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle
from pathlib import Path

def load_data(processed_dir):
    X_train = np.load(f"{processed_dir}/X_train.npy")
    X_val = np.load(f"{processed_dir}/X_val.npy")
    X_test = np.load(f"{processed_dir}/X_test.npy")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    y_val = np.load(f"{processed_dir}/y_val.npy")
    y_test = np.load(f"{processed_dir}/y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    print(f"Val Macro F1: {val_f1:.4f}")
    
    return model

def train_naive_bayes(X_train, y_train, X_val, y_val):
    print("Training Naive Bayes...")
    # Use only TF-IDF part (first 10000 features), make non-negative
    X_train_nb = np.abs(X_train[:, :10000])
    X_val_nb = np.abs(X_val[:, :10000])
    
    model = MultinomialNB()
    model.fit(X_train_nb, y_train)
    
    val_pred = model.predict(X_val_nb)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    print(f"Val Macro F1: {val_f1:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, class_names, use_tfidf_only=False):
    if use_tfidf_only:
        X_test = np.abs(X_test[:, :10000])
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test Macro F1: {macro_f1:.4f}")
    
    return macro_f1

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    classes = ["stress", "depression", "bipolar", "personality_disorder", "anxiety"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data("data/processed")
    
    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
    print("\n=== Logistic Regression ===")
    lr_f1 = evaluate_model(lr_model, X_test, y_test, classes)
    save_model(lr_model, "models/logistic_regression.pkl")
    
    # Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train, X_val, y_val)
    print("\n=== Naive Bayes ===")
    nb_f1 = evaluate_model(nb_model, X_test, y_test, classes, use_tfidf_only=True)
    save_model(nb_model, "models/naive_bayes.pkl")
    
    print(f"\nBaseline Results:")
    print(f"Logistic Regression F1: {lr_f1:.4f}")
    print(f"Naive Bayes F1: {nb_f1:.4f}")