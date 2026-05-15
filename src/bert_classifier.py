import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import pickle

def load_data(processed_dir):
    X_bert = np.load(f"{processed_dir}/X_bert.npy")
    y = np.load(f"{processed_dir}/y.npy")
    return X_bert, y

def split_bert_data(X, y, test_size=0.2, val_size=0.1):
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_bert_classifier(X_train, y_train, X_val, y_val):
    print("Training BERT classifier...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    print(f"Val Macro F1: {val_f1:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test Macro F1: {macro_f1:.4f}")
    
    return macro_f1

if __name__ == "__main__":
    classes = ["stress", "depression", "bipolar", "personality_disorder", "anxiety"]
    
    X_bert, y = load_data("data/processed")
    X_train, X_val, X_test, y_train, y_val, y_test = split_bert_data(X_bert, y)
    
    model = train_bert_classifier(X_train, y_train, X_val, y_val)
    
    print("\n=== BERT Classifier ===")
    bert_f1 = evaluate_model(model, X_test, y_test, classes)
    
    with open("models/bert_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nBERT F1: {bert_f1:.4f}")
    print(f"Target: 0.80")
    print(f"Beat baseline? {'Yes' if bert_f1 > 0.7812 else 'No'}")