import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_data(processed_dir, test_size=0.2, val_size=0.1):
    # Load features
    X_tfidf = np.load(f"{processed_dir}/X_tfidf.npy")
    X_sentiment = np.load(f"{processed_dir}/X_sentiment.npy")
    y = np.load(f"{processed_dir}/y.npy")
    
    # Combine TF-IDF + sentiment
    X = np.hstack([X_tfidf, X_sentiment])
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    # Save
    output_dir = Path(processed_dir)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    split_data("data/processed")