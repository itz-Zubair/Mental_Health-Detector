import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
import pickle

class FeatureExtractor:
    def __init__(self, max_features=10000):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.sentiment = SentimentIntensityAnalyzer()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        
    def extract_tfidf(self, texts, fit=True):
        if fit:
            return self.tfidf.fit_transform(texts)
        return self.tfidf.transform(texts)
    
    def extract_sentiment(self, texts):
        features = []
        for text in texts:
            scores = self.sentiment.polarity_scores(text)
            features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        return np.array(features)
    
    def extract_bert(self, texts, batch_size=32):
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                outputs = self.bert(**encoded)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
    
    def save_tfidf(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf, f)
    
    def load_tfidf(self, path):
        with open(path, 'rb') as f:
            self.tfidf = pickle.load(f)

def create_features(input_path, output_dir):
    df = pd.read_csv(input_path)
    texts = df['processed_text'].tolist()
    y = df['target'].values
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor()
    
    # TF-IDF
    print("Extracting TF-IDF...")
    X_tfidf = extractor.extract_tfidf(texts, fit=True)
    extractor.save_tfidf(output_dir / 'tfidf_vectorizer.pkl')
    np.save(output_dir / 'X_tfidf.npy', X_tfidf.toarray())
    
    # Sentiment
    print("Extracting sentiment...")
    X_sentiment = extractor.extract_sentiment(texts)
    np.save(output_dir / 'X_sentiment.npy', X_sentiment)
    
    # BERT (slow, skip if testing)
    print("Extracting BERT embeddings...")
    X_bert = extractor.extract_bert(texts[:100])  # Test with 100 first
    np.save(output_dir / 'X_bert.npy', X_bert)
    
    # Labels
    np.save(output_dir / 'y.npy', y)
    
    print(f"Features saved to {output_dir}")

if __name__ == "__main__":
    create_features("data/processed/cleaned_data.csv", "data/processed")