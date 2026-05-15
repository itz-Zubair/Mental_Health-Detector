import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from pathlib import Path

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Combine title and text
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['content'].apply(preprocessor.preprocess)
    
    # Remove empty rows
    df = df[df['processed_text'].str.len() > 0]
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "data/raw/data_to_be_cleansed.csv"
    output_file = "data/processed/cleaned_data.csv"
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    preprocess_data(input_file, output_file)