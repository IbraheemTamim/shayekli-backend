import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# Define paths
DATA_PATH = '../large_hybrid_sms_dataset.csv'
MODEL_PATH = 'checkley_model.pkl'

def train():
    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    # We only need text and label
    X = df['text']
    y = df['label']
    
    print("Building model pipeline...")
    # Pipeline: TF-IDF vectorizer (with character n-grams good for Arabic dialects) -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_df=0.9, min_df=2)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    
    accuracy = pipeline.score(X, y)
    print(f"Model trained successfully! Accuracy on training set: {accuracy * 100:.2f}%")
    
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Done!")

if __name__ == "__main__":
    train()
