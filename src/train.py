import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Load your real data CSV
df = pd.read_csv("./Data/news_sentiment.csv")  # adjust path as needed

# Check balance (optional)
print(df['sentiment'].value_counts())

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))  # use uni+bi-grams for better context
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

os.makedirs("model", exist_ok=True)
with open("model/logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer_logreg.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")






