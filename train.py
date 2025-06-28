# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Sample training data (replace this with your real dataset)
data = {
    "text": [
        "I love this!", "This is awful.", "Absolutely fantastic product.",
        "Worst experience ever.", "I'm happy with the results.", "I hate it"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Directory to save models
os.makedirs("model", exist_ok=True)

# Models dictionary
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "nb": MultinomialNB(),
    "rf": RandomForestClassifier(n_estimators=100),
    "svc": LinearSVC(max_iter=1000)
}

for name, model in models.items():
    print(f"Training {name}...")

    # Fit a fresh vectorizer for each model (to ensure matching features)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model.fit(X_train_vec, y_train)

    # Save model
    with open(f"model/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer separately for this model
    with open(f"model/vectorizer_{name}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

print("All models and vectorizers saved successfully.")

