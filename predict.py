# predict.py

import sys
import pickle
import numpy as np

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

if len(sys.argv) < 2:
    print("Usage: python predict.py \"<your headline>\"")
    sys.exit(1)

review = sys.argv[1]
X_vec = vectorizer.transform([review])
prediction = model.predict(X_vec)[0]
confidence = model.predict_proba(X_vec).max()

label = "positive" if prediction == 1 else "negative"
print(f"{label} ({confidence:.2f})")
