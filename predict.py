import pickle
import re
import sys
import os

def load_model_and_vectorizer():
    with open("model/logreg_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer_logreg.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # simple clean
    return text

def predict_sentiment(text):
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    # Boost confidence if strong negative words found
    strong_neg_words = ['hate', 'awful', 'terrible', 'worst', 'horrible', 'disgusting', 'bad']
    if any(word in cleaned for word in strong_neg_words):
        # Force negative with high confidence
        negative_class_index = list(model.classes_).index(0)
        confidence = max(round(proba[negative_class_index]*100, 1), 98.0)
        sentiment = "negative"
    else:
        confidence = round(max(proba)*100, 1)
        sentiment = "positive" if pred == 1 else "negative"

    return sentiment, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"your sentence here\"")
        sys.exit(1)

    text_input = ' '.join(sys.argv[1:])
    sentiment, confidence = predict_sentiment(text_input)
    print(f"Sentiment: {sentiment} (Confidence: {confidence}%)")


