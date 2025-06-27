import sys
import pickle

MODEL_FILES = {
    "logistic_regression": "model/logistic_regression.pkl",
    "naive_bayes": "model/naive_bayes.pkl",
    "random_forest": "model/random_forest.pkl",
    "linear_svc": "model/linear_svc.pkl"
}

VECTORIZER_PATH = "model/vectorizer.pkl"

def load_model_and_vectorizer(model_key):
    if model_key not in MODEL_FILES:
        print(f"Model '{model_key}' not found. Available models: {list(MODEL_FILES.keys())}")
        sys.exit(1)
    with open(MODEL_FILES[model_key], "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_name> \"<your headline>\"")
        print(f"Available models: {list(MODEL_FILES.keys())}")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    review = sys.argv[2]

    model, vectorizer = load_model_and_vectorizer(model_name)
    X_vec = vectorizer.transform([review])

    # For RandomForest and other classifiers without predict_proba, handle carefully
    try:
        confidence = model.predict_proba(X_vec).max()
    except AttributeError:
        confidence = None  # SVC or others may not have predict_proba
    
    prediction = model.predict(X_vec)[0]
    label = "positive" if prediction == 1 else "negative"

    if confidence is not None:
        print(f"{label} ({confidence:.2f})")
    else:
        print(f"{label} (confidence not available)")

if __name__ == "__main__":
    main()


