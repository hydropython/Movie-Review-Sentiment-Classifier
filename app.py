from flask import Flask, request, jsonify, render_template_string
import pickle
import os

app = Flask(__name__)

# Define your models and vectorizers paths
MODEL_FILES = {
    "Logistic Regression": ("model/logreg_model.pkl", "model/vectorizer_logreg.pkl"),
    "Naive Bayes": ("model/nb_model.pkl", "model/vectorizer_nb.pkl"),
    "Random Forest": ("model/rf_model.pkl", "model/vectorizer_rf.pkl"),
    "Linear SVC": ("model/svc_model.pkl", "model/vectorizer_svc.pkl"),
}

# Load models and vectorizers
models = {}
vectorizers = {}

for model_name, (model_path, vec_path) in MODEL_FILES.items():
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Missing model or vectorizer for {model_name}")
    with open(model_path, "rb") as f:
        models[model_name] = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizers[model_name] = pickle.load(f)

# Headings for UI dropdown
HEADINGS = {
    "ai_powered": "🧠 AI-Powered News Sentiment Classifier",
    "decode_mood": "📰 Decode the Mood of Headlines",
    "real_time": "🔍 Real-Time News Emotion Detector",
    "market_mood": "📈 Market Mood from Headlines",
    "your_news": "🤖 Your News, Our Judgment",
    "sentiment_radar": "🗞️ Sentiment Radar",
    "instant_analyzer": "💬 Instant Headline Analyzer",
    "news_dashboard": "🚦 News Sentiment Dashboard",
    "sentiment_thermometer": "🌡️ Headline Sentiment Thermometer",
    "emotional_classifier": "📢 Emotional Classifier for News Headlines"
}

# Stylish HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>News Sentiment Classifier</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        body {
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Montserrat', sans-serif;
            color: #f0f0f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 50px 20px 80px;
        }

        h1 {
            font-size: 3rem;
            margin-bottom: 40px;
            text-shadow: 0 3px 10px rgba(0, 0, 0, 0.4);
            user-select: none;
        }

        form {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 35px 45px;
            width: 100%;
            max-width: 650px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }

        label {
            display: block;
            margin-bottom: 12px;
            font-weight: 700;
            font-size: 1.1rem;
            color: #e0e0e8;
            user-select: none;
        }

        select, input[type=text] {
            width: 100%;
            padding: 15px 20px;
            margin-bottom: 30px;
            border-radius: 12px;
            border: none;
            font-size: 1.1rem;
            font-weight: 500;
            outline: none;
            box-sizing: border-box;
            transition: box-shadow 0.3s ease;
        }

        select:focus, input[type=text]:focus {
            box-shadow: 0 0 10px #ff6f91;
        }

        select {
            cursor: pointer;
        }

        input[type=submit] {
            width: 100%;
            background: #ff6f91;
            color: white;
            font-weight: 700;
            font-size: 1.2rem;
            padding: 18px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: background-color 0.35s ease;
            user-select: none;
        }

        input[type=submit]:hover {
            background: #ff4b69;
        }

        .result {
            margin-top: 40px;
            background: rgba(255, 255, 255, 0.22);
            border-radius: 15px;
            padding: 25px 35px;
            max-width: 650px;
            width: 100%;
            color: #fff;
            font-size: 1.3rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
            user-select: none;
        }

        .result p {
            margin: 10px 0;
        }

        .result strong {
            display: inline-block;
            min-width: 160px;
            font-weight: 700;
            color: #ffe;
        }

        @media (max-width: 700px) {
            h1 {
                font-size: 2.4rem;
            }
            form, .result {
                padding: 25px 20px;
            }
        }
    </style>
</head>
<body>
    <h1>{{ heading }}</h1>
    <form method="POST" action="/">
        <label for="heading_select">Choose a heading:</label>
        <select name="heading_select" id="heading_select" required>
            {% for key, val in headings.items() %}
                <option value="{{ key }}" {% if key == selected_heading %}selected{% endif %}>{{ val }}</option>
            {% endfor %}
        </select>

        <label for="model_select">Choose a model:</label>
        <select name="model_select" id="model_select" required>
            {% for model_name in models %}
                <option value="{{ model_name }}" {% if model_name == selected_model %}selected{% endif %}>{{ model_name }}</option>
            {% endfor %}
        </select>

        <label for="headline">Enter news headline:</label>
        <input type="text" name="headline" id="headline" placeholder="Enter news headline here..." required value="{{ headline|default('') }}"/>

        <input type="submit" value="Analyze Sentiment" />
    </form>

    {% if sentiment %}
    <div class="result">
        <p><strong>Headline:</strong> {{ headline }}</p>
        <p><strong>Model:</strong> {{ selected_model }}</p>
        <p><strong>Predicted Sentiment:</strong> {{ sentiment }}</p>
        <p><strong>Confidence:</strong> {{ confidence }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = confidence = headline = None
    selected_heading = "ai_powered"
    selected_model = list(MODEL_FILES.keys())[0]

    if request.method == "POST":
        selected_heading = request.form.get("heading_select", "ai_powered")
        selected_model = request.form.get("model_select", selected_model)
        headline = request.form.get("headline", "").strip()

        if headline:
            vectorizer = vectorizers[selected_model]
            model = models[selected_model]
            X_vec = vectorizer.transform([headline])
            pred = model.predict(X_vec)[0]

            try:
                prob = model.predict_proba(X_vec).max()
            except AttributeError:
                prob = None

            sentiment = "🟢 Positive" if pred == 1 else "🔴 Negative"
            confidence = f"{round(prob * 100, 2)}%" if prob is not None else "N/A"

    return render_template_string(
        HTML_TEMPLATE,
        headings=HEADINGS,
        selected_heading=selected_heading,
        heading=HEADINGS.get(selected_heading, "News Sentiment Classifier"),
        models=models.keys(),
        selected_model=selected_model,
        sentiment=sentiment,
        confidence=confidence,
        headline=headline
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "headline" not in data or "model" not in data:
        return jsonify({"error": "Please provide 'headline' and 'model' in JSON body"}), 400

    headline = data["headline"]
    model_name = data["model"]

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    vectorizer = vectorizers[model_name]
    model = models[model_name]

    X_vec = vectorizer.transform([headline])
    pred = model.predict(X_vec)[0]

    try:
        conf = model.predict_proba(X_vec).max()
    except AttributeError:
        conf = None

    sentiment = "positive" if pred == 1 else "negative"

    return jsonify({
        "headline": headline,
        "model": model_name,
        "sentiment": sentiment,
        "confidence": round(conf, 3) if conf is not None else None
    })

if __name__ == "__main__":
    app.run(debug=True)








