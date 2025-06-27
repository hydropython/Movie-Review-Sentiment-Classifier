from flask import Flask, request, jsonify, render_template_string
import pickle

app = Flask(__name__)

# Update these paths to match your saved models
MODEL_FILES = {
    "Logistic Regression": ("model/logistic_regression.pkl", "model/vectorizer.pkl"),
    "Naive Bayes": ("model/naive_bayes.pkl", "model/vectorizer.pkl"),
    "Random Forest": ("model/random_forest.pkl", "model/vectorizer.pkl"),
    "Linear SVC": ("model/linear_svc.pkl", "model/vectorizer.pkl"),
}

# Load all models and vectorizer once
models = {}
vectorizers = {}
for model_name, (model_path, vec_path) in MODEL_FILES.items():
    with open(model_path, "rb") as f:
        models[model_name] = pickle.load(f)
    # Vectorizer is the same file for all; just load once
    if not vectorizers:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>News Sentiment Classifier</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        body {
            margin: 0;
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #f0f0f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding: 60px 20px;
        }
        h1 {
            font-size: 2.8rem;
            margin-bottom: 30px;
            text-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        form {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 30px 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        label {
            font-weight: 700;
            font-size: 1rem;
            display: block;
            margin-bottom: 10px;
            color: #f0f0f5;
        }
        select, input[type=text] {
            width: 100%;
            padding: 14px 18px;
            margin-bottom: 20px;
            border-radius: 8px;
            border: none;
            font-size: 1.1rem;
            font-weight: 500;
            color: #333;
            outline: none;
            box-sizing: border-box;
        }
        select {
            cursor: pointer;
        }
        input[type=submit] {
            background-color: #ff6f91;
            color: white;
            font-weight: 700;
            border: none;
            padding: 15px 30px;
            font-size: 1.1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
        }
        input[type=submit]:hover {
            background-color: #ff4b69;
        }
        .result {
            margin-top: 30px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px 25px;
            max-width: 600px;
            width: 100%;
            color: #fff;
            font-size: 1.2rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            text-align: left;
        }
        .result strong {
            display: inline-block;
            min-width: 140px;
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

        <label for="model_select">Choose model:</label>
        <select name="model_select" id="model_select" required>
            {% for model_name in models %}
                <option value="{{ model_name }}" {% if model_name == selected_model %}selected{% endif %}>{{ model_name }}</option>
            {% endfor %}
        </select>

        <input type="text" name="headline" placeholder="Enter news headline here..." required value="{{ headline|default('') }}"/>

        <input type="submit" value="Analyze Sentiment"/>
    </form>

    {% if sentiment %}
        <div class="result">
            <p><strong>Headline:</strong> {{ headline }}</p>
            <p><strong>Model:</strong> {{ selected_model }}</p>
            <p><strong>Predicted Sentiment:</strong> {{ sentiment }} ({{ confidence }} confidence)</p>
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
        headline = request.form.get("headline", "")

        if headline:
            X_vec = vectorizer.transform([headline])
            model = models[selected_model]
            pred = model.predict(X_vec)[0]

            # Try to get probability/confidence if available
            try:
                prob = model.predict_proba(X_vec).max()
            except AttributeError:
                # For models like LinearSVC that do not support predict_proba
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

    X_vec = vectorizer.transform([headline])
    model = models[model_name]

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




