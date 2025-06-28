# 🎬 Movie Review Sentiment Classifier

A simple Flask app to classify movie review sentiment (positive/negative) using classical ML models — Logistic Regression, Naive Bayes, Random Forest, and Linear SVC — trained with scikit-learn and TF-IDF vectorization.

---

## Features

- Train and use multiple classical ML models  
- Interactive web UI with model and heading selection  
- REST API for predictions with JSON input  
- Stylish gradient background UI  

---

## Setup & Usage

1. Clone the repo and install dependencies:

   ```bash
   git clone <repo-url>
   cd news-sentiment-classifier
   pip install -r requirements.txt
   ```
2. Train models

   ```bash
   python src/train.py
   ```
3. Predict 

   ```bash
   python src/predict.py logistic_regression "I loved this movie!"
   ```
4. Run the Flask app:

   ```bash
    python src/app.py
    ```
## Project Structure
```
news-sentiment-classifier/
│
├── assets/
│   └── app_screenshot.png         # Screenshot of the Flask app
│
├── data/
│   └── news_sentiment.csv         # Dataset with 'review' & 'sentiment' columns
│
├── model/
│   ├── logistic_regression.pkl    # Saved Logistic Regression model
│   ├── naive_bayes.pkl            # Saved Naive Bayes model
│   ├── random_forest.pkl          # Saved Random Forest model
│   ├── linear_svc.pkl             # Saved Linear SVC model
│   └── vectorizer.pkl             # TF-IDF vectorizer
│
├── src/
│   ├── train.py                   # Training script
│   ├── predict.py                 # CLI prediction script
│   ├── app.py                     # Flask app with UI & API
│   └── prepare.py                 # Data cleaning/preprocessing utils
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and usage

```
## 🔍 App Preview

![App Screenshot](./assests/Screenshot%202025-06-28%20130214.png)
