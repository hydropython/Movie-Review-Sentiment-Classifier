# News Sentiment Classifier

A simple Flask app to classify news headline sentiment (positive/negative) using multiple classical ML models (Logistic Regression, Naive Bayes, Random Forest, Linear SVC) trained with scikit-learn and TF-IDF vectorization.

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
   pip install -r requirements.txt
   ```
2. Train models

   ```bash
   python train.py
   ```
3. Predict 

   ```bash
   python predict.py logistic_regression "I loved this movie!"
   ```
4. Run the Flask app:

   ```bash
    python app.py
    ```
## Project Structure
```
news-sentiment-classifier/
│
├── data/
│   └── news_sentiment.csv       # Dataset with 'review' & 'sentiment' columns
│
├── model/
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── linear_svc.pkl
│   └── vectorizer.pkl           # TF-IDF vectorizer
│
├── train.py                     # Train and save models/vectorizer
├── app.py                       # Flask app with UI and API endpoints
├── predict.py                   # CLI prediction script (optional)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```