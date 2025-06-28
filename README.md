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
│   └── news_sentiment.csv           # Dataset with 'review' & 'sentiment' columns
│
├── model/
│   ├── logistic_regression.pkl     # Saved Logistic Regression model
│   ├── naive_bayes.pkl             # Saved Naive Bayes model
│   ├── random_forest.pkl           # Saved Random Forest model
│   ├── linear_svc.pkl              # Saved Linear SVC model
│   └── vectorizer.pkl              # TF-IDF vectorizer (or multiple vectorizers if separate)
│
├── src/
│   ├── train.py                   # Training script (load data, train models, save models/vectorizer)
│   ├── predict.py                 # CLI prediction script
│   ├── app.py                     # Flask app with UI & API
│   └── prepare.py                 # (Optional) Data cleaning, preprocessing utilities
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and usage

```