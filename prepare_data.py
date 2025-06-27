import pandas as pd
import os

# Load IMDb dataset
df = pd.read_csv("data/IMDB Dataset.csv")  # make sure the file name is correct

# Clean and format
df.dropna(subset=["review", "sentiment"], inplace=True)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Save prepared data
os.makedirs("data", exist_ok=True)
df.to_csv("data/news_sentiment.csv", index=False)
print("✅ IMDb data formatted and saved to data/news_sentiment.csv")

