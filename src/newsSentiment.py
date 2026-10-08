"""Score news headlines in news/ with FinBERT and save daily sentiment per symbol."""
from pathlib import Path

import pandas as pd
from transformers import pipeline

NEWS_DIR = Path(__file__).resolve().parent.parent / "news"
OUT_FILE = NEWS_DIR / "daily_sentiment.csv"
MODEL_NAME = "ProsusAI/finbert"
SIGN = {"positive": 1, "neutral": 0, "negative": -1}


def load_headlines() -> pd.DataFrame:
    frames = []
    for path in sorted(NEWS_DIR.glob("*.csv")):
        if path.name == OUT_FILE.name:
            continue
        frames.append(pd.read_csv(path, parse_dates=["published"]))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["title", "published"]).drop_duplicates(subset=["symbol", "link"])
    # Google News titles end with " - Source"; drop that suffix before scoring
    df["text"] = df["title"].str.replace(r"\s+-\s+[^-]+$", "", regex=True)
    return df


def main() -> None:
    df = load_headlines()
    print(f"Scoring {len(df)} headlines with {MODEL_NAME} ...")

    clf = pipeline("text-classification", model=MODEL_NAME, truncation=True, max_length=128)
    results = clf(df["text"].tolist(), batch_size=32)

    df["label"] = [r["label"] for r in results]
    # score in [-1, 1]: +confidence if positive, -confidence if negative, 0 if neutral
    df["score"] = [SIGN[r["label"]] * r["score"] for r in results]
    df["date"] = df["published"].dt.tz_convert("Asia/Kolkata").dt.date

    daily = (
        df.groupby(["symbol", "date"])
        .agg(sentiment=("score", "mean"), news_count=("score", "size"))
        .reset_index()
    )
    daily.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(daily)} symbol-days -> {OUT_FILE.name}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
