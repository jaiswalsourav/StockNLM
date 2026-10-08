"""Compare the price-only model against price + news sentiment on the same time split."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from priceFeature import FEATURE_COLS as PRICE_COLS

ROOT = Path(__file__).resolve().parent.parent
NEWS_COLS = ["sentiment", "news_count"]


def load() -> pd.DataFrame:
    feats = pd.read_csv(ROOT / "data" / "features.csv", index_col=0, parse_dates=True)
    feats = feats.dropna(subset=["target"]).reset_index()

    sent = pd.read_csv(ROOT / "news" / "daily_sentiment.csv", parse_dates=["date"])
    sent = sent.rename(columns={"date": "Date"})

    df = feats.merge(sent, on=["symbol", "Date"], how="left")
    df["has_news"] = df["news_count"].notna()
    df[NEWS_COLS] = df[NEWS_COLS].fillna(0)
    return df.sort_values("Date").set_index("Date")


def fit_score(train, test, cols) -> float:
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    model.fit(train[cols], train["target"])
    return accuracy_score(test["target"], model.predict(test[cols]))


def main() -> None:
    df = load()
    print(f"Rows with news: {df['has_news'].sum()} of {len(df)} "
          f"({df.loc[df['has_news']].index.min().date()} to {df.index.max().date()})\n")

    cutoff = df.index[int(len(df) * 0.8)]
    train, test = df[df.index < cutoff], df[df.index >= cutoff]
    news_test = test[test["has_news"]]

    print(f"Test rows: {len(test)} | test rows with news: {len(news_test)}")
    print(f"Always-up baseline (all test):       {(test['target'] == 1).mean():.3f}")
    print(f"Always-up baseline (news days only): {(news_test['target'] == 1).mean():.3f}\n")

    for name, cols in [("price only", PRICE_COLS), ("price + news", PRICE_COLS + NEWS_COLS)]:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
        ).fit(train[cols], train["target"])
        acc_all = accuracy_score(test["target"], model.predict(test[cols]))
        acc_news = accuracy_score(news_test["target"], model.predict(news_test[cols]))
        print(f"{name:13s} accuracy: all test {acc_all:.3f} | news days only {acc_news:.3f}")


if __name__ == "__main__":
    main()
