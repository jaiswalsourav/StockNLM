"""Train a baseline up/down classifier on data/features.csv with a time-based split."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from priceFeature import BASE_COLS, FEATURE_COLS

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features.csv"
TEST_FRACTION = 0.2


def fit_score(train, test, cols):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    model.fit(train[cols], train["target"])
    return model, accuracy_score(test["target"], model.predict(test[cols]))


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    # Delivery data only covers NSE stocks for about the last year, so use those rows
    df = df.dropna(subset=["target"] + FEATURE_COLS).sort_index()

    # Split by date (not randomly) so the model never sees the future
    cutoff = df.index[int(len(df) * (1 - TEST_FRACTION))]
    train, test = df[df.index < cutoff], df[df.index >= cutoff]
    print(f"Train: {len(train)} rows from {train.index.min().date()} to {train.index.max().date()}")
    print(f"Test:  {len(test)} rows from {test.index.min().date()} to {test.index.max().date()}\n")

    up = (test["target"] == 1).mean()
    print(f"Always-guess-up baseline:       {up:.3f}")
    print(f"Always-guess-down baseline:     {1 - up:.3f}")
    _, acc_base = fit_score(train, test, BASE_COLS)
    print(f"Model without delivery:         {acc_base:.3f}")
    model, acc_all = fit_score(train, test, FEATURE_COLS)
    print(f"Model with delivery (all cols): {acc_all:.3f}\n")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("Feature importance:")
    print(importance.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
