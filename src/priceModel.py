"""Train a baseline up/down classifier on data/features.csv with a time-based split."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from priceFeature import FEATURE_COLS

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features.csv"
TEST_FRACTION = 0.2


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    df = df.dropna(subset=["target"]).sort_index()

    # Split by date (not randomly) so the model never sees the future
    cutoff = df.index[int(len(df) * (1 - TEST_FRACTION))]
    train, test = df[df.index < cutoff], df[df.index >= cutoff]
    print(f"Train: {len(train)} rows up to {train.index.max().date()}")
    print(f"Test:  {len(test)} rows from {test.index.min().date()}\n")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    model.fit(train[FEATURE_COLS], train["target"])
    pred = model.predict(test[FEATURE_COLS])

    always_up = (test["target"] == 1).mean()
    print(f"Always-guess-up baseline: {always_up:.3f}")
    print(f"Random forest accuracy:   {accuracy_score(test['target'], pred):.3f}\n")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("Feature importance:")
    print(importance.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
