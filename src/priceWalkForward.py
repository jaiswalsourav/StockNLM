"""Walk-forward test: retrain every month on all earlier data, then score that month.

Compares price-only features against price + NIFTY market features, month by month,
so the result doesn't depend on one lucky (or unlucky) test window.

Usage:
    python src/priceWalkForward.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from priceFeature import MARKET_COLS, PRICE_COLS

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features.csv"
MIN_TRAIN_MONTHS = 12
FEATURE_SETS = {"price only": PRICE_COLS, "price + market": PRICE_COLS + MARKET_COLS}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    model.fit(train[cols], train["target"])
    return pd.Series(model.predict(test[cols]), index=test.index)


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    df = df.dropna(subset=["target"] + PRICE_COLS + MARKET_COLS).sort_index()
    df["month"] = df.index.to_period("M")

    months = sorted(df["month"].unique())[MIN_TRAIN_MONTHS:]
    rows = []
    for month in months:
        train, test = df[df["month"] < month], df[df["month"] == month]
        row = {"month": str(month), "days": test.index.nunique(), "up": (test["target"] == 1).mean()}
        for name, cols in FEATURE_SETS.items():
            row[name] = (fit_predict(train, test, cols) == test["target"]).mean()
        rows.append(row)
        print(f"{row['month']}: up {row['up']:.3f} | " +
              " | ".join(f"{n} {row[n]:.3f}" for n in FEATURE_SETS))

    res = pd.DataFrame(rows).set_index("month")
    res["best_naive"] = res["up"].clip(lower=0.5).where(res["up"] >= 0.5, 1 - res["up"])
    print(f"\n{len(res)} monthly tests ({res.index[0]} to {res.index[-1]})")
    print(f"Always up:        {res['up'].mean():.3f}")
    print(f"Always up/down (best of the two, hindsight): {res['best_naive'].mean():.3f}")
    for name in FEATURE_SETS:
        beats_up = (res[name] > res["up"]).mean()
        beats_naive = (res[name] > res["best_naive"]).mean()
        print(f"{name:15s} mean {res[name].mean():.3f} | beats always-up in {beats_up:.0%} of months "
              f"| beats best naive in {beats_naive:.0%}")


if __name__ == "__main__":
    main()
