"""Predict tomorrow's direction (UP/DOWN) for a stock using the price-only model.

Usage:
    python src/pricePredict.py TCS_NSE
    python src/pricePredict.py TCS_NSE --retrain
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "data" / "features.csv"
MODEL_FILE = ROOT / "data" / "priceModel.joblib"
FEATURE_COLS = [
    "return_1d", "return_5d", "ma_5_ratio", "ma_20_ratio", "ma_50_ratio",
    "volatility_20d", "rsi_14", "macd", "volume_change",
]


def train(df: pd.DataFrame) -> RandomForestClassifier:
    labelled = df.dropna(subset=["target"])
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    model.fit(labelled[FEATURE_COLS], labelled["target"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="e.g. TCS_NSE (file name in data/ without .csv)")
    parser.add_argument("--retrain", action="store_true", help="retrain even if a saved model exists")
    args = parser.parse_args()

    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    if args.symbol not in set(df["symbol"]):
        raise SystemExit(f"Unknown symbol '{args.symbol}'. Options: {', '.join(sorted(set(df['symbol'])))}")

    if MODEL_FILE.exists() and not args.retrain:
        model = joblib.load(MODEL_FILE)
    else:
        model = train(df)
        joblib.dump(model, MODEL_FILE)
        print(f"Trained on {df['target'].notna().sum()} rows, saved {MODEL_FILE.name}")

    latest = df[df["symbol"] == args.symbol].sort_index().iloc[[-1]]
    prob_up = model.predict_proba(latest[FEATURE_COLS])[0][list(model.classes_).index(1.0)]

    print(f"\n{args.symbol} as of {latest.index[0].date()}")
    print(f"Prediction for next trading day: {'UP' if prob_up >= 0.5 else 'DOWN'}")
    print(f"Probability up: {prob_up:.1%}")
    print("(Model accuracy is ~48%, about the same as guessing. Not financial advice.)")


if __name__ == "__main__":
    main()
