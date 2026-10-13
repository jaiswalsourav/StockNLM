"""Rank the stocks for the next trading day with the model saved by priceModel.py.

The model predicts which stocks beat the median stock (not whether the market goes up),
so the output is a ranking of the latest day's stocks.

Usage:
    python src/pricePredict.py            # top and bottom 10 of the ranking
    python src/pricePredict.py TCS_NSE    # one stock's place in the ranking
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "data" / "features.csv"
MODEL_FILE = ROOT / "data" / "priceModel.joblib"
SHOW = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", help="e.g. TCS_NSE (file name in data/ without .csv)")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise SystemExit("No saved model: run priceModel.py first.")
    saved = joblib.load(MODEL_FILE)
    if not isinstance(saved, dict) or "cols" not in saved:
        raise SystemExit("priceModel.joblib is from an older version: rerun priceModel.py.")

    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    latest_date = df.index.max()
    day = df[df.index == latest_date]
    if args.symbol and args.symbol not in set(day["symbol"]):
        raise SystemExit(f"'{args.symbol}' has no row for {latest_date.date()}. "
                         f"Options: {', '.join(sorted(day['symbol']))}")

    # Gradient boosting accepts missing values, so a stock with a gap (e.g. no delivery) is still scored
    day = day.assign(score=saved["model"].predict_proba(day[saved["cols"]])[:, 1])
    day = day.sort_values("score", ascending=False).reset_index(drop=True)
    day["rank"] = day.index + 1

    print(f"Ranking for the trading day after {latest_date.date()} ({len(day)} stocks)")
    print(f"Model: '{saved['set']}' features, trained until {saved['trained_until']}, "
          f"test mean IC {saved['test_ic']:.4f} (t-stat {saved['test_ic_t']:.2f})\n")
    cols = ["rank", "symbol", "score"]
    if args.symbol:
        row = day[day["symbol"] == args.symbol].iloc[0]
        print(f"{args.symbol}: rank {row['rank']} of {len(day)}  (score {row['score']:.3f}; "
              f"above 0.5 leans 'beats the median stock')")
    else:
        print(f"Most likely to beat the median stock:\n{day[cols].head(SHOW).round(3).to_string(index=False)}\n")
        print(f"Least likely:\n{day[cols].tail(SHOW).round(3).to_string(index=False)}")
    if abs(saved["test_ic_t"]) < 2:
        print("\nThe test t-stat is below 2, so this ranking is not distinguishable from chance.")
    print("Not financial advice.")


if __name__ == "__main__":
    main()
