"""Turn raw price CSVs in data/ into model features and a next-day up/down target."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "features.csv"


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


MARKET_SYMBOL = "NIFTY50"
PRICE_COLS = [
    "return_1d", "return_5d", "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_50_ratio",
    "ma_200_ratio", "volatility_20d", "rsi_14", "macd", "volume_change",
]
MARKET_COLS = [
    "market_return_1d", "market_return_5d", "market_ma_20_ratio",
    "market_volatility_20d", "rel_return_1d", "rel_return_5d",
]
# NSE delivery data only exists for NSE stocks and roughly the last year (see priceCollect.py)
DELIVERY_COLS = ["delivery_pct", "delivery_vs_20d"]
BASE_COLS = PRICE_COLS + MARKET_COLS
FEATURE_COLS = BASE_COLS + DELIVERY_COLS


def market_features(nifty: pd.DataFrame) -> pd.DataFrame:
    """NIFTY 50 context, computed once and joined onto every stock by date."""
    close = nifty["Close"]
    m = pd.DataFrame(index=nifty.index)
    m["market_return_1d"] = close.pct_change()
    m["market_return_5d"] = close.pct_change(5)
    m["market_ma_20_ratio"] = close / close.rolling(20).mean() - 1
    m["market_volatility_20d"] = m["market_return_1d"].rolling(20).std()
    return m


def build_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["return_1d"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    for w in (5, 20, 50):
        out[f"ma_{w}_ratio"] = close / close.rolling(w).mean() - 1
    # 10 and 200-day averages come from the columns priceCollect.py already saved
    out["ma_10_ratio"] = close / df["MA_10"] - 1
    out["ma_200_ratio"] = close / df["MA_200"] - 1
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["rsi_14"] = rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = (ema12 - ema26) / close
    out["volume_change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    # Market context: NIFTY features aligned by date, plus the stock's move relative to NIFTY
    out = out.join(market)
    out["rel_return_1d"] = out["return_1d"] - out["market_return_1d"]
    out["rel_return_5d"] = out["return_5d"] - out["market_return_5d"]

    # Delivery: % of traded shares actually delivered, and today vs its own 20-day average
    if "Delivery_Pct" in df:
        out["delivery_pct"] = df["Delivery_Pct"]
        out["delivery_vs_20d"] = df["Delivery_Pct"] / df["Delivery_Pct"].rolling(20, min_periods=10).mean() - 1
    else:
        out["delivery_pct"] = np.nan
        out["delivery_vs_20d"] = np.nan

    # Target: 1 if tomorrow's close is higher than today's, else 0
    out["target"] = (close.shift(-1) > close).astype(int)
    out.loc[close.shift(-1).isna(), "target"] = np.nan
    return out


def main() -> None:
    nifty = pd.read_csv(DATA_DIR / f"{MARKET_SYMBOL}.csv", index_col=0, parse_dates=True)
    market = market_features(nifty)

    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name == OUT_FILE.name:
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) < 100:
            print(f"skip {path.name}: only {len(df)} rows")
            continue
        feats = build_features(df, market)
        # Delivery columns may stay empty; the models decide whether to require them
        feats = feats.dropna(subset=BASE_COLS)
        feats.insert(0, "symbol", path.stem)
        frames.append(feats)
        print(f"{path.stem}: {len(feats)} rows")

    result = pd.concat(frames)
    result.index.name = "Date"
    result.to_csv(OUT_FILE)
    print(f"\nSaved {len(result)} rows, {result['symbol'].nunique()} symbols -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
