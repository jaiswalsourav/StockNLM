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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["return_1d"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    for w in (5, 20, 50):
        out[f"ma_{w}_ratio"] = close / close.rolling(w).mean() - 1
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["rsi_14"] = rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = (ema12 - ema26) / close
    out["volume_change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    # Target: 1 if tomorrow's close is higher than today's, else 0
    out["target"] = (close.shift(-1) > close).astype(int)
    out.loc[close.shift(-1).isna(), "target"] = np.nan
    return out


def main() -> None:
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name == OUT_FILE.name:
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) < 100:
            print(f"skip {path.name}: only {len(df)} rows")
            continue
        feats = build_features(df)
        feats = feats.dropna(subset=[c for c in feats.columns if c != "target"])
        feats.insert(0, "symbol", path.stem)
        frames.append(feats)
        print(f"{path.stem}: {len(feats)} rows")

    result = pd.concat(frames)
    result.index.name = "Date"
    result.to_csv(OUT_FILE)
    print(f"\nSaved {len(result)} rows, {result['symbol'].nunique()} symbols -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
