"""Turn raw price CSVs in data/ into model features and targets.

Targets written to features.csv:
- target            1 if tomorrow's close is higher than today's (original next-day up/down)
- target_rel_1d     1 if the stock beats the median stock tomorrow (market direction removed)
- target_rel_5d     same over 5 days (rows overlap, so leave a >= 5 day gap between train/val/test)
- excess_return_1d/5d   continuous version of the above, for daily-IC scoring or regression
- target_vol_5d     realized volatility over the next 5 days
- fwd_return_1d/5d  raw forward returns (not features: never feed these to a model)
"""
import numpy as np
import pandas as pd

from priceCollect import CONTEXT_FILE, DATA_DIR, DII_FILE, FII_DERIVATIVES_FILE, STOCK_SECTOR

OUT_FILE = DATA_DIR / "features.csv"
MIN_STOCKS_PER_DAY = 30  # a daily cross-section smaller than this gets no relative target


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
# Other stocks in the same sector (leave-one-out average) and the stock's move relative to them
SECTOR_COLS = ["sector_return_1d", "sector_return_5d", "rel_sector_return_1d", "rel_sector_return_5d"]
# FII/FPI derivatives flow (Rs crore): daily net buying, 5-day sum, and open interest
FII_COLS = ["fii_index_fut_net", "fii_index_fut_net_5d", "fii_index_fut_oi", "fii_stock_fut_net", "fii_stock_fut_net_5d"]
# DII futures positioning (contracts): traded flow, 5-day flow and open-interest position
DII_COLS = [
    "dii_index_fut_net_vol", "dii_index_fut_net_vol_5d", "dii_index_fut_net_oi",
    "dii_stock_fut_net_vol", "dii_stock_fut_net_vol_5d", "dii_stock_fut_net_oi",
]
# India VIX, Bank Nifty, and global series (USD/INR, Brent, S&P 500; these are lagged, see context_features)
CONTEXT_COLS = [
    "india_vix", "vix_chg_1d", "vix_vs_20d", "banknifty_return_1d", "banknifty_return_5d",
    "usdinr_return_1d", "usdinr_return_5d", "brent_return_1d", "brent_return_5d",
    "sp500_return_1d", "sp500_return_5d",
]
# NSE delivery data only exists for NSE stocks, for as far back as priceCollect.py managed to backfill
DELIVERY_COLS = ["delivery_pct", "delivery_vs_20d"]

CORE_COLS = PRICE_COLS + MARKET_COLS + SECTOR_COLS + FII_COLS  # rows missing any of these are dropped
BASE_COLS = CORE_COLS + DII_COLS + CONTEXT_COLS  # new groups stay NaN when their data is missing
FEATURE_COLS = BASE_COLS + DELIVERY_COLS
TARGET_COLS = ["target", "target_rel_1d", "target_rel_5d", "excess_return_1d", "excess_return_5d", "target_vol_5d"]
LEAKY_COLS = ["fwd_return_1d", "fwd_return_5d"] + TARGET_COLS  # never use these as model inputs


def market_features(nifty: pd.DataFrame) -> pd.DataFrame:
    """NIFTY 50 context, computed once and joined onto every stock by date."""
    close = nifty["Close"]
    m = pd.DataFrame(index=nifty.index)
    m["market_return_1d"] = close.pct_change()
    m["market_return_5d"] = close.pct_change(5)
    m["market_ma_20_ratio"] = close / close.rolling(20).mean() - 1
    m["market_volatility_20d"] = m["market_return_1d"].rolling(20).std()
    return m


def fii_features() -> pd.DataFrame:
    """FII derivatives flow by date, collected by priceCollect.py."""
    fii = pd.read_csv(FII_DERIVATIVES_FILE, index_col=0, parse_dates=True).sort_index()
    fii["fii_index_fut_net_5d"] = fii["fii_index_fut_net"].rolling(5).sum()
    fii["fii_stock_fut_net_5d"] = fii["fii_stock_fut_net"].rolling(5).sum()
    return fii[FII_COLS]


def dii_features() -> pd.DataFrame:
    """DII futures flow and positioning by date; all NaN if the file has not been collected yet."""
    if not DII_FILE.exists():
        print("warning: dii_derivatives.csv missing, DII features will be empty (run priceCollect.py)")
        return pd.DataFrame(columns=DII_COLS, dtype=float)
    dii = pd.read_csv(DII_FILE, index_col=0, parse_dates=True).sort_index()
    dii["dii_index_fut_net_vol_5d"] = dii["dii_index_fut_net_vol"].rolling(5).sum()
    dii["dii_stock_fut_net_vol_5d"] = dii["dii_stock_fut_net_vol"].rolling(5).sum()
    return dii[DII_COLS]


def last_known_before(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """For each date in index, the latest value of series from a STRICTLY earlier date.

    Used for series that close after the Indian market (S&P 500, Brent, USD/INR on Yahoo): when the
    Indian market closes on day t, the latest value you could actually have is from before day t.
    """
    s = series.dropna()
    left = pd.DataFrame({"Date": pd.DatetimeIndex(index).astype("datetime64[ns]")}).sort_values("Date")
    right = pd.DataFrame({"Date": pd.DatetimeIndex(s.index).astype("datetime64[ns]"), "v": s.values}).sort_values("Date")
    merged = pd.merge_asof(left, right, on="Date", allow_exact_matches=False)
    return pd.Series(merged["v"].to_numpy(), index=pd.DatetimeIndex(left["Date"]))  # index must be sorted (NIFTY's is)


def context_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """India VIX, Bank Nifty, and lagged global series, aligned to the NIFTY trading calendar."""
    out = pd.DataFrame(index=index, columns=CONTEXT_COLS, dtype=float)
    if not CONTEXT_FILE.exists():
        print("warning: context.csv missing, context features will be empty (run priceCollect.py)")
        return out
    ctx = pd.read_csv(CONTEXT_FILE, index_col=0, parse_dates=True).sort_index()

    # Same-day series (they close at or before the Indian close)
    if "india_vix" in ctx:
        vix = ctx["india_vix"].reindex(index).ffill(limit=3)
        out["india_vix"] = vix
        out["vix_chg_1d"] = vix.pct_change()
        out["vix_vs_20d"] = vix / vix.rolling(20, min_periods=10).mean() - 1
    if "banknifty" in ctx:
        bn = ctx["banknifty"]
        out["banknifty_return_1d"] = bn.pct_change().reindex(index)
        out["banknifty_return_5d"] = bn.pct_change(5).reindex(index)

    # Series that close after India: use the last value from before each Indian trading day
    for name in ("usdinr", "brent", "sp500"):
        if name in ctx:
            s = ctx[name].dropna()
            out[f"{name}_return_1d"] = last_known_before(s.pct_change(), index)
            out[f"{name}_return_5d"] = last_known_before(s.pct_change(5), index)
    return out


def sector_features(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """For each NSE stock: average 1d/5d return of the OTHER stocks in its sector."""
    close = pd.DataFrame({name: df["Close"] for name, df in prices.items() if name in STOCK_SECTOR})
    ret1, ret5 = close.pct_change(), close.pct_change(5)
    out = {}
    for name in close.columns:
        peers = [n for n in close.columns if STOCK_SECTOR[n] == STOCK_SECTOR[name] and n != name]
        out[name] = pd.DataFrame({
            "sector_return_1d": ret1[peers].mean(axis=1),
            "sector_return_5d": ret5[peers].mean(axis=1),
        })
    return out


def build_features(
    df: pd.DataFrame,
    market: pd.DataFrame,
    fii: pd.DataFrame,
    dii: pd.DataFrame,
    context: pd.DataFrame,
    sector: pd.DataFrame | None,
) -> pd.DataFrame:
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

    # Sector context (NSE stocks only) and institutional flow / global context, all joined by date
    if sector is not None:
        out = out.join(sector)
    else:
        out["sector_return_1d"] = out["sector_return_5d"] = np.nan
    out["rel_sector_return_1d"] = out["return_1d"] - out["sector_return_1d"]
    out["rel_sector_return_5d"] = out["return_5d"] - out["sector_return_5d"]
    out = out.join(fii).join(dii).join(context)

    # Delivery: % of traded shares actually delivered, and today vs its own 20-day average
    if "Delivery_Pct" in df:
        out["delivery_pct"] = df["Delivery_Pct"]
        out["delivery_vs_20d"] = df["Delivery_Pct"] / df["Delivery_Pct"].rolling(20, min_periods=10).mean() - 1
    else:
        out["delivery_pct"] = np.nan
        out["delivery_vs_20d"] = np.nan

    # Forward-looking columns: targets only, never model inputs
    out["fwd_return_1d"] = close.shift(-1) / close - 1
    out["fwd_return_5d"] = close.shift(-5) / close - 1
    out["target"] = (out["fwd_return_1d"] > 0).astype(float).where(out["fwd_return_1d"].notna())
    out["target_vol_5d"] = out["return_1d"].rolling(5).std().shift(-5)  # std of days t+1 .. t+5
    return out


def add_relative_targets(result: pd.DataFrame) -> pd.DataFrame:
    """Beat-the-market targets: each stock's forward return minus the median stock's that same day."""
    for h in ("1d", "5d"):
        fwd = result[f"fwd_return_{h}"]
        by_day = fwd.groupby(level=0)
        enough = by_day.transform("count") >= MIN_STOCKS_PER_DAY
        excess = (fwd - by_day.transform("median")).where(enough)
        result[f"excess_return_{h}"] = excess
        result[f"target_rel_{h}"] = (excess > 0).astype(float).where(excess.notna())
    return result


def main() -> None:
    nifty = pd.read_csv(DATA_DIR / f"{MARKET_SYMBOL}.csv", index_col=0, parse_dates=True)
    market = market_features(nifty)

    if not FII_DERIVATIVES_FILE.exists():
        raise SystemExit("FII data missing: run priceCollect.py first.")
    fii = fii_features()
    dii = dii_features()
    context = context_features(nifty.index)

    prices = {
        path.stem: pd.read_csv(path, index_col=0, parse_dates=True)
        for path in sorted(DATA_DIR.glob("*.csv"))
        if path.name != OUT_FILE.name
    }
    sector = sector_features(prices)

    frames = []
    for name, df in prices.items():
        # Only the NSE stocks that have a sector; indices and TCS_BSE would distort the daily medians
        if name not in STOCK_SECTOR:
            continue
        if len(df) < 100:
            print(f"skip {name}: only {len(df)} rows")
            continue
        feats = build_features(df, market, fii, dii, context, sector.get(name))
        feats.insert(0, "symbol", name)
        frames.append(feats)

    result = pd.concat(frames)
    result.index.name = "Date"
    result = add_relative_targets(result)  # needs every stock's row for the day, so before any dropna
    # Only the core groups are required. DII / context / delivery may be missing for early dates;
    # the models decide whether to require them.
    result = result.dropna(subset=CORE_COLS).sort_index()
    result.to_csv(OUT_FILE)

    print(f"Saved {len(result)} rows, {result['symbol'].nunique()} symbols -> {OUT_FILE.name}")
    print(f"Dates: {result.index.min().date()} to {result.index.max().date()}")
    print("\nFirst date with every column of each group present:")
    for label, cols in (("core", CORE_COLS), ("dii", DII_COLS), ("context", CONTEXT_COLS), ("delivery", DELIVERY_COLS)):
        full = result[cols].dropna()
        print(f"  {label:9s} {full.index.min().date() if len(full) else 'none'}")
    print("\nTarget base rates (share of 1s):")
    for col in ("target", "target_rel_1d", "target_rel_5d"):
        print(f"  {col:14s} {result[col].mean():.3f}")


if __name__ == "__main__":
    main()