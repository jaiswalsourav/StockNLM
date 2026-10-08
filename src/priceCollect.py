"""Download daily price data for Indian stocks (NSE: .NS, BSE: .BO) and indices."""
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DELIVERY_DAYS = 365  # how far back to fetch NSE delivery data
BHAVCOPY_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{:%d%m%Y}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Yahoo Finance tickers: NSE -> ".NS", BSE -> ".BO"
TICKERS = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    # NSE stocks
    "RELIANCE_NSE": "RELIANCE.NS",
    "TCS_NSE": "TCS.NS",
    "INFY_NSE": "INFY.NS",
    "HDFCBANK_NSE": "HDFCBANK.NS",
    "ICICIBANK_NSE": "ICICIBANK.NS",
    "SBIN_NSE": "SBIN.NS",
    "BHARTIARTL_NSE": "BHARTIARTL.NS",
    "ITC_NSE": "ITC.NS",
    "LT_NSE": "LT.NS",
    "HINDUNILVR_NSE": "HINDUNILVR.NS",
    "AXISBANK_NSE": "AXISBANK.NS",
    "KOTAKBANK_NSE": "KOTAKBANK.NS",
    "MARUTI_NSE": "MARUTI.NS",
    "SUNPHARMA_NSE": "SUNPHARMA.NS",
    "TMPV_NSE": "TMPV.NS",
    "TITAN_NSE": "TITAN.NS",
    "HCLTECH_NSE": "HCLTECH.NS",
    "POWERGRID_NSE": "POWERGRID.NS",
    "TATASTEEL_NSE": "TATASTEEL.NS",
    "WIPRO_NSE": "WIPRO.NS",
    "ONGC_NSE": "ONGC.NS",
    "NTPC_NSE": "NTPC.NS",
    "ADANIENT_NSE": "ADANIENT.NS",
    "BAJFINANCE_NSE": "BAJFINANCE.NS",
    "ASIANPAINT_NSE": "ASIANPAINT.NS",
    "ULTRACEMCO_NSE": "ULTRACEMCO.NS",
    "NESTLEIND_NSE": "NESTLEIND.NS",
    "M&M_NSE": "M&M.NS",
    "BAJAJFINSV_NSE": "BAJAJFINSV.NS",
    "JSWSTEEL_NSE": "JSWSTEEL.NS",
    "COALINDIA_NSE": "COALINDIA.NS",
    "TECHM_NSE": "TECHM.NS",
    "GRASIM_NSE": "GRASIM.NS",
    "INDUSINDBK_NSE": "INDUSINDBK.NS",
    "CIPLA_NSE": "CIPLA.NS",
    "DRREDDY_NSE": "DRREDDY.NS",
    "EICHERMOT_NSE": "EICHERMOT.NS",
    "BPCL_NSE": "BPCL.NS",
    "HINDALCO_NSE": "HINDALCO.NS",
    "DIVISLAB_NSE": "DIVISLAB.NS",
    "BRITANNIA_NSE": "BRITANNIA.NS",
    "APOLLOHOSP_NSE": "APOLLOHOSP.NS",
    "HEROMOTOCO_NSE": "HEROMOTOCO.NS",
    "TATACONSUM_NSE": "TATACONSUM.NS",
    "ADANIPORTS_NSE": "ADANIPORTS.NS",
    "SBILIFE_NSE": "SBILIFE.NS",
    "HDFCLIFE_NSE": "HDFCLIFE.NS",
    "BAJAJ-AUTO_NSE": "BAJAJ-AUTO.NS",
    "SHRIRAMFIN_NSE": "SHRIRAMFIN.NS",
    "TRENT_NSE": "TRENT.NS",
    # BSE stocks
    "TCS_BSE": "TCS.BO",
}

# NSE symbol -> file name, e.g. "TCS" -> "TCS_NSE" (delivery data exists for NSE stocks only)
NSE_SYMBOLS = {t[:-3]: name for name, t in TICKERS.items() if t.endswith(".NS")}


def fetch_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages (10/50/200 day) and 14-day RSI, computed from Close."""
    close = df["Close"]
    for w in (10, 50, 200):
        df[f"MA_{w}"] = close.rolling(w).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + gain / loss)
    return df


def fetch_delivery_day(session: requests.Session, day: date) -> pd.DataFrame | None:
    """One NSE bhavcopy file: delivery qty and % for every tracked NSE stock that day."""
    resp = session.get(BHAVCOPY_URL.format(day), headers=HEADERS, timeout=30)
    if resp.status_code == 404:  # weekend or market holiday
        return None
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[(df["SERIES"].str.strip() == "EQ") & df["SYMBOL"].isin(NSE_SYMBOLS)].copy()
    df["name"] = df["SYMBOL"].map(NSE_SYMBOLS)
    df["Date"] = pd.Timestamp(day)
    df["Delivery_Qty"] = pd.to_numeric(df["DELIV_QTY"], errors="coerce")
    df["Delivery_Pct"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
    return df[["name", "Date", "Delivery_Qty", "Delivery_Pct"]]


def load_saved_delivery() -> pd.DataFrame:
    """Delivery values already stored in the stock CSVs, so each run only fetches new days."""
    frames = []
    for name in NSE_SYMBOLS.values():
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            old = pd.read_csv(path, parse_dates=["Date"])
            if "Delivery_Pct" in old:
                old = old.dropna(subset=["Delivery_Pct"])
                frames.append(old.assign(name=name)[["name", "Date", "Delivery_Qty", "Delivery_Pct"]])
    if frames:
        return pd.concat(frames)
    return pd.DataFrame({
        "name": pd.Series(dtype="object"),
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Delivery_Qty": pd.Series(dtype="float64"),
        "Delivery_Pct": pd.Series(dtype="float64"),
    })


def collect_delivery(days: int = DELIVERY_DAYS) -> pd.DataFrame:
    saved = load_saved_delivery()
    done = set(saved["Date"].dt.date)
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS, timeout=30)  # pick up cookies

    frames, today = [saved], date.today()
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5 or day in done:
            continue
        try:
            df = fetch_delivery_day(session, day)
        except requests.RequestException as e:
            print(f"delivery {day}: failed ({e})")
            continue
        if df is not None and len(df):
            frames.append(df)
            print(f"delivery {day}: {len(df)} stocks")
        time.sleep(0.3)
    return pd.concat(frames).drop_duplicates(subset=["name", "Date"])


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    delivery = collect_delivery()
    for name, ticker in TICKERS.items():
        df = add_indicators(fetch_prices(ticker))
        if name in NSE_SYMBOLS.values():
            d = delivery[delivery["name"] == name].set_index("Date")[["Delivery_Qty", "Delivery_Pct"]]
            df = df.join(d)
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out)
        print(f"{name} ({ticker}): {len(df)} rows -> {out.name}")


if __name__ == "__main__":
    main()
