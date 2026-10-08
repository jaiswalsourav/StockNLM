"""Download daily price data for Indian stocks (NSE: .NS, BSE: .BO) and indices."""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
    # BSE stocks
    "TCS_BSE": "TCS.BO",
}


def fetch_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, ticker in TICKERS.items():
        df = fetch_prices(ticker)
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out)
        print(f"{name} ({ticker}): {len(df)} rows -> {out.name}")


if __name__ == "__main__":
    main()
