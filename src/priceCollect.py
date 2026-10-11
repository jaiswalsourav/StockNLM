"""Download daily price data and market context for Indian stocks, built for a cross-sectional model.

Compared with the first version:
- 10 years of prices and flow history (more market regimes, bigger train/val/test blocks)
- Delivery % backfilled over the same span instead of only the last year
- Holidays / no-data days are remembered, so reruns only fetch genuinely new days
- Extra market context saved to data/market/context.csv: India VIX, USD/INR, Brent, S&P 500, Bank Nifty
- Open/High/Low are kept in every stock CSV (for range and gap features later)
"""
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_YEARS = 10
HISTORY_DAYS = 365 * HISTORY_YEARS + 10
DELIVERY_DAYS = HISTORY_DAYS  # NSE may not serve very old files; missing days are just marked as no-data
FII_DAYS = HISTORY_DAYS
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

# Market-wide series, saved together in data/market/context.csv (Close only).
# NOTE: sp500 and brent close AFTER the Indian market, so in priceFeature.py use them with shift(1).
CONTEXT_TICKERS = {
    "india_vix": "^INDIAVIX",
    "usdinr": "INR=X",
    "brent": "BZ=F",
    "sp500": "^GSPC",
    "banknifty": "^NSEBANK",
}

# NSE symbol -> file name, e.g. "TCS" -> "TCS_NSE" (delivery data exists for NSE stocks only)
NSE_SYMBOLS = {t[:-3]: name for name, t in TICKERS.items() if t.endswith(".NS")}

# Sector of each NSE stock; priceFeature.py averages returns within a sector
SECTORS = {
    "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "SBILIFE", "HDFCLIFE"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Auto": ["MARUTI", "TMPV", "M&M", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "Energy": ["RELIANCE", "ONGC", "BPCL"],
    "Utilities & Telecom": ["NTPC", "POWERGRID", "BHARTIARTL"],
    "Industrials": ["LT", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM"],
    "Consumer": ["TITAN", "ASIANPAINT", "TRENT"],
}
STOCK_SECTOR = {f"{s}_NSE": sector for sector, members in SECTORS.items() for s in members}

MARKET_DIR = DATA_DIR / "market"  # market-wide series; a subfolder so priceFeature.py's data/*.csv scan skips it
FII_DERIVATIVES_FILE = MARKET_DIR / "fii_derivatives.csv"
FII_DII_CASH_FILE = MARKET_DIR / "fii_dii_cash.csv"
FUNDAMENTALS_FILE = MARKET_DIR / "fundamentals.csv"
DII_FILE = MARKET_DIR / "dii_derivatives.csv"
CONTEXT_FILE = MARKET_DIR / "context.csv"
NO_DATA_FILE = MARKET_DIR / "no_data_days.json"  # holidays / unavailable days per source, so they aren't re-requested
PARTICIPANT_URL = "https://nsearchives.nseindia.com/content/nsccl/{}_{:%d%m%Y}.csv"
FII_DERIVATIVES_URL = "https://nsearchives.nseindia.com/content/fo/fii_stats_{:%d-%b-%Y}.xls"
FII_DII_CASH_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
RECENT_DAYS = 3  # a 404 this recent may just mean "not published yet", so don't record it as a holiday


def new_session() -> requests.Session:
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS, timeout=30)  # pick up cookies
    return session


def backfill(source: str, days: int, done: set, fetch, pause: float) -> list:
    """Call fetch(session, day) for every weekday not already saved or known to be a no-data day.

    fetch returns None for a holiday / unpublished file. Network errors are retried on the next run.
    Every 400 requests the session is refreshed, because NSE cookies expire.
    """
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    no_data = json.loads(NO_DATA_FILE.read_text()) if NO_DATA_FILE.exists() else {}
    skip = set(no_data.get(source, []))
    session, rows, today, calls = new_session(), [], date.today(), 0

    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5 or day in done or day.isoformat() in skip:
            continue
        calls += 1
        if calls % 400 == 0:
            session = new_session()
        try:
            row = fetch(session, day)
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            print(f"{source} {day}: skipped ({type(e).__name__})")
            continue
        if row is None:
            if day < today - timedelta(days=RECENT_DAYS):
                skip.add(day.isoformat())
            continue
        rows.append(row)
        time.sleep(pause)

    no_data[source] = sorted(skip)
    NO_DATA_FILE.write_text(json.dumps(no_data))
    return rows


def fetch_prices(ticker: str, period: str = f"{HISTORY_YEARS}y") -> pd.DataFrame:
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
    df["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    return df


# ---------------------------------------------------------------- delivery (per stock)

def fetch_delivery_day(session: requests.Session, day: date) -> pd.DataFrame | None:
    """One NSE bhavcopy file: delivery qty and % for every tracked NSE stock that day."""
    resp = session.get(BHAVCOPY_URL.format(day), headers=HEADERS, timeout=30)
    if resp.status_code == 404:  # weekend, market holiday, or file not in the archive
        return None
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[(df["SERIES"].str.strip() == "EQ") & df["SYMBOL"].isin(NSE_SYMBOLS)].copy()
    if df.empty:
        return None
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
    new = backfill("delivery", days, done, fetch_delivery_day, pause=0.3)
    print(f"delivery: {len(new)} new days")
    return pd.concat([saved] + new).drop_duplicates(subset=["name", "Date"])


# ---------------------------------------------------------------- FII / DII flow (market-wide)

def fetch_fii_derivatives_day(session: requests.Session, day: date) -> dict | None:
    """FII/FPI derivatives activity for one day: net buying (Rs crore) and open interest."""
    resp = session.get(FII_DERIVATIVES_URL.format(day), headers=HEADERS, timeout=30)
    if resp.status_code == 404:  # market holiday
        return None
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), header=None)
    raw[0] = raw[0].astype(str).str.strip()
    rows = raw.set_index(0)
    # columns: 1 buy contracts, 2 buy Rs cr, 3 sell contracts, 4 sell Rs cr, 5 OI contracts, 6 OI Rs cr
    out = {"Date": pd.Timestamp(day)}
    for label, key in (("INDEX FUTURES", "index_fut"), ("STOCK FUTURES", "stock_fut")):
        row = rows.loc[label]
        out[f"fii_{key}_net"] = float(row[2]) - float(row[4])
        out[f"fii_{key}_oi"] = float(row[6])
    return out


def fetch_dii_day(session: requests.Session, day: date) -> dict | None:
    """DII futures activity for one day from NSE's participant-wise files.

    vol file = contracts traded (flow), oi file = contracts held at the end of the day (position).
    Net = long - short, in contracts.
    """
    out = {"Date": pd.Timestamp(day)}
    for kind, tag in (("fao_participant_vol", "vol"), ("fao_participant_oi", "oi")):
        resp = session.get(PARTICIPANT_URL.format(kind, day), headers=HEADERS, timeout=30)
        if resp.status_code == 404:  # market holiday
            return None
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skiprows=1)
        df.columns = df.columns.str.strip()
        row = df.set_index(df["Client Type"].str.strip()).loc["DII"]
        out[f"dii_index_fut_net_{tag}"] = float(row["Future Index Long"]) - float(row["Future Index Short"])
        out[f"dii_stock_fut_net_{tag}"] = float(row["Future Stock Long"]) - float(row["Future Stock Short"])
    return out


def collect_daily_table(source: str, path: Path, fetch, days: int = FII_DAYS) -> pd.DataFrame:
    """Shared loop for the market-wide daily tables; later runs only fetch days not saved yet."""
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(path, parse_dates=["Date"]) if path.exists() else None
    done = set(saved["Date"].dt.date) if saved is not None else set()
    rows = backfill(source, days, done, fetch, pause=0.15)
    print(f"{source}: {len(rows)} new days")
    frames = ([saved] if saved is not None else []) + ([pd.DataFrame(rows)] if rows else [])
    result = pd.concat(frames).drop_duplicates(subset="Date").sort_values("Date")
    result.to_csv(path, index=False)
    return result


def collect_fii_dii_cash() -> None:
    """Cash-market FII/DII buy/sell. NSE only serves the latest day, so history builds up run by run."""
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    session = new_session()
    resp = session.get(
        FII_DII_CASH_URL, headers={**HEADERS, "Referer": "https://www.nseindia.com/reports/fii-dii"}, timeout=30
    )
    resp.raise_for_status()
    new = pd.DataFrame(resp.json())
    new["Date"] = pd.to_datetime(new["date"], format="%d-%b-%Y")
    new["category"] = new["category"].map({"FII/FPI": "fii", "DII": "dii"})
    for col in ("buyValue", "sellValue", "netValue"):
        new[col] = pd.to_numeric(new[col], errors="coerce")
    wide = new.pivot(index="Date", columns="category", values=["buyValue", "sellValue", "netValue"])
    wide.columns = [f"{cat}_{val.replace('Value', '')}" for val, cat in wide.columns]
    wide = wide.reset_index()
    if FII_DII_CASH_FILE.exists():
        wide = pd.concat([pd.read_csv(FII_DII_CASH_FILE, parse_dates=["Date"]), wide])
    wide = wide.drop_duplicates(subset="Date", keep="last").sort_values("Date")
    wide.to_csv(FII_DII_CASH_FILE, index=False)
    print(f"fii/dii cash: {len(wide)} day(s) saved, latest {wide['Date'].max().date()}")


# ---------------------------------------------------------------- other market context

def collect_market_context() -> pd.DataFrame:
    """Daily Close of India VIX, USD/INR, Brent, S&P 500 and Bank Nifty in one table (data/market/context.csv)."""
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, ticker in CONTEXT_TICKERS.items():
        try:
            cols[name] = fetch_prices(ticker)["Close"]
        except Exception as e:  # yfinance raises assorted errors on throttling/network trouble
            print(f"context {name}: skipped ({type(e).__name__})")
    if not cols:
        print("context: nothing fetched")
        return pd.DataFrame()
    ctx = pd.DataFrame(cols).sort_index()
    ctx.index.name = "Date"
    ctx.to_csv(CONTEXT_FILE)
    print(f"context: {len(ctx)} rows, columns {list(ctx.columns)} -> {CONTEXT_FILE.name}")
    return ctx


def collect_fundamentals() -> pd.DataFrame:
    """Today's P/E, industry P/E and debt/equity per NSE stock, appended to data/market/fundamentals.csv.

    Yahoo only reports current values (no history), so this file grows by one snapshot per run.
    Not usable as a model feature until months of snapshots exist (joining today's value to past rows leaks the future).
    """
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ticker in TICKERS.items():
        if name not in STOCK_SECTOR:
            continue
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            print(f"fundamentals {name}: skipped ({type(e).__name__})")
            continue
        de = info.get("debtToEquity")
        rows.append({
            "Date": pd.Timestamp(date.today()),
            "name": name,
            "sector": STOCK_SECTOR[name],
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "debt_to_equity": de / 100 if de is not None else None,  # Yahoo reports it as a percentage
        })
    snap = pd.DataFrame(rows)
    if snap.empty:
        print("fundamentals: nothing fetched")
        return snap
    snap[["pe", "forward_pe", "debt_to_equity"]] = snap[["pe", "forward_pe", "debt_to_equity"]].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_pe = snap["pe"].where(snap["pe"] > 0)  # negative P/E (losses) would distort the median

    def peer_median(row):
        peers = snap.index[(snap["sector"] == row["sector"]) & (snap["name"] != row["name"])]
        return valid_pe[peers].median()

    snap["industry_pe"] = snap.apply(peer_median, axis=1)
    snap["pe_vs_industry"] = snap["pe"] / snap["industry_pe"]

    out = snap
    if FUNDAMENTALS_FILE.exists():
        out = pd.concat([pd.read_csv(FUNDAMENTALS_FILE, parse_dates=["Date"]), snap])
    out = out.drop_duplicates(subset=["Date", "name"], keep="last").sort_values(["name", "Date"])
    out.to_csv(FUNDAMENTALS_FILE, index=False)
    print(f"fundamentals: {len(snap)} stocks saved for {date.today()} ({out['Date'].nunique()} snapshot day(s) total)")
    return snap


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    collect_daily_table("fii_derivatives", FII_DERIVATIVES_FILE, fetch_fii_derivatives_day)
    collect_daily_table("dii", DII_FILE, fetch_dii_day)
    try:
        collect_fii_dii_cash()
    except (requests.RequestException, ValueError) as e:
        print(f"fii/dii cash: skipped ({e})")
    collect_market_context()
    collect_fundamentals()
    delivery = collect_delivery()

    for name, ticker in TICKERS.items():
        try:
            df = add_indicators(fetch_prices(ticker))
        except Exception as e:
            print(f"{name} ({ticker}): FAILED ({type(e).__name__}), keeping old file")
            continue
        if df.empty:
            print(f"{name} ({ticker}): no data, keeping old file")
            continue
        if name in NSE_SYMBOLS.values():
            d = delivery[delivery["name"] == name].set_index("Date")[["Delivery_Qty", "Delivery_Pct"]]
            df = df.join(d)
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out)
        first = df.index.min().date()
        print(f"{name} ({ticker}): {len(df)} rows from {first} -> {out.name}")


if __name__ == "__main__":
    main()