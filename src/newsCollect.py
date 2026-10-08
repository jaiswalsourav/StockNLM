"""Collect recent news headlines for each tracked stock into news/ using free RSS feeds."""
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

NEWS_DIR = Path(__file__).resolve().parent.parent / "news"

# Symbol name (matches data/ file names) -> search phrase for Google News
COMPANIES = {
    "RELIANCE_NSE": "Reliance Industries",
    "TCS_NSE": "TCS Tata Consultancy Services",
    "INFY_NSE": "Infosys",
    "HDFCBANK_NSE": "HDFC Bank",
    "ICICIBANK_NSE": "ICICI Bank",
    "SBIN_NSE": "State Bank of India SBI",
    "BHARTIARTL_NSE": "Bharti Airtel",
    "ITC_NSE": "ITC Limited",
    "LT_NSE": "Larsen & Toubro",
    "HINDUNILVR_NSE": "Hindustan Unilever",
    "AXISBANK_NSE": "Axis Bank",
    "KOTAKBANK_NSE": "Kotak Mahindra Bank",
    "MARUTI_NSE": "Maruti Suzuki",
    "SUNPHARMA_NSE": "Sun Pharma",
    "TMPV_NSE": "Tata Motors",
    "TITAN_NSE": "Titan Company",
    "HCLTECH_NSE": "HCL Technologies",
    "POWERGRID_NSE": "Power Grid Corporation",
    "TATASTEEL_NSE": "Tata Steel",
    "WIPRO_NSE": "Wipro",
    "ONGC_NSE": "ONGC",
    "NTPC_NSE": "NTPC",
    "ADANIENT_NSE": "Adani Enterprises",
    "BAJFINANCE_NSE": "Bajaj Finance",
    "ASIANPAINT_NSE": "Asian Paints",
    "NIFTY50": "Nifty 50",
    "SENSEX": "Sensex",
}

# General market feeds (not tied to one stock)
MARKET_FEEDS = {
    "economictimes_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_latest": "https://www.moneycontrol.com/rss/latestnews.xml",
}


def google_news_url(query: str) -> str:
    return (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(query + ' stock when:30d')}&hl=en-IN&gl=IN&ceid=IN:en"
    )


def parse_feed(url: str, symbol: str) -> pd.DataFrame:
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        published = pd.to_datetime(e.get("published"), errors="coerce", utc=True)
        rows.append(
            {
                "symbol": symbol,
                "published": published,
                "title": e.get("title", "").strip(),
                "source": (e.get("source") or {}).get("title", ""),
                "link": e.get("link", ""),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "published", "title", "source", "link"])


def merge_with_existing(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Append to previous runs so history builds up over time."""
    if path.exists():
        old = pd.read_csv(path, parse_dates=["published"])
        new = pd.concat([old, new])
    return new.drop_duplicates(subset="link").sort_values("published", ascending=False)


def main() -> None:
    NEWS_DIR.mkdir(exist_ok=True)

    for symbol, query in COMPANIES.items():
        df = parse_feed(google_news_url(query), symbol)
        path = NEWS_DIR / f"{symbol}.csv"
        merged = merge_with_existing(path, df)
        merged.to_csv(path, index=False)
        print(f"{symbol}: {len(df)} fetched, {len(merged)} total")

    for name, url in MARKET_FEEDS.items():
        df = parse_feed(url, "MARKET")
        path = NEWS_DIR / f"market_{name}.csv"
        merged = merge_with_existing(path, df)
        merged.to_csv(path, index=False)
        print(f"market_{name}: {len(df)} fetched, {len(merged)} total")

    print(f"\nDone at {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
