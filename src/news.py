# src/news.py
import pandas as pd
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def _to_date_safe(x) -> pd.Timestamp:
    """Parse RSS timestamps robustly and normalize to date (no tz)."""
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_convert(None).normalize()
    except Exception:
        return pd.NaT


def _yahoo_rss_url(symbol: str, region: str, lang: str) -> str:
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={region}&lang={lang}"


def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    """Fetch RSS content with a user-agent to reduce empty/blocked responses."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception:
        # fall back to direct parse (sometimes works even if requests fails)
        return feedparser.parse(url)


def fetch_rss_items_for_symbol(symbol: str, url: str) -> pd.DataFrame:
    d = _fetch_feed(url)
    rows = []
    for e in getattr(d, "entries", []):
        title = getattr(e, "title", "") or ""
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        rows.append({"Symbol": symbol, "published": published, "title": title})
    return pd.DataFrame(rows)


def fetch_rss_items_global(urls: list[str]) -> pd.DataFrame:
    rows = []
    for url in urls:
        d = _fetch_feed(url)
        for e in getattr(d, "entries", []):
            title = getattr(e, "title", "") or ""
            published = getattr(e, "published", None) or getattr(e, "updated", None)
            rows.append({"Symbol": "GLOBAL", "published": published, "title": title})
    return pd.DataFrame(rows)


def daily_news_signals(symbols: list[str], news_cfg: dict) -> pd.DataFrame:
    """
    Build per-day, per-symbol news features from RSS headlines.

    Returns DataFrame with columns:
      Date, Symbol, news_count, news_sent_mean
    Optionally includes Symbol='GLOBAL' if global_rss_feeds provided.
    """
    items = []

    # Per-symbol Yahoo RSS
    if news_cfg.get("per_symbol_yahoo", True):
        region = news_cfg.get("yahoo_region", "US")
        lang = news_cfg.get("yahoo_lang", "en-US")
        for sym in symbols:
            url = _yahoo_rss_url(sym, region, lang)
            df = fetch_rss_items_for_symbol(sym, url)
            if not df.empty:
                items.append(df)

    # Optional global feeds (market-wide)
    global_urls = news_cfg.get("global_rss_feeds", []) or []
    if global_urls:
        gdf = fetch_rss_items_global(global_urls)
        if not gdf.empty:
            items.append(gdf)

    if not items:
        return pd.DataFrame(columns=["Date", "Symbol", "news_count", "news_sent_mean"])

    items = pd.concat(items, ignore_index=True)

    # Parse dates
    items["Date"] = items["published"].apply(_to_date_safe)
    items = items.dropna(subset=["Date"]).copy()

    # Sentiment
    items["compound"] = items["title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

    # Aggregate per (Date, Symbol)
    daily = (
        items.groupby(["Date", "Symbol"])
        .agg(
            news_count=("compound", "size"),
            news_sent_mean=("compound", "mean"),
        )
        .reset_index()
    )

    daily["news_sent_mean"] = daily["news_sent_mean"].astype(float)
    daily["news_count"] = daily["news_count"].astype(float)
    return daily