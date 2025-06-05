import feedparser
import os
import json
from datetime import datetime
from hashlib import sha256

NEWS_DIR = "../oraculum/src/app/data/news"

RSS_FEEDS = {
    "TradingEconomics": "https://tradingeconomics.com/rss/news",
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "Reuters": "https://www.reutersagency.com/feed/?best-topics=business",
    "OilPrice": "https://oilprice.com/rss/main",
    "DailyFX": "https://www.dailyfx.com/feeds/all",
    "CFTC": "https://www.cftc.gov/PressRoom/PressReleases/rss.xml",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Bloomberg": "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "MarketWatch": "https://www.marketwatch.com/rss/topstories",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "FT_UK": "https://www.ft.com/?format=rss",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",
    "Bank of Canada": "https://www.bankofcanada.ca/rss/press-releases/",
    "Financial Post": "https://financialpost.com/feed/",
    "RBA": "https://www.rba.gov.au/rss/rss-press-releases.xml",
    "ABC_AU_Business": "https://www.abc.net.au/news/feed/51120/rss.xml",
    "IMF": "https://www.imf.org/en/News/rss",
    "World Bank": "https://www.worldbank.org/en/news/all?format=rss",
    "OECD": "https://www.oecd.org/newsroom/publicationsdocuments/rss/",
}


def normalize_date(entry):
    try:
        return datetime(*entry.published_parsed[:6]).isoformat() + "Z"
    except Exception:
        return datetime.utcnow().isoformat() + "Z"


def load_existing_news(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def hash_entry(entry):
    key = (entry.get("title", "") + entry.get("link", "")).encode()
    return sha256(key).hexdigest()


def run():
    os.makedirs(NEWS_DIR, exist_ok=True)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    filepath = os.path.join(NEWS_DIR, f"{today_str}.json")

    print("Coletando notícias econômicas...")
    news_set = load_existing_news(filepath)
    existing_hashes = {hash_entry(e) for e in news_set}

    for source, url in RSS_FEEDS.items():
        print(f"🔍 Lendo {source}...")
        feed = feedparser.parse(url)
        for entry in feed.entries:
            item = {
                "title": entry.get("title", ""),
                "published": normalize_date(entry),
                "source": source,
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
            }
            h = hash_entry(item)
            if h not in existing_hashes:
                news_set.append(item)
                existing_hashes.add(h)

    with open(filepath, "w") as f:
        json.dump(news_set, f, indent=2, ensure_ascii=False)
    print(f"✅ {len(news_set)} notícias salvas em {filepath}")


if __name__ == "__main__":
    run()
