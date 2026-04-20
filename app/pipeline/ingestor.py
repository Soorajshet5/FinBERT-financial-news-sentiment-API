import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")


async def fetch_alpha_vantage_news(ticker: str = "NSE", limit: int = 20) -> List[Dict]:
    """Fetch news from Alpha Vantage for a given ticker."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            feed = data.get("feed", [])
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published_at": item.get("time_published", ""),
                    "ticker": ticker,
                }
                for item in feed
            ]
    except Exception as e:
        logger.error(f"Alpha Vantage fetch error: {e}")
        return []


async def fetch_newsapi(query: str = "NSE India stocks", limit: int = 20) -> List[Dict]:
    """Fetch news from NewsAPI."""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "summary": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "ticker": query,
                }
                for a in articles
            ]
    except Exception as e:
        logger.error(f"NewsAPI fetch error: {e}")
        return []


async def fetch_news(ticker: str = "RELIANCE", limit: int = 20) -> List[Dict]:
    """Unified news fetcher — tries Alpha Vantage first, falls back to NewsAPI."""
    results = await fetch_alpha_vantage_news(ticker, limit)
    if not results:
        results = await fetch_newsapi(f"{ticker} India stock market", limit)
    return results