from fastapi import APIRouter, Request, HTTPException
from models.schemas import (
    AnalyzeRequest, BatchAnalyzeRequest, SentimentResult,
    NewsAnalysisRequest, NewsAnalysisResponse, SearchRequest,
)
from pipeline.ingestor import fetch_news
from pipeline.vectorstore import NewsVectorStore
from collections import Counter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
vector_store = NewsVectorStore()


@router.post("/analyze", response_model=SentimentResult)
async def analyze_text(req: AnalyzeRequest, request: Request):
    """Analyze sentiment of a single text."""
    model = request.app.state.model
    result = model.predict_one(req.text)
    return result


@router.post("/analyze/batch")
async def analyze_batch(req: BatchAnalyzeRequest, request: Request):
    """Batch sentiment analysis for multiple texts."""
    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    model = request.app.state.model
    results = model.predict(req.texts)
    return {"count": len(results), "results": results}


@router.post("/news/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(req: NewsAnalysisRequest, request: Request):
    """Fetch live news for a ticker and run sentiment analysis."""
    model = request.app.state.model
    articles = await fetch_news(req.ticker, req.limit)

    if not articles:
        raise HTTPException(status_code=404, detail=f"No news found for ticker: {req.ticker}")

    texts = [f"{a['title']} {a['summary']}" for a in articles]
    sentiments = model.predict(texts)

    # Store in vector DB
    vector_store.add_articles(articles, sentiments)

    # Build response
    labels = [s["label"] for s in sentiments]
    summary = dict(Counter(labels))

    enriched = []
    for article, sentiment in zip(articles, sentiments):
        enriched.append({
            "title": article["title"],
            "source": article.get("source", ""),
            "label": sentiment["label"],
            "confidence": sentiment["confidence"],
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "scores": sentiment["scores"],
        })

    return {
        "ticker": req.ticker,
        "total": len(enriched),
        "sentiment_summary": summary,
        "articles": enriched,
    }


@router.post("/search")
async def semantic_search(req: SearchRequest):
    """Semantic search over stored news articles."""
    results = vector_store.search(req.query, req.n_results, req.label_filter)
    return {"query": req.query, "results": results}


@router.get("/stats")
async def get_stats():
    """Vector store stats."""
    return vector_store.stats()
