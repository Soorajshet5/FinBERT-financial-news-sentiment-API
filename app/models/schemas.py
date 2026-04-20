from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    text: str
    ticker: Optional[str] = None

class SentimentResult(BaseModel):
    label: str        # bullish / bearish / neutral
    confidence: float
    ticker: Optional[str] = None


# Batch request for multiple texts
from typing import List, Dict

class BatchAnalyzeRequest(BaseModel):
    texts: List[str]
    ticker: Optional[str] = None

# News analysis request/response
class NewsAnalysisRequest(BaseModel):
    ticker: str
    limit: Optional[int] = 10

class NewsArticle(BaseModel):
    title: str
    summary: str
    source: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None
    ticker: Optional[str] = None

class NewsAnalysisResponse(BaseModel):
    articles: List[NewsArticle]
    sentiments: List[SentimentResult]

# Semantic search request
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
