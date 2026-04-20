# FinBERT News Intelligence

End-to-end NLP pipeline for financial news sentiment analysis targeting NSE equities.
Built with FinBERT (ProsusAI), FastAPI, ChromaDB, and Docker.

---

## Architecture

```
Alpha Vantage / NewsAPI
        ↓
   [Ingestor]  app/pipeline/ingestor.py
        ↓
 [FinBERT Model]  app/core/model.py
        ↓
[ChromaDB Vector Store]  app/pipeline/vectorstore.py
        ↓
  [FastAPI Routes]  app/api/routes.py
        ↓
  REST API on :8000
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /api/v1/analyze | Single text sentiment |
| POST | /api/v1/analyze/batch | Batch sentiment (max 50) |
| POST | /api/v1/news/analyze | Fetch + analyze ticker news |
| POST | /api/v1/search | Semantic search over stored articles |
| GET | /api/v1/stats | Vector store stats |

---

## Quick Start

```bash
# 1. Clone and setup env
cp .env.example .env
# Add your API keys to .env

# 2. Install deps
pip install -r requirements.txt

# 3. Run locally
uvicorn app.main:app --reload --port 8000

# 4. Or with Docker
docker-compose up --build
```

---

## Example Requests

### Single text
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Reliance Industries Q4 profit surges 12%, beats estimates"}'
```

### Analyze news for a ticker
```bash
curl -X POST http://localhost:8000/api/v1/news/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE", "limit": 10}'
```

### Semantic search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Reliance green energy expansion", "n_results": 5, "label_filter": "bullish"}'
```

---

## Tech Stack

| Component | Tech |
|-----------|------|
| NLP Model | FinBERT (ProsusAI/finbert) via HuggingFace |
| Deep Learning | PyTorch |
| API Framework | FastAPI |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| News Sources | Alpha Vantage, NewsAPI |
| Containerization | Docker + docker-compose |
| Cloud Deploy | Railway / GCP Cloud Run |

---

## JD Alignment

| JD Requirement | Implementation |
|----------------|----------------|
| NLP model | FinBERT fine-tuned on financial text |
| PyTorch | Model inference layer |
| End-to-end pipeline | Ingest → Embed → Predict → Store → Serve |
| MLOps / Docker | Dockerized with health checks |
| Cloud deploy | Railway-ready (Dockerfile + env vars) |
| Scalability | Async FastAPI + batch inference |

---

## Environment Variables

```
ALPHA_VANTAGE_KEY=  # Get from alphavantage.co
NEWSAPI_KEY=        # Get from newsapi.org
CHROMA_PATH=./chroma_db
```