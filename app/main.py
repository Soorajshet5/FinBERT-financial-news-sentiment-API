from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.model import SentimentModel
from app.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FinBERT model...")
    app.state.model = SentimentModel()
    await app.state.model.load()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Financial News Intelligence API",
    description="End-to-end NLP pipeline for NSE market sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "FinBERT", "version": "1.0.0"}
