---
title: FinBERT Financial News Sentiment API
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---




# FinBERT Financial News Sentiment API

End-to-end NLP pipeline for Indian market sentiment analysis using fine-tuned FinBERT.

## Results

| Metric | Base FinBERT | Fine-tuned | Improvement |
|--------|-------------|------------|-------------|
| Accuracy | 53% | 89% | +36% |
| Macro F1 | 0.32 | 0.85 | +0.53 |
| Bearish F1 | 0.03 | 0.81 | +0.78 |
| Bullish F1 | 0.11 | 0.83 | +0.72 |

## Stack
- Model: ProsusAI/FinBERT fine-tuned on financial Twitter sentiment
- API: FastAPI with async batch inference
- Vector Store: ChromaDB for semantic search
- Training: HuggingFace Trainer + PyTorch

## API Endpoints
- POST /api/v1/analyze - single text sentiment
- POST /api/v1/batch - up to 50 texts
- GET /health - service health check

## Quick Start
pip install -r requirements.txt
uvicorn app.main:app --reload

## Key Finding
Base FinBERT achieved 53% accuracy on Indian financial news due to domain mismatch.
Fine-tuning improved macro F1 from 0.32 to 0.85.
