from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

LABELS = ["bearish", "neutral", "bullish"]
MODEL_NAME = "./finbert-finetuned"


class SentimentModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self):
        logger.info(f"Loading {MODEL_NAME} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("FinBERT loaded successfully")

    def predict(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        results = []
        for i, prob in enumerate(probs):
            scores = prob.cpu().numpy().tolist()
            pred_idx = int(prob.argmax())
            results.append({
                "text": texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                "label": LABELS[pred_idx],
                "confidence": round(scores[pred_idx], 4),
                "scores": {
                    "bullish": round(scores[2], 4),
                    "neutral": round(scores[1], 4),
                    "bearish": round(scores[0], 4),
                },
            })
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]
