import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


class NewsVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name="financial_news",
            metadata={"hnsw:space": "cosine"},
        )

    def _doc_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def add_articles(self, articles: List[Dict], sentiments: List[Dict]):
        """Store articles with their sentiment metadata."""
        docs, ids, metas = [], [], []
        for article, sentiment in zip(articles, sentiments):
            text = f"{article['title']} {article['summary']}"
            doc_id = self._doc_id(text)
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "title": article["title"][:200],
                "source": article.get("source", ""),
                "ticker": article.get("ticker", ""),
                "label": sentiment["label"],
                "confidence": str(sentiment["confidence"]),
                "published_at": article.get("published_at", ""),
                "url": article.get("url", ""),
            })

        if docs:
            embeddings = self.embedder.encode(docs).tolist()
            self.collection.upsert(documents=docs, ids=ids, metadatas=metas, embeddings=embeddings)
            logger.info(f"Stored {len(docs)} articles in vector store")

    def search(self, query: str, n_results: int = 5, label_filter: str = None) -> List[Dict]:
        """Semantic search with optional sentiment label filter."""
        query_embedding = self.embedder.encode([query]).tolist()
        where = {"label": label_filter} if label_filter else None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )
        items = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            items.append({
                "text": doc,
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "label": meta.get("label", ""),
                "confidence": float(meta.get("confidence", 0)),
                "ticker": meta.get("ticker", ""),
                "url": meta.get("url", ""),
                "published_at": meta.get("published_at", ""),
            })
        return items

    def stats(self) -> Dict:
        count = self.collection.count()
        return {"total_articles": count, "collection": "financial_news"}