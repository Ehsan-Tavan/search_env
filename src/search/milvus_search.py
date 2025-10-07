from typing import List
import numpy as np
from pymilvus import connections, Collection, db
from sentence_transformers import SentenceTransformer
from .base import SearchClient, SearchResult


class MilvusSearch(SearchClient):
    def __init__(
        self,
        database_name: str = "default",
        collection_name: str = "documents",
        host: str = "localhost",
        port: str = "19530",
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name, device=device)

        connections.connect(host=host, port=port)
        db.using_database(database_name)

        self._collection = Collection(collection_name)

    def search(
            self,
            query: str,
            top_k: int = 5,
            **kwargs
    ) -> List[SearchResult]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32).tolist()
        params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._collection.search(
            q_emb,
            "embedding",
            param=params,
            limit=top_k,
            output_fields=["text"]
        )
        return [
            SearchResult(id=int(hit.id), score=float(hit.distance), text=hit.entity.get("text"))
            for hit in results[0]
        ]

    def warmup(self) -> None:
        self._collection.load()
