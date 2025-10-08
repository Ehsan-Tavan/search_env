from typing import List, Dict, Any
import torch

from .milvus_search import MilvusSearch
from src.configurations import Config


def detect_device() -> str:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name} ({num_gpus} GPU(s) available)")
        return "cuda"
    else:
        print("No GPU detected. Using CPU.")
        return "cpu"


def search_tool(
        query: str,
        source: str = "milvus",
        top_k: int = 5,
        config: Config = None
) -> List[Dict[str, Any]]:
    """
    Search for the given query using Milvus or ....
    Args:
        query: text to search
        source: 'milvus' (local db) or '...' (web)
        top_k: number of results
        config: configuration
    """

    if source == "milvus":
        client = MilvusSearch(
            database_name=config.milvus.db_name,
            collection_name=config.milvus.collection_name,
            host=config.milvus.host,
            port=str(config.milvus.port),
            model_name=config.model.name,
            device=detect_device(),
        )
    else:
        raise ValueError("Invalid source. Choose 'milvus' or '...'.")
    return [dict(r) for r in client.search(query, top_k=top_k)]
