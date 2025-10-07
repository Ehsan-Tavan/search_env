from typing import List, Dict, Any
from .milvus_search import MilvusSearch

def search_tool(
        query: str,
        source: str = "milvus",
        top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for the given query using Milvus or ....
    Args:
        query: text to search
        source: 'milvus' (local db) or '...' (web)
        top_k: number of results
    """
    if source == "milvus":
        client = MilvusSearch()
    else:
        raise ValueError("Invalid source. Choose 'milvus' or '...'.")
    return [dict(r) for r in client.search(query, top_k=top_k)]
