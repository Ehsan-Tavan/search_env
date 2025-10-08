from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="The search query text (e.g., a question, phrase, or keyword to find relevant documents).")
    source: str = Field(
        "milvus",
        description="The data source to search in. Use 'milvus' for local database search or 'tavily' for web search."
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="The number of top results to return, ranked by similarity or relevance."
    )


class SearchResponse(BaseModel):
    id: int
    score: float
    text: str
    meta: Optional[Dict[str, Any]]
