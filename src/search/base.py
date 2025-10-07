from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import List


class SearchResult(BaseModel):
    """
    Schema for search results.
    Provides strong typing and JSON serialization for LangGraph/FastAPI.
    """
    id: int | str = Field(..., description="Unique identifier of the document or item")
    score: float = Field(..., description="Similarity or relevance score")
    text: str = Field(..., description="Text content of the result")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


class SearchClient(ABC):
    """Abstract search client interface."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Search for documents."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """Optional: pre-load resources or connect to DB."""
        raise NotImplementedError
