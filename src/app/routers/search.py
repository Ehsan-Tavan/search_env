from typing import List
from fastapi import APIRouter, Request, HTTPException

from src.app.deps import get_search_client
from src.app.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["Search"])

@router.post("/search", response_model=List[SearchResponse])
def search_endpoint(req: SearchRequest, request: Request):
    try:
        config = request.app.state.config
        client = get_search_client(req.source, config=config)
        results = client.search(req.query, top_k=req.top_k)
        return [r.model_dump() for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
