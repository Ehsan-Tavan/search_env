from fastapi import APIRouter

router = APIRouter(tags=["Root"])

@router.get("/")
def root():
    """
    Root endpoint to verify that the Search API is running.
    """
    return {
        "message": "Search API is running 🚀",
        "endpoints": ["/search"]
    }
