from fastapi import APIRouter
from analytics.tracker import get_tracker

router = APIRouter()


@router.get("/")
async def get_analytics():
    """Usage analytics — search counts, latency, top queries, failed searches."""
    return get_tracker().get_stats()
