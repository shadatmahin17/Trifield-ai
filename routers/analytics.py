from fastapi import APIRouter
from analytics.tracker import get_tracker

router = APIRouter()


@router.get("/")
async def get_analytics():
    """Usage analytics — search counts, latency, top queries, failed searches."""
    stats = get_tracker().get_stats()
    # BUG FIX: get_stats() returns {"message": "No search events recorded yet."}
    # when empty — ensure the response always has a consistent shape for the frontend.
    if "message" in stats:
        return {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "success_rate_pct": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_results_per_search": 0.0,
            "top_queries": [],
            "top_disciplines": [],
            "top_intents": [],
            "top_failed_queries": [],
            "total_pdf_uploads": 0,
            "note": stats["message"],
        }
    return stats
