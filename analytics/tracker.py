"""
Analytics tracker — writes to Supabase PostgreSQL when configured,
falls back to in-memory when Supabase is unavailable (dev / cold start).

Table: search_events
Columns: id, query, discipline, intent, result_count, latency_ms,
         success, source, created_at

All DB writes are fire-and-forget (asyncio.create_task) so they never
slow down a search response.
"""
import asyncio
import math
import time
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchEvent:
    query:        str
    discipline:   str
    result_count: int
    latency_ms:   float
    intent:       str
    success:      bool
    source:       str   = "api"
    timestamp:    float = field(default_factory=time.time)


@dataclass
class PDFEvent:
    filename:    str
    session_id:  str
    chunk_count: int
    latency_ms:  float
    timestamp:   float = field(default_factory=time.time)


# ── Supabase persistence helpers ────────────────────────────────────────────

def _insert_search_sync(event: SearchEvent):
    """Synchronous Supabase insert — run in a thread."""
    from db.client import get_db
    get_db().table("search_events").insert({
        "query":        event.query,
        "discipline":   event.discipline,
        "intent":       event.intent,
        "result_count": event.result_count,
        "latency_ms":   event.latency_ms,
        "success":      event.success,
        "source":       event.source,
    }).execute()


async def _persist_search(event: SearchEvent):
    """Fire-and-forget async wrapper — failures are logged, not raised."""
    try:
        await asyncio.to_thread(_insert_search_sync, event)
    except Exception as e:
        logger.warning(f"Analytics DB write failed (falling back to in-memory): {e}")


def _fetch_stats_sync() -> Optional[dict]:
    """
    Pull aggregated stats from Supabase using PostgREST queries.
    Returns None if Supabase is unavailable or the table is empty.
    """
    from db.client import get_db, is_configured
    if not is_configured():
        return None

    db = get_db()

    # All rows (capped at 10 000 most recent for aggregation)
    resp = (
        db.table("search_events")
          .select("query,discipline,intent,result_count,latency_ms,success,source,created_at")
          .order("created_at", desc=True)
          .limit(10_000)
          .execute()
    )
    rows = resp.data or []
    if not rows:
        return None

    now = time.time()
    import dateutil.parser

    latencies     = [r["latency_ms"] for r in rows]
    successful    = [r for r in rows if r["success"]]
    failed        = [r for r in rows if not r["success"]]

    # searches_last_hour using ISO timestamp from Supabase
    def _ts(row):
        try:
            return dateutil.parser.parse(row["created_at"]).timestamp()
        except Exception:
            return 0

    searches_last_hour = sum(1 for r in rows if (now - _ts(r)) < 3600)

    disc_counts   = Counter(r["discipline"] for r in rows)
    intent_counts = Counter(r["intent"]     for r in rows)
    source_counts = Counter(r["source"]     for r in rows)
    src_ok        = Counter(r["source"] for r in rows if r["success"])

    # top queries with avg latency
    q_counts  = Counter(r["query"].strip().lower() for r in rows)
    q_lat_sum: dict = defaultdict(float)
    for r in rows:
        q_lat_sum[r["query"].strip().lower()] += r["latency_ms"]

    top_queries = [
        {"query": q, "count": c, "avg_latency_ms": round(q_lat_sum[q] / c, 1)}
        for q, c in q_counts.most_common(15)
    ]

    fail_counts = Counter(r["query"].strip().lower() for r in failed)
    top_failed  = [{"query": q, "count": c} for q, c in fail_counts.most_common(10)]

    def p95(vals):
        if not vals: return 0.0
        s   = sorted(vals)
        idx = min(math.ceil(len(s) * 0.95) - 1, len(s) - 1)
        return round(s[max(idx, 0)], 1)

    # PDF uploads count from pdf_sessions table
    try:
        pdf_resp = db.table("pdf_sessions").select("session_id", count="exact").execute()
        total_pdfs = pdf_resp.count or 0
    except Exception:
        total_pdfs = 0

    # period_start: oldest event in window
    try:
        oldest = dateutil.parser.parse(rows[-1]["created_at"]).timestamp()
    except Exception:
        oldest = now - 86400

    return {
        "total_searches":          len(rows),
        "successful_searches":     len(successful),
        "failed_searches":         len(failed),
        "success_rate_pct":        round(len(successful) / len(rows) * 100, 1) if rows else 0.0,
        "avg_latency_ms":          round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms":          p95(latencies),
        "avg_results_per_search":  round(
            sum(r["result_count"] for r in successful) / len(successful), 1
        ) if successful else 0.0,
        "searches_last_hour":      searches_last_hour,
        "period_start":            oldest,
        "period_end":              now,
        "discipline_distribution": dict(disc_counts.most_common(10)),
        "intent_distribution":     dict(intent_counts.most_common(10)),
        "top_queries":             top_queries,
        "top_failed_queries":      top_failed,
        "source_success_rates":    {
            src: round(src_ok[src] / source_counts[src], 3)
            for src in source_counts
        },
        "cache_hit_rate":          0.0,
        "total_pdf_uploads":       total_pdfs,
    }


# ── In-memory tracker (fallback / dev) ─────────────────────────────────────

class AnalyticsTracker:
    def __init__(self):
        self.search_events: list[SearchEvent] = []
        self.pdf_events:    list[PDFEvent]    = []
        self._query_counts:      Counter          = Counter()
        self._failed_queries:    Counter          = Counter()
        self._discipline_counts: Counter          = Counter()
        self._intent_counts:     Counter          = Counter()
        self._query_latency_sum: dict[str, float] = defaultdict(float)
        self._started_at: float = time.time()

    def record_search(
        self,
        query:        str,
        discipline:   str,
        result_count: int,
        latency_ms:   float,
        intent:       str  = "general",
        success:      bool = True,
        source:       str  = "api",
    ):
        event = SearchEvent(
            query=query, discipline=discipline, result_count=result_count,
            latency_ms=latency_ms, intent=intent, success=success, source=source,
        )
        # ── Persist to Supabase (fire-and-forget) ──
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_persist_search(event))
            else:
                asyncio.run(_persist_search(event))
        except Exception:
            pass   # Supabase write failure never crashes the search

        # ── Keep in-memory copy for fast get_stats() when Supabase is slow ──
        key = query.strip().lower()
        self.search_events.append(event)
        self._query_counts[key]      += 1
        self._query_latency_sum[key] += latency_ms
        self._discipline_counts[discipline] += 1
        self._intent_counts[intent]  += 1
        if not success:
            self._failed_queries[key] += 1

        if len(self.search_events) > 10_000:
            self.search_events = self.search_events[-10_000:]
            self._rebuild_counters()

    def record_pdf(self, filename: str, session_id: str, chunk_count: int, latency_ms: float):
        # PDF events are recorded in the pdf_sessions table by the PDF router;
        # keep a local copy for the rare case where Supabase is unavailable.
        self.pdf_events.append(PDFEvent(
            filename=filename, session_id=session_id,
            chunk_count=chunk_count, latency_ms=latency_ms,
        ))
        if len(self.pdf_events) > 5_000:
            self.pdf_events = self.pdf_events[-5_000:]

    def _rebuild_counters(self):
        self._query_counts      = Counter()
        self._failed_queries    = Counter()
        self._discipline_counts = Counter()
        self._intent_counts     = Counter()
        self._query_latency_sum = defaultdict(float)
        for e in self.search_events:
            k = e.query.strip().lower()
            self._query_counts[k]      += 1
            self._query_latency_sum[k] += e.latency_ms
            self._discipline_counts[e.discipline] += 1
            self._intent_counts[e.intent]         += 1
            if not e.success:
                self._failed_queries[k] += 1

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values: return 0.0
        s   = sorted(values)
        idx = min(math.ceil(len(s) * 0.95) - 1, len(s) - 1)
        return round(s[max(idx, 0)], 1)

    def get_stats(self) -> dict:
        # ── Try Supabase first (live, persistent data) ──
        try:
            from db.client import is_configured
            if is_configured():
                db_stats = _fetch_stats_sync()
                if db_stats:
                    return db_stats
        except Exception as e:
            logger.warning(f"Supabase get_stats failed, using in-memory: {e}")

        # ── Fallback: aggregate from in-memory events ──
        if not self.search_events:
            return {
                "total_searches": 0, "successful_searches": 0, "failed_searches": 0,
                "success_rate_pct": 0.0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
                "avg_results_per_search": 0.0, "searches_last_hour": 0,
                "discipline_distribution": {}, "intent_distribution": {},
                "top_queries": [], "top_failed_queries": [],
                "source_success_rates": {}, "cache_hit_rate": 0.0,
                "total_pdf_uploads": len(self.pdf_events),
                "period_start": self._started_at, "period_end": time.time(),
            }

        now        = time.time()
        latencies  = [e.latency_ms for e in self.search_events]
        successful = [e for e in self.search_events if e.success]
        failed     = [e for e in self.search_events if not e.success]

        src_counts = Counter(e.source for e in self.search_events)
        src_ok     = Counter(e.source for e in self.search_events if e.success)

        top_queries = [
            {"query": q, "count": c,
             "avg_latency_ms": round(self._query_latency_sum[q] / c, 1)}
            for q, c in self._query_counts.most_common(15)
        ]
        top_failed = [
            {"query": q, "count": c}
            for q, c in self._failed_queries.most_common(10)
        ]

        return {
            "total_searches":          len(self.search_events),
            "successful_searches":     len(successful),
            "failed_searches":         len(failed),
            "success_rate_pct":        round(len(successful) / len(self.search_events) * 100, 1),
            "avg_latency_ms":          round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms":          self._p95(latencies),
            "avg_results_per_search":  round(
                sum(e.result_count for e in successful) / len(successful), 1
            ) if successful else 0.0,
            "searches_last_hour":      sum(1 for e in self.search_events if (now - e.timestamp) < 3600),
            "period_start":            self._started_at,
            "period_end":              now,
            "discipline_distribution": dict(self._discipline_counts.most_common(10)),
            "intent_distribution":     dict(self._intent_counts.most_common(10)),
            "top_queries":             top_queries,
            "top_failed_queries":      top_failed,
            "source_success_rates":    {
                src: round(src_ok[src] / src_counts[src], 3)
                for src in src_counts
            },
            "cache_hit_rate":          0.0,
            "total_pdf_uploads":       len(self.pdf_events),
        }


# ── Singleton ──────────────────────────────────────────────────────────────
_tracker: Optional[AnalyticsTracker] = None

def get_tracker() -> AnalyticsTracker:
    global _tracker
    if _tracker is None:
        _tracker = AnalyticsTracker()
    return _tracker
