"""
Usage analytics tracker — in-memory with optional persistence.
Tracks: searches, latency, failed queries, top topics, PDF uploads.
"""
import time
import math
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
    source:       str = "api"          # "api" | "stream"
    timestamp:    float = field(default_factory=time.time)


@dataclass
class PDFEvent:
    filename:    str
    session_id:  str
    chunk_count: int
    latency_ms:  float
    timestamp:   float = field(default_factory=time.time)


class AnalyticsTracker:
    def __init__(self):
        self.search_events: list[SearchEvent] = []
        self.pdf_events:    list[PDFEvent]    = []
        # Fast counters — kept in sync with search_events list
        self._query_counts:      Counter = Counter()
        self._failed_queries:    Counter = Counter()
        self._discipline_counts: Counter = Counter()
        self._intent_counts:     Counter = Counter()
        # Per-source success/fail counts for success-rate calculation
        self._source_success:    Counter = Counter()
        self._source_total:      Counter = Counter()
        # Per-query latency accumulator for avg_latency_ms in top_queries
        self._query_latency_sum: dict[str, float] = defaultdict(float)
        self._started_at: float = time.time()

    # ── Record events ──────────────────────────────────────────────────────

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
        key = query.strip().lower()
        event = SearchEvent(
            query=query, discipline=discipline, result_count=result_count,
            latency_ms=latency_ms, intent=intent, success=success, source=source,
        )
        self.search_events.append(event)
        self._query_counts[key] += 1
        self._query_latency_sum[key] += latency_ms
        self._discipline_counts[discipline] += 1
        self._intent_counts[intent] += 1
        if not success:
            self._failed_queries[key] += 1

        # Keep last 10 000 events to bound memory
        if len(self.search_events) > 10_000:
            dropped = self.search_events[:-10_000]
            self.search_events = self.search_events[-10_000:]
            # Rebuild fast counters from surviving events to stay in sync
            self._rebuild_counters()

    def record_pdf(self, filename: str, session_id: str, chunk_count: int, latency_ms: float):
        self.pdf_events.append(PDFEvent(
            filename=filename, session_id=session_id,
            chunk_count=chunk_count, latency_ms=latency_ms,
        ))
        if len(self.pdf_events) > 5_000:
            self.pdf_events = self.pdf_events[-5_000:]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _rebuild_counters(self):
        """Rebuild all Counter state from self.search_events after a trim."""
        self._query_counts      = Counter()
        self._failed_queries    = Counter()
        self._discipline_counts = Counter()
        self._intent_counts     = Counter()
        self._query_latency_sum = defaultdict(float)
        for e in self.search_events:
            key = e.query.strip().lower()
            self._query_counts[key] += 1
            self._query_latency_sum[key] += e.latency_ms
            self._discipline_counts[e.discipline] += 1
            self._intent_counts[e.intent] += 1
            if not e.success:
                self._failed_queries[key] += 1

    @staticmethod
    def _p95(values: list[float]) -> float:
        """True p95 using nearest-rank method — never raises IndexError."""
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(math.ceil(len(s) * 0.95) - 1, len(s) - 1)
        return round(s[max(idx, 0)], 1)

    # ── Stats ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
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

        now       = time.time()
        latencies = [e.latency_ms for e in self.search_events]
        successful = [e for e in self.search_events if e.success]
        failed     = [e for e in self.search_events if not e.success]

        # Searches in the last hour
        searches_last_hour = sum(
            1 for e in self.search_events if (now - e.timestamp) < 3600
        )

        # avg results (successful only)
        avg_results = (
            sum(e.result_count for e in successful) / len(successful)
            if successful else 0.0
        )

        # Discipline distribution as dict (frontend expects {str: int})
        discipline_distribution = dict(self._discipline_counts.most_common(10))

        # Intent distribution as dict
        intent_distribution = dict(self._intent_counts.most_common(10))

        # Top queries as list of dicts with avg_latency_ms
        top_queries = [
            {
                "query": q,
                "count": cnt,
                "avg_latency_ms": round(self._query_latency_sum[q] / cnt, 1),
            }
            for q, cnt in self._query_counts.most_common(15)
        ]

        # Top failed queries as list of dicts
        top_failed_queries = [
            {"query": q, "count": cnt}
            for q, cnt in self._failed_queries.most_common(10)
        ]

        # Source success rates — derived from search_events source field
        # (api vs stream breakdowns; extend if per-API-source data added)
        source_counts: Counter = Counter()
        source_success: Counter = Counter()
        for e in self.search_events:
            source_counts[e.source] += 1
            if e.success:
                source_success[e.source] += 1
        source_success_rates = {
            src: round(source_success[src] / source_counts[src], 3)
            for src in source_counts
        }

        return {
            # ── Core counts ──
            "total_searches":         len(self.search_events),
            "successful_searches":    len(successful),
            "failed_searches":        len(failed),
            "success_rate_pct":       round(len(successful) / len(self.search_events) * 100, 1),

            # ── Latency ──
            "avg_latency_ms":         round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms":         self._p95(latencies),
            "avg_results_per_search": round(avg_results, 1),

            # ── Time window ──
            "searches_last_hour":     searches_last_hour,
            "period_start":           self._started_at,
            "period_end":             now,

            # ── Distributions (dicts — frontend-compatible) ──
            "discipline_distribution": discipline_distribution,
            "intent_distribution":     intent_distribution,

            # ── Top queries (list of dicts — frontend-compatible) ──
            "top_queries":             top_queries,
            "top_failed_queries":      top_failed_queries,

            # ── Source rates ──
            "source_success_rates":    source_success_rates,

            # ── Extras ──
            "cache_hit_rate":          0.0,   # extend when cache logging is added
            "total_pdf_uploads":       len(self.pdf_events),
        }


# ── Singleton ──────────────────────────────────────────────────────────────
_tracker: Optional[AnalyticsTracker] = None

def get_tracker() -> AnalyticsTracker:
    global _tracker
    if _tracker is None:
        _tracker = AnalyticsTracker()
    return _tracker
