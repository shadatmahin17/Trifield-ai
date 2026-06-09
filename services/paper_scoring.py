"""
Paper quality scoring with weighted formula:
  score = 0.40 * relevance
        + 0.25 * citation_score
        + 0.15 * recency_score
        + 0.10 * journal_quality
        + 0.10 * open_access_bonus
"""
import math
import re

CURRENT_YEAR = 2026

# Tier-1 journals for composites/aerospace/textile
TIER1_JOURNALS = {
    "composites science and technology",
    "composites part a",
    "composites part b",
    "composite structures",
    "journal of composite materials",
    "carbon",
    "polymer composites",
    "textile research journal",
    "journal of the textile institute",
    "aiaa journal",
    "aerospace science and technology",
    "journal of aerospace engineering",
    "acta astronautica",
    "international journal of mechanical sciences",
    "materials & design",
    "engineering fracture mechanics",
    "international journal of fatigue",
}

TIER2_JOURNALS = {
    "materials today",
    "polymers",
    "fibers and polymers",
    "journal of materials science",
    "applied composite materials",
    "journal of reinforced plastics and composites",
    "industrial crops and products",
    "construction and building materials",
    "thin-walled structures",
    "polymer composites",
    "journal of natural fibres",
    "cellulose",
    "carbohydrate polymers",
    "bioresource technology",
    "heliyon",
    "materials",
}


def _relevance_score(paper: dict, query_terms: list[str], entities: list[str]) -> float:
    """Score 0-1 based on term/entity match in title + abstract."""
    title    = (paper.get("title")    or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text     = title + " " + abstract

    if not text.strip():
        return 0.1

    score = 0.0
    total_terms = len(query_terms) + len(entities)
    if total_terms == 0:
        return 0.5

    for term in query_terms:
        if len(term) > 2:
            if term in title:    score += 2.0
            elif term in abstract: score += 0.8

    for entity in entities:
        if entity in title:    score += 3.0
        elif entity in abstract: score += 1.5

    max_possible = len(query_terms) * 2 + len(entities) * 3
    return min(1.0, score / max(max_possible, 1))


def _citation_score(citation_count: int) -> float:
    """Log-normalised citation score 0-1. 100 citations ≈ 0.85."""
    if citation_count <= 0:
        return 0.0
    return min(1.0, math.log(citation_count + 1) / math.log(101))


def _recency_score(year: int | None) -> float:
    """Recency score 0-1. Recent papers score higher."""
    if not year:
        return 0.3
    age = CURRENT_YEAR - year
    if age <= 2:   return 1.0
    if age <= 4:   return 0.85
    if age <= 7:   return 0.65
    if age <= 12:  return 0.40
    if age <= 20:  return 0.20
    return 0.10


def _journal_quality_score(journal: str | None) -> float:
    """Journal tier score 0-1."""
    if not journal:
        return 0.3
    j = journal.lower()
    for t1 in TIER1_JOURNALS:
        if t1 in j:
            return 1.0
    for t2 in TIER2_JOURNALS:
        if t2 in j:
            return 0.65
    if "arxiv" in j:
        return 0.35   # preprint
    return 0.40


def _oa_score(open_access_url: str | None) -> float:
    return 1.0 if open_access_url else 0.0


def score_paper(paper: dict, query_terms: list[str], entities: list[str]) -> float:
    """
    Compute weighted quality score for a paper.

    Weights:
      0.40 relevance      — keyword/entity match
      0.25 citation       — log-normalised citation count
      0.15 recency        — publication year
      0.10 journal quality— tier-1/2/unknown
      0.10 open access    — has free PDF
    """
    r  = _relevance_score(paper, query_terms, entities)
    c  = _citation_score(paper.get("citation_count") or 0)
    t  = _recency_score(paper.get("year"))
    j  = _journal_quality_score(paper.get("journal"))
    oa = _oa_score(paper.get("open_access_url"))

    score = (0.40 * r) + (0.25 * c) + (0.15 * t) + (0.10 * j) + (0.10 * oa)
    return round(score, 4)


def rank_papers(papers: list[dict], query_terms: list[str], entities: list[str]) -> list[dict]:
    """Score and sort papers by quality score descending."""
    scored = []
    for p in papers:
        s = score_paper(p, query_terms, entities)
        p["_quality_score"] = s
        scored.append(p)
    scored.sort(key=lambda x: x["_quality_score"], reverse=True)
    return scored
