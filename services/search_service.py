import httpx
import asyncio
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from models.schemas import Paper, Author

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
CACHE_TTL = 3600

# ── Discipline config ──────────────────────────────────────────────────────
DISCIPLINE_KEYWORDS = {
    "aerospace": [
        "aerospace", "aeronautics", "aircraft", "spacecraft", "cfrp",
        "carbon fibre", "carbon fiber", "airframe", "aerostructure",
        "damage tolerance", "composite laminate", "sandwich panel",
        "aeroelastic", "fuselage", "wing structure", "fatigue crack",
        "structural health monitoring", "composite structure",
    ],
    "materials": [
        "composite", "hybrid composite", "nanocomposite", "polymer matrix",
        "epoxy", "fibre reinforced", "fiber reinforced", "tensile strength",
        "flexural strength", "void content", "fibre volume fraction",
        "delamination", "fracture toughness", "matrix cracking", "interlaminar",
        "mechanical properties", "impact resistance",
    ],
    "textile": [
        "textile composite", "woven composite", "woven fabric composite",
        "natural fibre composite", "natural fiber composite",
        "jute composite", "flax composite", "hemp composite",
        "hybrid composite", "bast fibre", "bast fiber",
        "fabric reinforced", "preform", "woven reinforcement",
        "technical textile", "braided composite",
    ],
}

# Minimum keyword hits required for a paper to pass discipline filter
DISCIPLINE_MIN_SCORE = {
    "aerospace": 1,
    "materials": 1,
    "textile":   2,   # stricter — textile has more noise
    "all":       0,
}

DISCIPLINE_CONCEPTS = {
    "aerospace": "C27206212",
    "materials": "C192562407",
    "textile":   "C107038049",
}

EXCLUDE_CONCEPTS = {
    "C127413603",  # Civil engineering
    "C144024400",  # Medicine
    "C185592680",  # Pure chemistry
}

# Junk result patterns — filter out standards docs, patents, etc.
JUNK_TITLE_PATTERNS = [
    r"^specification for",
    r"^standard for",
    r"^iso \d+",
    r"^bs \d+",
    r"^astm [a-z]",
    r"twines made from",
    r"^patent",
]


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _score(title: str, abstract: str | None, discipline: str) -> int:
    if discipline == "all":
        return 1
    text = (title + " " + (abstract or "")).lower()
    return sum(1 for kw in DISCIPLINE_KEYWORDS.get(discipline, []) if kw in text)


def _tag_discipline(title: str, abstract: str | None, concepts: list) -> str:
    concept_names = " ".join(c.get("display_name", "").lower() for c in concepts)
    text = (title + " " + (abstract or "") + " " + concept_names).lower()
    scores = {d: sum(1 for kw in kws if kw in text)
              for d, kws in DISCIPLINE_KEYWORDS.items()}
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def _is_excluded(concepts: list) -> bool:
    ids = {c.get("id", "").split("/")[-1] for c in concepts}
    return bool(ids & EXCLUDE_CONCEPTS)


def _is_junk(paper: dict) -> bool:
    """Filter out standards, specs, patents, and papers with no useful metadata."""
    title = (paper.get("title") or "").lower().strip()

    # No title or no year and no authors = useless
    if not title:
        return True
    has_authors = bool(paper.get("authors"))
    has_year    = bool(paper.get("year"))
    if not has_authors and not has_year:
        return True

    # Match junk title patterns
    for pat in JUNK_TITLE_PATTERNS:
        if re.match(pat, title, re.IGNORECASE):
            return True

    return False


def _reconstruct_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    try:
        pos = {}
        for word, positions in inv.items():
            for p in positions:
                pos[p] = word
        return " ".join(pos[i] for i in sorted(pos)) or None
    except Exception:
        return None


def _clean_abstract(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text if len(text) > 30 else None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:80]


def _cache_key(*args) -> str:
    return hashlib.md5("|".join(str(a) for a in args).encode()).hexdigest()


def _get_cached(key):
    e = _cache.get(key)
    return e["papers"] if e and time.time() < e["expires"] else None


def _set_cache(key, papers):
    _cache[key] = {"papers": papers, "expires": time.time() + CACHE_TTL}


# ══════════════════════════════════════════════════════════════════
#  ABSTRACT ENRICHMENT — Crossref direct DOI lookup
# ══════════════════════════════════════════════════════════════════

async def _fetch_abstract_crossref(doi: str, client: httpx.AsyncClient) -> str | None:
    """Direct Crossref DOI lookup for abstract."""
    if not doi:
        return None
    doi_clean = doi.replace("https://doi.org/", "").strip()
    try:
        resp = await client.get(
            f"https://api.crossref.org/works/{doi_clean}",
            timeout=8,
            headers={"User-Agent": "TriFieldAI/1.0 (contact@trifield.ai)"}
        )
        if resp.status_code == 200:
            abstract = resp.json().get("message", {}).get("abstract", "")
            return _clean_abstract(abstract)
    except Exception:
        pass
    return None


async def _fetch_abstract_s2(doi: str, client: httpx.AsyncClient) -> str | None:
    """
    Semantic Scholar paper lookup — good coverage for older pre-2000 papers.
    Free, no key required for low volume.
    """
    if not doi:
        return None
    doi_clean = doi.replace("https://doi.org/", "").strip()
    try:
        resp = await client.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_clean}",
            params={"fields": "abstract"},
            timeout=8,
            headers={"User-Agent": "TriFieldAI/1.0"}
        )
        if resp.status_code == 200:
            abstract = resp.json().get("abstract") or ""
            return _clean_abstract(abstract)
    except Exception:
        pass
    return None


async def _fetch_abstract_by_doi(doi: str, client: httpx.AsyncClient) -> str | None:
    """
    Abstract pipeline: Crossref first, Semantic Scholar as fallback.
    Crossref: best coverage post-2000.
    Semantic Scholar: best for older papers with curated abstracts.
    Both run sequentially only if needed — fast path exits early.
    """
    abstract = await _fetch_abstract_crossref(doi, client)
    if abstract:
        return abstract
    return await _fetch_abstract_s2(doi, client)
    doi_clean = doi.replace("https://doi.org/", "").strip()
    try:
        resp = await client.get(
            f"https://api.crossref.org/works/{doi_clean}",
            timeout=8,
            headers={"User-Agent": "TriFieldAI/1.0 (contact@trifield.ai)"}
        )
        if resp.status_code == 200:
            abstract = resp.json().get("message", {}).get("abstract", "")
            return _clean_abstract(abstract)
    except Exception:
        pass
    return None


async def _get_oa_url(doi: str, client: httpx.AsyncClient) -> str | None:
    """Unpaywall: find best free PDF for a DOI."""
    if not doi:
        return None
    doi_clean = doi.replace("https://doi.org/", "").strip()
    try:
        resp = await client.get(
            f"https://api.unpaywall.org/v2/{doi_clean}?email=contact@trifield.ai",
            timeout=6
        )
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


async def _enrich(paper: dict, client: httpx.AsyncClient) -> dict:
    """
    For papers missing abstract or OA url, fetch both concurrently.
    Returns updated paper dict.
    """
    doi      = paper.get("doi") or ""
    abstract = paper.get("abstract")
    oa_url   = paper.get("open_access_url")

    tasks = []
    need_abstract = not abstract and doi
    need_oa       = not oa_url and doi

    if need_abstract and need_oa:
        abstract_new, oa_new = await asyncio.gather(
            _fetch_abstract_by_doi(doi, client),
            _get_oa_url(doi, client),
        )
    elif need_abstract:
        abstract_new = await _fetch_abstract_by_doi(doi, client)
        oa_new = None
    elif need_oa:
        abstract_new = None
        oa_new = await _get_oa_url(doi, client)
    else:
        return paper  # nothing to enrich

    if abstract_new:
        paper["abstract"] = abstract_new
    if oa_new:
        paper["open_access_url"] = oa_new

    return paper


# ══════════════════════════════════════════════════════════════════
#  SOURCE 1 — OpenAlex
# ══════════════════════════════════════════════════════════════════

async def _search_openalex(
    query: str, discipline: str, year_from, year_to, limit: int,
    client: httpx.AsyncClient
) -> list[dict]:
    params = {
        "search":   query,
        "per_page": min(limit * 2, 50),
        "select": (
            "id,title,authorships,publication_year,"
            "abstract_inverted_index,cited_by_count,doi,"
            "primary_location,open_access,concepts"
        ),
        "sort":   "relevance_score:desc",
        "mailto": "contact@trifield.ai",
    }
    filters = []
    if discipline != "all" and discipline in DISCIPLINE_CONCEPTS:
        filters.append(f"concepts.id:{DISCIPLINE_CONCEPTS[discipline]}")
    if year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    if year_to:
        filters.append(f"publication_year:<{year_to + 1}")
    if filters:
        params["filter"] = ",".join(filters)

    try:
        resp = await client.get(
            "https://api.openalex.org/works", params=params, timeout=15
        )
        if resp.status_code != 200:
            return []
        papers = []
        for item in resp.json().get("results", []):
            doi      = item.get("doi") or ""
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
            oa       = item.get("open_access") or {}
            primary  = item.get("primary_location") or {}
            source   = primary.get("source") or {}
            authors  = [
                Author(name=a.get("author", {}).get("display_name", "Unknown"))
                for a in item.get("authorships", [])[:10]
            ]
            papers.append({
                "paper_id":        item.get("id", "").split("/")[-1],
                "title":           item.get("title") or "",
                "authors":         authors,
                "year":            item.get("publication_year"),
                "abstract":        abstract,
                "citation_count":  item.get("cited_by_count", 0),
                "doi":             doi,
                "url":             doi or "",
                "open_access_url": oa.get("oa_url"),
                "journal":         source.get("display_name"),
                "concepts":        item.get("concepts") or [],
                "source":          "openalex",
            })
        return papers
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  SOURCE 2 — Crossref
# ══════════════════════════════════════════════════════════════════

async def _search_crossref(
    query: str, discipline: str, year_from, year_to, limit: int,
    client: httpx.AsyncClient
) -> list[dict]:
    params = {
        "query":  query,
        "rows":   min(limit * 2, 40),
        "select": "DOI,title,author,published,abstract,container-title,is-referenced-by-count,type",
        "sort":   "relevance",
        "mailto": "contact@trifield.ai",
        # Only journal articles and conference papers — excludes standards
        "filter": "type:journal-article,type:proceedings-article",
    }
    if year_from:
        params["filter"] += f",from-pub-date:{year_from}"
    if year_to:
        params["filter"] += f",until-pub-date:{year_to}"

    try:
        resp = await client.get(
            "https://api.crossref.org/works", params=params, timeout=15,
            headers={"User-Agent": "TriFieldAI/1.0 (contact@trifield.ai)"}
        )
        if resp.status_code != 200:
            return []
        papers = []
        for item in resp.json().get("message", {}).get("items", []):
            doi        = item.get("DOI") or ""
            title_list = item.get("title") or []
            title      = title_list[0] if title_list else ""
            if not title:
                continue
            authors = [
                Author(name=f"{a.get('given','')} {a.get('family','')}".strip())
                for a in item.get("author", [])[:10]
            ]
            pub        = item.get("published") or {}
            date_parts = pub.get("date-parts") or [[None]]
            year       = date_parts[0][0] if date_parts and date_parts[0] else None
            abstract   = _clean_abstract(item.get("abstract") or "")
            journal    = (item.get("container-title") or [""])[0] or None

            papers.append({
                "paper_id":        f"CR_{doi.replace('/','_')}",
                "title":           title,
                "authors":         authors,
                "year":            year,
                "abstract":        abstract,
                "citation_count":  item.get("is-referenced-by-count") or 0,
                "doi":             f"https://doi.org/{doi}" if doi else "",
                "url":             f"https://doi.org/{doi}" if doi else "",
                "open_access_url": None,
                "journal":         journal,
                "concepts":        [],
                "source":          "crossref",
            })
        return papers
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  SOURCE 3 — arXiv
# ══════════════════════════════════════════════════════════════════

async def _search_arxiv(
    query: str, discipline: str, year_from, year_to, limit: int,
    client: httpx.AsyncClient
) -> list[dict]:
    cat_map = {
        "aerospace": "cat:cond-mat.mtrl-sci OR cat:physics.flu-dyn",
        "materials": "cat:cond-mat.mtrl-sci",
        "textile":   "cat:cond-mat.soft",
        "all":       "",
    }
    cat          = cat_map.get(discipline, "")
    search_query = f"all:{query}"
    if cat:
        search_query = f"({search_query}) AND ({cat})"

    try:
        resp = await client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": search_query,
                "start":        0,
                "max_results":  min(limit, 15),
                "sortBy":       "relevance",
                "sortOrder":    "descending",
            },
            timeout=15
        )
        if resp.status_code != 200:
            return []

        ns   = {"atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(resp.text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
            title    = (entry.findtext("atom:title", "", ns) or "").replace("\n", " ").strip()
            abstract = (entry.findtext("atom:summary", "", ns) or "").replace("\n", " ").strip()
            if not title:
                continue

            published = entry.findtext("atom:published", "", ns) or ""
            year      = int(published[:4]) if len(published) >= 4 else None

            if year_from and year and year < year_from:
                continue
            if year_to and year and year > year_to:
                continue

            authors = [
                Author(name=a.findtext("atom:name", "", ns) or "Unknown")
                for a in entry.findall("atom:author", ns)
            ][:10]

            doi_tag = entry.findtext("arxiv:doi", None, ns)
            doi_url = f"https://doi.org/{doi_tag}" if doi_tag else ""
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            papers.append({
                "paper_id":        f"AR_{arxiv_id.replace('.','_')}",
                "title":           title,
                "authors":         authors,
                "year":            year,
                "abstract":        abstract or None,
                "citation_count":  0,
                "doi":             doi_url,
                "url":             doi_url or f"https://arxiv.org/abs/{arxiv_id}",
                "open_access_url": pdf_url,
                "journal":         "arXiv preprint",
                "concepts":        [],
                "source":          "arxiv",
            })
        return papers
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  SOURCE 4 — PubMed
# ══════════════════════════════════════════════════════════════════

async def _search_pubmed(
    query: str, discipline: str, year_from, year_to, limit: int,
    client: httpx.AsyncClient
) -> list[dict]:
    if discipline == "aerospace":
        return []

    search_params = {
        "db":      "pubmed",
        "term":    f"{query} AND (composite OR textile OR fibre OR fiber)",
        "retmax":  min(limit, 15),
        "retmode": "json",
        "sort":    "relevance",
        "tool":    "TriFieldAI",
        "email":   "contact@trifield.ai",
    }
    if year_from:
        search_params["mindate"] = str(year_from)
    if year_to:
        search_params["maxdate"] = str(year_to)

    try:
        resp = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=search_params, timeout=12
        )
        if resp.status_code != 200:
            return []
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        fetch_resp = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(pmids),
                "retmode": "xml",
                "tool":    "TriFieldAI",
                "email":   "contact@trifield.ai",
            },
            timeout=12
        )
        if fetch_resp.status_code != 200:
            return []

        root   = ET.fromstring(fetch_resp.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            art = medline.find("Article")
            if art is None:
                continue

            pmid     = (medline.findtext("PMID") or "").strip()
            title    = (art.findtext("ArticleTitle") or "").strip()
            if not title:
                continue

            abs_el   = art.find("Abstract/AbstractText")
            abstract = abs_el.text.strip() if abs_el is not None and abs_el.text else None

            pub_date = art.find(".//PubDate")
            year     = None
            if pub_date is not None:
                yr   = pub_date.findtext("Year")
                year = int(yr) if yr and yr.isdigit() else None

            authors = []
            for a in art.findall("AuthorList/Author")[:10]:
                last = a.findtext("LastName") or ""
                fore = a.findtext("ForeName") or ""
                name = f"{fore} {last}".strip()
                if name:
                    authors.append(Author(name=name))

            journal = art.findtext("Journal/Title")
            doi     = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                    break

            papers.append({
                "paper_id":        f"PM_{pmid}",
                "title":           title,
                "authors":         authors,
                "year":            year,
                "abstract":        abstract,
                "citation_count":  0,
                "doi":             f"https://doi.org/{doi}" if doi else "",
                "url":             (f"https://doi.org/{doi}" if doi
                                    else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                "open_access_url": (f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{pmid}/"
                                    if pmid else None),
                "journal":         journal,
                "concepts":        [],
                "source":          "pubmed",
            })
        return papers
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  MERGE + DEDUPLICATE
# ══════════════════════════════════════════════════════════════════

def _deduplicate(papers: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}

    for p in papers:
        doi    = (p.get("doi") or "").strip()
        ntitle = _norm_title(p.get("title") or "")
        key    = doi if doi else ntitle
        if not key:
            continue

        if key in by_key:
            existing = by_key[key]
            if not existing.get("abstract") and p.get("abstract"):
                existing["abstract"] = p["abstract"]
            if not existing.get("open_access_url") and p.get("open_access_url"):
                existing["open_access_url"] = p["open_access_url"]
            if (p.get("citation_count") or 0) > (existing.get("citation_count") or 0):
                existing["citation_count"] = p["citation_count"]
            if p["source"] in ("crossref", "pubmed") and p.get("journal"):
                existing["journal"] = p["journal"]
        else:
            by_key[key] = p
            # Also index by title to catch DOI mismatches
            if ntitle and ntitle not in by_key:
                by_key[ntitle] = p

    seen:   set[str] = set()
    result: list[dict] = []
    for p in papers:
        ntitle = _norm_title(p.get("title") or "")
        if ntitle and ntitle not in seen:
            seen.add(ntitle)
            doi = (p.get("doi") or "").strip()
            result.append(by_key.get(doi) or by_key.get(ntitle) or p)
    return result


# ══════════════════════════════════════════════════════════════════
#  MAIN SEARCH FUNCTION
# ══════════════════════════════════════════════════════════════════

async def search_papers(
    query: str,
    discipline: str = "all",
    year_from: int | None = None,
    year_to:   int | None = None,
    limit:     int = 10,
) -> list[Paper]:

    # 1. Cache check
    key    = _cache_key(query, discipline, year_from, year_to, limit)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    # 2. Query all 4 sources concurrently
    async with httpx.AsyncClient(timeout=20) as client:
        results = await asyncio.gather(
            _search_openalex(query, discipline, year_from, year_to, limit, client),
            _search_crossref(query, discipline, year_from, year_to, limit, client),
            _search_arxiv(   query, discipline, year_from, year_to, limit, client),
            _search_pubmed(  query, discipline, year_from, year_to, limit, client),
            return_exceptions=True
        )

        # Flatten — skip errored sources
        all_raw: list[dict] = []
        for r in results:
            if isinstance(r, list):
                all_raw.extend(r)

        # 3. Remove junk results first
        all_raw = [p for p in all_raw if not _is_junk(p)]

        # 4. Deduplicate across sources
        merged = _deduplicate(all_raw)

        # 5. Filter by discipline relevance
        min_score = DISCIPLINE_MIN_SCORE.get(discipline, 1)
        relevant: list[dict] = []
        for p in merged:
            if _is_excluded(p.get("concepts") or []):
                continue
            if _score(p.get("title", ""), p.get("abstract"), discipline) >= min_score:
                relevant.append(p)

        # 6. Sort: discipline match first, then citation count desc
        relevant.sort(key=lambda p: (
            0 if (discipline == "all" or _tag_discipline(
                p.get("title", ""), p.get("abstract"), p.get("concepts", [])
            ) == discipline) else 1,
            -(p.get("citation_count") or 0)
        ))

        # 7. Take top N × 2 candidates, enrich abstracts + OA urls concurrently
        candidates = relevant[:limit * 2]
        enriched = await asyncio.gather(
            *[_enrich(p, client) for p in candidates],
            return_exceptions=True
        )

        # Replace with enriched versions (skip any that errored)
        final_candidates = []
        for orig, result in zip(candidates, enriched):
            final_candidates.append(result if isinstance(result, dict) else orig)

        # 8. Final relevance re-score after enrichment (abstracts now available)
        #    and take final top N
        final_candidates.sort(key=lambda p: (
            0 if (discipline == "all" or _tag_discipline(
                p.get("title", ""), p.get("abstract"), p.get("concepts", [])
            ) == discipline) else 1,
            -(p.get("citation_count") or 0)
        ))
        top = final_candidates[:limit]

    # 9. Build Paper objects
    papers: list[Paper] = []
    for p in top:
        abstract = p.get("abstract")
        concepts = p.get("concepts") or []
        papers.append(Paper(
            paper_id        = p["paper_id"],
            title           = p.get("title") or "Untitled",
            authors         = p.get("authors") or [],
            year            = p.get("year"),
            abstract        = abstract,
            citation_count  = p.get("citation_count") or 0,
            url             = p.get("url") or "",
            open_access_url = p.get("open_access_url"),
            journal         = p.get("journal"),
            discipline_tag  = _tag_discipline(p.get("title", ""), abstract, concepts),
        ))

    _set_cache(key, papers)
    return papers
