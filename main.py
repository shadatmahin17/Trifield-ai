from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import search, pdf, citations, health, copilot, analytics

app = FastAPI(
    title="TriField AI Backend",
    description="AI Research Workspace v2 — Aerospace · Materials · Textile Engineering",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,     tags=["Health"])
app.include_router(search.router,     prefix="/api/search",    tags=["Search"])
app.include_router(pdf.router,        prefix="/api/pdf",       tags=["PDF"])
app.include_router(citations.router,  prefix="/api/citations", tags=["Citations"])
app.include_router(copilot.router,    prefix="/api/copilot",   tags=["Copilot"])
app.include_router(analytics.router,  prefix="/api/analytics", tags=["Analytics"])

@app.get("/")
def root():
    return {
        "name":    "TriField AI",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "new_in_v2": [
            "Qdrant vector search for PDF RAG",
            "LLM query rewriting (Groq)",
            "Weighted paper quality scoring",
            "Research Copilot (gaps, trends, experiments)",
            "SSE streaming search with live progress",
            "Task-aware LLM routing",
            "Usage analytics",
        ],
    }
