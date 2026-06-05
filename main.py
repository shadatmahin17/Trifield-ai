import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TriField AI Backend",
    description="AI Research Workspace — Aerospace · Materials · Textile Engineering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers AFTER app is created — keeps startup fast
from routers import search, pdf, citations, health

app.include_router(health.router,    tags=["Health"])
app.include_router(search.router,    prefix="/api/search",    tags=["Search"])
app.include_router(pdf.router,       prefix="/api/pdf",       tags=["PDF"])
app.include_router(citations.router, prefix="/api/citations", tags=["Citations"])

@app.get("/")
def root():
    return {
        "name":        "TriField AI",
        "status":      "running",
        "docs":        "/docs",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "endpoints": {
            "search":    "/api/search/?query=carbon+fibre&discipline=aerospace&limit=5",
            "health":    "/health",
            "api_docs":  "/docs",
        }
    }
