from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import search, pdf, citations, health

app = FastAPI(
    title="TriField AI Backend",
    description="AI Research Workspace for Aerospace · Materials · Textile Engineering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(search.router,    prefix="/api/search",    tags=["Paper Search"])
app.include_router(pdf.router,       prefix="/api/pdf",       tags=["PDF Chat"])
app.include_router(citations.router, prefix="/api/citations", tags=["Citations"])

@app.get("/")
def root():
    return {
        "name": "TriField AI",
        "status": "running",
        "docs": "/docs",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"]
    }
