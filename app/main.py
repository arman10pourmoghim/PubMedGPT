# app/main.py
from fastapi import FastAPI, Request
from app.config import settings
from app.__about__ import __app_name__, __version__
from app.routers_pubmed import router as pubmed_router
from app.routers_answer import router as answer_router
from app.metrics import metrics
import time

app = FastAPI(
    title="PubMed-Grounded GPT",
    version=__version__,
    description="Retrieval-first literature assistant (PubMed/PMC → GPT).",
)

# ---------------------- Middleware (compatible with simple metrics.py) ----------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        path = request.url.path
        method = request.method
        metrics.inc("http.requests.total")
        # per-route latency key; safe for dev (in prod, consider path templating)
        metrics.observe_ms(f"http.latency.{method}.{path}", ms)

# ---------------------- Health endpoint ----------------------
@app.get("/health")
def health():
    """
    Operational heartbeat for platform observability.
    Confirms config wiring without leaking secrets.
    """
    return {
        "app": __app_name__,
        "version": __version__,
        "openai_key_loaded": bool(settings.openai_api_key),
        "ncbi_key_loaded": bool(settings.ncbi_api_key),
        "ncbi_email": bool(settings.ncbi_email),
        "ncbi_tool": bool(settings.ncbi_tool),
        "status": "ok",
    }

# ---------------------- Metrics endpoint (JSON snapshot) ----------------------
@app.get("/_metrics")
def get_metrics():
    """
    Simple JSON metrics snapshot (counters + p95 latency) from app.metrics.Metrics.
    """
    return metrics.snapshot()

# ---------------------- API routers ----------------------
app.include_router(pubmed_router, prefix="/pubmed")
app.include_router(answer_router, prefix="/pubmed")
