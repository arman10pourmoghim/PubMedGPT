# app/obs.py
from __future__ import annotations
import time, uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.metrics import metrics

class RequestObservability(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            return response
        finally:
            ms = (time.perf_counter() - start) * 1000
            path = request.url.path
            method = request.method
            metrics.inc(f"http.requests.total")
            metrics.observe_ms(f"http.latency.{method}.{path}", ms)
            # lightweight structured log to stdout (shows up in Uvicorn logs)
            print(f'{{"req_id":"{req_id}","method":"{method}","path":"{path}","ms":{ms:.1f}}}')
