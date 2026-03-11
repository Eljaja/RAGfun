"""HTTP middleware for metrics and request tracking."""

import time

from fastapi import Request, Response

from app.metrics import HTTP_INFLIGHT, HTTP_LAT, HTTP_REQ_SIZE, HTTP_REQS, HTTP_RESP_SIZE


async def http_metrics_middleware(request: Request, call_next):
    """Middleware to track HTTP metrics."""
    # Avoid self-scrape noise
    path = request.url.path
    if path in ("/v1/metrics", "/metrics"):
        return await call_next(request)

    method = request.method
    route_obj = request.scope.get("route")
    route = getattr(route_obj, "path", None) or path

    def _cl(headers) -> int | None:
        try:
            v = headers.get("content-length")
            return int(v) if v is not None else None
        except Exception:
            return None

    req_size = _cl(request.headers)

    HTTP_INFLIGHT.labels(method=method, route=route).inc()
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        dur_s = max(0.0, time.perf_counter() - start)
        status = response.status_code if response is not None else 500
        HTTP_REQS.labels(method=method, route=route, status=str(status)).inc()
        HTTP_LAT.labels(method=method, route=route, status=str(status)).observe(dur_s)
        if req_size is not None:
            HTTP_REQ_SIZE.labels(method=method, route=route).observe(req_size)
        resp_size = _cl(response.headers) if response is not None else None
        if resp_size is not None:
            HTTP_RESP_SIZE.labels(method=method, route=route, status=str(status)).observe(resp_size)
        HTTP_INFLIGHT.labels(method=method, route=route).dec()

