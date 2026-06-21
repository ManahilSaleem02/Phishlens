"""
AI-Powered URL Phishing Detection — FastAPI backend.

Endpoints
  GET  /health          liveness + model metadata
  POST /api/predict     {"url": "..."} -> structured explainable prediction
  GET  /api/history     recent scans
  GET  /api/stats       analytics for the dashboard
  GET  /                serves the dashboard (frontend/index.html)

Includes a simple in-memory token-bucket rate limiter and request logging,
so it has no extra dependencies beyond FastAPI/uvicorn.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import predictor          # noqa: E402
from backend import database         # noqa: E402
from backend import auth             # noqa: E402

# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(
                  os.path.dirname(os.path.abspath(__file__)), "predictions.log"))],
)
log = logging.getLogger("phishing-api")

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "frontend", "index.html")
LOGIN_PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "frontend", "login.html")
EVALUATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "model", "evaluation.json")

app = FastAPI(title="AI-Powered URL Phishing Detection", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# --------------------------- rate limiting ----------------------------------
RATE_LIMIT = 60          # requests
RATE_WINDOW = 60         # seconds
_hits: dict[str, deque] = defaultdict(deque)

# Paths reachable without a session. Everything else requires login.
PUBLIC_PATHS = {"/login", "/api/login", "/health"}


def _is_authenticated(request: Request) -> bool:
    return auth.verify_session(request.cookies.get(auth.COOKIE_NAME)) is not None


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    path = request.url.path

    # ---- authentication gate (runs before rate limiting / handlers) --------
    if path not in PUBLIC_PATHS and not path.startswith("/docs") \
            and not path.startswith("/openapi") and not _is_authenticated(request):
        if path.startswith("/api/"):
            return JSONResponse(status_code=401,
                                content={"detail": "Authentication required. Please log in."})
        # Browser navigation: send them to the login page.
        return RedirectResponse(url="/login", status_code=303)

    if path.startswith("/api/"):
        q = _hits[ip]
        while q and q[0] <= now - RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            log.warning(f"rate-limit {ip} on {path}")
            return JSONResponse(status_code=429,
                                content={"detail": "Rate limit exceeded. Try again shortly."})
        q.append(now)
    start = time.time()
    resp = await call_next(request)
    log.info(f"{ip} {request.method} {path} -> {resp.status_code} "
             f"({(time.time()-start)*1000:.0f}ms)")
    return resp


# ------------------------------- models -------------------------------------
class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["http://paypal-secure-login.tk/verify"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


# ------------------------------ lifecycle -----------------------------------
@app.on_event("startup")
def _startup():
    database.init_db()
    info = predictor.model_info()
    log.info(f"Model loaded. Trained {info.get('trained_at')} | "
             f"metrics={info.get('metrics')}")


# --------------------------------- auth -------------------------------------
@app.get("/login")
def login_page(request: Request):
    # Already signed in? Skip the form.
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    if os.path.exists(LOGIN_PAGE):
        return FileResponse(LOGIN_PAGE, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        })
    raise HTTPException(status_code=404, detail="Login page not found.")


@app.post("/api/login")
def api_login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not auth.verify_credentials(req.username, req.password):
        log.warning(f"failed login for '{req.username}' from {ip}")
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth.create_session(req.username)
    resp = JSONResponse(content={"status": "ok", "username": req.username})
    resp.set_cookie(
        key=auth.COOKIE_NAME, value=token,
        max_age=auth.SESSION_TTL, httponly=True, samesite="lax", path="/",
    )
    log.info(f"login ok for '{req.username}' from {ip}")
    return resp


@app.post("/api/logout")
def api_logout():
    resp = JSONResponse(content={"status": "ok"})
    resp.delete_cookie(key=auth.COOKIE_NAME, path="/")
    return resp


@app.get("/api/me")
def api_me(request: Request):
    user = auth.verify_session(request.cookies.get(auth.COOKIE_NAME))
    return {"username": user}


# ------------------------------ endpoints -----------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_info()}


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    try:
        result = predictor.predict(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:                       # pragma: no cover
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    database.record(result["url"], result["prediction"],
                    result["risk_score"], result["registered_domain"])
    return result


@app.get("/api/history")
def api_history(limit: int = 50):
    return {"history": database.history(min(max(limit, 1), 200))}


@app.get("/api/stats")
def api_stats():
    return database.stats()


@app.get("/api/evaluation")
def evaluation():
    """Plain-language model comparison for the dashboard's performance tab.
    Generated offline by `python -m model.evaluate`."""
    if os.path.exists(EVALUATION):
        with open(EVALUATION) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Evaluation not generated yet.")


@app.get("/")
def index():
    if os.path.exists(FRONTEND):
        # no-store so the browser always loads the latest dashboard build
        # (prevents a cached index.html from showing old behaviour after updates)
        return FileResponse(FRONTEND, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        })
    return {"message": "Backend running. Frontend not found.",
            "docs": "/docs", "health": "/health"}
