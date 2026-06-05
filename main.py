# api/main.py
import os
from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routers import auth, chat, diagnosis, user_stats, inquiry, ads, investment, agent

app = FastAPI(title="ASCEND API", version="1.0.0")

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8502,https://ys-consulting-frontend-cj2fjmijla-an.a.run.app,https://ys-consulting-admin-cj2fjmijla-an.a.run.app,https://ys-consulting-frontend-665881683479.asia-northeast1.run.app,https://ys-consulting-admin-665881683479.asia-northeast1.run.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(diagnosis.router)
app.include_router(user_stats.router)
app.include_router(inquiry.router)
app.include_router(ads.router)
app.include_router(investment.router)
app.include_router(agent.router)

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    import json as _j
    try:
        _errs = exc.errors()
    except Exception:
        _errs = str(exc)
    try:
        _body = getattr(exc, "body", None)
        print("[RESPONSE_VALIDATION_ERROR]", _j.dumps(_errs, ensure_ascii=False, default=str)[:5000], flush=True)
        print("[RESPONSE_VALIDATION_BODY]", _j.dumps(_body, ensure_ascii=False, default=str)[:5000] if _body else "None", flush=True)
    except Exception as _pe:
        print("[RESPONSE_VALIDATION_ERROR_RAW]", repr(_errs), flush=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "ResponseValidationError", "errors": _errs if isinstance(_errs, list) else str(_errs)},
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "ASCEND API"}
