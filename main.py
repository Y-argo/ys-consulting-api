# api/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import auth, chat, diagnosis, user_stats, inquiry, ads

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

@app.get("/health")
def health():
    return {"status": "ok", "service": "ASCEND API"}
