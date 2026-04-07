# api/routers/auth.py
import os
import base64
import hashlib
import datetime
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

JWT_SECRET = os.environ.get("JWT_SECRET", os.environ.get("COOKIE_SECRET", "change_me"))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

from api.core.firestore_client import get_db

# ── app.py と同一の PBKDF2 検証 ──────────────────────────
def _verify_pw_pbkdf2(password: str, salt_b64: str, hash_b64: str, iters: int) -> bool:
    """app.py の verify_pw と完全一致"""
    try:
        salt = base64.urlsafe_b64decode(salt_b64 + "==")
        stored = base64.urlsafe_b64decode(hash_b64 + "==")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters), dklen=32)
        return dk == stored
    except Exception:
        return False

def _verify_user(uid: str, password: str) -> bool:
    db = get_db()
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return False
    data = doc.to_dict() or {}
    if not bool(data.get("is_active", True)):
        return False
    return _verify_pw_pbkdf2(
        password,
        data.get("pw_salt", ""),
        data.get("pw_hash", ""),
        int(data.get("pw_iters", 150_000) or 150_000),
    )

def _get_user_tenant(uid: str) -> str:
    try:
        db = get_db()
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return (doc.to_dict() or {}).get("tenant_id", "default")
    except Exception:
        pass
    return "default"

def _make_token(uid: str, role: str, tenant_id: str = "default") -> str:
    payload = {
        "uid": uid,
        "role": role,
        "tenant_id": tenant_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="認証が必要です")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="セッションが期限切れです")
    except Exception:
        raise HTTPException(status_code=401, detail="無効なトークンです")

# ── エンドポイント ─────────────────────────────────────────
class LoginRequest(BaseModel):
    uid: str
    password: str
    role: str = "user"

class LoginResponse(BaseModel):
    token: str
    uid: str
    role: str
    tenant_id: str

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    if req.role == "admin":
        if not ADMIN_PASSWORD or req.password != ADMIN_PASSWORD:
            raise HTTPException(status_code=403, detail="管理者パスワードが正しくありません")
        token = _make_token(uid="admin", role="admin", tenant_id="default")
        return LoginResponse(token=token, uid="admin", role="admin", tenant_id="default")
    else:
        if not _verify_user(req.uid, req.password):
            raise HTTPException(status_code=403, detail="UID またはパスワードが正しくありません")
        # 有効期限チェック
        try:
            from api.core.firestore_client import get_db as _gdb_auth
            import datetime as _dt_auth
            _usnap = _gdb_auth().collection("users").document(req.uid).get()
            _ud = _usnap.to_dict() if _usnap.exists else {}
            _is_unlimited = bool(_ud.get("is_unlimited", False))
            _expires_at = str(_ud.get("expires_at", "")).strip()
            if not _is_unlimited and _expires_at:
                _exp_date = _dt_auth.date.fromisoformat(_expires_at[:10])
                if _dt_auth.date.today() > _exp_date:
                    raise HTTPException(status_code=403, detail="EXPIRED")
        except HTTPException:
            raise
        except Exception:
            pass
        tenant_id = _get_user_tenant(req.uid)
        token = _make_token(uid=req.uid, role="user", tenant_id=tenant_id)
        return LoginResponse(token=token, uid=req.uid, role="user", tenant_id=tenant_id)

@router.post("/logout")
def logout():
    return {"ok": True}

@router.get("/me")
def me(payload: dict = Depends(verify_token)):
    return {"uid": payload["uid"], "role": payload["role"], "tenant_id": payload.get("tenant_id", "default")}

from api.core.features import get_effective_feature_flags

@router.get("/me/features")
def me_features(payload: dict = Depends(verify_token)):
    """ログイン中ユーザーの有効 feature フラグを返す。admin は全 True。"""
    uid  = payload["uid"]
    role = payload.get("role", "user")
    if role == "admin":
        from api.core.features import FEATURE_REGISTRY
        flags = {fid: True for fid in FEATURE_REGISTRY}
    else:
        flags = get_effective_feature_flags(uid)
    return {"uid": uid, "role": role, "features": flags}

import secrets
import hashlib as _hashlib
import base64 as _base64

def _b64e(b: bytes) -> str:
    return _base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _make_pw_hash(password: str):
    salt = secrets.token_bytes(16)
    iters = 150_000
    dk = _hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    return _b64e(salt), _b64e(dk), iters

class RegisterRequest(BaseModel):
    uid: str
    password: str
    display_name: str = ""

@router.post("/register")
def register(req: RegisterRequest):
    db = get_db()
    uid = req.uid.strip()
    if not uid or len(uid) < 2:
        raise HTTPException(status_code=400, detail="UIDは2文字以上で入力してください")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="パスワードは6文字以上で入力してください")
    doc = db.collection("users").document(uid).get()
    if doc.exists:
        raise HTTPException(status_code=409, detail="このUIDは既に使用されています")
    salt_b64, hash_b64, iters = _make_pw_hash(req.password)
    from google.cloud import firestore as _fs
    db.collection("users").document(uid).set({
        "uid": uid,
        "tenant_id": "default",
        "pw_salt": salt_b64,
        "pw_hash": hash_b64,
        "pw_iters": iters,
        "is_active": True,
        "display_name": req.display_name or uid,
        "created_at": _fs.SERVER_TIMESTAMP,
        "updated_at": _fs.SERVER_TIMESTAMP,
        "level_score": 0,
        "use_count_since_report": 0,
        "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
    })
    token = _make_token(uid=uid, role="user", tenant_id="default")
    return LoginResponse(token=token, uid=uid, role="user", tenant_id="default")
