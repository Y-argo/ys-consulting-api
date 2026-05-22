# api/routers/inquiry.py
import uuid
import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from google.cloud import firestore as fs
from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT

router = APIRouter(prefix="/api/inquiry", tags=["inquiry"])

def _write_notification(uid: str, notif_type: str, title: str, body: str, link_tab: str = "overview"):
    """Firestoreに通知レコードを書き込む共通関数"""
    try:
        import uuid as _uuid
        import datetime as _dt
        db = get_db()
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() if snap.exists else {}
        settings = d.get("notification_settings", {})
        type_key_map = {
            "reply": "notify_reply",
            "rankup": "notify_rankup",
            "fc": "notify_fc",
            "inquiry": "notify_inquiry",
            "priority_action": "notify_reply",
        }
        setting_key = type_key_map.get(notif_type, "notify_reply")
        if settings.get(setting_key, True) is False:
            return
        notif_id = _uuid.uuid4().hex
        db.collection("notifications").document(uid).collection("items").document(notif_id).set({
            "notif_id": notif_id,
            "uid": uid,
            "type": notif_type,
            "title": title,
            "body": body,
            "link_tab": link_tab,
            "read": False,
            "created_at": (_dt.datetime.utcnow() + _dt.timedelta(hours=9)).isoformat(),
        })
    except Exception as e:
        print(f"[NOTIFICATION_ERROR] {e}", flush=True)

COL_INQUIRIES = "consulting_inquiries"
COL_INQ_MESSAGES = "consulting_inquiry_messages"

CATEGORY_OPTIONS = ["戦略・方針相談","売上・マーケティング","組織・人材","財務・資金調達","オペレーション改善","その他"]
INQUIRY_STATUSES = {"new":"未対応","in_progress":"対応中","replied":"返信済み","waiting_user":"返信待ち","closed":"完了"}

class CreateInquiryRequest(BaseModel):
    title: str
    body: str
    category: str = ""
    supplement: str = ""

class AddMessageRequest(BaseModel):
    inquiry_id: str
    body: str

@router.get("/list")
def list_inquiries(payload: dict = Depends(verify_token)):
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    if not is_feature_enabled(uid, "personal_consulting"):
        raise HTTPException(status_code=403, detail="個人相談は現在未開放のため使用できません。")
    db = get_db()
    try:
        docs = list(
            db.collection(COL_INQUIRIES)
            .where("user_id", "==", uid)
            .limit(50)
            .stream()
        )
        result = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_deleted"): continue
            result.append({
                "inquiry_id": data.get("inquiry_id", d.id),
                "title": data.get("title", ""),
                "category": data.get("category", ""),
                "status": data.get("status", "new"),
                "status_label": INQUIRY_STATUSES.get(data.get("status","new"),"未対応"),
                "created_at": str(data.get("created_at", "")),
                "updated_at": str(data.get("updated_at", "")),
                "unread_for_user": bool(data.get("unread_for_user", False)),
            })
        result.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"inquiries": result}
    except Exception as e:
        return {"inquiries": []}

@router.get("/messages/{inquiry_id}")
def get_messages(inquiry_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    inq = db.collection(COL_INQUIRIES).document(inquiry_id).get()
    if not inq.exists or (inq.to_dict() or {}).get("user_id") != uid:
        raise HTTPException(status_code=403, detail="アクセス権限がありません")
    try:
        docs = list(
            db.collection(COL_INQ_MESSAGES)
            .where("inquiry_id", "==", inquiry_id)
            .stream()
        )
        docs.sort(key=lambda d: str((d.to_dict() or {}).get("created_at", "")))
        msgs = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_deleted"): continue
            if not data.get("visible_to_user", True): continue
            msgs.append({
                "message_id": data.get("message_id", d.id),
                "sender_type": data.get("sender_type", "user"),
                "body": data.get("body", ""),
                "created_at": str(data.get("created_at", "")),
            })
        db.collection(COL_INQUIRIES).document(inquiry_id).update({"unread_for_user": False})
        return {"messages": msgs}
    except Exception as e:
        return {"messages": []}

@router.post("/create")
def create_inquiry(req: CreateInquiryRequest, payload: dict = Depends(verify_token)):
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    if not is_feature_enabled(uid, "personal_consulting"):
        raise HTTPException(status_code=403, detail="個人相談は現在未開放のため使用できません。")
    db = get_db()
    inquiry_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    try:
        snap = db.collection("users").document(uid).get()
        display_name = (snap.to_dict() or {}).get("display_name", uid) if snap.exists else uid
    except Exception:
        display_name = uid
    db.collection(COL_INQUIRIES).document(inquiry_id).set({
        "inquiry_id": inquiry_id,
        "tenant_id": tenant_id,
        "user_id": uid,
        "user_display_name": display_name,
        "title": req.title.strip(),
        "category": req.category,
        "supplement": req.supplement.strip(),
        "status": "new",
        "created_at": now,
        "updated_at": now,
        "last_message_at": now,
        "last_sender_type": "user",
        "unread_for_admin": True,
        "unread_for_user": False,
        "is_deleted": False,
    })
    message_id = str(uuid.uuid4())
    db.collection(COL_INQ_MESSAGES).document(message_id).set({
        "message_id": message_id,
        "inquiry_id": inquiry_id,
        "tenant_id": tenant_id,
        "sender_type": "user",
        "sender_id": uid,
        "body": req.body.strip(),
        "created_at": now,
        "visible_to_user": True,
        "visible_to_admin": True,
        "is_deleted": False,
    })
    return {"inquiry_id": inquiry_id, "ok": True}

@router.post("/message")
def add_message(req: AddMessageRequest, payload: dict = Depends(verify_token)):
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    if not is_feature_enabled(uid, "personal_consulting"):
        raise HTTPException(status_code=403, detail="個人相談は現在未開放のため使用できません。")
    db = get_db()
    inq = db.collection(COL_INQUIRIES).document(req.inquiry_id).get()
    if not inq.exists or (inq.to_dict() or {}).get("user_id") != uid:
        raise HTTPException(status_code=403, detail="アクセス権限がありません")
    message_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    db.collection(COL_INQ_MESSAGES).document(message_id).set({
        "message_id": message_id,
        "inquiry_id": req.inquiry_id,
        "tenant_id": tenant_id,
        "sender_type": "user",
        "sender_id": uid,
        "body": req.body.strip(),
        "created_at": now,
        "visible_to_user": True,
        "visible_to_admin": True,
        "is_deleted": False,
    })
    db.collection(COL_INQUIRIES).document(req.inquiry_id).update({
        "last_message_at": now,
        "updated_at": now,
        "last_sender_type": "user",
        "unread_for_admin": True,
        "unread_for_user": False,
        "status": "in_progress",
    })
    return {"message_id": message_id, "ok": True}

class AdminReplyRequest(BaseModel):
    inquiry_id: str
    body: str

@router.patch("/read/{inquiry_id}")
def mark_read_for_user(inquiry_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        db.collection("consulting_inquiries").document(inquiry_id).set(
            {"unread_for_user": False}, merge=True)
    except Exception:
        pass
    return {"ok": True}

@router.patch("/admin/read/{inquiry_id}")
def mark_read_for_admin(inquiry_id: str, payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    try:
        db.collection("consulting_inquiries").document(inquiry_id).set(
            {"unread_for_admin": False}, merge=True)
    except Exception:
        pass
    return {"ok": True}

@router.patch("/admin/status/{inquiry_id}")
def update_inquiry_status(inquiry_id: str, body: dict, payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    STATUS_MAP = {"open":"対応中","in_progress":"対応中","resolved":"解決済み","closed":"クローズ"}
    new_status = body.get("status","open")
    try:
        db.collection("consulting_inquiries").document(inquiry_id).set({
            "status": new_status,
            "status_label": STATUS_MAP.get(new_status, new_status),
            "updated_at": datetime.datetime.utcnow(),
        }, merge=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}

@router.patch("/admin/memo/{inquiry_id}")
def update_admin_memo(inquiry_id: str, body: dict, payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    try:
        db.collection("consulting_inquiries").document(inquiry_id).set(
            {"admin_memo": body.get("memo","")}, merge=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}

@router.get("/admin/list")
def admin_list_inquiries(payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        docs = list(
            db.collection("consulting_inquiries")
            .where("tenant_id", "==", tenant_id)
            .limit(100)
            .stream()
        )
        result = [d.to_dict() for d in docs]
        result.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        return {"inquiries": result}
    except Exception:
        return {"inquiries": []}

@router.post("/admin/reply")
def admin_reply(req: AdminReplyRequest, payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    inq = db.collection(COL_INQUIRIES).document(req.inquiry_id).get()
    if not inq.exists:
        raise HTTPException(status_code=404, detail="相談が見つかりません")
    tenant_id = (inq.to_dict() or {}).get("tenant_id", DEFAULT_TENANT)
    message_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()
    db.collection(COL_INQ_MESSAGES).document(message_id).set({
        "message_id": message_id,
        "inquiry_id": req.inquiry_id,
        "tenant_id": tenant_id,
        "sender_type": "admin",
        "sender_id": "admin",
        "body": req.body.strip(),
        "created_at": now,
        "visible_to_user": True,
        "visible_to_admin": True,
        "unread_for_user": True,
        "unread_for_admin": False,
        "is_deleted": False,
    })
    db.collection(COL_INQUIRIES).document(req.inquiry_id).update({
        "last_message_at": now,
        "updated_at": now,
        "last_sender_type": "admin",
        "unread_for_user": True,
        "unread_for_admin": False,
        "status": "replied",
    })
    # 個人相談返信通知→ユーザーに通知
    try:
        inq_d = inq.to_dict() or {}
        target_uid = inq_d.get("user_id", "")
        if target_uid:
            _write_notification(
                uid=target_uid,
                notif_type="inquiry",
                title="📩 個人相談に返信が届きました",
                body=req.body.strip()[:60] + "..." if len(req.body.strip()) > 60 else req.body.strip(),
                link_tab="dm",
            )
    except Exception:
        pass
    return {"message_id": message_id, "ok": True}
