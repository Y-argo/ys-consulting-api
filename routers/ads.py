# api/routers/ads.py
import datetime
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from api.routers.auth import verify_token
from api.core.firestore_client import get_db

router = APIRouter(prefix="/api/ads", tags=["ads"])

@router.get("")
def get_ad(
    position: str = Query(..., description="sidebar or mypage"),
    _user=Depends(verify_token),
):
    tenant_id = _user.get("tenant_id", "default")
    db = get_db()

    # "both"登録の広告はsidebar/mypage両方で表示する
    positions = [position, "both"]

    docs = []
    for _pos in positions:
        docs += list(
            db.collection("ad_banners")
              .document(tenant_id)
              .collection("ads")
              .where("position", "==", _pos)
              .where("is_active", "==", True)
              .limit(10)
              .stream()
        )

    if not docs and tenant_id != "default":
        for _pos in positions:
            docs += list(
                db.collection("ad_banners")
                  .document("default")
                  .collection("ads")
                  .where("position", "==", _pos)
                  .where("is_active", "==", True)
                  .limit(10)
                  .stream()
            )

    if not docs:
        return {"ad": None}

    items = [d.to_dict() | {"id": d.id} for d in docs]
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return {"ad": items[0]}


class ClickRequest(BaseModel):
    ad_id: str
    tenant_id: str

@router.post("/click")
def record_click(
    req: ClickRequest,
    _user=Depends(verify_token),
):
    """クリックをFirestoreに記録する。"""
    db = get_db()
    now = datetime.datetime.utcnow()
    month_key = now.strftime("%Y-%m")

    ad_ref = (
        db.collection("ad_banners")
          .document(req.tenant_id)
          .collection("ads")
          .document(req.ad_id)
    )
    doc = ad_ref.get()
    if not doc.exists:
        return {"ok": False, "reason": "ad not found"}

    # 月別クリック数をインクリメント
    from google.cloud.firestore import Increment
    ad_ref.set(
        {
            "click_total": Increment(1),
            f"click_{month_key}": Increment(1),
            "click_last_at": now.isoformat(),
        },
        merge=True,
    )
    return {"ok": True}
