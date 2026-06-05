# api/routers/agent.py
import uuid
import datetime
from fastapi import APIRouter, HTTPException, Depends, Header, Body
from pydantic import BaseModel, Field
from typing import Optional
import os

from api.core.firestore_client import get_db
from google.cloud import firestore
from api.core.agent_executor import execute_agent_task
from croniter import croniter
from api.routers.auth import verify_token

router = APIRouter(prefix="/api/agent", tags=["agent"])

APEX_PLANS = {"APEX", "ULTRA", "ultra_admin", "ultra_member"}
BLOCKED_PLANS = {"STARTER", "STANDARD"}
AGENT_TYPES = {"hp_update", "audit", "interview"}
OPERATION_TYPES = {
    "entity_register", "entity_update", "media_replace",
    "text_update", "schedule_update", "price_update",
    "news_post", "status_update",
}
INDUSTRY_TEMPLATES = {
    "nightlife":  {"entity_name": "キャスト",     "schedule": "出勤",         "news": "ニュース",     "media": "写真"},
    "beauty":     {"entity_name": "スタッフ",     "schedule": "予約枠",       "news": "キャンペーン", "media": "スタッフ写真"},
    "retail":     {"entity_name": "商品",         "schedule": "営業時間",     "news": "お知らせ",     "media": "商品写真"},
    "realestate": {"entity_name": "物件",         "schedule": "空室状況",     "news": "新着物件",     "media": "物件写真"},
    "btob":       {"entity_name": "サービス",     "schedule": "セミナー",     "news": "ニュース",     "media": "資料"},
    "fitness":    {"entity_name": "講師",         "schedule": "レッスン",     "news": "キャンペーン", "media": "講師写真"},
    "other":      {"entity_name": "エンティティ", "schedule": "スケジュール", "news": "お知らせ",     "media": "メディア"},
}

def _resolve_agent_user_context(user: dict) -> dict:
    """userdictからtenant_id/role/plansを正規化して返す補助関数"""
    uid = user.get("uid") or user.get("user_id") or user.get("sub") or ""
    tenant_id = user.get("tenant_id") or uid or "default"
    role = str(user.get("role", "") or "").lower()
    plans = {
        str(user.get("plan", "") or "").upper(),
        str(user.get("subscription_plan", "") or "").upper(),
        str(user.get("ai_tier", "") or "").upper(),
        str(user.get("tier", "") or "").upper(),
        role.upper(),
    }
    is_admin = (role == "admin")
    is_unlimited = bool(user.get("is_unlimited", False))
    return {
        "uid": uid,
        "tenant_id": tenant_id,
        "role": role,
        "plans": plans,
        "is_admin": is_admin,
        "is_unlimited": is_unlimited,
    }


def _get_agent_permissions(tenant_id: str) -> dict:
    try:
        db = get_db()
        doc = db.collection("agent_permissions").document(tenant_id).get()
        if doc.exists:
            return doc.to_dict() or {}
    except Exception as e:
        print(f"[agent_permissions] fetch error: {type(e).__name__}", flush=True)
    return {"admin_granted": False, "allowed_agents": [], "allowed_operations": [], "max_tasks_per_day": 0}


def _enforce_agent_permissions(ctx: dict, agent_type: str, operation_type: str):
    if ctx["is_admin"] or ctx["is_unlimited"]:
        return
    if any(p in APEX_PLANS for p in ctx["plans"]):
        return
    tenant_id = ctx["tenant_id"]
    perm = _get_agent_permissions(tenant_id)
    if not perm.get("admin_granted", False):
        raise HTTPException(status_code=403, detail="管理者によるエージェント利用許可がありません")
    allowed_agents = perm.get("allowed_agents") or []
    if allowed_agents and not agent_type:
        raise HTTPException(status_code=403, detail="agent_typeが未確定のため権限検査できません")
    if allowed_agents and agent_type and agent_type not in allowed_agents:
        raise HTTPException(status_code=403, detail=f"agent_type '{agent_type}' はこのテナントで許可されていません。許可済み: {allowed_agents}")
    allowed_ops = perm.get("allowed_operations") or []
    if allowed_ops and not operation_type:
        raise HTTPException(status_code=403, detail="operation_typeが未確定のため権限検査できません")
    if allowed_ops and operation_type and operation_type not in allowed_ops:
        raise HTTPException(status_code=403, detail=f"operation_type '{operation_type}' はこのテナントで許可されていません。許可済み: {allowed_ops}")
    max_tasks = perm.get("max_tasks_per_day", 0)
    if max_tasks and max_tasks > 0:
        try:
            import datetime as _dt
            db = get_db()
            today_start = _dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            count_docs = db.collection("agent_tasks").where("tenant_id", "==", tenant_id).where("created_at", ">=", today_start).stream()
            task_count = sum(1 for _ in count_docs)
            if task_count >= max_tasks:
                raise HTTPException(status_code=403, detail=f"本日のタスク上限({max_tasks}件)に達しています。現在: {task_count}件")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[enforce_permissions] task count error: {type(e).__name__}", flush=True)

def _check_agent_access(user: dict) -> bool:
    uid = user.get("uid") or user.get("user_id") or user.get("sub")
    tenant_id = user.get("tenant_id", "default")

    role = str(user.get("role", "") or "").lower()
    plans = {
        str(user.get("plan", "") or "").upper(),
        str(user.get("subscription_plan", "") or "").upper(),
        str(user.get("ai_tier", "") or "").upper(),
        str(user.get("tier", "") or "").upper(),
        role.upper(),
    }

    if role == "admin":
        return True

    try:
        db = get_db()

        # tokenだけでなくFirestore users/{uid} も見る
        if uid:
            udoc = db.collection("users").document(uid).get()
            if udoc.exists:
                u = udoc.to_dict() or {}
                urole = str(u.get("role", "") or "").lower()
                uplans = {
                    str(u.get("plan", "") or "").upper(),
                    str(u.get("subscription_plan", "") or "").upper(),
                    str(u.get("ai_tier", "") or "").upper(),
                    str(u.get("tier", "") or "").upper(),
                    urole.upper(),
                }

                if urole == "admin":
                    return True
                if bool(u.get("is_unlimited", False)):
                    return True
                if any(x in APEX_PLANS for x in uplans):
                    return True

                tenant_id = u.get("tenant_id", tenant_id)

        if any(x in APEX_PLANS for x in plans):
            return True

        if any(x in BLOCKED_PLANS for x in plans):
            return False

        perm = db.collection("agent_permissions").document(tenant_id).get()
        if perm.exists:
            return bool((perm.to_dict() or {}).get("admin_granted", False))

    except Exception as e:
        print(f"[agent_access] permission check error: {type(e).__name__}", flush=True)

    return False

# P2: risk_level判定マップ
_RISK_MAP = {
    "entity_register": "medium",
    "entity_update":   "medium",
    "media_replace":   "high",
    "text_update":     "low",
    "schedule_update": "low",
    "price_update":    "high",
    "news_post":       "medium",
    "status_update":   "medium",
}

def _build_preview(agent_type: str, operation_type: str, industry: str, payload: dict, operation_steps: list = None) -> dict:
    tmpl = INDUSTRY_TEMPLATES.get(industry, INDUSTRY_TEMPLATES["other"])
    affected = list(payload.keys())
    diff = [{"field": k, "before": "unknown", "after": v} for k, v in payload.items()]
    preview = {
        "agent_type":      agent_type,
        "operation_type":  operation_type,
        "industry":        industry,
        "entity_label":    tmpl["entity_name"],
        "summary":         tmpl["entity_name"] + "の" + operation_type + "を実行します",
        "payload_preview": payload,
        "before":          {},
        "after":           payload,
        "diff":            diff,
        "affected_fields": affected,
        "risk_level":      _RISK_MAP.get(operation_type, "low"),
    }
    # P14: operation_graphプレビュー
    if operation_steps:
        preview["operation_graph"] = True
        preview["step_count"]      = len(operation_steps)
        preview["steps_preview"]   = [{"step_id": s.get("step_id"), "step_type": s.get("step_type"), "order": s.get("order")} for s in operation_steps]
    return preview

def _save_log(
    task: dict,
    success: bool,
    before: dict,
    after: dict,
    error: str = "",
    execution_time_ms: int = 0,
    verification: dict = None,
    rollback: dict = None,
    selector_repair: dict = None,
    result: dict = None,
):
    """P9: 実行学習ログ。agent_logsに拡張フィールドを保存。"""
    try:
        from api.core.browser_executor import GENERIC_OPERATION_CONFIG
        db = get_db()
        log_id = str(uuid.uuid4())
        operation_type = task.get("operation_type", "")
        _retry_count = task.get("retry_count", 0)
        _verification = verification or {}
        _v_method = _verification.get("method", "")
        _v_verified = _verification.get("verified", False)
        _cap = GENERIC_OPERATION_CONFIG.get(operation_type, {}).get("capability_key", "")
        # selector_success_rate: verified=Trueかつmethod=selectorなら1.0、falseなら0.0、その他0.5
        if _v_method == "selector":
            _sel_rate = 1.0 if _v_verified else 0.0
        elif _v_method in ("url", "diff", "submit_disabled"):
            _sel_rate = 0.5
        else:
            _sel_rate = None
        db.collection("agent_logs").document(log_id).set({
            "log_id":               log_id,
            "task_id":              task.get("task_id", ""),
            "tenant_id":            task.get("tenant_id", ""),
            "operator_uid":         task.get("user_uid", ""),
            "agent_type":           task.get("agent_type", ""),
            "operation_type":       operation_type,
            "before_state":         before,
            "after_state":          after,
            "success":              success,
            "error_message":        error,
            "executed_at":          datetime.datetime.utcnow(),
            # P9: 学習ログ追加フィールド
            "execution_time_ms":     execution_time_ms,
            "retry_count":           _retry_count,
            "verification_method":   _v_method,
            "selector_success_rate": _sel_rate,
            "capability_used":       _cap,
            # P7-rollback: rollbackログ
            "rollback_attempted":    (rollback or {}).get("attempted", False),
            "rollback_success":      (rollback or {}).get("success", False),
            "rollback_failed_fields":(rollback or {}).get("failed_fields", []),
            "selector_repair_suggested": bool(selector_repair and selector_repair.get("suggested")),
            "selector_repair_count":     len(selector_repair.get("suggested_selectors", [])) if selector_repair else 0,
            # P13: self-healログ
            "self_heal_attempted":       bool((result or {}).get("self_heal", {}).get("attempted")),
            "self_heal_success":         bool((result or {}).get("self_heal", {}).get("success")),
            "self_heal_retry_succeeded": bool((result or {}).get("self_heal", {}).get("retry_succeeded")),
            # P16-7: selector_rankingログ
            "selector_rank_used":   bool((result or {}).get("self_heal", {}).get("selector_rank_used")),
            "selector_rank_score":  (result or {}).get("self_heal", {}).get("selector_rank_score"),
            "selector_rank_source": (result or {}).get("self_heal", {}).get("selector_rank_source", ""),
        })
    except Exception as e:
        print("[agent_logs] write error: " + str(e), flush=True)

class TaskCreateRequest(BaseModel):
    agent_type: str
    operation_type: str
    industry: str = "generic"
    entity_type: Optional[str] = None
    op_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    scheduled_at: Optional[str] = None

class TaskApproveRequest(BaseModel):
    task_id: str

class TaskRejectRequest(BaseModel):
    task_id: str
    reason: Optional[str] = None

@router.post("/task/create")
def create_task(req: TaskCreateRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="エージェントモードの利用権限がありません")

    db = get_db()

    # op_id指定時はFirestore op_dataを正として厳格検証
    if req.op_id:
        op_doc = db.collection("agent_ops").document(req.op_id).get()
        if not op_doc.exists:
            raise HTTPException(status_code=400, detail="存在しないop_idです")
        op_data = op_doc.to_dict()

        # active確認
        if not op_data.get("active", True):
            raise HTTPException(status_code=400, detail="このOperationは現在利用できません")

        # operation_type / entity_type 必須確認
        if not op_data.get("operation_type"):
            raise HTTPException(status_code=400, detail="このOperationはoperation_typeが未定義のため作成できません")
        if not op_data.get("entity_type"):
            raise HTTPException(status_code=400, detail="このOperationはentity_typeが未定義のため作成できません")

        # allowed_plans確認
        ctx = _resolve_agent_user_context(user)
        user_role = ctx["role"]
        user_plans = {p.upper() for p in ctx["plans"]}
        allowed = {p.upper() for p in op_data.get("allowed_plans", [])}
        if not ctx["is_admin"] and not user_plans.intersection(allowed):
            raise HTTPException(status_code=403, detail="このOperationを利用する権限がありません")

        # op_dataからフィールドを正として補完（req側は信用しない）
        agent_type     = op_data.get("category", "hp_update")
        operation_type = op_data["operation_type"]
        entity_type    = op_data["entity_type"]
        industry       = op_data.get("industry") or "generic"

        # payload_schema.fields の required検証 + 余計なキー除外
        schema_fields = op_data.get("payload_schema", {}).get("fields", [])
        allowed_keys = {f["key"] for f in schema_fields}
        required_keys = {f["key"] for f in schema_fields if f.get("required")}

        # required_fields フォールバック
        for rk in op_data.get("required_fields", []):
            required_keys.add(rk)

        missing = [k for k in required_keys if not req.payload.get(k)]
        if missing:
            raise HTTPException(status_code=400, detail=f"必須項目が不足しています: {', '.join(missing)}")

        # 余計なキーを除外（allowed_keysが空の場合は全許可）
        if allowed_keys:
            clean_payload = {k: v for k, v in req.payload.items() if k in allowed_keys}
        else:
            clean_payload = dict(req.payload)

        # op_snapshot
        op_snapshot = {
            "op_id": req.op_id,
            "display_name": op_data.get("display_name") or op_data.get("name") or req.op_id,
            "category": agent_type,
            "operation_type": operation_type,
            "entity_type": entity_type,
            "industry": industry,
            "payload_schema_version": op_data.get("payload_schema_version", "1"),
        }

    else:
        # op_id未指定時は従来の固定チェック
        if req.agent_type not in AGENT_TYPES:
            raise HTTPException(status_code=400, detail="無効なagent_typeです")
        if req.operation_type not in OPERATION_TYPES:
            raise HTTPException(status_code=400, detail="無効なoperation_typeです")
        agent_type     = req.agent_type
        operation_type = req.operation_type
        entity_type    = req.entity_type or ""
        industry       = req.industry or "generic"
        clean_payload  = dict(req.payload)
        op_snapshot    = {}

    # P0-1: operation_type / agent_type 確定後に enforcement（op_id解決後）
    _ctx_create = _resolve_agent_user_context(user)
    _enforce_agent_permissions(_ctx_create, agent_type, operation_type)

    # P14: operation_steps_templateをpayloadで展開
    _op_steps_template = op_data.get("operation_steps_template") if 'op_data' in locals() else None
    _operation_steps = _op_steps_template if _op_steps_template else None
    task_id    = str(uuid.uuid4())
    # P17: workflow_id生成（複数task chainなら共有可）
    workflow_id = str(uuid.uuid4())
    preview = _build_preview(agent_type, operation_type, industry, clean_payload, operation_steps=_operation_steps)
    _tenant_id_create = _resolve_agent_user_context(user)["tenant_id"]
    task = {
        "task_id": task_id,
        "tenant_id": _tenant_id_create,
        "user_uid": user.get("uid", ""),
        "agent_type": agent_type,
        "operation_type": operation_type,
        "industry": industry,
        "entity_type": entity_type,
        "op_id": req.op_id or "",
        "op_snapshot": op_snapshot,
        "status": "PENDING",
        "payload": clean_payload,
        "preview": preview,
        "operation_steps": _operation_steps,
        "approved_by": None,
        "approved_at": None,
        "scheduled_at": req.scheduled_at,
        "result": None,
        "created_at": datetime.datetime.utcnow(),
        # P17: operation chain memory schema拡張
        "workflow_id":             workflow_id,
        "chain_id":                "",
        "parent_task_id":          "",
        "depends_on":              [],
        "previous_operation":      "",
        "next_operation_candidates": [],
    }
    db.collection("agent_tasks").document(task_id).set(task)

    # P17: 類似workflow提案（自動実行禁止・提案のみ）
    _recommended_workflows = []
    try:
        from api.core.browser_executor import find_similar_workflows
        _media_name_p17 = clean_payload.get("media_name", "")
        _recommended_workflows = find_similar_workflows(
            db=db,
            tenant_id=_tenant_id_create,
            operation_type=operation_type,
            media_name=_media_name_p17,
            top_n=3,
        )
    except Exception as _p17_e:
        print(f"[P17 find_similar_workflows] エラー: {type(_p17_e).__name__}", flush=True)


    # P20: risk estimation + workflow session作成（自動実行禁止）
    _p20_risk       = {}
    _p20_session_id = ""
    try:
        from api.core.browser_executor import (
            estimate_workflow_risk,
            create_workflow_session,
        )
        _p20_mm = {}
        _p20_media_name = clean_payload.get("media_name", "")
        if _p20_media_name:
            _p20_docs = db.collection("media_mappings").where("tenant_id", "==", _tenant_id_create).stream()
            for _p20_d in _p20_docs:
                _p20_dm = _p20_d.to_dict()
                if _p20_dm.get("media_name") == _p20_media_name:
                    _p20_mm = _p20_dm
                    break
        _p20_risk = estimate_workflow_risk(
            db=db,
            tenant_id=_tenant_id_create,
            operation_type=operation_type,
            media_family=_p20_mm.get("media_family", ""),
            operation_steps=_operation_steps or [],
            media_mapping=_p20_mm,
        )
        _p20_policy = {
            "max_retry":                  2,
            "allow_self_heal":            True,
            "allow_replan":               True,
            "require_human_on_high_risk": True,
            "interruptible":              True,
        }
        _p20_session_id = create_workflow_session(
            db=db,
            tenant_id=_tenant_id_create,
            workflow_id=workflow_id,
            goal=f"{operation_type} for {_p20_media_name}",
            operation_type=operation_type,
            operation_steps=_operation_steps or [],
            execution_policy=_p20_policy,
            risk_estimation=_p20_risk,
        )
    except Exception as _p20_e:
        print(f"[P20 workflow_session] 作成エラー: {type(_p20_e).__name__}", flush=True)

    return {
        "task_id":                task_id,
        "status":                 "PENDING",
        "preview":                preview,
        "workflow_id":            workflow_id,
        "recommended_workflows":  _recommended_workflows,
        "workflow_session_id":    _p20_session_id,
        "risk_level":             _p20_risk.get("risk_level", ""),
        "risk_score":             _p20_risk.get("risk_score", 0.0),
        "risk_factors":           _p20_risk.get("risk_factors", []),
        "require_human_approval": _p20_risk.get("require_human_approval", False),
    }

@router.post("/task/approve")
def approve_task(req: TaskApproveRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_tasks").document(req.task_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict()
    if task.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="PENDING状態のタスクのみ承認できます。現在: " + str(task.get("status")))
    ref.update({
        "status": "APPROVED",
        "approved_by": user.get("uid", ""),
        "approved_at": datetime.datetime.utcnow(),
    })
    return {"task_id": req.task_id, "status": "APPROVED"}

@router.post("/task/reject")
def reject_task(req: TaskRejectRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_tasks").document(req.task_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict()
    if task.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="PENDING状態のタスクのみ却下できます")
    ref.update({
        "status": "REJECTED",
        "result": {"reason": req.reason or ""},
        "approved_by": user.get("uid", ""),
        "approved_at": datetime.datetime.utcnow(),
    })
    return {"task_id": req.task_id, "status": "REJECTED"}

@router.post("/task/execute/{task_id}")
def execute_task(task_id: str, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_tasks").document(task_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict()
    # tenant_id所有者チェック
    user_tenant_id = user.get("tenant_id") or user.get("uid") or "default"
    if task.get("tenant_id") != user_tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このタスクを実行する権限がありません")
    if task.get("status") != "APPROVED":
        raise HTTPException(status_code=400, detail="APPROVEDのタスクのみ実行できます")
    # P0-1: 実行直前の二重防御
    _ctx_exec = _resolve_agent_user_context(user)
    _enforce_agent_permissions(_ctx_exec, task.get("agent_type", ""), task.get("operation_type", ""))
    # 二重実行防止：RUNNINGへの遷移をトランザクションで実施
    transaction = db.transaction()

    @firestore.transactional
    def _set_running(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise ValueError("タスクが見つかりません")
        t = snap.to_dict()
        if t.get("status") != "APPROVED":
            raise ValueError("既に実行中または完了済みです: " + str(t.get("status")))
        transaction.update(ref, {"status": "RUNNING"})

    try:
        _set_running(transaction, ref)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    try:
        before_state = {"status": "APPROVED", "payload": task.get("payload", {}), "operation_type": task.get("operation_type", ""), "media_name": task.get("payload", {}).get("media_name", "")}
        import time as _time
        _exec_start = _time.time()
        # 媒体マッピングを取得（tenant_idで絞り込み）
        tenant_id = task.get("tenant_id", "default")
        payload   = task.get("payload", {})
        media_name = payload.get("media_name")
        media_mapping = None
        if media_name:
            docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
            for d in docs:
                dm = d.to_dict()
                if dm.get("media_name") == media_name:
                    media_mapping = dm
                    break

        # operation_mappingsのselectorsをdom_selectorsにマージ（deep scan結果を実行に反映）
        if media_mapping:
            _op_type_exec = task.get("operation_type", "")
            _op_maps_exec = media_mapping.get("operation_mappings") or {}
            _op_sel_exec  = _op_maps_exec.get(_op_type_exec, {}).get("selectors") or {}
            if _op_sel_exec:
                _dom_exec = dict(media_mapping.get("dom_selectors") or {})
                for _role, _sel_data in _op_sel_exec.items():
                    _sel_val = _sel_data.get("selector") if isinstance(_sel_data, dict) else str(_sel_data)
                    if _sel_val and _role not in _dom_exec:
                        _dom_exec[_role] = _sel_val
                media_mapping = dict(media_mapping)
                media_mapping["dom_selectors"] = _dom_exec
                print(f"[EXEC_SELECTOR_MERGE] op={_op_type_exec} merged={list(_op_sel_exec.keys())} dom_keys={list(_dom_exec.keys())}", flush=True)
        # P29: 実行前バリデーション（validation_rulesチェック）
        if media_mapping:
            _vr = media_mapping.get("validation_rules") or {}
            _payload_check = task.get("payload", {})
            _val_errors = []
            _max_title = _vr.get("max_title_length")
            _max_body  = _vr.get("max_body_length")
            _forbidden = _vr.get("forbidden_words") or []
            _max_img   = _vr.get("image_max_size_mb")
            if _max_title and len(str(_payload_check.get("title", ""))) > _max_title:
                _val_errors.append(f"titleが{_max_title}文字を超えています（" + str(len(str(_payload_check.get("title", "")))) + "文字）")
            if _max_body and len(str(_payload_check.get("body", ""))) > _max_body:
                _val_errors.append(f"bodyが{_max_body}文字を超えています（" + str(len(str(_payload_check.get("body", "")))) + "文字）")
            for _fw in _forbidden:
                for _field in ("title", "body", "text"):
                    if _fw and _fw in str(_payload_check.get(_field, "")):
                        _val_errors.append(f"禁止ワード「{_fw}」が{_field}に含まれています")
            if _val_errors:
                _val_msg = "、".join(_val_errors)
                ref.update({"status": "BLOCKED", "result": {"validation_errors": _val_errors}})
                raise HTTPException(status_code=400, detail=f"[P29_VALIDATION] {_val_msg}")
            print(f"[P29_VALIDATION] OK operation_type={task.get('operation_type','')} errors=0", flush=True)
        # executor層に委譲
        result = execute_agent_task(task, media_mapping)
        final_status = result.get("status", "WAITING_EXECUTOR")

        # P26: FAILED時の原因分類別リトライ（最大3回）
        import datetime as _dt
        def _classify_retry_reason(res: dict) -> str:
            msg = str(res.get("message", "")).lower()
            if any(k in msg for k in ("selector", "not found", "element", "locator")):
                return "SELECTOR_BROKEN"
            if any(k in msg for k in ("login", "session", "cookie", "auth", "expired")):
                return "LOGIN_EXPIRED"
            if any(k in msg for k in ("timeout", "timed out", "time out")):
                return "TIMEOUT"
            if any(k in msg for k in ("server", "500", "502", "503", "504", "connection")):
                return "SERVER_ERROR"
            return ""
        _retry_reason = _classify_retry_reason(result) if final_status == "FAILED" and not result.get("executed") else ""
        _retry_triggerable = bool(_retry_reason)
        if _retry_triggerable:
            _current_retry = task.get("retry_count", 0)
            if _current_retry < 3:
                _now_retry = _dt.datetime.utcnow()
                ref.update({
                    "retry_count":      _current_retry + 1,
                    "last_retry_at":    _now_retry,
                    "retry_reason":     _retry_reason,
                    "status":           "APPROVED",
                })
                # P26: 原因分類別処置
                if _retry_reason == "SELECTOR_BROKEN":
                    print(f"[P26_RETRY] reason=SELECTOR_BROKEN retry={_current_retry+1}", flush=True)
                elif _retry_reason == "LOGIN_EXPIRED":
                    from api.core.browser_executor import _clear_cached_session
                    _mid = task.get("payload", {}).get("media_mapping_id") or task.get("media_mapping_id", "")
                    if _mid: _clear_cached_session(_mid, reason="LOGIN_EXPIRED_retry")
                    print(f"[P26_RETRY] reason=LOGIN_EXPIRED session_cleared retry={_current_retry+1}", flush=True)
                elif _retry_reason in ("TIMEOUT", "SERVER_ERROR"):
                    import time as _t26
                    _t26.sleep(2)
                    print(f"[P26_RETRY] reason={_retry_reason} wait=2s retry={_current_retry+1}", flush=True)
                result = execute_agent_task(task, media_mapping)
                final_status = result.get("status", "WAITING_EXECUTOR")
                ref.update({"status": final_status, "result": result})
            else:
                result["retry_exhausted"] = True
                result["retry_reason"]    = _retry_reason
                print(f"[P26_RETRY] リトライ上限到達 reason={_retry_reason} count={_current_retry}", flush=True)
        # executorが返したstatusをそのままFirestoreに反映
        # DONE以外はDONEにしない
        ref.update({"status": final_status, "result": result})
        after_state = {"status": final_status, "result": result}
        _exec_ms = int((_time.time() - _exec_start) * 1000)
        _last_verification = result.get("verification") if isinstance(result, dict) else None
        _last_rollback = result.get("rollback") if isinstance(result, dict) else None
        _last_selector_repair = result.get("selector_repair") if isinstance(result, dict) else None
        # selector自己修復候補をmedia_mappingsへ保存（自動適用禁止・提案のみ）
        if _last_selector_repair and _last_selector_repair.get("suggested"):
            try:
                _media_id = task.get("media_mapping_id") or task.get("payload", {}).get("media_mapping_id")
                if _media_id:
                    import datetime as _dt
                    get_db().collection("media_mappings").document(_media_id).update({
                        "selector_repair_suggestions": {
                            "created_at":         _dt.datetime.utcnow(),
                            "operation_type":     task.get("operation_type", ""),
                            "failed_selectors":   _last_selector_repair.get("failed_selectors", []),
                            "suggested_selectors": _last_selector_repair.get("suggested_selectors", []),
                            "reason":             "実行失敗または検証失敗によるselector自己修復候補",
                        }
                    })
            except Exception as _sre:
                print("[selector_repair] media_mappings保存エラー: " + str(_sre), flush=True)
        # P16昇格: selector execution feedback保存
        try:
            _media_id_fb = task.get("media_mapping_id") or task.get("payload", {}).get("media_mapping_id")
            if _media_id_fb:
                _mm_fb = get_db().collection("media_mappings").document(_media_id_fb).get()
                if _mm_fb.exists:
                    _mm_data = _mm_fb.to_dict()
                    _mn = _mm_data.get("media_name", "")
                    _ot = task.get("operation_type", "")
                    _is_done = final_status == "DONE"
                    _verify_ok = bool(_last_verification and _last_verification.get("verified"))
                    from api.core.browser_executor import update_selector_learning_stats
                    # dom_selectorsの各selectorをfeedback保存
                    for _sk, _sv in (_mm_data.get("dom_selectors") or {}).items():
                        update_selector_learning_stats(
                            db=get_db(),
                            media_name=_mn,
                            operation_type=_ot,
                            selector=_sv,
                            success=_is_done,
                            timeout=False,
                            verify_success=_verify_ok,
                            latency_ms=float(_exec_ms),
                        )
        except Exception as _fb_e:
            print("[selector_feedback] 保存エラー: " + str(_fb_e), flush=True)
        # P17: operation chain memory保存
        try:
            from api.core.browser_executor import update_operation_chain_memory
            _p17_tenant   = task.get("tenant_id", "")
            _p17_wf_id    = task.get("workflow_id", "")
            _p17_media    = task.get("payload", {}).get("media_name", "")
            _p17_op_type  = task.get("operation_type", "")
            _p17_steps    = task.get("operation_steps") or []
            _p17_success  = (final_status == "DONE")
            _p17_retry    = task.get("retry_count", 0)
            _p17_repair   = len((_last_selector_repair or {}).get("suggested_selectors", []))
            update_operation_chain_memory(
                db=get_db(),
                tenant_id=_p17_tenant,
                workflow_id=_p17_wf_id,
                media_name=_p17_media,
                operation_type=_p17_op_type,
                operation_steps=_p17_steps,
                success=_p17_success,
                duration_ms=float(_exec_ms),
                retry_count=_p17_retry,
                selector_repair_count=_p17_repair,
            )
        except Exception as _p17_e:
            print(f"[P17 chain_memory] 保存エラー: {type(_p17_e).__name__}", flush=True)

        # P18: cross-media template保存（DONE時のみ）
        if final_status == "DONE":
            try:
                from api.core.browser_executor import update_cross_media_template
                _p18_media    = task.get("payload", {}).get("media_name", "")
                _p18_op_type  = task.get("operation_type", "")
                _p18_steps    = task.get("operation_steps") or []
                _p18_repair   = len((_last_selector_repair or {}).get("suggested_selectors", []))
                _p18_mm       = media_mapping or {}
                _p18_family   = _p18_mm.get("media_family", "")
                _p18_caps     = _p18_mm.get("capabilities") or {}
                _p18_dom_sel  = _p18_mm.get("dom_selectors") or {}
                _p18_industry = task.get("industry", "generic")
                _p18_tenant   = task.get("tenant_id", "")
                update_cross_media_template(
                    db=get_db(),
                    tenant_id=_p18_tenant,
                    media_name=_p18_media,
                    media_family=_p18_family,
                    industry=_p18_industry,
                    operation_type=_p18_op_type,
                    capabilities=_p18_caps,
                    operation_steps=_p18_steps,
                    dom_selectors=_p18_dom_sel,
                    success=True,
                    duration_ms=float(_exec_ms),
                    repair_count=_p18_repair,
                )
            except Exception as _p18_e:
                print(f"[P18 cross_media_template] 保存エラー: {type(_p18_e).__name__}", flush=True)

        # P19: failure pattern clustering（FAILED時）
        if final_status == "FAILED":
            try:
                from api.core.browser_executor import update_failure_pattern_cluster
                _p19_tenant    = task.get("tenant_id", "")
                _p19_op_type   = task.get("operation_type", "")
                _p19_media     = task.get("payload", {}).get("media_name", "")
                _p19_mm        = media_mapping or {}
                _p19_family    = _p19_mm.get("media_family", "")
                _p19_err_type  = type(result.get("error", "")).__name__ if isinstance(result, dict) else "FAILED"
                _p19_err_msg   = str(result.get("error", "")) if isinstance(result, dict) else ""
                _p19_failed_sel = list((_last_selector_repair or {}).get("failed_selectors", []))
                _p19_rollback  = str((_last_rollback or {}).get("reason", ""))
                _p19_heal_stat = "success" if (_last_verification or {}).get("verified") else "failed"
                update_failure_pattern_cluster(
                    db=get_db(),
                    tenant_id=_p19_tenant,
                    error_type=_p19_err_type,
                    error_msg=_p19_err_msg,
                    operation_type=_p19_op_type,
                    media_name=_p19_media,
                    media_family=_p19_family,
                    failed_selectors=_p19_failed_sel,
                    rollback_reason=_p19_rollback,
                    self_heal_status=_p19_heal_stat,
                )
            except Exception as _p19_e:
                print(f"[P19 failure_cluster] 保存エラー: {type(_p19_e).__name__}", flush=True)

        _save_log(task, final_status == "DONE", before_state, after_state,
                  execution_time_ms=_exec_ms, verification=_last_verification,
                  rollback=_last_rollback, selector_repair=_last_selector_repair, result=result)
        return {"task_id": task_id, "status": final_status, "result": result}
    except Exception as e:
        error_msg = str(e)
        ref.update({"status": "FAILED", "result": {"error": error_msg}})
        _exec_ms = int((_time.time() - _exec_start) * 1000)
        # P19: failure pattern clustering（exception FAILED時）
        try:
            from api.core.browser_executor import update_failure_pattern_cluster
            _p19_tenant2   = task.get("tenant_id", "")
            _p19_op_type2  = task.get("operation_type", "")
            _p19_media2    = task.get("payload", {}).get("media_name", "")
            _p19_mm2       = media_mapping if 'media_mapping' in dir() else {}
            _p19_family2   = (_p19_mm2 or {}).get("media_family", "")
            update_failure_pattern_cluster(
                db=get_db(),
                tenant_id=_p19_tenant2,
                error_type=type(e).__name__,
                error_msg=error_msg[:200],
                operation_type=_p19_op_type2,
                media_name=_p19_media2,
                media_family=_p19_family2,
                failed_selectors=[],
                rollback_reason="",
                self_heal_status="not_attempted",
            )
        except Exception as _p19_e2:
            print(f"[P19 failure_cluster except] 保存エラー: {type(_p19_e2).__name__}", flush=True)
        _save_log(task, False, before_state, {"status": "FAILED"}, error_msg,
                  execution_time_ms=_exec_ms, rollback=None, result=None)
        raise HTTPException(status_code=500, detail="実行エラー: " + error_msg)

@router.get("/task/list")
def list_tasks(
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    docs = db.collection("agent_tasks").where("tenant_id", "==", tenant_id).limit(200).stream()
    rows = []
    for d in docs:
        t = d.to_dict()
        for k in ("created_at", "approved_at", "scheduled_at"):
            if t.get(k) and hasattr(t[k], "isoformat"):
                t[k] = t[k].isoformat()
        if status and t.get("status") != status:
            continue
        if agent_type and t.get("agent_type") != agent_type:
            continue
        rows.append(t)
    return {"tasks": rows, "count": len(rows)}

@router.get("/task/{task_id}")
def get_task(task_id: str, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("agent_tasks").document(task_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    t = doc.to_dict()
    for k in ("created_at", "approved_at", "scheduled_at"):
        if t.get(k) and hasattr(t[k], "isoformat"):
            t[k] = t[k].isoformat()
    return t

@router.get("/log/list")
def list_logs(
    agent_type: Optional[str] = None,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    docs = db.collection("agent_logs").where("tenant_id", "==", tenant_id).limit(300).stream()
    rows = []
    for d in docs:
        lg = d.to_dict()
        if agent_type and lg.get("agent_type") != agent_type:
            continue
        if lg.get("executed_at") and hasattr(lg["executed_at"], "isoformat"):
            lg["executed_at"] = lg["executed_at"].isoformat()
        rows.append(lg)
    return {"logs": rows, "count": len(rows)}

@router.get("/permissions/{tenant_id}")
def get_permissions(tenant_id: str, user: dict = Depends(verify_token)):
    ctx = _resolve_agent_user_context(user)
    if not ctx["is_admin"] and ctx["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("agent_permissions").document(tenant_id).get()
    if not doc.exists:
        return {"tenant_id": tenant_id, "admin_granted": False, "allowed_agents": [], "allowed_operations": [], "max_tasks_per_day": 0}
    return doc.to_dict()

@router.get("/industry_templates")
def get_industry_templates(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    return {"templates": INDUSTRY_TEMPLATES, "agent_types": list(AGENT_TYPES), "operation_types": list(OPERATION_TYPES)}



# --- /plan: 自然言語指示からtask候補を生成 ---

class PlanRequest(BaseModel):
    instruction: str
    mapping_id: Optional[str] = None

@router.post("/plan")
def plan_agent_task(req: PlanRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.instruction or not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instructionが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    # media_mappings 取得
    mappings_docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
    mappings = [d.to_dict() for d in mappings_docs]
    if req.mapping_id:
        mappings = [m for m in mappings if m.get("mapping_id") == req.mapping_id] or mappings

    # agent_ops 取得
    ops_docs = db.collection("agent_ops").stream()
    ops = []
    for d in ops_docs:
        op = d.to_dict()
        if op.get("active") is False:
            continue
        ops.append({
            "op_id": op.get("op_id") or d.id,
            "display_name": op.get("display_name") or op.get("name") or "",
            "operation_type": op.get("operation_type", ""),
            "entity_type": op.get("entity_type", ""),
            "industry": op.get("industry", "other"),
            "payload_schema": op.get("payload_schema", {}),
        })

    # LLMで解析
    try:
        from api.core.llm_client import call_llm_json
        site_names = [m.get('media_name') for m in mappings]
        op_list = [{'op_id': o['op_id'], 'display_name': o['display_name'], 'operation_type': o['operation_type'], 'industry': o['industry'], 'fields': [f['key'] for f in o['payload_schema'].get('fields', [])]} for o in ops]
        prompt = (
            'You are an agent task analyzer. '
            'Read the user instruction and return JSON only (no markdown). '
            'Format: {"ready": bool, "media_name": str|null, "op_id": str|null, "operation_type": str|null, "payload": {}, "preview": str, "question": str|null} '
            'Sites: ' + str(site_names) + ' '
            'Ops: ' + str(op_list) + ' '
            'Instruction: ' + req.instruction + ' '
            'Fill payload from payload_schema fields. If info missing set ready=false and write question in Japanese. If site unclear set ready=false.'
        )
        result = call_llm_json(prompt)
    except Exception as e:
        print(f"[PLAN] LLM error: {type(e).__name__}", flush=True)
        return {"ok": False, "ready": False, "question": "AIによる解析に失敗しました。手動で自動化を選択してください。"}

    return {"ok": True, **result}


# --- agent_ops（Firestoreから動的取得） ---

@router.get("/ops")
def list_ops(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    user_plans = ctx["plans"]
    is_admin = ctx["is_admin"]
    docs = db.collection("agent_ops").stream()
    result = []
    for d in docs:
        op = d.to_dict()
        # 権限チェック（adminはすべて取得）
        allowed = {str(x).upper() for x in op.get("allowed_plans", [])}
        if not is_admin and allowed and not user_plans.intersection(allowed):
            continue
        # created_at シリアライズ
        if op.get("created_at") and hasattr(op["created_at"], "isoformat"):
            op["created_at"] = op["created_at"].isoformat()
        # display_name を name/op_name からフォールバック補完
        if not op.get("display_name"):
            op["display_name"] = op.get("name") or op.get("op_name") or op.get("op_id", "")
        # industry デフォルト補完
        if not op.get("industry"):
            op["industry"] = "generic"
        # 欠損チェック
        missing = []
        if not op.get("display_name"): missing.append("display_name")
        if not op.get("operation_type"): missing.append("operation_type")
        if not op.get("entity_type"): missing.append("entity_type")
        if not op.get("payload_schema"): missing.append("payload_schema")
        if missing:
            op["active"] = False
            op["invalid_reason"] = f"定義不足: {', '.join(missing)}"
        elif op.get("active") is None:
            op["active"] = True
        result.append(op)
    return {"ops": result, "count": len(result)}


# --- media_mappings ---

class MediaMappingCreateRequest(BaseModel):
    media_name: str
    media_url: str
    login_url: Optional[str] = None
    industry: Optional[str] = "other"
    operation_type: Optional[str] = None
    auth_type: Optional[str] = "login_form"
    dom_selectors: dict = Field(default_factory=dict)
    form_structure: dict = Field(default_factory=dict)
    credential_secret_name: Optional[str] = None
    verify_selector: Optional[str] = None
    capabilities: Optional[dict] = Field(default_factory=lambda: {
        "can_login": True,
        "can_upload_image": False,
        "can_post_news": False,
        "can_update_text": False,
        "can_verify": False,
    })

@router.post("/media/map")
def create_media_mapping(req: MediaMappingCreateRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    # 同一tenant内のmedia_name重複チェック
    existing = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
    for ex in existing:
        if ex.to_dict().get("media_name") == req.media_name:
            raise HTTPException(status_code=400, detail="同じ媒体名のマッピングが既に存在します")
    mapping_id = str(uuid.uuid4())
    doc = {
        "mapping_id": mapping_id,
        "tenant_id": tenant_id,
        "media_name": req.media_name,
        "media_url": req.media_url or "",
        "login_url": req.login_url or req.media_url or "",
        "industry": req.industry or "other",
        "operation_type": req.operation_type,
        "auth_type": req.auth_type,
        "dom_selectors": req.dom_selectors,
        "form_structure": req.form_structure,
        "credential_secret_name": req.credential_secret_name,
        "verify_selector": req.verify_selector,
        "capabilities": req.capabilities or {"can_login": True, "can_upload_image": False, "can_post_news": False, "can_update_text": False, "can_verify": False},
        "login_health": "UNKNOWN",
        "selector_health": "UNKNOWN",
        "health_score": 0,
        "last_success_at": None,
        "last_failure_at": None,
        "failure_count": 0,
        "consecutive_failures": 0,
        "last_error_type": None,
        "last_error_message": None,
        "last_verified_at": None,
        "created_at": datetime.datetime.utcnow(),
    }
    db.collection("media_mappings").document(mapping_id).set(doc)
    print(f"[AGENT_MEDIA_CREATE] mapping_id={mapping_id} tenant_id={tenant_id} media_name={req.media_name} login_url_set={bool(req.login_url)}", flush=True)
    saved = db.collection("media_mappings").document(mapping_id).get()
    print(f"[AGENT_MEDIA_CREATE_VERIFY] mapping_id={mapping_id} exists={saved.exists}", flush=True)
    return {"mapping_id": mapping_id, "status": "created"}

@router.get("/media/map")
def list_media_mappings(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
    result = []
    for d in docs:
        m = d.to_dict()
        for k in ("created_at", "last_verified_at", "crawler_last_run_at", "updated_at"):
            if m.get(k) and hasattr(m[k], "isoformat"):
                m[k] = m[k].isoformat()
        # crawl_state内のdatetimeもシリアライズ
        cs = m.get("crawl_state")
        if isinstance(cs, dict):
            for ck in ("updated_at", "started_at"):
                if cs.get(ck) and hasattr(cs[ck], "isoformat"):
                    cs[ck] = cs[ck].isoformat()
        result.append(m)
    print(f"[AGENT_MEDIA_LIST] tenant_id={tenant_id} count={len(result)}", flush=True)
    return {"mappings": result, "count": len(result)}

@router.delete("/media/map/{mapping_id}")
def delete_media_mapping(mapping_id: str, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは削除できません")
    ref.delete()
    return {"mapping_id": mapping_id, "status": "deleted"}


# --- agent_schedules ---

class ScheduleCreateRequest(BaseModel):
    op_id: str
    cron_expr: str
    payload_template: dict = Field(default_factory=dict)
    enabled: bool = True

@router.post("/schedule/create")
def create_schedule(req: ScheduleCreateRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    op_doc = db.collection("agent_ops").document(req.op_id).get()
    if not op_doc.exists:
        raise HTTPException(status_code=400, detail="存在しないop_idです")
    schedule_id = str(uuid.uuid4())
    _now_cs = datetime.datetime.utcnow()
    doc = {
        "schedule_id": schedule_id,
        "tenant_id": tenant_id,
        "op_id": req.op_id,
        "cron_expr": req.cron_expr,
        "payload_template": req.payload_template,
        "enabled": req.enabled,
        "last_run_at": None,
        "next_run_at": _calc_next_run(req.cron_expr, _now_cs, schedule_id),
        "schedule_health": "HEALTHY",
        "last_error_type": None,
        "last_error_message": None,
        "consecutive_failures": 0,
        "created_at": _now_cs,
    }
    db.collection("agent_schedules").document(schedule_id).set(doc)
    return {"schedule_id": schedule_id, "status": "created"}

@router.get("/schedule/list")
def list_schedules(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    docs = db.collection("agent_schedules").where("tenant_id", "==", tenant_id).stream()
    result = []
    for d in docs:
        s = d.to_dict()
        for k in ("created_at", "last_run_at", "next_run_at"):
            if s.get(k) and hasattr(s[k], "isoformat"):
                s[k] = s[k].isoformat()
        result.append(s)
    return {"schedules": result, "count": len(result)}

@router.patch("/schedule/{schedule_id}")
def update_schedule(schedule_id: str, enabled: bool, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_schedules").document(schedule_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="スケジュールが見つかりません")
    s = doc.to_dict()
    if s.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのスケジュールは変更できません")
    ref.update({"enabled": enabled})
    return {"schedule_id": schedule_id, "enabled": enabled}


# --- スケジューラトリガー ---

def _calc_next_run(
    cron_expr: str,
    base_time: datetime.datetime,
    schedule_id: str = "",
) -> datetime.datetime:
    """cron_exprとbase_timeからnext_run_atを計算する"""
    try:
        return croniter(cron_expr, base_time).get_next(datetime.datetime)
    except Exception as e:
        print(f"[SCHEDULER_CRON_ERROR] schedule_id={schedule_id} cron_expr={cron_expr!r} error={type(e).__name__}", flush=True)
        if schedule_id:
            try:
                get_db().collection("agent_schedules").document(schedule_id).update({
                    "schedule_health": "BROKEN",
                    "last_error_type": type(e).__name__,
                    "last_error_message": "不正なcron式のためスケジュール計算に失敗しました",
                })
            except Exception as fe:
                print(f"[SCHEDULER_CRON_ERROR] Firestore update failed: {type(fe).__name__}", flush=True)
        return base_time + datetime.timedelta(minutes=1)


@router.post("/schedule/trigger")
@router.post("/scheduler/run")
def trigger_schedules(x_ascend_scheduler_token: Optional[str] = Header(None)):
    """
    Cloud Schedulerから毎分呼び出される。
    next_run_at <= now のスケジュールのみ対象にagent_tasksを作成する。
    二重作成防止のためlast_run_atから一定時間経過しているものだけ処理する。
    """
    expected_token = os.environ.get("SCHEDULER_TOKEN", "")
    if expected_token and x_ascend_scheduler_token != expected_token:
        raise HTTPException(status_code=403, detail="scheduler token invalid")
    db = get_db()
    now = datetime.datetime.utcnow()
    docs = db.collection("agent_schedules").where("enabled", "==", True).stream()
    triggered = []
    skipped   = []
    failed    = []
    for d in docs:
        s = d.to_dict()
        schedule_id = s.get("schedule_id", d.id)

        # next_run_at が設定済みで未到達なものはスキップ
        next_run_at = s.get("next_run_at")
        if next_run_at and hasattr(next_run_at, "replace"):
            next_run_at_naive = next_run_at.replace(tzinfo=None)
            if next_run_at_naive > now:
                skipped.append(schedule_id)
                continue

        # last_run_atから60秒以内は二重作成防止でスキップ
        last_run_at = s.get("last_run_at")
        if last_run_at and hasattr(last_run_at, "replace"):
            last_run_naive = last_run_at.replace(tzinfo=None)
            if (now - last_run_naive).total_seconds() < 60:
                skipped.append(schedule_id)
                continue

        op_doc = db.collection("agent_ops").document(s.get("op_id", "")).get()
        if not op_doc.exists:
            failed.append({"schedule_id": schedule_id, "reason": "op_not_found"})
            continue
        op = op_doc.to_dict()
        required_approval = op.get("required_approval", True)
        task_id = str(uuid.uuid4())
        payload = s.get("payload_template", {})
        op_operation_type = op.get("operation_type", "schedule_update")
        # P0: scheduler経由でもpermission enforcement
        _sched_tenant = s.get("tenant_id", "default")
        _sched_perm = _get_agent_permissions(_sched_tenant)
        _allowed_ops_s = _sched_perm.get("allowed_operations") or []
        _allowed_ags_s = _sched_perm.get("allowed_agents") or []
        if _allowed_ops_s and op_operation_type and op_operation_type not in _allowed_ops_s:
            failed.append({"schedule_id": schedule_id, "reason": f"operation_type '{op_operation_type}' not in allowed_operations"})
            continue
        if _allowed_ags_s and op.get("category") and op.get("category") not in _allowed_ags_s:
            failed.append({"schedule_id": schedule_id, "reason": f"agent_type '{op.get('category')}' not in allowed_agents"})
            continue
        op_industry = op.get("industry", "other")
        op_category = op.get("category", "")
        op_entity_type = op.get("entity_type", "")
        op_display_name = op.get("display_name") or op.get("name") or op.get("op_name") or s.get("op_id", "")
        op_snapshot = {
            "op_id": s.get("op_id", ""),
            "display_name": op_display_name,
            "category": op_category,
            "operation_type": op_operation_type,
            "entity_type": op_entity_type,
            "industry": op_industry,
            "payload_schema_version": op.get("payload_schema_version", "1"),
        }
        task = {
            "task_id":        task_id,
            "tenant_id":      s.get("tenant_id", "default"),
            "user_uid":       "scheduler",
            "agent_type":     op_category,
            "operation_type": op_operation_type,
            "industry":       op_industry,
            "entity_type":    op_entity_type,
            "op_id":          s.get("op_id", ""),
            "op_snapshot":    op_snapshot,
            "status":         "PENDING" if required_approval else "APPROVED",
            "payload":        payload,
            "preview": {
                "agent_type":      op_category,
                "operation_type":  op_operation_type,
                "industry":        op_industry,
                "entity_label":    "",
                "summary":         op_display_name + "（スケジュール自動登録）",
                "payload_preview": payload,
            },
            "approved_by":  None if required_approval else "scheduler",
            "approved_at":  None if required_approval else now,
            "scheduled_at": now.isoformat(),
            "result":       None,
            "created_at":   now,
        }
        try:
            db.collection("agent_tasks").document(task_id).set(task)
            db.collection("agent_schedules").document(schedule_id).update({
                "last_run_at": now,
                "next_run_at": _calc_next_run(s.get("cron_expr", "* * * * *"), now, schedule_id),
                "schedule_health": "HEALTHY",
                "consecutive_failures": 0,
                "last_error_type": None,
                "last_error_message": None,
            })
            triggered.append(task_id)
        except Exception as te:
            print(f"[SCHEDULER_TASK_ERROR] schedule_id={schedule_id} error={type(te).__name__}", flush=True)
            failed.append({"schedule_id": schedule_id, "reason": type(te).__name__})
            try:
                s_current = get_db().collection("agent_schedules").document(schedule_id).get().to_dict() or {}
                consec = s_current.get("consecutive_failures", 0) + 1
                get_db().collection("agent_schedules").document(schedule_id).update({
                    "schedule_health": "FAILED",
                    "last_error_type": type(te).__name__,
                    "last_error_message": "タスク生成に失敗しました",
                    "consecutive_failures": consec,
                })
            except Exception as fe2:
                print(f"[SCHEDULER_TASK_ERROR] health update failed: {type(fe2).__name__}", flush=True)
    return {"triggered": len(triggered), "skipped": len(skipped), "failed": failed, "task_ids": triggered}


# --- 媒体マッピング検証（last_verified_at更新） ---

class CredentialSaveRequest(BaseModel):
    login_id: str
    password: str

@router.patch("/media/map/{mapping_id}/credential")
def save_credential(mapping_id: str, req: CredentialSaveRequest, user: dict = Depends(verify_token)):
    """
    Secret Managerへログイン情報を保存し、Firestoreのcredential_secret_nameを更新する。
    ID/PASSはログ・レスポンスへ絶対含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.login_id or not req.password:
        raise HTTPException(status_code=400, detail="IDとパスワードを入力してください")

    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    print(f"[AGENT_MEDIA_CREDENTIAL_ATTEMPT] mapping_id={mapping_id} tenant_id={tenant_id}", flush=True)

    # mapping ownership確認
    doc = db.collection("media_mappings").document(mapping_id).get()
    print(f"[AGENT_MEDIA_CREDENTIAL_LOOKUP] mapping_id={mapping_id} exists={doc.exists}", flush=True)
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != tenant_id and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")

    # secret名生成（平文保存禁止・secret名はレスポンスに含めない）
    secret_name = f"agent-media-{tenant_id}-{mapping_id}"

    # Secret Managerへ保存（失敗時はFirestore更新しない）
    from api.core.secret_manager import save_secret_json
    try:
        save_secret_json(secret_name, {"username": req.login_id, "password": req.password})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Firestore更新（Secret保存成功時のみ）
    db.collection("media_mappings").document(mapping_id).update({
        "credential_secret_name": secret_name,
        "credential_updated_at": datetime.datetime.utcnow(),
    })

    return {
        "status": "saved",
        "mapping_id": mapping_id,
        "credential_registered": True,
    }

@router.patch("/media/map/{mapping_id}/verify")
def verify_media_mapping(mapping_id: str, user: dict = Depends(verify_token)):
    """媒体マッピングの構造確認済みとしてlast_verified_atを更新する。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは更新できません")
    now = datetime.datetime.utcnow()
    ref.update({"last_verified_at": now})
    return {"mapping_id": mapping_id, "last_verified_at": now.isoformat()}


class SelectorUpdateRequest(BaseModel):
    dom_selectors: dict
    form_structure: dict = Field(default_factory=dict)
    verify_selector: Optional[str] = None

@router.patch("/media/map/{mapping_id}/selectors")
def update_media_selectors(
    mapping_id: str,
    req: SelectorUpdateRequest,
    user: dict = Depends(verify_token),
):
    """媒体マッピングのDOMセレクター・フォーム構造を更新する。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは更新できません")
    now = datetime.datetime.utcnow()
    update_data = {
        "dom_selectors": req.dom_selectors,
        "form_structure": req.form_structure,
        "last_verified_at": now,
    }
    if req.verify_selector is not None:
        update_data["verify_selector"] = req.verify_selector
    ref.update(update_data)
    return {"mapping_id": mapping_id, "status": "updated", "last_verified_at": now.isoformat()}


# --- 媒体マッピング ログイン確認 ---

@router.post("/media/map/{mapping_id}/login_check")
def login_check_media_mapping(mapping_id: str, user: dict = Depends(verify_token)):
    """
    対象mappingのcredential_secret_nameでSecret Managerからcredを取得し、
    Playwrightでログイン試行する。
    成功時のみlast_verified_atを更新。
    ID/PASSはFirestore・ログ・レスポンスに絶対含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")

    # credential_secret_name確認
    secret_name = m.get("credential_secret_name")
    if not secret_name:
        return {
            "status":   "BLOCKED",
            "executed": False,
            "message":  "credential_secret_nameが未設定です。Secret Managerへの登録が必要です。",
        }

    # Secret Manager取得（ID/PASSはここで止める・レスポンスに含めない）
    from api.core.secret_manager import get_secret_json
    from api.core.browser_executor import run_login_form_check
    creds = get_secret_json(secret_name)
    if not creds or creds.get("blocked"):
        return {
            "status":   "BLOCKED",
            "executed": False,
            "message":  creds.get("error", "認証情報の取得に失敗しました") if creds else "認証情報が取得できませんでした",
        }

    # ログイン試行（ID/PASSはrun_login_form_check内でのみ使用・外に出さない）
    result = run_login_form_check(m, creds)

    # 成功・失敗ともにFirestore更新（healthフィールド記録）
    _now = datetime.datetime.utcnow()
    if result.get("login_success"):
        ref.update({
            "last_verified_at": _now,
            "last_success_at": _now,
            "consecutive_failures": 0,
            "login_health": "HEALTHY",
            "selector_health": "HEALTHY",
            "health_score": 100,
            "operation_health": "UNKNOWN",
        })
    else:
        m_current = ref.get().to_dict() or {}
        consec = m_current.get("consecutive_failures", 0) + 1
        health_score = 40 if consec >= 3 else 70
        ref.update({
            "last_failure_at": _now,
            "failure_count": firestore.Increment(1),
            "consecutive_failures": consec,
            "last_error_type": result.get("error_type", "LOGIN_FAILED"),
            "last_error_message": {
                "STEP_ERROR": "selector_error",
                "TIMEOUT": "timeout",
                "LOGIN_FAILED": "login_failed",
            }.get(str(result.get("error_type", "LOGIN_FAILED")), "login_failed"),
            "login_health": "FAILED",
            "health_score": health_score,
        })

    # credentialsは絶対にresultに含めない
    # crawl_result要約（巨大化防止：counts + candidates のみ）
    _raw_crawl = result.get("crawl_result") or {}
    _crawl_summary = {}
    if _raw_crawl:
        _crawl_summary = {
            "status":                  _raw_crawl.get("status"),
            "pages_crawled":           _raw_crawl.get("pages_crawled"),
            "capabilities":            _raw_crawl.get("capabilities"),
            "operation_candidates":    _raw_crawl.get("operation_candidates"),
            "operation_steps_by_type": _raw_crawl.get("operation_steps_by_type"),
            "detected_summary":        _raw_crawl.get("detected_summary"),
            "note":                    _raw_crawl.get("note"),
            "error":                   _raw_crawl.get("error"),
        }
        # None値を除去
        _crawl_summary = {k: v for k, v in _crawl_summary.items() if v is not None}
    return {
        "mapping_id":    mapping_id,
        "status":        result.get("status"),
        "login_checked": result.get("login_checked", False),
        "login_success": result.get("login_success", False),
        "message":       result.get("message", ""),
        "crawl_result":  _crawl_summary,
    }


class DomScanRequest(BaseModel):
    max_pages: int = 200
    start_url: str = ""
    include_patterns: list = Field(default_factory=list)
    exclude_patterns: list = Field(default_factory=list)
    reset_resume: bool = False

@router.post("/media/map/{mapping_id}/dom_scan")
def dom_scan_media_mapping(mapping_id: str, body: DomScanRequest = None, user: dict = Depends(verify_token)):
    """
    P5: DOM自動候補抽出。
    Playwrightで対象URLのinput/textarea/button/select/file inputを収集し、
    suggested_structure と suggested_verify_selector を返す。
    ID/PASSはレスポンスに含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")

    from api.core.browser_executor import run_dom_scan
    _max_pages        = body.max_pages        if body else 200
    _start_url        = body.start_url        if body else ""
    _include_patterns = body.include_patterns if body else []
    _exclude_patterns = body.exclude_patterns if body else []
    _reset_resume     = body.reset_resume     if body else False
    result = run_dom_scan(
        m,
        max_pages=_max_pages,
        start_url=_start_url,
        include_patterns=_include_patterns,
        exclude_patterns=_exclude_patterns,
        reset_resume=_reset_resume,
    )
    # capabilities_candidateをmedia_mappingsへ保存（自動適用禁止・提案のみ）
    _caps_candidate = result.get("capabilities_candidate")
    if _caps_candidate and _caps_candidate.get("capabilities"):
        try:
            import datetime as _dt
            get_db().collection("media_mappings").document(mapping_id).update({
                "capabilities_candidate": {
                    "created_at":   _dt.datetime.utcnow(),
                    "capabilities": _caps_candidate.get("capabilities", {}),
                    "reason":       _caps_candidate.get("reason", {}),
                }
            })
        except Exception as _ce:
            print("[capabilities_candidate] 保存エラー: " + str(_ce), flush=True)

    # P15-3: semantic_selector_candidatesをmedia_mappingsへ保存
    _semantic_candidate = result.get("semantic_selector_candidates")
    _auto_applied = False
    _auto_apply_message = ""
    if _semantic_candidate and _semantic_candidate.get("labels"):
        try:
            get_db().collection("media_mappings").document(mapping_id).update({
                "semantic_selector_candidates": _semantic_candidate,
            })
        except Exception as _se:
            print("[semantic_selector_candidates] 保存エラー: " + str(_se), flush=True)
        import datetime as _dt2
        _labels = _semantic_candidate.get("labels", {})
        _auto_label_map = {
            "login_id": "username", "id": "username", "user_id": "username",
            "username": "username", "account": "username", "email": "username",
            "mail": "username", "loginId": "username",
            "login_password": "password", "pass": "password",
            "password": "password", "pwd": "password",
            "submit": "login_submit", "login_submit": "login_submit",
            "login_button": "login_submit", "button": "login_submit",
            "send": "login_submit",
        }
        from api.core.browser_executor import normalize_css_selector as _normalize_sel
        _normalized = {}
        for _lk, _lv in _labels.items():
            _nk = _auto_label_map.get(_lk)
            if _nk and _nk not in _normalized:
                _normalized[_nk] = _normalize_sel(str(_lv))
        if {"username", "password", "login_submit"}.issubset(_normalized.keys()):
            try:
                _current_dom = m.get("dom_selectors") or {}
                _current_dom.update(_normalized)
                _current_caps = m.get("capabilities") or {}
                _current_caps["can_login"] = True
                get_db().collection("media_mappings").document(mapping_id).update({
                    "dom_selectors": _current_dom,
                    "capabilities": _current_caps,
                    "dom_selectors_auto_applied_at": _dt2.datetime.utcnow(),
                })
                _auto_applied = True
                _auto_apply_message = "ログイン入力欄を自動設定しました。次に接続確認を押してください。"
            except Exception as _ae:
                print("[dom_scan auto_apply] 保存エラー: " + str(_ae), flush=True)

    # P0-1: detected_summary を Firestore に保存（画面再読み込み後も保持）
    _detected_summary = result.get("detected_summary")
    _detected_login_url = result.get("target_url")
    if _detected_summary or _detected_login_url:
        try:
            import datetime as _dt_ds
            _ds_update = {"detected_summary_at": _dt_ds.datetime.utcnow()}
            if _detected_summary:
                _ds_update["detected_summary"] = _detected_summary
            if _detected_login_url:
                _ds_update["detected_login_url"] = _detected_login_url
            get_db().collection("media_mappings").document(mapping_id).update(_ds_update)
        except Exception as _dse:
            print("[detected_summary] 保存エラー: " + str(_dse), flush=True)

    # dom_scan成功時にlogin_health=HEALTHYを保存
    if result.get("admin_crawl_completed") or result.get("executed"):
        try:
            import datetime as _dt_lh2
            get_db().collection("media_mappings").document(mapping_id).update({
                "login_health": "HEALTHY",
                "selector_health": "HEALTHY",
                "health_score": 100,
                "consecutive_failures": 0,
                "last_success_at": _dt_lh2.datetime.utcnow(),
                "last_verified_at": _dt_lh2.datetime.utcnow(),
            })
            print(f"[DOM_SCAN_LOGIN_HEALTH_SAVED] mapping_id={mapping_id} login_health=HEALTHY", flush=True)
        except Exception as _lh2_err:
            print(f"[DOM_SCAN_LOGIN_HEALTH_ERROR] {_lh2_err}", flush=True)
    return {
        "mapping_id":                    mapping_id,
        "status":                        result.get("status"),
        "executed":                      result.get("executed", False),
        "target_url":                    result.get("target_url"),
        "selectors":                     result.get("selectors", []),
        "suggested_structure":           result.get("suggested_structure", {}),
        "suggested_verify_selector":     result.get("suggested_verify_selector"),
        "capabilities_candidate":        result.get("capabilities_candidate"),
        "semantic_selector_candidates":  result.get("semantic_selector_candidates"),
        "detected_summary":              result.get("detected_summary"),
        "message":                       result.get("message", ""),
        "auto_applied":                  _auto_applied,
        "auto_apply_message":            _auto_apply_message,
        "dom_scan_completed":            result.get("executed", False),
        "admin_crawl_completed":         result.get("admin_crawl_completed", False),
        "pages_crawled":                 result.get("pages_crawled", 0),
        "operation_candidates_count":    result.get("operation_candidates_count", 0),
    }


class SelectorRepairApplyRequest(BaseModel):
    approved_selectors: dict


@router.post("/media/map/{mapping_id}/selector_repair/apply")
def apply_selector_repair(
    mapping_id: str,
    req: SelectorRepairApplyRequest,
    user=Depends(verify_token),
):
    """selector自己修復候補をユーザー承認後にdom_selectorsへ適用する。自動適用禁止。"""
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")
    suggestions = m.get("selector_repair_suggestions", {})
    if not suggestions or suggestions.get("cleared_at"):
        raise HTTPException(status_code=400, detail="適用可能なselector修復候補がありません")
    approved = req.approved_selectors
    if not approved:
        raise HTTPException(status_code=400, detail="approved_selectorsが空です")
    # 既存dom_selectorsを退避
    current_dom = m.get("dom_selectors", {})
    previous_selectors = {k: current_dom[k] for k in approved if k in current_dom}
    # approved_selectorsのみmerge（自動全上書き禁止）
    new_dom = dict(current_dom)
    for k, v in approved.items():
        new_dom[k] = v
    import datetime as _dt
    now = _dt.datetime.utcnow()
    ref.update({
        "dom_selectors": new_dom,
        "previous_selectors": previous_selectors,
        "last_selector_repair_applied_at": now,
        "selector_repair_suggestions": {
            "cleared_at": now,
            "applied": True,
        },
    })
    # agent_logsに適用ログを保存
    try:
        log_id = str(uuid.uuid4())
        db.collection("agent_logs").document(log_id).set({
            "log_id":                       log_id,
            "tenant_id":                    m.get("tenant_id", ""),
            "operator_uid":                 user.get("uid", ""),
            "operation_type":               "selector_repair_apply",
            "mapping_id":                   mapping_id,
            "selector_repair_applied":      True,
            "selector_repair_apply_count":  len(approved),
            "executed_at":                  now,
        })
    except Exception as _le:
        print("[selector_repair_apply] log error: " + str(_le), flush=True)
    return {
        "mapping_id":      mapping_id,
        "applied_keys":    list(approved.keys()),
        "previous_selectors": previous_selectors,
        "status":          "applied",
    }


class CapabilitiesApplyRequest(BaseModel):
    approved_capabilities: dict


@router.post("/media/map/{mapping_id}/capabilities/apply")
def apply_capabilities(
    mapping_id: str,
    req: CapabilitiesApplyRequest,
    user=Depends(verify_token),
):
    """capability候補をユーザー承認後にcapabilitiesへ適用する。自動適用禁止。"""
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")
    candidate = m.get("capabilities_candidate", {})
    if not candidate or candidate.get("cleared_at"):
        raise HTTPException(status_code=400, detail="適用可能なcapability候補がありません")
    approved = req.approved_capabilities
    if not approved:
        raise HTTPException(status_code=400, detail="approved_capabilitiesが空です")
    # 既存capabilitiesを退避
    current_caps = m.get("capabilities") or {}
    previous_capabilities = {k: current_caps.get(k) for k in approved if k in current_caps}
    # approved_capabilitiesのみmerge（自動全上書き禁止）
    new_caps = dict(current_caps)
    for k, v in approved.items():
        new_caps[k] = v
    import datetime as _dt
    now = _dt.datetime.utcnow()
    ref.update({
        "capabilities": new_caps,
        "previous_capabilities": previous_capabilities,
        "last_capabilities_applied_at": now,
        "capabilities_candidate": {
            "cleared_at": now,
            "applied": True,
        },
    })
    # agent_logsに適用ログを保存
    try:
        log_id = str(uuid.uuid4())
        db.collection("agent_logs").document(log_id).set({
            "log_id":                      log_id,
            "tenant_id":                   m.get("tenant_id", ""),
            "operator_uid":                user.get("uid", ""),
            "operation_type":              "capabilities_apply",
            "mapping_id":                  mapping_id,
            "capabilities_applied":        True,
            "capabilities_apply_count":    len(approved),
            "executed_at":                 now,
        })
    except Exception as _le:
        print("[capabilities_apply] log error: " + str(_le), flush=True)
    return {
        "mapping_id":           mapping_id,
        "applied_keys":         list(approved.keys()),
        "previous_capabilities": previous_capabilities,
        "status":               "applied",
    }



class CapabilitiesUpdateRequest(BaseModel):
    capabilities: dict

@router.patch("/media/map/{mapping_id}/capabilities")
def update_capabilities_direct(
    mapping_id: str,
    req: CapabilitiesUpdateRequest,
    user=Depends(verify_token),
):
    """capabilityを直接ON/OFFする（ユーザー手動トグル用）。"""
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")
    current_caps = m.get("capabilities") or {}
    new_caps = dict(current_caps)
    for k, v in req.capabilities.items():
        new_caps[k] = bool(v)
    import datetime as _dt
    ref.update({
        "capabilities": new_caps,
        "last_capabilities_updated_at": _dt.datetime.utcnow(),
    })
    return {
        "mapping_id": mapping_id,
        "capabilities": new_caps,
        "status": "updated",
    }

class SemanticSelectorApplyRequest(BaseModel):
    approved_labels: dict

@router.post("/media/map/{mapping_id}/semantic_selector/apply")
def apply_semantic_selector(
    mapping_id: str,
    req: SemanticSelectorApplyRequest,
    user=Depends(verify_token),
):
    """P15-5: semantic selector候補をユーザー承認後にdom_selectorsへ適用する。自動適用禁止。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")
    candidate = m.get("semantic_selector_candidates", {})
    if not candidate or not candidate.get("labels"):
        raise HTTPException(status_code=400, detail="適用可能なsemantic selector候補がありません")
    approved = req.approved_labels
    if not approved:
        raise HTTPException(status_code=400, detail="approved_labelsが空です")
    labels = candidate.get("labels", {})
    confidence = candidate.get("confidence", {})
    # label_key → dom_selectors正規化マップ
    label_map = {
        "login_id": "username",
        "id": "username",
        "user_id": "username",
        "username": "username",
        "account": "username",
        "email": "username",
        "mail": "username",
        "loginId": "username",
        "login_password": "password",
        "pass": "password",
        "password": "password",
        "pwd": "password",
        "submit": "login_submit",
        "login_submit": "login_submit",
        "login_button": "login_submit",
        "button": "login_submit",
        "send": "login_submit",
    }
    # approved_labelsのkeyのみdom_selectorsへmerge（自動全上書き禁止）
    current_dom = m.get("dom_selectors") or {}
    previous_selectors = {k: current_dom.get(k) for k in approved if k in current_dom}
    new_dom = dict(current_dom)
    applied_keys = []
    for label_key, apply_flag in approved.items():
        if apply_flag and label_key in labels:
            normalized_key = label_map.get(label_key, label_key)
            new_dom[normalized_key] = labels[label_key]
            applied_keys.append(normalized_key)
    import datetime as _dt
    now = _dt.datetime.utcnow()
    ref.update({
        "dom_selectors": new_dom,
        "previous_selectors": previous_selectors,
        "last_semantic_selector_applied_at": now,
        "semantic_selector_candidates": {
            "cleared_at": now,
            "applied": True,
            "applied_keys": applied_keys,
        },
    })
    # agent_logsに適用ログを保存
    try:
        log_id = str(uuid.uuid4())
        db.collection("agent_logs").document(log_id).set({
            "log_id":                          log_id,
            "tenant_id":                       m.get("tenant_id", ""),
            "operator_uid":                    user.get("uid", ""),
            "operation_type":                  "semantic_selector_apply",
            "mapping_id":                      mapping_id,
            "semantic_selector_applied":        True,
            "semantic_selector_apply_count":    len(applied_keys),
            "executed_at":                     now,
        })
    except Exception as _le:
        print("[semantic_selector_apply] log error: " + str(_le), flush=True)
    return {
        "mapping_id":         mapping_id,
        "applied_keys":       applied_keys,
        "previous_selectors": previous_selectors,
        "status":             "applied",
    }

@router.post("/media/map/{mapping_id}/selector_rank/recompute")
def recompute_selector_ranking(
    mapping_id: str,
    user=Depends(verify_token),
):
    """P16-4: selector候補をスコアリングしてselector_rankingsをFirestoreへ保存する。自動適用禁止。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    ctx = _resolve_agent_user_context(user)
    if m.get("tenant_id") != ctx["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")

    # agent_logs取得（Firestore制約: whereはtenant_idのみ、Python側でmapping_idをfilter）
    try:
        _logs_raw = db.collection("agent_logs").where("tenant_id", "==", ctx["tenant_id"]).limit(200).stream()
        agent_logs = [l.to_dict() for l in _logs_raw if l.to_dict().get("mapping_id") == mapping_id][:100]
    except Exception as _le:
        agent_logs = []

    # selector候補を集約
    candidates = []
    # semantic_selector_candidates
    _sem = m.get("semantic_selector_candidates") or {}
    _sem_labels = _sem.get("labels") or {}
    _sem_conf = _sem.get("confidence") or {}
    for label, sel in _sem_labels.items():
        candidates.append({
            "selector": sel,
            "source": "semantic",
            "label": label,
            "confidence": _sem_conf.get(label, "low"),
        })
    # selector_repair_suggestions
    _repair = m.get("selector_repair_suggestions") or {}
    for s in (_repair.get("suggested_selectors") or []):
        sel = s.get("suggested_selector")
        if sel:
            candidates.append({
                "selector": sel,
                "source": "repair",
                "label": s.get("name") or s.get("id") or "",
            })
    # 現在のdom_selectors
    for k, sel in (m.get("dom_selectors") or {}).items():
        candidates.append({
            "selector": sel,
            "source": "history",
            "label": k,
        })

    # P16昇格: selector_learning_stats取得（media_nameでfilter）
    _media_name = m.get("media_name", "")
    learning_stats = {}
    try:
        _stats_raw = db.collection("selector_learning_stats").where("media_name", "==", _media_name).limit(200).stream()
        for _sd in _stats_raw:
            _sd_dict = _sd.to_dict()
            _key = f"{_sd_dict.get('media_name','')}__{_sd_dict.get('operation_type','')}__{_sd_dict.get('selector_hash','')}"
            learning_stats[_sd.id] = _sd_dict
    except Exception as _sle:
        print("[selector_learning_stats] 取得エラー: " + str(_sle), flush=True)

    from api.core.browser_executor import rank_selector_candidates
    ranked = rank_selector_candidates(
        media_mapping=m,
        operation_type=m.get("operation_type") or "",
        step_type="",
        semantic_label="",
        candidates=candidates,
        agent_logs=agent_logs,
        learning_stats=learning_stats,
    )

    import datetime as _dt
    now = _dt.datetime.utcnow()
    selector_rankings = {
        "computed_at":          now,
        "operation_type":        m.get("operation_type") or "",
        "ranked_selectors":      ranked,
        "ranking_model_version": "p16_v1",
        "learning_enabled":      True,
    }

    ref.update({"selector_rankings": selector_rankings})
    return {
        "mapping_id": mapping_id,
        "status": "computed",
        "ranked_count": len(ranked),
        "selector_rankings": selector_rankings,
    }


@router.post("/media/map/{mapping_id}/learning/recompute")
def recompute_learning_health(
    mapping_id: str,
    user=Depends(verify_token),
):
    """実行学習ログを集計しmedia_mappingsのagent_learning_healthへ反映する。自動修正禁止。"""
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    if m.get("tenant_id") != _resolve_agent_user_context(user)["tenant_id"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")
    tenant_id = m.get("tenant_id", "")
    media_name = m.get("media_name", "")
    # Firestore制約: whereはtenant_idのみ。Python側でmapping_id/media_nameをfilter
    docs = db.collection("agent_logs").where("tenant_id", "==", tenant_id).limit(500).stream()
    logs = [d.to_dict() for d in docs if d.to_dict().get("mapping_id") == mapping_id or d.to_dict().get("media_name") == media_name]
    logs = sorted(logs, key=lambda x: x.get("executed_at") or "", reverse=True)[:100]
    # 集計
    execution_count = len(logs)
    success_count = sum(1 for l in logs if l.get("success"))
    failed_count = execution_count - success_count
    success_rate = round(success_count / execution_count, 3) if execution_count > 0 else 0.0
    exec_times = [l["execution_time_ms"] for l in logs if l.get("execution_time_ms")]
    avg_execution_time_ms = int(sum(exec_times) / len(exec_times)) if exec_times else 0
    sel_rates = [l["selector_success_rate"] for l in logs if l.get("selector_success_rate") is not None]
    selector_success_rate_avg = round(sum(sel_rates) / len(sel_rates), 3) if sel_rates else None
    failed_logs = [l for l in logs if not l.get("success")]
    error_msgs = [l.get("error_message", "") for l in failed_logs if l.get("error_message")]
    from collections import Counter
    most_common_failure = Counter(error_msgs).most_common(1)[0][0] if error_msgs else ""
    last_failure_at = next((l.get("executed_at") for l in logs if not l.get("success")), None)
    last_success_at = next((l.get("executed_at") for l in logs if l.get("success")), None)
    selector_repair_count = sum(1 for l in logs if l.get("selector_repair_suggested"))
    # recommendations生成
    recommendations = []
    if success_rate < 0.5:
        recommendations.append("selector再確認が必要です")
    if selector_success_rate_avg is not None and selector_success_rate_avg < 0.7:
        recommendations.append("DOM_SCANとselector修復を推奨します")
    if avg_execution_time_ms > 15000:
        recommendations.append("媒体処理が遅いためtimeout延長またはselector最適化を推奨します")
    if selector_repair_count >= 3:
        recommendations.append("selector構造変更の可能性があります。DOM再スキャンを推奨します")
    # media_mappingsへ保存
    import datetime as _dt
    now = _dt.datetime.utcnow()
    health = {
        "computed_at":               now,
        "execution_count":           execution_count,
        "success_count":             success_count,
        "failed_count":              failed_count,
        "success_rate":              success_rate,
        "avg_execution_time_ms":     avg_execution_time_ms,
        "selector_success_rate_avg": selector_success_rate_avg,
        "most_common_failure":       most_common_failure,
        "last_failure_at":           last_failure_at,
        "last_success_at":           last_success_at,
        "recommendations":           recommendations,
    }
    ref.update({"agent_learning_health": health})
    return {
        "mapping_id": mapping_id,
        "status":     "recomputed",
        "health":     health,
    }


# ── P16.7: recompute_all_rankings endpoint ───────────────────────────────

@router.post("/selector/recompute_all_rankings")
def recompute_all_rankings(user: dict = Depends(verify_token)):
    """
    P16.7: tenant内全media_mappingsのselector rankingを再計算。
    Firestore制約厳守: whereはtenant_idのみ、ソートはPython側。
    自動適用禁止。ranking提案のみ。dom_selectors更新禁止。
    """
    import datetime as _dt
    from api.core.browser_executor import (
        rank_selector_candidates,
        save_selector_ranking_result,
    )
    from api.core.firestore_client import get_db

    tenant_id = user.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id missing")

    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="DB unavailable")

    # media_mappings取得 (whereはtenant_idのみ・Firestore複合index不使用)
    try:
        docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
        mappings = [{"id": d.id, **d.to_dict()} for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"media_mappings取得エラー: {type(e).__name__}")

    # Python側でソート
    mappings.sort(key=lambda x: x.get("media_name", ""))

    results = []
    errors  = []

    for mapping in mappings:
        mapping_id  = mapping.get("id", "")
        media_name  = mapping.get("media_name", "")
        dom_selectors = mapping.get("dom_selectors") or {}

        if not dom_selectors:
            results.append({
                "mapping_id": mapping_id,
                "media_name": media_name,
                "status":     "skipped",
                "reason":     "dom_selectors未設定",
            })
            continue

        # agent_logsをtenant_id + mapping_idで取得 (whereは1フィールドのみ)
        try:
            log_docs = (
                db.collection("agent_logs")
                .where("mapping_id", "==", mapping_id)
                .stream()
            )
            agent_logs = [d.to_dict() for d in log_docs]
        except Exception:
            agent_logs = []

        # selector_learning_statsをmedia_nameプレフィックスで取得
        try:
            stats_docs = (
                db.collection("selector_learning_stats")
                .where("media_name", "==", media_name)
                .stream()
            )
            learning_stats = {}
            for sd in stats_docs:
                learning_stats[sd.id] = sd.to_dict()
        except Exception:
            learning_stats = {}

        # dom_selectorsのselectorをcandidatesとして変換
        candidates = []
        for label, selector in dom_selectors.items():
            if isinstance(selector, str) and selector:
                candidates.append({
                    "selector": selector,
                    "label":    label,
                    "source":   "dom_selectors",
                    "confidence": "medium",
                })

        if not candidates:
            results.append({
                "mapping_id": mapping_id,
                "media_name": media_name,
                "status":     "skipped",
                "reason":     "有効なselector候補なし",
            })
            continue

        try:
            ranked = rank_selector_candidates(
                media_mapping=mapping,
                operation_type="recompute_all",
                step_type="recompute",
                semantic_label="",
                candidates=candidates,
                agent_logs=agent_logs,
                learning_stats=learning_stats,
            )
            save_selector_ranking_result(
                db=db,
                media_name=media_name,
                operation_type="recompute_all",
                step_type="recompute",
                ranked=ranked,
            )
            results.append({
                "mapping_id":      mapping_id,
                "media_name":      media_name,
                "status":          "recomputed",
                "candidate_count": len(candidates),
                "top_selector":    ranked[0].get("selector") if ranked else None,
                "top_score":       ranked[0].get("score")    if ranked else None,
                "top_confidence":  ranked[0].get("confidence") if ranked else None,
            })
        except Exception as e:
            errors.append({
                "mapping_id": mapping_id,
                "media_name": media_name,
                "error":      type(e).__name__,
            })

    return {
        "tenant_id":       tenant_id,
        "total_mappings":  len(mappings),
        "recomputed":      len([r for r in results if r.get("status") == "recomputed"]),
        "skipped":         len([r for r in results if r.get("status") == "skipped"]),
        "errors":          len(errors),
        "results":         results,
        "error_details":   errors,
        "computed_at":     _dt.datetime.utcnow().isoformat(),
    }


# ── P18: cross-media reusable templates endpoints ────────────────────────

@router.post("/template/recommend")
def recommend_templates(
    operation_type: str,
    industry: str = "",
    media_family: str = "",
    user: dict = Depends(verify_token),
):
    """
    P18: 再利用可能なcross-media templateを提案。
    自動適用禁止。提案のみ。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")

    # media_mappingsからcapabilities取得（tenant_idのみwhere）
    capabilities = {}
    try:
        docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
        for d in docs:
            dm = d.to_dict()
            if dm.get("media_family") == media_family and dm.get("capabilities"):
                capabilities = dm.get("capabilities") or {}
                break
    except Exception:
        pass

    from api.core.browser_executor import find_reusable_templates
    templates = find_reusable_templates(
        db=db,
        tenant_id=tenant_id,
        operation_type=operation_type,
        industry=industry,
        media_family=media_family,
        capabilities=capabilities,
        top_n=5,
    )
    return {
        "tenant_id":              tenant_id,
        "operation_type":         operation_type,
        "industry":               industry,
        "media_family":           media_family,
        "recommended_templates":  templates,
        "count":                  len(templates),
    }


@router.post("/template/apply")
def apply_template(
    template_id: str,
    mapping_id: str,
    user: dict = Depends(verify_token),
):
    """
    P18: templateをmedia_mappingへ適用するpreviewを生成。
    自動適用禁止。preview必須。承認制維持。
    dom_selectors自動更新禁止。提案のみ返す。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")

    # template取得
    tmpl_doc = db.collection("cross_media_templates").document(template_id).get()
    if not tmpl_doc.exists:
        raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
    tmpl = tmpl_doc.to_dict()

    # media_mapping取得
    mm_doc = db.collection("media_mappings").document(mapping_id).get()
    if not mm_doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mm = mm_doc.to_dict()

    # tenant_id所有者チェック
    if mm.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmedia_mappingへのアクセス権がありません")

    # 不足selectorの算出（template inheritance）
    existing_selectors = set((mm.get("dom_selectors") or {}).keys())
    tmpl_patterns      = tmpl.get("selector_patterns") or []
    missing_selectors  = [
        p for p in tmpl_patterns
        if p.get("type") not in existing_selectors
    ]
    already_covered    = [
        p for p in tmpl_patterns
        if p.get("type") in existing_selectors
    ]

    # preview生成（自動適用しない）
    preview = {
        "template_id":        template_id,
        "template_name":      tmpl.get("template_name", ""),
        "template_signature": tmpl.get("template_signature", ""),
        "mapping_id":         mapping_id,
        "media_name":         mm.get("media_name", ""),
        "operation_steps":    tmpl.get("operation_steps", []),
        "capabilities":       tmpl.get("capabilities", {}),
        "selector_patterns":  tmpl_patterns,
        "missing_selectors":  missing_selectors,
        "already_covered":    already_covered,
        "missing_count":      len(missing_selectors),
        "covered_count":      len(already_covered),
        "success_rate":       tmpl.get("success_rate", 0.0),
        "source_media_names": tmpl.get("source_media_names", []),
        "apply_note":         "このpreviewは提案のみです。dom_selectorsの自動更新は行いません。適用する場合は管理者が手動で設定してください。",
    }
    return {
        "status":  "preview",
        "preview": preview,
    }


# ── P20: semi-autonomous workflow orchestration endpoints ─────────────────

@router.post("/workflow/session/create")
def create_workflow_session_endpoint(
    workflow_id: str,
    goal: str,
    operation_type: str,
    user: dict = Depends(verify_token),
    max_retry: int = 2,
    allow_self_heal: bool = True,
    allow_replan: bool = True,
    require_human_on_high_risk: bool = True,
    interruptible: bool = True,
):
    """
    P20: workflow_execution_sessionを新規作成。
    HIGH riskは自動的にWAITING_APPROVAL。LOW/MEDIUMはAPPROVED。
    自動実行禁止。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")

    from api.core.browser_executor import (
        estimate_workflow_risk,
        create_workflow_session,
    )

    # media_mapping取得（operation_typeから最初の1件）
    media_mapping = {}
    try:
        docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
        for d in docs:
            media_mapping = d.to_dict()
            break
    except Exception:
        pass

    execution_policy = {
        "max_retry":                max_retry,
        "allow_self_heal":          allow_self_heal,
        "allow_replan":             allow_replan,
        "require_human_on_high_risk": require_human_on_high_risk,
        "interruptible":            interruptible,
    }

    risk = estimate_workflow_risk(
        db=db,
        tenant_id=tenant_id,
        operation_type=operation_type,
        media_family=media_mapping.get("media_family", ""),
        operation_steps=[],
        media_mapping=media_mapping,
    )

    session_id = create_workflow_session(
        db=db,
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        goal=goal,
        operation_type=operation_type,
        operation_steps=[],
        execution_policy=execution_policy,
        risk_estimation=risk,
    )

    return {
        "session_id":      session_id,
        "workflow_id":     workflow_id,
        "approval_state":  "WAITING_APPROVAL" if risk["require_human_approval"] else "APPROVED",
        "risk_level":      risk["risk_level"],
        "risk_score":      risk["risk_score"],
        "risk_factors":    risk["risk_factors"],
        "execution_policy": execution_policy,
    }


@router.post("/workflow/approve")
def approve_workflow_session(
    session_id: str,
    user: dict = Depends(verify_token),
):
    """P20: workflow sessionを承認。HIGH risk承認に使用。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    import datetime as _dt
    ref = db.collection("workflow_execution_sessions").document(session_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    ref.update({
        "approval_state": "APPROVED",
        "status":         "READY",
        "approved_by":    user.get("uid", ""),
        "approved_at":    _dt.datetime.utcnow(),
        "updated_at":     _dt.datetime.utcnow(),
    })
    return {"session_id": session_id, "approval_state": "APPROVED"}


@router.post("/workflow/reject")
def reject_workflow_session(
    session_id: str,
    reason: str = "",
    user: dict = Depends(verify_token),
):
    """P20: workflow sessionを却下。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    import datetime as _dt
    ref = db.collection("workflow_execution_sessions").document(session_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    ref.update({
        "approval_state": "REJECTED",
        "status":         "REJECTED",
        "reject_reason":  reason,
        "rejected_by":    user.get("uid", ""),
        "rejected_at":    _dt.datetime.utcnow(),
        "updated_at":     _dt.datetime.utcnow(),
    })
    return {"session_id": session_id, "approval_state": "REJECTED"}


@router.post("/workflow/pause")
def pause_workflow_session(
    session_id: str,
    user: dict = Depends(verify_token),
):
    """P20: workflow sessionを一時停止。interruptible=Trueのみ許可。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    import datetime as _dt
    ref = db.collection("workflow_execution_sessions").document(session_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    if not d.get("interruptible", True):
        raise HTTPException(status_code=400, detail="このworkflowは中断不可です")
    ref.update({
        "paused":         True,
        "approval_state": "PAUSED",
        "status":         "PAUSED",
        "paused_at":      _dt.datetime.utcnow(),
        "updated_at":     _dt.datetime.utcnow(),
    })
    return {"session_id": session_id, "approval_state": "PAUSED"}


@router.post("/workflow/resume")
def resume_workflow_session(
    session_id: str,
    user: dict = Depends(verify_token),
):
    """P20: pause中のworkflow sessionを再開。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    import datetime as _dt
    ref = db.collection("workflow_execution_sessions").document(session_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    if not d.get("paused"):
        raise HTTPException(status_code=400, detail="このworkflowはpause中ではありません")
    ref.update({
        "paused":         False,
        "approval_state": "APPROVED",
        "status":         "READY",
        "resumed_at":     _dt.datetime.utcnow(),
        "updated_at":     _dt.datetime.utcnow(),
    })
    return {"session_id": session_id, "approval_state": "APPROVED", "status": "READY"}


@router.post("/workflow/cancel")
def cancel_workflow_session(
    session_id: str,
    reason: str = "",
    user: dict = Depends(verify_token),
):
    """P20: workflow sessionをキャンセル。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    import datetime as _dt
    ref = db.collection("workflow_execution_sessions").document(session_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    ref.update({
        "cancelled":      True,
        "approval_state": "CANCELLED",
        "status":         "CANCELLED",
        "cancel_reason":  reason,
        "cancelled_by":   user.get("uid", ""),
        "cancelled_at":   _dt.datetime.utcnow(),
        "updated_at":     _dt.datetime.utcnow(),
    })
    return {"session_id": session_id, "approval_state": "CANCELLED"}


@router.get("/workflow/session/{session_id}")
def get_workflow_session(
    session_id: str,
    user: dict = Depends(verify_token),
):
    """P20: workflow session詳細取得。frontend dashboard用。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    doc = db.collection("workflow_execution_sessions").document(session_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="sessionが見つかりません")
    d = doc.to_dict()
    if d.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このsessionへのアクセス権がありません")
    for k in ("created_at", "updated_at", "approved_at", "paused_at", "resumed_at", "cancelled_at"):
        if d.get(k) and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    return d


@router.get("/workflow/session/list")
def list_workflow_sessions(
    status: str = None,
    user: dict = Depends(verify_token),
):
    """P20: tenant内workflow session一覧。Firestore複合index不使用。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    docs = (
        db.collection("workflow_execution_sessions")
        .where("tenant_id", "==", tenant_id)
        .stream()
    )
    rows = []
    for d in docs:
        s = d.to_dict()
        # Python側フィルタ
        if status and s.get("status") != status:
            continue
        for k in ("created_at", "updated_at"):
            if s.get(k) and hasattr(s[k], "isoformat"):
                s[k] = s[k].isoformat()
        rows.append(s)
    # Python側ソート
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"sessions": rows, "count": len(rows)}


# ══════════════════════════════════════════════════════════════════════════════
# P23 Operation Deep Scan
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/media/map/{mapping_id}/operation/{operation_type}/deep_scan")
def operation_deep_scan(
    mapping_id: str,
    operation_type: str,
    body: dict = Body(default={}),
    user: dict = Depends(verify_token),
):
    """P23: operation_type別にログイン後ページを解析し必要selectorを検出する。
    実際の保存・投稿・更新は行わない（検出のみ）。
    結果を media_mappings/{mapping_id}.operation_mappings.{operation_type} に保存。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    tenant_id = user.get("tenant_id", "default")

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")

    mapping = doc.to_dict()
    mapping["id"] = mapping_id

    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    valid_ops = [
        "news_post", "text_update", "media_replace",
        "schedule_update", "price_update", "entity_register", "entity_update",
    ]
    if operation_type not in valid_ops:
        raise HTTPException(status_code=400, detail=f"未対応operation_type: {operation_type}")

    # deep_scan 実行
    import datetime as _dt
    now = _dt.datetime.utcnow()
    # 実行前にSCANNING状態を書き込み（ERROR残骸を上書き）
    try:
        db.collection("media_mappings").document(mapping_id).update({
            f"operation_mappings.{operation_type}": {
                "status": "SCANNING",
                "last_scanned_at": now.isoformat(),
            },
            "updated_at": now,
        })
    except Exception as e:
        print(f"[P23] SCANNING書き込みエラー: {e}", flush=True)

    from api.core.browser_executor import deep_scan_operation
    result = deep_scan_operation(mapping, operation_type, hint_url=body.get("hint_url", ""))

    # Firestore 保存（成功・失敗どちらも上書き）
    now2 = _dt.datetime.utcnow()
    result["last_scanned_at"] = now2.isoformat()
    if result.get("status") == "ERROR":
        result["error_message"] = result.get("error", "不明なエラー")
    try:
        # P24のtarget_url等を既存から引き継ぐ（P23保存で上書きしない）
        _existing_mm_p23 = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _existing_op_p23 = _existing_mm_p23.get("operation_mappings", {}).get(operation_type, {})
        _detail_list_p23 = _existing_mm_p23.get("operation_candidates_detail") or []
        _detail_url_p23 = next((d.get("source_url","") for d in _detail_list_p23 if isinstance(d,dict) and d.get("operation_type")==operation_type), "")
        if _existing_op_p23.get("p24_source_url") and _existing_op_p23.get("p24_source_url") == _detail_url_p23:
            result["p24_source_url"]    = _existing_op_p23["p24_source_url"]
            result["p24_confidence"]    = _existing_op_p23.get("p24_confidence", 0.0)
            result["p24_classified_at"] = _existing_op_p23.get("p24_classified_at")
            result["target_url"]        = _existing_op_p23["p24_source_url"]
        db.collection("media_mappings").document(mapping_id).update({
            f"operation_mappings.{operation_type}": result,
            "updated_at": now2,
        })
        print(f"[P23_SAVE] operation_type={operation_type} target_url={result.get('target_url')} status={result.get('status')}", flush=True)
    except Exception as e:
        print(f"[P23] Firestore保存エラー: {e}", flush=True)


    # ── P23完了後にoperation_steps_by_typeを再生成 ──────────────────
    try:
        _latest_mm = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _nav_graph   = _latest_mm.get("navigation_graph") or {}
        _op_cands    = _latest_mm.get("operation_candidates") or []
        _op_mappings = _latest_mm.get("operation_mappings") or {}
        _detail_list = _latest_mm.get("operation_candidates_detail") or []
        _detail_by_op = {
            d["operation_type"]: d
            for d in _detail_list
            if isinstance(d, dict) and d.get("operation_type")
        }
        from api.core.browser_executor import rebuild_operation_steps
        _new_steps = rebuild_operation_steps(_op_cands, _nav_graph, _op_mappings, _detail_by_op)
        if _new_steps:
            db.collection("media_mappings").document(mapping_id).update({
                "operation_steps_by_type": _new_steps,
                "updated_at": now2,
            })
            print(f"[P23_STEPS_REBUILT] mapping_id={mapping_id} ops={list(_new_steps.keys())}", flush=True)
    except Exception as _rb_err:
        print(f"[P23_STEPS_REBUILD_ERROR] {_rb_err}", flush=True)

    return {
        "mapping_id":     mapping_id,
        "operation_type": operation_type,
        "result":         result,
    }

def _sync_ready_operation_steps(mapping_id: str, db) -> None:
    """READY/PARTIAL op の operation_steps_by_type に同期する（末尾依存禁止）"""
    try:
        import datetime as _dt_sync
        from api.core.browser_executor import rebuild_operation_steps
        _doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _op_maps = _doc.get("operation_mappings", {})
        _nav = _doc.get("navigation_graph", {})
        _dlist = _doc.get("operation_candidates_detail") or []
        _det = {d["operation_type"]: d for d in _dlist if isinstance(d, dict) and d.get("operation_type")}
        _ready_ops = [
            op for op, m in _op_maps.items()
            if isinstance(m, dict)
            and m.get("status") in ("READY", "PARTIAL")
            and m.get("target_url")
        ]
        _steps = rebuild_operation_steps(_ready_ops, _nav, _op_maps, _det) if _ready_ops else {}
        db.collection("media_mappings").document(mapping_id).update({
            "operation_steps_by_type": _steps,
            "updated_at": _dt_sync.datetime.utcnow(),
        })
        print(f"[P24_7_STEPS_SYNC] mapping_id={mapping_id} ready_count={len(_ready_ops)} steps={list(_steps.keys())}", flush=True)
        # [修正A2] candidates正規化: operation_mappings保存直後に必ず実行
        _exist_cands = _doc.get("operation_candidates") or []
        _op_keys_sync = list(_op_maps.keys())
        _step_keys_sync = list(_steps.keys())
        print(f"[P24_CANDIDATE_NORMALIZE_BEFORE] mapping_id={mapping_id} candidates={_exist_cands}", flush=True)
        _norm_cands = []
        for _op_nc in _exist_cands + _op_keys_sync + _step_keys_sync:
            if not _op_nc or _op_nc == "admin_crawl":
                continue
            if _op_nc not in _norm_cands:
                _norm_cands.append(_op_nc)
        db.collection("media_mappings").document(mapping_id).update({
            "operation_candidates": _norm_cands,
        })
        print(f"[P24_CANDIDATE_NORMALIZE_AFTER] mapping_id={mapping_id} candidates={_norm_cands}", flush=True)
    except Exception as _e_sync:
        print(f"[P24_7_STEPS_SYNC_ERROR] mapping_id={mapping_id} {_e_sync}", flush=True)
        try:
            import datetime as _dt_sync_fb
            db.collection("media_mappings").document(mapping_id).update({
                "operation_steps_by_type": {},
                "updated_at": _dt_sync_fb.datetime.utcnow(),
            })
            print(f"[P24_7_STEPS_SYNC_FALLBACK] mapping_id={mapping_id} cleared", flush=True)
        except Exception as _e_fb:
            print(f"[P24_7_STEPS_SYNC_FALLBACK_ERROR] {_e_fb}", flush=True)



@router.post("/media/map/{mapping_id}/multi_deep_scan")
def multi_deep_scan(
    mapping_id: str,
    user: dict = Depends(verify_token),
):
    """P24.7: operation_candidates全件を順にdeep_scanし、各operationのREADY/PARTIAL/NEEDS_MAPPINGを必ず保存する。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = user.get("tenant_id", "default")
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict()
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    import datetime as _dt47
    valid_ops = [
        "news_post", "text_update", "media_replace",
        "schedule_update", "price_update", "entity_register", "entity_update",
    ]
    # required_fields定義（未解析保存用）
    _required_map = {
        "news_post":       ["title", "body", "save"],
        "text_update":     ["body", "save"],
        "media_replace":   ["file", "save"],
        "schedule_update": ["date_input", "save"],
        "price_update":    ["price_input", "save"],
        "entity_register": ["required_inputs", "save"],
        "entity_update":   ["editable_inputs", "save"],
    }
    # operation_candidates から対象を決定。なければ全valid_ops
    op_candidates = [op for op in (mapping.get("operation_candidates") or valid_ops) if op in valid_ops]

    now47 = _dt47.datetime.utcnow()
    print(f"[P24_7_MULTI_SCAN_START] mapping_id={mapping_id} operations_count={len(op_candidates)}", flush=True)

    results = {}
    ready_ops = []
    partial_ops = []
    waiting_ops = []
    failed_ops = []
    needs_mapping_ops = []

    from api.core.browser_executor import deep_scan_operation, rebuild_operation_steps

    for op in op_candidates:
        print(f"[P24_7_OP_SCAN_START] operation_type={op}", flush=True)
        _required = _required_map.get(op, ["save"])
        now47a = _dt47.datetime.utcnow()

        # SCANNING状態を書き込み
        try:
            db.collection("media_mappings").document(mapping_id).update({
                f"operation_mappings.{op}": {
                    "status": "SCANNING",
                    "last_scanned_at": now47a.isoformat(),
                },
            })
        except Exception as _e_scan:
            print(f"[P24_7_SCANNING_WRITE_ERROR] op={op} {_e_scan}", flush=True)

        # 最新mappingを取得して渡す
        _cur_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _cur_doc["id"] = mapping_id
        _cur_op = _cur_doc.get("operation_mappings", {}).get(op, {})
        _cur_status = _cur_op.get("status", "")

        # ── entity_update list系URL事前ガード ──
        _eu_list_patterns  = ["cast_list", "readlog", "review_list", "/list", "price", "fee", "course", "pricelist", "料金", "料金表", "systemlist", "multifee"]
        _eu_allow_patterns = ["cast_edit", "edit", "regist", "form", "profile_edit"]
        _eu_src_url = (_cur_op.get("p24_source_url") or _cur_op.get("target_url") or "").lower()
        if op == "entity_update" and _eu_src_url:
            _eu_is_list  = any(p in _eu_src_url for p in _eu_list_patterns)
            _eu_is_allow = any(p in _eu_src_url for p in _eu_allow_patterns)
            if _eu_is_list and not _eu_is_allow:
                print(f"[P24_7_ENTITY_UPDATE_LIST_GUARD] op={op} src_url={_eu_src_url} -> NEEDS_MAPPING", flush=True)
                result = {
                    "status":           "NEEDS_MAPPING",
                    "missing":          ["edit_trigger", "save"],
                    "selectors":        {},
                    "target_url":       None,
                    "validation_score": 0,
                    "error_reason":     "list page excluded for entity_update",
                    "executable":       False,
                    "last_scanned_at":  _dt47.datetime.utcnow().isoformat(),
                }
                print(f"[P24_7_FINAL_SAVE_GUARD] op={op} status=NEEDS_MAPPING", flush=True)
                db.collection("media_mappings").document(mapping_id).update({
                    f"operation_mappings.{op}": result,
                    "updated_at": _dt47.datetime.utcnow(),
                })
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status=NEEDS_MAPPING missing={result['missing']}", flush=True)
                _sync_ready_operation_steps(mapping_id, db)
                results[op] = result
                needs_mapping_ops.append(op)
                continue
        # ── deep_scan実行（with不使用・timeoutガード） ──
        import concurrent.futures as _cf47
        _DEEP_SCAN_TIMEOUT_SEC = 90
        _now47_before = _dt47.datetime.utcnow()
        print(f"[P24_7_DEEP_SCAN_BEFORE] op={op}", flush=True)
        _exe47 = _cf47.ThreadPoolExecutor(max_workers=1)
        _fut47 = _exe47.submit(deep_scan_operation, _cur_doc, op)
        try:
            result = _fut47.result(timeout=_DEEP_SCAN_TIMEOUT_SEC)
            _exe47.shutdown(wait=False, cancel_futures=True)
            _elapsed47 = (_dt47.datetime.utcnow() - _now47_before).total_seconds()
            print(f"[P24_7_DEEP_SCAN_AFTER] op={op} elapsed={_elapsed47:.1f}s status={result.get('status')}", flush=True)
        except _cf47.TimeoutError:
            _fut47.cancel()
            _exe47.shutdown(wait=False, cancel_futures=True)
            _elapsed47 = (_dt47.datetime.utcnow() - _now47_before).total_seconds()
            print(f"[P24_7_DEEP_SCAN_TIMEOUT] op={op} elapsed={_elapsed47:.1f}s", flush=True)
            _now47b_to = _dt47.datetime.utcnow()
            result = {
                "status":           "NEEDS_MAPPING",
                "missing":          _required,
                "selectors":        {},
                "target_url":       None,
                "validation_score": 0,
                "error_reason":     "deep_scan timeout",
                "executable":       False,
                "last_scanned_at":  _now47b_to.isoformat(),
            }
            print(f"[P24_7_FINAL_SAVE_GUARD] op={op} status=NEEDS_MAPPING reason=timeout", flush=True)
            try:
                db.collection("media_mappings").document(mapping_id).update({
                    f"operation_mappings.{op}": result,
                    "updated_at": _now47b_to,
                })
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status=NEEDS_MAPPING missing={_required}", flush=True)
            except Exception as _e_to_save:
                print(f"[P24_7_TIMEOUT_SAVE_ERROR] op={op} {_e_to_save}", flush=True)
            _sync_ready_operation_steps(mapping_id, db)
            results[op] = result
            needs_mapping_ops.append(op)
            continue
        except Exception as _e_exec:
            _exe47.shutdown(wait=False, cancel_futures=True)
            _elapsed47 = (_dt47.datetime.utcnow() - _now47_before).total_seconds()
            _now47b_ex = _dt47.datetime.utcnow()
            result = {
                "status":           "FAILED",
                "missing":          _required,
                "selectors":        {},
                "target_url":       None,
                "validation_score": 0,
                "error_reason":     str(_e_exec),
                "last_scanned_at":  _now47b_ex.isoformat(),
            }
            print(f"[P24_7_EXEC_ERROR] op={op} elapsed={_elapsed47:.1f}s {_e_exec}", flush=True)
            print(f"[P24_7_FINAL_SAVE_GUARD] op={op} status=FAILED reason=exception", flush=True)
            try:
                db.collection("media_mappings").document(mapping_id).update({
                    f"operation_mappings.{op}": result,
                    "updated_at": _now47b_ex,
                })
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status=FAILED missing={_required}", flush=True)
            except Exception as _e_ex_save:
                print(f"[P24_7_EXEC_SAVE_ERROR] op={op} {_e_ex_save}", flush=True)
            _sync_ready_operation_steps(mapping_id, db)
            results[op] = result
            failed_ops.append(op)
            continue
        now47b = _dt47.datetime.utcnow()
        result["last_scanned_at"] = now47b.isoformat()
        # selectors/missing/target_urlの保証
        if "selectors" not in result:
            result["selectors"] = {}
        if "missing" not in result:
            result["missing"] = _required
        if "target_url" not in result:
            result["target_url"] = None
        if "error_reason" not in result and result.get("error"):
            result["error_reason"] = result.get("error", "")

        # READY→PARTIAL/WAITING_EXECUTOR downgrade禁止
        # ── status正規化（必須仕様） ──────────────────────────────────
        _r_missing      = result.get("missing", [])
        _r_error_reason = result.get("error_reason", "") or ""
        _r_target_url   = result.get("target_url") or ""
        _r_status_raw   = result.get("status", "")
        if "missing fields:" in _r_error_reason and not _r_missing:
            import re as _re_status
            _mf = _re_status.findall(r"missing fields:\s*(?:\[([^\]]+)\]|([^;]+))", _r_error_reason)
            if _mf:
                _raw_missing = _mf[0][0] or _mf[0][1]
                _extracted = [
                    f.strip().strip("'").strip('"')
                    for f in _raw_missing.split(",")
                    if f.strip()
                ]
                result["missing"] = _extracted
                _r_missing = _extracted
        _has_missing_in_reason = "missing fields:" in _r_error_reason
        if _r_status_raw == "READY":
            if _r_missing or _has_missing_in_reason or not _r_target_url:
                result["status"] = "PARTIAL"
                result["executable"] = False
                if not _r_missing and _has_missing_in_reason:
                    result["missing"] = ["unknown_missing_field"]
                print(f"[STATUS_NORMALIZE] op={op} READY->PARTIAL missing={result['missing']} target_url={_r_target_url} reason={_r_error_reason[:80]}", flush=True)
        # ── status正規化ここまで ───────────────────────────────────────────
        _new_status = result.get("status", "")
        if _cur_status == "READY" and _new_status in ("PARTIAL", "WAITING_EXECUTOR", "FAILED", "NEEDS_MAPPING"):
            _new_vscore = result.get("validation_score", 0)
            _cur_vscore = _cur_op.get("validation_score", 0)
            _new_missing = result.get("missing", ["dummy"])
            # [修正B] READY+executable=True+stepsあり -> downgrade完全ブロック
            _cur_steps = (_cur_op.get("steps") or {})
            _cur_exec  = _cur_op.get("executable") is True
            if _cur_exec and _cur_steps:
                print(f"[P24_READY_DOWNGRADE_BLOCKED] op={op} cur=READY new={_new_status} vscore={_new_vscore} steps={list(_cur_steps.keys())[:3]} -> kept READY", flush=True)
                result["status"]           = "READY"
                result["executable"]       = True
                result["missing"]          = _cur_op.get("missing", [])
                result["target_url"]       = _cur_op.get("target_url", result.get("target_url"))
                result["validation_score"] = _cur_op.get("validation_score", _new_vscore)
            elif _new_vscore >= 70 and not _new_missing and _new_vscore > _cur_vscore:
                print(f"[P24_9_READY_UPDATE] op={op} old_score={_cur_vscore} new_score={_new_vscore} updating READY->READY", flush=True)
            else:
                print(f"[P24_7_DOWNGRADE_ALLOW] op={op} cur=READY new={_new_status} vscore={_new_vscore} proceeding.", flush=True)
        # P24のtarget_url等を引き継ぐ
        _detail_list_p247 = _cur_doc.get("operation_candidates_detail") or []
        _detail_url_p247 = next((d.get("source_url","") for d in _detail_list_p247 if isinstance(d,dict) and d.get("operation_type")==op), "")
        if _cur_op.get("p24_source_url") and _cur_op.get("p24_source_url") == _detail_url_p247:
            result["p24_source_url"]    = _cur_op["p24_source_url"]
            result["p24_confidence"]    = _cur_op.get("p24_confidence", 0.0)
            result["p24_classified_at"] = _cur_op.get("p24_classified_at")
            if not result.get("target_url"):
                result["target_url"] = _cur_op["p24_source_url"]

        # Firestore保存前READYガード（P24.9汚染防止）
        _bad_url_words = [
            "signin", "login", "readlog", "review_list", "navi",
            "manual", ".pdf", "C1Main",
        ]
        _save_target_url = (result.get("target_url") or "").lower()
        _save_missing = result.get("missing") or []
        _save_score = int(result.get("validation_score") or result.get("score") or 0)

        if result.get("status") == "READY":
            _invalid_url = any(w.lower() in _save_target_url for w in _bad_url_words)
            _list_like = (
                "/list" in _save_target_url
                and not any(x in _save_target_url for x in ["cast_edit", "edit", "regist", "form"])
            )
            if _save_missing or _save_score < 70 or _invalid_url or _list_like:
                result["status"] = "PARTIAL"
                result["executable"] = False
                result["validation_score"] = min(_save_score, 69)
                result["error_reason"] = (
                    (result.get("error_reason") or "")
                    + " ready_downgraded_before_save"
                ).strip()
                print(
                    f"[P24_7_READY_DOWNGRADE] op={op} score={_save_score} missing={_save_missing} target_url={result.get('target_url')}",
                    flush=True,
                )

        # P24.6がFirestoreに保存した最新結果を読み直してresultを更新（上書き防止）
        try:
            _cur_op_check = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _cur_op_data = _cur_op_check.get("operation_mappings", {}).get(op, {})
            if _cur_op_data.get("executable") is True and _cur_op_data.get("status") == "READY":
                result["status"]           = "READY"
                result["missing"]          = _cur_op_data.get("missing", [])
                result["selectors"]        = _cur_op_data.get("selectors", result.get("selectors", {}))
                result["target_url"]       = _cur_op_data.get("target_url", result.get("target_url"))
                result["validation_score"] = _cur_op_data.get("validation_score", result.get("validation_score", 0))
                result["executable"]       = True
                print(f"[P24_7_RESULT_SYNC] op={op} status=READY executable=True synced from Firestore", flush=True)
        except Exception as _sync_err:
            print(f"[P24_7_RESULT_SYNC_ERROR] op={op} {_sync_err}", flush=True)
        # Firestore保存（成功・失敗・WAITING_EXECUTOR 全て保存）
        try:
            db.collection("media_mappings").document(mapping_id).update({
                f"operation_mappings.{op}": result,
                "updated_at": now47b,
            })
            print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status={result.get('status')} missing={result.get('missing', [])}", flush=True)
            print(f"[P24_7_OP_SCAN_DONE] operation_type={op} status={result.get('status')} missing={result.get('missing', [])}", flush=True)
            _sync_ready_operation_steps(mapping_id, db)
        except Exception as _e_save:
            print(f"[P24_7_SAVE_ERROR] op={op} {_e_save}", flush=True)
            failed_ops.append(op)
            results[op] = {"status": "FAILED", "missing": _required, "selectors": {}, "target_url": None, "error_reason": str(_e_save)}
            continue

        _st = result.get("status", "")
        if _st == "READY":
            ready_ops.append(op)
        elif _st in ("PARTIAL",):
            partial_ops.append(op)
        elif _st in ("WAITING_EXECUTOR", "BLOCKED", "SCANNING"):
            waiting_ops.append(op)
        else:
            failed_ops.append(op)
        results[op] = result

    # 未処理operationを保証（念のため）
    try:
        _final_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _final_op_map = _final_doc.get("operation_mappings", {})
        _ensure_update = {}
        _now_ensure = _dt47.datetime.utcnow().isoformat()
        for _op in op_candidates:
            if _op not in _final_op_map:
                _required = _required_map.get(_op, ["save"])
                _ensure_update[f"operation_mappings.{_op}"] = {
                    "status": "NEEDS_MAPPING",
                    "missing": _required,
                    "selectors": {},
                    "target_url": None,
                    "error_reason": "multi_deep_scan did not produce result",
                    "last_scanned_at": _now_ensure,
                    "executable":       False,
                }
                needs_mapping_ops.append(_op)
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={_op} status=NEEDS_MAPPING missing={_required}", flush=True)
        if _ensure_update:
            db.collection("media_mappings").document(mapping_id).update(_ensure_update)
    except Exception as _e_ensure:
        print(f"[P24_7_ENSURE_ERROR] {_e_ensure}", flush=True)

    # [修正A] candidates union を先に実行 -> union済みでsteps再生成
    try:
        _latest_mm47 = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _nav47   = _latest_mm47.get("navigation_graph") or {}
        _ops47   = _latest_mm47.get("operation_mappings") or {}
        _dlist47 = _latest_mm47.get("operation_candidates_detail") or []
        _det47   = {d["operation_type"]: d for d in _dlist47 if isinstance(d, dict) and d.get("operation_type")}
        _existing_candidates47 = list(op_candidates or [])
        _op_keys47 = list((_ops47 or {}).keys())
        _step_keys47 = list((_latest_mm47.get("operation_steps_by_type") or {}).keys())
        _normalized_candidates = []
        for _op_nc in _existing_candidates47 + _op_keys47 + _step_keys47:
            if not _op_nc or _op_nc == "admin_crawl":
                continue
            if _op_nc not in _normalized_candidates:
                _normalized_candidates.append(_op_nc)
        print(f"[P24_7_OPERATION_CANDIDATES_NORMALIZED] before={_existing_candidates47} after={_normalized_candidates}", flush=True)
        _step_ops47 = [
            op for op in _normalized_candidates
            if _ops47.get(op, {}).get("status") in ("READY", "PARTIAL")
            and _ops47.get(op, {}).get("target_url")
        ]
        print(f"[P24_7_STEPS_READY_PARTIAL] step_ops={_step_ops47} skipped={[op for op in _normalized_candidates if op not in _step_ops47]}", flush=True)
        _steps47 = rebuild_operation_steps(_step_ops47, _nav47, _ops47, _det47) if _step_ops47 else {}
        db.collection("media_mappings").document(mapping_id).update({
            "operation_steps_by_type": _steps47,
            "operation_candidates": _normalized_candidates,
            "updated_at": _dt47.datetime.utcnow(),
        })
        print(f"[P24_7_STEPS_REBUILT] ops={list(_steps47.keys())} step_count={len(_step_ops47)}", flush=True)
    except Exception as _rb47:
        print(f"[P24_7_STEPS_REBUILD_ERROR] {_rb47}", flush=True)
    return {
        "ok": True,
        "mapping_id": mapping_id,
        "operations_count": len(op_candidates),
        "ready": ready_ops,
        "partial": partial_ops,
        "waiting": waiting_ops,
        "failed": failed_ops,
        "needs_mapping": needs_mapping_ops,
        "results": [
            {
                "operation_type": op,
                "status": r.get("status", ""),
                "missing": r.get("missing", []),
                "target_url": r.get("target_url"),
                "error_reason":         r.get("error_reason", r.get("error", "")),
                "validation_score":     r.get("validation_score", 0),
                "validation_breakdown": r.get("validation_breakdown", {}),
                "evidence":             r.get("evidence", []),
            }
            for op, r in results.items()
        ],
    }
