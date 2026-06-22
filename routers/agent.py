# api/routers/agent.py
import uuid
import datetime
from fastapi import APIRouter, HTTPException, Depends, Header, Body, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import os
import re

from api.core.firestore_client import get_db
from google.cloud import firestore
from api.core.agent_executor import execute_agent_task
from croniter import croniter
from api.routers.auth import verify_token

router = APIRouter(prefix="/api/agent", tags=["agent"])

APEX_PLANS = {"APEX", "ULTRA", "ULTRA_ADMIN", "ULTRA_MEMBER", "ultra_admin", "ultra_member"}
BLOCKED_PLANS = {"STARTER", "STANDARD"}
AGENT_TYPES = {"hp_update", "audit", "interview", "post_monitoring", "page_monitor"}
OPERATION_TYPES = {
    "entity_register", "entity_update", "media_replace",
    "text_update", "schedule_update", "price_update",
    "news_post", "blog_post", "post_monitoring",
    "interview_assist", "page_monitor", "offer_send",
    "recruit_reply", "recruit_inbox_scan",
    # status_update は廃止（情報更新に統合）。過去データ互換のため受理のみ継続。
    "status_update",
}

# P31: 自動化禁止Operation（いかなる場合もauto_enabled禁止）
AUTO_APPROVE_FORBIDDEN_OPERATIONS = {"price_update", "entity_delete"}
# P31: 自動化許可Operation（低リスクのみ）
AUTO_APPROVE_LOW_RISK_OPERATIONS = {"news_post", "schedule_update", "text_update"}
INDUSTRY_TEMPLATES = {
    "nightlife":  {"entity_name": "キャスト",     "schedule": "出勤",         "news": "ニュース",     "media": "写真"},
    "beauty":     {"entity_name": "スタッフ",     "schedule": "予約枠",       "news": "キャンペーン", "media": "スタッフ写真"},
    "retail":     {"entity_name": "商品",         "schedule": "営業時間",     "news": "お知らせ",     "media": "商品写真"},
    "realestate": {"entity_name": "物件",         "schedule": "空室状況",     "news": "新着物件",     "media": "物件写真"},
    "btob":       {"entity_name": "サービス",     "schedule": "セミナー",     "news": "ニュース",     "media": "資料"},
    "fitness":    {"entity_name": "講師",         "schedule": "レッスン",     "news": "キャンペーン", "media": "講師写真"},
    "other":      {"entity_name": "エンティティ", "schedule": "スケジュール", "news": "お知らせ",     "media": "メディア"},
}

INDUSTRY_ALIASES = {
    "real_estate": "realestate",
    "b2b":         "btob",
    "realestate":  "realestate",
    "btob":        "btob",
}

DEFAULT_AGENT_OPS = [
    {
        "op_id": "default_news_post",
        "display_name": "ニュース投稿",
        "category": "hp_update",
        "operation_type": "news_post",
        "entity_type": "content",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "title", "label": "タイトル", "type": "text", "required": False},
            {"key": "body", "label": "本文", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_text_update",
        "display_name": "テキスト更新",
        "category": "hp_update",
        "operation_type": "text_update",
        "entity_type": "content",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "text", "label": "更新テキスト", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_blog_post",
        "display_name": "店長ブログ",
        "category": "hp_update",
        "operation_type": "blog_post",
        "entity_type": "recruitment",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "title", "label": "タイトル", "type": "text", "required": False},
            {"key": "body", "label": "本文（店長ブログ・求人サイト）", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_media_replace",
        "display_name": "画像・資料差し替え",
        "category": "hp_update",
        "operation_type": "media_replace",
        "entity_type": "media",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "file_path", "label": "ファイルパス", "type": "file", "required": True},
        ]},
    },
    {
        "op_id": "default_schedule_update",
        "display_name": "予定更新",
        "category": "hp_update",
        "operation_type": "schedule_update",
        "entity_type": "schedule",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "schedule_value", "label": "予定/出勤内容", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_price_update",
        "display_name": "料金更新",
        "category": "hp_update",
        "operation_type": "price_update",
        "entity_type": "price",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "price_value", "label": "料金内容", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_entity_register",
        "display_name": "情報登録",
        "category": "hp_update",
        "operation_type": "entity_register",
        "entity_type": "entity",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "name", "label": "登録名", "type": "text", "required": True},
        ]},
    },
    {
        "op_id": "default_entity_update",
        "display_name": "情報更新",
        "category": "hp_update",
        "operation_type": "entity_update",
        "entity_type": "entity",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "value", "label": "更新内容", "type": "textarea", "required": True},
        ]},
    },
    {
        "op_id": "default_post_monitoring",
        "display_name": "投稿数監視",
        "category": "post_monitoring",
        "operation_type": "post_monitoring",
        "entity_type": "monitoring",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "monitoring_target", "label": "監視URL", "type": "text", "required": False},
            {"key": "cast_names", "label": "キャスト名（カンマ区切り）", "type": "text", "required": False},
            {"key": "monitoring_date", "label": "対象日（YYYY-MM-DD）", "type": "text", "required": False},
            {"key": "market_keywords", "label": "市場・マーケ監視キーワード", "type": "text", "required": False},
            {"key": "competitor_urls", "label": "競合URL（改行区切り）", "type": "textarea", "required": False},
        ]},
    },
    {
        "op_id": "default_page_monitor",
        "display_name": "日記・投稿監視",
        "category": "page_monitor",
        "operation_type": "page_monitor",
        "entity_type": "monitoring",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "check_points", "label": "確認ポイント（任意）例: キャラとの一致・投稿頻度", "type": "text", "required": False},
        ]},
    },
    {
        "op_id": "default_interview_assist",
        "display_name": "面接補助",
        "category": "interview",
        "operation_type": "interview_assist",
        "entity_type": "candidate",
        "industry": "generic",
        "active": True,
        "payload_schema": {"fields": [
            {"key": "use_case", "label": "面接目的", "type": "text", "required": True},
            {"key": "candidate_memo", "label": "候補者メモ", "type": "textarea", "required": False},
            {"key": "requirements", "label": "採用条件・確認観点", "type": "textarea", "required": False},
            {"key": "role_name", "label": "対象職種", "type": "text", "required": False},
        ]},
    },
    # ── スカウト型求人サイト専用 ops ──
    {
        "op_id": "default_offer_send",
        "display_name": "スカウト精査＋オファー送信",
        "category": "hp_update",
        "operation_type": "offer_send",
        "entity_type": "recruit",
        "industry": "generic",
        "active": True,
        "site_purpose": "scout",
        "payload_schema": {"fields": [
            {"key": "body", "label": "オファー文（空欄なら業務条件のひな形を使用）", "type": "textarea", "required": False},
            {"key": "max_send", "label": "最大送信件数", "type": "number", "required": False},
            {"key": "fi_scout_only", "label": "スカウト受付中の候補者のみ対象", "type": "boolean", "required": False},
            {"key": "fi_offer_unset_only", "label": "オファー未送信の候補者のみ", "type": "boolean", "required": False},
            {"key": "fi_free_text", "label": "フリーテキスト検索キーワード", "type": "text", "required": False},
        ]},
    },
    {
        "op_id": "default_recruit_inbox_scan",
        "display_name": "受信ボックスをスキャン",
        "category": "hp_update",
        "operation_type": "recruit_inbox_scan",
        "entity_type": "recruit",
        "industry": "generic",
        "active": True,
        "site_purpose": "scout",
        "payload_schema": {"fields": []},
    },
    {
        "op_id": "default_recruit_reply",
        "display_name": "候補者に返信",
        "category": "hp_update",
        "operation_type": "recruit_reply",
        "entity_type": "recruit",
        "industry": "generic",
        "active": True,
        "site_purpose": "scout",
        "payload_schema": {"fields": [
            {"key": "body", "label": "返信本文", "type": "textarea", "required": True},
            {"key": "reply_url", "label": "返信先URL（会話スレッドURL）", "type": "text", "required": False},
        ]},
    },
]

OPERATION_AGENT_TYPE_MAP = {
    "news_post": "hp_update",
    "blog_post": "hp_update",
    "offer_send": "hp_update",
    "recruit_reply": "hp_update",
    "recruit_inbox_scan": "hp_update",
    "text_update": "hp_update",
    "status_update": "hp_update",
    "media_replace": "hp_update",
    "schedule_update": "hp_update",
    "price_update": "hp_update",
    "entity_register": "hp_update",
    "entity_update": "hp_update",
    "post_monitoring": "post_monitoring",
    "page_monitor":    "page_monitor",
    "interview_assist": "interview",
}

def _validate_agent_operation_pair(agent_type: str, operation_type: str) -> None:
    expected = OPERATION_AGENT_TYPE_MAP.get(operation_type)
    if expected and agent_type != expected:
        raise HTTPException(
            status_code=400,
            detail=f"operation_type '{operation_type}' は agent_type '{expected}' 専用です。指定値: '{agent_type}'",
        )

def _normalize_industry(industry: str) -> str:
    key = (industry or "other").strip()
    return INDUSTRY_ALIASES.get(key, key if key in INDUSTRY_TEMPLATES else "other")


def _load_agent_ops_for_user(db, ctx: dict) -> list[dict]:
    import copy
    ops_by_id = {
        op["op_id"]: copy.deepcopy(op)
        for op in DEFAULT_AGENT_OPS
    }
    try:
        for d in db.collection("agent_ops").stream():
            op = d.to_dict() or {}
            op_id = op.get("op_id") or d.id
            base = ops_by_id.get(op_id, {})
            merged = {**base, **op, "op_id": op_id}
            ops_by_id[op_id] = merged
    except Exception as e:
        print(f"[AGENT_OPS_LOAD_ERROR] {type(e).__name__}:{e}", flush=True)

    user_plans = ctx.get("plans", set())
    is_admin = ctx.get("is_admin", False)
    result = []
    for op in ops_by_id.values():
        allowed = {str(x).upper() for x in op.get("allowed_plans", [])}
        if not is_admin and allowed and not user_plans.intersection(allowed):
            continue
        if op.get("created_at") and hasattr(op["created_at"], "isoformat"):
            op["created_at"] = op["created_at"].isoformat()
        if not op.get("display_name"):
            op["display_name"] = op.get("name") or op.get("op_name") or op.get("op_id", "")
        if not op.get("category"):
            op["category"] = "hp_update"
        if not op.get("industry"):
            op["industry"] = "generic"
        if not op.get("payload_schema"):
            op["payload_schema"] = {"fields": []}
        missing = []
        if not op.get("display_name"): missing.append("display_name")
        if not op.get("operation_type"): missing.append("operation_type")
        if not op.get("entity_type"): missing.append("entity_type")
        if not isinstance(op.get("payload_schema"), dict): missing.append("payload_schema")
        if missing:
            op["active"] = False
            op["invalid_reason"] = f"定義不足: {', '.join(missing)}"
        elif op.get("active") is None:
            op["active"] = True
        result.append(op)
    result.sort(key=lambda x: (x.get("active") is False, x.get("display_name", ""), x.get("op_id", "")))
    return result


def _fallback_plan_from_instruction(instruction: str) -> dict:
    t = (instruction or "").lower()
    rules = [
        # スカウト系を最優先（面接・採用より前に判定）
        (r"スカウト.*精査|精査.*候補|オファー送信|候補者.*オファー|offer.?send|スカウト.*送信", "offer_send", "スカウト精査＋オファー一括送信タスクを作成します。"),
        (r"受信ボックス|受信.*スキャン|inbox.*スキャン|メッセージ.*確認|recruit.?inbox|着信.*確認|応募.*確認|応募者.*連絡|新着.*確認|受信.*確認|応募チェック", "recruit_inbox_scan", "受信ボックス監視タスクを作成します。応募者・候補者からのメッセージを確認します。"),
        (r"候補者.*返信|スカウト.*返信|recruit.?reply|求人.*返信|返信.*候補|応募者.*返信|応募.*返信|連絡.*返信|問い合わせ.*返信", "recruit_reply", "返信送信タスクを作成します。応募者・候補者の会話スレッドに返信します。"),
        (r"面接|ヒアリング|面談|interview", "interview_assist", "面接メモ作成タスクを作成します。質問案・評価軸・判断メモを整理します。"),
        (r"投稿数|投稿頻度|写メ日記|未投稿|監視|diary count|post count", "post_monitoring", "投稿数監視タスクを作成します。"),
        (r"写真|画像|差し替え|media|アップロード|サムネ", "media_replace", "画像・資料差し替えタスクを作成します。"),
        (r"出勤|予定|スケジュール|schedule|シフト|カレンダー", "schedule_update", "予定更新タスクを作成します。"),
        (r"ニュース|お知らせ|投稿|news|ブログ|日記|写メ", "news_post", "ニュース投稿タスクを作成します。"),
        (r"料金|価格|price|コース|費用", "price_update", "料金更新タスクを作成します。"),
        (r"ステータス|status|状態|公開|非公開|表示|非表示|停止|有効|無効", "status_update", "ステータス更新タスクを作成します。"),
        (r"登録|追加|新規|entity|キャスト追加|スタッフ追加", "entity_register", "情報登録タスクを作成します。"),
        (r"編集|変更|修正|プロフィール|紹介|本文|テキスト|文章|説明|更新|text", "text_update", "テキスト更新タスクを作成します。"),
    ]
    import re
    for pattern, op, preview in rules:
        if re.search(pattern, t):
            return {"ok": True, "ready": True, "media_name": None, "op_id": None, "operation_type": op, "payload": {}, "preview": preview, "question": None}
    return {
        "ok": True,
        "ready": False,
        "media_name": None,
        "op_id": None,
        "operation_type": None,
        "payload": {},
        "preview": "",
        "question": "操作タイプを特定できません。投稿数監視、投稿、画像差し替え、予定更新、料金更新、ステータス更新、情報登録、情報更新のどれかを含めてください。",
    }


def _goal_mapping_ready_summary(mapping: dict) -> dict:
    ready_ops = set()
    op_map = mapping.get("operation_mappings") or {}
    steps_by_type = mapping.get("operation_steps_by_type") or {}
    if isinstance(op_map, dict):
        for op, st in op_map.items():
            if not _operation_mapping_is_production_ready(st):
                continue
            steps = steps_by_type.get(op) or st.get("steps")
            step_count = st.get("step_count")
            has_steps = bool(steps) if isinstance(steps, list) else bool(step_count or steps)
            if has_steps or st.get("selectors"):
                ready_ops.add(str(op))

    menu_ready = 0
    for item in ((mapping.get("manual_menu_scan_results") or {}).get("items") or []):
        if not isinstance(item, dict):
            continue
        for op, st in (item.get("operations") or {}).items():
            if not isinstance(st, dict) or st.get("status") != "READY" or st.get("production_ready") is not True:
                continue
            steps = st.get("steps")
            step_count = st.get("step_count")
            has_steps = bool(steps) if isinstance(steps, list) else bool(step_count or steps)
            if has_steps:
                ready_ops.add(str(op))
                menu_ready += 1
    return {
        "mapping_id": mapping.get("mapping_id"),
        "media_name": mapping.get("media_name", ""),
        "industry": _normalize_industry(mapping.get("industry", "other")),
        "ready_ops": sorted(ready_ops),
        "ready_count": len(ready_ops),
        "menu_ready_count": menu_ready,
        "has_credential": bool(mapping.get("credential_secret_name")),
        "verified": bool(mapping.get("last_verified_at")),
    }


def _goal_has(text: str, words: list[str]) -> bool:
    return any(w.lower() in text for w in words)


def _goal_extract_context(goal: str) -> dict:
    import re
    text = goal or ""
    urls = [u.rstrip("。、)）]】") for u in re.findall(r"https?://\S+", text)]
    date_match = re.search(r"(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?|今日|明日|昨日|今週|来週|毎日|毎週)", text)
    top_match = re.search(r"(?:上位|最大|top|TOP)\s*(\d+)|(\d+)\s*件", text)
    max_items = 1
    if top_match:
        max_items = max(1, min(50, int(top_match.group(1) or top_match.group(2) or 1)))

    known_media_terms = ["スーモ", "SUUMO", "ホームズ", "HOMES", "LIFULL", "CHINTAI", "レインズ", "Instagram", "X", "Twitter", "Google", "ホットペッパー"]
    mentioned_terms = [x for x in known_media_terms if x.lower() in text.lower()]

    keyword_pool = []
    for kw in ["新人", "イベント", "割引", "キャンペーン", "予約", "本指名", "口コミ", "ランキング", "SNS", "2LDK", "駅近", "家賃", "エリア"]:
        if kw.lower() in text.lower():
            keyword_pool.append(kw)

    return {
        "urls": urls,
        "date_hint": date_match.group(0) if date_match else "",
        "max_items": max_items,
        "mentioned_media_terms": mentioned_terms,
        "keywords": keyword_pool,
        "raw_conditions": text,
    }


# site_purpose → 推奨 operation_type（全機能連動用）
_SITE_PURPOSE_OPS: dict[str, list[str]] = {
    "scout":   ["offer_send", "recruit_inbox_scan", "recruit_reply"],
    "reply":   ["recruit_reply", "page_monitor"],
    "post":    ["blog_post", "news_post", "text_update", "entity_register", "entity_update"],
    "monitor": ["page_monitor", "recruit_inbox_scan", "post_monitoring"],
    "other":   ["news_post", "blog_post", "text_update", "entity_register", "entity_update", "page_monitor"],
}

_SITE_PURPOSE_NEXT_ACTIONS: dict[str, list[dict]] = {
    "scout": [
        {"op": "offer_send",         "label": "🎯 スカウト精査＋オファー",   "note": "候補者をAI精査してオファー一括送信"},
        {"op": "recruit_inbox_scan", "label": "📬 受信ボックスをスキャン",   "note": "候補者の返信を確認して会話スレッド更新"},
        {"op": "recruit_reply",      "label": "💬 返信を送信",               "note": "会話スレッドに返信"},
    ],
    "reply": [
        {"op": "recruit_reply", "label": "💬 返信を送信",       "note": ""},
        {"op": "page_monitor",  "label": "👁 受信状況を確認",   "note": ""},
    ],
    "post": [
        {"op": "blog_post",   "label": "📝 ブログを投稿",   "note": ""},
        {"op": "news_post",   "label": "📰 ニュースを投稿", "note": ""},
        {"op": "text_update", "label": "✏️ 情報を更新",     "note": ""},
    ],
    "monitor": [
        {"op": "page_monitor",       "label": "👁 ページを監視",       "note": ""},
        {"op": "recruit_inbox_scan", "label": "📬 求人受信を確認",     "note": ""},
    ],
}


def _build_goal_plan(goal: str, mappings: list[dict], tasks: list[dict], batches: list[dict], cross_tasks: list[dict], schedules: list[dict]) -> dict:
    text = (goal or "").strip()
    t = text.lower()
    extracted = _goal_extract_context(text)
    mapping_rows = [_goal_mapping_ready_summary(m) for m in mappings]
    ready_total = sum(r["ready_count"] + r["menu_ready_count"] for r in mapping_rows)
    verified_count = sum(1 for r in mapping_rows if r["verified"])
    monitoring_tasks = [x for x in tasks if x.get("agent_type") == "post_monitoring" or x.get("operation_type") == "post_monitoring"]
    pending_tasks = [x for x in tasks if x.get("status") == "PENDING"]
    blocked_tasks = [x for x in tasks if x.get("status") in {"FAILED", "BLOCKED", "WAITING_MAPPING", "WAITING_EXECUTOR"}]

    mode = "operation_task"
    route_tab = "create"
    confidence = 0.55
    summary = "ゴールから実行タスクを作成します。"

    cross_terms = ["媒体間", "クロスメディア", "移す", "コピー", "転載", "転記", "他媒体", "別媒体", "アップ", "suumo", "スーモ", "homes", "ホームズ", "chintai", "レインズ", "上位5", "上位"]
    monitoring_terms = ["監視", "投稿数", "投稿頻度", "未投稿", "写メ日記", "日記", "競合", "市場", "マーケ", "ランキング", "sns"]
    mapping_terms = ["接続", "ログイン", "媒体登録", "マッピング", "html", "メニュー", "検出", "ready", "未検出", "解析"]
    interview_terms = ["面接", "ヒアリング", "面談"]
    schedule_terms = ["毎日", "毎週", "定期", "予約", "スケジュール", "自動運用", "巡回"]
    health_terms = ["止まった", "失敗", "エラー", "原因", "ログ", "異常", "なぜ", "blocked", "failed"]
    growth_terms = ["売上", "集客", "予約を増", "問い合わせ", "反響", "応募を増", "選ばれる", "改善", "伸ば", "上げたい", "勝ちたい"]
    scout_terms = ["スカウト", "scout", "オファー送信", "候補者", "候補者を探", "精査", "スカウト型", "採用候補", "応募者", "採用"]
    recruit_terms = ["求人", "採用", "応募", "返信", "求職", "オファー", "recruit"]
    mentioned_mappings = []
    for r in mapping_rows:
        name = str(r.get("media_name") or "")
        if name and name.lower() in t:
            mentioned_mappings.append(r)
    # Step 4: 求人系マッピングを検出（scout=スカウト型 / reply=返信型 / monitor=監視型を含む）
    scout_mappings = [
        m for m in mappings
        if (m.get("business_conditions") or {}).get("site_purpose") in {"scout", "reply", "monitor"}
    ]

    if _goal_has(t, health_terms):
        mode, route_tab, confidence = "health_management", "health", 0.82
        summary = "失敗・未検出・停止理由を確認し、復旧に必要な設定へ誘導します。"
    elif _goal_has(t, cross_terms):
        mode, route_tab, confidence = "cross_media", "cross", 0.86
        summary = "取得元の情報を別媒体のREADY操作へ展開するクロスメディア案件として扱います。"
    elif _goal_has(t, monitoring_terms) or _goal_has(t, growth_terms):
        mode, route_tab, confidence = "market_monitoring", "monitoring", 0.84
        summary = "投稿量・未投稿・競合・マーケティング信号を監視し、改善アクションへつなげます。"
    elif _goal_has(t, scout_terms):
        mode, route_tab, confidence = "scout_recruit", "create", 0.88
        summary = "スカウト型求人の候補者精査→オファー→返信→面接確定の自動フローを実行します。"
    elif _goal_has(t, recruit_terms):
        mode, route_tab, confidence = "scout_recruit", "create", 0.75
        summary = "求人対応（オファー・返信・掲載文）をAI支援で実行します。"
    elif _goal_has(t, interview_terms):
        mode, route_tab, confidence = "interview_assist", "interview", 0.78
        summary = "面接・応募対応の質問、評価軸、判断メモを整理する補助案件として扱います。"
    elif _goal_has(t, schedule_terms):
        mode, route_tab, confidence = "scheduled_operations", "schedule", 0.72
        summary = "定期実行・予約実行として扱います。"
    elif _goal_has(t, mapping_terms) or not mappings:
        mode, route_tab, confidence = "mapping_setup", "sites", 0.8
        summary = "媒体登録・HTML/DOM解析・READY化を先に進める必要があります。"

    if not mappings and mode != "mapping_setup":
        route_tab = "sites"
        summary = "媒体が未登録のため、まず媒体マッピングから開始します。"

    op_plan = _fallback_plan_from_instruction(text)
    can_create_task = bool(op_plan.get("operation_type")) and mode in {"operation_task", "market_monitoring", "scheduled_operations", "interview_assist", "scout_recruit"}
    if mode == "scout_recruit":
        # fallback_plan が既にスカウト系 op を特定していれば優先、なければ offer_send をデフォルトに
        _scout_ops = {"offer_send", "recruit_inbox_scan", "recruit_reply"}
        _scout_op = op_plan.get("operation_type") if op_plan.get("operation_type") in _scout_ops else "offer_send"
        _scout_preview = {
            "offer_send":         "スカウト精査→オファー一括送信タスクを作成します。",
            "recruit_inbox_scan": "受信ボックス監視タスクを作成します。候補者からの返信を確認します。",
            "recruit_reply":      "候補者への返信タスクを作成します。",
        }.get(_scout_op, "スカウト精査→オファー一括送信タスクを作成します。")
        op_plan = {**op_plan, "operation_type": _scout_op, "ready": True, "preview": _scout_preview}
        can_create_task = True
    if mode == "market_monitoring":
        op_plan = {**op_plan, "operation_type": "post_monitoring", "ready": True, "preview": "投稿・市場監視タスクを作成します。"}
        can_create_task = True

    source_url = extracted["urls"][0] if extracted["urls"] else ""
    monitoring_keywords = extracted["keywords"] or ["新人", "イベント", "割引", "キャンペーン", "予約", "本指名", "SNS"]
    prefill = {
        "monitoring": {
            "monitoring_target": source_url,
            "monitoring_date": extracted["date_hint"],
            "market_keywords": ",".join(monitoring_keywords),
        },
        "cross_media": {
            "instruction": text,
            "source_mode": "public_url" if source_url else "manual_payload",
            "source_url": source_url,
            "query": text,
            "max_items": extracted["max_items"],
            "target_operation_type": op_plan.get("operation_type") or "entity_register",
        },
        "task": {
            "instruction": text,
            "operation_type": op_plan.get("operation_type"),
            "payload": op_plan.get("payload") or {},
        },
        "interview": {
            "use_case": text,
            "requirements": ",".join(extracted["keywords"] or ["継続性", "接客適性", "ルール理解", "条件一致"]),
        },
        "schedule": {
            "instruction": text,
            "operation_type": op_plan.get("operation_type") or "news_post",
            "payload": op_plan.get("payload") or {"body": text},
        },
        "scout_recruit": {
            "recruit_mode": "offer",
            "instruction": text,
            "max_send": 5,
            "filter_intent": {"scout_only": True, "offer_unset_only": True},
            "scout_mapping_ids": [m.get("mapping_id") for m in scout_mappings if m.get("mapping_id")],
        },
    }

    missing_capabilities = []
    if not mappings:
        missing_capabilities.append("媒体マッピング未登録")
    if mappings and ready_total == 0 and mode not in {"market_monitoring", "health_management"}:
        missing_capabilities.append("READY+steps未作成")
    if mode == "cross_media" and not source_url and not mentioned_mappings:
        missing_capabilities.append("取得元URLまたは取得元媒体が未指定")
    if mode in {"operation_task", "scheduled_operations"} and not can_create_task:
        missing_capabilities.append("操作種別または必須payloadが不足")
    if mode == "scout_recruit" and not scout_mappings:
        missing_capabilities.append("求人対応マッピング未設定（🏢業務条件でsite_purpose=scout/reply/monitorを設定してください）")

    if not mappings:
        autonomy_level = "FOUNDATION_REQUIRED"
    elif missing_capabilities:
        autonomy_level = "NEEDS_SETUP"
    elif pending_tasks or blocked_tasks:
        autonomy_level = "NEEDS_MANAGEMENT_REVIEW"
    elif mode in {"cross_media", "market_monitoring", "operation_task", "scheduled_operations", "scout_recruit"}:
        autonomy_level = "READY_TO_ORCHESTRATE"
    else:
        autonomy_level = "READY_TO_GUIDE"

    def _tool(name: str, tab: str, score: int, reason: str) -> dict:
        return {"tool": name, "tab": tab, "score": max(0, min(100, score)), "reason": reason}

    tool_selection = [
        _tool("媒体マッピング", "sites", 100 if mode == "mapping_setup" or not mappings else (70 if ready_total == 0 else 35), "媒体登録・HTML/DOM解析・READY化の土台"),
        _tool("媒体クロスメディア", "cross", 100 if mode == "cross_media" else (65 if _goal_has(t, growth_terms) else 20), "取得元情報を別媒体へ展開"),
        _tool("投稿・市場監視", "monitoring", 100 if mode == "market_monitoring" else (75 if _goal_has(t, growth_terms) else 25), "投稿量・未投稿・競合・マーケティング信号"),
        _tool("🧲 求人対応", "cross", 100 if mode == "scout_recruit" else (80 if scout_mappings else (50 if _goal_has(t, recruit_terms) else 15)), "スカウト精査・オファー・返信・面接確定の自動化（🏢業務条件が連動）"),
        _tool("面接補助", "interview", 100 if mode == "interview_assist" else 20, "候補者メモから質問・評価軸・合否判断材料を整理"),
        _tool("タスク生成", "create", 95 if mode in {"operation_task", "scheduled_operations", "interview_assist"} else 40, "自然文から承認制タスクへ変換"),
        _tool("スケジュール", "schedule", 90 if mode == "scheduled_operations" else 35, "定期実行・予約実行"),
        _tool("一括実行", "batch", 80 if "複数" in text or "全媒体" in text or "まとめて" in text else 30, "複数媒体への同一操作"),
        _tool("異常確認", "health", 100 if mode == "health_management" or blocked_tasks else 30, "停止理由・未検出・設定不足の確認"),
    ]
    tool_selection.sort(key=lambda x: x["score"], reverse=True)

    phases = [
        {"phase": "media_foundation", "title": "媒体を登録し、ログイン/公開URLを保持", "tab": "sites", "status": "OK" if mappings else "NEEDS_ACTION"},
        {"phase": "structure_ready", "title": "HTML/DOMを解析してREADY+stepsを作成", "tab": "sites", "status": "OK" if ready_total else "NEEDS_ACTION"},
        {"phase": "goal_execution", "title": "ゴールに合うタスクを作成し承認へ回す", "tab": route_tab, "status": "READY" if can_create_task or mode in {"cross_media", "mapping_setup", "health_management"} else "NEEDS_INFO"},
        {"phase": "monitoring", "title": "投稿・市場・競合を継続監視", "tab": "monitoring", "status": "OK" if monitoring_tasks else ("RECOMMENDED" if mode in {"market_monitoring", "cross_media"} else "OPTIONAL")},
        {"phase": "management", "title": "承認待ち・失敗・スケジュールを管理", "tab": "tasks", "status": "NEEDS_REVIEW" if pending_tasks or blocked_tasks else "OK"},
    ]

    actions = []
    if not mappings:
        actions.append({"label": "媒体を登録する", "tab": "sites", "status": "NEEDS_ACTION", "reason": "自動実行・監視・クロスメディアの土台がありません。"})
    elif ready_total == 0 and mode not in {"market_monitoring", "health_management"}:
        actions.append({"label": "HTMLメニュー/DOM深掘りでREADY化する", "tab": "sites", "status": "NEEDS_ACTION", "reason": "出力先として使えるREADY操作がありません。"})
    if mode == "cross_media":
        actions.append({"label": "媒体クロスメディアを開く", "tab": "cross", "status": "NEXT", "reason": "取得元と出力先READY媒体を選びます。"})
    elif mode == "market_monitoring":
        actions.append({"label": "投稿・市場監視を開く", "tab": "monitoring", "status": "NEXT", "reason": "投稿量、未投稿、競合URL、マーケキーワードを監視します。"})
    elif mode == "interview_assist":
        actions.append({"label": "面接補助を開く", "tab": "interview", "status": "NEXT", "reason": "面接目的、候補者メモ、採用条件から質問と評価軸を作ります。"})
    elif mode == "health_management":
        actions.append({"label": "異常確認を開く", "tab": "health", "status": "NEXT", "reason": "止まった箇所、未検出、設定不足を確認します。"})
    elif can_create_task:
        actions.append({"label": "ゴールからタスク生成", "tab": "create", "status": "NEXT", "reason": op_plan.get("preview") or "自然文から実行タスクを作成します。"})
    if pending_tasks:
        actions.append({"label": "承認待ちを処理", "tab": "tasks", "status": "REVIEW", "reason": f"承認待ちが{len(pending_tasks)}件あります。"})
    if blocked_tasks:
        actions.append({"label": "停止理由を確認", "tab": "health", "status": "REVIEW", "reason": f"要確認タスクが{len(blocked_tasks)}件あります。"})

    return {
        "ok": True,
        "goal": text,
        "mode": mode,
        "route_tab": route_tab,
        "confidence": confidence,
        "summary": summary,
        "autonomy_level": autonomy_level,
        "can_create_task": can_create_task,
        "operation_plan": op_plan,
        "extracted": extracted,
        "prefill": prefill,
        "missing_capabilities": missing_capabilities,
        "tool_selection": tool_selection,
        "readiness": {
            "mappings": len(mappings),
            "verified_mappings": verified_count,
            "ready_operations": ready_total,
            "tasks": len(tasks),
            "pending_tasks": len(pending_tasks),
            "blocked_tasks": len(blocked_tasks),
            "batches": len(batches),
            "cross_tasks": len(cross_tasks),
            "monitoring_tasks": len(monitoring_tasks),
            "schedules": len(schedules),
        },
        "media": mapping_rows[:20],
        "workstream": phases,
        "next_actions": actions,
        "mapping_purpose_summary": [
            {
                "mapping_id": m.get("mapping_id"),
                "media_name": m.get("media_name"),
                "industry": m.get("industry") or "other",
                "site_purpose": (m.get("business_conditions") or {}).get("site_purpose") or "other",
                "suggested_ops": _SITE_PURPOSE_OPS.get(
                    (m.get("business_conditions") or {}).get("site_purpose") or "other",
                    _SITE_PURPOSE_OPS["other"]
                ),
                "next_actions": _SITE_PURPOSE_NEXT_ACTIONS.get(
                    (m.get("business_conditions") or {}).get("site_purpose") or "other",
                    []
                ),
            }
            for m in mappings
        ],
    }


def _resolve_agent_user_context(user: dict) -> dict:
    """userdictからtenant_id/role/plansを正規化して返す補助関数"""
    uid = user.get("uid") or user.get("user_id") or user.get("sub") or ""
    tenant_id = user.get("tenant_id") or uid or "default"
    role = str(user.get("role", "") or "").lower()
    is_unlimited = bool(user.get("is_unlimited", False))
    # C-3: JWTのroleクレームが失効後も古い権限を保持するリスクを防ぐため、
    # Firestoreから最新のroleを取得してJWTのclaimより優先する
    if uid:
        try:
            _fs_db = get_db()
            if _fs_db is not None:
                _fs_snap = _fs_db.collection("users").document(uid).get()
                if _fs_snap.exists:
                    _fs_u = _fs_snap.to_dict() or {}
                    _fs_role = str(_fs_u.get("role", "") or "").lower()
                    if _fs_role:
                        role = _fs_role
                    _fs_unlimited = _fs_u.get("is_unlimited")
                    if _fs_unlimited is not None:
                        is_unlimited = bool(_fs_unlimited)
        except Exception as _c3_e:
            print(f"[ROLE_REFRESH_ERROR] uid={uid} err={type(_c3_e).__name__}", flush=True)
    plans = {
        str(user.get("plan", "") or "").upper(),
        str(user.get("subscription_plan", "") or "").upper(),
        str(user.get("ai_tier", "") or "").upper(),
        str(user.get("tier", "") or "").upper(),
        role.upper(),
    }
    is_admin = (role == "admin")
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


def _is_agent_admin(user: dict) -> bool:
    return _resolve_agent_user_context(user)["is_admin"]


def _can_manage_agent_permissions(user: dict) -> bool:
    ctx = _resolve_agent_user_context(user)
    if ctx["is_admin"] or ctx["is_unlimited"] or any(p in APEX_PLANS for p in ctx["plans"]):
        return True
    uid = ctx.get("uid") or ""
    if not uid:
        return False
    try:
        db = get_db()
        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return False
        u = doc.to_dict() or {}
        role = str(u.get("role", "") or "").lower()
        plans = {
            str(u.get("plan", "") or "").upper(),
            str(u.get("subscription_plan", "") or "").upper(),
            str(u.get("ai_tier", "") or "").upper(),
            str(u.get("tier", "") or "").upper(),
            role.upper(),
        }
        return role == "admin" or bool(u.get("is_unlimited", False)) or any(p in APEX_PLANS for p in plans)
    except Exception as e:
        print(f"[agent_permissions_manage] check error: {type(e).__name__}", flush=True)
        return False


def _agent_credential_secret_name(tenant_id: str, mapping_id: str) -> str:
    return f"agent-media-{tenant_id}-{mapping_id}"


def _assert_tenant_access(resource: dict, user: dict, detail: str = "このリソースへのアクセス権がありません") -> dict:
    ctx = _resolve_agent_user_context(user)
    if resource.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail=detail)
    return ctx


def _assert_url_in_mapping_scope(mapping: dict, url: str, field_name: str = "url") -> None:
    if not url:
        return
    try:
        from urllib.parse import urlparse
        parsed = urlparse(str(url))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("invalid_url")
        target_host = parsed.netloc.lower().split("@")[-1].split(":")[0]
        allowed_hosts = set()
        for key in ("media_url", "login_url", "detected_login_url"):
            raw = str(mapping.get(key) or "")
            if not raw:
                continue
            base = urlparse(raw if "://" in raw else "https://" + raw)
            if base.netloc:
                allowed_hosts.add(base.netloc.lower().split("@")[-1].split(":")[0])
        if not allowed_hosts:
            raise HTTPException(status_code=400, detail=f"{field_name}の許可元ホストを判定できません")
        allowed = any(target_host == host or target_host.endswith("." + host) for host in allowed_hosts)
        if not allowed:
            raise HTTPException(status_code=400, detail=f"{field_name}は登録媒体のホスト外です")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field_name}が不正です")


def _menu_scope_root(pathname: str) -> str:
    parts = [p for p in str(pathname or "/").split("/") if p]
    if not parts:
        return "/"
    first = parts[0]
    if "." in first:
        return "/"
    return f"/{first}/"


def _url_in_mapping_menu_scope(mapping: dict, url: str) -> bool:
    if not url:
        return False
    try:
        from urllib.parse import urlparse
        target = urlparse(str(url))
        if target.scheme not in ("http", "https") or not target.netloc:
            return False
        target_host = target.netloc.lower().split("@")[-1].split(":")[0]
        bases = [str(mapping.get(k) or "") for k in ("media_url", "login_url", "detected_login_url") if mapping.get(k)]
        if not bases:
            return True
        for raw in bases:
            base = urlparse(raw if "://" in raw else "https://" + raw)
            if not base.netloc:
                continue
            base_host = base.netloc.lower().split("@")[-1].split(":")[0]
            if not (target_host == base_host or target_host.endswith("." + base_host)):
                continue
            root = _menu_scope_root(base.path)
            if root == "/" or target.path == root[:-1] or target.path.startswith(root):
                return True
        return False
    except Exception:
        return False


def _menu_item_has_editable_evidence(item: dict) -> bool:
    diag = item.get("diagnostics") or {}
    if not isinstance(diag, dict):
        return False
    if int(diag.get("editable_pages_count") or 0) > 0:
        return True
    for page in diag.get("page_evidence") or []:
        if not isinstance(page, dict):
            continue
        if any(int(page.get(k) or 0) > 0 for k in ("forms", "inputs", "buttons", "selects", "textareas", "file_inputs")):
            return True
    return False


def _normalize_menu_item_for_response(mapping: dict, item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or item.get("absolute_url") or item.get("href") or "")
    if not _url_in_mapping_menu_scope(mapping, url):
        return None
    row = dict(item)
    row_confirmed = bool(
        row.get("production_ready") is True
        or row.get("confirmation_status") == "AI_CONFIRMED"
        or row.get("source") in CONFIRMED_OPERATION_SOURCES
    )
    if row.get("status") == "READY" and not row_confirmed:
        row["stored_status"] = row.get("status")
        row["status"] = "NEEDS_REVIEW"
        row["candidate_only"] = True
        row["production_ready"] = False
        row["message"] = row.get("message") or "AI整備前の候補です。実行にはAI整備で対象ページ・入力項目・保存操作を保存する必要があります"
        diag = dict(row.get("diagnostics") or {})
        diag["response_adjusted_status"] = "NEEDS_REVIEW"
        row["diagnostics"] = diag
    if row.get("status") == "NO_OPERATION" and _menu_item_has_editable_evidence(row):
        row["stored_status"] = row.get("status")
        row["status"] = "NEEDS_REVIEW"
        row["message"] = row.get("message") or "編集DOMは検出済み。AI整備で項目と保存操作を確定する必要があります"
        diag = dict(row.get("diagnostics") or {})
        diag["response_adjusted_status"] = "NEEDS_REVIEW"
        row["diagnostics"] = diag
    return row


def _filter_menu_items_for_response(mapping: dict, items: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    seen: set[str] = set()
    for item in items or []:
        row = _normalize_menu_item_for_response(mapping, item)
        if not row:
            continue
        norm = _normalize_menu_scan_url(row.get("canonical_url") or row.get("url") or row.get("absolute_url") or row.get("href") or "")
        if not norm or norm in seen:
            continue
        seen.add(norm)
        filtered.append(row)
    return filtered


def _filter_mapping_menu_scan_for_response(mapping: dict) -> dict:
    scan = mapping.get("manual_menu_scan_results") or {}
    if not isinstance(scan, dict):
        return mapping
    raw_items = [i for i in (scan.get("items") or []) if isinstance(i, dict)]
    if not raw_items:
        return mapping
    filtered = _filter_menu_items_for_response(mapping, raw_items)
    next_scan = dict(scan)
    next_scan["items"] = filtered
    next_scan["summary"] = _menu_scan_summary(_compact_menu_scan_items(filtered, parent=True)) if filtered else {}
    next_scan["response_filtered"] = True
    next_scan["response_filtered_count"] = max(0, len(raw_items) - len(filtered))
    mapping["manual_menu_scan_results"] = next_scan
    return mapping


CONFIRMED_OPERATION_SOURCES = {"AI_CONFIRMED", "TASK_OVERRIDE"}


def _operation_mapping_is_confirmed(op_data: dict) -> bool:
    if not isinstance(op_data, dict):
        return False
    source = str(op_data.get("source") or "")
    status = str(op_data.get("confirmation_status") or "")
    return (
        bool(op_data.get("production_ready") is True)
        or status in {"AI_CONFIRMED"}
        or source in CONFIRMED_OPERATION_SOURCES
    )


def _operation_mapping_is_production_ready(op_data: dict) -> bool:
    return bool(
        isinstance(op_data, dict)
        and op_data.get("status") == "READY"
        and op_data.get("executable") is True
        and _operation_mapping_is_confirmed(op_data)
    )


def _operation_has_ai_production_evidence(op_data: dict, steps: list[dict], failed_required: bool) -> bool:
    if not isinstance(op_data, dict):
        return False
    if op_data.get("status") != "READY" or op_data.get("executable") is not True:
        return False
    if not op_data.get("target_url") or op_data.get("missing"):
        return False
    if not steps or failed_required:
        return False
    selectors = op_data.get("selectors") or {}
    form_schema = op_data.get("form_schema") or {}
    form_fields = 0
    if isinstance(form_schema, dict):
        form_fields = int(form_schema.get("fields_count") or len(form_schema.get("fields") or []) or 0)
    if not selectors and form_fields <= 0:
        return False
    return True


def _annotate_operation_mappings_for_response(mapping: dict) -> dict:
    op_maps = mapping.get("operation_mappings") or {}
    if not isinstance(op_maps, dict):
        return mapping
    annotated: dict = {}
    for op, data in op_maps.items():
        if not isinstance(data, dict):
            annotated[op] = data
            continue
        row = dict(data)
        confirmed = _operation_mapping_is_confirmed(row)
        production_ready = bool(row.get("status") == "READY" and row.get("executable") is True and confirmed)
        row["confirmed"] = confirmed
        row["production_ready"] = production_ready
        if not confirmed:
            row["candidate_only"] = True
            row["execution_block_reason"] = "AI整備前の候補のため本番実行不可"
        annotated[op] = row
    mapping["operation_mappings"] = annotated
    return mapping


def _delete_legacy_operation_mappings_for_ai_contract(db, mapping_id: str, mapping: dict) -> dict:
    """Remove non-AI-confirmed operation mappings from the executable contract."""
    op_maps = mapping.get("operation_mappings") or {}
    if not isinstance(op_maps, dict) or not op_maps:
        return mapping
    legacy_ops = [
        op for op, row in op_maps.items()
        if isinstance(row, dict) and not _operation_mapping_is_production_ready(row)
    ]
    if not legacy_ops:
        mapping["mapping_contract_version"] = mapping.get("mapping_contract_version") or "ai_confirmed_v1"
        return mapping

    updates: dict[str, object] = {
        "mapping_contract_version": "ai_confirmed_v1",
        "legacy_operation_mappings_deleted_at": datetime.datetime.utcnow(),
        "legacy_operation_mappings_deleted_count": len(legacy_ops),
        "updated_at": datetime.datetime.utcnow(),
    }
    for op in legacy_ops:
        updates[f"operation_mappings.{op}"] = firestore.DELETE_FIELD
        updates[f"operation_steps_by_type.{op}"] = firestore.DELETE_FIELD
    try:
        db.collection("media_mappings").document(mapping_id).update(updates)
        next_ops = dict(op_maps)
        for op in legacy_ops:
            next_ops.pop(op, None)
        mapping["operation_mappings"] = next_ops
        steps = dict(mapping.get("operation_steps_by_type") or {})
        for op in legacy_ops:
            steps.pop(op, None)
        mapping["operation_steps_by_type"] = steps
        mapping["mapping_contract_version"] = "ai_confirmed_v1"
        mapping["legacy_operation_mappings_deleted_count"] = len(legacy_ops)
        print(f"[AI_CONTRACT_LEGACY_OPS_DELETED] mapping_id={mapping_id} ops={legacy_ops}", flush=True)
    except Exception as e:
        print(f"[AI_CONTRACT_LEGACY_OPS_DELETE_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    return mapping


def _parse_progress_time(value):
    if not value:
        return None
    if hasattr(value, "replace") and hasattr(value, "isoformat"):
        return value.replace(tzinfo=None)
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _guard_mapping_scan_not_running(db, mapping_id: str, mapping: dict, kind: str = "", stale_minutes: int = 20) -> None:
    progress = mapping.get("scan_progress") or {}
    if progress.get("status") != "RUNNING":
        return
    now = datetime.datetime.utcnow()
    stamp = _parse_progress_time(progress.get("updated_at") or progress.get("started_at"))
    is_stale = not stamp or (now - stamp).total_seconds() > stale_minutes * 60
    if is_stale:
        try:
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress.status": "FAILED",
                "scan_progress.error": "stale RUNNING scan recovered before new request",
                "scan_progress.recovered_at": now.isoformat(),
                "scan_progress.updated_at": now.isoformat(),
            })
        except Exception as e:
            print(f"[SCAN_PROGRESS_STALE_RECOVERY_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
        return
    raise HTTPException(
        status_code=409,
        detail=f"スキャン実行中です。完了後に再実行してください: {progress.get('kind') or kind or 'scan'}",
    )


def _enforce_agent_permissions(ctx: dict, agent_type: str, operation_type: str):
    if ctx["is_admin"] or ctx["is_unlimited"]:
        return
    if any(p in APEX_PLANS for p in ctx["plans"]):
        return
    tenant_id = ctx["tenant_id"]
    perm = _get_agent_permissions(tenant_id)
    # P31: tenant許可または低リスクoperationの自動昇格で利用可。
    auto_pass = False
    if operation_type and operation_type not in AUTO_APPROVE_FORBIDDEN_OPERATIONS:
        _p31_ops = perm.get("operations", {})
        _p31_op_data = _p31_ops.get(operation_type, {})
        auto_pass = bool(_p31_op_data.get("auto_enabled", False))
        if auto_pass:
            print(f"[P31_AUTO_PASS] tenant={tenant_id} op={operation_type} auto_enabled=True", flush=True)
    elif operation_type in AUTO_APPROVE_FORBIDDEN_OPERATIONS:
        print(f"[P31_FORBIDDEN_BLOCKED] tenant={tenant_id} op={operation_type} auto_enabled_ignored=True", flush=True)

    if not perm.get("admin_granted", False) and not auto_pass:
        raise HTTPException(status_code=403, detail="管理者によるエージェント利用許可がありません")

    allowed_agents = perm.get("allowed_agents") or []
    if allowed_agents and not auto_pass and not agent_type:
        raise HTTPException(status_code=403, detail="agent_typeが未確定のため権限検査できません")
    if allowed_agents and not auto_pass and agent_type and agent_type not in allowed_agents:
        raise HTTPException(status_code=403, detail=f"agent_type '{agent_type}' はこのテナントで許可されていません。許可済み: {allowed_agents}")
    allowed_ops = perm.get("allowed_operations") or []
    if allowed_ops and not auto_pass and not operation_type:
        raise HTTPException(status_code=403, detail="operation_typeが未確定のため権限検査できません")
    if allowed_ops and not auto_pass and operation_type and operation_type not in allowed_ops:
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

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
    "post_monitoring":  "low",
    "interview_assist": "low",
}

OPERATION_LABELS = {
    "entity_register": "情報登録",
    "entity_update": "情報更新",
    "media_replace": "画像・資料差し替え",
    "text_update": "テキスト更新",
    "schedule_update": "予定更新",
    "price_update": "料金更新",
    "news_post": "ニュース投稿",
    "status_update": "ステータス更新",
    "post_monitoring": "投稿数監視",
    "interview_assist": "面接メモ作成",
}

OPERATION_SUMMARIES = {
    "post_monitoring": "投稿数、未投稿、競合・市場キーワードを確認します。",
    "interview_assist": "面接で使う質問案・評価軸・判断メモを作成します。面接そのものを自動実施するものではありません。",
}

def _build_preview(agent_type: str, operation_type: str, industry: str, payload: dict, operation_steps: list = None, before_values: dict = None) -> dict:
    tmpl = INDUSTRY_TEMPLATES.get(industry, INDUSTRY_TEMPLATES["other"])
    affected = list(payload.keys())
    op_label = OPERATION_LABELS.get(operation_type, operation_type)
    summary = OPERATION_SUMMARIES.get(operation_type) or f"{op_label}の内容を確認し、承認後に実行します。"
    # P28: before_valuesが取得済みならdiffに反映、なければunknown
    _bv = before_values or {}
    diff = [
        {
            "field":  k,
            "before": _bv.get(k, {}).get("value", "unknown") if _bv.get(k) else "unknown",
            "after":  v,
            "changed": _bv.get(k, {}).get("value") != v if _bv.get(k) else True,
        }
        for k, v in payload.items()
    ]
    _before_display = {k: v.get("value") for k, v in _bv.items() if isinstance(v, dict)}
    preview = {
        "agent_type":      agent_type,
        "operation_type":  operation_type,
        "industry":        industry,
        "entity_label":    tmpl["entity_name"],
        "summary":         summary,
        "operation_label":  op_label,
        "payload_preview": payload,
        "before":          _before_display,
        "after":           payload,
        "diff":            diff,
        "affected_fields": affected,
        "risk_level":      _RISK_MAP.get(operation_type, "low"),
        "before_captured": bool(_bv),
    }
    # P14: operation_graphプレビュー
    if operation_steps:
        preview["operation_graph"] = True
        preview["step_count"]      = len(operation_steps)
        preview["steps_preview"]   = [{"step_id": s.get("step_id"), "step_type": s.get("step_type"), "order": s.get("order")} for s in operation_steps]
        # P13: 空selectorのstepを検出してmissing_step_selectorsをpreviewに出す
        _missing_step_sels = []
        for _s in operation_steps:
            _sel_key = _s.get("selector_key") or ""
            _sel_val = _s.get("selector") or ""
            if _sel_key and not _sel_val:
                _missing_step_sels.append({"step_id": _s.get("step_id", ""), "selector_key": _sel_key, "step_type": _s.get("step_type", "")})
        if _missing_step_sels:
            preview["missing_step_selectors"] = _missing_step_sels
            preview["missing_step_selector_count"] = len(_missing_step_sels)
    return preview


def _create_task_workflow_session(
    db,
    tenant_id: str,
    workflow_id: str,
    operation_type: str,
    operation_steps: list | None = None,
    media_mapping: dict | None = None,
    media_mapping_id: str = "",
    media_name: str = "",
    goal_context: str = "",
) -> tuple[str, dict]:
    """Create the P20 workflow safety session for every task creation path."""
    try:
        from api.core.browser_executor import (
            estimate_workflow_risk,
            create_workflow_session,
        )

        mm = dict(media_mapping or {})
        if media_mapping_id and not mm:
            try:
                snap = db.collection("media_mappings").document(media_mapping_id).get()
                if snap.exists:
                    cand = snap.to_dict() or {}
                    if cand.get("tenant_id") == tenant_id:
                        mm = cand
                        mm["mapping_id"] = media_mapping_id
                        mm["id"] = media_mapping_id
            except Exception as e:
                print(f"[P20_MEDIA_MAPPING_LOAD_ERROR] {type(e).__name__}", flush=True)

        display_media = media_name or mm.get("media_name", "")
        steps = operation_steps or []
        risk = estimate_workflow_risk(
            db=db,
            tenant_id=tenant_id,
            operation_type=operation_type,
            media_family=mm.get("media_family", ""),
            operation_steps=steps,
            media_mapping=mm,
        )
        policy = {
            "max_retry": 2,
            "allow_self_heal": True,
            "allow_replan": True,
            "require_human_on_high_risk": True,
            "interruptible": True,
        }
        session_id = create_workflow_session(
            db=db,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            goal=f"{goal_context or operation_type} for {display_media or media_mapping_id or 'mapping'}",
            operation_type=operation_type,
            operation_steps=steps,
            execution_policy=policy,
            risk_estimation=risk,
        )
        return session_id, risk
    except Exception as e:
        print(f"[P20 workflow_session] 作成エラー: {type(e).__name__}", flush=True)
        return "", {}


def _structure_page_doc_id(url: str) -> str:
    import hashlib as _hashlib_structure
    import re as _re_structure
    raw = str(url or "")
    slug = _re_structure.sub(r"[^0-9a-zA-Z_-]+", "_", raw)[:70].strip("_") or "page"
    digest = _hashlib_structure.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:14]
    return f"page_{slug}_{digest}"[:140]


def _role_aliases(role: str) -> set[str]:
    aliases = {
        "save": {"save", "submit"},
        "submit": {"save", "submit"},
        "date_input": {"date_input", "date", "schedule"},
        "required_inputs": {"required_inputs", "name", "entity_name"},
        "editable_inputs": {"editable_inputs", "body", "name", "text", "content"},
        "body": {"body", "text", "content", "description", "comment"},
        "title": {"title"},
        "file": {"file", "image", "media"},
        "price": {"price"},
    }
    return aliases.get(role, {role})


def _infer_structure_role(el: dict) -> tuple[str, float, str]:
    tag = str(el.get("tag") or "").lower()
    typ = str(el.get("type") or "").lower()
    blob = " ".join(
        str(el.get(k) or "")
        for k in ("canonical", "name", "id", "label", "placeholder", "text", "value", "section", "aria_label")
    ).lower()

    if typ == "file":
        return "file", 1.0, "type=file"
    if tag == "textarea":
        return "body", 0.95, "textarea"
    if typ in ("submit", "image"):
        return "save", 0.9, "submit input"
    if typ == "button" or tag == "button":
        if any(k in blob for k in ("保存", "登録", "更新", "投稿", "送信", "確定", "反映", "save", "submit", "register", "regist", "update", "post", "send", "publish", "apply")):
            return "save", 0.95, "submit/save text"
    if typ in ("date", "time", "datetime-local", "datetime"):
        return "date_input", 0.9, "date/time input"
    if any(k in blob for k in ("price", "料金", "価格", "金額", "コース", "fee")):
        return "price", 0.85, "price label"
    if any(k in blob for k in ("title", "タイトル", "件名", "見出し")):
        return "title", 0.85, "title label"
    if any(k in blob for k in ("schedule", "shift", "出勤", "予定", "日付", "シフト", "カレンダー")):
        return "date_input", 0.75, "schedule label"
    if any(k in blob for k in ("name", "名前", "氏名", "源氏名", "店舗名", "物件名", "商品名")):
        return "required_inputs", 0.72, "name label"
    if any(k in blob for k in ("body", "content", "本文", "内容", "コメント", "紹介", "説明", "プロフィール", "comment", "description", "text")):
        return "body", 0.72, "body label"
    if tag == "select":
        return "editable_inputs", 0.65, "select input"
    if tag == "input" and typ not in ("hidden", "password", "submit", "button", "reset", "image"):
        return "editable_inputs", 0.55, "editable input"
    return "", 0.0, ""


def _selector_from_structure_field(field: dict) -> str:
    return str(field.get("selector") or field.get("suggested_selector") or "")


def _normalize_structure_page(page: dict, mapping_id: str, source: str = "deep_scan") -> dict:
    if not isinstance(page, dict):
        return {}
    url = str(page.get("url") or (page.get("form_schema") or {}).get("url") or "")
    if not url:
        return {}

    raw_fields: list[dict] = []
    form_schema = page.get("form_schema") or {}
    if isinstance(form_schema, dict):
        for f in form_schema.get("fields") or []:
            if isinstance(f, dict):
                raw_fields.append({**f, "source_group": "form_schema"})
    for group, tag in (
        ("inputs", "input"),
        ("textareas", "textarea"),
        ("selects", "select"),
        ("file_inputs", "input"),
        ("buttons", "button"),
    ):
        for f in page.get(group) or []:
            if not isinstance(f, dict):
                continue
            row = dict(f)
            row.setdefault("tag", tag)
            if group == "file_inputs":
                row.setdefault("type", "file")
            row["source_group"] = group
            raw_fields.append(row)

    seen = set()
    fields: list[dict] = []
    for idx, f in enumerate(raw_fields[:500]):
        selector = _selector_from_structure_field(f)
        key = selector or f"{f.get('source_group','field')}:{f.get('name','')}:{f.get('id','')}:{idx}"
        if key in seen:
            continue
        seen.add(key)
        role, conf, reason = _infer_structure_role(f)
        fields.append({
            "order": int(f.get("order") or idx),
            "selector": selector,
            "tag": str(f.get("tag") or "").lower(),
            "type": str(f.get("type") or "").lower(),
            "name": str(f.get("name") or ""),
            "id": str(f.get("id") or ""),
            "label": str(f.get("label") or f.get("text") or f.get("value") or f.get("placeholder") or "")[:180],
            "role": role,
            "role_confidence": conf,
            "role_source": reason,
            "source": f.get("source_group") or source,
        })

    role_counts: dict[str, int] = {}
    for f in fields:
        if f.get("role"):
            role_counts[f["role"]] = role_counts.get(f["role"], 0) + 1

    return {
        "page_id": _structure_page_doc_id(url),
        "mapping_id": mapping_id,
        "url": url,
        "title": str(page.get("title") or page.get("html_title") or form_schema.get("title") or "")[:180],
        "category": str(page.get("category") or ""),
        "page_purpose": str(page.get("page_purpose") or ""),
        "page_purpose_source": str(page.get("page_purpose_source") or ""),
        "forms": [{
            "form_index": 0,
            "fields": fields,
            "fields_count": len(fields),
            "role_counts": role_counts,
            "source": source,
        }],
        "fields_count": len(fields),
        "role_counts": role_counts,
        "source": source,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }


def _write_structure_pages(db, mapping_id: str, pages: list[dict], source: str = "deep_scan") -> int:
    ref = db.collection("media_mappings").document(mapping_id).collection("pages")
    written = 0
    for pg in pages or []:
        normalized = _normalize_structure_page(pg, mapping_id, source=source)
        if not normalized or normalized.get("fields_count", 0) <= 0:
            continue
        try:
            existing = ref.document(normalized["page_id"]).get().to_dict() or {}
            existing_fields = []
            for form in existing.get("forms") or []:
                existing_fields.extend([f for f in (form.get("fields") or []) if isinstance(f, dict)])
            next_fields = []
            for form in normalized.get("forms") or []:
                next_fields.extend([f for f in (form.get("fields") or []) if isinstance(f, dict)])
            if existing_fields:
                by_key = {}
                for f in existing_fields + next_fields:
                    key = f.get("selector") or f"{f.get('source')}:{f.get('name')}:{f.get('id')}:{f.get('label')}"
                    if not key:
                        continue
                    cur = by_key.get(key)
                    if not cur or float(f.get("role_confidence") or 0) >= float(cur.get("role_confidence") or 0):
                        by_key[key] = f
                merged_fields = list(by_key.values())[:500]
                role_counts: dict[str, int] = {}
                for f in merged_fields:
                    if f.get("role"):
                        role_counts[f["role"]] = role_counts.get(f["role"], 0) + 1
                normalized["forms"] = [{
                    "form_index": 0,
                    "fields": merged_fields,
                    "fields_count": len(merged_fields),
                    "role_counts": role_counts,
                    "source": source,
                }]
                normalized["fields_count"] = len(merged_fields)
                normalized["role_counts"] = role_counts
        except Exception as e:
            print(f"[STRUCTURE_PAGE_MERGE_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
        # H-3: Firestore write失敗がサイレントだとデータ欠損。ログ出力して残ページは継続
        try:
            ref.document(normalized["page_id"]).set(normalized, merge=True)
            written += 1
        except Exception as _ws_e:
            print(f"[STRUCTURE_PAGE_WRITE_ERROR] mapping_id={mapping_id} page_id={normalized.get('page_id')} err={type(_ws_e).__name__}", flush=True)
    return written


def _read_structure_pages(db, mapping_id: str, limit: int = 500) -> list[dict]:
    pages = []
    try:
        for snap in (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("pages")
            .limit(max(1, min(int(limit or 500), 500)))
            .stream()
        ):
            row = snap.to_dict() or {}
            if row:
                row["page_id"] = row.get("page_id") or snap.id
                pages.append(row)
    except Exception as e:
        print(f"[STRUCTURE_PAGES_READ_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
    return pages


def _role_candidates_for_page(page: dict, selector_key: str) -> list[dict]:
    aliases = _role_aliases(selector_key)
    rows = []
    for form in page.get("forms") or []:
        for f in form.get("fields") or []:
            if not isinstance(f, dict) or not f.get("selector"):
                continue
            if str(f.get("role") or "") in aliases:
                rows.append(f)
    rows.sort(key=lambda r: float(r.get("role_confidence") or 0), reverse=True)
    return rows


def _selector_record_from_field(field: dict) -> dict:
    return {
        "selector": field.get("selector") or "",
        "role": field.get("role") or "",
        "tag": field.get("tag") or "",
        "type": field.get("type") or "",
        "label": field.get("label") or "",
        "source": "structure_pages",
        "confidence": field.get("role_confidence") or 0,
    }


def _structural_status_for_page(op: str, page: dict, cfg: dict) -> dict:
    required = list(cfg.get("required_selector_keys") or [])
    selectors = {}
    missing = []
    review_roles = []
    field_schema = []
    for key in required:
        candidates = _role_candidates_for_page(page, key)
        if not candidates:
            missing.append(key)
            continue
        best = candidates[0]
        selectors[key] = _selector_record_from_field(best)
        if len(candidates) > 1 or float(best.get("role_confidence") or 0) < 0.6:
            review_roles.append(key)
        field_schema.append({
            "selector_key": key,
            "selector": best.get("selector") or "",
            "label": best.get("label") or "",
            "role": best.get("role") or "",
            "role_confidence": best.get("role_confidence") or 0,
            "candidate_count": len(candidates),
        })

    if missing:
        status = "UNDISCOVERED"
    elif review_roles:
        status = "NEEDS_REVIEW"
    else:
        status = "READY"
    return {
        "status": status,
        "selectors": selectors,
        "missing": missing,
        "review_roles": review_roles,
        "field_schema": field_schema,
        "target_url": page.get("url") or "",
        "page_id": page.get("page_id") or _structure_page_doc_id(page.get("url") or ""),
        "title": page.get("title") or "",
    }


def _selector_text_from_record(sel) -> str:
    if isinstance(sel, dict):
        return str(sel.get("selector") or "")
    return str(sel or "")


def _build_steps_from_structural_op(op_type: str, op_map: dict, cfg: dict) -> list[dict]:
    selectors = op_map.get("selectors") or {}
    target_url = str(op_map.get("target_url") or "")
    computed_status = op_map.get("status") or "UNDISCOVERED"
    if computed_status not in ("READY", "NEEDS_REVIEW") or not target_url.startswith(("http://", "https://")):
        return []

    def _has_selector(key: str) -> bool:
        return bool(_selector_text_from_record(selectors.get(key)))

    required_keys = list(cfg.get("required_selector_keys") or [])
    missing = [key for key in required_keys if not _has_selector(key) and not (key == "save" and _has_selector("submit"))]
    if missing and computed_status == "READY":
        computed_status = "NEEDS_REVIEW"

    steps: list[dict] = []
    order = 0
    steps.append({
        "order": order,
        "step_id": f"{op_type}_login",
        "step_type": "login",
        "display_name": "ログイン",
        "status": "READY",
        "required": True,
        "source_url": None,
        "target_url": None,
        "selector_key": None,
        "selector": None,
        "source": "structure_pages",
    })
    order += 1
    steps.append({
        "order": order,
        "step_id": f"{op_type}_navigate",
        "step_type": "navigate",
        "display_name": "画面へ移動",
        "status": computed_status,
        "required": True,
        "source_url": target_url,
        "target_url": target_url,
        "url": target_url,
        "selector_key": None,
        "selector": None,
        "source": "structure_pages",
    })
    order += 1

    for field in cfg.get("fields") or []:
        if not isinstance(field, dict):
            continue
        skey = str(field.get("selector_key") or "")
        if not skey:
            continue
        required = skey in required_keys
        selector = _selector_text_from_record(selectors.get(skey))
        if not selector and not required:
            continue
        input_type = str(field.get("input_type") or "text")
        step_type = "upload_file" if input_type == "file" else ("select" if input_type == "select" else "fill")
        steps.append({
            "order": order,
            "step_id": f"{op_type}_{step_type}_{skey}",
            "step_type": step_type,
            "display_name": "ファイルアップロード" if step_type == "upload_file" else "入力",
            "status": computed_status if selector else "FAILED",
            "required": required,
            "source_url": target_url,
            "target_url": target_url,
            "selector_key": skey,
            "payload_key": field.get("payload_key"),
            "selector": selector or None,
            "missing_required_fields": missing if missing else [],
            "source": "structure_pages",
        })
        order += 1

    submit_key = str(cfg.get("submit_selector_key") or "save")
    if not _has_selector(submit_key) and _has_selector("submit"):
        submit_key = "submit"
    submit_selector = _selector_text_from_record(selectors.get(submit_key))
    steps.append({
        "order": order,
        "step_id": f"{op_type}_click_save",
        "step_type": "click",
        "display_name": "保存",
        "required": True,
        "terminal": True,
        "once": True,
        "status": computed_status if submit_selector else "FAILED",
        "source_url": target_url,
        "target_url": target_url,
        "selector_key": submit_key,
        "selector": submit_selector or None,
        "missing_required_fields": missing if missing else [],
        "source": "structure_pages",
    })
    order += 1
    steps.append({
        "order": order,
        "step_id": f"{op_type}_verify",
        "step_type": "verify",
        "display_name": "反映確認",
        "status": computed_status,
        "required": True,
        "source_url": target_url,
        "target_url": target_url,
        "selector_key": None,
        "selector": None,
        "source": "structure_pages",
    })
    return steps


def _build_structural_capability_view(mapping_id: str, pages: list[dict], admin_host: str = "") -> tuple[dict, dict, dict]:
    from api.core.browser_executor import GENERIC_OPERATION_CONFIG

    # 操作対象になり得ないページを除外（FAQヘルプ・外部フォーム・PDF・ガイドライン等）。
    # deep scanがメニュー内の「〜について」ヘルプリンクを操作対象に誤登録する根本原因への対策。
    _all_pages = pages or []
    _candidate_pages = [p for p in _all_pages if not _is_blocked_operation_url(p.get("url") or "")]
    # admin_hostが分かる場合は同一サイト外のページも除外（FAQ別ドメイン等を確実に弾く）
    if admin_host:
        _same_site = [p for p in _candidate_pages
                      if _registrable_domain(p.get("url") or "") == admin_host]
        # 全滅回避: 同一サイトページが1つも無ければドメイン条件は使わない
        if _same_site:
            _candidate_pages = _same_site
    # 安全策: ブロックリストで全滅した場合は元のページに戻す（機能停止を防ぐ）
    if not _candidate_pages and _all_pages:
        _candidate_pages = _all_pages

    def _same_domain_rank(url: str) -> int:
        if admin_host and _registrable_domain(url or "") == admin_host:
            return 0
        return 1

    operations = {}
    op_maps = {}
    for op, cfg in GENERIC_OPERATION_CONFIG.items():
        page_rows = []
        for page in _candidate_pages:
            row = _structural_status_for_page(op, page, cfg)
            if row["status"] == "UNDISCOVERED":
                continue
            page_rows.append(row)

        # ステータス優先（READY>NEEDS_REVIEW）、同点なら同一adminドメインを優先
        page_rows.sort(key=lambda r: (
            {"READY": 0, "NEEDS_REVIEW": 1}.get(r["status"], 9),
            _same_domain_rank(r.get("target_url") or ""),
        ))
        best = page_rows[0] if page_rows else {}
        status = best.get("status") or "UNDISCOVERED"
        operations[op] = {
            "status": status,
            "ready_count": len([p for p in page_rows if p.get("status") == "READY"]),
            "review_count": len([p for p in page_rows if p.get("status") == "NEEDS_REVIEW"]),
            "target_url": best.get("target_url", ""),
            "page_id": best.get("page_id", ""),
            "missing": best.get("missing", list(cfg.get("required_selector_keys") or [])),
            "field_schema": best.get("field_schema", []),
            "pages": page_rows[:60],
            "source": "structure_pages",
        }
        if best:
            op_maps[op] = {
                "status": status,
                "selectors": best.get("selectors", {}),
                "missing": best.get("missing", []),
                "target_url": best.get("target_url", ""),
                "validation_score": 100 if status == "READY" else 60,
                "executable": status == "READY",
                "structural_ready": status == "READY",
                "source": "structure_pages",
                "page_id": best.get("page_id", ""),
                "form_schema": {"fields": best.get("field_schema", []), "source": "structure_pages"},
            }

    steps = {
        op: built
        for op, built in (
            (op, _build_steps_from_structural_op(op, op_map, GENERIC_OPERATION_CONFIG.get(op, {}) or {}))
            for op, op_map in op_maps.items()
        )
        if built
    } if op_maps else {}
    for op, op_steps in (steps or {}).items():
        if op in operations:
            operations[op]["step_count"] = len(op_steps or [])
            operations[op]["taskable"] = operations[op].get("status") == "READY" and bool(op_steps)
            if operations[op].get("status") == "READY" and not op_steps:
                operations[op]["status"] = "NEEDS_REVIEW"
                operations[op]["taskable"] = False
                missing = list(operations[op].get("missing") or [])
                if "operation_steps" not in missing:
                    missing.append("operation_steps")
                operations[op]["missing"] = missing
        if op in op_maps:
            op_maps[op]["step_count"] = len(op_steps or [])
            if not op_steps and op_maps[op].get("status") == "READY":
                op_maps[op]["status"] = "NEEDS_REVIEW"
                op_maps[op]["executable"] = False
                op_maps[op]["missing"] = list(set((op_maps[op].get("missing") or []) + ["operation_steps"]))
    for op, row in operations.items():
        if row.get("status") == "READY" and not row.get("step_count"):
            row["status"] = "NEEDS_REVIEW"
            row["taskable"] = False
            missing = list(row.get("missing") or [])
            if "operation_steps" not in missing:
                missing.append("operation_steps")
            row["missing"] = missing
            if op in op_maps:
                op_maps[op]["status"] = "NEEDS_REVIEW"
                op_maps[op]["executable"] = False
                op_maps[op]["missing"] = list(set((op_maps[op].get("missing") or []) + ["operation_steps"]))

    view = {
        "version": 1,
        "source": "structure_pages",
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "pages_count": len(pages or []),
        "operations": operations,
        "ready_count": len([op for op, row in operations.items() if row.get("taskable")]),
        "review_count": len([op for op, row in operations.items() if row.get("status") == "NEEDS_REVIEW"]),
    }
    return view, op_maps, steps or {}


def _refresh_capability_view_for_mapping(db, mapping_id: str, mapping: dict | None = None, merge_operation_cache: bool = True) -> dict:
    pages = _read_structure_pages(db, mapping_id)
    if not pages:
        return {}
    _login_for_host = ((mapping or {}).get("login_url")
                       or (mapping or {}).get("media_url")
                       or (mapping or {}).get("detected_login_url") or "")
    _admin_host = _registrable_domain(_login_for_host)
    view, op_maps, steps = _build_structural_capability_view(mapping_id, pages, admin_host=_admin_host)
    update = {
        "capability_view": view,
        "structure_model": {
            "source": "pages",
            "pages_count": len(pages),
            "updated_at": view["generated_at"],
        },
        "updated_at": datetime.datetime.utcnow(),
    }
    if merge_operation_cache:
        # AI整備済みoperationだけを保護する。旧MANUAL/旧READYは再整備で上書き対象。
        try:
            _cur_snap = db.collection("media_mappings").document(mapping_id).get()
            _cur_op_maps = (_cur_snap.to_dict() or {}).get("operation_mappings") or {}
        except Exception:
            _cur_op_maps = {}
        _protected_ops = {op for op, v in _cur_op_maps.items() if _operation_mapping_is_production_ready(v)}
        if _protected_ops:
            for _pop in _protected_ops:
                if _pop in op_maps:
                    op_maps[_pop] = _cur_op_maps[_pop]
                    print(f"[AI_READY_PROTECT] op={_pop} mapping_id={mapping_id} AI整備済みのため上書き防止", flush=True)
                else:
                    op_maps[_pop] = _cur_op_maps[_pop]
        update["operation_mappings"] = op_maps
        update["operation_steps_by_type"] = steps
        update["operation_cache_source"] = "structure_pages"
    try:
        db.collection("media_mappings").document(mapping_id).set(update, merge=True)
    except Exception as e:
        print(f"[CAPABILITY_VIEW_SAVE_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    return view


def _operation_from_capability_view(mapping: dict, operation_type: str) -> dict:
    ops = ((mapping.get("capability_view") or {}).get("operations") or {})
    if isinstance(ops, dict) and ops:
        return ops.get(operation_type) or {
            "status": "UNDISCOVERED",
            "missing": ["structure_role"],
            "taskable": False,
            "source": "structure_pages",
        }
    return {}


def _seed_structure_pages_from_menu_items(db, mapping_id: str, items: list[dict], source: str = "menu_scan_backfill") -> int:
    pages: list[dict] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_url = item.get("url") or item.get("absolute_url") or item.get("href") or ""
        title = item.get("title") or item_url
        category = item.get("category") or ""
        for rec in (item.get("operations") or {}).values():
            if not isinstance(rec, dict):
                continue
            form_schema = rec.get("form_schema") or {}
            if not isinstance(form_schema, dict) or not (form_schema.get("fields") or []):
                continue
            url = rec.get("target_url") or item_url
            if not url:
                continue
            key = f"{url}:{rec.get('op') or ''}"
            if key in seen:
                continue
            seen.add(key)
            pages.append({
                "url": url,
                "title": title,
                "category": category,
                "form_schema": form_schema,
                "page_purpose": rec.get("op") or "",
                "page_purpose_source": source,
            })
    if not pages:
        return 0
    return _write_structure_pages(db, mapping_id, pages, source=source)


def _seed_structure_pages_from_schema_forms(db, mapping_id: str, source: str = "schema_forms_backfill") -> int:
    pages: list[dict] = []
    try:
        for snap in (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("schema_forms")
            .limit(500)
            .stream()
        ):
            form = snap.to_dict() or {}
            fields = form.get("fields") or []
            if not isinstance(fields, list) or not fields:
                continue
            pages.append({
                "url": form.get("url") or form.get("page_url") or form.get("target_url") or f"schema_form://{snap.id}",
                "title": form.get("title") or form.get("page_title") or form.get("entity_type") or snap.id,
                "category": form.get("category") or form.get("entity_type") or "",
                "page_purpose": form.get("page_purpose") or form.get("entity_type") or "",
                "page_purpose_source": source,
                "form_schema": {
                    "fields": fields,
                    "fields_count": form.get("fields_count") or len(fields),
                    "title": form.get("title") or form.get("page_title") or snap.id,
                    "source": source,
                },
            })
    except Exception as e:
        print(f"[STRUCTURE_BACKFILL_SCHEMA_FORMS_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
        return 0
    if not pages:
        return 0
    return _write_structure_pages(db, mapping_id, pages, source=source)


def _role_from_manual_selector_key(key: str) -> str:
    k = str(key or "")
    if k in {"body", "title", "file", "save", "submit", "date_input", "price", "required_inputs", "editable_inputs"}:
        return "save" if k == "submit" else k
    if k in {"username", "password", "login_submit", "login_id", "login_password"}:
        return ""
    return ""


def _sync_manual_selectors_to_structure_pages(
    db,
    mapping_id: str,
    mapping: dict,
    selectors: dict,
    source: str = "manual_selector",
) -> int:
    # 旧DOMセレクターを操作構造へ昇格させる経路はAI整備契約と混線するため停止。
    # ログイン用dom_selectorsの保存自体は残すが、実行可能operationはAI整備のみで作る。
    return 0
    if not isinstance(selectors, dict) or not selectors:
        return 0
    pages = _read_structure_pages(db, mapping_id)
    if not pages:
        return 0
    try:
        from api.core.browser_executor import GENERIC_OPERATION_CONFIG
    except Exception:
        GENERIC_OPERATION_CONFIG = {}

    role_target_urls: dict[str, set[str]] = {}
    op_maps = mapping.get("operation_mappings") or {}
    for op, op_map in op_maps.items():
        if not isinstance(op_map, dict) or not op_map.get("target_url"):
            continue
        cfg = GENERIC_OPERATION_CONFIG.get(op, {}) or {}
        roles = set(cfg.get("required_selector_keys") or [])
        roles.update(str(f.get("selector_key") or "") for f in (cfg.get("fields") or []) if isinstance(f, dict))
        if cfg.get("submit_selector_key"):
            roles.add(str(cfg.get("submit_selector_key")))
        for role in roles:
            if role:
                role_target_urls.setdefault("save" if role == "submit" else role, set()).add(str(op_map.get("target_url")))

    pages_ref = db.collection("media_mappings").document(mapping_id).collection("pages")
    updated = 0
    now_iso = datetime.datetime.utcnow().isoformat()
    for page in pages:
        page_url = str(page.get("url") or "")
        page_changed = False
        forms = page.get("forms") or []
        if not forms:
            forms = [{"form_index": 0, "fields": []}]
        fields = []
        for form in forms:
            if isinstance(form, dict):
                fields.extend([f for f in (form.get("fields") or []) if isinstance(f, dict)])

        for key, raw_selector in selectors.items():
            role = _role_from_manual_selector_key(key)
            selector = _selector_text_from_record(raw_selector)
            if not role or not selector:
                continue
            matched = False
            for field in fields:
                if field.get("selector") == selector or field.get("role") == role:
                    field["selector"] = selector
                    field["role"] = role
                    field["role_confidence"] = 1.0
                    field["role_source"] = source
                    field["source"] = source
                    matched = True
                    page_changed = True
            if not matched and page_url in role_target_urls.get(role, set()):
                fields.append({
                    "order": len(fields),
                    "selector": selector,
                    "tag": "",
                    "type": "",
                    "name": str(key),
                    "id": "",
                    "label": str(key),
                    "role": role,
                    "role_confidence": 1.0,
                    "role_source": source,
                    "source": source,
                })
                page_changed = True

        if not page_changed:
            continue
        role_counts: dict[str, int] = {}
        for f in fields:
            if f.get("role"):
                role_counts[f["role"]] = role_counts.get(f["role"], 0) + 1
        page["forms"] = [{
            "form_index": 0,
            "fields": fields[:500],
            "fields_count": min(len(fields), 500),
            "role_counts": role_counts,
            "source": source,
        }]
        page["fields_count"] = min(len(fields), 500)
        page["role_counts"] = role_counts
        page["updated_at"] = now_iso
        pages_ref.document(page.get("page_id") or _structure_page_doc_id(page_url)).set(page, merge=True)
        updated += 1
    if updated:
        _refresh_capability_view_for_mapping(db, mapping_id, mapping)
    return updated


def _ensure_capability_view_for_mapping(db, mapping_id: str, mapping: dict | None = None) -> tuple[dict, dict]:
    current = dict(mapping or {})
    view = current.get("capability_view") or {}
    if view.get("source") == "structure_pages" and view.get("operations"):
        return current, view

    try:
        written = 0
        scan_items = ((current.get("manual_menu_scan_results") or {}).get("items") or [])
        if scan_items:
            written += _seed_structure_pages_from_menu_items(db, mapping_id, scan_items, source="menu_scan_parent_backfill")
        if not written:
            try:
                detail_items = [
                    snap.to_dict() or {}
                    for snap in (
                        db.collection("media_mappings")
                        .document(mapping_id)
                        .collection("menu_scan_items")
                        .limit(500)
                        .stream()
                    )
                ]
                written += _seed_structure_pages_from_menu_items(db, mapping_id, detail_items, source="menu_scan_detail_backfill")
            except Exception as e:
                print(f"[STRUCTURE_BACKFILL_DETAIL_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
        if not written:
            written += _seed_structure_pages_from_schema_forms(db, mapping_id, source="schema_forms_backfill")
        refreshed = _refresh_capability_view_for_mapping(db, mapping_id, current)
        if refreshed:
            latest_snap = db.collection("media_mappings").document(mapping_id).get()
            if latest_snap.exists:
                current = latest_snap.to_dict() or current
                current["mapping_id"] = mapping_id
                current["id"] = mapping_id
            return current, refreshed
    except Exception as e:
        print(f"[CAPABILITY_VIEW_ENSURE_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    return current, current.get("capability_view") or {}


def _capability_op_is_taskable(mapping: dict, operation_type: str) -> bool:
    op_map = ((mapping.get("operation_mappings") or {}).get(operation_type) or {})
    if not _operation_mapping_is_production_ready(op_map):
        return False
    row = _operation_from_capability_view(mapping, operation_type)
    return bool(row.get("status") == "READY" and row.get("taskable"))


def _capability_steps_for_mapping(db, mapping_id: str, mapping: dict, operation_type: str) -> tuple[list, dict, dict]:
    mapping, view = _ensure_capability_view_for_mapping(db, mapping_id, mapping)
    cap_op = _operation_from_capability_view(mapping, operation_type)
    op_map = ((mapping.get("operation_mappings") or {}).get(operation_type) or {})
    if not _operation_mapping_is_production_ready(op_map):
        return [], cap_op, mapping
    steps = ((mapping.get("operation_steps_by_type") or {}).get(operation_type)) or []
    if cap_op and (cap_op.get("status") != "READY" or not cap_op.get("taskable")):
        return [], cap_op, mapping
    if steps:
        return steps, cap_op, mapping
    return [], cap_op, mapping

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
    media_mapping_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    operation_mapping_override: dict = Field(default_factory=dict)
    scheduled_at: Optional[str] = None

class TaskApproveRequest(BaseModel):
    task_id: str

class TaskRejectRequest(BaseModel):
    task_id: str
    reason: Optional[str] = None

class AgentPermissionsUpdateRequest(BaseModel):
    admin_granted: Optional[bool] = None
    allowed_agents: Optional[list[str]] = None
    allowed_operations: Optional[list[str]] = None
    max_tasks_per_day: Optional[int] = None

class BatchTaskCreateRequest(BaseModel):
    agent_type: str = "hp_update"
    operation_type: str
    industry: str = "generic"
    entity_type: Optional[str] = None
    media_mapping_ids: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    scheduled_at: Optional[str] = None
    include_needs_review: bool = False

class BatchTaskActionRequest(BaseModel):
    batch_id: str

class CrossMediaTaskCreateRequest(BaseModel):
    instruction: str = ""
    industry: str = "generic"
    source_mode: str = "manual_payload"  # manual_payload | public_url | source_mapping
    source_url: Optional[str] = None
    source_mapping_id: Optional[str] = None
    target_mapping_ids: list[str] = Field(default_factory=list)
    target_operation_type: str = "entity_update"
    source_payload: dict = Field(default_factory=dict)
    query: str = ""
    max_items: int = 1
    source_access_confirmed: bool = False
    scheduled_at: Optional[str] = None
    # 対象指定（誰を）: 取得元から選んだエンティティの詳細URL/ラベル
    source_entity_url: Optional[str] = None
    source_entity_label: str = ""
    # 更新範囲（何を）: 反映するフィールドのラベル/index（空=全項目）
    selected_field_keys: list[str] = Field(default_factory=list)

class CrossMediaTaskActionRequest(BaseModel):
    cross_task_id: str

@router.post("/task/create")
def create_task(req: TaskCreateRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="エージェントモードの利用権限がありません")

    db = get_db()

    # op_id指定時はFirestore op_dataを正として厳格検証
    if req.op_id:
        op_doc = db.collection("agent_ops").document(req.op_id).get()
        default_op = next((dict(o) for o in DEFAULT_AGENT_OPS if o.get("op_id") == req.op_id), None)
        if not op_doc.exists and not default_op:
            raise HTTPException(status_code=400, detail="存在しないop_idです")
        firestore_op = op_doc.to_dict() if op_doc.exists else {}
        op_data = {**(default_op or {}), **(firestore_op or {}), "op_id": req.op_id}

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
        if not ctx["is_admin"] and allowed and not user_plans.intersection(allowed):
            raise HTTPException(status_code=403, detail="このOperationを利用する権限がありません")

        # op_dataからフィールドを正として補完（req側は信用しない）
        agent_type     = op_data.get("category", "hp_update")
        operation_type = op_data["operation_type"]
        entity_type    = op_data["entity_type"]
        industry       = _normalize_industry(op_data.get("industry") or "generic")
        _validate_agent_operation_pair(agent_type, operation_type)

        # payload_schema.fields の required検証 + 余計なキー除外
        schema_fields = op_data.get("payload_schema", {}).get("fields", [])
        allowed_keys = {f["key"] for f in schema_fields}
        required_keys = {f["key"] for f in schema_fields if f.get("required")}

        # required_fields フォールバック
        for rk in op_data.get("required_fields", []):
            required_keys.add(rk)

        # Mapping-derived fields are first-class payload keys. Without this,
        # a custom/Firestore op drops fields that were correctly mapped from the site.
        _schema_mapping_id = req.media_mapping_id or (req.payload or {}).get("media_mapping_id", "")
        if _schema_mapping_id:
            try:
                _schema_mm_doc = db.collection("media_mappings").document(str(_schema_mapping_id)).get()
                if _schema_mm_doc.exists:
                    _schema_mm = _schema_mm_doc.to_dict() or {}
                    if _schema_mm.get("tenant_id") == ctx["tenant_id"] or ctx.get("is_admin"):
                        _mapped_keys = _payload_keys_for_mapping_fields(
                            _mapping_fields_for_operation(_schema_mm, op_data.get("operation_type") or "")
                        )
                        allowed_keys.update(_mapped_keys)
                        if _mapped_keys:
                            allowed_keys.add("structured_fields")
            except Exception as _mapped_key_e:
                print(f"[CREATE_TASK_MAPPED_KEYS_ERROR] op={op_data.get('operation_type')} {type(_mapped_key_e).__name__}", flush=True)

        # offer_send: filter_fields（mapping取得フィールドの絞り込み条件dict）を追加許可
        if op_data.get("operation_type") == "offer_send":
            allowed_keys.add("filter_fields")
            allowed_keys.add("filter_intent")

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
        if req.operation_type == "admin_crawl":
            raise HTTPException(status_code=400, detail="admin_crawl は管理画面解析専用Operationのため、タスク作成できません。媒体マッピングの自動解析から実行してください。")
        if req.operation_type not in OPERATION_TYPES:
            raise HTTPException(status_code=400, detail="無効なoperation_typeです")
        agent_type     = req.agent_type
        operation_type = req.operation_type
        entity_type    = req.entity_type or ""
        industry       = _normalize_industry(req.industry or "generic")
        clean_payload  = dict(req.payload)
        op_snapshot    = {}
        _validate_agent_operation_pair(agent_type, operation_type)
        # P12: hp_update系はmedia_mapping_id必須（audit/interview/post_monitoringは対象外）
        if agent_type == "hp_update" and operation_type in OPERATION_TYPES:
            _req_mm_id = req.media_mapping_id or (req.payload or {}).get("media_mapping_id", "")
            if not _req_mm_id:
                raise HTTPException(status_code=400, detail="media_mapping_idが必要です。媒体を選択してください。")

    # P0-1: operation_type / agent_type 確定後に enforcement（op_id解決後）
    _ctx_create = _resolve_agent_user_context(user)
    _enforce_agent_permissions(_ctx_create, agent_type, operation_type)

    # P14: operation_steps取得（operation_steps_by_type優先 → operation_steps_template fallback）
    _op_steps_template = op_data.get("operation_steps_template") if 'op_data' in locals() else None
    _operation_steps = None
    _media_mapping_id = req.media_mapping_id or clean_payload.get("media_mapping_id", "")
    if _media_mapping_id and not clean_payload.get("media_mapping_id"):
        clean_payload["media_mapping_id"] = _media_mapping_id
    _mm_id_create = _media_mapping_id
    _task_mapping_for_steps = {}
    _task_capability_op = {}
    _structural_gate_checked = False
    # スカウト系ops: capability_view/steps不要。credential+site_purposeで代替チェック。
    _SCOUT_OPS_CREATE = {"offer_send", "recruit_inbox_scan", "recruit_reply"}
    _SCOUT_SP_REQ_CREATE = {
        "offer_send":         {"scout"},
        "recruit_inbox_scan": {"scout", "reply", "monitor"},
        "recruit_reply":      {"scout", "reply"},
    }
    _is_scout_op_create = operation_type in _SCOUT_OPS_CREATE
    if _mm_id_create:
        try:
            _mm_doc_create = db.collection("media_mappings").document(_mm_id_create).get()
            if _mm_doc_create.exists:
                _task_mapping_for_steps = _mm_doc_create.to_dict() or {}
                _task_mapping_for_steps["mapping_id"] = _mm_id_create
                _task_mapping_for_steps["id"] = _mm_id_create
                if _task_mapping_for_steps.get("tenant_id") != _ctx_create["tenant_id"] and not _ctx_create["is_admin"]:
                    raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")
                _create_op_override = req.operation_mapping_override or {}
                if _create_op_override:
                    _op_maps_create = dict(_task_mapping_for_steps.get("operation_mappings") or {})
                    _prev_op_create = dict(_op_maps_create.get(operation_type) or {})
                    _merged_override = {**_prev_op_create, **_create_op_override}
                    if _merged_override.get("selectors") or _merged_override.get("target_url"):
                        _merged_override.setdefault("status", "READY")
                        _merged_override.setdefault("executable", True)
                    if not _operation_mapping_is_production_ready(_merged_override):
                        raise HTTPException(status_code=400, detail={
                            "message": "operation_mapping_override はAI整備済みではありません。任意指定ではなく媒体基盤のAI整備結果を使用してください。",
                            "operation_type": operation_type,
                            "status": "MAPPING_OVERRIDE_NOT_PRODUCTION_READY",
                        })
                    _op_maps_create[operation_type] = _merged_override
                    _task_mapping_for_steps["operation_mappings"] = _op_maps_create
                    print(f"[CREATE_TASK_OP_OVERRIDE] op={operation_type} selectors={len((_merged_override.get('selectors') or {}))}", flush=True)
                if _is_scout_op_create:
                    # スカウト系: site_purpose + credential のみ確認
                    _bc_sp_c = (_task_mapping_for_steps.get("business_conditions") or {}).get("site_purpose", "")
                    _req_sp_c = _SCOUT_SP_REQ_CREATE.get(operation_type, set())
                    if _bc_sp_c not in _req_sp_c:
                        raise HTTPException(status_code=400, detail={
                            "message": f"このサイト目的（{_bc_sp_c or '未設定'}）は {operation_type} に対応していません。業務条件でサイト目的を設定してください。",
                            "operation_type": operation_type,
                            "status": "SITE_PURPOSE_MISMATCH",
                        })
                    if not _task_mapping_for_steps.get("credential_secret_name"):
                        raise HTTPException(status_code=400, detail={
                            "message": "認証情報が設定されていません。先にログイン情報を登録してください。",
                            "operation_type": operation_type,
                            "status": "CREDENTIAL_MISSING",
                        })
                    _scout_op_map = (_task_mapping_for_steps.get("operation_mappings") or {}).get(operation_type, {})
                    if operation_type == "offer_send":
                        if not _operation_mapping_is_production_ready(_scout_op_map):
                            raise HTTPException(status_code=400, detail={
                                "message": "offer_send はAI整備済みではありません。媒体基盤のAI整備でスカウト候補者検索ページを保存してください。",
                                "operation_type": operation_type,
                                "status": "MAPPING_NOT_PRODUCTION_READY",
                                "source": _scout_op_map.get("source") or "unknown",
                            })
                        _offer_url = (
                            _scout_op_map.get("target_url")
                            or clean_payload.get("search_url")
                            or ""
                        )
                        if not _offer_url:
                            raise HTTPException(status_code=400, detail={
                                "message": "スカウト候補者検索ページURLがAI整備済みではありません。媒体基盤のAI整備で offer_send を使える状態にしてください。",
                                "operation_type": operation_type,
                                "status": "RECRUIT_URL_MISSING",
                                "missing": ["offer_send.search_url"],
                            })
                        clean_payload.setdefault("search_url", _offer_url)
                    elif operation_type == "recruit_inbox_scan":
                        if not _operation_mapping_is_production_ready(_scout_op_map):
                            raise HTTPException(status_code=400, detail={
                                "message": "recruit_inbox_scan はAI整備済みではありません。媒体基盤のAI整備で応募/返信の受信ボックスURLを保存してください。",
                                "operation_type": operation_type,
                                "status": "MAPPING_NOT_PRODUCTION_READY",
                                "source": _scout_op_map.get("source") or "unknown",
                            })
                        _inbox_url = (
                            _scout_op_map.get("target_url")
                            or clean_payload.get("inbox_url")
                            or ""
                        )
                        if not _inbox_url:
                            raise HTTPException(status_code=400, detail={
                            "message": "応募/返信の受信ボックスURLがAI整備済みではありません。媒体基盤のAI整備で recruit_inbox_scan を使える状態にしてください。",
                                "operation_type": operation_type,
                                "status": "RECRUIT_URL_MISSING",
                                "missing": ["recruit_inbox_scan.inbox_url"],
                            })
                        clean_payload.setdefault("inbox_url", _inbox_url)
                    elif operation_type == "recruit_reply" and not clean_payload.get("reply_url"):
                        raise HTTPException(status_code=400, detail={
                            "message": "返信先URL(reply_url)が必要です。受信ボックス監視で取得した会話スレッド、または候補者ページのURLを指定してください。",
                            "operation_type": operation_type,
                            "status": "REPLY_URL_MISSING",
                            "missing": ["reply_url"],
                        })
                    _operation_steps = []  # スカウト系はsteps不要（専用executorが処理）
                    print(f"[CREATE_TASK_STEPS] source=scout_op op={operation_type} site_purpose={_bc_sp_c}", flush=True)
                else:
                    if agent_type == "hp_update" and not req.operation_mapping_override:
                        _gate_op_map = (_task_mapping_for_steps.get("operation_mappings") or {}).get(operation_type, {})
                        if not _operation_mapping_is_production_ready(_gate_op_map):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "message": "この操作はAI整備前の候補のため本番実行できません。媒体基盤のAI整備で対象ページ・入力項目・保存操作を保存してください。",
                                    "operation_type": operation_type,
                                    "status": "MAPPING_NOT_PRODUCTION_READY",
                                    "source": _gate_op_map.get("source") or "unknown",
                                    "target_url": _gate_op_map.get("target_url") or "",
                                },
                            )
                    _task_mapping_for_steps, _ = _ensure_capability_view_for_mapping(db, _mm_id_create, _task_mapping_for_steps)
                    _task_capability_op = _operation_from_capability_view(_task_mapping_for_steps, operation_type)
                    if _task_capability_op:
                        _structural_gate_checked = True
                        if _task_capability_op.get("status") != "READY" or not _task_capability_op.get("taskable"):
                            # AI整備済みoperationだけはcapability_viewの遅延更新を吸収して通す。
                            _manual_op_map_c = (_task_mapping_for_steps.get("operation_mappings") or {}).get(operation_type, {})
                            _manual_ready_c = _operation_mapping_is_production_ready(_manual_op_map_c)
                            if not _manual_ready_c and agent_type == "hp_update":
                                raise HTTPException(
                                    status_code=400,
                                    detail={
                                        "message": "この媒体は指定operationを構造的READYとして実行できません。媒体構造schemaの候補を確認してください。",
                                        "operation_type": operation_type,
                                        "status": _task_capability_op.get("status") or "UNDISCOVERED",
                                        "missing": _task_capability_op.get("missing", []),
                                        "source": "capability_view",
                                    },
                                )
                        clean_payload.setdefault("target_url", _task_capability_op.get("target_url") or "")
                    _mm_steps_by_type = (_task_mapping_for_steps.get("operation_steps_by_type") or {})
                    _steps_from_mm = None if req.operation_mapping_override else _mm_steps_by_type.get(operation_type)
                    if _steps_from_mm:
                        _operation_steps = _steps_from_mm
                        print(f"[CREATE_TASK_STEPS] source=capability_view op={operation_type} steps_count={len(_steps_from_mm)}", flush=True)
                    else:
                        # AI整備済みoperation: 保存済みselectorからstepsをオンデマンド生成
                        _manual_op_map_s = (_task_mapping_for_steps.get("operation_mappings") or {}).get(operation_type, {})
                        if _operation_mapping_is_production_ready(_manual_op_map_s) and _manual_op_map_s.get("selectors"):
                            _target_url_s = _manual_op_map_s.get("target_url", "")
                            _inline_steps = [
                                {"step_type": "login", "display_name": "ログイン", "status": "READY", "required": True},
                                {"step_type": "navigate", "display_name": "画面へ移動", "status": "READY", "required": True, "target_url": _target_url_s},
                            ]
                            for _sk, _sv in _manual_op_map_s["selectors"].items():
                                if _sk in ("save", "submit"):
                                    continue
                                _st = "upload_file" if (isinstance(_sv, dict) and _sv.get("type") == "file") else "fill"
                                _inline_steps.append({
                                    "step_type": _st, "display_name": isinstance(_sv, dict) and _sv.get("label") or _sk,
                                    "status": "READY", "required": False,
                                    "selector_key": _sk, "payload_key": _sk,
                                    "selector": isinstance(_sv, dict) and _sv.get("selector") or _sv,
                                })
                            if _manual_op_map_s.get("save_selector") or (_manual_op_map_s["selectors"].get("save") or {}).get("selector"):
                                _save_sel = _manual_op_map_s.get("save_selector") or (_manual_op_map_s["selectors"].get("save") or {}).get("selector")
                                _inline_steps.append({"step_type": "click", "display_name": "保存", "status": "READY", "required": True, "selector": _save_sel})
                            _inline_steps.append({"step_type": "verify", "display_name": "反映確認", "status": "READY", "required": True})
                            _operation_steps = _inline_steps
                            print(f"[CREATE_TASK_STEPS] source=ai_confirmed_selectors op={operation_type} steps={len(_inline_steps)}", flush=True)
            elif agent_type == "hp_update":
                raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
        except Exception as _steps_e:
            if isinstance(_steps_e, HTTPException):
                raise
            print(f"[CREATE_TASK_STEPS_ERROR] {_steps_e}", flush=True)
    if _operation_steps is None:
        _operation_steps = _op_steps_template if _op_steps_template and not (_mm_id_create and agent_type == "hp_update" and _structural_gate_checked) else None
        if _operation_steps:
            print(f"[CREATE_TASK_STEPS] source=operation_steps_template op={operation_type}", flush=True)
    if _mm_id_create and agent_type == "hp_update" and _operation_steps is None and not _is_scout_op_create:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "この媒体は指定operationの実行stepsが未生成です。媒体構造schemaの確認または深掘り解析を実行してください。",
                "operation_type": operation_type,
                "status": _task_capability_op.get("status") or "UNDISCOVERED",
                "missing": _task_capability_op.get("missing", ["operation_steps"]),
            },
        )
    task_id    = str(uuid.uuid4())
    # P17: workflow_id生成（複数task chainなら共有可）
    workflow_id = str(uuid.uuid4())
    # P28: task作成時にbefore値をPlaywrightで取得（設計図P2-1準拠）
    _before_values = {}
    if _mm_id_create and agent_type == "hp_update":
        _p28_before_capture_enabled = os.environ.get("P28_BEFORE_CAPTURE_ENABLED", "false").lower() == "true"
        if not _p28_before_capture_enabled:
            print(f"[P28_BEFORE_CAPTURE_SKIPPED] op={operation_type} reason=P28_BEFORE_CAPTURE_ENABLED_FALSE", flush=True)
        else:
            try:
                from api.core.browser_executor import is_playwright_enabled as _is_pw_enabled
                if not _is_pw_enabled():
                    print(f"[P28_BEFORE_CAPTURE_SKIPPED] op={operation_type} reason=PLAYWRIGHT_DISABLED", flush=True)
                else:
                    from api.core.secret_manager import get_secret_json as _get_secret
                    from api.core.browser_executor import _capture_before_values as _cbv
                    from playwright.sync_api import sync_playwright as _swp
                    _mm_doc_bv = db.collection("media_mappings").document(_mm_id_create).get()
                    if _mm_doc_bv.exists:
                        _mm_bv = _mm_doc_bv.to_dict() or {}
                        _secret_bv = _mm_bv.get("credential_secret_name", "")
                        _creds_bv = _get_secret(_secret_bv) if _secret_bv else None
                        if _creds_bv and not _creds_bv.get("blocked"):
                            from api.core.browser_executor import create_authenticated_page as _cap
                            with _swp() as _p_bv:
                                _auth_bv = _cap(_p_bv, _mm_bv, _creds_bv)
                                _page_bv = _auth_bv.get("page")
                                _browser_bv = _auth_bv.get("browser")
                                try:
                                    if _page_bv:
                                        _before_values = _cbv(_page_bv, _mm_bv, operation_type)
                                        print(f"[P28_BEFORE_CAPTURE] op={operation_type} fields={list(_before_values.keys())}", flush=True)
                                finally:
                                    try:
                                        _browser_bv.close()
                                    except Exception:
                                        pass
            except Exception as _bv_err:
                print(f"[P28_BEFORE_CAPTURE_ERROR] op={operation_type} error={type(_bv_err).__name__}", flush=True)
                _before_values = {}
    preview = _build_preview(agent_type, operation_type, industry, clean_payload, operation_steps=_operation_steps, before_values=_before_values)
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
        "media_mapping_id":        _media_mapping_id,
        "operation_mapping_override": req.operation_mapping_override or {},
        # P25: workflow_session_id（_p20_session_id確定後に上書き）
        "workflow_session_id":     "",
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
        if _media_mapping_id:
            try:
                _p20_doc = db.collection("media_mappings").document(_media_mapping_id).get()
                if _p20_doc.exists:
                    _p20_candidate = _p20_doc.to_dict() or {}
                    if _p20_candidate.get("tenant_id") == _tenant_id_create or _ctx_create.get("is_admin"):
                        _p20_mm = _p20_candidate
                        _p20_mm["mapping_id"] = _media_mapping_id
                        _p20_mm["id"] = _media_mapping_id
                        _p20_media_name = _p20_mm.get("media_name", _p20_media_name)
            except Exception as _p20_mm_e:
                print(f"[P20_MEDIA_MAPPING_ID_ERROR] {type(_p20_mm_e).__name__}", flush=True)
        if not _p20_mm and _p20_media_name:
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

    # workflow_session_idをFirestoreに保存（FAILED時のrebuild_execution_planで参照）
    if _p20_session_id:
        try:
            db.collection("agent_tasks").document(task_id).update({
                "workflow_session_id": _p20_session_id,
                "risk_level": _p20_risk.get("risk_level", ""),
                "risk_score": _p20_risk.get("risk_score", 0.0),
                "risk_factors": _p20_risk.get("risk_factors", []),
                "require_human_approval": _p20_risk.get("require_human_approval", False),
            })
            print(f"[P20_SESSION_SAVED] task_id={task_id} session_id={_p20_session_id}", flush=True)
        except Exception as _wss_e:
            print(f"[P20_SESSION_SAVE_ERROR] {_wss_e}", flush=True)
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
    _assert_tenant_access(task, user, "このタスクを承認する権限がありません")
    if task.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="PENDING状態のタスクのみ承認できます。現在: " + str(task.get("status")))
    ref.update({
        "status": "APPROVED",
        "approved_by": user.get("uid", ""),
        "approved_at": datetime.datetime.utcnow(),
    })
    try:
        session_id = task.get("workflow_session_id") or ""
        if session_id:
            sref = db.collection("workflow_execution_sessions").document(session_id)
            ssnap = sref.get()
            if ssnap.exists:
                session = ssnap.to_dict() or {}
                _assert_tenant_access(session, user, "このワークフローを承認する権限がありません")
                if not session.get("cancelled"):
                    now = datetime.datetime.utcnow()
                    sref.update({
                        "approval_state": "APPROVED",
                        "status": "READY",
                        "approved_by": user.get("uid", ""),
                        "approved_at": now,
                        "updated_at": now,
                    })
    except Exception as _wf_approve_e:
        print(f"[TASK_APPROVE_WORKFLOW_SYNC_ERROR] {type(_wf_approve_e).__name__}:{_wf_approve_e}", flush=True)
    # P31: 承認カウントアップ → 6回で auto_enabled 昇格
    try:
        _p31_tenant = task.get("tenant_id", "")
        _p31_op     = task.get("operation_type", "")
        if _p31_tenant and _p31_op:
            _p31_ref = db.collection("agent_permissions").document(_p31_tenant)
            _p31_tx = db.transaction()

            @firestore.transactional
            def _increment_approval_count(transaction, ref):
                snap = ref.get(transaction=transaction)
                data = snap.to_dict() if snap.exists else {}
                ops = dict(data.get("operations", {}) or {})
                op_data = dict(ops.get(_p31_op, {}) or {})
                count = int(op_data.get("approval_count") or 0) + 1
                auto = (
                    count >= 6
                    and _p31_op in AUTO_APPROVE_LOW_RISK_OPERATIONS
                    and _p31_op not in AUTO_APPROVE_FORBIDDEN_OPERATIONS
                )
                ops[_p31_op] = {
                    **op_data,
                    "approval_count": count,
                    "auto_enabled": auto,
                    "last_approved_at": datetime.datetime.utcnow(),
                    "last_approved_by": user.get("uid", ""),
                }
                transaction.set(ref, {"operations": ops, "tenant_id": _p31_tenant}, merge=True)
                return count, auto

            _p31_count, _p31_auto = _increment_approval_count(_p31_tx, _p31_ref)
            print(f"[P31_APPROVAL_COUNT] tenant={_p31_tenant} op={_p31_op} count={_p31_count} auto_enabled={_p31_auto}", flush=True)
            if _p31_auto:
                print(f"[P31_AUTO_PROMOTED] tenant={_p31_tenant} op={_p31_op} auto_enabled=True", flush=True)
    except Exception as _p31_e:
        print(f"[P31_ERROR] {_p31_e}", flush=True)
    # P31: auto_promotedフラグをレスポンスに含める（UI側で「自動化しますか？」ボタン表示に使用）
    _p31_promoted = False
    _p31_count_resp = 0
    try:
        _p31_resp_ref = db.collection("agent_permissions").document(task.get("tenant_id", "")).get()
        if _p31_resp_ref.exists:
            _p31_resp_ops = (_p31_resp_ref.to_dict() or {}).get("operations", {})
            _p31_resp_op  = _p31_resp_ops.get(task.get("operation_type", ""), {})
            _p31_promoted = _p31_resp_op.get("auto_enabled", False)
            _p31_count_resp = _p31_resp_op.get("approval_count", 0)
    except Exception:
        pass
    return {
        "task_id":        req.task_id,
        "status":         "APPROVED",
        "auto_promoted":  _p31_promoted,
        "approval_count": _p31_count_resp,
    }

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
    _assert_tenant_access(task, user, "このタスクを却下する権限がありません")
    if task.get("status") != "PENDING":
        raise HTTPException(status_code=400, detail="PENDING状態のタスクのみ却下できます")
    ref.update({
        "status": "REJECTED",
        "result": {"reason": req.reason or ""},
        "approved_by": user.get("uid", ""),
        "approved_at": datetime.datetime.utcnow(),
    })
    try:
        session_id = task.get("workflow_session_id") or ""
        if session_id:
            sref = db.collection("workflow_execution_sessions").document(session_id)
            ssnap = sref.get()
            if ssnap.exists:
                session = ssnap.to_dict() or {}
                _assert_tenant_access(session, user, "このワークフローを却下する権限がありません")
                now = datetime.datetime.utcnow()
                sref.update({
                    "approval_state": "REJECTED",
                    "status": "REJECTED",
                    "reject_reason": req.reason or "",
                    "rejected_by": user.get("uid", ""),
                    "rejected_at": now,
                    "updated_at": now,
                })
    except Exception as _wf_reject_e:
        print(f"[TASK_REJECT_WORKFLOW_SYNC_ERROR] {type(_wf_reject_e).__name__}:{_wf_reject_e}", flush=True)
    return {"task_id": req.task_id, "status": "REJECTED"}

@router.delete("/task/{task_id}")
def delete_task(task_id: str, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_tasks").document(task_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict() or {}
    _assert_tenant_access(task, user, "このタスクを削除する権限がありません")
    if task.get("status") == "RUNNING":
        raise HTTPException(status_code=400, detail="実行中のタスクは削除できません")
    try:
        session_id = task.get("workflow_session_id") or ""
        if session_id:
            sref = db.collection("workflow_execution_sessions").document(session_id)
            ssnap = sref.get()
            if ssnap.exists:
                session = ssnap.to_dict() or {}
                _assert_tenant_access(session, user, "このワークフローを削除する権限がありません")
                now = datetime.datetime.utcnow()
                sref.update({
                    "approval_state": "CANCELLED",
                    "status": "CANCELLED",
                    "cancelled": True,
                    "cancel_reason": "task deleted",
                    "cancelled_by": user.get("uid", ""),
                    "cancelled_at": now,
                    "updated_at": now,
                })
    except Exception as _wf_delete_e:
        print(f"[TASK_DELETE_WORKFLOW_SYNC_ERROR] {type(_wf_delete_e).__name__}:{_wf_delete_e}", flush=True)
    ref.delete()
    return {"task_id": task_id, "status": "DELETED"}

@router.post("/task/{task_id}/force-reset")
def force_reset_task(task_id: str, user: dict = Depends(verify_token)):
    """
    H-1/J-1: RUNNING詰まりタスクを強制FAILEDに戻す（admin専用）。
    Cloud RunインスタンスOOM/timeout後にRUNNINGのまま残ったタスクを回復する。
    """
    if user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ使用できます")
    db = get_db()
    ref = db.collection("agent_tasks").document(task_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict() or {}
    if task.get("status") != "RUNNING":
        raise HTTPException(status_code=400, detail=f"RUNNINGでないタスクはforce-resetできません (status={task.get('status')})")
    now = datetime.datetime.utcnow()
    ref.update({
        "status":          "FAILED",
        "result":          {"status": "FAILED", "message": "管理者によるforce-reset (インスタンス異常終了の疑い)", "executed": False},
        "force_reset_at":  now,
        "force_reset_by":  user.get("uid", ""),
        "updated_at":      now,
    })
    print(f"[FORCE_RESET] task_id={task_id} by={user.get('uid','')} tenant={task.get('tenant_id','')}", flush=True)
    return {"task_id": task_id, "status": "FAILED", "reset": True}


@router.post("/task/watchdog/recover-running")
def recover_stuck_running_tasks(x_ascend_watchdog_token: Optional[str] = Header(None)):
    """
    H-1/J-1: RUNNING状態が20分以上続いているタスクを自動FAILED化する。
    Cloud Schedulerから10分間隔で呼び出す。
    認証: X-Ascend-Watchdog-Token ヘッダー（env: WATCHDOG_TOKEN）
    """
    expected = os.environ.get("WATCHDOG_TOKEN", "")
    if not expected or x_ascend_watchdog_token != expected:
        raise HTTPException(status_code=403, detail="watchdog token invalid")
    db = get_db()
    _STUCK_MINUTES = 15
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=_STUCK_MINUTES)
    recovered = []
    try:
        docs = db.collection("agent_tasks").where("status", "==", "RUNNING").stream()
        for doc in docs:
            t = doc.to_dict() or {}
            _updated = t.get("updated_at") or t.get("created_at")
            if _updated is None:
                continue
            if hasattr(_updated, "tzinfo") and _updated.tzinfo is not None:
                import datetime as _dt_wd
                cutoff_aware = cutoff.replace(tzinfo=_dt_wd.timezone.utc)
                is_stuck = _updated < cutoff_aware
            else:
                is_stuck = _updated < cutoff
            if not is_stuck:
                continue
            now = datetime.datetime.utcnow()
            _ckpt = t.get("checkpoint_step_results") or []
            _resume_count = t.get("resume_count", 0)
            _MAX_RESUME = 5
            if _ckpt and _resume_count < _MAX_RESUME:
                _new_status = "APPROVED"
                _new_result = {"status": "APPROVED", "message": f"watchdog: チェックポイントから再開 (resume {_resume_count + 1}/{_MAX_RESUME})", "executed": False}
                _extra = {"resume_count": _resume_count + 1}
            else:
                _new_status = "FAILED"
                _new_result = {"status": "FAILED", "message": f"watchdog: {_STUCK_MINUTES}分以上RUNNINGのため自動回復", "executed": False}
                _extra = {}
            db.collection("agent_tasks").document(doc.id).update({
                "status":         _new_status,
                "result":         _new_result,
                "force_reset_at": now,
                "updated_at":     now,
                **_extra,
            })
            recovered.append(doc.id)
            print(f"[WATCHDOG_RECOVER] task_id={doc.id} status={_new_status} resume_count={_resume_count} tenant={t.get('tenant_id','')} updated_at={_updated}", flush=True)
    except Exception as _e_wd:
        print(f"[WATCHDOG_ERROR] {type(_e_wd).__name__}:{_e_wd}", flush=True)
        raise HTTPException(status_code=500, detail=f"watchdog error: {_e_wd}")
    return {"recovered": recovered, "count": len(recovered)}


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
    _assert_tenant_access(task, user, "このタスクを実行する権限がありません")
    if task.get("status") != "APPROVED":
        raise HTTPException(status_code=400, detail="APPROVEDのタスクのみ実行できます")
    # J-2: op_idが存在する場合、実行時にop定義と operation_type が一致するか再検証
    _exec_op_id = task.get("op_id") or ""
    if _exec_op_id:
        _exec_op_doc = db.collection("agent_ops").document(_exec_op_id).get()
        _exec_default_op = next((dict(o) for o in DEFAULT_AGENT_OPS if o.get("op_id") == _exec_op_id), None)
        if _exec_op_doc.exists or _exec_default_op:
            _exec_op_data = {**(_exec_default_op or {}), **(_exec_op_doc.to_dict() if _exec_op_doc.exists else {})}
            _exec_defined_op_type = _exec_op_data.get("operation_type") or ""
            if _exec_defined_op_type and task.get("operation_type") != _exec_defined_op_type:
                raise HTTPException(
                    status_code=403,
                    detail=f"operation_typeがop_id定義と一致しません: task={task.get('operation_type')} op_id_defines={_exec_defined_op_type}",
                )
        else:
            print(f"[J2_OP_ID_NOT_FOUND] op_id={_exec_op_id} task_id={task_id}", flush=True)
    # P0-1: 実行直前の二重防御
    _ctx_exec = _resolve_agent_user_context(user)
    _enforce_agent_permissions(_ctx_exec, task.get("agent_type", ""), task.get("operation_type", ""))
    workflow_session_id = task.get("workflow_session_id") or ""
    if workflow_session_id:
        try:
            from api.core.browser_executor import check_workflow_approval
            approval = check_workflow_approval(db, workflow_session_id, tenant_id=task.get("tenant_id", ""))
        except Exception as approval_error:
            approval = {
                "approved": False,
                "approval_state": "ERROR",
                "error_type": type(approval_error).__name__,
            }
        if not approval.get("approved"):
            ref.update({
                "workflow_approval": approval,
                "blocked_reason": "workflow approval required",
                "updated_at": datetime.datetime.utcnow(),
            })
            raise HTTPException(
                status_code=403,
                detail=f"P20承認が未完了です: {approval.get('approval_state', 'UNKNOWN')}",
            )
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
        # 媒体マッピングを取得（media_mapping_id優先 → media_name fallback）
        tenant_id = task.get("tenant_id", "default")
        payload   = task.get("payload", {})
        media_name = payload.get("media_name")
        media_mapping = None
        _mm_id = task.get("media_mapping_id") or payload.get("media_mapping_id") or ""
        if _mm_id:
            _mm_doc = db.collection("media_mappings").document(_mm_id).get()
            if _mm_doc.exists:
                _mm_data = _mm_doc.to_dict()
                if _mm_data.get("tenant_id") == tenant_id or user.get("role", "").lower() == "admin":
                    media_mapping = _mm_data
                    media_mapping["_doc_id"] = _mm_id
                    media_mapping["id"] = _mm_id
                    media_mapping["mapping_id"] = _mm_id
                    print(f"[EXEC_MM_ID] media_mapping_id={_mm_id} media_name={_mm_data.get('media_name')}", flush=True)
        if media_mapping is None and media_name:
            docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream()
            for d in docs:
                dm = d.to_dict()
                if dm.get("media_name") == media_name:
                    media_mapping = dm
                    media_mapping["id"] = d.id
                    media_mapping["mapping_id"] = d.id
                    break

        if media_mapping and task.get("operation_mapping_override"):
            _op_type_override = task.get("operation_type", "")
            _override = task.get("operation_mapping_override") or {}
            _op_maps_override = dict(media_mapping.get("operation_mappings") or {})
            _op_maps_override[_op_type_override] = _override
            media_mapping = dict(media_mapping)
            media_mapping["operation_mappings"] = _op_maps_override
            try:
                from api.core.browser_executor import GENERIC_OPERATION_CONFIG as _generic_cfg_override
                _cap_key_override = (_generic_cfg_override.get(_op_type_override) or {}).get("capability_key", "")
                if _cap_key_override:
                    _caps_override = dict(media_mapping.get("capabilities") or {})
                    _caps_override[_cap_key_override] = True
                    media_mapping["capabilities"] = _caps_override
            except Exception as _cap_override_err:
                print(f"[EXEC_MENU_ITEM_CAP_OVERRIDE_ERROR] {type(_cap_override_err).__name__}:{_cap_override_err}", flush=True)
            print(f"[EXEC_MENU_ITEM_OVERRIDE] op={_op_type_override} target={str(_override.get('target_url',''))[:80]}", flush=True)

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
                raise HTTPException(status_code=400, detail=f"[P29_VALIDATION_FAILED] {_val_msg}")
            print(f"[P29_VALIDATION_OK] operation_type={task.get('operation_type','')} errors=0", flush=True)
        # Checkpoint/resume: read prior completed steps
        _checkpoint = task.get("checkpoint_step_results") or []
        if _checkpoint:
            _ckpt_done = len([r for r in _checkpoint if r.get("status") == "DONE"])
            print(f"[CHECKPOINT_RESUME] task_id={task_id} prior_done={_ckpt_done}", flush=True)
        # executor層に委譲
        result = execute_agent_task(task, media_mapping, task_id=task_id, db=db, prior_step_results=_checkpoint)
        final_status = result.get("status", "WAITING_EXECUTOR")

        # P26: FAILED時の原因分類別リトライ（task-levelは最大1回）
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
        def _is_retry_safe(res: dict) -> bool:
            if res.get("retry_safe") is False or res.get("terminal_step_done") is True:
                return False
            try:
                from api.core.browser_executor import _terminal_step_done as _terminal_done_check
                return not _terminal_done_check(task.get("operation_steps") or [], res.get("step_results") or [])
            except Exception:
                return True
        _retry_reason = _classify_retry_reason(result) if final_status == "FAILED" and not result.get("executed") else ""
        _retry_triggerable = bool(_retry_reason) and _is_retry_safe(result)
        if _retry_reason and not _retry_triggerable:
            result["retry_skipped_reason"] = "terminal_step_already_done_or_not_retry_safe"
            print(f"[P26_RETRY_SKIP] reason={_retry_reason} task_id={task_id} retry_safe=False", flush=True)
        if _retry_triggerable:
            _current_retry = task.get("retry_count", 0)
            if _current_retry < 1:  # P26: task-level retry 最大1回（step-level retryはbrowser_executor側で処理）
                _now_retry = _dt.datetime.utcnow()
                # J-3: APPROVEDを経由すると二重実行ウィンドウが生じる。
                # リトライはRUNNING状態を維持したまま同一コンテキスト内で再実行する
                ref.update({
                    "retry_count":   _current_retry + 1,
                    "last_retry_at": _now_retry,
                    "retry_reason":  _retry_reason,
                })
                task["retry_count"] = _current_retry + 1
                task["last_retry_at"] = _now_retry
                task["retry_reason"] = _retry_reason
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
                # [P13_STATUS_UNIFIED] 最終ref.updateは813行目に一本化。ここでは更新しない
            else:
                result["retry_exhausted"] = True
                result["retry_reason"]    = _retry_reason
                print(f"[P26_RETRY] リトライ上限到達 reason={_retry_reason} count={_current_retry}", flush=True)
        # executorが返したstatusをそのままFirestoreに反映
        # DONE以外はDONEにしない
        _final_update = {"status": final_status, "result": result}
        if final_status == "DONE":
            _final_update["checkpoint_step_results"] = []
            _final_update["checkpoint_step_id"] = ""
            _final_update["resume_count"] = 0
        ref.update(_final_update)
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
                    from api.core.browser_executor import GENERIC_OPERATION_CONFIG, update_selector_learning_stats
                    _cfg_fb = GENERIC_OPERATION_CONFIG.get(_ot, {}) or {}
                    _feedback_keys = {
                        str(f.get("selector_key") or "")
                        for f in (_cfg_fb.get("fields") or [])
                        if isinstance(f, dict) and f.get("selector_key")
                    }
                    if _cfg_fb.get("submit_selector_key"):
                        _feedback_keys.add(str(_cfg_fb.get("submit_selector_key")))
                    _dom_feedback = dict(_mm_data.get("dom_selectors") or {})
                    _op_feedback = (((media_mapping or _mm_data).get("operation_mappings") or {}).get(_ot) or {}).get("selectors") or {}
                    for _role, _sel_data in _op_feedback.items():
                        _sel_val = _sel_data.get("selector") if isinstance(_sel_data, dict) else str(_sel_data)
                        if _sel_val:
                            _dom_feedback[_role] = _sel_val
                    for _sk in sorted(_feedback_keys):
                        _sv = _dom_feedback.get(_sk)
                        if not _sv:
                            continue
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
                    # B-1 deferred: update_selector_transition_graph requires prev_selector and next_selector; single selector feedback path cannot call it safely.
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

        # monitoring_results: post_monitoringタスクDONE時にGemini分析して専用コレクションへ保存
        if final_status == "DONE" and task.get("operation_type") == "post_monitoring":
            try:
                import datetime as _mr_dt
                _mr_page_text  = result.get("page_text", "")
                _mr_tenant     = task.get("tenant_id", "")
                _mr_mapping_id = task.get("media_mapping_id") or task.get("payload", {}).get("media_mapping_id", "")
                _mr_raw        = result.get("monitoring_result", {})
                _mr_mkt        = _mr_raw.get("marketing", {})
                _mr_industry   = task.get("payload", {}).get("industry") or task.get("industry") or "nightlife"
                _mr_ai         = _monitoring_ai_analyze(_mr_page_text, _mr_industry) if _mr_page_text else {}
                get_db().collection("monitoring_results").add({
                    "tenant_id":       _mr_tenant,
                    "mapping_id":      _mr_mapping_id,
                    "task_id":         task_id,
                    "executed_at":     _mr_dt.datetime.utcnow(),
                    "monitoring_target": result.get("monitoring_target", ""),
                    "industry":        _mr_industry,
                    "trending_phrases": _mr_ai.get("trending_phrases", []),
                    "popular_types":   _mr_ai.get("popular_types", []),
                    "avoid_phrases":   _mr_ai.get("avoid_phrases", []),
                    "ai_summary":      _mr_ai.get("ai_summary", ""),
                    "recommendations": (_mr_mkt.get("recommendations", []) + _mr_ai.get("recommendations", [])),
                    "keyword_hits":    _mr_mkt.get("keyword_hits", {}),
                    "active_casts":    _mr_mkt.get("active_casts", []),
                    "silent_casts":    _mr_mkt.get("silent_casts", []),
                    "total_posts":     _mr_raw.get("total_posts", 0),
                    "competitors":     _mr_raw.get("competitors", []),
                })
                print(f"[MONITORING_RESULTS] 保存完了 tenant={_mr_tenant} mapping={_mr_mapping_id}", flush=True)
            except Exception as _mr_e:
                print(f"[MONITORING_RESULTS] 保存エラー: {type(_mr_e).__name__}: {_mr_e}", flush=True)

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

        # P20: rebuild_execution_plan（FAILED時・再計画提案のみ・自動実行禁止）
        if final_status == "FAILED":
            try:
                from api.core.browser_executor import rebuild_execution_plan
                _p20_session_id  = task.get("workflow_session_id", "")
                _p20_tenant      = task.get("tenant_id", "")
                _p20_op_type     = task.get("operation_type", "")
                _p20_steps       = task.get("operation_steps") or []
                _p20_failed_step = ((_last_selector_repair or {}).get("failed_selectors") or [""])[0]
                _p20_reason      = str((result or {}).get("error", "") or "unknown")
                _p20_policy      = task.get("execution_policy") or {}
                if _p20_session_id and _p20_steps:
                    _replan = rebuild_execution_plan(
                        db=get_db(),
                        session_id=_p20_session_id,
                        tenant_id=_p20_tenant,
                        operation_type=_p20_op_type,
                        operation_steps=_p20_steps,
                        failed_step=_p20_failed_step,
                        failure_reason=_p20_reason,
                        execution_policy=_p20_policy,
                    )
                    print(f"[P20_REPLAN] session={_p20_session_id} replanned={_replan.get('replanned')} branch={_replan.get('branch_taken')}", flush=True)
            except Exception as _p20_e:
                print(f"[P20_REPLAN_ERROR] {type(_p20_e).__name__}: {_p20_e}", flush=True)
        # P27: 成功/失敗 anomaly check（operation単位・全ケース）
        try:
            from api.core.browser_executor import _p27_anomaly_check
            _p27_mapping_id = str((media_mapping or {}).get("id") or (media_mapping or {}).get("mapping_id") or "")
            _p27_op_type    = task.get("operation_type", "")
            if _p27_mapping_id:
                if final_status == "DONE":
                    _p27_anomaly_check(
                        mapping_id=_p27_mapping_id,
                        event_type="success",
                        operation_type=_p27_op_type,
                    )
                elif final_status == "FAILED":
                    _p27_anomaly_check(
                        mapping_id=_p27_mapping_id,
                        event_type="operation_failed",
                        operation_type=_p27_op_type,
                    )
        except Exception as _p27_e:
            print(f"[P27_ANOMALY_CALL_ERROR] {type(_p27_e).__name__}: {_p27_e}", flush=True)
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


def _iso_top_level(doc: dict) -> dict:
    out = dict(doc or {})
    for k, v in list(out.items()):
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


def _load_batch_mapping_targets(db, tenant_id: str, mapping_ids: list[str], user: dict) -> list[tuple[str, dict]]:
    targets = []
    _is_admin = user.get("role", "").lower() == "admin"
    if mapping_ids:
        for mid in mapping_ids[:50]:
            snap = db.collection("media_mappings").document(str(mid)).get()
            if not snap.exists:
                targets.append((str(mid), {"_missing": True}))
                continue
            mm = snap.to_dict() or {}
            if not _is_admin and mm.get("tenant_id") != tenant_id:
                continue
            mm["id"] = str(mid)
            mm["mapping_id"] = str(mid)
            targets.append((str(mid), mm))
        return targets

    docs = db.collection("media_mappings").where("tenant_id", "==", tenant_id).limit(50).stream()
    for d in docs:
        mm = d.to_dict() or {}
        mm["id"] = d.id
        mm["mapping_id"] = d.id
        targets.append((d.id, mm))
    return targets


# 操作対象として絶対に使ってはいけないURL（FAQヘルプ・外部フォーム・PDF・ガイドライン等）。
# deep scanがメニュー内の「〜について」ヘルプリンクを誤って操作対象に登録するため、
# 実行・プレビュー時にこれらを除外して正しい管理画面URLのみを採用する。
_CROSS_MEDIA_URL_BLOCKLIST_SUBSTR = (
    "hgjn-help.com",          # ヘブンFAQヘルプセンター
    "site_domain=admin_faq",  # FAQ判定パラメータ
    "/faq",
    "zohopublic.com",         # 外部Zohoフォーム
    "forms.office.com",
    "forms.cloud.microsoft",
    "guideline",
    "disclaimer",
    "/fairness",
    "teamviewer.com",
    "lin.ee",
    ".pdf",
)


def _is_blocked_operation_url(url: str) -> bool:
    if not url:
        return True
    u = str(url).strip().lower()
    if not u.startswith(("http://", "https://")):
        return True
    if "{" in u:  # 未解決プレースホルダー（{id}等）
        return True
    return any(bad in u for bad in _CROSS_MEDIA_URL_BLOCKLIST_SUBSTR)


def _registrable_domain(url_or_host: str) -> str:
    """URL/ホストから登録可能ドメインを大まかに抽出（例: newmanager.cityheaven.net -> cityheaven.net）。
    co.jp等の3階層JP-TLDのみ3ラベル保持。完全な公開接尾辞解決ではないが、
    同一サイト判定（FAQ/外部ドメイン除外）には十分。"""
    import re as _re_dom
    s = str(url_or_host or "").strip().lower()
    m = _re_dom.match(r"^[a-z][a-z0-9+.\-]*://([^/:?#]+)", s)
    host = m.group(1) if m else s.split("/")[0]
    labels = [l for l in host.split(".") if l]
    if len(labels) <= 2:
        return ".".join(labels)
    if labels[-1] == "jp" and labels[-2] in ("co", "ne", "or", "go", "ac", "ad", "ed", "gr", "lg"):
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _mapping_fields_for_operation(mapping: dict, operation_type: str) -> list[dict]:
    """Return form fields only from AI-confirmed executable operation mappings."""
    out: list[dict] = []
    seen = set()

    def _add_fields(fields, source: str = "") -> None:
        for idx, field in enumerate(fields or []):
            if not isinstance(field, dict):
                continue
            if not _is_actionable_mapping_field(field):
                continue
            selector = str(field.get("selector") or "").strip()
            canonical = str(field.get("canonical") or "").strip()
            name = str(field.get("name") or "").strip()
            fid = str(field.get("id") or "").strip()
            label = str(field.get("label") or "").strip()
            key = (selector, canonical, name, fid, label)
            if key in seen:
                continue
            seen.add(key)
            row = dict(field)
            row.setdefault("source", source or row.get("source") or "mapping")
            row.setdefault("_field_index", idx)
            out.append(row)

    op_map = (mapping.get("operation_mappings") or {}).get(operation_type) or {}
    if _operation_mapping_is_production_ready(op_map):
        _add_fields(op_map.get("fields") or (op_map.get("form_schema") or {}).get("fields") or [], "ai_confirmed_operation")

    return out


def _payload_keys_for_mapping_fields(fields: list[dict]) -> set[str]:
    """Derive payload keys used by manual/capability steps from mapped form fields."""
    import re as _re_payload_keys

    keys: set[str] = set()
    for idx, field in enumerate(fields or []):
        if not isinstance(field, dict):
            continue
        if not _is_actionable_mapping_field(field):
            continue
        raw_candidates = [
            str(field.get("canonical") or "").split(".")[-1],
            field.get("name"),
            field.get("id"),
            field.get("key"),
            f"field_{idx}",
        ]
        for raw in raw_candidates:
            raw_s = str(raw or "").strip()
            if not raw_s:
                continue
            key = _re_payload_keys.sub(r"[^0-9A-Za-z_\-]+", "_", raw_s).strip("_")
            if key:
                keys.add(key[:80])
                break
    return keys


def _normalize_mapping_payload_field_type(field_type: str) -> str:
    t = str(field_type or "text").strip().lower()
    if t == "textarea":
        return "textarea"
    if t == "select":
        return "select"
    if t in {"number", "range"}:
        return "number"
    if t in {"checkbox", "radio", "boolean"}:
        return "boolean"
    if t in {"datetime", "datetime-local", "date", "time"}:
        return "datetime"
    if t == "file":
        return "file"
    return "text"


def _sanitize_mapping_payload_key(raw: str, fallback: str, used: set[str]) -> str:
    import re as _re_payload_keys

    key = _re_payload_keys.sub(r"[^0-9A-Za-z_\-]+", "_", str(raw or fallback or "field")).strip("_")
    if not key:
        key = fallback or "field"
    uniq = key[:80]
    suffix = 2
    while uniq in used:
        uniq = f"{key[:70]}_{suffix}"[:80]
        suffix += 1
    used.add(uniq)
    return uniq


def _payload_schema_fields_for_operation(mapping: dict | None, op_data: dict, operation_type: str) -> list[dict]:
    """
    Build the effective payload schema for an operation.
    Base op schema is kept, and AI-confirmed mapping fields are appended as
    first-class payload fields so chat/planning sees the same executable
    structure as cross-media and task creation.
    """
    used: set[str] = set()
    fields: list[dict] = []

    for field in ((op_data.get("payload_schema") or {}).get("fields") or []):
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key or key in used:
            continue
        used.add(key)
        fields.append(dict(field))

    if not mapping:
        return fields

    for idx, field in enumerate(_mapping_fields_for_operation(mapping, operation_type)):
        tail = str(field.get("canonical") or "").split(".")[-1]
        raw_key = str(field.get("key") or tail or field.get("name") or field.get("id") or f"field_{idx}")
        key = _sanitize_mapping_payload_key(raw_key, f"field_{idx}", used)
        fields.append({
            "key": key,
            "label": str(field.get("label") or field.get("name") or field.get("canonical") or field.get("id") or key),
            "type": _normalize_mapping_payload_field_type(field.get("type") or "text"),
            "required": bool(field.get("required")),
            "options": field.get("options") if isinstance(field.get("options"), list) else None,
            "canonical": str(field.get("canonical") or ""),
            "source": str(field.get("source") or "mapping"),
        })
    return fields


def _resolve_cross_media_target(db, mapping_id: str, mapping: dict, operation_type: str) -> dict:
    """
    クロスメディアの送信先フォームURLを解決する。
    旧MANUAL/関連操作/FALLBACKは誤投稿の原因になるため使わない。
    AI整備済み(operation_mappings[op].production_ready/AI_CONFIRMED)だけを本番対象にする。
    """
    op_maps = mapping.get("operation_mappings") or {}
    _om = op_maps.get(operation_type) or {}

    def _candidate_ok(candidate_url: str) -> tuple[bool, str]:
        if not candidate_url:
            return False, ""
        page = _find_mapping_page_by_url(mapping, candidate_url)
        reason = _cross_media_target_mismatch_reason(operation_type, mapping, candidate_url, page)
        return not bool(reason), reason

    url = str(_om.get("target_url") or "").strip()
    if _operation_mapping_is_production_ready(_om) and url and not _is_blocked_operation_url(url):
        ok, reason = _candidate_ok(url)
        if ok:
            return {"url": url, "source": "AI_CONFIRMED", "verified": True}
        return {"url": "", "source": "NONE", "verified": False, "reason": reason or "AI整備済みURLが対象操作のフォームに見えません"}

    return {
        "url": "",
        "source": "NONE",
        "verified": False,
        "reason": "AI整備済みの対象URLがありません。媒体基盤のAI整備で対象ページ・入力項目・保存操作を保存してください。",
    }


def _resolve_cross_media_target_url(db, mapping_id: str, mapping: dict, operation_type: str) -> str:
    """後方互換: URL文字列のみ返す薄いラッパー。"""
    return _resolve_cross_media_target(db, mapping_id, mapping, operation_type).get("url", "")


def _batch_steps_for_mapping(db, mapping_id: str, mapping: dict, operation_type: str) -> list:
    cap_steps, cap_op, _ = _capability_steps_for_mapping(db, mapping_id, mapping, operation_type)
    if cap_op:
        return cap_steps
    op_maps = mapping.get("operation_mappings") or {}
    op_map = op_maps.get(operation_type) or {}
    if not _operation_mapping_is_production_ready(op_map):
        return []
    steps = ((mapping.get("operation_steps_by_type") or {}).get(operation_type)) or []
    if steps:
        return steps
    if op_map.get("status") != "READY" or not op_map.get("target_url"):
        return []
    try:
        from api.core.browser_executor import rebuild_operation_steps
        nav = mapping.get("navigation_graph") or {}
        dlist = mapping.get("operation_candidates_detail") or []
        detail = {d["operation_type"]: d for d in dlist if isinstance(d, dict) and d.get("operation_type")}
        built = rebuild_operation_steps([operation_type], nav, op_maps, detail).get(operation_type) or []
        if built:
            cur_steps = dict(mapping.get("operation_steps_by_type") or {})
            cur_steps[operation_type] = built
            db.collection("media_mappings").document(mapping_id).update({
                "operation_steps_by_type": cur_steps,
                "updated_at": datetime.datetime.utcnow(),
            })
        return built
    except Exception as e:
        print(f"[P35_STEPS_BUILD_ERROR] mapping_id={mapping_id} op={operation_type} error={type(e).__name__}", flush=True)
        return []


def _extract_public_url_snapshot(source_url: str) -> dict:
    """公開URLから軽量な中間データを作る。認証回避や規約回避はしない。"""
    if not source_url or not str(source_url).startswith(("http://", "https://")):
        return {"ok": False, "error": "source_url_invalid"}
    try:
        import re
        import html as _html
        from urllib.request import Request, urlopen
        req = Request(
            source_url,
            headers={
                "User-Agent": "ASCEND-Agent-OS/1.0 (+authorized cross-media extraction)",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.2",
            },
        )
        with urlopen(req, timeout=12) as res:
            content_type = res.headers.get("Content-Type", "")
            raw = res.read(1_000_000)
        if "html" not in content_type.lower() and "text" not in content_type.lower():
            return {"ok": False, "error": f"unsupported_content_type:{content_type}"}
        text = raw.decode("utf-8", errors="ignore")
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
        if m:
            title = _html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
        desc = ""
        m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', text, flags=re.I | re.S)
        if m:
            desc = _html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
        cleaned = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = _html.unescape(re.sub(r"\s+", " ", cleaned)).strip()
        image_urls = []
        for img in re.findall(r'<img[^>]+src=["\'](.*?)["\']', text, flags=re.I | re.S)[:20]:
            try:
                from urllib.parse import urljoin
                image_urls.append(urljoin(source_url, img))
            except Exception:
                image_urls.append(img)
        items = []
        try:
            import json
            for block in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', text, flags=re.I | re.S)[:10]:
                try:
                    data = json.loads(_html.unescape(block).strip())
                except Exception:
                    continue
                stack = data if isinstance(data, list) else [data]
                expanded = []
                for obj in stack:
                    if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                        expanded.extend(obj.get("@graph") or [])
                    else:
                        expanded.append(obj)
                for obj in expanded:
                    if not isinstance(obj, dict):
                        continue
                    name = obj.get("name") or obj.get("headline") or obj.get("title") or ""
                    desc2 = obj.get("description") or obj.get("text") or ""
                    url2 = obj.get("url") or obj.get("@id") or source_url
                    img2 = obj.get("image") or []
                    if isinstance(img2, str):
                        img2 = [img2]
                    if name or desc2:
                        items.append({
                            "name": str(name),
                            "title": str(name),
                            "body": str(desc2),
                            "text": str(desc2),
                            "value": str(desc2),
                            "source_url": str(url2),
                            "image_urls": img2[:10] if isinstance(img2, list) else [],
                        })
                if len(items) >= 20:
                    break
        except Exception:
            items = []
        return {
            "ok": True,
            "source_url": source_url,
            "title": title,
            "description": desc,
            "body_text": cleaned[:8000],
            "image_urls": image_urls,
            "items": items[:20],
            "content_type": content_type,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}:{e}"}


def _cross_media_source_candidates(req: CrossMediaTaskCreateRequest, source_mapping: dict) -> list[dict]:
    candidates: list[dict] = []
    seen = set()

    def add(url: str, label: str = "", source: str = "", score: int = 0):
        url = str(url or "").strip()
        if not url or not url.startswith(("http://", "https://")) or url in seen:
            return
        seen.add(url)
        candidates.append({"url": url, "label": label or url, "source": source, "score": score})

    payload = req.source_payload or {}
    # 対象指定（誰を）: 選択エンティティの詳細URLを最優先
    add(getattr(req, "source_entity_url", None) or "", "選択した対象", "entity", 130)
    add(payload.get("source_url") or payload.get("url") or "", "payload URL", "payload", 120)
    add(req.source_url or "", "指定URL", "request", 110)

    query = " ".join([req.query or "", req.instruction or ""]).strip().lower()

    def text_score(text: str, base: int) -> int:
        text_l = str(text or "").lower()
        if not query:
            return base
        score = base
        if query and query in text_l:
            score += 60
        for token in [t for t in query.replace("　", " ").split(" ") if len(t) >= 2][:12]:
            if token in text_l:
                score += 10
        return score

    for item in (source_mapping.get("manual_menu_scan_results") or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        text = " ".join([item.get("text") or "", item.get("label") or "", item.get("title") or "", item.get("url") or ""])
        add(item.get("url") or "", text[:80], "manual_menu_scan_results", text_score(text, 70))

    for item in source_mapping.get("manual_menu_items") or []:
        if not isinstance(item, dict):
            continue
        text = " ".join([item.get("text") or "", item.get("label") or "", item.get("title") or "", item.get("url") or ""])
        add(item.get("url") or "", text[:80], "manual_menu_items", text_score(text, 60))

    for page in (source_mapping.get("navigation_graph") or {}).get("pages") or []:
        if not isinstance(page, dict):
            continue
        text = " ".join([
            page.get("html_title") or "",
            page.get("page_purpose") or "",
            page.get("url") or "",
        ])
        add(page.get("url") or "", text[:80], "navigation_graph", text_score(text, 50))

    add(source_mapping.get("media_url") or "", source_mapping.get("media_name") or "media_url", "media_url", 20)
    add(source_mapping.get("login_url") or "", source_mapping.get("media_name") or "login_url", "login_url", 10)
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:10]


def _stored_source_snapshot_from_mapping(req: CrossMediaTaskCreateRequest, source_mapping: dict, mapping_id: str, candidates: list[dict]) -> dict:
    best = candidates[0] if candidates else {}
    url = best.get("url") or source_mapping.get("media_url") or source_mapping.get("login_url") or ""
    title = best.get("label") or source_mapping.get("media_name") or ""
    lines = []
    structured_fields = []
    for page in (source_mapping.get("navigation_graph") or {}).get("pages") or []:
        if not isinstance(page, dict):
            continue
        if url and page.get("url") != url:
            continue
        for key in ("html_title", "page_purpose"):
            if page.get(key):
                lines.append(str(page.get(key)))
        for group in ("links", "buttons", "inputs", "textareas", "selects"):
            for el in page.get(group) or []:
                if not isinstance(el, dict):
                    continue
                label = " ".join([
                    str(el.get("text") or ""),
                    str(el.get("label") or ""),
                    str(el.get("placeholder") or ""),
                    str(el.get("name") or ""),
                    str(el.get("value") or ""),
                ]).strip()
                if label:
                    lines.append(label)
        break
    try:
        db = get_db()
        schema_generation = ((source_mapping.get("schema_first") or {}).get("schema_generation") or (source_mapping.get("media_schema") or {}).get("schema_generation") or "")
        field_docs = db.collection("media_mappings").document(mapping_id).collection("schema_fields").limit(180).stream()
        for fd in field_docs:
            row = fd.to_dict() or {}
            if schema_generation and row.get("schema_generation") != schema_generation:
                continue
            structured_fields.append({
                "canonical": row.get("canonical", ""),
                "entity_type": row.get("entity_type", ""),
                "label": row.get("label", ""),
                "type": row.get("type", ""),
                "aliases": row.get("aliases", [])[:8],
                "target_count": row.get("target_count", 0),
            })
            if row.get("label") or row.get("canonical"):
                lines.append(f"{row.get('canonical','')} {row.get('label','')}")
    except Exception as _schema_snap_err:
        print(f"[CROSS_MEDIA_SCHEMA_SNAPSHOT_ERROR] mapping_id={mapping_id} {_schema_snap_err}", flush=True)
    body_text = " ".join(" ".join(lines).split())[:8000]
    return {
        "ok": bool(url or body_text or title),
        "source_mode": "source_mapping",
        "source_mapping_id": mapping_id,
        "media_name": source_mapping.get("media_name", ""),
        "source_url": url,
        "title": title,
        "description": "",
        "body_text": body_text,
        "image_urls": [],
        "content_type": "stored/navigation_graph",
        "candidates": candidates[:5],
        "structured_fields": structured_fields[:180],
        "media_schema_summary": source_mapping.get("schema_first") or {},
        "message": "既存の媒体マッピング保存データから中間データを作成しました。",
    }


def _extract_source_mapping_snapshot(req: CrossMediaTaskCreateRequest, source_mapping: dict, mapping_id: str) -> dict:
    candidates = _cross_media_source_candidates(req, source_mapping)
    best = candidates[0] if candidates else {}
    target_url = best.get("url") or ""
    if target_url:
        try:
            from api.core.browser_executor import fetch_content_snapshot_for_url
            snap = fetch_content_snapshot_for_url({**source_mapping, "mapping_id": mapping_id, "id": mapping_id}, target_url)
            if snap.get("ok"):
                snap.update({
                    "source_mode": "source_mapping",
                    "source_mapping_id": mapping_id,
                    "media_name": source_mapping.get("media_name", ""),
                    "candidates": candidates[:5],
                    "message": "認証済み媒体マッピングからページ本文を抽出しました。",
                })
                return snap
            stored = _stored_source_snapshot_from_mapping(req, source_mapping, mapping_id, candidates)
            stored["live_extract_status"] = snap.get("status", "")
            stored["live_extract_message"] = snap.get("message", snap.get("error", ""))
            return stored
        except Exception as e:
            stored = _stored_source_snapshot_from_mapping(req, source_mapping, mapping_id, candidates)
            stored["live_extract_status"] = "FAILED"
            stored["live_extract_message"] = f"{type(e).__name__}:{e}"
            return stored
    return {
        "ok": False,
        "source_mode": "source_mapping",
        "source_mapping_id": mapping_id,
        "media_name": source_mapping.get("media_name", ""),
        "error": "source_url_not_resolved",
        "message": "取得元媒体で読むURLを特定できませんでした。HTMLメニュー取り込み、source_url指定、またはsource_payload指定が必要です。",
    }


def _cross_media_payload_has_source_data(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("title", "body", "text", "value", "name", "source_url", "image_urls", "file_path", "schedule_value", "price_value"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return True
        if isinstance(val, (list, tuple, dict)) and val:
            return True
    return False


def _cross_media_source_items(base_payload: dict, source_snapshot: dict, max_items: int) -> list[dict]:
    limit = max(1, min(int(max_items or 1), 50))
    raw_items = []
    if isinstance(base_payload.get("items"), list):
        raw_items = base_payload.get("items") or []
    elif isinstance(source_snapshot.get("items"), list):
        raw_items = source_snapshot.get("items") or []
    items = []
    for idx, item in enumerate(raw_items[:limit]):
        if not isinstance(item, dict):
            continue
        payload = dict(base_payload)
        payload.pop("items", None)
        payload.update(item)
        payload["cross_media_item_index"] = idx
        items.append(payload)
    if items:
        return items
    single = dict(base_payload)
    single.pop("items", None)
    return [single]


def _cross_media_payload_from_snapshot(req: CrossMediaTaskCreateRequest, snapshot: dict) -> dict:
    payload = dict(req.source_payload or {})
    if snapshot.get("ok"):
        payload.setdefault("title", snapshot.get("title") or "")
        payload.setdefault("body", snapshot.get("description") or snapshot.get("body_text") or "")
        payload.setdefault("text", snapshot.get("body_text") or snapshot.get("description") or "")
        payload.setdefault("value", snapshot.get("body_text") or snapshot.get("description") or "")
        payload.setdefault("name", snapshot.get("title") or "")
        payload.setdefault("source_url", snapshot.get("source_url") or req.source_url or "")
        if snapshot.get("image_urls"):
            payload.setdefault("image_urls", snapshot.get("image_urls"))
        if snapshot.get("structured_fields"):
            payload.setdefault("structured_fields", snapshot.get("structured_fields"))
        if snapshot.get("media_schema_summary"):
            payload.setdefault("source_media_schema", snapshot.get("media_schema_summary"))
    if not _cross_media_payload_has_source_data(payload) and req.source_mode == "manual_payload":
        fallback_text = (req.instruction or req.query or "").strip()
        if fallback_text:
            payload.setdefault("body", fallback_text)
            payload.setdefault("text", fallback_text)
            payload.setdefault("value", fallback_text)
    if req.query:
        payload.setdefault("cross_media_query", req.query)
    if req.instruction:
        payload.setdefault("cross_media_instruction", req.instruction)
    return payload


def _cross_media_target_mappings(db, tenant_id: str, mapping_ids: list[str], user: dict) -> list[tuple[str, dict]]:
    targets = []
    ids = [x for x in (mapping_ids or []) if x]
    if ids:
        for mid in ids:
            snap = db.collection("media_mappings").document(mid).get()
            if snap.exists:
                mm = snap.to_dict() or {}
                mm["mapping_id"] = mid
                if mm.get("tenant_id") == tenant_id or user.get("role", "").lower() == "admin":
                    targets.append((mid, mm))
    else:
        for d in db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream():
            mm = d.to_dict() or {}
            mm["mapping_id"] = d.id
            targets.append((d.id, mm))
    return targets


@router.post("/task/batch/create")
def create_task_batch(req: BatchTaskCreateRequest, user: dict = Depends(verify_token)):
    """P35: 複数媒体へ同一operationの子taskを安全に展開する。実行はしない。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if req.agent_type not in AGENT_TYPES:
        raise HTTPException(status_code=400, detail="無効なagent_typeです")
    if req.operation_type not in OPERATION_TYPES:
        raise HTTPException(status_code=400, detail="無効なoperation_typeです")

    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    _enforce_agent_permissions(ctx, req.agent_type, req.operation_type)

    db = get_db()
    from api.core.browser_executor import GENERIC_OPERATION_CONFIG
    cap_key = GENERIC_OPERATION_CONFIG.get(req.operation_type, {}).get("capability_key", "")
    now = datetime.datetime.utcnow()
    batch_id = str(uuid.uuid4())
    workflow_id = str(uuid.uuid4())
    base_payload = dict(req.payload or {})

    targets = _load_batch_mapping_targets(db, tenant_id, req.media_mapping_ids or [], user)
    created_tasks = []
    skipped = []

    for idx, (mapping_id, mm) in enumerate(targets):
        media_name = mm.get("media_name", "")
        if mm.get("_missing"):
            skipped.append({"mapping_id": mapping_id, "media_name": "", "reason": "mapping_not_found"})
            continue
        if mm.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
            skipped.append({"mapping_id": mapping_id, "media_name": media_name, "reason": "forbidden_mapping"})
            continue

        mm, _ = _ensure_capability_view_for_mapping(db, mapping_id, mm)

        # ── スカウト系 op: credential + site_purpose で READY 判定（HTML steps不要）──
        _SCOUT_BATCH_SP: dict[str, set] = {
            "offer_send":         {"scout"},
            "recruit_inbox_scan": {"scout", "reply", "monitor"},
            "recruit_reply":      {"scout", "reply"},
        }
        _scout_payload_add: dict = {}
        if req.operation_type in _SCOUT_BATCH_SP:
            _bc_sp = (mm.get("business_conditions") or {}).get("site_purpose", "")
            if _bc_sp not in _SCOUT_BATCH_SP[req.operation_type]:
                skipped.append({"mapping_id": mapping_id, "media_name": media_name, "reason": "site_purpose_mismatch", "site_purpose": _bc_sp})
                continue
            if not mm.get("credential_secret_name"):
                skipped.append({"mapping_id": mapping_id, "media_name": media_name, "reason": "credential_missing"})
                continue
            _scout_payload_probe = dict(base_payload)
            _scout_op_map_b = (mm.get("operation_mappings") or {}).get(req.operation_type, {})
            if req.operation_type == "offer_send":
                if not _operation_mapping_is_production_ready(_scout_op_map_b):
                    skipped.append({
                        "mapping_id": mapping_id,
                        "media_name": media_name,
                        "reason": "mapping_not_production_ready",
                        "operation_type": req.operation_type,
                    })
                    continue
                _offer_url_b = (
                    _scout_op_map_b.get("target_url")
                    or _scout_payload_probe.get("search_url")
                    or ""
                )
                if not _offer_url_b:
                    skipped.append({
                        "mapping_id": mapping_id,
                        "media_name": media_name,
                        "reason": "recruit_url_missing",
                        "missing": ["offer_send.search_url"],
                    })
                    continue
                _scout_payload_add["search_url"] = _offer_url_b
            elif req.operation_type == "recruit_inbox_scan":
                if not _operation_mapping_is_production_ready(_scout_op_map_b):
                    skipped.append({
                        "mapping_id": mapping_id,
                        "media_name": media_name,
                        "reason": "mapping_not_production_ready",
                        "operation_type": req.operation_type,
                    })
                    continue
                _inbox_url_b = (
                    _scout_op_map_b.get("target_url")
                    or _scout_payload_probe.get("inbox_url")
                    or ""
                )
                if not _inbox_url_b:
                    skipped.append({
                        "mapping_id": mapping_id,
                        "media_name": media_name,
                        "reason": "recruit_url_missing",
                        "missing": ["recruit_inbox_scan.inbox_url"],
                    })
                    continue
                _scout_payload_add["inbox_url"] = _inbox_url_b
            elif req.operation_type == "recruit_reply" and not _scout_payload_probe.get("reply_url"):
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "reply_url_missing",
                    "missing": ["reply_url"],
                })
                continue
            steps = []  # スカウト系はステップ不要
        else:
            cap_op = _operation_from_capability_view(mm, req.operation_type)
            op_map = (mm.get("operation_mappings") or {}).get(req.operation_type) or {}
            op_status = cap_op.get("status") if cap_op else op_map.get("status", "")
            if not _operation_mapping_is_production_ready(op_map):
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "mapping_not_production_ready",
                    "operation_status": op_status or op_map.get("status") or "UNDISCOVERED",
                    "missing": op_map.get("missing", []),
                })
                continue
            if cap_op and not (cap_op.get("status") == "READY" and cap_op.get("taskable")):
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "operation_not_structurally_ready",
                    "operation_status": op_status or "UNDISCOVERED",
                    "missing": cap_op.get("missing", []),
                })
                continue
            if not cap_op and op_status != "READY":
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "operation_not_ready",
                    "operation_status": op_status or "UNDISCOVERED",
                    "missing": op_map.get("missing", []),
                })
                continue

            caps = mm.get("capabilities") or {}
            if cap_key and not caps.get(cap_key, False) and not _capability_op_is_taskable(mm, req.operation_type):
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "capability_missing",
                    "missing_capability": cap_key,
                })
                continue

            steps = _batch_steps_for_mapping(db, mapping_id, mm, req.operation_type)
            if not steps:
                skipped.append({
                    "mapping_id": mapping_id,
                    "media_name": media_name,
                    "reason": "operation_steps_missing",
                    "operation_status": op_status,
                })
                continue

        payload = dict(base_payload)
        payload.update(_scout_payload_add)
        payload["media_mapping_id"] = mapping_id
        payload["media_name"] = media_name
        preview = _build_preview(
            req.agent_type,
            req.operation_type,
            _normalize_industry(req.industry or "generic"),
            payload,
            operation_steps=steps,
            before_values={},
        )
        task_id = str(uuid.uuid4())
        workflow_session_id, workflow_risk = _create_task_workflow_session(
            db=db,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            operation_type=req.operation_type,
            operation_steps=steps,
            media_mapping=mm,
            media_mapping_id=mapping_id,
            media_name=media_name,
            goal_context="batch",
        )
        task = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "user_uid": user.get("uid", ""),
            "agent_type": req.agent_type,
            "operation_type": req.operation_type,
            "industry": _normalize_industry(req.industry or "generic"),
            "entity_type": req.entity_type or "",
            "op_id": "",
            "op_snapshot": {},
            "status": "PENDING",
            "payload": payload,
            "preview": preview,
            "operation_steps": steps,
            "approved_by": None,
            "approved_at": None,
            "scheduled_at": req.scheduled_at,
            "result": None,
            "created_at": now,
            "media_mapping_id": mapping_id,
            "workflow_session_id": workflow_session_id,
            "risk_level": workflow_risk.get("risk_level", ""),
            "risk_score": workflow_risk.get("risk_score", 0.0),
            "risk_factors": workflow_risk.get("risk_factors", []),
            "require_human_approval": workflow_risk.get("require_human_approval", False),
            "workflow_id": workflow_id,
            "chain_id": batch_id,
            "parent_task_id": "",
            "depends_on": [],
            "previous_operation": "",
            "next_operation_candidates": [],
            "batch_id": batch_id,
            "batch_index": idx,
        }
        db.collection("agent_tasks").document(task_id).set(task)
        created_tasks.append({
            "task_id": task_id,
            "mapping_id": mapping_id,
            "media_name": media_name,
            "status": "PENDING",
            "step_count": len(steps),
            "risk_level": workflow_risk.get("risk_level", ""),
            "require_human_approval": workflow_risk.get("require_human_approval", False),
        })

    batch_status = "PENDING" if created_tasks else "BLOCKED"
    batch_doc = {
        "batch_id": batch_id,
        "tenant_id": tenant_id,
        "user_uid": user.get("uid", ""),
        "agent_type": req.agent_type,
        "operation_type": req.operation_type,
        "industry": _normalize_industry(req.industry or "generic"),
        "payload": base_payload,
        "workflow_id": workflow_id,
        "status": batch_status,
        "task_ids": [t["task_id"] for t in created_tasks],
        "created_tasks": created_tasks,
        "skipped_targets": skipped,
        "counts": {
            "targets": len(targets),
            "created": len(created_tasks),
            "skipped": len(skipped),
            "approved": 0,
            "done": 0,
            "failed": 0,
        },
        "created_at": now,
        "approved_at": None,
        "executed_at": None,
        "updated_at": now,
    }
    db.collection("agent_task_batches").document(batch_id).set(batch_doc)
    return _iso_top_level(batch_doc)


class CrossMediaSourceEntitiesRequest(BaseModel):
    source_mapping_id: str
    target_operation_type: str = "entity_update"
    list_url: Optional[str] = None  # 明示指定があればそのURL、無ければ操作から解決


class CrossMediaPreviewRequest(BaseModel):
    source_mapping_id: Optional[str] = None
    source_url: Optional[str] = None
    source_payload: dict = Field(default_factory=dict)
    target_mapping_ids: list[str] = Field(default_factory=list)
    target_operation_type: str = "entity_update"
    instruction: str = ""
    # 対象指定（誰を）: 取得元から選んだエンティティの詳細URL
    source_entity_url: Optional[str] = None


@router.get("/cross_media/snapshot")
def get_cross_media_snapshot(
    source_mapping_id: str,
    dest_mapping_id: str,
    entity_url: str,
    user: dict = Depends(verify_token),
):
    """前回同期時のスナップショットを取得。差分更新の基準データとして使用。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    import hashlib
    db = get_db()
    snap_id = hashlib.sha256(
        f"{source_mapping_id}_{dest_mapping_id}_{entity_url}".encode()
    ).hexdigest()[:32]
    doc = db.collection("cross_media_snapshots").document(snap_id).get()
    if not doc.exists:
        return {"ok": False, "snapshot": None, "message": "スナップショットなし（初回同期）"}
    snap = doc.to_dict() or {}
    tenant_id = user.get("tenant_id") or user.get("email") or ""
    if snap.get("tenant_id") not in ("", tenant_id) and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="アクセス権がありません")
    # Firestore Timestamp → ISO文字列
    synced_at = snap.get("synced_at")
    try:
        synced_at = synced_at.isoformat() if hasattr(synced_at, "isoformat") else str(synced_at)
    except Exception:
        synced_at = None
    return {
        "ok": True,
        "snapshot": {
            "synced_at":    synced_at,
            "source_data":  snap.get("source_data") or {},
            "mapped_fields": snap.get("mapped_fields") or {},
            "entity_label": snap.get("entity_label") or "",
            "industry":     snap.get("industry") or "generic",
        },
    }


@router.post("/cross_media/source_entities")
def cross_media_source_entities(req: CrossMediaSourceEntitiesRequest, user: dict = Depends(verify_token)):
    """取得元媒体のエンティティ一覧（例: キャスト一覧）を取得して返す。
    ID/PASS/Cookieはレスポンスに含めない。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("media_mappings").document(req.source_mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="取得元マッピングが見つかりません")
    mm = doc.to_dict() or {}
    mm["mapping_id"] = req.source_mapping_id
    _assert_tenant_access(mm, user, "他テナントのマッピングは操作できません")

    # 一覧ページURLを解決（明示指定 > ログイン後リダイレクト先 > admin_url > media_url）
    # NOTE: _resolve_cross_media_target_url は「操作フォームURL」を返すため使わない。
    # 取得元のエンティティ一覧はログイン後の最初のページ（一覧/ダッシュボード）が最も確実。
    list_url = req.list_url or ""
    if not list_url or _is_blocked_operation_url(list_url):
        _login_url = mm.get("login_url") or ""
        for _key in ("login_success_redirect_url", "admin_url", "media_url"):
            _cand = mm.get(_key) or ""
            if _cand and _cand != _login_url and not _is_blocked_operation_url(_cand):
                list_url = _cand
                break
        if not list_url:
            list_url = _login_url

    industry = _normalize_industry(mm.get("industry") or "generic")
    entity_label = (INDUSTRY_TEMPLATES.get(industry) or INDUSTRY_TEMPLATES["other"]).get("entity_name", "対象")

    from api.core.browser_executor import extract_entity_list
    result = extract_entity_list(mm, list_url, entity_label=entity_label)
    _entities = result.get("entities", [])
    _visible = [e for e in _entities if not e.get("hidden")]
    _hidden  = [e for e in _entities if e.get("hidden")]
    return {
        "ok": result.get("ok", False),
        "status": result.get("status", ""),
        "message": result.get("message", ""),
        "entity_label": entity_label,
        "list_url": list_url,
        "entities": _entities,
        "count": len(_entities),
        "visible_count": len(_visible),
        "hidden_count": len(_hidden),
    }

@router.post("/cross_media/preview")
def cross_media_preview(req: CrossMediaPreviewRequest, user: dict = Depends(verify_token)):
    """
    タスク作成前プレビュー: 送信先フォームのスクリーンショット + Gemini AI マッピング提案を返す。
    ID/PASS/Cookie はレスポンスに含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    results = []

    # 取得元データを構築
    source_payload = dict(req.source_payload or {})
    if req.source_mapping_id and not source_payload:
        try:
            src_doc = db.collection("media_mappings").document(req.source_mapping_id).get()
            if src_doc.exists:
                src_mapping = src_doc.to_dict() or {}
                src_mapping["mapping_id"] = req.source_mapping_id
                _assert_tenant_access(src_mapping, user, "他テナントのマッピングは操作できません")
                # 対象指定（誰を）: source_entity_url があればそのキャストの詳細ページを取得元にする
                _src_url = req.source_entity_url or req.source_url or ""
                fake_req = CrossMediaTaskCreateRequest(
                    source_mapping_id=req.source_mapping_id,
                    source_url=_src_url,
                    target_operation_type=req.target_operation_type,
                    instruction=req.instruction,
                )
                snap = _extract_source_mapping_snapshot(fake_req, src_mapping, req.source_mapping_id)
                source_payload = _cross_media_payload_from_snapshot(fake_req, snap)
        except Exception as _se:
            print(f"[CROSS_MEDIA_PREVIEW] source extraction error: {_se}", flush=True)

    import json as _preview_json
    _SKIP_KEYS = {"media_mapping_id", "media_name", "cross_media_task_id",
                  "source_mode", "source_url", "source_mapping_id",
                  "cross_media_target_index", "cross_media_item_index"}
    fill_data = {k: v for k, v in source_payload.items()
                 if isinstance(v, str) and v.strip() and k not in _SKIP_KEYS}
    # structured_fieldsはdictなので通常フィルタで除外される → JSON文字列としてAIに渡す
    if isinstance(source_payload.get("structured_fields"), (dict, list)):
        fill_data["structured_fields"] = _preview_json.dumps(
            source_payload["structured_fields"], ensure_ascii=False
        )

    for mid in req.target_mapping_ids:
        try:
            doc = db.collection("media_mappings").document(mid).get()
            if not doc.exists:
                continue
            mm = doc.to_dict() or {}
            mm["mapping_id"] = mid
            _assert_tenant_access(mm, user, "他テナントのマッピングは操作できません")

            # フォームスナップショット（スクリーンショット + フィールド一覧）
            # 実行時と同一ロジックでtarget_urlを解決（FAQ/外部フォーム等は除外）
            from api.core.browser_executor import take_site_preview
            _login_url = mm.get("login_url") or ""
            _post_login_url = mm.get("login_success_redirect_url") or ""

            _resolved_meta = _resolve_cross_media_target(db, mid, mm, req.target_operation_type)
            _resolved = _resolved_meta.get("url", "")
            _url_source = _resolved_meta.get("source", "NONE")
            _url_verified = bool(_resolved_meta.get("verified"))
            _resolve_reason = str(_resolved_meta.get("reason") or "")
            # URLが見つからない場合は明確なエラーを返す（無関係なページを表示しない）
            if not _resolved:
                results.append({
                    "mapping_id": mid,
                    "media_name": mm.get("media_name", mid),
                    "error": (
                        f"【{mm.get('media_name', mid)}】{req.target_operation_type} のフォームURLが見つかりません。"
                        + (f" { _resolve_reason }" if _resolve_reason else "")
                        + " 媒体基盤のAI整備でこの操作のURLを保存してください。"
                    ),
                    "field_count": 0, "mapped_count": 0,
                    "mapping_detail": [], "source_data_keys": [],
                    "url_source": _url_source, "url_verified": False,
                })
                continue
            # 解決URLがログインURLそのものなら、ログイン後ページのまま撮影（再ログイン回避）
            if _resolved != _login_url:
                _preview_target = _resolved
            else:
                _preview_target = _post_login_url if (_post_login_url and _post_login_url != _login_url) else ""
            snap_result = take_site_preview(mm, target_url=_preview_target, extract_all_form_elements=True)
            screenshot_b64 = snap_result.get("screenshot_b64", "")
            raw_form_elements = [f for f in (snap_result.get("form_elements", []) or []) if isinstance(f, dict)]
            known_mapped_fields = _mapping_fields_for_operation(mm, req.target_operation_type)
            _preview_login_reason = _manual_page_login_reason(
                mm.get("media_name", mid),
                snap_result.get("current_url") or _resolved,
                "",
                raw_form_elements,
            )
            if _preview_login_reason:
                results.append({
                    "mapping_id": mid,
                    "media_name": mm.get("media_name", mid),
                    "screenshot_b64": screenshot_b64,
                    "current_url": snap_result.get("current_url", ""),
                    "resolved_url": _resolved,
                    "url_source": _url_source,
                    "url_verified": False,
                    "error": f"【{mm.get('media_name', mid)}】ログイン画面が表示されています。{req.target_operation_type} の投稿・更新フォームとしては使えません。媒体基盤のAI整備でログイン後の正しいフォームURL/HTMLを保存してください。",
                    "field_count": 0,
                    "mapped_count": 0,
                    "mapping_detail": [],
                    "mapped_field_count": len(known_mapped_fields),
                    "known_mapped_fields": [
                        {
                            "label": f.get("label", ""),
                            "name": f.get("name", ""),
                            "id": f.get("id", ""),
                            "canonical": f.get("canonical", ""),
                            "selector": f.get("selector", ""),
                            "type": f.get("type", "text"),
                            "source": f.get("source", "mapping"),
                        }
                        for f in known_mapped_fields[:200]
                    ],
                    "source_data_keys": list(fill_data.keys()),
                })
                continue
            form_elements = [f for f in raw_form_elements if _is_actionable_mapping_field(f)]

            # Gemini AI マッピング（ドライラン）
            proposed_mapping = {}
            if fill_data and form_elements:
                import json as _json
                from api.core.llm_client import call_llm_json
                _fields_for_ai = [
                    {"index": i, "label": f.get("label",""), "name": f.get("name",""),
                     "placeholder": f.get("placeholder",""), "type": f.get("type","text")}
                    for i, f in enumerate(form_elements[:200])
                ]
                _known_fields_for_ai = [
                    {
                        "label": f.get("label", ""),
                        "name": f.get("name", ""),
                        "id": f.get("id", ""),
                        "canonical": f.get("canonical", ""),
                        "selector": f.get("selector", ""),
                        "type": f.get("type", "text"),
                        "source": f.get("source", "mapping"),
                    }
                    for f in known_mapped_fields[:200]
                ]
                # structured_fieldsがある場合は優先説明を追加
                _has_structured = "structured_fields" in fill_data
                _src_hint = (
                    "※ structured_fieldsキーには取得元ページのフォーム/テーブルから抽出した実データが含まれます。これを最優先で参照してください。"
                    if _has_structured else
                    "※ body/text/valueキーには取得元ページの本文テキストが含まれます。その中から該当フィールドの値を読み取ってください。"
                )
                _prompt = f"""あなたはWebフォーム自動入力AIです。
操作タイプ: {req.target_operation_type}
追加指示: {req.instruction or 'なし'}
{_src_hint}

【取得元データ】
{_json.dumps(fill_data, ensure_ascii=False, indent=2)}

【送信先フォームフィールド一覧（index, label, name, placeholder, type）】
{_json.dumps(_fields_for_ai, ensure_ascii=False, indent=2)}

【AI整備済みマッピング構造（operation_mappings由来）】
{_json.dumps(_known_fields_for_ai, ensure_ascii=False, indent=2)}

フィールドのlabel・name・placeholderを手がかりに意味的にマッピングし、
入力すべき値を {{"index": "値"}} 形式のJSONのみで返してください。
マッピングできないフィールドはスキップ（含めない）。"""
                try:
                    proposed_mapping = call_llm_json(
                        prompt=_prompt,
                        system_prompt="JSONのみ出力。```json等のMarkdownブロック禁止。",
                        ai_tier="core",
                        max_tokens=2048,
                    )
                except Exception as _ae:
                    print(f"[CROSS_MEDIA_PREVIEW] AI mapping error mid={mid}: {_ae}", flush=True)

            # レスポンス用マッピング詳細: [{field_label, field_name, value}]
            mapping_detail = []
            for idx_str, val in proposed_mapping.items():
                try:
                    idx = int(idx_str)
                    if idx < len(form_elements):
                        fe = form_elements[idx]
                        mapping_detail.append({
                            "index": idx,
                            "label": fe.get("label") or fe.get("name") or fe.get("placeholder") or f"field_{idx}",
                            "name": fe.get("name", ""),
                            "value": str(val),
                        })
                except Exception:
                    pass

            results.append({
                "mapping_id": mid,
                "media_name": mm.get("media_name", mid),
                "screenshot_b64": screenshot_b64,
                "current_url": snap_result.get("current_url", ""),
                "resolved_url": _resolved,
                "url_source": _url_source,
                "url_verified": _url_verified,
                "mapping_detail": mapping_detail,
                "field_count": len(form_elements),
                "mapped_field_count": len(known_mapped_fields),
                "known_mapped_fields": [
                    {
                        "label": f.get("label", ""),
                        "name": f.get("name", ""),
                        "id": f.get("id", ""),
                        "canonical": f.get("canonical", ""),
                        "selector": f.get("selector", ""),
                        "type": f.get("type", "text"),
                        "source": f.get("source", "mapping"),
                    }
                    for f in known_mapped_fields[:200]
                ],
                "mapped_count": len(mapping_detail),
                "source_data_keys": list(fill_data.keys()),
            })
        except Exception as _de:
            print(f"[CROSS_MEDIA_PREVIEW] dest error mid={mid}: {_de}", flush=True)
            results.append({"mapping_id": mid, "error": str(_de)})

    return {"results": results, "source_data": fill_data}


@router.post("/cross_media/task/create")
def create_cross_media_task(req: CrossMediaTaskCreateRequest, user: dict = Depends(verify_token)):
    """媒体A/公開URL/手入力データを中間データ化し、複数媒体Bへ子タスクを作る第2段入口。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if req.target_operation_type not in OPERATION_TYPES:
        raise HTTPException(status_code=400, detail="無効なtarget_operation_typeです")
    if req.source_mode not in ("manual_payload", "public_url", "source_mapping"):
        raise HTTPException(status_code=400, detail="source_modeが不正です")
    if req.source_mode in ("public_url", "source_mapping") and not req.source_access_confirmed:
        raise HTTPException(status_code=400, detail="取得元の利用権限・規約確認が必要です")
    if req.source_mode == "public_url" and not req.source_url:
        raise HTTPException(status_code=400, detail="source_urlが必要です")
    if req.source_mode == "source_mapping" and not req.source_mapping_id:
        raise HTTPException(status_code=400, detail="source_mapping_idが必要です")

    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    _enforce_agent_permissions(ctx, "hp_update", req.target_operation_type)
    db = get_db()
    now = datetime.datetime.utcnow()
    cross_task_id = str(uuid.uuid4())
    workflow_id = str(uuid.uuid4())

    source_snapshot = {"ok": True, "source_mode": req.source_mode}
    source_status = "READY"
    if req.source_mode == "public_url":
        source_snapshot = _extract_public_url_snapshot(req.source_url or "")
        source_status = "READY" if source_snapshot.get("ok") else "NEEDS_REVIEW"
    elif req.source_mode == "source_mapping":
        snap = db.collection("media_mappings").document(req.source_mapping_id or "").get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail="取得元media_mappingが見つかりません")
        source_mapping = snap.to_dict() or {}
        if source_mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
            raise HTTPException(status_code=403, detail="取得元mappingへのアクセス権がありません")
        source_snapshot = _extract_source_mapping_snapshot(req, source_mapping, req.source_mapping_id or "")
        if source_snapshot.get("ok") or req.source_payload:
            source_status = "READY"
        else:
            source_status = "NEEDS_EXTRACTION"

    base_payload = _cross_media_payload_from_snapshot(req, source_snapshot)
    source_items = _cross_media_source_items(base_payload, source_snapshot, req.max_items)
    has_source_data = any(_cross_media_payload_has_source_data(item) for item in source_items)
    targets = _cross_media_target_mappings(db, tenant_id, req.target_mapping_ids, user)
    created_tasks = []
    skipped_targets = []
    task_index = 0

    if source_status == "READY" and has_source_data:
        for target_idx, (mapping_id, mm) in enumerate(targets):
            mm, _ = _ensure_capability_view_for_mapping(db, mapping_id, mm)
            cap_op = _operation_from_capability_view(mm, req.target_operation_type)
            op_map = ((mm.get("operation_mappings") or {}).get(req.target_operation_type) or {})

            # クロスメディアは常にAI自動マッピング(cross_media_ai_fill)で実行する。
            # 既存のoperation_stepsはdom_selectors依存で複雑な日本語フォームに対応できないため、
            # ここでは「正しいtarget_urlの解決」にのみ既存データを使い、実行ステップはAIフィルに統一する。
            cm_target_meta = _resolve_cross_media_target(db, mapping_id, mm, req.target_operation_type)
            cm_target = cm_target_meta.get("url", "")
            if not cm_target:
                skipped_targets.append({
                    "mapping_id": mapping_id,
                    "media_name": mm.get("media_name", ""),
                    "reason": cm_target_meta.get("reason") or "target_url_not_resolved",
                    "operation_type": req.target_operation_type,
                })
                continue
            steps = [{
                "step_id": "cross_media_ai_fill_1",
                "step_type": "cross_media_ai_fill",
                "required": True,
                "description": "AI が出力先フォームを自動検出・入力します（クロスメディア自動マッピング）",
                "target_url": cm_target,
                "operation_type": req.target_operation_type,
            }]
            for item_idx, source_item in enumerate(source_items):
                payload = dict(source_item)
                payload["media_mapping_id"] = mapping_id
                payload["media_name"] = mm.get("media_name", "")
                payload["cross_media_task_id"] = cross_task_id
                payload["source_mode"] = req.source_mode
                payload["source_url"] = payload.get("source_url") or req.source_url or source_snapshot.get("source_url") or ""
                payload["source_mapping_id"] = req.source_mapping_id or ""
                payload["cross_media_target_index"] = target_idx
                payload["cross_media_item_index"] = payload.get("cross_media_item_index", item_idx)
                # 対象指定（誰を）・更新範囲（何を）を実行ペイロードに保存
                if req.source_entity_label:
                    payload["cross_media_entity_label"] = req.source_entity_label
                if req.source_entity_url:
                    payload["cross_media_entity_url"] = req.source_entity_url
                if req.selected_field_keys:
                    # 反映する宛先フィールドのラベル一覧（executorがこれだけ入力）
                    payload["cross_media_selected_fields"] = list(req.selected_field_keys)
                preview = _build_preview(
                    "hp_update",
                    req.target_operation_type,
                    _normalize_industry(req.industry or mm.get("industry") or "generic"),
                    payload,
                    operation_steps=steps,
                    before_values={},
                )
                task_id = str(uuid.uuid4())
                workflow_session_id, workflow_risk = _create_task_workflow_session(
                    db=db,
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                    operation_type=req.target_operation_type,
                    operation_steps=steps,
                    media_mapping=mm,
                    media_mapping_id=mapping_id,
                    media_name=mm.get("media_name", ""),
                    goal_context="cross_media",
                )
                task = {
                    "task_id": task_id,
                    "tenant_id": tenant_id,
                    "user_uid": user.get("uid", ""),
                    "agent_type": "hp_update",
                    "operation_type": req.target_operation_type,
                    "industry": _normalize_industry(req.industry or mm.get("industry") or "generic"),
                    "entity_type": "",
                    "op_id": "",
                    "op_snapshot": {},
                    "status": "PENDING",
                    "payload": payload,
                    "preview": preview,
                    "operation_steps": steps,
                    "approved_by": None,
                    "approved_at": None,
                    "scheduled_at": req.scheduled_at,
                    "result": None,
                    "created_at": now,
                    "media_mapping_id": mapping_id,
                    "workflow_session_id": workflow_session_id,
                    "risk_level": workflow_risk.get("risk_level", ""),
                    "risk_score": workflow_risk.get("risk_score", 0.0),
                    "risk_factors": workflow_risk.get("risk_factors", []),
                    "require_human_approval": workflow_risk.get("require_human_approval", False),
                    "workflow_id": workflow_id,
                    "chain_id": cross_task_id,
                    "parent_task_id": "",
                    "depends_on": [],
                    "previous_operation": "cross_media_extract",
                    "next_operation_candidates": [],
                    "cross_media_task_id": cross_task_id,
                    "cross_media_role": "target_upload",
                    "cross_media_source": {
                        "mode": req.source_mode,
                        "url": req.source_url or "",
                        "mapping_id": req.source_mapping_id or "",
                    },
                    "cross_media_item_index": item_idx,
                    "batch_index": task_index,
                }
                db.collection("agent_tasks").document(task_id).set(task)
                created_tasks.append({
                    "task_id": task_id,
                    "mapping_id": mapping_id,
                    "media_name": mm.get("media_name", ""),
                    "status": "PENDING",
                    "step_count": len(steps),
                    "item_index": item_idx,
                    "risk_level": workflow_risk.get("risk_level", ""),
                    "require_human_approval": workflow_risk.get("require_human_approval", False),
                })
                task_index += 1

    if created_tasks:
        status = "PENDING"
    elif source_status == "NEEDS_EXTRACTION":
        status = "NEEDS_EXTRACTION"
    elif not has_source_data:
        status = "NEEDS_PAYLOAD"
    else:
        status = "NEEDS_REVIEW"
    doc = {
        "cross_task_id": cross_task_id,
        "tenant_id": tenant_id,
        "user_uid": user.get("uid", ""),
        "workflow_id": workflow_id,
        "instruction": req.instruction,
        "industry": _normalize_industry(req.industry or "generic"),
        "source_mode": req.source_mode,
        "source_url": req.source_url or "",
        "source_mapping_id": req.source_mapping_id or "",
        "source_access_confirmed": req.source_access_confirmed,
        "source_status": source_status,
        "source_snapshot": source_snapshot,
        "target_operation_type": req.target_operation_type,
        "target_mapping_ids": req.target_mapping_ids,
        "payload": base_payload,
        "source_item_count": len(source_items) if has_source_data else 0,
        "has_source_data": has_source_data,
        "query": req.query,
        "max_items": max(1, min(int(req.max_items or 1), 50)),
        "status": status,
        "task_ids": [t["task_id"] for t in created_tasks],
        "created_tasks": created_tasks,
        "skipped_targets": skipped_targets,
        "counts": {
            "targets": len(targets),
            "created": len(created_tasks),
            "skipped": len(skipped_targets),
        },
        "created_at": now,
        "updated_at": now,
    }
    db.collection("cross_media_tasks").document(cross_task_id).set(doc)
    return _iso_top_level(doc)


@router.get("/cross_media/task/list")
def list_cross_media_tasks(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    rows = []
    for d in db.collection("cross_media_tasks").where("tenant_id", "==", tenant_id).limit(200).stream():
        item = d.to_dict() or {}
        item["cross_task_id"] = item.get("cross_task_id") or d.id
        rows.append(_iso_top_level(item))
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"tasks": rows, "count": len(rows)}


@router.delete("/cross_media/task/{cross_task_id}")
def delete_cross_media_task(cross_task_id: str, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("cross_media_tasks").document(cross_task_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    task = doc.to_dict() or {}
    if task.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このタスクへのアクセス権がありません")
    db.collection("cross_media_tasks").document(cross_task_id).delete()
    return {"deleted": True, "cross_task_id": cross_task_id}


def _monitoring_ai_analyze(page_text: str, industry: str = "nightlife") -> dict:
    """監視ページテキストからGeminiでトレンドフレーズ・人気タイプ・避けるべき表現を抽出する"""
    try:
        from api.core.llm_client import call_llm_json
        _prompt = (
            f"以下は{industry}業界のキャスト管理サイトから取得したページテキストです。\n"
            "このテキストから以下の情報を分析してJSONで返してください。\n\n"
            f"【テキスト】\n{page_text[:5000]}\n\n"
            "【抽出項目】\n"
            "1. trending_phrases: よく出てくるキャッチコピー・売り文句フレーズ（最大10件）\n"
            "2. popular_types: 人気のキャスタータイプ・雰囲気（例：天然系、癒し系、攻め系）（最大8件）\n"
            "3. avoid_phrases: 使い回しが多く差別化できない陳腐な表現（最大5件）\n"
            "4. ai_summary: 市場傾向の要約（100字以内）\n"
            "5. recommendations: プロフィール作成時の具体的な推奨事項（最大5件）\n\n"
            'JSONのみ返してください: {"trending_phrases":[],"popular_types":[],"avoid_phrases":[],"ai_summary":"","recommendations":[]}'
        )
        _r = call_llm_json(
            prompt=_prompt,
            system_prompt="市場分析AI。JSONのみ出力。Markdownや説明文は禁止。",
            ai_tier="core",
            max_tokens=1024,
        )
        return {
            "trending_phrases": _r.get("trending_phrases") or [],
            "popular_types":    _r.get("popular_types") or [],
            "avoid_phrases":    _r.get("avoid_phrases") or [],
            "ai_summary":       _r.get("ai_summary") or "",
            "recommendations":  _r.get("recommendations") or [],
        }
    except Exception as _e:
        print(f"[MONITORING_AI_ANALYZE] error: {_e}", flush=True)
        return {"trending_phrases": [], "popular_types": [], "avoid_phrases": [], "ai_summary": "", "recommendations": []}


@router.get("/monitoring/results")
def get_monitoring_results(mapping_id: str = "", limit: int = 20, user: dict = Depends(verify_token)):
    """市場監視結果の一覧を返す（最新順）"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    _ctx = _resolve_agent_user_context(user)
    tenant_id = _ctx["tenant_id"]
    db = get_db()
    docs = db.collection("monitoring_results").where("tenant_id", "==", tenant_id).stream()
    results = []
    for d in docs:
        row = d.to_dict() or {}
        row["id"] = d.id
        if row.get("executed_at") and hasattr(row["executed_at"], "isoformat"):
            row["executed_at"] = row["executed_at"].isoformat()
        if not mapping_id or row.get("mapping_id") == mapping_id:
            results.append(row)
    results.sort(key=lambda x: x.get("executed_at") or "", reverse=True)
    return {"results": results[:limit]}


class ProfileGenerateRequest(BaseModel):
    cast_name: str
    age: str = ""
    height: str = ""
    bust: str = ""
    cup: str = ""
    waist: str = ""
    hip: str = ""
    type_hint: str = ""          # 「清楚系」「元気系」など
    custom_instructions: str = ""
    industry: str = "nightlife"
    target_mapping_ids: list[str] = []
    source_html: str = ""        # HTMLを直貼りした場合はPlaywrightなしで解析
    source_html_mapping_id: str = ""  # source_html対応の送信先マッピングID
    source_html_target_url: str = ""  # source_html用フォームURL（submit先）


# ─────────────────────────────────────────────────────────────────────
# プロフィール生成: フォームHTML解析 → 全テキストフィールドをAI一括生成
# 戻り値の fill_fields は {CSSセレクタ: 値} 形式 — execute で直接使用可能
# ─────────────────────────────────────────────────────────────────────
@router.post("/cross_media/generate_profile_preview")
def generate_profile_preview(req: ProfileGenerateRequest, user: dict = Depends(verify_token)):
    """
    キャストプロフィールをAI生成してプレビューを返す。
    画像なし・テキストフィールドのみ。ID/PASSはレスポンスに含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    import json as _jj
    from api.core.llm_client import call_llm_json
    from api.core.browser_executor import take_site_preview

    industry = _normalize_industry(req.industry or "nightlife")
    tmpl = INDUSTRY_TEMPLATES.get(industry, INDUSTRY_TEMPLATES["other"])
    entity_name = tmpl.get("entity_name", "キャスト")

    _industry_ctx = {
        "nightlife": "キャバクラ・クラブ・ラウンジ等の夜職ホームページ掲載用プロフィール。魅力的・親しみやすく・集客力のある文体で。",
        "beauty":    "エステ・マッサージ・美容サロンのスタッフ紹介ページ。清潔感・専門性・温かみを重視。",
        "fitness":   "フィットネス・スポーツジムのインストラクター紹介。活発・健康的・信頼感のある文体で。",
    }.get(industry, "ホームページ掲載用スタッフプロフィール。魅力的かつ読みやすい文体で。")

    # 基本情報文字列（ユーザーが入力した値 — AIが生成せずそのまま使う）
    _basic_lines = [f"{entity_name}名: {req.cast_name}"]
    if req.age:    _basic_lines.append(f"年齢: {req.age}歳")
    if req.height: _basic_lines.append(f"身長: {req.height}cm")
    _size_parts = []
    if req.bust:   _size_parts.append(f"B{req.bust}" + (f"({req.cup}カップ)" if req.cup else ""))
    if req.waist:  _size_parts.append(f"W{req.waist}")
    if req.hip:    _size_parts.append(f"H{req.hip}")
    if _size_parts: _basic_lines.append("スリーサイズ: " + " / ".join(_size_parts))
    if req.type_hint: _basic_lines.append(f"タイプ・雰囲気ヒント: {req.type_hint}")
    if req.custom_instructions: _basic_lines.append(f"追加指示: {req.custom_instructions}")
    _basic_str = "\n".join(_basic_lines)

    # ── db + 市場監視コンテキスト + RAGナレッジを全パス共通で先取りする ──────────
    db = get_db()
    _tenant_id_pre = _resolve_agent_user_context(user)["tenant_id"]

    _market_context_pre = ""
    try:
        _mr_docs_pre = db.collection("monitoring_results").where("tenant_id", "==", _tenant_id_pre).stream()
        def _pre_ts_key(x: dict) -> str:
            v = x.get("executed_at")
            if v is None: return ""
            return v.isoformat() if hasattr(v, "isoformat") else str(v)
        _mr_list_pre = sorted([d.to_dict() for d in _mr_docs_pre], key=_pre_ts_key, reverse=True)
        if _mr_list_pre:
            _l = _mr_list_pre[0]
            _ts = _pre_ts_key(_l)[:10]
            _t  = ", ".join(_l.get("trending_phrases") or []) or "データなし"
            _pt = ", ".join(_l.get("popular_types") or []) or "データなし"
            _av = ", ".join(_l.get("avoid_phrases") or [])
            _rc = "; ".join((_l.get("recommendations") or [])[:3])
            _market_context_pre = (
                f"【市場監視データ（{_ts}時点）】\n"
                f"トレンドフレーズ: {_t}\n人気タイプ: {_pt}\n"
                + (f"避けるべき表現: {_av}\n" if _av else "")
                + (f"推奨事項: {_rc}\n" if _rc else "")
                + (f"市場サマリー: {_l.get('ai_summary', '')}" if _l.get("ai_summary") else "")
            ).strip()
    except Exception as _mc_pre_e:
        print(f"[GENERATE_PROFILE] 監視データ取得エラー: {_mc_pre_e}", flush=True)

    _rag_context_pre = ""
    try:
        from api.core.rag import rag_retrieve_chunks_with_vec, embed_text as _rag_embed_pre
        _rag_vec_pre = _rag_embed_pre(f"{req.industry} キャスト プロフィール {req.type_hint} キャッチコピー 売れる 予約")
        _rag_chunks_pre = rag_retrieve_chunks_with_vec(_tenant_id_pre, _rag_vec_pre, top_k=4, threshold=0.25)
        if _rag_chunks_pre:
            _rag_context_pre = "【専用ナレッジ（登録済み業界知識）】\n" + "\n".join(c.get("text", "") for c in _rag_chunks_pre)
    except Exception as _rag_pre_e:
        print(f"[GENERATE_PROFILE] RAGエラー: {_rag_pre_e}", flush=True)

    # ── HTMLを直貼りした場合: Playwrightなし → GeminiでHTMLを解析（私(Claude Code)が行う処理と同じ） ──
    if req.source_html and req.source_html_mapping_id:
        doc = db.collection("media_mappings").document(req.source_html_mapping_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="マッピングが見つかりません")
        mm_h = doc.to_dict() or {}
        mm_h["mapping_id"] = req.source_html_mapping_id
        _assert_tenant_access(mm_h, user, "他テナントのマッピングは操作できません")

        _html_parse_prompt = f"""以下のHTMLからフォームの入力フィールドを全て抽出してください。

HTML（最大15000文字）:
{req.source_html[:15000]}

抽出対象: input, textarea, select（hidden/password/file/submit/button は除外）
各フィールドについて以下の情報を返してください:
- selector: CSSセレクタ（idがあれば#id、なければinput[name='xxx']形式）
- label: ラベルテキスト（labelタグ or 直前のテキスト or placeholder）
- type: フィールドタイプ（text/textarea/select/number等）

JSON配列で返してください:
[{{"selector": "#txt_name", "label": "名前", "type": "text"}}, ...]
説明不要。JSONのみ。"""

        _parsed_elements = call_llm_json(
            prompt=_html_parse_prompt,
            system_prompt="JSON配列のみ出力。",
            ai_tier="core",
            max_tokens=2000,
        )
        _html_form_elements = _parsed_elements if isinstance(_parsed_elements, list) else []
        print(f"[GENERATE_PROFILE_HTML] cast={req.cast_name} html_fields={len(_html_form_elements)}", flush=True)

        # パススルー + AI生成（Playwrightパスと同一ロジック）
        _PT = [
            (["名前", "キャスト名", "name", "氏名"], req.cast_name),
            (["ふりがな", "フリガナ", "kana"], None),
            (["年齢", "age"], req.age),
            (["身長", "height"], req.height),
            (["バスト", "bust", "b:"], req.bust),
            (["ウエスト", "waist", "w:"], req.waist),
            (["ヒップ", "hip", "h:"], req.hip),
            (["カップ", "cup"], req.cup),
        ]
        _SKIP_H = {"file", "hidden", "submit", "button", "image", "reset", "password"}
        fill_h: dict[str, str] = {}
        ai_q_h: list[dict] = []
        for f in _html_form_elements:
            sel = f.get("selector", "")
            if not sel:
                continue
            if (f.get("type", "") or "").lower() in _SKIP_H:
                continue
            label_l = (f.get("label") or "").lower()
            matched = False
            for patterns, val in _PT:
                if val and any(p.lower() in label_l for p in patterns):
                    fill_h[sel] = val
                    matched = True
                    break
            if not matched:
                ai_q_h.append({"idx": len(ai_q_h), "selector": sel, "label": f.get("label", ""), "type": f.get("type", "text")})

        if ai_q_h:
            _h_ctx_parts = [p for p in [_market_context_pre, _rag_context_pre] if p]
            _h_ctx = "\n\n".join(_h_ctx_parts) + "\n\n" if _h_ctx_parts else ""
            _ai_r = call_llm_json(
                prompt=f"""{_h_ctx}あなたは{_industry_ctx}のトッププロデューサーです。市場データと業界知識を活用してください。

【{entity_name}基本情報】
{_basic_str}

【フィールド一覧】
{_jj.dumps(ai_q_h, ensure_ascii=False)}

各フィールドに適切な内容を生成し {{"0":"値",...}} 形式で返してください。JSONのみ。""",
                system_prompt="JSONのみ。",
                ai_tier="core",
                max_tokens=2500,
            )
            if isinstance(_ai_r, dict):
                for idx_s, val in _ai_r.items():
                    try:
                        i = int(idx_s)
                        if 0 <= i < len(ai_q_h) and val:
                            fill_h[ai_q_h[i]["selector"]] = str(val)
                    except (ValueError, TypeError):
                        pass

        _disp_h = [{"selector": f.get("selector"), "label": f.get("label") or f.get("selector"), "value": fill_h.get(f.get("selector",""), ""), "type": f.get("type","text")} for f in _html_form_elements if f.get("selector") in fill_h]
        _compat_h = {d["label"]: d["value"] for d in _disp_h}
        return {
            "ok": True,
            "cast_name": req.cast_name,
            "generated_fields": _compat_h,
            "mapping_results": [{
                "mapping_id": req.source_html_mapping_id,
                "media_name": mm_h.get("media_name", req.source_html_mapping_id),
                "target_url": req.source_html_target_url,
                "fill_fields": fill_h,
                "display_fields": _disp_h,
            }],
        }

    # ── フォーム要素取得 & AI一括生成（マッピングごと） ──
    # db / _market_context_pre / _rag_context_pre は上の共通ブロックで定義済み
    _market_context = _market_context_pre  # Playwrightパスのプロンプトで参照するエイリアス
    _rag_context    = _rag_context_pre
    mapping_results = []

    # パススルーフィールド判定（ユーザー入力値をそのまま埋める）
    _PASSTHROUGH_PATTERNS = [
        (["名前", "お名前", "キャスト名", "name", "氏名"], req.cast_name),
        (["ふりがな", "フリガナ", "読み", "yomi", "kana", "ruby"], None),  # AI生成
        (["年齢", "age", "nenrei"], req.age),
        (["身長", "height", "shincho"], req.height),
        (["バスト", "bust", "b:", "b :"], req.bust),
        (["ウエスト", "waist", "w:", "w :"], req.waist),
        (["ヒップ", "hip", "h:", "h :"], req.hip),
        (["カップ", "cup", "cup_size"], req.cup),
    ]

    _SKIP_TYPES = {"file", "hidden", "submit", "button", "image", "reset", "password"}
    _TEXT_TYPES = {"text", "textarea", "number", "email", "tel", "url", "date", "time",
                   "search", "select-one", "select", "textarea"}

    import urllib.parse as _up

    for mid in req.target_mapping_ids:
        _target_url = ""  # エラー時にも含められるよう先に初期化
        try:
            doc = db.collection("media_mappings").document(mid).get()
            if not doc.exists:
                mapping_results.append({"mapping_id": mid, "error": f"マッピングID {mid} が存在しません", "target_url": ""})
                continue
            mm = doc.to_dict() or {}
            mm["mapping_id"] = mid
            _assert_tenant_access(mm, user, "他テナントのマッピングは操作できません")

            _target_url = _resolve_cross_media_target_url(db, mid, mm, "entity_register")

            # entity_register URLが未発見の場合は明確なエラーで即返却
            if not _target_url:
                _mname = mm.get("media_name", mid)
                mapping_results.append({
                    "mapping_id": mid,
                    "media_name": _mname,
                    "error": f"【{_mname}】AI整備済みの新規登録フォームURLがありません。媒体基盤のAI整備で entity_register を使える状態にしてください。",
                    "target_url": "",
                    "display_fields": [],
                    "fill_fields": {},
                })
                continue

            # entity_register は新規登録フォーム → URLにIDパラメータがあれば除去
            # （深堀で cast_edit?id=168 が保存されている場合に cast_edit として解釈）
            _parsed_u = _up.urlparse(_target_url)
            _qparams = _up.parse_qs(_parsed_u.query, keep_blank_values=True)
            _id_keys = [k for k in _qparams if k.lower() in ("id", "cast_id", "girl_id", "member_id", "staff_id", "user_id")]
            if _id_keys:
                for k in _id_keys:
                    _qparams.pop(k)
                _target_url = _up.urlunparse(_parsed_u._replace(query=_up.urlencode(_qparams, doseq=True)))
                print(f"[GENERATE_PROFILE] entity_register URL: stripped ID → {_target_url}", flush=True)
            snap = take_site_preview(mm, target_url=_target_url, extract_all_form_elements=True)
            form_elements = snap.get("form_elements", [])

            # 1. パススルー（基本情報）をセレクタで埋める
            fill_fields: dict[str, str] = {}
            ai_queue: list[dict] = []

            for f in form_elements:
                el_type = (f.get("type") or "").lower()
                el_tag = (f.get("tag") or "").lower()
                selector = f.get("selector") or ""
                if not selector:
                    continue
                if el_type in _SKIP_TYPES:
                    continue
                label_lower = (f.get("label") or "").lower()
                name_lower = (f.get("name") or "").lower()

                # パススルー判定
                matched_passthrough = False
                for patterns, val in _PASSTHROUGH_PATTERNS:
                    if val and any(p.lower() in label_lower or p.lower() in name_lower for p in patterns):
                        fill_fields[selector] = val
                        matched_passthrough = True
                        break

                if not matched_passthrough and (el_type in _TEXT_TYPES or el_tag == "textarea"):
                    ai_queue.append({
                        "idx": len(ai_queue),
                        "selector": selector,
                        "label": f.get("label", ""),
                        "name": f.get("name", ""),
                        "type": el_type or el_tag,
                    })

            # 2. AI一括生成（フォームフィールド全体を渡して一発生成）
            if ai_queue:
                _fields_str = _jj.dumps(
                    [{"idx": f["idx"], "label": f["label"], "name": f["name"], "type": f["type"]} for f in ai_queue],
                    ensure_ascii=False, indent=2
                )
                _ai_prompt = f"""あなたは{_industry_ctx}のトッププロデューサーです。
Geminiとして業界の最新トレンド・流行語・予約につながる表現の知識を全て活用してください。

{_market_context}

{_rag_context}

【{entity_name}基本情報】
{_basic_str}

【入力が必要なフォームフィールド一覧】
{_fields_str}

上記の市場データ・業界知識・専用ナレッジを最大限活用し、予約につながる高品質なプロフィールを生成してください。
- 市場監視のトレンドフレーズを自然に取り入れ、陳腐な表現は避ける
- 「タイプ」「性格」は10文字以内の短い形容詞（人気タイプを参考に）
- 「コメント」「自己紹介」は150〜200文字の自然な文章（読んで予約したくなる内容）
- 「キャッチコピー」は20〜30文字のインパクトある一文（トレンドフレーズを活かす）
- 「ふりがな」はキャスト名の読みをひらがなで
- 「好きな下着」「チャームポイント」「マイブーム」等は{industry}向けに具体的・魅力的に
- 数値フィールド(年齢/身長等)は基本情報に記載がある場合はその値を、なければ空文字で

以下のJSON形式で返してください（idxをキーにした辞書）:
{{"0": "フィールド0の値", "1": "フィールド1の値", ...}}
説明・前置き不要。JSONのみ。"""

                _ai_values = call_llm_json(
                    prompt=_ai_prompt,
                    system_prompt="JSONのみ出力。各フィールドの値のみ。",
                    ai_tier="core",
                    max_tokens=2500,
                )
                if isinstance(_ai_values, dict):
                    for idx_str, val in _ai_values.items():
                        try:
                            idx = int(idx_str)
                            if 0 <= idx < len(ai_queue) and val:
                                fill_fields[ai_queue[idx]["selector"]] = str(val)
                        except (ValueError, TypeError):
                            pass

            # 3. 表示用フィールドリスト（フォーム要素 + 生成値を結合）
            display_fields = []
            for f in form_elements:
                sel = f.get("selector") or ""
                if sel and sel in fill_fields:
                    display_fields.append({
                        "selector": sel,
                        "label": f.get("label") or f.get("name") or sel,
                        "value": fill_fields[sel],
                        "type": (f.get("type") or f.get("tag") or "text").lower(),
                    })

            print(f"[GENERATE_PROFILE] cast={req.cast_name} mid={mid} fill_fields={len(fill_fields)} display={len(display_fields)}", flush=True)
            mapping_results.append({
                "mapping_id": mid,
                "media_name": mm.get("media_name", mid),
                "target_url": _target_url,
                "fill_fields": fill_fields,        # CSSセレクタ→値（execute直通）
                "display_fields": display_fields,  # ラベル→値（UI表示用）
            })

        except Exception as _me:
            print(f"[GENERATE_PROFILE_PREVIEW] mapping={mid} url={_target_url} error={_me}", flush=True)
            mapping_results.append({"mapping_id": mid, "error": str(_me), "target_url": _target_url})

    # generated_fields: 後方互換のため先頭マッピングのdisplay_fieldsをラベル→値に変換
    _first = next((r for r in mapping_results if "display_fields" in r), None)
    _compat_generated = {d["label"]: d["value"] for d in (_first or {}).get("display_fields", [])} if _first else {}

    return {
        "ok": True,
        "cast_name": req.cast_name,
        "generated_fields": _compat_generated,  # 後方互換
        "mapping_results": mapping_results,
    }


class ProfileExecuteRequest(BaseModel):
    cast_name: str
    fill_fields: dict = {}              # {CSSセレクタ: 値} — fill_and_submit_form へ直通
    final_fields: dict = {}             # 後方互換（旧フロントエンドのみ使用）
    target_mapping_id: str
    target_url: str = ""


@router.post("/cross_media/generate_profile_execute")
def generate_profile_execute(req: ProfileExecuteRequest, user: dict = Depends(verify_token)):
    """
    生成プロフィールをフォームへ書き込み登録する。
    fill_fields に {CSSセレクタ: 値} を渡すことで fill_and_submit_form へ直通する。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    doc = db.collection("media_mappings").document(req.target_mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    mm = doc.to_dict() or {}
    mm["mapping_id"] = req.target_mapping_id
    _assert_tenant_access(mm, user, "他テナントのマッピングは操作できません")

    target_url = req.target_url or _resolve_cross_media_target_url(db, req.target_mapping_id, mm, "entity_register")

    from api.core.browser_executor import fill_and_submit_form

    # fill_fields (新形式) を優先。なければ後方互換でfinal_fieldsを使う
    selector_values = req.fill_fields if req.fill_fields else req.final_fields
    if not selector_values:
        raise HTTPException(status_code=400, detail="fill_fieldsが空です")

    result = fill_and_submit_form(mm, target_url=target_url, field_values=selector_values)

    print(f"[GENERATE_PROFILE_EXECUTE] cast={req.cast_name} mapping={req.target_mapping_id} ok={result.get('ok')} fields={len(selector_values)}", flush=True)
    return {
        "ok": result.get("ok", False),
        "cast_name": req.cast_name,
        "message": result.get("message", ""),
        "target_url": target_url,
        "filled_count": len(selector_values),
    }


# ─────────────────────────────────────────────────────────────────────
# 求人対応（項目7）: 専用ナレッジ＋市場調査でAI文面生成 → 人が承認 → 送信
#   - generate: 文面をAI生成（送信しない）。AIに丸投げせず登録知識・市場に基づく。
#   - 送信は既存の createAgentTask(blog_post) 経由でPENDING化→承認→実行で行う。
# ─────────────────────────────────────────────────────────────────────
def _build_recruit_knowledge_context(db, tenant_id: str, query: str) -> dict:
    """
    ✨AI新規生成と同じ「専用ナレッジ(RAG)＋市場監視データ」を取得して返す。
    求人対応の文面生成でAIに丸投げせず、登録済み知識・市場調査に基づかせるために使う。
    戻り値: {"market": str, "knowledge": str, "has_market": bool, "has_knowledge": bool}
    """
    market_context = ""
    try:
        _mr_docs = db.collection("monitoring_results").where("tenant_id", "==", tenant_id).stream()
        def _ts_key(x: dict) -> str:
            v = x.get("executed_at")
            if v is None:
                return ""
            return v.isoformat() if hasattr(v, "isoformat") else str(v)
        _mr_list = sorted([d.to_dict() for d in _mr_docs], key=_ts_key, reverse=True)
        if _mr_list:
            _l = _mr_list[0]
            _ts = _ts_key(_l)[:10]
            _t  = ", ".join(_l.get("trending_phrases") or []) or "データなし"
            _pt = ", ".join(_l.get("popular_types") or []) or "データなし"
            _av = ", ".join(_l.get("avoid_phrases") or [])
            _rc = "; ".join((_l.get("recommendations") or [])[:3])
            market_context = (
                f"【市場監視データ（{_ts}時点）】\n"
                f"トレンドフレーズ: {_t}\n人気タイプ: {_pt}\n"
                + (f"避けるべき表現: {_av}\n" if _av else "")
                + (f"推奨事項: {_rc}\n" if _rc else "")
                + (f"市場サマリー: {_l.get('ai_summary', '')}" if _l.get("ai_summary") else "")
            ).strip()
    except Exception as _e:
        print(f"[RECRUIT_CTX] 監視データ取得エラー: {_e}", flush=True)

    knowledge_context = ""
    try:
        from api.core.rag import rag_retrieve_chunks_with_vec, embed_text as _embed
        _vec = _embed(query)
        _chunks = rag_retrieve_chunks_with_vec(tenant_id, _vec, top_k=5, threshold=0.25)
        if _chunks:
            knowledge_context = "【専用ナレッジ（登録済み業界知識）】\n" + "\n".join(c.get("text", "") for c in _chunks)
    except Exception as _e:
        print(f"[RECRUIT_CTX] RAGエラー: {_e}", flush=True)

    return {
        "market": market_context,
        "knowledge": knowledge_context,
        "has_market": bool(market_context),
        "has_knowledge": bool(knowledge_context),
    }


class RecruitGenerateRequest(BaseModel):
    target_mapping_id: str = ""
    recruit_mode: str = "reply"      # offer=オファー送信文 / reply=応募・問合せ返信 / text=求人掲載文言
    applicant_context: str = ""      # 相手のメッセージ・プロフィール・状況
    conditions: str = ""             # 条件・対応方針（給与/シフト/歓迎条件 等）
    instruction: str = ""            # 追加指示
    industry: str = "nightlife"


@router.post("/cross_media/recruit/generate")
def recruit_generate(req: RecruitGenerateRequest, user: dict = Depends(verify_token)):
    """
    求人対応の文面をAI生成する（送信はしない）。
    ✨AI新規生成と同じ専用ナレッジ＋市場調査に基づき、条件・対応方法に合った文面を作る。
    生成→人が承認→送信、の「生成」段階を担う。ID/PASSはレスポンスに含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    from api.core.llm_client import call_llm_json
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    industry = _normalize_industry(req.industry or "nightlife")

    _mode_spec = {
        "offer": ("求人オファー送信文", "指定条件に合う応募候補者へ送る、誠実で魅力的なスカウト/オファーメッセージ"),
        "reply": ("応募・問い合わせへの返信文", "応募者・問い合わせ者のメッセージに対する、丁寧で具体的な返信"),
        "text":  ("求人掲載文言", "求人サイトに掲載する募集文・キャッチコピー・条件説明文"),
    }.get(req.recruit_mode, ("応募・問い合わせへの返信文", "応募者への返信"))
    _doc_label, _doc_desc = _mode_spec

    _q = f"求人 採用 {req.recruit_mode} {req.conditions} {req.applicant_context} 返信 オファー 募集 {industry}"
    ctx = _build_recruit_knowledge_context(db, tenant_id, _q)
    _ctx_parts = [p for p in [ctx["market"], ctx["knowledge"]] if p]
    _ctx_str = ("\n\n".join(_ctx_parts) + "\n\n") if _ctx_parts else ""

    _industry_ctx = {
        "nightlife": "ナイトワーク（キャバクラ・クラブ・ラウンジ等）の求人。応募者の不安に寄り添い、安心感と具体的メリットを示す文体で。",
    }.get(industry, "求人対応。誠実で具体的、相手の不安を解消する文体で。")

    _prompt = f"""{_ctx_str}あなたは{_industry_ctx}に精通した採用担当のプロです。
上記の市場データ・専用ナレッジ（登録済み業界知識）を最大限活用し、AIの一般論ではなく
当社の方針・市場状況に合った文面を作成してください。

【作成する文書】{_doc_label}
（{_doc_desc}）

【相手の情報・メッセージ】
{req.applicant_context or '（特になし）'}

【条件・対応方針】
{req.conditions or '（特になし）'}

【追加指示】
{req.instruction or 'なし'}

要件:
- 相手の状況に具体的に対応する（テンプレ丸写しにしない）
- 条件・対応方針を正確に反映する
- {industry}の応募者が安心して次の行動に進める内容にする
- 日本語。そのまま送信できる完成文。

JSON形式で返してください:
{{"title": "件名/見出し（任意・無ければ空文字）", "body": "本文"}}"""

    try:
        result = call_llm_json(
            prompt=_prompt,
            system_prompt="JSONのみ出力。```等のMarkdownブロック禁止。",
            ai_tier="core",
            max_tokens=2000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成に失敗しました: {type(e).__name__}")

    if not isinstance(result, dict):
        result = {"title": "", "body": str(result)}

    print(f"[RECRUIT_GENERATE] mode={req.recruit_mode} knowledge={ctx['has_knowledge']} market={ctx['has_market']}", flush=True)
    return {
        "recruit_mode": req.recruit_mode,
        "doc_label": _doc_label,
        "title": result.get("title", ""),
        "body": result.get("body", ""),
        "knowledge_used": ctx["has_knowledge"],
        "market_used": ctx["has_market"],
        "note": (
            "登録済み専用ナレッジ＋市場調査に基づき生成しました（AIに丸投げしていません）。"
            if (ctx["has_knowledge"] or ctx["has_market"]) else
            "専用ナレッジ・市場データが未登録のため一般知識で生成しました。ナレッジ登録で精度が上がります。"
        ),
    }


@router.post("/task/batch/approve")
def approve_task_batch(req: BatchTaskActionRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_task_batches").document(req.batch_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="batchが見つかりません")
    batch = snap.to_dict() or {}
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    if batch.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このbatchへのアクセス権がありません")
    if batch.get("status") not in ("PENDING", "PARTIAL_APPROVED"):
        raise HTTPException(status_code=400, detail="PENDING状態のbatchのみ承認できます。現在: " + str(batch.get("status")))

    now = datetime.datetime.utcnow()
    approved = []
    skipped = []
    for task_id in batch.get("task_ids", []):
        tref = db.collection("agent_tasks").document(task_id)
        tsnap = tref.get()
        if not tsnap.exists:
            skipped.append({"task_id": task_id, "reason": "task_not_found"})
            continue
        task = tsnap.to_dict() or {}
        if task.get("status") != "PENDING":
            skipped.append({"task_id": task_id, "reason": "not_pending", "status": task.get("status")})
            continue
        tref.update({
            "status": "APPROVED",
            "approved_by": user.get("uid", ""),
            "approved_at": now,
        })
        try:
            session_id = task.get("workflow_session_id") or ""
            if session_id:
                sref = db.collection("workflow_execution_sessions").document(session_id)
                ssnap = sref.get()
                if ssnap.exists:
                    session = ssnap.to_dict() or {}
                    if session.get("tenant_id") == tenant_id or user.get("role", "").lower() == "admin":
                        sref.update({
                            "approval_state": "APPROVED",
                            "status": "READY",
                            "approved_by": user.get("uid", ""),
                            "approved_at": now,
                            "updated_at": now,
                        })
        except Exception as _batch_wf_e:
            print(f"[BATCH_APPROVE_WORKFLOW_SYNC_ERROR] task_id={task_id} {type(_batch_wf_e).__name__}:{_batch_wf_e}", flush=True)
        approved.append(task_id)

    counts = dict(batch.get("counts") or {})
    counts["approved"] = counts.get("approved", 0) + len(approved)
    new_status = "APPROVED" if approved else "PENDING"
    ref.update({
        "status": new_status,
        "approved_at": now if approved else batch.get("approved_at"),
        "approved_by": user.get("uid", "") if approved else batch.get("approved_by", ""),
        "approval_skipped": skipped,
        "counts": counts,
        "updated_at": now,
    })
    return {
        "batch_id": req.batch_id,
        "status": new_status,
        "approved_task_ids": approved,
        "skipped": skipped,
        "counts": counts,
    }


@router.post("/task/batch/execute")
def execute_task_batch(req: BatchTaskActionRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("agent_task_batches").document(req.batch_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="batchが見つかりません")
    batch = snap.to_dict() or {}
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    if batch.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このbatchへのアクセス権がありません")
    if batch.get("status") not in ("APPROVED", "PARTIAL_FAILED", "PARTIAL_DONE", "FAILED", "NEEDS_REVIEW"):
        raise HTTPException(status_code=400, detail="APPROVED済みbatchのみ実行できます。現在: " + str(batch.get("status")))

    now = datetime.datetime.utcnow()
    ref.update({"status": "RUNNING", "started_at": now, "updated_at": now})
    results = []
    done = failed = skipped = needs_review = 0

    for task_id in batch.get("task_ids", []):
        tsnap = db.collection("agent_tasks").document(task_id).get()
        task_status = (tsnap.to_dict() or {}).get("status") if tsnap.exists else ""
        if task_status == "DONE":
            done += 1
            results.append({"task_id": task_id, "status": "DONE", "skipped": True, "reason": "already_done"})
            continue
        if task_status != "APPROVED":
            skipped += 1
            results.append({"task_id": task_id, "status": task_status or "MISSING", "skipped": True, "reason": "not_approved"})
            continue
        try:
            res = execute_task(task_id, user)
            st = res.get("status", "")
            if st == "DONE":
                done += 1
            elif st == "NEEDS_REVIEW":
                needs_review += 1
            elif st in ("FAILED", "BLOCKED", "WAITING_MAPPING", "WAITING_EXECUTOR"):
                failed += 1
            results.append({
                "task_id": task_id,
                "status": st,
                "result": res.get("result", {}),
            })
        except HTTPException as e:
            failed += 1
            results.append({
                "task_id": task_id,
                "status": "FAILED",
                "error": e.detail,
                "http_status": e.status_code,
            })
        except Exception as e:
            failed += 1
            results.append({
                "task_id": task_id,
                "status": "FAILED",
                "error": str(e),
            })

    total = len(batch.get("task_ids", []))
    if total and done == total:
        final_status = "DONE"
    elif failed > 0 and (done > 0 or needs_review > 0):
        final_status = "PARTIAL_FAILED"
    elif done > 0:
        final_status = "PARTIAL_DONE"
    elif needs_review > 0 and failed == 0:
        final_status = "NEEDS_REVIEW"
    else:
        final_status = "FAILED"
    counts = dict(batch.get("counts") or {})
    counts.update({
        "done": done,
        "failed": failed,
        "skipped": (batch.get("counts") or {}).get("skipped", 0) + skipped,
        "needs_review": needs_review,
    })
    finished = datetime.datetime.utcnow()
    ref.update({
        "status": final_status,
        "execution_results": results,
        "counts": counts,
        "executed_at": finished,
        "updated_at": finished,
    })
    return {
        "batch_id": req.batch_id,
        "status": final_status,
        "results": results,
        "counts": counts,
    }


@router.get("/task/batch/list")
def list_task_batches(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    docs = db.collection("agent_task_batches").where("tenant_id", "==", tenant_id).limit(100).stream()
    rows = []
    for d in docs:
        row = _iso_top_level(d.to_dict() or {})
        rows.append(row)
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"batches": rows, "count": len(rows)}


@router.get("/task/list")
def list_tasks(
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    # P31: agent_permissions から approval_count/auto_enabled を各taskに合成
    try:
        _perm_doc = db.collection("agent_permissions").document(tenant_id).get()
        _perm_ops = (_perm_doc.to_dict() or {}).get("operations", {}) if _perm_doc.exists else {}
        for _t in rows:
            _op = _t.get("operation_type", "")
            _op_data = _perm_ops.get(_op, {})
            _t["approval_count"] = _op_data.get("approval_count", 0)
            _t["auto_enabled"]   = _op_data.get("auto_enabled", False)
    except Exception as _perm_e:
        print(f"[task_list_perm_merge] error: {type(_perm_e).__name__}", flush=True)
    rows.sort(key=lambda x: x.get("created_at") or "", reverse=True)
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
    _assert_tenant_access(t, user, "このタスクへのアクセス権がありません")
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
    rows.sort(key=lambda x: x.get("executed_at") or "", reverse=True)
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


@router.patch("/permissions/{tenant_id}")
def update_permissions(tenant_id: str, req: AgentPermissionsUpdateRequest, user: dict = Depends(verify_token)):
    ctx = _resolve_agent_user_context(user)
    if not _can_manage_agent_permissions(user):
        raise HTTPException(status_code=403, detail="管理者のみ変更できます")
    if not ctx["is_admin"] and ctx["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="他テナントの権限は変更できません")
    db = get_db()
    update: dict = {
        "tenant_id": tenant_id,
        "updated_at": datetime.datetime.utcnow(),
        "updated_by": user.get("uid", ""),
    }
    if req.admin_granted is not None:
        update["admin_granted"] = bool(req.admin_granted)
    if req.allowed_agents is not None:
        update["allowed_agents"] = [str(x) for x in req.allowed_agents if str(x)]
    if req.allowed_operations is not None:
        update["allowed_operations"] = [str(x) for x in req.allowed_operations if str(x)]
    if req.max_tasks_per_day is not None:
        update["max_tasks_per_day"] = max(0, int(req.max_tasks_per_day or 0))
    ref = db.collection("agent_permissions").document(tenant_id)
    ref.set(update, merge=True)
    doc = ref.get()
    return doc.to_dict() or update


@router.get("/permission/operations")
def get_permission_operations(user: dict = Depends(verify_token)):
    """
    P31: operation別のapproval_count / auto_enabled一覧を返す。
    UIの「自動化設定」タブで表示・解除ボタン表示に使用。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("agent_permissions").document(tenant_id).get()
    ops = (doc.to_dict() or {}).get("operations", {}) if doc.exists else {}
    result = []
    for op_type, op_data in ops.items():
        result.append({
            "operation_type":   op_type,
            "approval_count":   op_data.get("approval_count", 0),
            "auto_enabled":     op_data.get("auto_enabled", False),
            "last_approved_at": op_data.get("last_approved_at").isoformat() if op_data.get("last_approved_at") and hasattr(op_data.get("last_approved_at"), "isoformat") else None,
        })
    return {"tenant_id": tenant_id, "operations": result}


@router.post("/permission/auto_disable")
def disable_auto_enabled(
    operation_type: str,
    user: dict = Depends(verify_token),
):
    """
    P31: 指定operation_typeのauto_enabledをFalseに戻す（解除）。
    UIの「自動化解除」ボタンから呼ぶ。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    ref = db.collection("agent_permissions").document(tenant_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="権限設定が見つかりません")
    ops = (doc.to_dict() or {}).get("operations", {})
    if operation_type not in ops:
        raise HTTPException(status_code=404, detail=f"operation_type '{operation_type}' の権限設定が見つかりません")
    ops[operation_type]["auto_enabled"] = False
    ops[operation_type]["auto_disabled_at"] = datetime.datetime.utcnow()
    ops[operation_type]["auto_disabled_by"] = user.get("uid", "")
    ref.set({"operations": ops}, merge=True)
    print(f"[P31_AUTO_DISABLED] tenant={tenant_id} op={operation_type} by={user.get('uid','')}", flush=True)
    return {
        "tenant_id":      tenant_id,
        "operation_type": operation_type,
        "auto_enabled":   False,
        "message":        f"'{operation_type}' の自動化を解除しました",
    }

@router.get("/industry_templates")
def get_industry_templates(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    return {"templates": INDUSTRY_TEMPLATES, "agent_types": list(AGENT_TYPES), "operation_types": list(OPERATION_TYPES)}



# --- /plan: 自然言語指示からtask候補を生成 ---

class PlanRequest(BaseModel):
    instruction: str
    mapping_id: Optional[str] = None

class GoalPlanRequest(BaseModel):
    goal: str
    mapping_id: Optional[str] = None

class InstructionTaskCreateRequest(BaseModel):
    instruction: str
    mapping_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    scheduled_at: Optional[str] = None

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
    mappings = []
    for d in mappings_docs:
        m = d.to_dict() or {}
        m["mapping_id"] = m.get("mapping_id") or d.id
        mappings.append(m)
    if req.mapping_id:
        mappings = [m for m in mappings if m.get("mapping_id") == req.mapping_id] or mappings
    hydrated_mappings = []
    for m in mappings[:50]:
        mid = m.get("mapping_id") or ""
        if not mid:
            hydrated_mappings.append(m)
            continue
        m2, _ = _ensure_capability_view_for_mapping(db, mid, m)
        m2["mapping_id"] = mid
        hydrated_mappings.append(m2)
    mappings = hydrated_mappings
    selected_mapping = mappings[0] if req.mapping_id and len(mappings) == 1 else None

    # agent_ops 取得
    ops = [
        {
            "op_id": op.get("op_id") or "",
            "display_name": op.get("display_name") or op.get("name") or "",
            "operation_type": op.get("operation_type", ""),
            "entity_type": op.get("entity_type", ""),
            "industry": _normalize_industry(op.get("industry", "other")),
            "payload_schema": op.get("payload_schema", {}),
        }
        for op in _load_agent_ops_for_user(db, ctx)
        if op.get("active") is not False
    ]

    # LLMで解析
    try:
        from api.core.llm_client import call_llm_json
        site_names = [m.get('media_name') for m in mappings]
        op_list = []
        for o in ops:
            plan_fields = _payload_schema_fields_for_operation(selected_mapping, o, o.get("operation_type", ""))
            op_list.append({
                'op_id': o['op_id'],
                'display_name': o['display_name'],
                'operation_type': o['operation_type'],
                'industry': o['industry'],
                'fields': [
                    {
                        "key": f.get("key", ""),
                        "label": f.get("label", ""),
                        "type": f.get("type", "text"),
                        "required": bool(f.get("required")),
                        "canonical": f.get("canonical", ""),
                    }
                    for f in plan_fields[:200]
                ],
            })
        site_capabilities = []
        for m in mappings:
            ops_view = (m.get("capability_view") or {}).get("operations") or {}
            ready_ops = []
            review_ops = []
            for op_name, row in ops_view.items():
                if not isinstance(row, dict):
                    continue
                mapped_fields = _mapping_fields_for_operation(m, op_name)
                mapped_field_descriptors = [
                    {
                        "key": next(iter(_payload_keys_for_mapping_fields([f])), ""),
                        "label": str(f.get("label") or f.get("name") or f.get("canonical") or ""),
                        "canonical": str(f.get("canonical") or ""),
                        "type": str(f.get("type") or "text"),
                    }
                    for f in mapped_fields[:200]
                ]
                rec = {
                    "operation_type": op_name,
                    "target_url": row.get("target_url", ""),
                    "fields": mapped_field_descriptors or [
                        {
                            "key": f.get("selector_key") or f.get("role") or "",
                            "label": f.get("label") or f.get("selector_key") or f.get("role") or "",
                            "canonical": f.get("canonical") or "",
                            "type": f.get("type") or "text",
                        }
                        for f in row.get("field_schema") or []
                        if isinstance(f, dict)
                    ],
                }
                if row.get("status") == "READY" and row.get("taskable"):
                    ready_ops.append(rec)
                elif row.get("status") == "NEEDS_REVIEW":
                    rec["missing"] = row.get("missing", [])
                    review_ops.append(rec)
            site_capabilities.append({
                "mapping_id": m.get("mapping_id"),
                "media_name": m.get("media_name"),
                "ready_operations": ready_ops,
                "needs_review_operations": review_ops[:20],
            })
        prompt = (
            'You are an agent task analyzer. '
            + 'Read the user instruction and return JSON only (no markdown). '
            + 'Format: {"ready": bool, "media_name": str|null, "op_id": str|null, "operation_type": str|null, "payload": {}, "preview": str, "question": str|null} '
            + ('Selected site: ' + str(selected_mapping.get("media_name")) + ' ' if selected_mapping else '')
            + 'Sites: ' + str(site_names) + ' '
            + 'Ops: ' + str(op_list) + ' '
            + 'Actual site capabilities from structural HTML model: ' + str(site_capabilities) + ' '
            + 'Instruction: ' + req.instruction + ' '
            + 'Fill payload from operation fields. When a selected site has saved mapping fields, prefer those labels/keys and do not collapse entity registration or profile update into only name/body unless no richer fields are known. '
            + 'Set ready=true only when the requested site has that operation in ready_operations. '
            + 'If it is only in needs_review_operations, set ready=false and ask the user to review the media structure in Japanese. '
            + 'If info missing set ready=false and write question in Japanese, naming the exact missing field labels when possible. If site unclear set ready=false.'
        )
        result = call_llm_json(prompt)
    except Exception as e:
        print(f"[PLAN] LLM error: {type(e).__name__}", flush=True)
        result = _fallback_plan_from_instruction(req.instruction)

    if not result.get("operation_type"):
        fb = _fallback_plan_from_instruction(req.instruction)
        if fb.get("operation_type"):
            result = {**result, **fb}
    if result.get("operation_type"):
        op_type = result.get("operation_type") or ""
        if selected_mapping and not result.get("ready"):
            selected_ready = False
            if _capability_op_is_taskable(selected_mapping, op_type):
                selected_ready = True
            elif op_type in {"post_monitoring", "page_monitor"} and selected_mapping.get("mapping_id"):
                selected_ready = True
            elif op_type in {"offer_send", "recruit_inbox_scan", "recruit_reply"} and selected_mapping.get("credential_secret_name"):
                selected_ready = True
            if selected_ready:
                result["ready"] = True
        matched_op = next((o for o in ops if o.get("op_id") == result.get("op_id")), None)
        if not matched_op:
            matched_op = next((o for o in ops if o.get("operation_type") == op_type), None)
        plan_fields = _payload_schema_fields_for_operation(selected_mapping, matched_op or {"payload_schema": {"fields": []}}, op_type)
        if plan_fields:
            result["available_fields"] = [
                {
                    "key": f.get("key", ""),
                    "label": f.get("label", ""),
                    "type": f.get("type", "text"),
                    "required": bool(f.get("required")),
                    "canonical": f.get("canonical", ""),
                }
                for f in plan_fields[:200]
            ]
            result["mapped_field_count"] = len(_mapping_fields_for_operation(selected_mapping or {}, op_type))
            if selected_mapping and result.get("ready"):
                base_preview = str(result.get("preview") or "")
                field_note = f" 保存済みフィールド {len(plan_fields)} 項目を会話入力に連動できます。"
                if field_note.strip() not in base_preview:
                    result["preview"] = (base_preview + field_note).strip()
    return {"ok": True, **result}


@router.post("/goal/plan")
def plan_agent_goal(req: GoalPlanRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.goal or not req.goal.strip():
        raise HTTPException(status_code=400, detail="goalが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    mappings = []
    for d in db.collection("media_mappings").where("tenant_id", "==", tenant_id).limit(200).stream():
        m = d.to_dict() or {}
        m["mapping_id"] = m.get("mapping_id") or d.id
        if req.mapping_id and m.get("mapping_id") != req.mapping_id:
            continue
        m, _ = _ensure_capability_view_for_mapping(db, m["mapping_id"], m)
        m["mapping_id"] = m.get("mapping_id") or d.id
        mappings.append(m)

    tasks = []
    for d in db.collection("agent_tasks").where("tenant_id", "==", tenant_id).limit(300).stream():
        item = d.to_dict() or {}
        item["task_id"] = item.get("task_id") or d.id
        tasks.append(item)

    batches = []
    try:
        for d in db.collection("agent_task_batches").where("tenant_id", "==", tenant_id).limit(100).stream():
            item = d.to_dict() or {}
            item["batch_id"] = item.get("batch_id") or d.id
            batches.append(item)
    except Exception:
        batches = []

    cross_tasks = []
    try:
        for d in db.collection("cross_media_tasks").where("tenant_id", "==", tenant_id).limit(100).stream():
            item = d.to_dict() or {}
            item["cross_task_id"] = item.get("cross_task_id") or d.id
            cross_tasks.append(item)
    except Exception:
        cross_tasks = []

    schedules = []
    try:
        for d in db.collection("agent_schedules").where("tenant_id", "==", tenant_id).limit(100).stream():
            item = d.to_dict() or {}
            item["schedule_id"] = item.get("schedule_id") or d.id
            schedules.append(item)
    except Exception:
        schedules = []

    return _build_goal_plan(req.goal, mappings, tasks, batches, cross_tasks, schedules)


# --- agent_ops（Firestoreから動的取得） ---

@router.get("/ops")
def list_ops(user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    result = _load_agent_ops_for_user(db, ctx)
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

def _find_template_by_url(db, media_url: str, login_url: str) -> Optional[dict]:
    """URLドメインが一致するadminテナントのmappingをテンプレートとして返す。"""
    from urllib.parse import urlparse
    def _domain(url: str) -> str:
        try:
            return urlparse(str(url or "")).netloc.lower().lstrip("www.")
        except Exception:
            return ""
    candidates = {_domain(u) for u in [media_url, login_url] if u}
    candidates.discard("")
    if not candidates:
        return None
    best: Optional[dict] = None
    best_score: Optional[tuple] = None

    def _noisy_menu_count(items: list) -> int:
        noisy = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get(k) or "") for k in ("title", "text", "label", "url", "href", "absolute_url"))
            if "entry_sid=" in text or "entry_time=" in text or " sid=" in text or re.search(r"[A-Za-z0-9_-]{40,}", text):
                noisy += 1
        return noisy

    def _template_score(data: dict) -> tuple:
        manual_items = data.get("manual_menu_items") or []
        scan = data.get("manual_menu_scan_results") or {}
        summary = scan.get("summary") or {}
        ready = int(summary.get("ready") or 0)
        review = int(summary.get("needs_review") or 0)
        scanned = int(summary.get("scanned") or 0)
        total = int(summary.get("total") or len(manual_items) or 0)
        failed = int(summary.get("failed") or 0)
        no_op = int(summary.get("no_operation") or 0)
        no_dom = int(summary.get("no_editable_dom") or 0)
        actionable = ready + review
        clean_bonus = 1 if total > 0 and scanned >= total and failed == 0 else 0
        detail_count = int(scan.get("items_written") or len(scan.get("items") or []) or 0)
        noisy = _noisy_menu_count(manual_items)
        return (
            clean_bonus,
            actionable,
            ready,
            scanned,
            -failed,
            -(no_op + no_dom),
            -noisy,
            detail_count,
            len(manual_items),
        )

    for snap in db.collection("media_mappings").where("tenant_id", "==", "default").stream():
        data = snap.to_dict() or {}
        tmpl_domain = _domain(data.get("media_url") or data.get("login_url") or "")
        if not tmpl_domain or tmpl_domain not in candidates:
            continue
        score = _template_score(data)
        if best_score is None or score > best_score:
            best_score = score
            best = {**data, "_doc_id": snap.id}
    return best


@router.post("/media/map")
def create_media_mapping(
    req: MediaMappingCreateRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token),
):
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
    credential_secret_name = _agent_credential_secret_name(tenant_id, mapping_id)
    doc = {
        "mapping_id": mapping_id,
        "tenant_id": tenant_id,
        "media_name": req.media_name,
        "media_url": req.media_url or "",
        "login_url": req.login_url or req.media_url or "",
        "industry": _normalize_industry(req.industry or "other"),
        "operation_type": req.operation_type,
        "auth_type": req.auth_type,
        "dom_selectors": req.dom_selectors,
        "form_structure": req.form_structure,
        "credential_secret_name": credential_secret_name,
        "credential_registered": False,
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

    cloned_from = None
    if tenant_id != "default":
        template = _find_template_by_url(db, req.media_url or "", req.login_url or "")
        if template:
            tmpl_doc_id = template.get("_doc_id") or template.get("mapping_id") or ""
            copy_fields = [
                "business_conditions", "capabilities",
                "manual_menu_items", "manual_menu_source_url",
                "login_selectors", "dom_selectors",
            ]
            updates: dict = {
                "cloned_from_template": tmpl_doc_id,
                "legacy_mapping_clone_disabled": True,
                "mapping_contract_version": "ai_confirmed_v1",
            }
            for field in copy_fields:
                val = template.get(field)
                if val:
                    updates[field] = val
            db.collection("media_mappings").document(mapping_id).update(updates)
            cloned_from = tmpl_doc_id
            print(f"[TEMPLATE_CLONE] mapping_id={mapping_id} cloned_from={tmpl_doc_id} legacy_mapping_clone_disabled=True", flush=True)

    # ── 共有キャッシュから即時適用（同じURLを別テナントが解析済みなら0手間でREADY）──
    cache_applied = False
    try:
        _cache_snap = db.collection("media_html_cache").where("url", "==", req.media_url or "").limit(1).stream()
        for _c in _cache_snap:
            _cdata = _c.to_dict() or {}
            _ready_ops_cache = _cdata.get("ready_ops") or []
            _op_maps_cache   = _cdata.get("operation_mappings") or {}
            import datetime as _dt_clone
            _now_clone = _dt_clone.datetime.utcnow()
            _clone_updates: dict = {"updated_at": _now_clone}
            _CAP_MAP_CLONE = {
                "news_post": "can_post_news", "blog_post": "can_post_news",
                "text_update": "can_update_text", "media_replace": "can_upload_image",
                "schedule_update": "can_update_schedule", "price_update": "can_update_price",
                "entity_register": "can_register_entity", "entity_update": "can_update_entity",
            }
            for _op, _op_data in _op_maps_cache.items():
                if not isinstance(_op_data, dict):
                    continue
                _clone_updates[f"operation_mappings.{_op}"] = {
                    **_op_data,
                    "production_ready":    True,
                    "confirmation_status": "AI_CONFIRMED",
                    "source":              "AI_CONFIRMED",
                    "cloned_from_cache":   True,
                }
                _cap_key = _CAP_MAP_CLONE.get(_op)
                if _cap_key:
                    _clone_updates[f"capabilities.{_cap_key}"] = True
            if _clone_updates:
                db.collection("media_mappings").document(mapping_id).update(_clone_updates)
                cache_applied = True
                print(f"[MEDIA_CREATE_CACHE_CLONE] mapping_id={mapping_id} ready_ops={_ready_ops_cache}", flush=True)
            break
    except Exception as _ce:
        print(f"[MEDIA_CREATE_CACHE_CLONE_ERROR] {type(_ce).__name__}:{_ce}", flush=True)

    # キャッシュなし時: 認証情報登録完了後（save_credential）に自動解析が起動する

    return {
        "mapping_id":           mapping_id,
        "status":               "created",
        "cloned_from_template": cloned_from,
        "cache_applied":        cache_applied,
    }


def _hydrate_menu_scan_for_mapping_response(db, mapping_id: str, mapping: dict, limit: int = 500) -> dict:
    scan = mapping.get("manual_menu_scan_results") or {}
    if not isinstance(scan, dict):
        return mapping

    detail_items: list[dict] = []
    try:
        for snap in (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("menu_scan_items")
            .limit(max(1, min(int(limit or 500), 500)))
            .stream()
        ):
            row = snap.to_dict() or {}
            if isinstance(row, dict) and row.get("url"):
                detail_items.append(row)
    except Exception as e:
        print(f"[MENU_SCAN_RESPONSE_HYDRATE_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
        detail_items = []

    if not detail_items:
        return mapping

    by_url: dict[str, dict] = {}
    order: list[str] = []

    def add_item(item: dict) -> None:
        if not isinstance(item, dict):
            return
        url = str(item.get("url") or item.get("absolute_url") or item.get("href") or "")
        norm = _normalize_menu_scan_url(item.get("canonical_url") or url)
        if not norm:
            return
        if norm not in by_url:
            by_url[norm] = {"url": url, "canonical_url": norm}
            order.append(norm)
        elif url and not by_url[norm].get("url"):
            by_url[norm]["url"] = url
        merged = dict(item)
        merged["canonical_url"] = norm
        by_url[norm].update(merged)

    for item in scan.get("items") or []:
        add_item(item)
    for item in mapping.get("manual_menu_items") or []:
        add_item({
            "url": item.get("absolute_url") or item.get("href") or "",
            "title": item.get("title") or item.get("absolute_url") or item.get("href") or "",
            "category": item.get("category") or "その他",
        })
    for item in detail_items:
        add_item(item)

    merged = [by_url[url] for url in order if url in by_url]
    compact_items = _compact_menu_scan_items(merged, parent=True)
    hydrated_scan = dict(scan)
    hydrated_scan["items"] = compact_items
    hydrated_scan["stored_summary"] = scan.get("summary") or {}
    hydrated_scan["summary"] = _menu_scan_summary(compact_items)
    hydrated_scan["items_storage_mode"] = scan.get("items_storage_mode") or "subcollection"
    hydrated_scan["items_subcollection"] = scan.get("items_subcollection") or "menu_scan_items"
    hydrated_scan["items_hydrated_from"] = "menu_scan_items"
    hydrated_scan["items_hydrated_count"] = len(detail_items)
    mapping["manual_menu_scan_results"] = hydrated_scan
    if not ((mapping.get("capability_view") or {}).get("operations")):
        try:
            written = _seed_structure_pages_from_menu_items(db, mapping_id, detail_items, source="menu_scan_detail_hydrate")
            if written:
                view = _refresh_capability_view_for_mapping(db, mapping_id, mapping)
                if view:
                    latest = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                    mapping["capability_view"] = latest.get("capability_view") or view
                    mapping["structure_model"] = latest.get("structure_model") or mapping.get("structure_model") or {}
                    mapping["operation_mappings"] = latest.get("operation_mappings") or mapping.get("operation_mappings") or {}
                    mapping["operation_steps_by_type"] = latest.get("operation_steps_by_type") or mapping.get("operation_steps_by_type") or {}
        except Exception as e:
            print(f"[MENU_SCAN_CAPABILITY_HYDRATE_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
    return mapping


def _backfill_menu_scan_for_mapping_response(db, mapping_id: str, mapping: dict) -> dict:
    """Restore menu-scan status for older mappings that have URLs and operation mappings but no scan summary."""
    scan = mapping.get("manual_menu_scan_results") or {}
    manual_items = [it for it in (mapping.get("manual_menu_items") or []) if isinstance(it, dict)]
    detail_items = []
    try:
        for snap in (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("menu_scan_items")
            .limit(500)
            .stream()
        ):
            row = snap.to_dict() or {}
            if isinstance(row, dict):
                detail_items.append(row)
    except Exception as e:
        print(f"[MENU_SCAN_BACKFILL_DETAIL_READ_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)

    if not manual_items and not detail_items and not (scan.get("items") or []):
        return mapping

    op_maps = mapping.get("operation_mappings") or {}
    op_steps = mapping.get("operation_steps_by_type") or {}
    by_norm: dict[str, dict] = {}
    order: list[str] = []

    def ensure_item(url: str, title: str = "", category: str = "") -> dict | None:
        norm = _normalize_menu_scan_url(url)
        if not norm:
            return None
        if norm not in by_norm:
            by_norm[norm] = {
                "url": url,
                "canonical_url": norm,
                "title": title or url,
                "category": category or "その他",
                "updated_ops": [],
            }
            order.append(norm)
        else:
            row = by_norm[norm]
            if url and not row.get("url"):
                row["url"] = url
            if title and (not row.get("title") or row.get("title") == row.get("url")):
                row["title"] = title
            if category and not row.get("category"):
                row["category"] = category
        return by_norm[norm]

    for item in scan.get("items") or []:
        if not isinstance(item, dict):
            continue
        row = ensure_item(
            str(item.get("url") or item.get("absolute_url") or item.get("href") or ""),
            str(item.get("title") or item.get("text") or ""),
            str(item.get("category") or ""),
        )
        if row is not None:
            row.update(dict(item))
            row["canonical_url"] = _normalize_menu_scan_url(row.get("canonical_url") or row.get("url") or "")

    for item in manual_items:
        row = ensure_item(
            str(item.get("absolute_url") or item.get("href") or item.get("url") or ""),
            str(item.get("title") or item.get("text") or item.get("label") or ""),
            str(item.get("category") or ""),
        )
        if row is not None and not row.get("message"):
            row.setdefault("message", "")

    for item in detail_items:
        row = ensure_item(
            str(item.get("url") or item.get("absolute_url") or item.get("href") or ""),
            str(item.get("title") or item.get("text") or ""),
            str(item.get("category") or ""),
        )
        if row is not None:
            row.update(dict(item))
            row["canonical_url"] = _normalize_menu_scan_url(row.get("canonical_url") or row.get("url") or "")

    for op, op_data in op_maps.items():
        if not isinstance(op_data, dict):
            continue
        target_url = str(op_data.get("target_url") or "")
        row = ensure_item(
            target_url,
            str(op_data.get("manual_title") or (op_data.get("form_schema") or {}).get("title") or op),
            "",
        )
        if row is None:
            continue
        steps = op_steps.get(op) or []
        step_count = len(steps) if isinstance(steps, list) else int(op_data.get("step_count") or 0)
        selectors_count = len(op_data.get("selectors") or {})
        existing_rows = [r for r in (row.get("updated_ops") or []) if isinstance(r, dict) and r.get("op") != op]
        existing_rows.append({
            "op": op,
            "status": op_data.get("status") or "",
            "target_url": target_url or row.get("url") or "",
            "missing": list(op_data.get("missing") or [])[:8],
            "selectors": selectors_count,
            "steps": step_count,
            "protected": bool(op_data.get("executable")),
        })
        row["updated_ops"] = existing_rows
        current_status = str(row.get("status") or "")
        if op_data.get("status") == "READY" and step_count > 0:
            row["status"] = "READY"
        elif not current_status and op_data.get("status") == "NEEDS_REVIEW":
            row["status"] = "NEEDS_REVIEW"

    merged_items = [_compact_menu_item_result(by_norm[norm], parent=True) for norm in order if norm in by_norm]
    if not merged_items:
        return mapping

    summary = _menu_scan_summary(merged_items)
    compact_scan = dict(scan)
    compact_scan["items"] = merged_items
    compact_scan["summary"] = summary
    compact_scan["items_count"] = len(merged_items)
    compact_scan["items_storage_mode"] = compact_scan.get("items_storage_mode") or ("subcollection" if detail_items else "response_backfill")
    compact_scan["backfilled_for_response"] = True
    compact_scan["updated_at"] = compact_scan.get("updated_at") or datetime.datetime.utcnow().isoformat()
    mapping["manual_menu_scan_results"] = compact_scan

    if not scan.get("summary") or not (scan.get("items") or []):
        try:
            db.collection("media_mappings").document(mapping_id).update({
                "manual_menu_scan_results": compact_scan,
                "updated_at": datetime.datetime.utcnow(),
            })
        except Exception as e:
            print(f"[MENU_SCAN_BACKFILL_WRITE_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)

    return mapping


def _backfill_manual_operation_mappings_for_response(db, mapping_id: str, mapping: dict) -> dict:
    """Legacy manual-page backfill is disabled under the AI-confirmed mapping contract."""
    mapping["legacy_manual_operation_backfill_disabled"] = True
    return mapping


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
        m["mapping_id"] = m.get("mapping_id") or d.id
        m = _delete_legacy_operation_mappings_for_ai_contract(db, m["mapping_id"], m)
        for k in ("created_at", "last_verified_at", "crawler_last_run_at", "updated_at"):
            if m.get(k) and hasattr(m[k], "isoformat"):
                m[k] = m[k].isoformat()
        # crawl_state内のdatetimeもシリアライズ
        cs = m.get("crawl_state")
        if isinstance(cs, dict):
            for ck in ("updated_at", "started_at"):
                if cs.get(ck) and hasattr(cs[ck], "isoformat"):
                    cs[ck] = cs[ck].isoformat()
        ms = m.get("media_schema") or {}
        sf = m.get("schema_first") or {}
        if isinstance(ms, dict) and (ms.get("forms_count") or ms.get("canonical_fields_count")) and not sf.get("forms_count"):
            m["schema_first"] = {
                **sf,
                "status": "READY" if ms.get("forms_count") else (sf.get("status") or "NO_FORM_SCHEMA"),
                "storage_mode": ms.get("storage_mode", "subcollections"),
                "schema_generation": sf.get("schema_generation") or ms.get("schema_generation", ""),
                "forms_count": ms.get("forms_count", 0),
                "canonical_fields_count": ms.get("canonical_fields_count", 0),
                "entities_count": ms.get("entities_count", 0),
                "derived_from": "media_schema",
            }
        m = _hydrate_menu_scan_for_mapping_response(db, d.id, m)
        m = _backfill_manual_operation_mappings_for_response(db, d.id, m)
        m = _backfill_menu_scan_for_mapping_response(db, d.id, m)
        m = _filter_mapping_menu_scan_for_response(m)
        m = _annotate_operation_mappings_for_response(m)
        if not ((m.get("capability_view") or {}).get("operations")) and (m.get("media_schema") or m.get("schema_first")):
            try:
                _rebuild_media_schema_for_mapping(db, d.id)
                latest = db.collection("media_mappings").document(d.id).get().to_dict() or {}
                m["capability_view"] = latest.get("capability_view") or m.get("capability_view") or {}
                m["structure_model"] = latest.get("structure_model") or m.get("structure_model") or {}
                m["operation_mappings"] = latest.get("operation_mappings") or m.get("operation_mappings") or {}
                m["operation_steps_by_type"] = latest.get("operation_steps_by_type") or m.get("operation_steps_by_type") or {}
                m = _annotate_operation_mappings_for_response(m)
            except Exception as e:
                print(f"[MEDIA_LIST_CAPABILITY_BACKFILL_ERROR] mapping_id={d.id} {type(e).__name__}", flush=True)
        result.append(m)
    print(f"[AGENT_MEDIA_LIST] tenant_id={tenant_id} count={len(result)}", flush=True)
    return {"mappings": result, "count": len(result)}

@router.get("/media/map/{mapping_id}/schema")
def get_media_mapping_schema(
    mapping_id: str,
    include_forms: bool = True,
    include_fields: bool = True,
    forms_limit: int = 80,
    fields_limit: int = 240,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    ref = db.collection("media_mappings").document(mapping_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    mapping = snap.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and str(user.get("role", "")).lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    schema = mapping.get("media_schema") or {}
    generation = (mapping.get("schema_first") or {}).get("schema_generation") or schema.get("schema_generation") or ""

    def _dedupe_response_forms(rows: list[dict]) -> list[dict]:
        import re as _re_schema_resp
        out = []
        seen = set()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            field_sig = tuple(sorted({
                str(f.get("canonical") or f.get("name") or f.get("label") or "")
                for f in (row.get("fields") or [])
                if isinstance(f, dict) and (f.get("canonical") or f.get("name") or f.get("label"))
            }))
            title_key = _re_schema_resp.sub(r"\s+", " ", str(row.get("title") or row.get("page_purpose") or "")).strip().lower()
            key = (
                row.get("entity_type") or "",
                title_key,
                int(row.get("fields_count") or len(row.get("fields") or []) or 0),
                field_sig,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    forms = []
    if include_forms:
        for d in ref.collection("schema_forms").limit(max(1, min(int(forms_limit or 80), 200))).stream():
            row = d.to_dict() or {}
            if generation and row.get("schema_generation") != generation:
                continue
            forms.append(row)
    forms = _dedupe_response_forms(forms)

    fields = []
    if include_fields:
        for d in ref.collection("schema_fields").limit(max(1, min(int(fields_limit or 240), 500))).stream():
            row = d.to_dict() or {}
            if generation and row.get("schema_generation") != generation:
                continue
            fields.append(row)

    schema_first = dict(mapping.get("schema_first") or {})
    schema_forms_total = schema.get("forms_count", 0) if isinstance(schema, dict) else 0
    schema_fields_total = schema.get("canonical_fields_count", 0) if isinstance(schema, dict) else 0
    schema_entities_total = schema.get("entities_count", 0) if isinstance(schema, dict) else 0
    if (not schema_first.get("forms_count")) and (schema_forms_total or forms):
        schema_first.update({
            "status": "READY",
            "storage_mode": schema.get("storage_mode", "subcollections") if isinstance(schema, dict) else "subcollections",
            "schema_generation": generation,
            "forms_count": schema_forms_total or len(forms),
            "canonical_fields_count": schema_fields_total or len(fields),
            "entities_count": schema_entities_total or len(mapping.get("entity_schema") or {}),
            "derived_from": "media_schema",
        })
    mapping, capability_view = _ensure_capability_view_for_mapping(db, mapping_id, mapping)

    return {
        "mapping_id": mapping_id,
        "media_name": mapping.get("media_name", ""),
        "schema_first": schema_first,
        "media_schema": schema,
        "structure_model": mapping.get("structure_model") or {},
        "capability_view": capability_view or mapping.get("capability_view") or {},
        "entity_schema": mapping.get("entity_schema") or {},
        "schema_generation": generation,
        "forms": forms,
        "fields": fields,
        "counts": {
            "forms": len(forms),
            "fields": len(fields),
            "forms_total": schema_forms_total or len(forms),
            "fields_total": schema_fields_total or len(fields),
        },
    }

@router.get("/media/map/{mapping_id}/menu_items")
def list_media_menu_scan_items(
    mapping_id: str,
    limit: int = 300,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    ref = db.collection("media_mappings").document(mapping_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    mapping = snap.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and str(user.get("role", "")).lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    scan = mapping.get("manual_menu_scan_results") or {}
    max_limit = max(1, min(int(limit or 300), 500))
    items = []
    try:
        for d in ref.collection("menu_scan_items").limit(max_limit).stream():
            row = d.to_dict() or {}
            if row:
                items.append(row)
    except Exception as e:
        print(f"[MENU_SCAN_ITEMS_READ_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
        items = []
    if not items:
        items = [i for i in (scan.get("items") or [])[:max_limit] if isinstance(i, dict)]
    items = _filter_menu_items_for_response(mapping, items)[:max_limit]
    return {
        "mapping_id": mapping_id,
        "media_name": mapping.get("media_name", ""),
        "items": items,
        "count": len(items),
        "summary": _menu_scan_summary(_compact_menu_scan_items(items, parent=True)) if items else (scan.get("summary") or {}),
        "stored_summary": scan.get("summary") or {},
        "storage_mode": scan.get("items_storage_mode") or "parent",
        "items_subcollection": scan.get("items_subcollection") or "",
    }

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
    ctx = _assert_tenant_access(m, user, "他テナントのマッピングは削除できません")
    if m.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="他テナントのマッピングは削除できません")
    cascade = {"secret": "skipped", "session": "skipped", "schedules_disabled": 0, "subcollections_deleted": 0}
    try:
        secret_name = m.get("credential_secret_name") or _agent_credential_secret_name(m.get("tenant_id") or ctx["tenant_id"], mapping_id)
        expected_secret = _agent_credential_secret_name(m.get("tenant_id") or ctx["tenant_id"], mapping_id)
        if secret_name == expected_secret:
            from api.core.secret_manager import delete_secret
            cascade["secret"] = delete_secret(secret_name).get("status", "unknown")
        else:
            cascade["secret"] = "skipped_untrusted_name"
    except Exception as e:
        cascade["secret"] = f"failed:{type(e).__name__}"
    try:
        from api.core.browser_executor import _clear_cached_session
        _clear_cached_session(mapping_id, reason="media_mapping_deleted")
        cascade["session"] = "cleared"
    except Exception as e:
        cascade["session"] = f"failed:{type(e).__name__}"
    try:
        for sched in db.collection("agent_schedules").where("media_mapping_id", "==", mapping_id).stream():
            sd = sched.to_dict() or {}
            if sd.get("tenant_id") == m.get("tenant_id") or ctx["is_admin"]:
                sched.reference.update({
                    "enabled": False,
                    "schedule_health": "DISABLED",
                    "last_error_type": "MEDIA_MAPPING_DELETED",
                    "last_error_message": "参照先の媒体マッピングが削除されたため停止しました",
                    "updated_at": datetime.datetime.utcnow(),
                })
                cascade["schedules_disabled"] += 1
    except Exception as e:
        cascade["schedules_error"] = type(e).__name__
    for sub_name in ("menu_scan_items", "schema_forms", "schema_fields", "pages"):
        try:
            for child in ref.collection(sub_name).limit(500).stream():
                child.reference.delete()
                cascade["subcollections_deleted"] += 1
        except Exception as e:
            cascade[f"{sub_name}_error"] = type(e).__name__
    ref.delete()
    return {"mapping_id": mapping_id, "status": "deleted", "cascade": cascade}


# --- agent_schedules ---

class ScheduleCreateRequest(BaseModel):
    op_id: Optional[str] = None
    operation_type: Optional[str] = None
    media_mapping_id: Optional[str] = None
    menu_item_target_url: Optional[str] = None
    cron_expr: str
    payload_template: dict = Field(default_factory=dict)
    enabled: bool = True

@router.post("/schedule/create")
def create_schedule(req: ScheduleCreateRequest, user: dict = Depends(verify_token)):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    ops = _load_agent_ops_for_user(db, ctx)
    op = next((o for o in ops if req.op_id and o.get("op_id") == req.op_id), None)
    if not op and req.operation_type:
        op = next((o for o in ops if o.get("operation_type") == req.operation_type), None)
    if not op:
        raise HTTPException(status_code=400, detail="存在しないoperationです")
    if op.get("active") is False:
        raise HTTPException(status_code=400, detail="このOperationは現在利用できません")

    operation_type = op.get("operation_type") or req.operation_type or ""
    if operation_type not in OPERATION_TYPES:
        raise HTTPException(status_code=400, detail="無効なoperation_typeです")
    _enforce_agent_permissions(ctx, op.get("category", "hp_update"), operation_type)

    mapping = {}
    operation_steps = []
    operation_override = {}
    menu_item = {}
    media_mapping_id = req.media_mapping_id or (req.payload_template or {}).get("media_mapping_id", "")
    if media_mapping_id:
        mm_doc = db.collection("media_mappings").document(media_mapping_id).get()
        if not mm_doc.exists:
            raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
        mapping = mm_doc.to_dict() or {}
        mapping["mapping_id"] = media_mapping_id
        mapping["id"] = media_mapping_id
        if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
            raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

        mapping, _ = _ensure_capability_view_for_mapping(db, media_mapping_id, mapping)
        if req.menu_item_target_url:
            items = ((mapping.get("manual_menu_scan_results") or {}).get("items") or [])
            menu_item = next((it for it in items if isinstance(it, dict) and it.get("url") == req.menu_item_target_url), {})
            detail_item = _get_menu_scan_item_document(db, media_mapping_id, req.menu_item_target_url)
            if detail_item:
                menu_item = {**menu_item, **detail_item}
            operation = ((menu_item.get("operations") or {}).get(operation_type) or {})
            if operation.get("status") != "READY" or not operation.get("steps"):
                raise HTTPException(status_code=400, detail="このHTMLメニューURLは指定operationのREADY予約対象ではありません")
            if not (
                _operation_mapping_is_production_ready(operation)
                or menu_item.get("production_ready") is True
                or menu_item.get("confirmation_status") == "AI_CONFIRMED"
            ):
                raise HTTPException(status_code=400, detail="このHTMLメニューURLはAI整備前の候補のため予約実行できません。媒体基盤のAI整備で保存してください。")
            operation_steps = operation.get("steps") or []
            operation_override = {
                "status": "READY",
                "target_url": operation.get("target_url") or req.menu_item_target_url,
                "selectors": operation.get("selectors") or {},
                "missing": operation.get("missing", []),
                "validation_score": operation.get("validation_score", 0),
                "executable": True,
                "source": "TASK_OVERRIDE",
                "confirmed": True,
                "production_ready": True,
                "last_scanned_at": operation.get("scanned_at") or datetime.datetime.utcnow().isoformat(),
            }
        else:
            cap_op = _operation_from_capability_view(mapping, operation_type)
            op_map = ((mapping.get("operation_mappings") or {}).get(operation_type) or {})
            operation_steps = ((mapping.get("operation_steps_by_type") or {}).get(operation_type) or [])
            if not _operation_mapping_is_production_ready(op_map) or not operation_steps:
                raise HTTPException(status_code=400, detail="この媒体は指定operationのREADY予約対象ではありません")

    schedule_id = str(uuid.uuid4())
    _now_cs = datetime.datetime.utcnow()
    payload_template = dict(req.payload_template or {})
    if media_mapping_id:
        payload_template["media_mapping_id"] = media_mapping_id
        payload_template["media_name"] = mapping.get("media_name", "")
    if req.menu_item_target_url:
        payload_template["menu_item_target_url"] = req.menu_item_target_url
        payload_template["target_url"] = req.menu_item_target_url
        payload_template["menu_item_title"] = menu_item.get("title", "")
        payload_template["menu_item_category"] = menu_item.get("category", "")

    doc = {
        "schedule_id": schedule_id,
        "tenant_id": tenant_id,
        "op_id": op.get("op_id", ""),
        "operation_type": operation_type,
        "agent_type": op.get("category", "hp_update"),
        "entity_type": op.get("entity_type", ""),
        "industry": _normalize_industry(mapping.get("industry") or op.get("industry") or "generic"),
        "op_display_name": op.get("display_name") or op.get("name") or op.get("op_id", ""),
        "media_mapping_id": media_mapping_id,
        "media_name": mapping.get("media_name", ""),
        "menu_item_target_url": req.menu_item_target_url or "",
        "menu_item_title": menu_item.get("title", ""),
        "menu_item_category": menu_item.get("category", ""),
        "operation_steps": operation_steps,
        "operation_mapping_override": operation_override,
        "cron_expr": req.cron_expr,
        "payload_template": payload_template,
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
    _assert_tenant_access(s, user, "他テナントのスケジュールは変更できません")
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

        op = {}
        if s.get("op_id"):
            op_doc = db.collection("agent_ops").document(s.get("op_id", "")).get()
            if op_doc.exists:
                op = op_doc.to_dict() or {}
        required_approval = op.get("required_approval", True)
        task_id = str(uuid.uuid4())
        payload = dict(s.get("payload_template", {}) or {})
        op_operation_type = s.get("operation_type") or op.get("operation_type", "schedule_update")
        op_category = s.get("agent_type") or op.get("category", "hp_update")
        try:
            _validate_agent_operation_pair(op_category, op_operation_type)
        except HTTPException as ve:
            failed.append({"schedule_id": schedule_id, "reason": str(ve.detail)})
            continue
        # P0: scheduler経由でもpermission enforcement
        _sched_tenant = s.get("tenant_id", "default")
        _sched_perm = _get_agent_permissions(_sched_tenant)
        if not _sched_perm.get("admin_granted", False):
            failed.append({"schedule_id": schedule_id, "reason": "admin_granted is OFF"})
            continue
        _allowed_ops_s = _sched_perm.get("allowed_operations") or []
        _allowed_ags_s = _sched_perm.get("allowed_agents") or []
        if _allowed_ops_s and op_operation_type and op_operation_type not in _allowed_ops_s:
            failed.append({"schedule_id": schedule_id, "reason": f"operation_type '{op_operation_type}' not in allowed_operations"})
            continue
        if _allowed_ags_s and op_category and op_category not in _allowed_ags_s:
            failed.append({"schedule_id": schedule_id, "reason": f"agent_type '{op_category}' not in allowed_agents"})
            continue
        # P4: media_mapping_id取得・operation_steps取得
        _sched_media_mapping_id = s.get("media_mapping_id", "") or payload.get("media_mapping_id", "")
        _sched_operation_steps = s.get("operation_steps") or []
        _sched_operation_override = s.get("operation_mapping_override") or {}
        _sched_media_mapping = {}
        if _sched_media_mapping_id:
            try:
                _mm_doc = db.collection("media_mappings").document(_sched_media_mapping_id).get()
                if _mm_doc.exists:
                    _sched_media_mapping = _mm_doc.to_dict() or {}
                    _sched_media_mapping, _ = _ensure_capability_view_for_mapping(db, _sched_media_mapping_id, _sched_media_mapping)
                    if not _sched_operation_steps:
                        if s.get("menu_item_target_url"):
                            _items = ((_sched_media_mapping.get("manual_menu_scan_results") or {}).get("items") or [])
                            _menu_item = next((it for it in _items if isinstance(it, dict) and it.get("url") == s.get("menu_item_target_url")), {})
                            _detail_item = _get_menu_scan_item_document(db, _sched_media_mapping_id, s.get("menu_item_target_url"))
                            if _detail_item:
                                _menu_item = {**_menu_item, **_detail_item}
                            _operation = ((_menu_item.get("operations") or {}).get(op_operation_type) or {})
                            _sched_operation_steps = _operation.get("steps") or []
                            if not _sched_operation_override and _operation:
                                _sched_operation_override = {
                                    "status": "READY",
                                    "target_url": _operation.get("target_url") or s.get("menu_item_target_url"),
                                    "selectors": _operation.get("selectors") or {},
                                    "missing": _operation.get("missing", []),
                                    "validation_score": _operation.get("validation_score", 0),
                                    "executable": True,
                                    "source": "manual_menu_item_schedule_runtime",
                                    "last_scanned_at": _operation.get("scanned_at") or now.isoformat(),
                                }
                        else:
                            _cap_runtime = _operation_from_capability_view(_sched_media_mapping, op_operation_type)
                            if _cap_runtime and not (_cap_runtime.get("status") == "READY" and _cap_runtime.get("taskable")):
                                failed.append({
                                    "schedule_id": schedule_id,
                                    "reason": "operation_not_structurally_ready",
                                    "operation_status": _cap_runtime.get("status", "UNDISCOVERED"),
                                    "missing": _cap_runtime.get("missing", []),
                                })
                                continue
                            _steps_by_type = _sched_media_mapping.get("operation_steps_by_type", {})
                            _sched_operation_steps = _steps_by_type.get(op_operation_type, [])
            except Exception as _mm_e:
                print(f"[SCHEDULER_MEDIA_MAPPING_ERROR] {type(_mm_e).__name__}", flush=True)
        if _sched_media_mapping_id:
            payload = dict(payload)
            payload["media_mapping_id"] = _sched_media_mapping_id
            if _sched_media_mapping.get("media_name"):
                payload.setdefault("media_name", _sched_media_mapping.get("media_name", ""))
        if s.get("menu_item_target_url"):
            payload.setdefault("menu_item_target_url", s.get("menu_item_target_url"))
            payload.setdefault("target_url", s.get("menu_item_target_url"))
            payload.setdefault("menu_item_title", s.get("menu_item_title", ""))
            payload.setdefault("menu_item_category", s.get("menu_item_category", ""))
        _sched_workflow_id = str(uuid.uuid4())
        op_industry = _normalize_industry(s.get("industry") or op.get("industry", "other"))
        op_entity_type = s.get("entity_type") or op.get("entity_type", "")
        op_display_name = s.get("op_display_name") or op.get("display_name") or op.get("name") or op.get("op_name") or s.get("op_id", "")
        op_snapshot = {
            "op_id": s.get("op_id", ""),
            "display_name": op_display_name,
            "category": op_category,
            "operation_type": op_operation_type,
            "entity_type": op_entity_type,
            "industry": op_industry,
            "payload_schema_version": op.get("payload_schema_version", "1"),
        }
        workflow_session_id, workflow_risk = _create_task_workflow_session(
            db=db,
            tenant_id=_sched_tenant,
            workflow_id=_sched_workflow_id,
            operation_type=op_operation_type,
            operation_steps=_sched_operation_steps,
            media_mapping=_sched_media_mapping,
            media_mapping_id=_sched_media_mapping_id,
            media_name=_sched_media_mapping.get("media_name", "") or s.get("media_name", ""),
            goal_context="schedule",
        )
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
            "status":         "PENDING" if (required_approval or (op_category == "hp_update" and not _sched_operation_steps)) else "APPROVED",
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
            "schedule_id":       schedule_id,
            "media_mapping_id":  _sched_media_mapping_id,
            "operation_steps":   _sched_operation_steps,
            "operation_mapping_override": _sched_operation_override,
            "menu_item_target_url": s.get("menu_item_target_url", ""),
            "menu_item_title": s.get("menu_item_title", ""),
            "menu_item_category": s.get("menu_item_category", ""),
            "source": "schedule_menu_item" if s.get("menu_item_target_url") else "schedule",
            "workflow_id":       _sched_workflow_id,
            "workflow_session_id": workflow_session_id,
            "risk_level": workflow_risk.get("risk_level", ""),
            "risk_score": workflow_risk.get("risk_score", 0.0),
            "risk_factors": workflow_risk.get("risk_factors", []),
            "require_human_approval": workflow_risk.get("require_human_approval", False),
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
def save_credential(mapping_id: str, req: CredentialSaveRequest, background_tasks: BackgroundTasks, user: dict = Depends(verify_token)):
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
    if m.get("tenant_id") != tenant_id and not _is_agent_admin(user):
        raise HTTPException(status_code=403, detail="他テナントのマッピングは操作できません")

    # secret名生成（平文保存禁止・secret名はレスポンスに含めない）
    secret_name = _agent_credential_secret_name(m.get("tenant_id") or tenant_id, mapping_id)

    # Secret Managerへ保存（失敗時はFirestore更新しない）
    from api.core.secret_manager import save_secret_json
    try:
        save_secret_json(secret_name, {"username": req.login_id, "password": req.password})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Firestore更新（Secret保存成功時のみ）
    db.collection("media_mappings").document(mapping_id).update({
        "credential_secret_name": secret_name,
        "credential_registered": True,
        "credential_updated_at": datetime.datetime.utcnow(),
    })

    # 認証情報登録完了 → バックグラウンドで自動解析を起動
    def _bg_auto_setup_after_cred(_mid: str):
        try:
            from api.core.browser_executor import auto_setup_mapping_ai as _asa
            from api.core.firestore_client import get_db as _gdb
            _db2 = _gdb()
            _snap2 = _db2.collection("media_mappings").document(_mid).get()
            if _snap2.exists:
                _md = _snap2.to_dict() or {}
                _md["mapping_id"] = _mid
                result = _asa(_md, db=_db2)
                print(f"[CRED_AUTO_SETUP_DONE] mapping_id={_mid} ok={result.get('ok')} ready_ops={result.get('ready_ops')}", flush=True)
        except Exception as _e:
            print(f"[CRED_AUTO_SETUP_ERROR] mapping_id={_mid} {type(_e).__name__}:{_e}", flush=True)

    background_tasks.add_task(_bg_auto_setup_after_cred, mapping_id)

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
    _assert_tenant_access(m, user, "他テナントのマッピングは更新できません")
    now = datetime.datetime.utcnow()
    ref.update({"last_verified_at": now})
    return {"mapping_id": mapping_id, "last_verified_at": now.isoformat()}


class ScreeningCriteria(BaseModel):
    height_min: Optional[int] = None        # cm
    height_max: Optional[int] = None
    weight_max: Optional[int] = None        # kg
    cup_min: Optional[str] = None           # A/B/C/D...
    tattoo_ok: Optional[bool] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    custom_conditions: Optional[str] = None # 追加条件（フリーテキスト）
    image_check: Optional[bool] = True      # 添付画像もVision AIで判定

class ReplyPolicy(BaseModel):
    tone: Optional[str] = "polite"          # polite / casual
    interview_info: Optional[str] = None    # 面接時の標準案内
    shop_conditions: Optional[str] = None  # PR条件（返信生成に使う）

class BusinessConditionsRequest(BaseModel):
    site_purpose: Optional[str] = None      # scout / reply / post / monitor / other
    screening: Optional[ScreeningCriteria] = None
    reply_policy: Optional[ReplyPolicy] = None
    offer_template: Optional[str] = None    # オファーひな形文

@router.patch("/media/map/{mapping_id}/business_conditions")
def update_business_conditions(
    mapping_id: str,
    req: BusinessConditionsRequest,
    user: dict = Depends(verify_token),
):
    """マッピングに業務条件（スクリーニング基準・返信ポリシー・ひな形文）を保存する。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    _assert_tenant_access(doc.to_dict(), user, "他テナントのマッピングは更新できません")

    bc: dict = {}
    if req.site_purpose is not None:
        bc["site_purpose"] = req.site_purpose
    if req.screening is not None:
        bc["screening"] = {k: v for k, v in req.screening.dict().items() if v is not None}
    if req.reply_policy is not None:
        bc["reply_policy"] = {k: v for k, v in req.reply_policy.dict().items() if v is not None}
    if req.offer_template is not None:
        bc["offer_template"] = req.offer_template

    ref.update({"business_conditions": bc, "updated_at": datetime.datetime.utcnow()})
    return {"ok": True, "mapping_id": mapping_id, "business_conditions": bc}


# ─── Step 3: recruit_conversations 管理 ──────────────────────────────────


class RecruitReplyRequest(BaseModel):
    conversation_id: str = ""
    new_message: str = ""
    instruction: Optional[str] = ""
    mapping_id: Optional[str] = ""


@router.get("/recruit/conversations")
def list_recruit_conversations(
    mapping_id: Optional[str] = None,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    q = db.collection("recruit_conversations").where("tenant_id", "==", tenant_id)
    if mapping_id:
        q = q.where("mapping_id", "==", mapping_id)
    try:
        from google.cloud import firestore as _fs_rc
        docs = list(q.order_by("updated_at", direction=_fs_rc.Query.DESCENDING).limit(200).stream())
    except Exception:
        docs = list(q.limit(200).stream())
    result = []
    for d in docs:
        item = d.to_dict() or {}
        item["conversation_id"] = d.id
        for _df in ("created_at", "updated_at", "offer_sent_at"):
            if _df in item and hasattr(item[_df], "isoformat"):
                item[_df] = item[_df].isoformat()
        for _msg in (item.get("messages") or []):
            if "sent_at" in _msg and hasattr(_msg.get("sent_at"), "isoformat"):
                _msg["sent_at"] = _msg["sent_at"].isoformat()
        result.append(item)
    return {"conversations": result, "count": len(result)}


@router.post("/recruit/conversations/{conversation_id}/reply/generate")
async def generate_recruit_reply(
    conversation_id: str,
    req: RecruitReplyRequest,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    conv_doc = db.collection("recruit_conversations").document(conversation_id).get()
    if not conv_doc.exists:
        raise HTTPException(status_code=404, detail="会話スレッドが見つかりません")
    conv = conv_doc.to_dict() or {}
    if conv.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    mid = req.mapping_id or conv.get("mapping_id", "")
    bc: dict = {}
    if mid:
        m_doc = db.collection("media_mappings").document(mid).get()
        if m_doc.exists:
            bc = (m_doc.to_dict() or {}).get("business_conditions") or {}

    reply_policy = bc.get("reply_policy") or {}
    tone = reply_policy.get("tone") or "polite"
    interview_info = reply_policy.get("interview_info") or ""
    shop_conditions = reply_policy.get("shop_conditions") or ""

    current_phase = conv.get("phase", "offer_sent")
    messages_list = conv.get("messages") or []
    candidate_name = conv.get("candidate_name") or "候補者"

    history_text = ""
    for msg in messages_list[-6:]:
        role_label = "【店舗】" if msg.get("role") == "shop" else f"【{candidate_name}】"
        history_text += f"{role_label} {msg.get('content', '')}\n"

    phase_goal = {
        "offer_sent":          "オファーへの反応・関心度を確認する返信",
        "waiting_reply":       "候補者の状況を穏やかに確認する返信",
        "replied":             "条件・環境を確認しながら面接へ誘導する返信",
        "interview_info_sent": "面接日程の調整・確認をする返信",
        "scheduled":           "面接前日リマインド・持ち物案内の返信",
    }.get(current_phase, "状況に応じた適切な返信")

    system_p = (
        "あなたは求人担当者として候補者との会話を自然に進めるアシスタントです。\n"
        f"口調: {'丁寧・ですます調' if tone == 'polite' else tone}\n"
        + (f"面接案内情報: {interview_info}\n" if interview_info else "")
        + (f"店舗PR条件: {shop_conditions}\n" if shop_conditions else "")
        + "返信文のみを出力してください。前置き・説明は不要です。"
    )
    _new_msg = (req.new_message or "").strip()
    _instr = (req.instruction or "").strip()
    user_p = (
        f"【現在フェーズ】{current_phase} → 目標: {phase_goal}\n\n"
        + (f"【会話履歴】\n{history_text}\n" if history_text else "")
        + f"【{candidate_name}の最新メッセージ】\n{_new_msg}\n\n"
        + (f"【追加指示】{_instr}\n\n" if _instr else "")
        + "返信文を作成してください。"
    )

    from api.core.llm_client import call_llm as _call_llm_rr
    generated = _call_llm_rr(
        system_prompt=system_p,
        messages=[{"role": "user", "content": user_p}],
        ai_tier="core",
        temperature=0.7,
    )

    next_phase = {
        "offer_sent":          "replied",
        "waiting_reply":       "replied",
        "replied":             "interview_info_sent",
        "interview_info_sent": "scheduled",
        "scheduled":           "scheduled",
    }.get(current_phase, current_phase)

    import datetime as _dt_rr
    _now_rr = _dt_rr.datetime.utcnow()
    new_messages = list(messages_list) + [
        {"role": "candidate", "content": _new_msg, "sent_at": _now_rr},
    ]
    db.collection("recruit_conversations").document(conversation_id).update({
        "messages": new_messages,
        "updated_at": _now_rr,
        "last_candidate_message": _new_msg,
    })

    return {
        "generated_reply": generated,
        "current_phase": current_phase,
        "suggested_next_phase": next_phase,
        "conversation_id": conversation_id,
        "candidate_name": candidate_name,
        "note": "返信を確認・編集してから送信してください",
    }


@router.patch("/recruit/conversations/{conversation_id}/phase")
def update_recruit_conversation_phase(
    conversation_id: str,
    phase: str,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    conv_doc = db.collection("recruit_conversations").document(conversation_id).get()
    if not conv_doc.exists:
        raise HTTPException(status_code=404, detail="会話スレッドが見つかりません")
    if (conv_doc.to_dict() or {}).get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="アクセス権がありません")
    valid_phases = {"offer_sent", "waiting_reply", "replied", "interview_info_sent", "scheduled", "declined"}
    if phase not in valid_phases:
        raise HTTPException(status_code=400, detail=f"無効なphase: {phase}")
    db.collection("recruit_conversations").document(conversation_id).update({
        "phase": phase, "updated_at": datetime.datetime.utcnow()
    })
    return {"ok": True, "conversation_id": conversation_id, "phase": phase}


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
    _assert_tenant_access(m, user, "他テナントのマッピングは更新できません")
    now = datetime.datetime.utcnow()
    update_data = {
        "dom_selectors": req.dom_selectors,
        "form_structure": req.form_structure,
        "last_verified_at": now,
    }
    if req.verify_selector is not None:
        update_data["verify_selector"] = req.verify_selector
    ref.update(update_data)
    try:
        updated_pages = _sync_manual_selectors_to_structure_pages(
            db,
            mapping_id,
            {**m, "dom_selectors": req.dom_selectors},
            req.dom_selectors,
            source="manual_selector_update",
        )
        if not updated_pages:
            _refresh_capability_view_for_mapping(db, mapping_id, {**m, "dom_selectors": req.dom_selectors})
    except Exception as e:
        print(f"[SELECTOR_UPDATE_STRUCTURE_SYNC_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
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
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")

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
        _err = (creds.get("error", "") if creds else "") or "認証情報が取得できませんでした"
        print(f"[LOGIN_CHECK_CREDS_BLOCKED] secret_name={secret_name!r} err_type={_err[:60]}", flush=True)
        return {
            "status":       "BLOCKED",
            "executed":     False,
            "login_success": False,
            "message":      "ログイン情報の再登録が必要です。下のフォームからID・パスワードを入力してください。",
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
            # [P21_COLLECT_ONLY] operation_candidatesはP23/P24で生成するためP21戻り値から削除済み
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


class PublicUrlPreviewRequest(BaseModel):
    url: str

@router.post("/media/preview_url")
def preview_public_url(body: PublicUrlPreviewRequest, user: dict = Depends(verify_token)):
    """
    認証不要なURLを開いてスクリーンショット+HTMLを返す（クロスメディアの取得元確認用）。
    ログイン不要なページのみ対象。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not body.url or not body.url.startswith("http"):
        raise HTTPException(status_code=400, detail="有効なURLを入力してください")

    from api.core.browser_executor import take_site_preview
    result = take_site_preview({}, target_url=body.url)

    if result.get("status") == "WAITING_EXECUTOR":
        raise HTTPException(status_code=503, detail=result.get("message", "PLAYWRIGHT無効"))
    if result.get("status") in ("BLOCKED", "FAILED"):
        raise HTTPException(status_code=400, detail=result.get("message", "URLの取得に失敗しました"))

    return {
        "screenshot_b64": result.get("screenshot_b64", ""),
        "page_html": result.get("page_html", ""),
        "current_url": result.get("current_url", body.url),
        "title": result.get("title", ""),
        "field_boxes": [],
        "login_used": False,
        "viewport": result.get("viewport", {"width": 1280, "height": 800}),
    }


class SitePreviewRequest(BaseModel):
    target_url: str = ""

@router.post("/media/map/{mapping_id}/site_preview")
def site_preview_media_mapping(mapping_id: str, body: SitePreviewRequest = None, user: dict = Depends(verify_token)):
    """
    対象サイトをPlaywrightでログイン後にスクリーンショットし、
    base64エンコードPNG + マッピング済みフィールドの位置情報を返す。
    ID/PASS・Cookieはレスポンスに絶対含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")

    from api.core.browser_executor import take_site_preview
    target_url = (body.target_url if body else "") or ""
    result = take_site_preview(m, target_url=target_url)

    if result.get("status") == "WAITING_EXECUTOR":
        raise HTTPException(status_code=503, detail=result.get("message", "PLAYWRIGHT無効"))
    if result.get("status") == "BLOCKED":
        raise HTTPException(status_code=400, detail=result.get("message", "URL未設定"))
    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail=result.get("message", "スクリーンショット取得に失敗しました"))

    return {
        "mapping_id": mapping_id,
        "screenshot_b64": result.get("screenshot_b64", ""),
        "page_html": result.get("page_html", ""),
        "current_url": result.get("current_url", ""),
        "title": result.get("title", ""),
        "field_boxes": result.get("field_boxes", []),
        "login_used": result.get("login_used", False),
        "viewport": result.get("viewport", {"width": 1280, "height": 800}),
    }


class FormSnapshotRequest(BaseModel):
    target_url: str = ""

@router.post("/media/map/{mapping_id}/form_snapshot")
def form_snapshot_media_mapping(mapping_id: str, body: FormSnapshotRequest = None, user: dict = Depends(verify_token)):
    """
    ログイン後のページの全インタラクティブ要素（input/textarea/select/button）を
    座標・ラベル付きで返す。クロスメディアの視覚的フィールドマッピングに使用。
    ID/PASS・Cookieはレスポンスに絶対含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")

    from api.core.browser_executor import take_site_preview
    target_url = (body.target_url if body else "") or ""
    result = take_site_preview(m, target_url=target_url, extract_all_form_elements=True)

    if result.get("status") == "WAITING_EXECUTOR":
        raise HTTPException(status_code=503, detail=result.get("message", "PLAYWRIGHT無効"))
    if result.get("status") == "BLOCKED":
        raise HTTPException(status_code=400, detail=result.get("message", "URL未設定"))
    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail=result.get("message", "スナップショット取得に失敗しました"))

    return {
        "mapping_id": mapping_id,
        "screenshot_b64": result.get("screenshot_b64", ""),
        "page_html": result.get("page_html", ""),
        "current_url": result.get("current_url", ""),
        "title": result.get("title", ""),
        "field_boxes": result.get("field_boxes", []),
        "form_elements": result.get("form_elements", []),
        "login_used": result.get("login_used", False),
        "viewport": result.get("viewport", {"width": 1280, "height": 800}),
    }


class FormFillRequest(BaseModel):
    target_url: str = ""
    field_values: dict = {}

@router.post("/media/map/{mapping_id}/form_fill")
def form_fill_media_mapping(mapping_id: str, body: FormFillRequest, user: dict = Depends(verify_token)):
    """
    ログイン後に target_url へ遷移し、field_values (CSSセレクタ→値) でフォームを埋めて更新ボタンをクリックする。
    ASCENDの中から直接サイトのフォームを保存する「模倣」機能。
    ID/PASS・Cookieはレスポンスに絶対含めない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")

    from api.core.browser_executor import fill_and_submit_form
    result = fill_and_submit_form(m, target_url=body.target_url or "", field_values=body.field_values or {})

    if result.get("status") == "WAITING_EXECUTOR":
        raise HTTPException(status_code=503, detail=result.get("message", "PLAYWRIGHT無効"))
    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail=result.get("message", "フォーム送信に失敗しました"))

    return {
        "mapping_id": mapping_id,
        "submit_clicked": result.get("submit_clicked", False),
        "screenshot_b64": result.get("screenshot_b64", ""),
        "current_url": result.get("current_url", ""),
        "field_errors": result.get("field_errors", []),
        "viewport": result.get("viewport", {"width": 1280, "height": 800}),
        "message": result.get("message", ""),
    }


class DomScanRequest(BaseModel):
    max_pages: int = 200
    start_url: str = ""
    include_patterns: list = Field(default_factory=list)
    exclude_patterns: list = Field(default_factory=list)
    reset_resume: bool = False

@router.post("/media/map/{mapping_id}/dialog/scan")
def dialog_scan(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    対話型マッピング: 指定ページURLを解析してステップ・候補を動的生成する。
    body: {page_url, page_name, intent?}
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    page_url  = body.get("page_url", "").strip()
    page_name = body.get("page_name", "").strip()
    intent    = body.get("intent", "").strip()
    if not page_url:
        raise HTTPException(status_code=400, detail="page_url は必須です")
    if not page_name:
        raise HTTPException(status_code=400, detail="page_name は必須です")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    if not m.get("credential_secret_name"):
        raise HTTPException(status_code=400, detail="ログイン情報が未登録です")
    if not m.get("login_url"):
        raise HTTPException(status_code=400, detail="login_urlが未設定です")
    from api.core.browser_executor import scan_page_for_mapping
    from api.core.secret_manager import get_secret_json
    creds = get_secret_json(m["credential_secret_name"])
    if not creds or creds.get("blocked"):
        raise HTTPException(status_code=400, detail=f"認証情報が無効です: {creds.get('error', '取得不可') if creds else '取得不可'}")
    result = scan_page_for_mapping(m, creds, page_url, page_name, intent=intent)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "スキャン失敗"))
    return result


@router.post("/media/map/{mapping_id}/dialog/confirm")
def dialog_confirm(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    対話型マッピング: 1項目を確定してFirestoreに保存する。
    body: {page_name, role, value, type: "url"|"selector"}
    保存先: page_mappings.{page_name}.target_url or page_mappings.{page_name}.selectors.{role}
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    page_name = body.get("page_name", "")
    role      = body.get("role", "")
    value     = body.get("value", "")
    item_type = body.get("type", "selector")
    if not all([page_name, role, value]):
        raise HTTPException(status_code=400, detail="page_name / role / value は必須です")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    import datetime as _dt_dlg
    now = _dt_dlg.datetime.utcnow()
    if item_type == "url":
        ref.update({
            f"page_mappings.{page_name}.target_url": value,
            f"page_mappings.{page_name}.confirmed_at": now,
        })
    else:
        ref.update({
            f"page_mappings.{page_name}.selectors.{role}": {
                "selector":     value,
                "source":       "human_confirmed",
                "confirmed_at": now,
            },
            f"page_mappings.{page_name}.confirmed_at": now,
        })
    print(f"[DIALOG_CONFIRM] mapping={mapping_id} page={page_name} role={role} type={item_type}", flush=True)
    return {"ok": True, "page_name": page_name, "role": role, "saved": value}


@router.post("/media/map/{mapping_id}/dialog/element_preview")
def dialog_element_preview(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    セレクター要素をハイライトしたスクリーンショットを返す。
    body: {selector, navigate_url, operation_type}
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    selector     = body.get("selector", "").strip()
    navigate_url = body.get("navigate_url", "").strip()
    if not selector:
        raise HTTPException(status_code=400, detail="selector は必須です")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    if not m.get("credential_secret_name"):
        raise HTTPException(status_code=400, detail="ログイン情報が未登録です")
    from api.core.browser_executor import preview_selector_element
    from api.core.secret_manager import get_secret_json
    creds = get_secret_json(m["credential_secret_name"])
    if not creds or creds.get("blocked"):
        raise HTTPException(status_code=400, detail=f"認証情報が無効です: {creds.get('error', '取得不可') if creds else '取得不可'}")
    result = preview_selector_element(m, creds, selector, navigate_url or m.get("login_url", ""))
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "プレビュー失敗"))
    return result


@router.post("/media/map/{mapping_id}/menu/auto_discover")
def menu_auto_discover(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    Playwrightでログインして管理トップのナビリンクを自動取得し manual_menu_items に保存する。
    body: {start_url?}  省略時は login_url を使用
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    start_url = body.get("start_url", "").strip()
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    m["mapping_id"] = mapping_id
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    if not m.get("credential_secret_name"):
        raise HTTPException(status_code=400, detail="ログイン情報が未登録です")
    if not m.get("login_url"):
        raise HTTPException(status_code=400, detail="login_urlが未設定です")
    from api.core.browser_executor import auto_discover_menu_links
    from api.core.secret_manager import get_secret_json
    creds = get_secret_json(m["credential_secret_name"])
    if not creds or creds.get("blocked"):
        raise HTTPException(status_code=400, detail=f"認証情報が無効です: {creds.get('error', '取得不可') if creds else '取得不可'}")
    result = auto_discover_menu_links(m, creds, start_url or m.get("login_url", ""))
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "自動取得失敗"))
    menu_items = result.get("items", [])
    if not menu_items:
        raise HTTPException(status_code=400, detail="ナビリンクが検出できませんでした。ログイン後の管理ページURLを指定してください。")
    import datetime as _dt_auto
    now = _dt_auto.datetime.utcnow()
    ref.update({
        "manual_menu_items": menu_items,
        "manual_menu_source": "auto_discover",
        "manual_menu_imported_at": now,
        "manual_menu_source_url": result.get("source_url", ""),
    })
    print(f"[MENU_AUTO_DISCOVER] mapping={mapping_id} items={len(menu_items)}", flush=True)
    return {"ok": True, "items_count": len(menu_items), "source_url": result.get("source_url", "")}


@router.post("/media/map/{mapping_id}/menu_items/add")
def add_menu_item(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    発見されたタブメニュー等を manual_menu_items に追記する。
    body: {absolute_url, title, category?}
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    url   = (body.get("absolute_url") or body.get("url") or "").strip()
    title = (body.get("title") or body.get("text") or url).strip()
    cat   = (body.get("category") or "その他").strip()
    if not url:
        raise HTTPException(status_code=400, detail="absolute_url は必須です")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    existing = list(m.get("manual_menu_items") or [])
    already = any(
        (item.get("absolute_url") or item.get("href") or "") == url
        for item in existing
        if isinstance(item, dict)
    )
    if already:
        return {"ok": True, "added": False, "reason": "already_exists"}
    existing.append({"absolute_url": url, "href": url, "title": title, "text": title, "category": cat})
    ref.update({"manual_menu_items": existing})
    print(f"[MENU_ITEM_ADD] mapping={mapping_id} url={url}", flush=True)
    return {"ok": True, "added": True, "url": url}


@router.patch("/media/map/{mapping_id}/menu_items/category")
def update_menu_item_categories(mapping_id: str, body: dict = Body(...), user: dict = Depends(verify_token)):
    """
    manual_menu_items 内の指定タイトルのカテゴリを一括更新する。
    body: { "updates": [{"title": "旧応募情報", "category": "随時確認推奨項目"}, ...] }
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    updates: list = body.get("updates") or []
    if not updates:
        raise HTTPException(status_code=400, detail="updates は必須です")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    update_map = {u["title"]: u["category"] for u in updates if u.get("title") and u.get("category")}
    items = list(m.get("manual_menu_items") or [])
    changed = 0
    for item in items:
        t = item.get("title") or item.get("text") or ""
        if t in update_map:
            item["category"] = update_map[t]
            changed += 1
    ref.update({"manual_menu_items": items})
    print(f"[MENU_CATEGORY_UPDATE] mapping={mapping_id} changed={changed}", flush=True)
    return {"ok": True, "changed": changed}


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
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")

    from api.core.browser_executor import run_dom_scan
    _max_pages        = body.max_pages        if body else 200
    _start_url        = body.start_url        if body else ""
    _include_patterns = body.include_patterns if body else []
    _exclude_patterns = body.exclude_patterns if body else []
    _reset_resume     = body.reset_resume     if body else False
    if _start_url:
        _assert_url_in_mapping_scope(m, _start_url, "start_url")
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
    # crawl完了時にpage_purpose付与（設計図3章・完了条件）
    if result.get("admin_crawl_completed"):
        try:
            import datetime as _dt_pp
            _db_pp = get_db()
            _mm_pp = _db_pp.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _nav_pp = (_mm_pp.get("navigation_graph") or {}).get("pages") or []
            _updated = False
            for _pg_pp in _nav_pp:
                if not isinstance(_pg_pp, dict):
                    continue
                if _pg_pp.get("page_purpose"):
                    continue
                _ev_pp = {
                    "url":                _pg_pp.get("url", ""),
                    "title":              _pg_pp.get("title", ""),
                    "headings":           [],
                    "body_text_summary":  "",
                    "inputs":             _pg_pp.get("inputs", []),
                    "textareas":          _pg_pp.get("textareas", []),
                    "buttons":            _pg_pp.get("buttons", []),
                    "file_inputs":        _pg_pp.get("file_inputs", []),
                    "forms":              _pg_pp.get("forms", []),
                    "save_candidates":    [],
                    "editor_candidates":  [],
                    "editable_dom_score": 1 if (_pg_pp.get("inputs_count", 0) > 0 or _pg_pp.get("textareas_count", 0) > 0) else 0,
                    "source_chain":       [],
                }
                _pp_result = classify_page_purpose_with_llm(_ev_pp)
                _pg_pp["page_purpose"]        = _pp_result.get("page_purpose", "unknown")
                _pg_pp["is_operation_target"] = _pp_result.get("is_operation_target", False)
                _pg_pp["page_purpose_source"] = _pp_result.get("page_purpose_source", "rule_fallback")
                _pg_pp["llm_confidence"]      = _pp_result.get("confidence", 0.0)
                _updated = True
            if _updated and _nav_pp:
                _db_pp.collection("media_mappings").document(mapping_id).update({
                    "navigation_graph.pages": _nav_pp,
                    "updated_at": _dt_pp.datetime.utcnow(),
                })
                print(f"[CRAWL_PAGE_PURPOSE_SAVED] mapping_id={mapping_id} pages={len(_nav_pp)}", flush=True)
        except Exception as _pp_err:
            print(f"[CRAWL_PAGE_PURPOSE_ERROR] {_pp_err}", flush=True)

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
        # [P21_COLLECT_ONLY] P21候補件数は削除済み。常に0。
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
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
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
    try:
        updated_pages = _sync_manual_selectors_to_structure_pages(
            db,
            mapping_id,
            {**m, "dom_selectors": new_dom},
            approved,
            source="selector_repair_apply",
        )
        if not updated_pages:
            _refresh_capability_view_for_mapping(db, mapping_id, {**m, "dom_selectors": new_dom})
    except Exception as e:
        print(f"[SELECTOR_REPAIR_STRUCTURE_SYNC_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
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
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
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
    try:
        _refresh_capability_view_for_mapping(db, mapping_id, {**m, "capabilities": new_caps})
    except Exception as e:
        print(f"[CAPABILITIES_APPLY_VIEW_REFRESH_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
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
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
    current_caps = m.get("capabilities") or {}
    new_caps = dict(current_caps)
    for k, v in req.capabilities.items():
        new_caps[k] = bool(v)
    import datetime as _dt
    ref.update({
        "capabilities": new_caps,
        "last_capabilities_updated_at": _dt.datetime.utcnow(),
    })
    try:
        _refresh_capability_view_for_mapping(db, mapping_id, {**m, "capabilities": new_caps})
    except Exception as e:
        print(f"[CAPABILITIES_DIRECT_VIEW_REFRESH_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
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
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
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
    try:
        applied_selector_map = {key: new_dom.get(key) for key in applied_keys if key in new_dom}
        updated_pages = _sync_manual_selectors_to_structure_pages(
            db,
            mapping_id,
            {**m, "dom_selectors": new_dom},
            applied_selector_map,
            source="semantic_selector_apply",
        )
        if not updated_pages:
            _refresh_capability_view_for_mapping(db, mapping_id, {**m, "dom_selectors": new_dom})
    except Exception as e:
        print(f"[SEMANTIC_SELECTOR_STRUCTURE_SYNC_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
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
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ref = db.collection("media_mappings").document(mapping_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="マッピングが見つかりません")
    m = doc.to_dict()
    _assert_tenant_access(m, user, "他テナントのマッピングは操作できません")
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
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    import datetime as _dt
    from api.core.browser_executor import (
        rank_selector_candidates,
        save_selector_ranking_result,
    )
    from api.core.firestore_client import get_db

    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

    # template取得: agent_templates優先、fallbackにcross_media_templates
    tmpl_doc = db.collection("agent_templates").document(template_id).get()
    if not tmpl_doc.exists:
        tmpl_doc = db.collection("cross_media_templates").document(template_id).get()
    if not tmpl_doc.exists:
        raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
    tmpl = tmpl_doc.to_dict()
    tmpl_tenant = str(tmpl.get("tenant_id") or "")
    tmpl_scope = str(tmpl.get("scope") or tmpl.get("visibility") or "").lower()
    if tmpl_tenant and tmpl_tenant != tenant_id and tmpl_tenant not in ("global", "public") and tmpl_scope not in ("global", "public") and not _is_agent_admin(user):
        raise HTTPException(status_code=403, detail="このテンプレートへのアクセス権がありません")

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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
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
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")

    mapping = doc.to_dict()
    mapping["id"] = mapping_id

    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    valid_ops = [
        "news_post", "text_update", "media_replace",
        "schedule_update", "price_update", "entity_register", "entity_update", "status_update",
    ]
    if operation_type not in valid_ops:
        raise HTTPException(status_code=400, detail=f"未対応operation_type: {operation_type}")

    # deep_scan 実行
    import datetime as _dt
    now = _dt.datetime.utcnow()

    # AI整備済みのoperationだけは自動解析で上書きしない
    try:
        _pre_snap = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _pre_op = (_pre_snap.get("operation_mappings") or {}).get(operation_type) or {}
        if _operation_mapping_is_production_ready(_pre_op):
            print(f"[P23_AI_READY_PROTECT] op={operation_type} mapping_id={mapping_id} AI整備済みのためスキップ", flush=True)
            return {"ok": True, "skipped": True, "reason": "AI整備済みのため自動解析をスキップしました。再整備は媒体基盤のAI整備から実行してください。"}
    except Exception as _pre_e:
        print(f"[P23_MANUAL_CHECK_ERROR] {type(_pre_e).__name__}", flush=True)

    # 実行前にSCANNING状態を書き込み（ERROR残骸を上書き）
    try:
        db.collection("media_mappings").document(mapping_id).update({
            f"operation_mappings.{operation_type}.status": "SCANNING",
            f"operation_mappings.{operation_type}.last_scanned_at": now.isoformat(),
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
        # [LEGACY] operation_candidates_detail: legacy name。実体はoperation_detail_by_type用補助情報。operation_candidates依存ではない。
        _detail_list_p23 = _existing_mm_p23.get("operation_candidates_detail") or []
        _detail_url_p23 = next((d.get("source_url","") for d in _detail_list_p23 if isinstance(d,dict) and d.get("operation_type")==operation_type), "")
        if _existing_op_p23.get("p24_source_url") and _existing_op_p23.get("p24_source_url") == _detail_url_p23:
            result["p24_source_url"]    = _existing_op_p23["p24_source_url"]
            result["p24_confidence"]    = _existing_op_p23.get("p24_confidence", 0.0)
            result["p24_classified_at"] = _existing_op_p23.get("p24_classified_at")
            result["target_url"]        = _existing_op_p23["p24_source_url"]
        # AI整備済みは自動スキャンで上書きしない。
        if _operation_mapping_is_production_ready(_existing_op_p23) and _existing_op_p23.get("target_url"):
            result["target_url"]    = _existing_op_p23["target_url"]
            result["source"]        = _existing_op_p23.get("source") or "AI_CONFIRMED"
            if _existing_op_p23.get("fields"):
                result["fields"] = _existing_op_p23["fields"]
            if _existing_op_p23.get("save_selector"):
                result["save_selector"] = _existing_op_p23["save_selector"]
            if _existing_op_p23.get("form_action"):
                result["form_action"] = _existing_op_p23["form_action"]
            result["status"] = "READY"
            result["executable"] = _existing_op_p23.get("executable", True)
            result["production_ready"] = True
            result["confirmation_status"] = _existing_op_p23.get("confirmation_status") or "AI_CONFIRMED"
            print(f"[P23_AI_READY_PROTECT] op={operation_type} AI整備済みURLを保持: {_existing_op_p23.get('target_url')}", flush=True)
        result = _normalize_operation_status(result)
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
        # [P23_P24] operation_mappingsのキーをop_candidatesとして使う
        _op_cands    = list((_latest_mm.get("operation_mappings") or {}).keys())
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

    schema = _rebuild_media_schema_for_mapping(db, mapping_id)

    return {
        "mapping_id":     mapping_id,
        "operation_type": operation_type,
        "result":         result,
        "media_schema_summary": {
            "forms_count": schema.get("forms_count", 0),
            "entities_count": schema.get("entities_count", 0),
            "canonical_fields_count": schema.get("canonical_fields_count", 0),
        },
    }

def classify_page_purpose_with_llm(page_evidence: dict, operation_type: str = None) -> dict:
    """
    設計図3章: LLMによるページ目的判定。
    browser_executor.pyから受け取ったpage_evidenceをLLMに渡しpage_purposeを返す。
    LLM失敗時はrule_fallbackで継続する。
    """
    _PURPOSE_CANDIDATES = [
        "dashboard", "global_navigation", "external_link_setting", "media_library",
        "list_page", "edit_page", "create_page", "news_post_page", "text_edit_page",
        "entity_list_page", "entity_edit_page", "media_upload_page", "schedule_page",
        "price_page", "settings_page", "unknown",
    ]
    _OPERATION_TARGET_PURPOSES = {
        "edit_page", "create_page", "news_post_page", "text_edit_page",
        "entity_edit_page", "media_upload_page", "schedule_page", "price_page",
    }

    # ── rule_fallback（LLM失敗時・または高速判定用） ──
    def _rule_fallback(ev: dict) -> dict:
        url   = (ev.get("url") or "").lower()
        title = (ev.get("title") or "").lower()
        editable_score = ev.get("editable_dom_score", 0)
        save_cands = ev.get("save_candidates", [])
        purpose = "unknown"
        is_target = False
        if any(k in url or k in title for k in ("login", "signin", "ログイン")):
            purpose = "global_navigation"
        elif any(k in url or k in title for k in ("dashboard", "home", "top", "c1main")):
            purpose = "dashboard"
        elif any(k in url or k in title for k in ("news", "blog", "post", "topics", "投稿", "記事")):
            purpose = "news_post_page"
            is_target = editable_score > 0 or bool(save_cands)
        elif any(k in url or k in title for k in ("schedule", "shift", "スケジュール", "出勤")):
            purpose = "schedule_page"
            is_target = editable_score > 0 or bool(save_cands)
        elif any(k in url or k in title for k in ("price", "fee", "料金", "course")):
            purpose = "price_page"
            is_target = editable_score > 0 or bool(save_cands)
        elif any(k in url or k in title for k in ("cast", "staff", "member", "girl", "キャスト", "スタッフ")):
            if any(k in url for k in ("edit", "new", "create", "regist", "form")):
                purpose = "entity_edit_page"
                is_target = editable_score > 0 or bool(save_cands)
            else:
                purpose = "entity_list_page"
        elif any(k in url for k in ("edit", "new", "create", "regist", "form", "add")):
            purpose = "edit_page"
            is_target = editable_score > 0 or bool(save_cands)
        elif any(k in url or k in title for k in ("list", "一覧")):
            purpose = "list_page"
        elif any(k in url or k in title for k in ("media", "image", "photo", "upload", "画像")):
            purpose = "media_upload_page"
            is_target = editable_score > 0 or bool(save_cands)
        elif editable_score > 0 and save_cands:
            purpose = "edit_page"
            is_target = True
        return {
            "page_purpose":            purpose,
            "is_operation_target":     is_target,
            "operation_type_candidates": [],
            "confidence":              0.5,
            "reason":                  "rule_fallback",
            "negative_reasons":        [],
            "page_purpose_source":     "rule_fallback",
        }

    # ── LLM判定 ──
    try:
        from api.core.llm_client import call_llm_json
        _url   = page_evidence.get("url", "")
        _title = page_evidence.get("title", "")
        _headings = page_evidence.get("headings", [])
        _body  = (page_evidence.get("body_text_summary") or "")[:300]
        _inputs = [i.get("name","") + "/" + i.get("type","") for i in (page_evidence.get("inputs") or [])[:10]]
        _buttons = [(b.get("text") or b.get("value") or "") for b in (page_evidence.get("buttons") or [])[:10]]
        _save_cands = page_evidence.get("save_candidates", [])
        _editable_score = page_evidence.get("editable_dom_score", 0)
        _has_textarea = bool(page_evidence.get("textareas"))
        _has_file = bool(page_evidence.get("file_inputs"))
        _has_editor = bool(page_evidence.get("editor_candidates"))
        _op_hint = f" 操作タイプヒント: {operation_type}" if operation_type else ""
        prompt = f"""以下のページ情報からページ目的を判定してください。{_op_hint}

URL: {_url}
タイトル: {_title}
見出し: {_headings}
ボディ要約: {_body}
input要素: {_inputs}
ボタン: {_buttons}
保存候補: {_save_cands}
テキストエリア: {_has_textarea}
ファイル入力: {_has_file}
エディタ候補: {_has_editor}
編集可能DOMスコア: {_editable_score}

以下のJSONのみを返してください（説明不要）:
{{
  "page_purpose": "<以下から1つ選択: dashboard|global_navigation|external_link_setting|media_library|list_page|edit_page|create_page|news_post_page|text_edit_page|entity_list_page|entity_edit_page|media_upload_page|schedule_page|price_page|settings_page|unknown>",
  "is_operation_target": <true/false>,
  "operation_type_candidates": [],
  "confidence": <0.0-1.0>,
  "reason": "<判定理由>",
  "negative_reasons": []
}}
"""
        result = call_llm_json(prompt)
        if not isinstance(result, dict):
            raise ValueError("LLM result is not dict")
        purpose = result.get("page_purpose", "unknown")
        if purpose not in _PURPOSE_CANDIDATES:
            purpose = "unknown"
        result["page_purpose"] = purpose
        result["is_operation_target"] = bool(result.get("is_operation_target", False))
        result["page_purpose_source"] = "llm"
        print(f"[PAGE_PURPOSE_LLM] url={_url[:60]} purpose={purpose} is_target={result['is_operation_target']} confidence={result.get('confidence',0)}", flush=True)
        return result
    except Exception as _e_llm:
        print(f"[PAGE_PURPOSE_LLM_FALLBACK] error={_e_llm}", flush=True)
        fb = _rule_fallback(page_evidence)
        print(f"[PAGE_PURPOSE_RULE_FALLBACK] url={page_evidence.get('url','')[:60]} purpose={fb['page_purpose']}", flush=True)
        return fb


def _normalize_op_status_legacy(status: str) -> str:
    """既存Firestoreデータの旧statusを新4状態に互換変換する（読み込み時のみ使用・新規保存には使わない）"""
    _legacy_map = {
        "NEEDS_MAPPING":  "UNDISCOVERED",
        "PARTIAL":        "NEEDS_REVIEW",
        "BLOCKED":        "FAILED",
        "UNKNOWN":        "UNDISCOVERED",
        "WAITING_EXECUTOR": "UNDISCOVERED",
    }
    return _legacy_map.get(status, status)
def _normalize_operation_status(result: dict) -> dict:
    """保存直前にstatus正規化を必ず通す（指示書11番）"""
    old = result.get("status", "")
    selectors = result.get("selectors") or {}
    target_url = result.get("target_url")
    error_reason = result.get("error_reason", "")
    if old == "PARTIAL":
        result["status"] = "NEEDS_REVIEW"
        result["executable"] = False
        result["human_review_required"] = True
    elif old == "NEEDS_MAPPING":
        if target_url or selectors:
            result["status"] = "NEEDS_REVIEW"
            result["executable"] = False
            result["human_review_required"] = True
        elif "timeout" in str(error_reason).lower():
            result["status"] = "FAILED"
            result["executable"] = False
            result["human_review_required"] = False
        else:
            result["status"] = "UNDISCOVERED"
            result["executable"] = False
            result["human_review_required"] = False
    elif old not in ("READY", "NEEDS_REVIEW", "FAILED", "UNDISCOVERED"):
        result["status"] = "FAILED"
        result["executable"] = False
        result["human_review_required"] = False
        result["error_reason"] = (result.get("error_reason") or "") + f" invalid_status:{old}"
    if old != result.get("status"):
        print(f"[OP_MAPPING_STATUS_NORMALIZED] old_status={old} new_status={result.get('status')}", flush=True)
    return result



def _sync_ready_operation_steps(mapping_id: str, db) -> None:
    """AI整備済みopだけ operation_steps_by_type に同期する（候補のsteps化禁止）。"""
    try:
        import datetime as _dt_sync
        from api.core.browser_executor import rebuild_operation_steps
        _doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _structure_view = _refresh_capability_view_for_mapping(db, mapping_id, _doc)
        if _structure_view:
            _ready_ops_struct = [
                op for op, row in ((_structure_view.get("operations") or {}).items())
                if isinstance(row, dict) and row.get("status") == "READY" and row.get("production_ready") is True
            ]
            print(f"[P24_7_STEPS_SYNC] source=structure_pages mapping_id={mapping_id} ready_count={len(_ready_ops_struct)}", flush=True)
            return
        _op_maps = _doc.get("operation_mappings", {})
        _nav = _doc.get("navigation_graph", {})
        _dlist = _doc.get("operation_candidates_detail") or []
        _det = {d["operation_type"]: d for d in _dlist if isinstance(d, dict) and d.get("operation_type")}
        _ready_ops = [
            op for op, m in _op_maps.items()
            if isinstance(m, dict)
            and _operation_mapping_is_production_ready(m)
            and m.get("target_url")
        ]
        _steps = rebuild_operation_steps(_ready_ops, _nav, _op_maps, _det) if _ready_ops else {}
        db.collection("media_mappings").document(mapping_id).update({
            "operation_steps_by_type": _steps,
            "updated_at": _dt_sync.datetime.utcnow(),
        })
        print(f"[P24_7_STEPS_SYNC] mapping_id={mapping_id} ready_count={len(_ready_ops)} steps={list(_steps.keys())}", flush=True)
        # [P23_P24] operation_candidatesではなくoperation_mappingsキーを正規とする
        _op_keys_sync = [k for k in _op_maps.keys() if k and k != "admin_crawl"]
        _step_keys_sync = [k for k in _steps.keys() if k and k != "admin_crawl"]
        _norm_cands = list(dict.fromkeys(_op_keys_sync + _step_keys_sync))
        print(f"[P24_CANDIDATE_NORMALIZE] mapping_id={mapping_id} candidates={_norm_cands}", flush=True)
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
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict()
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")
    _guard_mapping_scan_not_running(db, mapping_id, mapping, "multi_deep_scan")

    # バックグラウンドで処理実行・即時返却
    background_tasks.add_task(_run_multi_deep_scan_bg, mapping_id, mapping, tenant_id)
    return {"ok": True, "status": "processing", "mapping_id": mapping_id}


def _run_multi_deep_scan_bg(mapping_id: str, mapping: dict, tenant_id: str):
    """multi_deep_scanのバックグラウンド処理本体"""
    print(f"[BG_TASK_START] mapping_id={mapping_id}", flush=True)
    db = get_db()
    import datetime as _dt47
    # BGタスク開始: scan_progress.status=RUNNINGを書き込み
    try:
        db.collection("media_mappings").document(mapping_id).update({
            "scan_progress.status": "RUNNING",
            "scan_progress.kind": "multi_deep_scan",
            "scan_progress.started_at": _dt47.datetime.utcnow().isoformat() if "datetime" in dir(_dt47) else "",
            "scan_progress.updated_at": _dt47.datetime.utcnow().isoformat() if "datetime" in dir(_dt47) else "",
        })
    except Exception as _e_prog_start:
        print(f"[BG_TASK_PROGRESS_START_ERROR] {_e_prog_start}", flush=True)
    valid_ops = [
        "news_post", "text_update", "media_replace",
        "schedule_update", "price_update", "entity_register", "entity_update", "status_update",
    ]
    # required_fields定義（未解析保存用）
    _required_map = {
        "news_post":       ["title", "body", "save"],
        "text_update":     ["body", "save"],
        "media_replace":   ["file", "save"],
        "schedule_update": ["save"],
        "price_update":    ["price", "save"],
        "entity_register": ["required_inputs", "save"],
        "status_update":   ["body", "save"],
        "entity_update":   ["editable_inputs", "save"],
    }
    # [P23_P24] operation_candidatesには依存しない。operation_mappingsキーまたはvalid_opsを基準に対象を決定する。
    # [P23_P24] operation_mappingsベースで全valid_opsを試す。operation_candidatesに依存しない。
    op_candidates = list(valid_ops)
    # [MENU_DOM_SCAN] Playwrightスキャンはdom_scanエンドポイントで実施（multi_deep_scanから除外）
    now47 = _dt47.datetime.utcnow()
    _start_time47 = _dt47.datetime.utcnow()
    print(f"[P24_7_MULTI_SCAN_START] mapping_id={mapping_id} operations_count={len(op_candidates)}", flush=True)

    results = {}
    ready_ops = []
    review_ops = []
    waiting_ops = []
    failed_ops = []
    undiscovered_ops = []

    # ── [PRE_LOOP] manual_importページにpage_purposeを一括付与（ループ前） ──
    try:
        _pre_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _pre_pages = (_pre_doc.get("navigation_graph") or {}).get("pages") or []
        _MI_PURPOSE_PRE = [
            (["news","post","blog","diary","topic","topics","event","coupon","お知らせ","ニュース","投稿","写メ","日記","freetext","contents","campaign","realtime","marquee","速報"], "news_post_page"),
            (["profile","about","text","プロフィール","自己紹介","説明","紹介文","freetext","contents","con_txt","seo","concept","フリー","編集"], "text_edit_page"),
            (["photo","image","gallery","写真","画像","メディア","upload","アップロード"], "media_upload_page"),
            (["schedule","shift","出勤","予定","calendar","シフト","カレンダー"], "schedule_page"),
            (["price","course","料金","コース","fee","システム"], "price_page"),
            (["register","new","add","登録","新規","追加","cast","staff","キャスト","スタッフ"], "create_page"),
            (["edit","update","編集","更新","cast","staff","キャスト","スタッフ","profile","プロフィール"], "edit_page"),
            (["status","public","private","表示","非表示","公開","停止","有効","無効","ステータス","state","visible","hidden","active","inactive","enabled","disabled","standby","girl","cast"], "status_page"),
        ]
        _pre_updated = False
        for _pre_pg in _pre_pages:
            if not isinstance(_pre_pg, dict):
                continue
            if not _pre_pg.get("manual_import"):
                continue
            if _pre_pg.get("page_purpose") and _pre_pg.get("page_purpose") != "unknown":
                continue
            _mi_url2   = (_pre_pg.get("url") or "").lower()
            _mi_title2 = (_pre_pg.get("title") or "").lower()
            _mi_cat2   = (_pre_pg.get("category") or "").lower()
            _mi_text2  = _mi_url2 + " " + _mi_title2 + " " + _mi_cat2
            _mi_purpose2 = "unknown"
            for _kws2, _purp2 in _MI_PURPOSE_PRE:
                if any(k in _mi_text2 for k in _kws2):
                    _mi_purpose2 = _purp2
                    break
            _pre_pg["page_purpose"]        = _mi_purpose2
            _pre_pg["page_purpose_source"] = "manual_import_keyword"
            _pre_pg["is_operation_target"] = _mi_purpose2 != "unknown"
            _pre_updated = True
        if _pre_updated and _pre_pages:
            db.collection("media_mappings").document(mapping_id).update({
                "navigation_graph.pages": _pre_pages,
            })
            print(f"[PRE_LOOP_PAGE_PURPOSE] mapping_id={mapping_id} pages={len(_pre_pages)} updated", flush=True)
    except Exception as _e_pre:
        print(f"[PRE_LOOP_PAGE_PURPOSE_ERROR] {_e_pre}", flush=True)
    from api.core.browser_executor import deep_scan_operation, rebuild_operation_steps

    for op in op_candidates:
        _required = _required_map.get(op, ["save"])
        print(f"[P24_7_OP_SCAN_START] operation_type={op}", flush=True)
        # ── 続きから再開: READY済みopはスキップ ──
        _resume_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _resume_op = _resume_doc.get("operation_mappings", {}).get(op, {})
        _resume_status = _normalize_op_status_legacy(_resume_op.get("status", ""))
        # AI整備済みopだけ自動解析でスキップ。旧MANUAL/旧READYは再整備対象。
        if _operation_mapping_is_production_ready(_resume_op):
            print(f"[P24_7_SKIP_AI_READY] op={op} AI整備済みのためスキップ", flush=True)
            results[op] = _resume_op
            ready_ops.append(op)
            continue
        # ── 進捗をFirestoreに書き込み（フロントポーリング用） ──
        _total_ops = len(op_candidates)
        _done_ops = len(ready_ops) + len(review_ops) + len(waiting_ops) + len(failed_ops) + len(undiscovered_ops)
        try:
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress": {
                    "status": "RUNNING",
                    "kind": "multi_deep_scan",
                    "current_op": op,
                    "done": _done_ops,
                    "total": _total_ops,
                    "updated_at": _dt47.datetime.utcnow().isoformat(),
                }
            })
        except Exception as _e_prog:
            print(f"[P24_7_PROGRESS_WRITE_ERROR] {_e_prog}", flush=True)
        _elapsed47 = (_dt47.datetime.utcnow() - _start_time47).total_seconds()
        if _elapsed47 > 240:
            print(f"[P24_7_TIMEOUT_BREAK] elapsed={_elapsed47:.1f}s op={op} 強制保存して終了", flush=True)
            # タイムアウト時: 残り未処理opのSCANNING状態を既存ステータスに戻す
            _remaining_ops = op_candidates[op_candidates.index(op):]
            for _rem_op in _remaining_ops:
                try:
                    _rem_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                    _rem_cur = _rem_doc.get("operation_mappings", {}).get(_rem_op, {})
                    _rem_status = _rem_cur.get("status", "")
                    if _rem_status == "SCANNING":
                        _rem_prev = _rem_cur.get("prev_status") or "NEEDS_REVIEW"
                        db.collection("media_mappings").document(mapping_id).update({
                            f"operation_mappings.{_rem_op}.status": _rem_prev,
                            f"operation_mappings.{_rem_op}.last_scanned_at": _dt47.datetime.utcnow().isoformat(),
                        })
                        print(f"[P24_7_TIMEOUT_RESTORE] op={_rem_op} status={_rem_prev}", flush=True)
                except Exception as _e_restore:
                    print(f"[P24_7_TIMEOUT_RESTORE_ERROR] op={_rem_op} {_e_restore}", flush=True)
            break
        now47a = _dt47.datetime.utcnow()

        # SCANNING状態を書き込み
        try:
            db.collection("media_mappings").document(mapping_id).update({
                f"operation_mappings.{op}.status": "SCANNING",
                f"operation_mappings.{op}.last_scanned_at": now47a.isoformat(),
            })
        except Exception as _e_scan:
            print(f"[P24_7_SCANNING_WRITE_ERROR] op={op} {_e_scan}", flush=True)

        # 最新mappingを取得して渡す
        _cur_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _cur_doc["id"] = mapping_id
        _cur_op = _cur_doc.get("operation_mappings", {}).get(op, {})
        _cur_status = _normalize_op_status_legacy(_cur_op.get("status", ""))

        # ── entity_update list系URL事前ガード ──
        _eu_list_patterns  = ["cast_list", "readlog", "review_list", "/list", "price", "fee", "course", "pricelist", "料金", "料金表", "systemlist", "multifee"]
        _eu_allow_patterns = ["cast_edit", "edit", "regist", "form", "profile_edit"]
        _eu_src_url = (_cur_op.get("p24_source_url") or _cur_op.get("target_url") or "").lower()
        if op == "entity_update" and _eu_src_url:
            _eu_is_list  = any(p in _eu_src_url for p in _eu_list_patterns)
            _eu_is_allow = any(p in _eu_src_url for p in _eu_allow_patterns)
            if _eu_is_list and not _eu_is_allow:
                print(f"[P24_7_ENTITY_UPDATE_LIST_GUARD] op={op} src_url={_eu_src_url} -> FAILED", flush=True)
                result = {
                    "status":           "FAILED",
                    "missing":          ["edit_trigger", "save"],
                    "selectors":        {},
                    "target_url":       None,
                    "validation_score": 0,
                    "error_reason":     "list page excluded for entity_update",
                    "executable":       False,
                    "last_scanned_at":  _dt47.datetime.utcnow().isoformat(),
                }
                print(f"[P24_7_FINAL_SAVE_GUARD] op={op} status=FAILED", flush=True)
                result = _normalize_operation_status(result)
                db.collection("media_mappings").document(mapping_id).update({
                    f"operation_mappings.{op}": result,
                    "updated_at": _dt47.datetime.utcnow(),
                })
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status=FAILED missing={result['missing']}", flush=True)
                _sync_ready_operation_steps(mapping_id, db)
                results[op] = result
                failed_ops.append(op)
                continue
        # ── P21/P23貫通: target_url判明済みmanual_importページのDOM個別スキャン ──
        _op_target_url = _cur_op.get("target_url") or ""
        # [P24_MANUAL_IMPORT_FALLBACK] target_url未設定時: manual_importページからpage_purpose一致のURLを探してDOMスキャン
        if not _op_target_url:
            _PURPOSE_MAP = {
                "news_post":       ("news_post_page",),
                "text_update":     ("text_edit_page", "edit_page"),
                "price_update":    ("price_page",),
                "schedule_update": ("schedule_page",),
                "media_replace":   ("media_upload_page",),
                "entity_register": ("create_page",),
                "status_update":  ("status_page",),
                "entity_update":   ("edit_page", "entity_edit_page"),
            }
            _target_purposes = _PURPOSE_MAP.get(op, ())
            _nav_pages_fb = (_cur_doc.get("navigation_graph") or {}).get("pages") or []
            for _fb_pg in _nav_pages_fb:
                if (_fb_pg.get("manual_import") and
                    _fb_pg.get("page_purpose") in _target_purposes and
                    not (_fb_pg.get("forms") or _fb_pg.get("inputs") or _fb_pg.get("buttons"))):
                    _op_target_url = _fb_pg.get("url") or ""
                    print(f"[P24_MANUAL_IMPORT_FALLBACK] op={op} found_url={_op_target_url[:60]}", flush=True)
                    break
        if _op_target_url:
            _nav_pages_now = (_cur_doc.get("navigation_graph") or {}).get("pages") or []
            _target_pg_now = next((p for p in _nav_pages_now if p.get("url") == _op_target_url), None)
            # navigation_graph URL直接キーも確認（manual_menu経由で保存されたDOM）
            _nav_graph_direct = (_cur_doc.get("navigation_graph") or {}).get(_op_target_url) or {}
            _has_dom_now = bool(
                (_target_pg_now and (_target_pg_now.get("forms") or _target_pg_now.get("inputs") or _target_pg_now.get("buttons")))
                or (_nav_graph_direct.get("forms_count", 0) > 0 or _nav_graph_direct.get("inputs_count", 0) > 0)
            )
            if not _has_dom_now:
                print(f"[P24_7_INDIVIDUAL_DOM_SCAN] op={op} target_url={_op_target_url[:60]}", flush=True)
                try:
                    from api.core.browser_executor import fetch_dom_for_url as _fetch_dom47
                    _dom_result = _fetch_dom47(_cur_doc, _op_target_url)
                    print(f"[P24_7_INDIVIDUAL_DOM_SCAN_DONE] op={op} status={_dom_result.get('status')}", flush=True)
                    # navigation_graphを再読込してcur_docを更新
                    _cur_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                    _cur_doc["id"] = mapping_id
                except Exception as _e_ids:
                    print(f"[P24_7_INDIVIDUAL_DOM_SCAN_ERROR] op={op} {_e_ids}", flush=True)
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
                "status":           "FAILED",
                "missing":          _required,
                "selectors":        {},
                "target_url":       None,
                "validation_score": 0,
                "error_reason":     "deep_scan timeout",
                "executable":       False,
                "last_scanned_at":  _now47b_to.isoformat(),
            }
            print(f"[P24_7_FINAL_SAVE_GUARD] op={op} status=FAILED reason=timeout", flush=True)
            try:
                result = _normalize_operation_status(result)
                db.collection("media_mappings").document(mapping_id).update({
                    f"operation_mappings.{op}": result,
                    "updated_at": _now47b_to,
                })
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={op} status=FAILED missing={_required}", flush=True)
            except Exception as _e_to_save:
                print(f"[P24_7_TIMEOUT_SAVE_ERROR] op={op} {_e_to_save}", flush=True)
            _sync_ready_operation_steps(mapping_id, db)
            results[op] = result
            failed_ops.append(op)
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
                result = _normalize_operation_status(result)
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
        # ── LLMページ目的判定（設計図3章）──
        try:
            _nav_pages = (_cur_doc.get("navigation_graph") or {}).get("pages") or []
            _target_url_for_llm = result.get("target_url") or ""
            _llm_page_updates = {}
            for _pg in _nav_pages:
                if not isinstance(_pg, dict):
                    continue
                if _pg.get("page_purpose"):
                    continue  # 既に判定済みはスキップ
                # [MANUAL_IMPORT_SKIP_LLM] manual_importページはURLキーワードでpage_purpose付与
                if _pg.get("manual_import"):
                    _mi_url2 = (_pg.get("url") or "").lower()
                    _mi_title2 = (_pg.get("title") or "").lower()
                    _mi_cat2 = (_pg.get("category") or "").lower()
                    _mi_text2 = _mi_url2 + " " + _mi_title2 + " " + _mi_cat2
                    _MI_PURPOSE = [
                        (["news","post","blog","diary","topic","event","coupon","お知らせ","ニュース","投稿","写メ","日記"], "news_post_page"),
                        (["profile","about","text","プロフィール","自己紹介","説明","紹介文"], "text_edit_page"),
                        (["photo","image","gallery","写真","画像","upload","アップロード"], "media_upload_page"),
                        (["schedule","shift","出勤","予定","calendar","シフト"], "schedule_page"),
                        (["price","course","料金","コース","fee","システム"], "price_page"),
                        (["register","new","add","登録","新規","追加","cast","staff","キャスト","スタッフ"], "create_page"),
                        (["edit","update","編集","更新"], "edit_page"),
                        (["status","public","private","表示","非表示","公開","停止","有効","無効","ステータス","state","visible","hidden","active","inactive","standby","girl","cast"], "status_page"),
                    ]
                    _mi_purpose2 = "unknown"
                    for _kws2, _purp2 in _MI_PURPOSE:
                        if any(k in _mi_text2 for k in _kws2):
                            _mi_purpose2 = _purp2
                            break
                    _pg["page_purpose"] = _mi_purpose2
                    _pg["page_purpose_source"] = "manual_import_keyword"
                    _pg["is_operation_target"] = _mi_purpose2 != "unknown"
                    continue
                _pg_evidence = {
                    "title":              _pg.get("title", ""),
                    "headings":           [],
                    "body_text_summary":  "",
                    "inputs":             _pg.get("inputs", []),
                    "textareas":          _pg.get("textareas", []),
                    "buttons":            _pg.get("buttons", []),
                    "file_inputs":        _pg.get("file_inputs", []),
                    "forms":              _pg.get("forms", []),
                    "save_candidates":    [],
                    "editor_candidates":  [],
                    "editable_dom_score": 1 if (_pg.get("inputs_count",0) > 0 or _pg.get("textareas_count",0) > 0) else 0,
                    "source_chain":       [],
                }
                _llm_result = classify_page_purpose_with_llm(_pg_evidence, operation_type=op)
                _pg["page_purpose"]        = _llm_result.get("page_purpose", "unknown")
                _pg["is_operation_target"] = _llm_result.get("is_operation_target", False)
                _pg["page_purpose_source"] = _llm_result.get("page_purpose_source", "rule_fallback")
                _pg["llm_confidence"]      = _llm_result.get("confidence", 0.0)
                _pg["llm_reason"]          = _llm_result.get("reason", "")
            # navigation_graph.pages を更新保存
            if _nav_pages:
                db.collection("media_mappings").document(mapping_id).update({
                    "navigation_graph.pages": _nav_pages,
                })
                print(f"[PAGE_PURPOSE_SAVED] op={op} pages={len(_nav_pages)}", flush=True)
            # target_url の LLM検証: is_operation_target=False なら NEEDS_REVIEW に降格
            if _target_url_for_llm:
                _target_pg = next((p for p in _nav_pages if p.get("url") == _target_url_for_llm), None)
                if _target_pg and not _target_pg.get("is_operation_target", True):
                    if result.get("status") == "READY":
                        result["status"] = "NEEDS_REVIEW"
                        result["executable"] = False
                        result["error_reason"] = (result.get("error_reason") or "") + " llm_target_not_confirmed"
                        print(f"[PAGE_PURPOSE_TARGET_DOWNGRADE] op={op} url={_target_url_for_llm[:60]} -> NEEDS_REVIEW", flush=True)
        except Exception as _e_llm_pg:
            print(f"[PAGE_PURPOSE_LLM_ERROR] op={op} {_e_llm_pg}", flush=True)

        # selectors/missing/target_urlの保証
        if "selectors" not in result:
            result["selectors"] = {}
        if "missing" not in result:
            result["missing"] = _required
        if "target_url" not in result:
            result["target_url"] = None
        if "error_reason" not in result and result.get("error"):
            result["error_reason"] = result.get("error", "")

        # READY→NEEDS_REVIEW/WAITING_EXECUTOR downgrade禁止
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
                result["status"] = "NEEDS_REVIEW"
                result["executable"] = False
                if not _r_missing and _has_missing_in_reason:
                    result["missing"] = ["unknown_missing_field"]
                print(f"[STATUS_NORMALIZE] op={op} READY->NEEDS_REVIEW missing={result['missing']} target_url={_r_target_url} reason={_r_error_reason[:80]}", flush=True)
        # ── status正規化ここまで ───────────────────────────────────────────
        _new_status = result.get("status", "")
        if _cur_status == "READY" and _new_status in ("NEEDS_REVIEW", "WAITING_EXECUTOR", "FAILED", "UNDISCOVERED"):
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
                and op != "media_replace"
            )
            if _save_missing or _save_score < 70 or _invalid_url or _list_like:
                result["status"] = "NEEDS_REVIEW"
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
            result = _normalize_operation_status(result)
            db.collection("media_mappings").document(mapping_id).update({
                f"operation_mappings.{op}": result,
                "updated_at": now47b,
            })
            try:
                if result.get("target_url") and isinstance(result.get("form_schema"), dict) and result["form_schema"].get("fields"):
                    _write_structure_pages(
                        db,
                        mapping_id,
                        [{
                            "url": result.get("target_url"),
                            "title": (result.get("form_schema") or {}).get("title") or op,
                            "form_schema": result.get("form_schema"),
                            "page_purpose": op,
                            "page_purpose_source": "multi_deep_scan_operation",
                        }],
                        source="multi_deep_scan_operation",
                    )
            except Exception as _e_structure_op:
                print(f"[STRUCTURE_PAGE_FROM_OP_ERROR] op={op} {type(_e_structure_op).__name__}", flush=True)
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
        elif _st in ("NEEDS_REVIEW",):
            review_ops.append(op)
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
                    "status": "UNDISCOVERED",
                    "missing": _required,
                    "selectors": {},
                    "target_url": None,
                    "error_reason": "multi_deep_scan did not produce result",
                    "last_scanned_at": _now_ensure,
                    "executable": False,
                }
                _ensure_update[f"operation_mappings.{_op}"] = _normalize_operation_status(_ensure_update[f"operation_mappings.{_op}"])
                undiscovered_ops.append(_op)
                print(f"[P24_7_OP_MAPPING_ENSURE] operation_type={_op} status=UNDISCOVERED missing={_required}", flush=True)
            elif _final_op_map.get(_op, {}).get("status") == "SCANNING":
                # SCANNING残骸救済: 前回ステータスまたはNEEDS_REVIEWに戻す
                _rem_prev2 = _final_op_map[_op].get("prev_status") or "NEEDS_REVIEW"
                _ensure_update[f"operation_mappings.{_op}.status"] = _rem_prev2
                _ensure_update[f"operation_mappings.{_op}.last_scanned_at"] = _now_ensure
                print(f"[P24_7_SCANNING_RESTORE] op={_op} status={_rem_prev2}", flush=True)
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
            if _ops47.get(op, {}).get("status") in ("READY", "NEEDS_REVIEW")
            and _ops47.get(op, {}).get("target_url")
        ]
        print(f"[P24_7_STEPS_READY_REVIEW] step_ops={_step_ops47} skipped={[op for op in _normalized_candidates if op not in _step_ops47]}", flush=True)
        _steps47 = rebuild_operation_steps(_step_ops47, _nav47, _ops47, _det47) if _step_ops47 else {}
        db.collection("media_mappings").document(mapping_id).update({
            "operation_steps_by_type": _steps47,
            # [P23_P24] operation_candidatesはoperation_mappingsキーから動的に取得するため保存不要
            "updated_at": _dt47.datetime.utcnow(),
        })
        # ── capabilities再計算（設計図11章）──
        _cap_map = {
            "news_post":       "can_post_news",
            "text_update":     "can_update_text",
            "media_replace":   "can_upload_image",
            "schedule_update": "can_update_schedule",
            "price_update":    "can_update_price",
            "entity_register": "can_register_entity",
            "entity_update":   "can_update_entity",
        }
        _new_caps = {}
        for _op_cap, _cap_key in _cap_map.items():
            _op_st = (_ops47.get(_op_cap) or {}).get("status", "")
            _new_caps[_cap_key] = _op_st in ("READY", "NEEDS_REVIEW")
        # status_updateはtext_updateと同じcan_update_textキーを共有 → ORで上書き
        _status_st = (_ops47.get("status_update") or {}).get("status", "")
        if _status_st in ("READY", "NEEDS_REVIEW"):
            _new_caps["can_update_text"] = True
        _new_caps["can_login"] = True
        db.collection("media_mappings").document(mapping_id).update({
            "capabilities": _new_caps,
        })
        print(f"[P24_7_CAPABILITIES_RECALC] caps={_new_caps}", flush=True)
        print(f"[P24_7_STEPS_REBUILT] ops={list(_steps47.keys())} step_count={len(_step_ops47)}", flush=True)
    except Exception as _rb47:
        print(f"[P24_7_STEPS_REBUILD_ERROR] {_rb47}", flush=True)
    try:
        _rebuild_media_schema_for_mapping(db, mapping_id)
    except Exception as _schema47:
        print(f"[P24_7_MEDIA_SCHEMA_REBUILD_ERROR] {_schema47}", flush=True)
    # BGタスク完了: scan_progress.status=DONEを書き込み
    try:
        db.collection("media_mappings").document(mapping_id).update({
            "scan_progress.status": "DONE",
            "scan_progress.kind": "multi_deep_scan",
            "scan_progress.done": len(op_candidates),
            "scan_progress.total": len(op_candidates),
            "scan_progress.finished_at": _dt47.datetime.utcnow().isoformat(),
            "scan_progress.updated_at": _dt47.datetime.utcnow().isoformat(),
        })
    except Exception as _e_prog_done:
        print(f"[BG_TASK_PROGRESS_DONE_ERROR] {_e_prog_done}", flush=True)
    return {
        "ok": True,
        "mapping_id": mapping_id,
        "operations_count": len(op_candidates),
        "ready": ready_ops,
        "review": review_ops,
        "waiting": waiting_ops,
        "failed": failed_ops,
        "undiscovered": undiscovered_ops,
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

# ==============================================================
# HTML Menu Import
# POST /api/agent/media/map/{mapping_id}/html_menu/import
# ==============================================================

class HtmlMenuImportRequest(BaseModel):
    source_url: str = ""
    raw_html:   str
    follow_links: bool = True
    max_follow_pages: int = 25

@router.post("/media/map/{mapping_id}/html_menu/import")
def html_menu_import(
    mapping_id: str,
    req: HtmlMenuImportRequest,
    user: dict = Depends(verify_token),
):
    """
    管理メニューHTMLを貼り付けて媒体構造を保存する。
    ログイン可能な媒体では、貼り付けHTMLで見えたURLを起点に配下ページやタブ候補も追跡する。
    Operation確定・capability確定・steps生成はここでは行わない。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    if not req.raw_html.strip():
        raise HTTPException(status_code=400, detail="raw_htmlが空です")

    import datetime as _dt_menu
    from api.core.browser_executor import parse_menu_html, expand_menu_links_from_seed_urls

    # HTML解析
    menu_items = parse_menu_html(req.raw_html, source_url=req.source_url)
    if not menu_items:
        raise HTTPException(status_code=400, detail="メニューリンクが検出できませんでした。HTMLを確認してください。")

    mapping["id"] = mapping_id
    now = _dt_menu.datetime.utcnow()
    follow_result: dict = {"status": "SKIPPED", "reason": "not_requested"}
    saved_menu_items = list(menu_items)

    if req.follow_links:
        secret_name = str(mapping.get("credential_secret_name") or "").strip()
        login_url = str(mapping.get("login_url") or "").strip()
        if not secret_name:
            follow_result = {"status": "SKIPPED", "reason": "credential_secret_name_missing"}
        elif not login_url:
            follow_result = {"status": "SKIPPED", "reason": "login_url_missing"}
        else:
            try:
                from api.core.secret_manager import get_secret_json
                creds = get_secret_json(secret_name)
                if not creds or creds.get("blocked"):
                    follow_result = {"status": "SKIPPED", "reason": "credential_blocked_or_missing"}
                else:
                    follow_result = expand_menu_links_from_seed_urls(
                        mapping,
                        creds,
                        menu_items,
                        start_url=req.source_url,
                        max_pages=max(1, min(int(req.max_follow_pages or 25), 60)),
                    ) or {"status": "ERROR", "reason": "empty_follow_result"}
                    if follow_result.get("ok") and isinstance(follow_result.get("items"), list):
                        expanded_items = [it for it in (follow_result.get("items") or []) if isinstance(it, dict)]
                        if expanded_items:
                            saved_menu_items = expanded_items
            except Exception as follow_exc:
                print(f"[HTML_MENU_IMPORT_FOLLOW_ERROR] mapping_id={mapping_id} {type(follow_exc).__name__}:{follow_exc}", flush=True)
                follow_result = {"status": "ERROR", "reason": type(follow_exc).__name__}

    # カテゴリ集計
    category_counts = {}
    for item in saved_menu_items:
        cat = item.get("category") or "その他"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    _MI_PURPOSE_RULES = [
        (["news","post","blog","diary","topic","topics","event","coupon","お知らせ","ニュース","投稿","写メ","日記","freetext","contents","campaign","realtime","marquee","速報"], "news_post_page"),
        (["profile","about","text","プロフィール","自己紹介","説明","紹介文","freetext","contents","con_txt","seo","concept","フリー","編集"], "text_edit_page"),
        (["photo","image","gallery","写真","画像","メディア","upload","アップロード"], "media_upload_page"),
        (["schedule","shift","出勤","予定","calendar","シフト","カレンダー"], "schedule_page"),
        (["price","course","料金","コース","fee","システム"], "price_page"),
        (["register","new","add","登録","新規","追加","cast","staff","キャスト","スタッフ"], "create_page"),
        (["edit","update","編集","更新","cast","staff","キャスト","スタッフ","profile","プロフィール"], "edit_page"),
        (["status","public","private","表示","非表示","公開","停止","有効","無効","ステータス","state","visible","hidden","active","inactive","enabled","disabled","standby","girl","cast"], "status_page"),
    ]

    def _infer_manual_purpose(_url: str, _title: str, _category: str) -> str:
        _text = f"{_url} {_title} {_category}".lower()
        for _kws, _purpose in _MI_PURPOSE_RULES:
            if any(_kw in _text for _kw in _kws):
                return _purpose
        return "unknown"

    # navigation_graph.pagesにmanual_importページとして保存
    nav_page = {
        "url":           req.source_url or mapping.get("media_url", ""),
        "title":         "manual_html_import",
        "raw_snapshot":  True,
        "manual_import": True,
        "menu_items":    saved_menu_items,
        "links":         [
            {"href": item["href"], "text": item["title"], "absolute_url": item["absolute_url"]}
            for item in saved_menu_items
        ],
        "collected_at":  now.isoformat(),
    }

    # Firestore保存
    try:
        # manual_menu_items保存
        # 既存のpagesを取得してmanual_importページを追加
        _cur_doc2 = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _cur_pages = (_cur_doc2.get("navigation_graph") or {}).get("pages") or []
        # manual_importページを先頭に追加（重複除去）
        _cur_pages = [p for p in _cur_pages if not p.get("manual_import")]
        # menu_itemsからページリストを生成してpagesに追加
        _menu_pages = []
        _seen_urls = set()
        for _mi in saved_menu_items:
            _abs = _mi.get("absolute_url") or _mi.get("href") or ""
            if _abs and _abs not in _seen_urls and _abs.startswith("http"):
                _seen_urls.add(_abs)
                _purpose = _infer_manual_purpose(
                    _abs,
                    _mi.get("title") or _mi.get("href") or "",
                    _mi.get("category") or "",
                )
                _menu_pages.append({
                    "url":          _abs,
                    "title":        _mi.get("title") or _mi.get("href") or "",
                    "category":     _mi.get("category") or "",
                    "manual_import": True,
                    "page_purpose":  _purpose,
                    "page_purpose_source": "manual_import_keyword",
                    "is_operation_target": _purpose != "unknown",
                    "collected_at": now.isoformat(),
                })
        _new_pages = _menu_pages + [p for p in _cur_pages if p.get("url") not in _seen_urls]
        db.collection("media_mappings").document(mapping_id).update({
            "manual_menu_items":              saved_menu_items,
            "manual_menu_imported_at":        now,
            "manual_menu_raw_html":          req.raw_html,
            "manual_menu_source_url":         req.source_url,
            "manual_menu_follow_result":      follow_result,
            "navigation_graph.manual_import": nav_page,
            "navigation_graph.pages":         _new_pages,
            "updated_at":                     now,
        })
        print(
            f"[HTML_MENU_IMPORT_SAVED] mapping_id={mapping_id} seed_items={len(menu_items)} "
            f"saved_items={len(saved_menu_items)} categories={list(category_counts.keys())}",
            flush=True,
        )
    except Exception as e:
        print(f"[HTML_MENU_IMPORT_SAVE_ERROR] {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"保存に失敗しました: {type(e).__name__}")

    return {
        "ok":               True,
        "mapping_id":       mapping_id,
        "seed_items_count": len(menu_items),
        "items_count":      len(saved_menu_items),
        "category_counts":  category_counts,
        "categories":       list(category_counts.keys()),
        "menu_items":       saved_menu_items,
        "follow_status":    str(follow_result.get("status") or ""),
        "follow_reason":    str(follow_result.get("reason") or follow_result.get("error") or ""),
        "follow_scanned_pages": int(follow_result.get("visited_pages") or 0),
        "follow_raw_count": int(follow_result.get("raw_count") or 0),
    }


# ==============================================================
# 手動ページ登録: POST /api/agent/media/map/{mapping_id}/manual_pages
# ユーザーがページタイトル+URL+HTMLを貼って operation_mappings に直接保存
# ==============================================================

class ManualPageEntry(BaseModel):
    title:            str = ""
    url:              str
    html:             str
    op_type_override: str = ""  # ユーザー指定のoperation_type（空の場合はGeminiの判定を使う）

class ManualPagesImportRequest(BaseModel):
    pages: list[ManualPageEntry]

_PAGE_TYPE_TO_OP = {
    "entity_register":  "entity_register",
    "entity_update":    "entity_update",
    "schedule_update":  "schedule_update",
    "news_post":        "news_post",
    "text_update":      "text_update",
    "media_replace":    "media_replace",
    "price_update":     "price_update",
    "status_update":    "entity_update",   # 廃止: 公開/非公開/在籍は情報更新に統合
    "page_monitor":     "page_monitor",
}

# canonical → role_key マッピング（browser_executor の OP_FIELD_STEPS に対応）
_CANONICAL_TO_ROLE: dict[str, str] = {
    "content.body":    "body",
    "content.title":   "title",
    "image.file":      "file",
    "media.file":      "file",
    "price.amount":    "price",
    "price.name":      "price",
    "price.fee":       "price",
    "schedule.date":   "date_input",
    "schedule.day":    "date_input",
    "schedule.start_time": "date_input",
    "reply.body":      "body",
    "inquiry.body":    "body",
    "contact.message": "body",
    "survey.body":     "body",
    "form.field":      "body",
}

_CONTROL_FIELD_TYPES = {"hidden", "password", "submit", "button", "reset", "image"}
_CONTROL_FIELD_KEYWORDS = (
    "login", "signin", "sign_in", "ログイン", "管理者id", "管理者", "admin",
    "password", "passwd", "pass", "パスワード",
    "hidden", "(hidden)", "csrf", "_token", "token", "nonce",
    "submit", "send", "送信", "保存する", "ログインid/パスワードを保存する",
    "remember", "button", "open_field", "select_girl_review",
    "sort", "ソート", "削除", "delete", "戻る", "back", "確認",
)

def _field_text_blob(field: dict) -> str:
    parts = [
        field.get("label"), field.get("name"), field.get("id"), field.get("key"),
        field.get("canonical"), field.get("selector"), field.get("placeholder"),
    ]
    return " ".join(str(p or "") for p in parts).strip().lower()

def _is_actionable_mapping_field(field: dict) -> bool:
    """Return True for business-editable fields, False for login/control fields."""
    if not isinstance(field, dict):
        return False
    ftype = str(field.get("type") or "text").lower().strip()
    if ftype in _CONTROL_FIELD_TYPES:
        return False
    blob = _field_text_blob(field)
    if not blob:
        return False
    if "input[type=password]" in blob or "type='password'" in blob or 'type="password"' in blob:
        return False
    return not any(k in blob for k in _CONTROL_FIELD_KEYWORDS)

def _manual_page_login_reason(title: str, url: str, html: str, fields: list[dict] | None = None) -> str:
    """Detect login/credential screens so they are never saved as operation forms."""
    fields = [f for f in (fields or []) if isinstance(f, dict)]
    title_l = str(title or "").lower()
    url_l = str(url or "").lower()
    html_l = str(html or "")[:200000].lower()
    page_blob = f"{title_l} {url_l} {html_l[:20000]}"
    has_password = any(str(f.get("type") or "").lower() == "password" for f in fields)
    has_password = has_password or "type=\"password\"" in html_l or "type='password'" in html_l
    loginish = any(k in page_blob for k in ("login", "signin", "sign in", "ログイン", "パスワード", "管理者id"))
    actionable = [f for f in fields if _is_actionable_mapping_field(f)]
    if has_password and (loginish or len(actionable) <= 2):
        return "ログイン画面です。投稿・更新フォームではないためマッピングとして保存しません。ログイン後の投稿画面HTMLを貼るか、ログイン後に到達できる正しいURLを登録してください。"
    if loginish and fields and not actionable:
        return "ログイン/認証用の項目だけが検出されています。業務フォームではないため保存しません。"
    return ""


def _find_mapping_page_by_url(mapping: dict, url: str) -> dict:
    target_norm = _normalize_menu_scan_url(url)
    if not target_norm:
        return {}
    for page in ((mapping.get("navigation_graph") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        if _normalize_menu_scan_url(page.get("url") or "") == target_norm:
            return page
    for page in mapping.get("manual_form_pages") or []:
        if not isinstance(page, dict):
            continue
        if _normalize_menu_scan_url(page.get("url") or "") == target_norm:
            return page
    for op_data in (mapping.get("operation_mappings") or {}).values():
        if not isinstance(op_data, dict):
            continue
        if _normalize_menu_scan_url(op_data.get("target_url") or "") == target_norm:
            return op_data
    return {}


def _cross_media_target_mismatch_reason(operation_type: str, mapping: dict, url: str, page: dict | None = None) -> str:
    """
    クロスメディアの送信先が問い合わせ/連絡フォームに見える場合は弾く。
    news_post/blog_post の誤割当を防ぐための軽量ガード。
    """
    if operation_type not in {"news_post", "blog_post"}:
        return ""
    page = page or _find_mapping_page_by_url(mapping, url)
    page_blob_parts = [
        url,
        page.get("title") or "",
        page.get("html_title") or "",
        page.get("manual_title") or "",
        page.get("page_purpose") or "",
        page.get("page_type") or "",
        page.get("category") or "",
        page.get("form_action") or "",
    ]
    page_blob = " ".join(str(v or "") for v in page_blob_parts).lower()
    contact_terms = ("contact", "inquiry", "toiawase", "お問い合わせ", "お問合せ", "問い合わせ")
    news_terms = ("news", "blog", "post", "お知らせ", "ニュース", "投稿", "記事", "日記", "topic", "topics")
    contact_like = any(term in page_blob for term in contact_terms)
    news_like = any(term in page_blob for term in news_terms)
    field_blob = " ".join(_field_text_blob(f) for f in (page.get("fields") or []) if isinstance(f, dict)).lower()
    field_contact_like = any(term in field_blob for term in ("email", "mail", "tel", "phone", "contact", "inquiry", "お問い合わせ", "お問合せ", "問い合わせ", "message"))
    field_news_like = any(term in field_blob for term in ("title", "body", "content", "news", "blog", "投稿", "記事", "お知らせ", "ニュース"))

    if contact_like and not news_like:
        return "問い合わせ/連絡フォームに見えるため news_post/blog_post の対象ではありません"
    if field_contact_like and not field_news_like:
        return "問い合わせ/連絡フォームの入力項目が検出されたため news_post/blog_post の対象ではありません"
    if str(page.get("page_purpose") or "") in {"global_navigation", "external_link_setting", "settings_page", "media_library", "list_page"}:
        return "投稿フォームではないページ種別です"
    return ""

def _fields_to_selectors(fields: list[dict], save_selector: str = "") -> dict:
    """Gemini解析fields配列 → browser_executorが期待するselectorsディクショナリへ変換。
    entity_register/entity_update: 全フィールドを canonical-tail キーで保存（動的展開）。
    その他の op: role_key (body/title/file/price/date_input) にもエイリアス保存。
    """
    selectors: dict = {}
    for f in fields:
        if not isinstance(f, dict):
            continue
        if not _is_actionable_mapping_field(f):
            continue
        selector = str(f.get("selector") or "").strip()
        if not selector:
            continue
        ftype = str(f.get("type") or "text").lower()
        canonical = str(f.get("canonical") or "")
        label     = str(f.get("label") or f.get("name") or canonical or "")
        name_attr = str(f.get("name") or f.get("id") or "")

        # role_key: canonical末尾部分 / name属性 / 連番フォールバック
        tail = canonical.split(".")[-1] if canonical else ""
        role_key = tail or name_attr or f"field_{len(selectors)}"

        rec = {"selector": selector, "source": "manual", "label": label[:80],
               "type": ftype, "confidence": "high"}
        selectors[role_key] = rec

        # generic role alias（non-entity opsのOP_FIELD_STEPSキーに対応）
        alias = _CANONICAL_TO_ROLE.get(canonical)
        if alias and alias not in selectors:
            selectors[alias] = rec
        # textarea は常に body エイリアス
        if ftype == "textarea" and "body" not in selectors:
            selectors["body"] = rec
        # file input は常に file エイリアス
        if ftype == "file" and "file" not in selectors:
            selectors["file"] = rec

    if save_selector:
        selectors["save"] = {"selector": save_selector, "source": "manual",
                             "label": "保存", "type": "button", "confidence": "high"}
    return selectors


def _extract_fields_from_html_direct(html: str) -> list:
    """html.parserでHTMLから可視フォームフィールドを直接抽出するフォールバック。
    Geminiがfieldsを返せなかった場合のみ使用。
    """
    from html.parser import HTMLParser

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fields = []
            self._labels: dict = {}
            self._cur_label_for: str = ""
            self._cur_label_buf: str = ""
            self._in_label = False

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            a = dict(attrs)
            if tag == "label":
                self._in_label = True
                self._cur_label_for = a.get("for") or a.get("htmlfor") or ""
                self._cur_label_buf = ""
            elif tag in ("input", "textarea", "select"):
                tp = (a.get("type") or "text").lower() if tag == "input" else tag
                if tp in ("hidden", "submit", "button", "reset", "image"):
                    return
                name = a.get("name") or ""
                id_ = a.get("id") or ""
                sel = f"#{id_}" if id_ else (f"{tag}[name='{name}']" if name else tag)
                self.fields.append({
                    "selector": sel,
                    "label": "",
                    "name": name,
                    "id": id_,
                    "type": tp,
                    "required": "required" in a,
                    "canonical": "",
                    "_id_key": id_,
                })

        def handle_endtag(self, tag):
            if tag.lower() == "label" and self._in_label:
                self._in_label = False
                if self._cur_label_for:
                    self._labels[self._cur_label_for] = self._cur_label_buf.strip()
                self._cur_label_for = ""
                self._cur_label_buf = ""

        def handle_data(self, data):
            if self._in_label:
                self._cur_label_buf += data

        def finalize(self):
            seen = set()
            out = []
            for f in self.fields:
                key = f["selector"]
                if key in seen:
                    continue
                seen.add(key)
                if f["_id_key"] in self._labels:
                    f["label"] = self._labels[f["_id_key"]]
                elif not f["label"]:
                    f["label"] = f["name"] or f["id"]
                del f["_id_key"]
                out.append(f)
            return out

    p = _Parser()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.finalize()


def _extract_save_selector_from_html(html: str) -> str:
    """HTMLからsubmit/登録ボタンのCSSセレクターを抽出する。"""
    from html.parser import HTMLParser

    _SAVE_KEYWORDS = ("登録", "保存", "更新", "送信", "確定", "完了", "submit", "save", "register", "post")

    class _BtnParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.result = ""
        def handle_starttag(self, tag, attrs):
            if self.result:
                return
            tag = tag.lower()
            if tag not in ("input", "button"):
                return
            a = dict(attrs)
            tp = (a.get("type") or "").lower()
            if tp not in ("submit", "button", "image", ""):
                return
            val = (a.get("value") or "").strip()
            id_ = a.get("id") or ""
            nm = a.get("name") or ""
            if any(kw in val.lower() for kw in _SAVE_KEYWORDS):
                self.result = f"#{id_}" if id_ else (f"{tag}[name='{nm}']" if nm else f"{tag}[type='submit']")
            elif tp == "submit":
                self.result = f"#{id_}" if id_ else (f"input[type='submit']" if tag == "input" else "button[type='submit']")
        def handle_data(self, data):
            pass

    bp = _BtnParser()
    try:
        bp.feed(html)
    except Exception:
        pass
    return bp.result


@router.post("/media/map/{mapping_id}/manual_pages")
def manual_pages_import(
    mapping_id: str,
    req: ManualPagesImportRequest,
    user: dict = Depends(verify_token),
):
    """
    ユーザーが直接貼ったページ(title+url+html)を解析してoperation_mappingsに保存。
    自動深掘りが失敗した媒体でも確実にマッピングを確定できる手動ルート。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.pages:
        raise HTTPException(status_code=400, detail="pagesが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    from api.core.browser_executor import _gemini_analyze_page_html as _analyze_html
    import hashlib as _hs_mp
    import datetime as _dt_mp

    industry = str(mapping.get("industry") or "other")
    results = []
    op_updates: dict = {}
    saved_pages = list(mapping.get("manual_form_pages") or [])
    saved_urls = {p.get("url") for p in saved_pages}

    for entry in req.pages:
        if not entry.url.startswith("http"):
            results.append({"url": entry.url, "ok": False, "error": "URLはhttpから始まる必要があります"})
            continue

        _html_to_analyze = entry.html.strip()
        if not _html_to_analyze:
            # HTMLなし → Playwrightで自動取得
            try:
                from playwright.sync_api import sync_playwright, TimeoutError as _PWTimeout2
                from api.core.browser_executor import create_authenticated_page as _cap
                from api.core.browser_executor import is_playwright_enabled as _ipe
                from api.core.secret_manager import get_secret_json as _gsj
                if _ipe():
                    _secret_name = mapping.get("credential_secret_name")
                    _creds = _gsj(_secret_name) if _secret_name else None
                    with sync_playwright() as _p2:
                        if _creds:
                            _auth2 = _cap(_p2, {**mapping, "id": mapping_id, "mapping_id": mapping_id}, _creds)
                            _br2, _pg2 = _auth2["browser"], _auth2["page"]
                        else:
                            _br2 = _p2.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
                            _pg2 = _br2.new_page()
                        try:
                            def _on_login_pg2(pg) -> bool:
                                try:
                                    if pg.locator("input[type=password]").count() > 0:
                                        return True
                                    if any(s in pg.title().lower() for s in ["ログイン", "login", "signin"]):
                                        return True
                                except Exception:
                                    pass
                                return False

                            _nav_url2 = entry.url
                            _pg2.goto(_nav_url2, timeout=30000, wait_until="domcontentloaded")
                            _pg2.wait_for_timeout(2000)
                            if _creds and _on_login_pg2(_pg2):
                                from urllib.parse import urlparse, urlencode, parse_qs, urlunparse as _uup
                                _EPHEM2 = {"sid", "z", "token", "nonce", "_token", "csrf"}
                                _prs2 = urlparse(_nav_url2)
                                _qs2 = {k: v for k, v in parse_qs(_prs2.query, keep_blank_values=True).items() if k.lower() not in _EPHEM2}
                                _clean2 = _uup(_prs2._replace(query=urlencode(_qs2, doseq=True)))
                                if _clean2 != _nav_url2:
                                    print(f"[IMPORT_PW] Login redirect, retry without session params: {_clean2[:100]}", flush=True)
                                    _pg2.goto(_clean2, timeout=30000, wait_until="domcontentloaded")
                                    _pg2.wait_for_timeout(2000)
                            try:
                                _pg2.wait_for_selector("input:not([type=hidden]), textarea, select", timeout=5000, state="attached")
                            except Exception:
                                pass
                            try:
                                _pg2.wait_for_load_state("networkidle", timeout=4000)
                            except Exception:
                                pass
                            _html_to_analyze = _pg2.content()
                        except _PWTimeout2:
                            results.append({"url": entry.url, "ok": False, "error": "Playwrightタイムアウト"})
                            continue
                        finally:
                            _br2.close()
            except Exception as _pw_e:
                results.append({"url": entry.url, "ok": False, "error": f"Playwright取得失敗: {type(_pw_e).__name__}"})
                continue

        if not _html_to_analyze:
            results.append({"url": entry.url, "ok": False, "error": "HTMLが取得できませんでした"})
            continue

        try:
            gemini_result = _analyze_html(_html_to_analyze, entry.url, entry.title, industry=industry, force_refresh=True)
        except Exception as _ge:
            results.append({"url": entry.url, "ok": False, "error": f"解析エラー: {type(_ge).__name__}"})
            continue

        page_type = gemini_result.get("page_type") or "other"
        # ユーザー指定のop_typeを優先（Geminiの誤判定を上書き）
        op_type   = entry.op_type_override.strip() if entry.op_type_override.strip() else _PAGE_TYPE_TO_OP.get(page_type)
        raw_fields = [f for f in (gemini_result.get("fields") or []) if isinstance(f, dict)]
        login_reason = _manual_page_login_reason(entry.title, entry.url, _html_to_analyze, raw_fields)
        if login_reason:
            results.append({
                "url": entry.url,
                "ok": False,
                "status": "LOGIN_PAGE_NOT_TARGET",
                "error": login_reason,
                "page_type": page_type,
                "op_type": op_type,
                "fields_count": 0,
            })
            continue
        fields = [f for f in raw_fields if _is_actionable_mapping_field(f)]

        _save_sel = gemini_result.get("save_selector") or ""
        page_record = {
            "page_id":   _hs_mp.md5(entry.url.encode()).hexdigest()[:12],
            "title":     entry.title,
            "url":       entry.url,
            "page_type": page_type,
            "op_type":   op_type or "",
            "fields":    fields,
            "form_action": gemini_result.get("form_action") or "",
            "save_selector": _save_sel,
            "source":    "AUXILIARY_PAGE",
            "op_type_user_specified": bool(entry.op_type_override.strip()),
            "saved_at":  _dt_mp.datetime.utcnow().isoformat(),
        }
        # 既存ページ更新 or 追加
        if entry.url in saved_urls:
            saved_pages = [page_record if p.get("url") == entry.url else p for p in saved_pages]
        else:
            saved_pages.append(page_record)
            saved_urls.add(entry.url)

        if op_type:
            # 補助ページはAI整備の材料として保存し、実行可能operationには昇格させない。
            _selectors = _fields_to_selectors(fields, _save_sel)
            _url_only_ready_op = op_type in {"offer_send", "recruit_inbox_scan", "page_monitor"}
            _manual_ready = bool(fields or _save_sel or _url_only_ready_op)
            op_updates[f"operation_mappings.{op_type}.status"]      = "NEEDS_REVIEW"
            op_updates[f"operation_mappings.{op_type}.executable"]  = False
            op_updates[f"operation_mappings.{op_type}.target_url"]  = entry.url
            op_updates[f"operation_mappings.{op_type}.source"]      = "AUXILIARY_PAGE"
            op_updates[f"operation_mappings.{op_type}.candidate_only"] = True
            op_updates[f"operation_mappings.{op_type}.production_ready"] = False
            op_updates[f"operation_mappings.{op_type}.confirmation_status"] = "AI_CANDIDATE"
            op_updates[f"operation_mappings.{op_type}.fields"]      = fields
            op_updates[f"operation_mappings.{op_type}.selectors"]   = _selectors
            op_updates[f"operation_mappings.{op_type}.form_schema"] = {"fields": fields}
            op_updates[f"operation_mappings.{op_type}.form_action"] = gemini_result.get("form_action") or ""
            op_updates[f"operation_mappings.{op_type}.save_selector"] = _save_sel
            op_updates[f"operation_mappings.{op_type}.manual_title"]  = entry.title
            op_updates[f"operation_mappings.{op_type}.updated_at"]    = _dt_mp.datetime.utcnow()
            results.append({"url": entry.url, "ok": True, "op_type": op_type, "page_type": page_type, "fields_count": len(fields), "selectors_count": len(_selectors)})
        else:
            results.append({"url": entry.url, "ok": True, "op_type": None, "page_type": page_type, "fields_count": len(fields), "note": "操作タイプを特定できませんでした。URLや内容を確認してください。"})

    # Firestore一括保存
    try:
        update_payload = {
            "manual_form_pages": saved_pages,
            "updated_at": _dt_mp.datetime.utcnow(),
            **op_updates,
        }
        db.collection("media_mappings").document(mapping_id).update(update_payload)
    except Exception as _save_e:
        raise HTTPException(status_code=500, detail=f"保存に失敗しました: {type(_save_e).__name__}")

    try:
        _sync_ready_operation_steps(mapping_id, db)
        _rebuild_media_schema_for_mapping(db, mapping_id)
    except Exception as _sync_e:
        print(f"[MANUAL_PAGES_SCHEMA_SYNC_ERROR] mapping_id={mapping_id} {type(_sync_e).__name__}:{_sync_e}", flush=True)

    mapped_count = sum(1 for r in results if r.get("op_type"))
    print(f"[MANUAL_PAGES_IMPORT] mapping_id={mapping_id} pages={len(req.pages)} mapped={mapped_count}", flush=True)
    return {
        "ok": True,
        "mapping_id":   mapping_id,
        "total":        len(req.pages),
        "mapped_count": mapped_count,
        "results":      results,
        "saved_pages":  saved_pages,
    }


@router.delete("/media/map/{mapping_id}/manual_pages/{page_id}")
def manual_page_delete(
    mapping_id: str,
    page_id: str,
    user: dict = Depends(verify_token),
):
    """登録済み手動ページを削除する。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")
    pages = list(mapping.get("manual_form_pages") or [])
    deleted_page = next((p for p in pages if p.get("page_id") == page_id), {})
    new_pages = [p for p in pages if p.get("page_id") != page_id]
    if len(new_pages) == len(pages):
        raise HTTPException(status_code=404, detail="ページが見つかりません")
    import datetime as _dt_del
    _delete_updates = {
        "manual_form_pages": new_pages,
        "updated_at": _dt_del.datetime.utcnow(),
    }
    deleted_url = (deleted_page or {}).get("url") or ""
    for op, op_data in (mapping.get("operation_mappings") or {}).items():
        if not isinstance(op_data, dict):
            continue
        if op_data.get("page_id") == page_id or (deleted_url and op_data.get("target_url") == deleted_url):
            _delete_updates[f"operation_mappings.{op}"] = firestore.DELETE_FIELD
    db.collection("media_mappings").document(mapping_id).update(_delete_updates)
    try:
        _sync_ready_operation_steps(mapping_id, db)
        _rebuild_media_schema_for_mapping(db, mapping_id)
    except Exception as _sync_e:
        print(f"[MANUAL_PAGE_DELETE_SCHEMA_SYNC_ERROR] mapping_id={mapping_id} {type(_sync_e).__name__}:{_sync_e}", flush=True)
    return {"ok": True, "deleted_page_id": page_id, "remaining": len(new_pages)}


class ManualPagePatchRequest(BaseModel):
    op_type: str = ""


@router.patch("/media/map/{mapping_id}/manual_pages/{page_id}")
def manual_page_patch(
    mapping_id: str,
    page_id: str,
    req: ManualPagePatchRequest,
    user: dict = Depends(verify_token),
):
    """登録済み手動ページのop_typeだけを更新する（再解析不要）。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")
    pages = list(mapping.get("manual_form_pages") or [])
    new_pages = []
    found = False
    target_page = {}
    previous_op_type = ""
    for p in pages:
        if p.get("page_id") == page_id:
            updated = dict(p)
            previous_op_type = str(updated.get("op_type") or "")
            updated["op_type"] = req.op_type
            new_pages.append(updated)
            target_page = updated
            found = True
        else:
            new_pages.append(p)
    if not found:
        raise HTTPException(status_code=404, detail="ページが見つかりません")
    import datetime as _dt_patch
    updates: dict = {
        "manual_form_pages": new_pages,
        "updated_at": _dt_patch.datetime.utcnow(),
    }
    if previous_op_type and previous_op_type != req.op_type:
        old_map = (mapping.get("operation_mappings") or {}).get(previous_op_type) or {}
        if old_map.get("page_id") == page_id or old_map.get("target_url") == target_page.get("url"):
            updates[f"operation_mappings.{previous_op_type}"] = firestore.DELETE_FIELD
    if req.op_type:
        updates[f"operation_mappings.{req.op_type}.page_id"] = page_id
        existing_url = target_page.get("url", "")
        if existing_url:
            updates[f"operation_mappings.{req.op_type}.target_url"] = existing_url
        raw_fields = [f for f in (target_page.get("fields") or []) if isinstance(f, dict)]
        login_reason = _manual_page_login_reason(target_page.get("title") or "", existing_url, "", raw_fields)
        if login_reason:
            raise HTTPException(status_code=400, detail=login_reason)
        fields = [f for f in raw_fields if _is_actionable_mapping_field(f)]
        save_selector = target_page.get("save_selector") or ""
        selectors = _fields_to_selectors(fields, save_selector)
        updates[f"operation_mappings.{req.op_type}.status"] = "NEEDS_REVIEW"
        updates[f"operation_mappings.{req.op_type}.executable"] = False
        updates[f"operation_mappings.{req.op_type}.source"] = "AUXILIARY_PAGE"
        updates[f"operation_mappings.{req.op_type}.candidate_only"] = True
        updates[f"operation_mappings.{req.op_type}.production_ready"] = False
        updates[f"operation_mappings.{req.op_type}.confirmation_status"] = "AI_CANDIDATE"
        updates[f"operation_mappings.{req.op_type}.fields"] = fields
        updates[f"operation_mappings.{req.op_type}.selectors"] = selectors
        updates[f"operation_mappings.{req.op_type}.form_schema"] = {"fields": fields}
        updates[f"operation_mappings.{req.op_type}.form_action"] = target_page.get("form_action") or ""
        updates[f"operation_mappings.{req.op_type}.save_selector"] = save_selector
        updates[f"operation_mappings.{req.op_type}.manual_title"] = target_page.get("title") or ""
        updates[f"operation_mappings.{req.op_type}.updated_at"] = _dt_patch.datetime.utcnow()
    db.collection("media_mappings").document(mapping_id).update(updates)
    try:
        _sync_ready_operation_steps(mapping_id, db)
        _rebuild_media_schema_for_mapping(db, mapping_id)
    except Exception as _sync_e:
        print(f"[MANUAL_PAGE_PATCH_SCHEMA_SYNC_ERROR] mapping_id={mapping_id} {type(_sync_e).__name__}:{_sync_e}", flush=True)
    return {"ok": True, "page_id": page_id, "op_type": req.op_type}


class FetchAndPreviewRequest(BaseModel):
    url:              str
    title:            str = ""
    op_type_override: str = ""


@router.post("/media/map/{mapping_id}/manual_pages/fetch_and_preview")
def manual_pages_fetch_and_preview(
    mapping_id: str,
    req: FetchAndPreviewRequest,
    user: dict = Depends(verify_token),
):
    """
    PlaywrightでURLにアクセスして描画済みHTMLを取得し、Geminiで解析してプレビューを返す。
    ユーザーはURLを入力するだけでよい（HTMLの貼り付け不要）。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="URLはhttpから始まる必要があります")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    industry = str(mapping.get("industry") or "other")

    # Playwright でレンダリング済みHTMLを取得
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as _PWTimeout
        from api.core.browser_executor import (
            create_authenticated_page,
            _gemini_analyze_page_html as _analyze_html,
            is_playwright_enabled,
        )
        from api.core.secret_manager import get_secret_json
    except ImportError as _ie:
        raise HTTPException(status_code=500, detail=f"Playwright未対応: {_ie}")

    if not is_playwright_enabled():
        raise HTTPException(status_code=503, detail="Playwrightが無効です（PLAYWRIGHT_ENABLED=falseの可能性）")

    secret_name = mapping.get("credential_secret_name")
    creds = get_secret_json(secret_name) if secret_name else None

    rendered_html = None
    current_url = req.url
    screenshot_b64 = None
    fetch_error = None
    try:
        with sync_playwright() as p:
            if creds:
                auth = create_authenticated_page(p, {**mapping, "id": mapping_id, "mapping_id": mapping_id}, creds)
                browser, page = auth["browser"], auth["page"]
            else:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
                page = browser.new_page()
            try:
                def _on_login_page(pg) -> bool:
                    try:
                        if pg.locator("input[type=password]").count() > 0:
                            return True
                        if any(s in pg.title().lower() for s in ["ログイン", "login", "signin"]):
                            return True
                    except Exception:
                        pass
                    return False

                from urllib.parse import urlparse, urlencode, parse_qs, urlunparse as _uup

                _auth_qs = parse_qs(urlparse(page.url).query, keep_blank_values=True)
                _target_prs = urlparse(req.url)
                _target_base = f"{_target_prs.scheme}://{_target_prs.netloc}{_target_prs.path}"
                _orig_qs = parse_qs(_target_prs.query, keep_blank_values=True)

                # パラメータ処理の汎用ルール（サイト非依存）:
                # - z/nonce/csrf: ページ固有CSRFトークン → 常に除去
                # - sid/session等: 認証後URLに同名があれば新鮮値で置換、なければ除去
                # - lid等（両URLに存在）: 認証後URLの値で置換（新鮮化）
                # - shopdir/regmode等（元URLのみ）: そのまま保持
                _NEVER_SEND = {"z", "nonce", "csrf", "_token"}
                _REFRESH_IF_STALE = {"sid", "token", "session", "sess"}

                def _build_nav_qs(orig: dict, fresh: dict) -> dict:
                    result = {}
                    for k, v in orig.items():
                        kl = k.lower()
                        if kl in _NEVER_SEND:
                            continue  # CSRFは常に除去
                        if k in fresh:
                            result[k] = fresh[k]  # 認証後URLに同名あり → 新鮮値で置換
                        elif kl in _REFRESH_IF_STALE:
                            continue  # 期限切れの可能性があるsid等で認証後URLにないものは除去
                        else:
                            result[k] = v  # shopdir/regmode等のページパラメータはそのまま保持
                    return result

                # Strategy1: 認証済みページ上のリンクをクリック（サーバーが新鮮なzを生成）
                _nav_url = None
                if creds:
                    try:
                        _link_count = page.locator(f"a[href*='{_target_prs.path}']").count()
                        if _link_count > 0:
                            print(f"[FETCH_PREVIEW] Clicking nav link to {_target_prs.path}", flush=True)
                            page.locator(f"a[href*='{_target_prs.path}']").first.click(timeout=5000)
                            page.wait_for_load_state("domcontentloaded", timeout=15000)
                            page.wait_for_timeout(1500)
                            _nav_url = page.url
                    except Exception as _le:
                        print(f"[FETCH_PREVIEW] Link click failed: {_le}", flush=True)

                # Strategy2: 汎用パラメータ処理（CSRF除去・ページパラメータ保持・セッション新鮮化）
                if _nav_url is None or _on_login_page(page):
                    _nav_qs2 = _build_nav_qs(_orig_qs, _auth_qs)
                    _nav_url = _target_base + ("?" + urlencode(_nav_qs2, doseq=True) if _nav_qs2 else "")
                    print(f"[FETCH_PREVIEW] Strategy2 universal nav: {_nav_url[:120]}", flush=True)
                    page.goto(_nav_url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(2000)

                # Strategy3: パスのみで遷移（最終手段）
                if _on_login_page(page):
                    print(f"[FETCH_PREVIEW] Strategy3 bare URL: {_target_base}", flush=True)
                    page.goto(_target_base, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(2000)

                # SPA描画待機: networkidle → フォーム要素出現 → 追加安定待ち
                try:
                    page.wait_for_load_state("networkidle", timeout=6000)
                except Exception:
                    pass
                try:
                    page.wait_for_selector("input:not([type=hidden]), textarea, select", timeout=7000, state="visible")
                except Exception:
                    pass
                page.wait_for_timeout(1500)
                current_url = page.url
                rendered_html = page.content()
                import base64 as _b64
                try:
                    # full_page=True でページ全体をキャプチャ
                    _ss_bytes = page.screenshot(type="jpeg", quality=60, full_page=True)
                    screenshot_b64 = _b64.b64encode(_ss_bytes).decode()
                except Exception:
                    try:
                        _ss_bytes = page.screenshot(type="jpeg", quality=60, full_page=False)
                        screenshot_b64 = _b64.b64encode(_ss_bytes).decode()
                    except Exception:
                        screenshot_b64 = None
            except _PWTimeout:
                fetch_error = "ページ読み込みタイムアウト"
            finally:
                browser.close()
    except Exception as _pe:
        fetch_error = str(_pe)[:200]

    if fetch_error or not rendered_html:
        return {"ok": False, "url": req.url, "error": fetch_error or "HTML取得失敗"}

    # Gemini解析
    try:
        gemini_result = _analyze_html(rendered_html, req.url, req.title, industry=industry, force_refresh=True)
    except Exception as _ge:
        return {"ok": False, "url": req.url, "error": f"Gemini解析エラー: {type(_ge).__name__}"}

    _pt = gemini_result.get("page_type") or "other"
    _ot = req.op_type_override.strip() if req.op_type_override.strip() else _PAGE_TYPE_TO_OP.get(_pt)
    _raw_fields = [f for f in (gemini_result.get("fields") or []) if isinstance(f, dict)]

    # Geminiがfields=0の場合: html.parserで直接DOM抽出をフォールバック
    if not _raw_fields and rendered_html:
        _raw_fields = _extract_fields_from_html_direct(rendered_html)
        if _raw_fields:
            print(f"[FETCH_PREVIEW_FALLBACK] html.parser fields={len(_raw_fields)} url={req.url[:60]}", flush=True)

    _login_reason = _manual_page_login_reason(req.title, current_url or req.url, rendered_html, _raw_fields)
    if _login_reason:
        return {
            "ok": False,
            "url": req.url,
            "current_url": current_url,
            "status": "LOGIN_PAGE_NOT_TARGET",
            "error": _login_reason,
            "page_type": _pt,
            "op_type": _ot,
            "fields": [],
            "html_length": len(rendered_html),
            "screenshot": screenshot_b64,
        }

    _fields = [f for f in _raw_fields if _is_actionable_mapping_field(f)]

    return {
        "ok": True,
        "url": req.url,
        "current_url": current_url,
        "title": req.title,
        "page_type": _pt,
        "op_type": _ot,
        "op_type_user_specified": bool(req.op_type_override.strip()),
        "form_action": gemini_result.get("form_action") or "",
        "save_selector": gemini_result.get("save_selector") or _extract_save_selector_from_html(rendered_html),
        "fields": _fields,
        "html_length": len(rendered_html),
        "screenshot": screenshot_b64,
    }


@router.post("/media/map/{mapping_id}/manual_pages/preview")
def manual_pages_preview(
    mapping_id: str,
    req: ManualPagesImportRequest,
    user: dict = Depends(verify_token),
):
    """保存なしでHTMLを解析してフィールド一覧を返す。プレビュー確認用。"""
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.pages:
        raise HTTPException(status_code=400, detail="pagesが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    from api.core.browser_executor import _gemini_analyze_page_html as _analyze_html
    industry = str(mapping.get("industry") or "other")
    results = []
    for entry in req.pages:
        if not entry.html.strip() or not entry.url.startswith("http"):
            results.append({"url": entry.url, "ok": False, "error": "URLまたはHTMLが無効"})
            continue
        try:
            gemini_result = _analyze_html(entry.html, entry.url, entry.title, industry=industry, force_refresh=True)
        except Exception as _ge:
            results.append({"url": entry.url, "ok": False, "error": type(_ge).__name__})
            continue
        _pt = gemini_result.get("page_type") or "other"
        _ot = entry.op_type_override.strip() if entry.op_type_override.strip() else _PAGE_TYPE_TO_OP.get(_pt)
        _raw_fields = [f for f in (gemini_result.get("fields") or []) if isinstance(f, dict)]
        _login_reason = _manual_page_login_reason(entry.title, entry.url, entry.html, _raw_fields)
        if _login_reason:
            results.append({
                "url": entry.url,
                "title": entry.title,
                "ok": False,
                "status": "LOGIN_PAGE_NOT_TARGET",
                "error": _login_reason,
                "page_type": _pt,
                "op_type": _ot,
                "fields": [],
            })
            continue
        results.append({
            "url":        entry.url,
            "title":      entry.title,
            "ok":         True,
            "page_type":  _pt,
            "op_type":    _ot,
            "op_type_user_specified": bool(entry.op_type_override.strip()),
            "form_action": gemini_result.get("form_action") or "",
            "fields":     [f for f in _raw_fields if _is_actionable_mapping_field(f)],
        })
    return {"ok": True, "results": results}


# ==============================================================
# POST /api/agent/media/map/{mapping_id}/menu_item/deep_scan
# 指定URL1件をDOM取得→全8op一括抽出→mapping更新→steps再生成→即返却
# ==============================================================

class MenuItemDeepScanRequest(BaseModel):
    target_url: str
    max_follow_per_url: int = 50

class MenuItemsDeepScanRequest(BaseModel):
    max_urls: int = 200
    max_follow_per_url: int = 50
    force_rescan: bool = False  # デフォルトFalse: 再開時にREADY済みURLをスキップして途中再開を実現
    # Chunked synchronous scan: process targets[offset : offset+chunk_size] per
    # request so it completes under Cloud Run CPU throttling (CPU is allocated
    # during request handling). The frontend drives chunks until finished.
    offset: int = 0
    chunk_size: Optional[int] = None

class MenuItemTaskCreateRequest(BaseModel):
    target_url: str
    operation_type: str
    payload: dict = Field(default_factory=dict)
    scheduled_at: Optional[str] = None


MENU_ITEM_OPERATION_TYPES = [
    "news_post", "blog_post", "text_update", "media_replace",
    "schedule_update", "price_update", "entity_register",
    "entity_update",
]


def _manual_menu_targets_from_mapping(mapping: dict, max_urls: int = 200) -> list[dict]:
    targets: list[dict] = []
    seen: set[str] = set()

    def _add(url: str, title: str = "", category: str = ""):
        if not url or not str(url).startswith("http"):
            return
        norm = _normalize_menu_scan_url(url)
        if norm in seen:
            return
        seen.add(norm)
        targets.append({
            "url": url,
            "canonical_url": norm,
            "title": title or url,
            "category": category or "その他",
        })

    for item in mapping.get("manual_menu_items") or []:
        if not isinstance(item, dict):
            continue
        _add(
            item.get("absolute_url") or item.get("href") or "",
            item.get("title") or item.get("text") or item.get("href") or "",
            item.get("category") or "",
        )

    for pg in ((mapping.get("navigation_graph") or {}).get("pages") or []):
        if not isinstance(pg, dict) or not pg.get("manual_import"):
            continue
        _add(pg.get("url") or "", pg.get("title") or "", pg.get("category") or "")

    return targets[: max(1, int(max_urls or 200))]


def _agent_page_has_editable_dom(pg: dict | None) -> bool:
    if not pg:
        return False
    return bool(
        pg.get("forms")
        or pg.get("inputs")
        or pg.get("buttons")
        or pg.get("textareas")
        or pg.get("file_inputs")
        or pg.get("selects")
        or int(pg.get("forms_count") or 0) > 0
        or int(pg.get("inputs_count") or 0) > 0
        or int(pg.get("buttons_count") or 0) > 0
        or int(pg.get("textareas_count") or 0) > 0
        or int(pg.get("file_inputs_count") or 0) > 0
        or int(pg.get("selects_count") or 0) > 0
    )


def _normalize_menu_scan_url(url: str | None) -> str:
    try:
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
        raw = str(url or "").strip()
        if not raw:
            return ""
        p = urlsplit(raw)
        path = (p.path or "").rstrip("/")
        drop_exact = {
            "sid", "z", "token", "nonce", "_token", "csrf",
            "phpsessid", "session", "sessionid",
            "entry_sid", "entry_time",
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "gclid", "fbclid", "yclid",
        }
        kept_pairs: list[tuple[str, str]] = []
        for key, value in parse_qsl(p.query, keep_blank_values=True):
            key_l = str(key or "").strip().lower()
            if not key_l:
                continue
            if key_l in drop_exact:
                continue
            if key_l.startswith("utm_"):
                continue
            if key_l.endswith("_sid") or key_l.endswith("sid"):
                continue
            if key_l.endswith("_token") or key_l.endswith("token"):
                continue
            kept_pairs.append((key, value))
        kept_pairs.sort(key=lambda row: (str(row[0]), str(row[1])))
        query = urlencode(kept_pairs, doseq=True)
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, query, ""))
    except Exception:
        return str(url or "").strip().rstrip("/")


def _menu_scan_health_status(summary: dict) -> str:
    total = int(summary.get("total") or 0)
    ready = int(summary.get("ready") or 0)
    review = int(summary.get("needs_review") or 0)
    failed = int(summary.get("failed") or 0)
    no_op = int(summary.get("no_operation") or 0)
    no_dom = int(summary.get("no_editable_dom") or 0)
    scanned = int(summary.get("scanned") or 0)
    if total <= 0:
        return "EMPTY"
    if scanned <= 0:
        return "UNSCANNED"
    if scanned < total:
        return "INCOMPLETE"
    if failed >= max(3, total // 3):
        return "FAILED_MANY"
    if ready + review == 0:
        return "NO_READY"
    if failed or no_op or review or no_dom:
        return "PARTIAL"
    return "READY"


def _menu_record_step_count(rec: dict) -> int:
    if not isinstance(rec, dict):
        return 0
    steps = rec.get("steps")
    if isinstance(steps, list):
        return len(steps)
    for key in ("step_count", "steps"):
        try:
            return int(rec.get(key) or 0)
        except Exception:
            continue
    return 0


def _menu_item_ready_operations(item: dict) -> dict[str, int]:
    ready: dict[str, int] = {}
    if not isinstance(item, dict):
        return ready
    for op, rec in (item.get("operations") or {}).items():
        if not isinstance(rec, dict):
            continue
        count = _menu_record_step_count(rec)
        if rec.get("status") == "READY" and count > 0 and rec.get("production_ready") is True:
            ready[str(op)] = count
    for rec in item.get("updated_ops") or []:
        if not isinstance(rec, dict):
            continue
        op = str(rec.get("op") or "")
        if not op or op in ready:
            continue
        count = _menu_record_step_count(rec)
        if rec.get("status") == "READY" and count > 0 and rec.get("production_ready") is True:
            ready[op] = count
    return ready


def _menu_scan_summary(items: list[dict]) -> dict:
    from collections import Counter

    stop_stages = Counter()
    stop_reasons = Counter()
    summary = {
        "total": len(items),
        "ready": 0,
        "needs_review": 0,
        "no_editable_dom": 0,
        "no_operation": 0,
        "failed": 0,
        "scanned": 0,
        "unknown": 0,
        "ready_operations": 0,
    }
    for item in items:
        st = item.get("status")
        if st:
            summary["scanned"] += 1
        ready_ops = _menu_item_ready_operations(item)
        if ready_ops:
            summary["ready"] += 1
            summary["ready_operations"] += len(ready_ops)
        elif st == "NEEDS_REVIEW":
            summary["needs_review"] += 1
        elif st == "NO_EDITABLE_DOM":
            summary["no_editable_dom"] += 1
        elif st == "NO_OPERATION":
            summary["no_operation"] += 1
        elif st == "FAILED":
            summary["failed"] += 1
        else:
            summary["unknown"] += 1
        if st in ("FAILED", "NO_EDITABLE_DOM", "NO_OPERATION", "NEEDS_REVIEW"):
            diag = item.get("diagnostics") or {}
            stage = str(diag.get("stop_stage") or st or "unknown")[:80]
            reason = str(diag.get("stop_reason") or item.get("message") or st or "unknown")[:180]
            stop_stages[stage] += 1
            stop_reasons[reason] += 1
    summary["ready_or_review"] = summary["ready"] + summary["needs_review"]
    summary["action_required"] = summary["needs_review"] + summary["no_operation"] + summary["failed"]
    summary["non_actionable"] = summary["no_editable_dom"]
    summary["task_ready"] = summary["ready"]
    summary["unscanned"] = max(0, summary["total"] - summary["scanned"])
    summary["completed"] = bool(summary["total"] > 0 and summary["scanned"] >= summary["total"])
    summary["health_status"] = _menu_scan_health_status(summary)
    summary["top_stop_stages"] = [
        {"stage": k, "count": v} for k, v in stop_stages.most_common(5)
    ]
    summary["top_stop_reasons"] = [
        {"reason": k, "count": v} for k, v in stop_reasons.most_common(5)
    ]
    return summary


def _compact_menu_operation_record(rec: dict, parent: bool = False) -> dict:
    if not isinstance(rec, dict):
        return {}
    status = rec.get("status") or "UNDISCOVERED"
    steps = rec.get("steps") or []
    form_schema = rec.get("form_schema") or {}
    compact_schema = {}
    if isinstance(form_schema, dict):
        compact_schema = {
            "source": form_schema.get("source", ""),
            "fields_count": int(form_schema.get("fields_count") or len(form_schema.get("fields") or []) or 0),
            "profile_fields_count": int(form_schema.get("profile_fields_count") or 0),
            "is_profile_form": bool(form_schema.get("is_profile_form")),
            "fields": [] if parent else [
                {
                    "order": f.get("order"),
                    "canonical": f.get("canonical", ""),
                    "name": f.get("name", ""),
                    "label": f.get("label", ""),
                    "section": f.get("section", ""),
                    "type": f.get("type", ""),
                    "selector": f.get("selector", ""),
                }
                for f in (form_schema.get("fields") or [])[:80]
                if isinstance(f, dict)
            ],
        }
    compact = {
        "op": rec.get("op"),
        "status": status,
        "target_url": rec.get("target_url") or "",
        "missing": list(rec.get("missing") or [])[:10],
        "selectors_count": int(rec.get("selectors_count", len(rec.get("selectors") or {})) or 0),
        "validation_score": rec.get("validation_score", 0),
        "executable": bool(rec.get("executable")),
        "confirmed": bool(rec.get("confirmed")),
        "production_ready": bool(rec.get("production_ready")),
        "candidate_only": bool(rec.get("candidate_only", not rec.get("confirmed"))),
        "confirmation_status": rec.get("confirmation_status") or "",
        "step_count": int(rec.get("step_count", len(steps)) or 0),
        "form_fields_count": int(rec.get("form_fields_count") or compact_schema.get("fields_count") or 0),
        "form_schema": compact_schema,
        "scanned_at": rec.get("scanned_at", ""),
        "source": rec.get("source", ""),
        "analysis_source": rec.get("analysis_source", ""),
    }
    if status == "READY" and steps and not parent:
        compact["steps"] = steps[:12]
    return compact


def _compact_menu_item_result(item: dict, ultra: bool = False, parent: bool = False) -> dict:
    if not isinstance(item, dict):
        return {}
    diag = item.get("diagnostics") or {}
    compact_diag = {
        "stop_stage": diag.get("stop_stage", ""),
        "stop_reason": str(diag.get("stop_reason") or "")[:220],
        "dom_status": diag.get("dom_status", ""),
        "dom_message": str(diag.get("dom_message") or "")[:180],
        "followed_count": int(diag.get("followed_count") or item.get("followed_count") or 0),
        "followed_urls": list(diag.get("followed_urls") or [])[:3],
        "inspected_urls": list(diag.get("inspected_urls") or [])[:5],
        "editable_pages_count": int(diag.get("editable_pages_count") or 0),
        "page_evidence": list(diag.get("page_evidence") or [])[:3],
        "operation_results": list(diag.get("operation_results") or [])[:8],
    }
    operations = {}
    for op, rec in (item.get("operations") or {}).items():
        if isinstance(rec, dict):
            compact_rec = _compact_menu_operation_record({**rec, "op": rec.get("op") or op}, parent=parent or ultra)
            if ultra:
                compact_rec.pop("steps", None)
            operations[op] = compact_rec
    item_production_ready = any(
        isinstance(rec, dict) and rec.get("production_ready") is True
        for rec in operations.values()
    )
    item_confirmed = any(
        isinstance(rec, dict) and rec.get("confirmed") is True
        for rec in operations.values()
    )
    updated_ops = []
    for row in (item.get("updated_ops") or [])[:8]:
        if isinstance(row, dict):
            updated_ops.append({
                "op": row.get("op"),
                "status": row.get("status"),
                "target_url": row.get("target_url") or "",
                "missing": list(row.get("missing") or [])[:8],
                "selectors": row.get("selectors", 0),
                "steps": row.get("steps", row.get("step_count", 0)),
                "protected": bool(row.get("protected")),
                "confirmed": bool(row.get("confirmed")),
                "production_ready": bool(row.get("production_ready")),
                "candidate_only": bool(row.get("candidate_only")),
            })
    return {
        "url": item.get("url") or "",
        "canonical_url": _normalize_menu_scan_url(item.get("canonical_url") or item.get("url") or ""),
        "title": item.get("title") or item.get("url") or "",
        "category": item.get("category") or "その他",
        "status": item.get("status") or "",
        "message": str(item.get("message") or "")[:240],
        "confirmed": bool(item.get("confirmed") or item_confirmed),
        "production_ready": bool(item.get("production_ready") or item_production_ready),
        "candidate_only": bool(item.get("candidate_only", not (item.get("production_ready") or item_production_ready))),
        "confirmation_status": item.get("confirmation_status") or ("AI_CONFIRMED" if item_production_ready else ""),
        "updated_ops": updated_ops,
        "operations": operations,
        "scope_urls": list(item.get("scope_urls") or [])[:10],
        "followed_count": int(item.get("followed_count") or 0),
        "diagnostics": compact_diag if not ultra else {
            "stop_stage": compact_diag["stop_stage"],
            "stop_reason": compact_diag["stop_reason"],
            "followed_count": compact_diag["followed_count"],
        },
        "scanned_at": item.get("scanned_at", ""),
    }


def _compact_menu_scan_items(items: list[dict], ultra: bool = False, parent: bool = False) -> list[dict]:
    return [_compact_menu_item_result(item, ultra=ultra, parent=parent) for item in (items or []) if isinstance(item, dict)]


def _menu_scan_item_doc_id(url: str) -> str:
    import hashlib as _hashlib_menu_doc
    import re as _re_menu_doc
    raw = _normalize_menu_scan_url(url) or str(url or "")
    slug = _re_menu_doc.sub(r"[^0-9a-zA-Z_-]+", "_", raw)[:70].strip("_") or "url"
    digest = _hashlib_menu_doc.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:14]
    return f"item_{slug}_{digest}"[:140]


def _menu_scan_item_has_detail(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    for rec in (item.get("operations") or {}).values():
        if not isinstance(rec, dict):
            continue
        form_schema = rec.get("form_schema") or {}
        if rec.get("steps") or rec.get("selectors") or (isinstance(form_schema, dict) and form_schema.get("fields")):
            return True
    return bool((item.get("diagnostics") or {}).get("page_evidence"))


def _write_menu_scan_item_documents(db, mapping_id: str, items: list[dict], generation: str) -> int:
    ref = db.collection("media_mappings").document(mapping_id).collection("menu_scan_items")
    written = 0
    for item in items or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if not _menu_scan_item_has_detail(item):
            continue
        doc_id = _menu_scan_item_doc_id(item.get("url") or "")
        row = dict(item)
        row.update({
            "mapping_id": mapping_id,
            "canonical_url": _normalize_menu_scan_url(item.get("canonical_url") or item.get("url") or ""),
            "menu_scan_generation": generation,
            "updated_at": generation,
        })
        ref.document(doc_id).set(row, merge=True)
        written += 1
    return written


def _get_menu_scan_item_document(db, mapping_id: str, target_url: str) -> dict:
    if not target_url:
        return {}
    try:
        doc_ref = (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("menu_scan_items")
            .document(_menu_scan_item_doc_id(target_url))
        )
        snap = doc_ref.get()
        row = snap.to_dict() or {}
        if row:
            return row
        target_norm = _normalize_menu_scan_url(target_url)
        for alt in (
            db.collection("media_mappings")
            .document(mapping_id)
            .collection("menu_scan_items")
            .limit(500)
            .stream()
        ):
            cand = alt.to_dict() or {}
            if _normalize_menu_scan_url(cand.get("canonical_url") or cand.get("url") or "") == target_norm:
                return cand
        return {}
    except Exception:
        return {}


def _compact_parent_navigation_pages(pages: list[dict], limit: int = 180) -> list[dict]:
    compacted = []
    seen = set()
    for pg in pages or []:
        if not isinstance(pg, dict):
            continue
        url = pg.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        compacted.append({
            "url": url,
            "title": pg.get("title") or pg.get("html_title") or "",
            "html_title": pg.get("html_title", ""),
            "category": pg.get("category", ""),
            "manual_import": bool(pg.get("manual_import")),
            "followed_from": pg.get("followed_from", ""),
            "forms_count": int(pg.get("forms_count") or len(pg.get("forms") or []) or 0),
            "inputs_count": int(pg.get("inputs_count") or len(pg.get("inputs") or []) or 0),
            "buttons_count": int(pg.get("buttons_count") or len(pg.get("buttons") or []) or 0),
            "textareas_count": int(pg.get("textareas_count") or len(pg.get("textareas") or []) or 0),
            "file_inputs_count": int(pg.get("file_inputs_count") or len(pg.get("file_inputs") or []) or 0),
            "selects_count": int(pg.get("selects_count") or len(pg.get("selects") or []) or 0),
            "page_purpose": pg.get("page_purpose", ""),
            "page_purpose_source": pg.get("page_purpose_source", ""),
            "raw_snapshot": bool(pg.get("raw_snapshot")),
            "collected_at": pg.get("collected_at", ""),
            "dom_evidence": pg.get("dom_evidence") or {},
        })
        if len(compacted) >= limit:
            break
    return compacted


def _update_menu_scan_document(
    db,
    mapping_id: str,
    scan_payload: dict,
    extra_update: dict | None = None,
) -> None:
    payload = dict(extra_update or {})
    now_iso = datetime.datetime.utcnow().isoformat()
    detail_items = _compact_menu_scan_items(scan_payload.get("items") or [], parent=False)
    parent_items = _compact_menu_scan_items(scan_payload.get("items") or [], parent=True)
    try:
        written = _write_menu_scan_item_documents(db, mapping_id, detail_items, now_iso)
    except Exception as e:
        written = 0
        print(f"[MENU_SCAN_ITEM_SUBCOLLECTION_WRITE_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    structure_pages = []
    for item in scan_payload.get("items") or []:
        if isinstance(item, dict):
            structure_pages.extend([pg for pg in (item.get("structure_pages") or []) if isinstance(pg, dict)])
    try:
        pages_written = _write_structure_pages(db, mapping_id, structure_pages, source="menu_scan")
    except Exception as e:
        pages_written = 0
        print(f"[STRUCTURE_PAGES_WRITE_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    if pages_written:
        try:
            _refresh_capability_view_for_mapping(db, mapping_id)
        except Exception as e:
            print(f"[CAPABILITY_VIEW_REFRESH_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
    compact_scan = dict(scan_payload)
    compact_scan["items"] = parent_items
    compact_scan["summary"] = _menu_scan_summary(parent_items)
    compact_scan["items_storage_mode"] = "subcollection"
    compact_scan["items_subcollection"] = "menu_scan_items"
    compact_scan["items_written"] = written
    compact_scan["structure_pages_written"] = pages_written
    compact_scan["items_count"] = len(detail_items)
    compact_scan["updated_at"] = scan_payload.get("updated_at") or now_iso
    payload["manual_menu_scan_results"] = compact_scan
    try:
        db.collection("media_mappings").document(mapping_id).update(payload)
    except Exception as e:
        if "exceeds the maximum allowed size" not in str(e) and "maximum allowed size" not in str(e):
            raise
        try:
            existing_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            existing_pages = (existing_doc.get("navigation_graph") or {}).get("pages") or []
            payload["navigation_graph.pages"] = _compact_parent_navigation_pages(existing_pages)
            payload["navigation_graph.storage_mode"] = "compact_parent"
            payload["navigation_graph.updated_at"] = datetime.datetime.utcnow().isoformat()
        except Exception:
            payload["navigation_graph.pages"] = []
            payload["navigation_graph.storage_mode"] = "compact_parent_empty"
        ultra_items = _compact_menu_scan_items(scan_payload.get("items") or [], ultra=True, parent=True)
        compact_scan["items"] = ultra_items
        compact_scan["summary"] = _menu_scan_summary(ultra_items)
        compact_scan["items_storage_mode"] = "subcollection_ultra_parent"
        compact_scan["items_written"] = written
        payload["manual_menu_scan_results"] = compact_scan
        if isinstance(payload.get("scan_progress"), dict):
            payload["scan_progress"]["storage_mode"] = "compact_parent"
        else:
            payload["scan_progress.storage_mode"] = "compact_parent"
        try:
            db.collection("media_mappings").document(mapping_id).update(payload)
        except Exception as e2:
            if "exceeds the maximum allowed size" not in str(e2) and "maximum allowed size" not in str(e2):
                raise
            payload["navigation_graph.pages"] = []
            payload["navigation_graph.storage_mode"] = "scan_results_only"
            db.collection("media_mappings").document(mapping_id).update(payload)


def _rebuild_media_schema_for_mapping(db, mapping_id: str) -> dict:
    """navigation_graph / operation_mappings / menu scan results から媒体構造schemaを再構築する。"""
    try:
        from api.core.browser_executor import build_media_schema_from_pages
        import datetime as _dt_schema
        import hashlib as _hashlib_schema
        import re as _re_schema

        doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        nav_pages = [
            p for p in ((doc.get("navigation_graph") or {}).get("pages") or [])
            if isinstance(p, dict)
        ]
        op_maps = doc.get("operation_mappings") or {}
        menu_scan = doc.get("manual_menu_scan_results") or {}
        menu_items = [i for i in (menu_scan.get("items") or []) if isinstance(i, dict)]
        if menu_scan.get("items_storage_mode") in ("subcollection", "subcollection_ultra_parent"):
            try:
                detail_items = [
                    d.to_dict() or {}
                    for d in db.collection("media_mappings").document(mapping_id).collection("menu_scan_items").limit(500).stream()
                ]
                if detail_items:
                    menu_items = detail_items
            except Exception as _mi_e:
                print(f"[MEDIA_SCHEMA_MENU_ITEMS_READ_ERROR] mapping_id={mapping_id} {type(_mi_e).__name__}:{_mi_e}", flush=True)

        pseudo_pages = []
        for op, op_data in op_maps.items():
            if not isinstance(op_data, dict):
                continue
            form_schema = op_data.get("form_schema") or {}
            if not isinstance(form_schema, dict) or not form_schema.get("fields"):
                continue
            pseudo_pages.append({
                "url": op_data.get("target_url") or form_schema.get("url") or "",
                "title": form_schema.get("title") or op,
                "page_purpose": op_data.get("page_purpose") or "",
                "page_purpose_source": op_data.get("page_purpose_source") or "operation_mapping",
                "form_schema": form_schema,
            })

        for item in menu_items:
            if not isinstance(item, dict):
                continue
            for op, rec in (item.get("operations") or {}).items():
                if not isinstance(rec, dict):
                    continue
                form_schema = rec.get("form_schema") or {}
                if not isinstance(form_schema, dict) or not form_schema.get("fields"):
                    continue
                pseudo_pages.append({
                    "url": rec.get("target_url") or item.get("url") or form_schema.get("url") or "",
                    "title": item.get("title") or form_schema.get("title") or op,
                    "page_purpose": rec.get("page_purpose") or "",
                    "page_purpose_source": rec.get("page_purpose_source") or "manual_menu_scan_results",
                    "form_schema": form_schema,
                })

        # URL + fields_count で軽く重複排除
        pages = []
        seen = set()
        for pg in nav_pages + pseudo_pages:
            fs = pg.get("form_schema") or {}
            key = (pg.get("url") or fs.get("url") or "", int(fs.get("fields_count") or len(fs.get("fields") or []) or 0))
            if key in seen:
                continue
            seen.add(key)
            pages.append(pg)

        try:
            structure_written = _write_structure_pages(db, mapping_id, pages, source="media_schema_rebuild")
            if structure_written:
                _refresh_capability_view_for_mapping(db, mapping_id, doc)
        except Exception as _structure_schema_e:
            print(f"[MEDIA_SCHEMA_STRUCTURE_SAVE_ERROR] mapping_id={mapping_id} {type(_structure_schema_e).__name__}:{_structure_schema_e}", flush=True)

        schema = build_media_schema_from_pages(pages, operation_mappings=op_maps, menu_items=menu_items)
        now = _dt_schema.datetime.utcnow()
        generation = now.isoformat()
        mapping_ref = db.collection("media_mappings").document(mapping_id)

        def _doc_id(prefix: str, value: str) -> str:
            raw = str(value or "")
            slug = _re_schema.sub(r"[^0-9a-zA-Z_-]+", "_", raw)[:60].strip("_") or "item"
            digest = _hashlib_schema.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]
            return f"{prefix}_{slug}_{digest}"[:120]

        form_index = []
        forms_written = 0
        forms_ref = mapping_ref.collection("schema_forms")
        for idx, form in enumerate(schema.get("forms") or []):
            if not isinstance(form, dict):
                continue
            form_id = _doc_id("form", f"{idx}|{form.get('url','')}|{form.get('entity_type','')}|{form.get('fields_count',0)}")
            form_doc = dict(form)
            form_doc.update({
                "form_id": form_id,
                "mapping_id": mapping_id,
                "schema_generation": generation,
                "updated_at": generation,
            })
            forms_ref.document(form_id).set(form_doc, merge=True)
            forms_written += 1
            form_index.append({
                "form_id": form_id,
                "url": form.get("url", ""),
                "title": form.get("title", ""),
                "entity_type": form.get("entity_type", ""),
                "fields_count": form.get("fields_count", 0),
                "profile_fields_count": form.get("profile_fields_count", 0),
                "is_profile_form": bool(form.get("is_profile_form")),
            })

        field_index = []
        fields_written = 0
        fields_ref = mapping_ref.collection("schema_fields")
        for entity_type, entity in (schema.get("entities") or {}).items():
            if not isinstance(entity, dict):
                continue
            for canonical, field in (entity.get("fields") or {}).items():
                if not isinstance(field, dict):
                    continue
                field_id = _doc_id("field", canonical)
                field_doc = {
                    "field_id": field_id,
                    "mapping_id": mapping_id,
                    "schema_generation": generation,
                    "updated_at": generation,
                    "entity_type": entity_type,
                    "canonical": canonical,
                    "label": field.get("label", ""),
                    "type": field.get("type", ""),
                    "required": bool(field.get("required")),
                    "aliases": (field.get("aliases") or [])[:20],
                    "targets": (field.get("targets") or [])[:30],
                    "target_count": len(field.get("targets") or []),
                }
                fields_ref.document(field_id).set(field_doc, merge=True)
                fields_written += 1
                field_index.append({
                    "field_id": field_id,
                    "entity_type": entity_type,
                    "canonical": canonical,
                    "label": field.get("label", ""),
                    "type": field.get("type", ""),
                    "target_count": len(field.get("targets") or []),
                })

        entity_summary = {}
        for entity_type, entity in (schema.get("entities") or {}).items():
            if not isinstance(entity, dict):
                continue
            field_keys = list((entity.get("fields") or {}).keys())
            entity_summary[entity_type] = {
                "entity_type": entity_type,
                "field_count": len(field_keys),
                "field_keys": field_keys[:60],
                "forms": (entity.get("forms") or [])[:10],
            }

        parent_schema = {
            "schema_version": schema.get("schema_version"),
            "source": schema.get("source"),
            "storage_mode": "subcollections",
            "updated_at": generation,
            "schema_generation": generation,
            "forms_count": schema.get("forms_count", 0),
            "source_urls_count": schema.get("source_urls_count", 0),
            "entities_count": schema.get("entities_count", 0),
            "canonical_fields_count": schema.get("canonical_fields_count", 0),
            "forms_written": forms_written,
            "fields_written": fields_written,
            "forms_index": form_index[:40],
            "fields_index": field_index[:80],
            "entities": entity_summary,
            "operation_coverage": schema.get("operation_coverage", {}),
            "subcollections": {
                "forms": "schema_forms",
                "fields": "schema_fields",
            },
            "menu_items_count": schema.get("menu_items_count", 0),
        }
        db.collection("media_mappings").document(mapping_id).update({
            "media_schema": parent_schema,
            "entity_schema": entity_summary,
            "schema_first": {
                "status": "READY" if schema.get("forms_count") else "NO_FORM_SCHEMA",
                "schema_version": schema.get("schema_version"),
                "storage_mode": "subcollections",
                "schema_generation": generation,
                "forms_count": schema.get("forms_count", 0),
                "entities_count": schema.get("entities_count", 0),
                "canonical_fields_count": schema.get("canonical_fields_count", 0),
                "forms_written": forms_written,
                "fields_written": fields_written,
                "updated_at": now.isoformat(),
            },
            "updated_at": now,
        })
        print(
            f"[MEDIA_SCHEMA_REBUILT] mapping_id={mapping_id} forms={schema.get('forms_count')} entities={schema.get('entities_count')} fields={schema.get('canonical_fields_count')} storage=subcollections forms_written={forms_written} fields_written={fields_written}",
            flush=True,
        )
        return parent_schema
    except Exception as e:
        print(f"[MEDIA_SCHEMA_REBUILD_ERROR] mapping_id={mapping_id} {type(e).__name__}:{e}", flush=True)
        try:
            db.collection("media_mappings").document(mapping_id).update({
                "schema_first.status": "FAILED",
                "schema_first.error": f"{type(e).__name__}:{e}",
                "schema_first.updated_at": datetime.datetime.utcnow().isoformat(),
            })
        except Exception:
            pass
        return {}


def _menu_item_scope_pages(mapping: dict, target_url: str, extra_urls: list[str] | None = None) -> list[dict]:
    pages = ((mapping.get("navigation_graph") or {}).get("pages") or [])
    target_norm = _normalize_menu_scan_url(target_url)
    extra = {_normalize_menu_scan_url(u) for u in (extra_urls or []) if u}
    scope_pages = []
    for pg in pages:
        if not isinstance(pg, dict):
            continue
        pg_url = pg.get("url") or ""
        pg_norm = _normalize_menu_scan_url(pg_url)
        followed_norm = _normalize_menu_scan_url(pg.get("followed_from") or "")
        if pg_norm == target_norm or pg_norm in extra or followed_norm == target_norm:
            if _agent_page_has_editable_dom(pg):
                scope_pages.append(pg)

    seen = set()
    deduped = []
    for pg in scope_pages:
        u = pg.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(pg)
    return deduped


def _selector_value(sel_row) -> str:
    if isinstance(sel_row, dict):
        return str(sel_row.get("selector") or "")
    return str(sel_row or "")


def _selector_record(selector: str, source: str, label: str = "", confidence: str = "schema_fallback") -> dict:
    return {
        "selector": selector,
        "source": source,
        "label": label,
        "confidence": confidence,
    }


def _find_save_selector_from_pages(scope_pages: list[dict], target_url: str = "") -> dict | None:
    save_words = (
        "保存", "登録", "更新", "投稿", "送信", "確定", "反映",
        "save", "submit", "register", "regist", "update", "post", "send", "publish", "commit", "apply",
    )
    reject_words = ("検索", "絞込", "戻る", "削除", "cancel", "delete", "search", "filter", "back", "preview")
    target_norm = _normalize_menu_scan_url(target_url)
    for pg in scope_pages or []:
        if not isinstance(pg, dict):
            continue
        pg_norm = _normalize_menu_scan_url(pg.get("url") or "")
        if target_norm and pg_norm != target_norm:
            continue
        for group in ("buttons", "inputs"):
            for el in (pg.get(group) or [])[:120]:
                if not isinstance(el, dict):
                    continue
                selector = el.get("selector") or el.get("suggested_selector") or ""
                if not selector:
                    continue
                blob = " ".join(str(el.get(k) or "") for k in ("text", "value", "name", "id", "label", "placeholder", "aria_label")).lower()
                typ = str(el.get("type") or "").lower()
                if group == "inputs" and typ not in ("submit", "button", "image"):
                    continue
                if any(w in blob for w in reject_words):
                    continue
                if typ == "submit" or any(w in blob for w in save_words):
                    return _selector_record(selector, "form_schema_button_fallback", blob[:80], "medium")
    return None


def _field_selector_candidates_from_form(form_schema: dict) -> list[dict]:
    if not isinstance(form_schema, dict):
        return []
    rows = []
    for field in (form_schema.get("fields") or [])[:220]:
        if not isinstance(field, dict):
            continue
        selector = str(field.get("selector") or "")
        if not selector:
            continue
        typ = str(field.get("type") or "").lower()
        tag = str(field.get("tag") or "").lower()
        if typ in ("hidden", "password"):
            continue
        blob = " ".join(str(field.get(k) or "") for k in ("canonical", "name", "id", "label", "section")).lower()
        rows.append({
            "selector": selector,
            "type": typ,
            "tag": tag,
            "blob": blob,
            "label": str(field.get("label") or field.get("name") or field.get("canonical") or "")[:120],
        })
    return rows


def _pick_field_selector(form_schema: dict, op: str, selector_key: str) -> dict | None:
    rows = _field_selector_candidates_from_form(form_schema)
    if not rows:
        return None

    def _first(predicate):
        for row in rows:
            try:
                if predicate(row):
                    return _selector_record(row["selector"], "form_schema_field_fallback", row.get("label", ""), "medium")
            except Exception:
                continue
        return None

    if selector_key == "file":
        return _first(lambda r: r["type"] == "file" or "media.file" in r["blob"] or "image" in r["blob"] or "画像" in r["blob"] or "写真" in r["blob"])
    if selector_key == "price":
        return _first(lambda r: "price" in r["blob"] or "料金" in r["blob"] or "価格" in r["blob"] or "金額" in r["blob"] or "コース" in r["blob"])
    if selector_key == "date_input":
        return _first(lambda r: "schedule" in r["blob"] or "date" in r["blob"] or "出勤" in r["blob"] or "予定" in r["blob"] or "日付" in r["blob"]) or _first(lambda r: r["type"] in ("date", "time", "datetime-local"))
    if selector_key == "title":
        return _first(lambda r: "title" in r["blob"] or "タイトル" in r["blob"] or "件名" in r["blob"])
    if selector_key == "body":
        return (
            _first(lambda r: r["tag"] == "textarea")
            or _first(lambda r: "content.body" in r["blob"] or "本文" in r["blob"] or "内容" in r["blob"] or "コメント" in r["blob"] or "紹介" in r["blob"] or "説明" in r["blob"])
            or _first(lambda r: r["type"] in ("text", "search", ""))
        )
    if selector_key == "required_inputs":
        return _first(lambda r: "name" in r["blob"] or "名前" in r["blob"] or "源氏名" in r["blob"]) or _first(lambda r: r["type"] in ("text", ""))
    if selector_key == "editable_inputs":
        return (
            _first(lambda r: r["tag"] == "textarea")
            or _first(lambda r: r["type"] in ("text", "number", "tel", "url", "email", ""))
            or _first(lambda r: r["tag"] == "select")
        )
    return None


def _enrich_operation_selectors_from_schema(op: str, op_result: dict, scope_pages: list[dict]) -> dict:
    if not isinstance(op_result, dict):
        return op_result
    from api.core.browser_executor import GENERIC_OPERATION_CONFIG

    enriched = dict(op_result)
    selectors = dict(enriched.get("selectors") or {})
    form_schema = enriched.get("form_schema") or {}
    cfg = GENERIC_OPERATION_CONFIG.get(op, {}) or {}
    added = []

    for field in cfg.get("fields") or []:
        if not isinstance(field, dict):
            continue
        key = str(field.get("selector_key") or "")
        if not key or _selector_value(selectors.get(key)):
            continue
        picked = _pick_field_selector(form_schema, op, key)
        if picked:
            selectors[key] = picked
            added.append(key)

    submit_key = str(cfg.get("submit_selector_key") or "save")
    if submit_key and not _selector_value(selectors.get(submit_key)) and not _selector_value(selectors.get("submit")):
        picked_save = _find_save_selector_from_pages(scope_pages, enriched.get("target_url") or "")
        if picked_save:
            selectors[submit_key] = picked_save
            added.append(submit_key)

    if added:
        missing = [m for m in (enriched.get("missing") or []) if m not in added and not (m == "save" and ("save" in added or "submit" in added))]
        enriched["selectors"] = selectors
        enriched["missing"] = missing
        enriched["schema_selector_fallback_keys"] = added
        if enriched.get("target_url") and not missing:
            enriched["status"] = "READY"
            enriched["executable"] = True
    return enriched


def _menu_item_build_operation_records(
    mapping_id: str,
    scope_pages: list[dict],
    all_mappings: dict | None = None,
    source: str = "manual_menu_item_scan",
) -> tuple[dict, dict]:
    from api.core.browser_executor import (
        build_operation_mappings_from_dom_evidence,
        rebuild_operation_steps,
    )

    if all_mappings is None:
        all_mappings = build_operation_mappings_from_dom_evidence(mapping_id, scope_pages)

    scope_urls = {_normalize_menu_scan_url(pg.get("url")) for pg in scope_pages if isinstance(pg, dict) and pg.get("url")}
    nav_graph = {"pages": scope_pages}
    structural_maps: dict = {}
    structural_steps: dict = {}
    try:
        normalized_scope_pages = [
            pg for pg in (
                _normalize_structure_page(page, mapping_id, source=source)
                for page in (scope_pages or [])
                if isinstance(page, dict)
            )
            if pg
        ]
        if normalized_scope_pages:
            _, structural_maps, structural_steps = _build_structural_capability_view(mapping_id, normalized_scope_pages)
    except Exception as e:
        print(f"[MENU_ITEM_STRUCTURE_BUILD_ERROR] mapping_id={mapping_id} {type(e).__name__}", flush=True)
    if not structural_maps:
        return {}, {}
    records: dict = {}
    normalized_maps: dict = {}
    now_iso = datetime.datetime.utcnow().isoformat()

    for op in MENU_ITEM_OPERATION_TYPES:
        raw_result = structural_maps.get(op) or {}
        if not isinstance(raw_result, dict):
            continue
        op_result = dict(raw_result)
        if not structural_maps.get(op):
            op_result = _enrich_operation_selectors_from_schema(op, op_result, scope_pages)
        elif (all_mappings.get(op) or {}).get("form_schema"):
            op_result["form_schema"] = (all_mappings.get(op) or {}).get("form_schema")
        op_target = op_result.get("target_url") or ""
        if _normalize_menu_scan_url(op_target) not in scope_urls:
            continue
        op_status = op_result.get("status") or "UNDISCOVERED"
        if op_status not in ("READY", "NEEDS_REVIEW"):
            continue

        op_result["last_scanned_at"] = now_iso
        op_result["source"] = source
        op_result = _normalize_operation_status(op_result)

        steps = structural_steps.get(op) or rebuild_operation_steps([op], nav_graph, {op: op_result}, {}).get(op) or []
        failed_required = any(
            isinstance(step, dict)
            and step.get("required", True)
            and step.get("status") == "FAILED"
            for step in steps
        )
        if steps and not failed_required and op_result.get("missing"):
            op_result["missing"] = [m for m in (op_result.get("missing") or []) if m != "operation_steps"]
        if op_result.get("status") == "READY" and (not steps or failed_required):
            missing = list(op_result.get("missing") or [])
            if "operation_steps" not in missing:
                missing.append("operation_steps")
            op_result["missing"] = missing
            op_result["status"] = "NEEDS_REVIEW"
            op_result["executable"] = False
            op_result["human_review_required"] = True
        elif op_result.get("status") == "NEEDS_REVIEW" and steps and not failed_required and not op_result.get("missing"):
            op_result["status"] = "READY"
            op_result["executable"] = True
            op_result["human_review_required"] = False
            op_result["ready_promoted_by"] = "structure_pages" if structural_maps.get(op) else "schema_selector_fallback"

        ai_confirmed = _operation_has_ai_production_evidence(op_result, steps, failed_required)
        confirmed = _operation_mapping_is_confirmed(op_result) or ai_confirmed
        if ai_confirmed:
            op_result["analysis_source"] = source
            op_result["source"] = "AI_CONFIRMED"
            op_result["confirmation_status"] = "AI_CONFIRMED"
            op_result["ai_confirmed_at"] = now_iso
            op_result["candidate_only"] = False
        op_result["confirmed"] = confirmed
        op_result["production_ready"] = bool(op_result.get("status") == "READY" and op_result.get("executable") is True and confirmed)
        if not confirmed:
            op_result["candidate_only"] = True
            op_result["confirmation_status"] = op_result.get("confirmation_status") or "AI_CANDIDATE"
            op_result["execution_block_reason"] = "AI整備前の候補のため本番実行不可"
        else:
            op_result.pop("execution_block_reason", None)

        selectors = op_result.get("selectors") or {}
        form_schema = op_result.get("form_schema") or {}
        record = {
            "op": op,
            "status": op_result.get("status") or "UNDISCOVERED",
            "target_url": op_result.get("target_url") or "",
            "missing": op_result.get("missing", []),
            "selectors": selectors,
            "selectors_count": len(selectors),
            "validation_score": op_result.get("validation_score", 0),
            "executable": bool(op_result.get("status") == "READY" and steps and not failed_required and op_result.get("production_ready") is True),
            "confirmed": confirmed,
            "production_ready": op_result.get("production_ready") is True,
            "candidate_only": not confirmed,
            "confirmation_status": op_result.get("confirmation_status") or ("AI_CONFIRMED" if confirmed else "AI_CANDIDATE"),
            "steps": steps,
            "step_count": len(steps),
            "form_schema": form_schema,
            "form_fields_count": int(form_schema.get("fields_count") or len(form_schema.get("fields") or []) or 0) if isinstance(form_schema, dict) else 0,
            "scanned_at": now_iso,
            "source": op_result.get("source") or source,
            "analysis_source": op_result.get("analysis_source") or source,
        }
        records[op] = record
        normalized_maps[op] = op_result

    return records, normalized_maps


def _menu_item_updated_ops_from_records(records: dict, protected_by_op: dict | None = None) -> list[dict]:
    protected_by_op = protected_by_op or {}
    rows = []
    for op, rec in records.items():
        if not isinstance(rec, dict):
            continue
        rows.append({
            "op": op,
            "status": rec.get("status"),
            "target_url": rec.get("target_url"),
            "missing": rec.get("missing", []),
            "selectors": rec.get("selectors_count", len(rec.get("selectors") or {})),
            "steps": rec.get("step_count", len(rec.get("steps") or [])),
            "protected": bool(protected_by_op.get(op)),
            "confirmed": bool(rec.get("confirmed")),
            "production_ready": bool(rec.get("production_ready")),
            "candidate_only": bool(rec.get("candidate_only", not rec.get("confirmed"))),
        })
    return rows


def _menu_item_status_from_records(records: dict, scope_pages: list[dict]) -> str:
    vals = [v for v in (records or {}).values() if isinstance(v, dict)]
    if any(v.get("status") == "READY" and int(v.get("step_count") or 0) > 0 and v.get("production_ready") is True for v in vals):
        return "READY"
    if any(v.get("status") in ("READY", "NEEDS_REVIEW") for v in vals):
        return "NEEDS_REVIEW"
    if not scope_pages:
        return "NO_EDITABLE_DOM"
    return "NO_OPERATION"


def _menu_item_scan_diagnostics(
    target_url: str,
    status: str,
    message: str = "",
    dom_result: dict | None = None,
    scope_pages: list[dict] | None = None,
    operation_records: dict | None = None,
) -> dict:
    scope_pages = scope_pages or []
    operation_records = operation_records or {}
    followed_pages = (dom_result or {}).get("followed_pages") or []
    followed_urls = [
        p.get("url")
        for p in followed_pages
        if isinstance(p, dict) and p.get("url")
    ]

    def _page_counts(pg: dict) -> dict:
        return {
            "url": pg.get("url", ""),
            "forms": len(pg.get("forms") or []) or int(pg.get("forms_count") or 0),
            "inputs": len(pg.get("inputs") or []) or int(pg.get("inputs_count") or 0),
            "textareas": len(pg.get("textareas") or []) or int(pg.get("textareas_count") or 0),
            "buttons": len(pg.get("buttons") or []) or int(pg.get("buttons_count") or 0),
            "file_inputs": len(pg.get("file_inputs") or []) or int(pg.get("file_inputs_count") or 0),
            "selects": len(pg.get("selects") or []) or int(pg.get("selects_count") or 0),
            "page_purpose": pg.get("page_purpose", ""),
            "page_purpose_source": pg.get("page_purpose_source", ""),
        }

    if status == "FAILED":
        stop_stage = "dom_fetch"
        stop_reason = message or "DOM取得失敗"
    elif status == "NO_EDITABLE_DOM":
        stop_stage = "editable_dom_detection"
        stop_reason = "対象URLと追跡先に編集フォーム・入力欄・保存ボタン候補が見つかりません"
    elif status == "NO_OPERATION":
        stop_stage = "operation_mapping"
        stop_reason = "編集DOMは見つかりましたが、AI整備済みに必要なselector組み合わせが作れません"
    elif status == "NEEDS_REVIEW":
        stop_stage = "human_review"
        stop_reason = "候補はありますが、必須selectorまたは実行stepsが不足しています"
    else:
        stop_stage = "ready" if status == "READY" else "unknown"
        stop_reason = message or ""

    return {
        "stop_stage": stop_stage,
        "stop_reason": stop_reason,
        "target_url": target_url,
        "dom_status": (dom_result or {}).get("status", ""),
        "dom_message": (dom_result or {}).get("message", ""),
        "followed_count": len(followed_urls),
        "followed_urls": followed_urls[:20],
        "inspected_urls": [pg.get("url") for pg in scope_pages if isinstance(pg, dict) and pg.get("url")],
        "editable_pages_count": len(scope_pages),
        "page_evidence": [_page_counts(pg) for pg in scope_pages[:10] if isinstance(pg, dict)],
        "operations_checked": list(MENU_ITEM_OPERATION_TYPES),
        "operation_results": [
            {
                "op": op,
                "status": rec.get("status"),
                "missing": rec.get("missing", []),
                "target_url": rec.get("target_url", ""),
                "selectors_count": rec.get("selectors_count", len(rec.get("selectors") or {})),
                "step_count": rec.get("step_count", len(rec.get("steps") or [])),
            }
            for op, rec in operation_records.items()
            if isinstance(rec, dict)
        ],
    }


def _save_menu_item_scan_result(
    db,
    mapping_id: str,
    mapping: dict,
    target_url: str,
    item_patch: dict,
) -> dict:
    targets = _manual_menu_targets_from_mapping(mapping, max_urls=500)
    target_norm = _normalize_menu_scan_url(target_url)
    target_meta = next((t for t in targets if _normalize_menu_scan_url(t.get("url") or "") == target_norm), {})
    scan = mapping.get("manual_menu_scan_results") or {}
    items = list(scan.get("items") or [])

    found = False
    for idx, item in enumerate(items):
        if isinstance(item, dict) and _normalize_menu_scan_url(item.get("canonical_url") or item.get("url") or "") == target_norm:
            next_item = dict(item)
            next_item.update(item_patch)
            next_item["canonical_url"] = target_norm
            items[idx] = next_item
            found = True
            break

    if not found:
        items.append({
            "url": target_url,
            "canonical_url": target_norm,
            "title": target_meta.get("title") or target_url,
            "category": target_meta.get("category") or "その他",
            **item_patch,
        })

    now_iso = datetime.datetime.utcnow().isoformat()
    started_at = scan.get("started_at") or now_iso
    summary = _menu_scan_summary(items)
    _update_menu_scan_document(
        db,
        mapping_id,
        {
            "items": items,
            "summary": summary,
            "started_at": started_at,
            "updated_at": now_iso,
        },
        {"updated_at": datetime.datetime.utcnow()},
    )
    return {"items": items, "summary": summary}


def _run_menu_items_deep_scan_bg(mapping_id: str, tenant_id: str, options: dict,
                                 start_index: int = 0, chunk_size: Optional[int] = None):
    db = get_db()
    import datetime as _dt_menu_full
    print(f"[MENU_ITEMS_DEEP_SCAN_BG_START] mapping_id={mapping_id} start_index={start_index} chunk_size={chunk_size}", flush=True)

    max_urls = max(1, min(int(options.get("max_urls") or 200), 500))
    max_follow = max(0, min(int(options.get("max_follow_per_url") or 50), 100))
    force_rescan = bool(options.get("force_rescan", False))
    offset = max(0, int(start_index or 0))

    try:
        doc = db.collection("media_mappings").document(mapping_id).get()
        if not doc.exists:
            print(f"[MENU_ITEMS_DEEP_SCAN_BG_ABORT] mapping_not_found mapping_id={mapping_id}", flush=True)
            return
        mapping = doc.to_dict() or {}
        mapping["id"] = mapping_id
        if mapping.get("tenant_id") != tenant_id:
            print(f"[MENU_ITEMS_DEEP_SCAN_BG_ABORT] tenant_mismatch mapping_id={mapping_id}", flush=True)
            return

        targets = _manual_menu_targets_from_mapping(mapping, max_urls=max_urls)
        if not targets:
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress": {
                    "status": "DONE",
                    "kind": "menu_items_deep_scan",
                    "done": 0,
                    "total": 0,
                    "updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    "message": "manual_menu_items_empty",
                }
            })
            return {"ok": True, "status": "DONE", "done": 0, "total": 0, "next_offset": None, "finished": True}

        total = len(targets)
        end = total if not chunk_size else min(total, offset + max(1, int(chunk_size)))
        is_final_chunk = end >= total

        existing_items = ((mapping.get("manual_menu_scan_results") or {}).get("items") or [])
        by_url = {
            item.get("url"): item
            for item in existing_items
            if isinstance(item, dict) and item.get("url")
        }
        result_items = []
        for t in targets:
            prior = dict(by_url.get(t["url"]) or {})
            prior.setdefault("url", t["url"])
            prior.setdefault("title", t.get("title", ""))
            prior.setdefault("category", t.get("category", ""))
            result_items.append(prior)
        result_by_url = {item["url"]: item for item in result_items}

        from api.core.browser_executor import (
            fetch_dom_for_url,
            build_operation_mappings_from_dom_evidence,
        )

        _last_error_url = ""
        _last_error_msg = ""

        if offset == 0:
            started = _dt_menu_full.datetime.utcnow()
            _update_menu_scan_document(
                db,
                mapping_id,
                {
                    "items": result_items,
                    "summary": _menu_scan_summary(result_items),
                    "started_at": started.isoformat(),
                    "updated_at": started.isoformat(),
                },
                {
                    "scan_progress": {
                    "status": "RUNNING",
                    "kind": "menu_items_deep_scan",
                    "done": 0,
                    "total": total,
                    "updated_at": started.isoformat(),
                    "current_url": "",
                    "follow_limit": max_follow,
                    },
                },
            )
        else:
            started = _parse_progress_time(
                (mapping.get("manual_menu_scan_results") or {}).get("started_at")
            ) or _dt_menu_full.datetime.utcnow()

        for idx, target in enumerate(targets):
            if not (offset <= idx < end):
                continue
            url = target["url"]
            now = _dt_menu_full.datetime.utcnow()
            current_item = result_by_url.get(url) or {
                "url": url,
                "title": target.get("title", ""),
                "category": target.get("category", ""),
            }

            try:
                _assert_url_in_mapping_scope(mapping, url, "target_url")
            except HTTPException as scope_error:
                current_item.update({
                    "status": "FAILED",
                    "updated_ops": [],
                    "operations": {},
                    "message": scope_error.detail,
                    "diagnostics": _menu_item_scan_diagnostics(
                        url,
                        "FAILED",
                        message=scope_error.detail,
                        dom_result={"status": "BLOCKED_OUT_OF_SCOPE"},
                        scope_pages=[],
                        operation_records={},
                    ),
                    "followed_count": 0,
                    "scanned_at": _dt_menu_full.datetime.utcnow().isoformat(),
                })
                result_by_url[url] = current_item
                result_items = [result_by_url[t["url"]] for t in targets]
                summary = _menu_scan_summary(result_items)
                _update_menu_scan_document(
                    db,
                    mapping_id,
                    {
                        "items": result_items,
                        "summary": summary,
                        "started_at": started.isoformat(),
                        "updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    },
                    {
                        "scan_progress": {
                        "status": "RUNNING",
                        "kind": "menu_items_deep_scan",
                        "done": idx + 1,
                        "total": len(targets),
                        "current_url": url,
                        "updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                        "follow_limit": max_follow,
                        },
                    },
                )
                continue

            if current_item.get("status") in ("READY", "NEEDS_REVIEW") and not force_rescan:
                db.collection("media_mappings").document(mapping_id).update({
                    "scan_progress.done": idx + 1,
                    "scan_progress.current_url": url,
                    "scan_progress.updated_at": now.isoformat(),
                })
                continue

            print(f"[MENU_ITEMS_DEEP_SCAN_URL] ({idx+1}/{len(targets)}) {url[:90]}", flush=True)
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress.status": "RUNNING",
                "scan_progress.kind": "menu_items_deep_scan",
                "scan_progress.done": idx,
                "scan_progress.total": len(targets),
                "scan_progress.current_url": url,
                "scan_progress.updated_at": now.isoformat(),
            })

            try:
                latest = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                latest["id"] = mapping_id
                dom_result = fetch_dom_for_url(latest, url, max_follow_urls=max_follow)
                if dom_result.get("status") != "OK":
                    _fail_msg = dom_result.get("message") or dom_result.get("status") or "DOM取得失敗"
                    _last_error_url = url
                    _last_error_msg = _fail_msg
                    current_item.update({
                        "status": "FAILED",
                        "updated_ops": [],
                        "operations": {},
                        "message": _fail_msg,
                        "diagnostics": _menu_item_scan_diagnostics(
                            url,
                            "FAILED",
                            message=_fail_msg,
                            dom_result=dom_result,
                            scope_pages=[],
                            operation_records={},
                        ),
                        "followed_count": 0,
                        "scanned_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    })
                else:
                    latest = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                    latest["id"] = mapping_id
                    pages = (latest.get("navigation_graph") or {}).get("pages") or []
                    followed_pages = dom_result.get("followed_pages") or []
                    in_memory_pages = list(pages)
                    if isinstance(dom_result.get("page_data"), dict):
                        in_memory_pages.append(dom_result["page_data"])
                    in_memory_pages.extend([p for p in followed_pages if isinstance(p, dict)])
                    followed_urls = {
                        p.get("url")
                        for p in followed_pages
                        if isinstance(p, dict) and p.get("url")
                    }
                    followed_url_norms = {_normalize_menu_scan_url(u) for u in followed_urls}
                    target_norm = _normalize_menu_scan_url(url)
                    scope_pages = []
                    for pg in in_memory_pages:
                        if not isinstance(pg, dict):
                            continue
                        pg_url = pg.get("url")
                        pg_norm = _normalize_menu_scan_url(pg_url)
                        followed_from_norm = _normalize_menu_scan_url(pg.get("followed_from") or "")
                        if pg_norm == target_norm or pg_norm in followed_url_norms or followed_from_norm == target_norm:
                            if _agent_page_has_editable_dom(pg):
                                scope_pages.append(pg)

                    seen_scope = set()
                    scope_pages = [
                        pg for pg in scope_pages
                        if pg.get("url") and not (pg.get("url") in seen_scope or seen_scope.add(pg.get("url")))
                    ]

                    updated_ops = []
                    if scope_pages:
                        all_mappings = build_operation_mappings_from_dom_evidence(mapping_id, scope_pages)
                        operation_records, local_op_maps = _menu_item_build_operation_records(
                            mapping_id,
                            scope_pages,
                            all_mappings=all_mappings,
                            source="manual_menu_full_scan",
                        )
                        existing_ops = latest.get("operation_mappings") or {}
                        op_update = {}
                        protected_by_op = {}
                        for op, op_result in local_op_maps.items():
                            existing = existing_ops.get(op) or {}
                            protected = _operation_mapping_is_production_ready(existing)
                            protected_by_op[op] = protected
                            if not protected:
                                op_update[f"operation_mappings.{op}"] = op_result
                        updated_ops = _menu_item_updated_ops_from_records(operation_records, protected_by_op)

                        if op_update:
                            op_update["updated_at"] = _dt_menu_full.datetime.utcnow()
                            db.collection("media_mappings").document(mapping_id).update(op_update)
                    else:
                        operation_records = {}

                    item_status = _menu_item_status_from_records(operation_records, scope_pages)
                    item_message = "" if updated_ops else ("フォーム候補なし" if not scope_pages else "operation候補なし")
                    current_item.update({
                        "status": item_status,
                        "updated_ops": updated_ops,
                        "operations": operation_records,
                        "structure_pages": scope_pages,
                        "scope_urls": [pg.get("url") for pg in scope_pages if pg.get("url")],
                        "followed_count": len(followed_urls),
                        "message": item_message,
                        "diagnostics": _menu_item_scan_diagnostics(
                            url,
                            item_status,
                            message=item_message,
                            dom_result=dom_result,
                            scope_pages=scope_pages,
                            operation_records=operation_records,
                        ),
                        "scanned_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    })

            except Exception as e:
                import traceback as _tb_menu_scan
                print(f"[MENU_ITEMS_DEEP_SCAN_URL_ERROR] url={url[:80]} error={type(e).__name__}:{e}", flush=True)
                print(_tb_menu_scan.format_exc(), flush=True)
                _last_error_url = url
                _last_error_msg = f"{type(e).__name__}: {e}"
                current_item.update({
                    "status": "FAILED",
                    "updated_ops": [],
                    "operations": {},
                    "message": str(e),
                    "diagnostics": _menu_item_scan_diagnostics(
                        url,
                        "FAILED",
                        message=str(e),
                        dom_result={},
                        scope_pages=[],
                        operation_records={},
                    ),
                    "followed_count": 0,
                    "scanned_at": _dt_menu_full.datetime.utcnow().isoformat(),
                })

            result_by_url[url] = current_item
            result_items = [result_by_url[t["url"]] for t in targets]
            summary = _menu_scan_summary(result_items)
            _update_menu_scan_document(
                db,
                mapping_id,
                {
                    "items": result_items,
                    "summary": summary,
                    "started_at": started.isoformat(),
                    "updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                },
                {
                    "scan_progress": {
                    "status": "RUNNING",
                    "kind": "menu_items_deep_scan",
                    "done": idx + 1,
                    "total": len(targets),
                    "current_url": url,
                    "updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    "follow_limit": max_follow,
                    "last_error_url": _last_error_url,
                    "last_error_msg": _last_error_msg,
                    },
                },
            )

        if not is_final_chunk:
            # More targets remain: persist progress and let the frontend drive the
            # next chunk. Status stays RUNNING (recent timestamp) so a *different*
            # new scan is still blocked, while the continuation reuses next_offset.
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress.status": "RUNNING",
                "scan_progress.kind": "menu_items_deep_scan",
                "scan_progress.done": end,
                "scan_progress.total": total,
                "scan_progress.next_offset": end,
                "scan_progress.current_url": "",
                "scan_progress.updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
            })
            print(f"[MENU_ITEMS_DEEP_SCAN_CHUNK_DONE] mapping_id={mapping_id} {offset}->{end}/{total}", flush=True)
            return {"ok": True, "status": "RUNNING", "done": end, "total": total, "next_offset": end, "finished": False}

        _sync_ready_operation_steps(mapping_id, db)
        final_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        final_items = ((final_doc.get("manual_menu_scan_results") or {}).get("items") or result_items)
        final_summary = _menu_scan_summary(final_items)
        finished = _dt_menu_full.datetime.utcnow()
        _update_menu_scan_document(
            db,
            mapping_id,
            {
                "items": final_items,
                "summary": final_summary,
                "started_at": started.isoformat(),
                "updated_at": finished.isoformat(),
                "finished_at": finished.isoformat(),
            },
            {
                "scan_progress": {
                "status": "DONE",
                "kind": "menu_items_deep_scan",
                "done": total,
                "total": total,
                "current_url": "",
                "next_offset": None,
                "updated_at": finished.isoformat(),
                "follow_limit": max_follow,
                "health_status": final_summary.get("health_status"),
                "summary": final_summary,
                },
                "updated_at": finished,
            },
        )
        _rebuild_media_schema_for_mapping(db, mapping_id)
        print(f"[MENU_ITEMS_DEEP_SCAN_BG_DONE] mapping_id={mapping_id} summary={final_summary}", flush=True)

        # C検証: 全URLスキャン完了後、operation_mappingsのtarget_urlを自動確認
        try:
            from api.core.browser_executor import run_operation_url_verification as _run_verify
            _verify_mapping = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _verify_mapping["id"] = mapping_id
            _op_types_to_verify = [
                op for op, op_data in (_verify_mapping.get("operation_mappings") or {}).items()
                if op_data.get("target_url") and op_data.get("status") in ("READY", "NEEDS_REVIEW")
            ]
            if _op_types_to_verify:
                print(f"[C_VERIFY_START] mapping_id={mapping_id} ops={_op_types_to_verify}", flush=True)
                db.collection("media_mappings").document(mapping_id).update({
                    "scan_progress.status": "VERIFYING",
                    "scan_progress.kind": "operation_url_verification",
                    "scan_progress.updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                })
                _vresult = _run_verify(_verify_mapping, operation_types=_op_types_to_verify)
                db.collection("media_mappings").document(mapping_id).update({
                    "scan_progress.status": "DONE",
                    "scan_progress.kind": "operation_url_verification",
                    "scan_progress.verification_summary": {
                        "verified":     _vresult.get("verified", 0),
                        "corrected":    _vresult.get("corrected", 0),
                        "needs_review": _vresult.get("needs_review", 0),
                    },
                    "scan_progress.updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                })
                print(f"[C_VERIFY_DONE] mapping_id={mapping_id} result={_vresult.get('status')} "
                      f"verified={_vresult.get('verified',0)} corrected={_vresult.get('corrected',0)} "
                      f"needs_review={_vresult.get('needs_review',0)}", flush=True)
        except Exception as _ve:
            print(f"[C_VERIFY_ERROR] mapping_id={mapping_id} err={type(_ve).__name__}:{_ve}", flush=True)

        return {"ok": True, "status": "DONE", "done": total, "total": total, "next_offset": None, "finished": True}
    except Exception as e:
        print(f"[MENU_ITEMS_DEEP_SCAN_BG_ERROR] mapping_id={mapping_id} error={type(e).__name__}:{e}", flush=True)
        try:
            db.collection("media_mappings").document(mapping_id).update({
                "scan_progress.status": "FAILED",
                "scan_progress.kind": "menu_items_deep_scan",
                "scan_progress.error": str(e),
                "scan_progress.updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
            })
        except Exception as update_error:
            if "maximum allowed size" in str(update_error):
                try:
                    db.collection("media_mappings").document(mapping_id).update({
                        "navigation_graph.pages": [],
                        "navigation_graph.storage_mode": "cleared_after_scan_error",
                        "scan_progress.status": "FAILED",
                        "scan_progress.kind": "menu_items_deep_scan",
                        "scan_progress.error": str(e),
                        "scan_progress.updated_at": _dt_menu_full.datetime.utcnow().isoformat(),
                    })
                except Exception:
                    pass
        return {"ok": False, "status": "FAILED", "error": str(e), "next_offset": None, "finished": True}


@router.post("/media/map/{mapping_id}/menu_items/deep_scan")
def menu_items_deep_scan(
    mapping_id: str,
    req: Optional[MenuItemsDeepScanRequest] = Body(None),
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")
    req = req or MenuItemsDeepScanRequest()
    offset = max(0, int(req.offset or 0))
    # Only the first chunk takes the lock; later chunks continue the same scan.
    if offset == 0:
        _guard_mapping_scan_not_running(db, mapping_id, mapping, "menu_items_deep_scan", stale_minutes=5)

    targets = _manual_menu_targets_from_mapping(mapping, max_urls=req.max_urls)
    if not targets:
        raise HTTPException(status_code=400, detail="HTMLメニュー由来のURLがありません")

    # Process one chunk synchronously so it completes under CPU throttling.
    # The frontend re-calls with next_offset until finished.
    result = _run_menu_items_deep_scan_bg(
        mapping_id,
        tenant_id,
        {
            "max_urls": req.max_urls,
            "max_follow_per_url": req.max_follow_per_url,
            "force_rescan": req.force_rescan,
        },
        start_index=offset,
        chunk_size=req.chunk_size,
    )
    if isinstance(result, dict):
        result.setdefault("mapping_id", mapping_id)
        result.setdefault("targets_count", len(targets))
    return result


@router.post("/media/map/{mapping_id}/verify_operation_urls")
def verify_operation_urls(
    mapping_id: str,
    operation_types: Optional[list] = Body(None),
    user: dict = Depends(verify_token),
):
    """
    C検証: 保存済みoperation_mappingsのtarget_urlが正しいフォームページか実ブラウザで確認。
    - VERIFIED: そのURLで正しいフォームを確認
    - URL_CORRECTED: リンク追跡で正しいURLを自動発見・更新
    - NEEDS_REVIEW: フォーム未検出 → スクリーンショット付きで人間確認
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and _resolve_agent_user_context(user)["role"] != "admin":
        raise HTTPException(status_code=403, detail="アクセス権がありません")
    from api.core.browser_executor import run_operation_url_verification
    result = run_operation_url_verification(mapping, operation_types=operation_types)
    result.setdefault("mapping_id", mapping_id)
    return result


@router.get("/media/map/{mapping_id}/verification_reviews")
def get_verification_reviews(
    mapping_id: str,
    user: dict = Depends(verify_token),
):
    """
    NEEDS_REVIEW状態のoperation一覧をスクリーンショットURL付きで返す。
    管理者がURLを手修正する前にこのエンドポイントで確認対象を把握する。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    op_maps = mapping.get("operation_mappings") or {}
    reviews = []
    for op_type, op_data in op_maps.items():
        v = op_data.get("verification") or {}
        reviews.append({
            "operation_type":  op_type,
            "current_url":     op_data.get("target_url") or op_data.get("url") or "",
            "original_url":    v.get("original_url") or "",
            "verification_status": v.get("status") or "UNVERIFIED",
            "screenshot_url":  v.get("screenshot_url") or "",
            "verified_at":     v.get("verified_at") or "",
            "reason":          v.get("reason") or "",
            "op_status":       op_data.get("status") or "",
        })
    # NEEDS_REVIEW → UNVERIFIED → VERIFIED の順に並べる
    _order = {"NEEDS_REVIEW": 0, "UNVERIFIED": 1, "URL_CORRECTED": 2, "VERIFIED": 3}
    reviews.sort(key=lambda r: _order.get(r["verification_status"], 9))
    return {
        "mapping_id": mapping_id,
        "media_name": mapping.get("media_name") or "",
        "reviews": reviews,
        "needs_review_count": sum(1 for r in reviews if r["verification_status"] == "NEEDS_REVIEW"),
        "unverified_count":   sum(1 for r in reviews if r["verification_status"] == "UNVERIFIED"),
    }


class OperationUrlPatchRequest(BaseModel):
    operation_type: str
    target_url: str


@router.patch("/media/map/{mapping_id}/operation_url")
def patch_operation_url(
    mapping_id: str,
    req: OperationUrlPatchRequest,
    user: dict = Depends(verify_token),
):
    """
    管理者がNEEDS_REVIEWのoperation_urlを手修正して保存する。
    保存後にC検証を再実行してVERIFIED/NEEDS_REVIEWを更新する。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.target_url.startswith("http"):
        raise HTTPException(status_code=400, detail="target_urlはhttpから始まる必要があります")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != ctx["tenant_id"] and not ctx["is_admin"]:
        raise HTTPException(status_code=403, detail="アクセス権がありません")

    op_type = req.operation_type
    if op_type not in (mapping.get("operation_mappings") or {}):
        raise HTTPException(status_code=400, detail=f"operation_type '{op_type}' が存在しません")

    # URL を更新
    db.collection("media_mappings").document(mapping_id).update({
        f"operation_mappings.{op_type}.target_url":             req.target_url,
        f"operation_mappings.{op_type}.target_url_manual_set":  True,
        f"operation_mappings.{op_type}.verification.status":    "PENDING_REVERIFY",
        "updated_at": datetime.datetime.utcnow(),
    })

    # 手修正後に即C検証を再実行
    try:
        from api.core.browser_executor import run_operation_url_verification as _reverify
        updated_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        updated_doc["id"] = mapping_id
        v_result = _reverify(updated_doc, operation_types=[op_type])
        op_v = v_result.get("results", {}).get(op_type) or {}
        return {
            "ok": True,
            "operation_type":       op_type,
            "target_url":           req.target_url,
            "verification_status":  op_v.get("status") or "UNKNOWN",
            "screenshot_url":       op_v.get("screenshot_url") or "",
            "verified_at":          op_v.get("verified_at") or "",
        }
    except Exception as _ve:
        print(f"[PATCH_OP_URL_REVERIFY_ERROR] op={op_type} err={type(_ve).__name__}", flush=True)
        return {
            "ok": True,
            "operation_type": op_type,
            "target_url":     req.target_url,
            "verification_status": "PENDING_REVERIFY",
            "note": "URL保存済み。検証は次回スキャン時に実行されます。",
        }


@router.post("/media/map/{mapping_id}/menu_item/deep_scan")
def menu_item_deep_scan(
    mapping_id: str,
    req: MenuItemDeepScanRequest,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.target_url or not req.target_url.startswith("http"):
        raise HTTPException(status_code=400, detail="target_urlが不正です")

    db = get_db()
    tenant_id = _resolve_agent_user_context(user)["tenant_id"]
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")
    _assert_url_in_mapping_scope(mapping, req.target_url, "target_url")

    import datetime as _dt_mi
    from api.core.browser_executor import (
        fetch_dom_for_url,
        build_operation_mappings_from_dom_evidence,
        rebuild_operation_steps,
    )

    # ── Step1: 指定URL1件のDOM取得 ──
    print(f"[MENU_ITEM_DEEP_SCAN_START] mapping_id={mapping_id} url={req.target_url[:80]}", flush=True)
    _single_follow_limit = max(0, min(int(req.max_follow_per_url or 50), 100))
    dom_result = fetch_dom_for_url(mapping, req.target_url, max_follow_urls=_single_follow_limit)
    if dom_result.get("status") not in ("OK",):
        _fail_msg = f"DOM取得失敗: {dom_result.get('message', dom_result.get('status', 'unknown'))}"
        _save_menu_item_scan_result(db, mapping_id, mapping, req.target_url, {
            "status": "FAILED",
            "updated_ops": [],
            "operations": {},
            "message": _fail_msg,
            "diagnostics": _menu_item_scan_diagnostics(
                req.target_url,
                "FAILED",
                message=_fail_msg,
                dom_result=dom_result,
                scope_pages=[],
                operation_records={},
            ),
            "followed_count": 0,
            "scanned_at": _dt_mi.datetime.utcnow().isoformat(),
        })
        return {
            "ok": False,
            "target_url": req.target_url,
            "message": f"DOM取得失敗: {dom_result.get('message', dom_result.get('status', 'unknown'))}",
            "updated_ops": [],
        }

    # ── Step2: 最新pagesを読み込み全8op一括抽出 ──
    _cur_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
    _cur_doc["id"] = mapping_id
    _pages = (_cur_doc.get("navigation_graph") or {}).get("pages") or []
    if isinstance(dom_result.get("page_data"), dict):
        _pages = list(_pages) + [dom_result["page_data"]]
    _pages = list(_pages) + [p for p in (dom_result.get("followed_pages") or []) if isinstance(p, dict)]

    # 対象ページのDOMが取れているか確認
    _target_norm = _normalize_menu_scan_url(req.target_url)
    _target_pg = next((p for p in _pages if _normalize_menu_scan_url(p.get("url") if isinstance(p, dict) else "") == _target_norm), None)
    _followed_pages = dom_result.get("followed_pages") or []
    _followed_urls = {
        p.get("url")
        for p in _followed_pages
        if isinstance(p, dict) and p.get("url")
    }
    _followed_url_norms = {_normalize_menu_scan_url(u) for u in _followed_urls}

    def _has_editable_dom_page(_pg: dict | None) -> bool:
        if not _pg:
            return False
        return bool(
            _pg.get("forms")
            or _pg.get("inputs")
            or _pg.get("buttons")
            or _pg.get("textareas")
            or _pg.get("file_inputs")
            or _pg.get("selects")
            or int(_pg.get("forms_count") or 0) > 0
            or int(_pg.get("inputs_count") or 0) > 0
            or int(_pg.get("buttons_count") or 0) > 0
            or int(_pg.get("textareas_count") or 0) > 0
            or int(_pg.get("file_inputs_count") or 0) > 0
            or int(_pg.get("selects_count") or 0) > 0
        )

    _scope_pages = []
    if _has_editable_dom_page(_target_pg):
        _scope_pages.append(_target_pg)
    for _pg in _pages:
        if _normalize_menu_scan_url(_pg.get("url") if isinstance(_pg, dict) else "") in _followed_url_norms and _has_editable_dom_page(_pg):
            _scope_pages.append(_pg)

    # URL完全一致で重複除去
    _seen_scope = set()
    _scope_pages_dedup = []
    for _pg in _scope_pages:
        _u = _pg.get("url")
        if not _u or _u in _seen_scope:
            continue
        _seen_scope.add(_u)
        _scope_pages_dedup.append(_pg)
    _scope_pages = _scope_pages_dedup

    if not _scope_pages:
        _no_dom_msg = "DOM取得後もフォーム・入力欄が検出できませんでした。このURLまたは追跡先は編集画面ではない可能性があります。"
        _save_menu_item_scan_result(db, mapping_id, _cur_doc, req.target_url, {
            "status": "NO_EDITABLE_DOM",
            "updated_ops": [],
            "operations": {},
            "scope_urls": [],
            "followed_count": len(_followed_urls),
            "message": _no_dom_msg,
            "diagnostics": _menu_item_scan_diagnostics(
                req.target_url,
                "NO_EDITABLE_DOM",
                message=_no_dom_msg,
                dom_result=dom_result,
                scope_pages=[],
                operation_records={},
            ),
            "scanned_at": _dt_mi.datetime.utcnow().isoformat(),
        })
        return {
            "ok": False,
            "target_url": req.target_url,
            "message": "DOM取得後もフォーム・入力欄が検出できませんでした。このURLまたは追跡先は編集画面ではない可能性があります。",
            "updated_ops": [],
        }

    all_mappings = build_operation_mappings_from_dom_evidence(mapping_id, _scope_pages)
    _operation_records, _local_op_maps = _menu_item_build_operation_records(
        mapping_id,
        _scope_pages,
        all_mappings=all_mappings,
        source="manual_menu_item_scan",
    )

    # ── Step3: このURLで有効なopのmappingを更新 ──
    _now_mi = _dt_mi.datetime.utcnow()
    _op_update = {}
    _existing_op_maps = _cur_doc.get("operation_mappings") or {}
    _protected_by_op = {}

    for op, op_result in _local_op_maps.items():
        # READY保護: AI整備済みの既存だけ保護し、旧READY候補は今回のAI整備結果で上書きする
        _existing = _existing_op_maps.get(op) or {}
        if _operation_mapping_is_production_ready(_existing):
            print(f"[MENU_ITEM_READY_PROTECT] op={op} already READY+executable -> skip", flush=True)
            _protected_by_op[op] = True
            continue

        _op_update[f"operation_mappings.{op}"] = op_result
        _protected_by_op[op] = False
        print(f"[MENU_ITEM_DEEP_SCAN_OP] op={op} status={op_result.get('status')} sel={len(op_result.get('selectors') or {})}", flush=True)

    _updated_ops = _menu_item_updated_ops_from_records(_operation_records, _protected_by_op)

    if _op_update:
        _op_update["updated_at"] = _now_mi
        db.collection("media_mappings").document(mapping_id).update(_op_update)

    _item_status = _menu_item_status_from_records(_operation_records, _scope_pages)
    _item_msg = "" if _updated_ops else "このURLではOperation候補が検出されませんでした"
    _save_menu_item_scan_result(db, mapping_id, _cur_doc, req.target_url, {
        "status": _item_status,
        "updated_ops": _updated_ops,
        "operations": _operation_records,
        "structure_pages": _scope_pages,
        "scope_urls": [pg.get("url") for pg in _scope_pages if pg.get("url")],
        "followed_count": len(_followed_urls),
        "message": _item_msg,
        "diagnostics": _menu_item_scan_diagnostics(
            req.target_url,
            _item_status,
            message=_item_msg,
            dom_result=dom_result,
            scope_pages=_scope_pages,
            operation_records=_operation_records,
        ),
        "scanned_at": _now_mi.isoformat(),
    })

    # ── Step4: steps即時再生成 ──
    _sync_ready_operation_steps(mapping_id, db)
    _schema_mi = _rebuild_media_schema_for_mapping(db, mapping_id)

    # 最新状態を返却
    _final_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
    _final_ops = _final_doc.get("operation_mappings") or {}
    _final_steps = _final_doc.get("operation_steps_by_type") or {}

    print(f"[MENU_ITEM_DEEP_SCAN_DONE] mapping_id={mapping_id} url={req.target_url[:60]} updated={[u['op'] for u in _updated_ops]}", flush=True)

    return {
        "ok": True,
        "target_url": req.target_url,
        "max_follow_per_url": _single_follow_limit,
        "followed_count": len(_followed_urls),
        "updated_ops": _updated_ops,
        "operation_mappings": {
            op: {
                "status": v.get("status"),
                "executable": v.get("executable"),
                "selectors": len(v.get("selectors") or {}),
                "missing": v.get("missing", []),
                "steps": len(_final_steps.get(op) or []),
            }
            for op, v in _final_ops.items()
        },
        "media_schema_summary": {
            "forms_count": _schema_mi.get("forms_count", 0),
            "entities_count": _schema_mi.get("entities_count", 0),
            "canonical_fields_count": _schema_mi.get("canonical_fields_count", 0),
        },
        "message": f"{len(_updated_ops)}件のOperationを更新しました" if _updated_ops else "このURLではOperation候補が検出されませんでした",
    }


@router.post("/media/map/{mapping_id}/menu_item/task/create")
def create_menu_item_task(
    mapping_id: str,
    req: MenuItemTaskCreateRequest,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.target_url or not req.target_url.startswith("http"):
        raise HTTPException(status_code=400, detail="target_urlが不正です")
    if req.operation_type not in OPERATION_TYPES:
        raise HTTPException(status_code=400, detail="無効なoperation_typeです")

    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    _enforce_agent_permissions(ctx, "hp_update", req.operation_type)

    db = get_db()
    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    mapping["id"] = mapping_id
    if mapping.get("tenant_id") != tenant_id and user.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    scan = mapping.get("manual_menu_scan_results") or {}
    items = scan.get("items") or []
    item = next((it for it in items if isinstance(it, dict) and it.get("url") == req.target_url), None)
    detail_item = _get_menu_scan_item_document(db, mapping_id, req.target_url)
    if detail_item:
        item = {**(item or {}), **detail_item}
    operations = (item or {}).get("operations") or {}
    operation = operations.get(req.operation_type) or {}

    if operation.get("status") != "READY" or not operation.get("steps"):
        scope_pages = _menu_item_scope_pages(mapping, req.target_url, (item or {}).get("scope_urls") or [])
        if scope_pages:
            records, _ = _menu_item_build_operation_records(
                mapping_id,
                scope_pages,
                source="manual_menu_item_task_rebuild",
            )
            operation = records.get(req.operation_type) or {}
            patch = {
                "status": _menu_item_status_from_records(records, scope_pages),
                "updated_ops": _menu_item_updated_ops_from_records(records, {}),
                "operations": records,
                "structure_pages": scope_pages,
                "scope_urls": [pg.get("url") for pg in scope_pages if pg.get("url")],
                "followed_count": len([pg for pg in scope_pages if pg.get("followed_from") == req.target_url]),
                "message": "" if records else "operation候補なし",
                "scanned_at": datetime.datetime.utcnow().isoformat(),
            }
            _save_menu_item_scan_result(db, mapping_id, mapping, req.target_url, patch)
            item = {
                **(item or {}),
                "url": req.target_url,
                **patch,
            }

    steps = operation.get("steps") or []
    if operation.get("status") != "READY" or not steps or operation.get("production_ready") is not True:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "このHTMLメニューURLはAI整備済みの実行対象ではありません。媒体基盤のAI整備で対象ページ・入力項目・保存操作を保存してください。",
                "target_url": req.target_url,
                "operation_type": req.operation_type,
                "status": operation.get("status") or (item or {}).get("status") or "UNDISCOVERED",
                "missing": operation.get("missing", []),
            },
        )

    payload = dict(req.payload or {})
    payload["media_mapping_id"] = mapping_id
    payload["media_name"] = mapping.get("media_name", "")
    payload["menu_item_target_url"] = req.target_url
    payload["target_url"] = req.target_url
    payload["menu_item_title"] = (item or {}).get("title", "")
    payload["menu_item_category"] = (item or {}).get("category", "")

    now = datetime.datetime.utcnow()
    task_id = str(uuid.uuid4())
    workflow_id = str(uuid.uuid4())
    industry = _normalize_industry(mapping.get("industry") or "generic")
    preview = _build_preview(
        "hp_update",
        req.operation_type,
        industry,
        payload,
        operation_steps=steps,
        before_values={},
    )
    operation_override = {
        "status": "READY",
        "target_url": operation.get("target_url") or req.target_url,
        "selectors": operation.get("selectors") or {},
        "missing": operation.get("missing", []),
        "validation_score": operation.get("validation_score", 0),
        "executable": True,
        "source": "TASK_OVERRIDE",
        "confirmed": True,
        "production_ready": True,
        "confirmation_status": "AI_CONFIRMED",
        "last_scanned_at": operation.get("scanned_at") or now.isoformat(),
    }
    workflow_session_id, workflow_risk = _create_task_workflow_session(
        db=db,
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        operation_type=req.operation_type,
        operation_steps=steps,
        media_mapping=mapping,
        media_mapping_id=mapping_id,
        media_name=mapping.get("media_name", ""),
        goal_context="manual_menu_item",
    )
    task = {
        "task_id": task_id,
        "tenant_id": tenant_id,
        "user_uid": user.get("uid", ""),
        "agent_type": "hp_update",
        "operation_type": req.operation_type,
        "industry": industry,
        "entity_type": "",
        "op_id": "",
        "op_snapshot": {},
        "status": "PENDING",
        "payload": payload,
        "preview": preview,
        "operation_steps": steps,
        "operation_mapping_override": operation_override,
        "approved_by": None,
        "approved_at": None,
        "scheduled_at": req.scheduled_at,
        "result": None,
        "created_at": now,
        "media_mapping_id": mapping_id,
        "workflow_session_id": workflow_session_id,
        "risk_level": workflow_risk.get("risk_level", ""),
        "risk_score": workflow_risk.get("risk_score", 0.0),
        "risk_factors": workflow_risk.get("risk_factors", []),
        "require_human_approval": workflow_risk.get("require_human_approval", False),
        "workflow_id": workflow_id,
        "chain_id": "",
        "parent_task_id": "",
        "depends_on": [],
        "previous_operation": "",
        "next_operation_candidates": [],
        "menu_item_target_url": req.target_url,
        "menu_item_title": (item or {}).get("title", ""),
        "menu_item_category": (item or {}).get("category", ""),
        "source": "manual_menu_item",
    }
    db.collection("agent_tasks").document(task_id).set(task)

    return {
        "task_id": task_id,
        "status": "PENDING",
        "preview": preview,
        "mapping_id": mapping_id,
        "media_name": mapping.get("media_name", ""),
        "target_url": req.target_url,
        "operation_type": req.operation_type,
        "step_count": len(steps),
        "workflow_session_id": workflow_session_id,
        "risk_level": workflow_risk.get("risk_level", ""),
        "require_human_approval": workflow_risk.get("require_human_approval", False),
    }


@router.post("/task/from_instruction")
def create_task_from_instruction(
    req: InstructionTaskCreateRequest,
    user: dict = Depends(verify_token),
):
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.instruction or not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instructionが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]
    plan = plan_agent_task(PlanRequest(instruction=req.instruction, mapping_id=req.mapping_id), user)
    if not plan.get("ready") or not plan.get("operation_type"):
        return {
            "ok": True,
            "created": False,
            "status": "NEEDS_INFO",
            "question": plan.get("question") or "対象媒体、操作内容、更新内容を指定してください。",
            "plan": plan,
        }

    op = plan.get("operation_type")
    if op == "interview_assist":
        payload = dict(plan.get("payload") or {})
        payload.update(req.payload or {})
        payload.setdefault("use_case", req.instruction)
        created = create_task(
            TaskCreateRequest(
                agent_type="interview",
                operation_type="interview_assist",
                industry="generic",
                entity_type="candidate",
                op_id="" if str(plan.get("op_id") or "").startswith("default_") else (plan.get("op_id") or ""),
                media_mapping_id=None,
                payload=payload,
                scheduled_at=req.scheduled_at,
            ),
            user,
        )
        return {
            "ok": True,
            "created": True,
            "source": "interview_assist",
            "operation_type": op,
            "plan": plan,
            **created,
        }
    agent_type_for_op = "post_monitoring" if op == "post_monitoring" else "hp_update"
    _enforce_agent_permissions(ctx, agent_type_for_op, op)
    payload = dict(plan.get("payload") or {})
    payload.update(req.payload or {})

    mappings = []
    for d in db.collection("media_mappings").where("tenant_id", "==", tenant_id).stream():
        m = d.to_dict() or {}
        m["mapping_id"] = m.get("mapping_id") or d.id
        m, _ = _ensure_capability_view_for_mapping(db, m["mapping_id"], m)
        m["mapping_id"] = m.get("mapping_id") or d.id
        mappings.append(m)
    if req.mapping_id:
        mappings = [m for m in mappings if m.get("mapping_id") == req.mapping_id]
        if not mappings:
            raise HTTPException(status_code=404, detail="media_mappingが見つかりません")

    media_hint = str(plan.get("media_name") or "").strip().lower()
    instruction_lower = req.instruction.lower()

    def _mapping_matches(m: dict) -> bool:
        if req.mapping_id:
            return m.get("mapping_id") == req.mapping_id
        if not media_hint:
            return False
        name = str(m.get("media_name") or "").lower()
        url = str(m.get("media_url") or "").lower()
        return bool(name and (name in media_hint or media_hint in name)) or bool(url and url in media_hint)

    if op == "post_monitoring":
        import re as _re_mon
        url_match = _re_mon.search(r"https?://\S+", req.instruction)
        if url_match and not payload.get("monitoring_target"):
            payload["monitoring_target"] = url_match.group(0).rstrip("。、)")
        if not payload.get("monitoring_target"):
            payload["monitoring_target"] = ""
        if not payload.get("monitoring_date"):
            payload["monitoring_date"] = ""
        if not payload.get("cast_names"):
            payload["cast_names"] = ""

        mapping_matches = [m for m in mappings if _mapping_matches(m)]
        if not mapping_matches and len(mappings) == 1 and not media_hint:
            mapping_matches = mappings
        if not mapping_matches and payload.get("monitoring_target"):
            payload["media_mapping_id"] = ""
            payload["media_name"] = "公開URL監視"
            created = create_task(
                TaskCreateRequest(
                    agent_type="post_monitoring",
                    operation_type="post_monitoring",
                    industry="generic",
                    entity_type="monitoring",
                    op_id="" if str(plan.get("op_id") or "").startswith("default_") else (plan.get("op_id") or ""),
                    media_mapping_id=None,
                    payload=payload,
                    scheduled_at=req.scheduled_at,
                ),
                user,
            )
            return {
                "ok": True,
                "created": True,
                "source": "public_url_monitoring",
                "media_name": "公開URL監視",
                "mapping_id": None,
                "operation_type": op,
                "plan": plan,
                **created,
            }
        if len(mapping_matches) == 1:
            m = mapping_matches[0]
            payload["media_mapping_id"] = m.get("mapping_id")
            payload["media_name"] = m.get("media_name", "")
            created = create_task(
                TaskCreateRequest(
                    agent_type="post_monitoring",
                    operation_type="post_monitoring",
                    industry=_normalize_industry(m.get("industry") or "generic"),
                    entity_type="monitoring",
                    op_id="" if str(plan.get("op_id") or "").startswith("default_") else (plan.get("op_id") or ""),
                    media_mapping_id=m.get("mapping_id"),
                    payload=payload,
                    scheduled_at=req.scheduled_at,
                ),
                user,
            )
            return {
                "ok": True,
                "created": True,
                "source": "post_monitoring",
                "media_name": m.get("media_name", ""),
                "mapping_id": m.get("mapping_id"),
                "operation_type": op,
                "plan": plan,
                **created,
            }
        if len(mapping_matches) > 1 or len(mappings) > 1:
            return {
                "ok": True,
                "created": False,
                "status": "NEEDS_MEDIA",
                "operation_type": op,
                "question": "投稿数を監視する媒体を一意に特定できません。媒体名、または監視URLを含めて指示してください。",
                "candidates": [
                    {"mapping_id": m.get("mapping_id"), "media_name": m.get("media_name", ""), "media_url": m.get("media_url", "")}
                    for m in mappings[:20]
                ],
                "plan": plan,
            }
        return {
            "ok": True,
            "created": False,
            "status": "NEEDS_MEDIA",
            "operation_type": op,
            "question": "投稿数監視には媒体マッピングまたは監視URLが必要です。",
            "plan": plan,
        }

    # ── スカウト系 op 専用ルート（credential＋site_purpose ベース）──
    _SCOUT_OPS_SET = {"offer_send", "recruit_inbox_scan", "recruit_reply"}
    if op in _SCOUT_OPS_SET:
        _sp_allowed = {
            "offer_send":         {"scout"},
            "recruit_inbox_scan": {"scout", "reply", "monitor"},
            "recruit_reply":      {"scout", "reply"},
        }[op]
        scout_candidates = [
            m for m in mappings
            if (m.get("business_conditions") or {}).get("site_purpose") in _sp_allowed
            and m.get("credential_secret_name")
            and (op == "recruit_reply" or _operation_mapping_is_production_ready(((m.get("operation_mappings") or {}).get(op) or {})))
        ]
        if not scout_candidates:
            return {
                "ok": True,
                "created": False,
                "status": "NEEDS_MEDIA",
                "operation_type": op,
                "question": f"{op} に対応するサイト（site_purpose={'/'.join(sorted(_sp_allowed))}）がありません。媒体マッピングでsite_purposeとログイン情報を設定してください。",
                "plan": plan,
            }
        m = scout_candidates[0]
        bc = m.get("business_conditions") or {}
        # offer_template があれば body をデフォルト注入
        if op == "offer_send" and not payload.get("body") and bc.get("offer_template"):
            payload["body"] = bc["offer_template"]
        payload.setdefault("media_mapping_id", m.get("mapping_id", ""))
        payload.setdefault("media_name", m.get("media_name", ""))
        created = create_task(
            TaskCreateRequest(
                agent_type="hp_update",
                operation_type=op,
                industry=_normalize_industry(m.get("industry") or "generic"),
                entity_type="recruit",
                op_id="",
                media_mapping_id=m.get("mapping_id"),
                payload=payload,
                scheduled_at=req.scheduled_at,
            ),
            user,
        )
        return {
            "ok": True,
            "created": True,
            "source": "scout_mapping",
            "media_name": m.get("media_name", ""),
            "mapping_id": m.get("mapping_id"),
            "operation_type": op,
            "plan": plan,
            **created,
        }

    ready_menu_items = []
    for m in mappings:
        scan_items = ((m.get("manual_menu_scan_results") or {}).get("items") or [])
        for item in scan_items:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            operation = ((item.get("operations") or {}).get(op) or {})
            if operation.get("status") != "READY" or not operation.get("steps") or operation.get("production_ready") is not True:
                continue
            ready_menu_items.append({
                "mapping": m,
                "url": item.get("url"),
                "title": item.get("title") or item.get("url"),
                "category": item.get("category") or "",
                "step_count": len(operation.get("steps") or []),
            })

    def _menu_item_matches(item: dict) -> bool:
        m = item["mapping"]
        if req.mapping_id and m.get("mapping_id") != req.mapping_id:
            return False
        media_name = str(m.get("media_name") or "").lower()
        media_url = str(m.get("media_url") or "").lower()
        title = str(item.get("title") or "").lower()
        url = str(item.get("url") or "").lower()
        category = str(item.get("category") or "").lower()
        if media_hint and (media_name in media_hint or media_hint in media_name or media_url in media_hint):
            return True
        if title and len(title) >= 2 and title in instruction_lower:
            return True
        if url and url in instruction_lower:
            return True
        if category and len(category) >= 3 and category in instruction_lower:
            return True
        return False

    menu_matches = [item for item in ready_menu_items if _menu_item_matches(item)]
    if not menu_matches and len(ready_menu_items) == 1 and not media_hint:
        menu_matches = ready_menu_items

    if len(menu_matches) == 1:
        item = menu_matches[0]
        m = item["mapping"]
        created = create_menu_item_task(
            m["mapping_id"],
            MenuItemTaskCreateRequest(
                target_url=item["url"],
                operation_type=op,
                payload=payload,
                scheduled_at=req.scheduled_at,
            ),
            user,
        )
        return {
            "ok": True,
            "created": True,
            "source": "menu_item",
            "plan": plan,
            **created,
        }

    if len(menu_matches) > 1 or (req.mapping_id and len(ready_menu_items) > 1 and not menu_matches):
        candidates = (menu_matches or ready_menu_items)[:20]
        return {
            "ok": True,
            "created": False,
            "status": "NEEDS_TARGET",
            "operation_type": op,
            "question": "対象リンクを一意に特定できません。リンク名、カテゴリ、URLのどれかを含めて指示してください。",
            "candidates": [
                {
                    "mapping_id": c["mapping"].get("mapping_id"),
                    "media_name": c["mapping"].get("media_name", ""),
                    "title": c.get("title", ""),
                    "category": c.get("category", ""),
                    "url": c.get("url", ""),
                    "step_count": c.get("step_count", 0),
                }
                for c in candidates
            ],
            "plan": plan,
        }

    ready_mappings = []
    for m in mappings:
        cap_op = _operation_from_capability_view(m, op)
        op_map = ((m.get("operation_mappings") or {}).get(op) or {})
        steps = ((m.get("operation_steps_by_type") or {}).get(op) or [])
        if _operation_mapping_is_production_ready(op_map) and (
            (cap_op and cap_op.get("status") == "READY" and cap_op.get("taskable") and steps)
            or (not cap_op and op_map.get("status") == "READY" and steps)
            or op_map.get("selectors")
        ):
            ready_mappings.append(m)

    mapping_matches = [m for m in ready_mappings if _mapping_matches(m)]
    if not mapping_matches and len(ready_mappings) == 1 and not media_hint:
        mapping_matches = ready_mappings

    if len(mapping_matches) == 1:
        m = mapping_matches[0]
        payload["media_mapping_id"] = m.get("mapping_id")
        payload["media_name"] = m.get("media_name", "")
        created = create_task(
            TaskCreateRequest(
                agent_type="hp_update",
                operation_type=op,
                industry=_normalize_industry(m.get("industry") or "generic"),
                entity_type="",
                op_id="" if str(plan.get("op_id") or "").startswith("default_") else (plan.get("op_id") or ""),
                media_mapping_id=m.get("mapping_id"),
                payload=payload,
                scheduled_at=req.scheduled_at,
            ),
            user,
        )
        return {
            "ok": True,
            "created": True,
            "source": "media_mapping",
            "media_name": m.get("media_name", ""),
            "mapping_id": m.get("mapping_id"),
            "operation_type": op,
            "plan": plan,
            **created,
        }

    if len(ready_mappings) > 1:
        return {
            "ok": True,
            "created": False,
            "status": "NEEDS_MEDIA",
            "operation_type": op,
            "question": "対象媒体を一意に特定できません。媒体名を含めて指示してください。",
            "candidates": [
                {"mapping_id": m.get("mapping_id"), "media_name": m.get("media_name", ""), "media_url": m.get("media_url", "")}
                for m in ready_mappings[:20]
            ],
            "plan": plan,
        }

    return {
        "ok": True,
        "created": False,
        "status": "NO_READY_TARGET",
        "operation_type": op,
        "question": f"{op} を実行できるREADY媒体またはREADYリンクがありません。HTMLメニュー解析と深掘り解析を完了してください。",
        "plan": plan,
    }


# ==============================================================
# 外部LLM HTML解析 新設計 (AI_HTML_ANALYZE)
# 旧設計: ASCEND内部Playwright+deep_scan → 廃止
# 新設計: ユーザーがHTMLを貼り付け → 外部LLM解析 → Firestore保存
# 共有キャッシュ: media_html_cache/{url_hash} (テナント横断)
# ==============================================================

class AiHtmlAnalyzeRequest(BaseModel):
    raw_html:       str
    page_url:       str = ""
    page_type_hint: str = "auto"   # auto/login/news_post/text_update/...
    mapping_id:     Optional[str] = None   # 指定時は解析後に即適用
    force_reanalyze: bool = False


class AiSetupBatchPage(BaseModel):
    raw_html:       str
    page_url:       str = ""
    page_type_hint: str = "auto"


class AiSetupBatchRequest(BaseModel):
    pages: list[AiSetupBatchPage]


@router.post("/media/ai_analyze_html")
def ai_analyze_html(
    req: AiHtmlAnalyzeRequest,
    user: dict = Depends(verify_token),
):
    """
    HTMLを外部LLMで解析し、セレクタ・ステップ・capability を返す。
    - mapping_id を指定した場合は解析後に即座に media_mappings へ適用
    - 解析結果は media_html_cache に保存（テナント横断共有）
    - 同一URLを別テナントが登録した場合はキャッシュから即適用
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.raw_html.strip():
        raise HTTPException(status_code=400, detail="raw_htmlが空です")

    from api.core.html_analyzer import (
        analyze_html_page, get_cached_analysis,
        save_analysis_to_cache, apply_analysis_to_mapping, url_to_hash,
    )
    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    # キャッシュ確認
    cached = None
    url_hash = ""
    if req.page_url and not req.force_reanalyze:
        cached = get_cached_analysis(db, req.page_url)
        url_hash = url_to_hash(req.page_url)

    if cached:
        analysis = cached
        from_cache = True
    else:
        analysis = analyze_html_page(
            raw_html=req.raw_html,
            page_url=req.page_url,
            page_type_hint=req.page_type_hint,
        )
        from_cache = False
        if req.page_url and analysis.get("confidence", 0) >= 0.5:
            url_hash = save_analysis_to_cache(db, req.page_url, analysis)

    # mapping_id が指定されていれば適用
    apply_result = None
    if req.mapping_id:
        doc = db.collection("media_mappings").document(req.mapping_id).get()
        if doc.exists and (doc.to_dict() or {}).get("tenant_id") == tenant_id:
            apply_result = apply_analysis_to_mapping(db, req.mapping_id, analysis)
        else:
            apply_result = {"ok": False, "error": "mapping_idが見つからないか権限がありません"}

    return {
        "ok":          True,
        "from_cache":  from_cache,
        "url_hash":    url_hash,
        "page_type":   analysis.get("page_type"),
        "confidence":  analysis.get("confidence"),
        "site_purpose": analysis.get("site_purpose"),
        "login_selectors":     analysis.get("login_selectors") or {},
        "operation_selectors": analysis.get("operation_selectors") or {},
        "operation_steps":     analysis.get("operation_steps") or [],
        "capabilities":        analysis.get("capabilities") or {},
        "analysis_notes":      analysis.get("analysis_notes") or "",
        "apply_result":        apply_result,
    }


@router.post("/media/map/{mapping_id}/ai_setup")
def ai_setup_mapping(
    mapping_id: str,
    req: AiHtmlAnalyzeRequest,
    user: dict = Depends(verify_token),
):
    """
    特定 mapping_id に対してHTMLを解析し、結果を即適用する。
    - ログインページ → dom_selectors 更新
    - 操作ページ    → operation_mappings[op] を READY 状態で保存
    - 共有キャッシュに保存（他テナントも利用可能になる）
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.raw_html.strip():
        raise HTTPException(status_code=400, detail="raw_htmlが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and not ctx.get("is_admin"):
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    from api.core.html_analyzer import (
        analyze_html_page, get_cached_analysis,
        save_analysis_to_cache, apply_analysis_to_mapping, url_to_hash,
    )

    # キャッシュ確認
    effective_url = req.page_url or mapping.get("login_url") or mapping.get("media_url") or ""
    cached = None
    if effective_url and not req.force_reanalyze:
        cached = get_cached_analysis(db, effective_url)

    from_cache = False
    if cached:
        analysis = cached
        from_cache = True
    else:
        analysis = analyze_html_page(
            raw_html=req.raw_html,
            page_url=effective_url,
            page_type_hint=req.page_type_hint,
        )
        if effective_url and analysis.get("confidence", 0) >= 0.5:
            save_analysis_to_cache(db, effective_url, analysis)

    apply_result = apply_analysis_to_mapping(db, mapping_id, analysis)

    return {
        "ok":           apply_result.get("ok", False),
        "mapping_id":   mapping_id,
        "from_cache":   from_cache,
        "page_type":    analysis.get("page_type"),
        "confidence":   analysis.get("confidence"),
        "updated_fields": apply_result.get("updated_fields") or [],
        "analysis":     {
            "login_selectors":     analysis.get("login_selectors") or {},
            "operation_selectors": analysis.get("operation_selectors") or {},
            "operation_steps":     analysis.get("operation_steps") or [],
            "capabilities":        analysis.get("capabilities") or {},
            "site_purpose":        analysis.get("site_purpose"),
            "analysis_notes":      analysis.get("analysis_notes") or "",
        },
        "error": apply_result.get("error"),
    }


@router.post("/media/map/{mapping_id}/ai_setup_batch")
def ai_setup_batch(
    mapping_id: str,
    req: AiSetupBatchRequest,
    user: dict = Depends(verify_token),
):
    """
    複数ページを一括解析して mapping に適用する。
    例: ログインページ + ニュース投稿ページ + テキスト更新ページを同時指定。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not req.pages:
        raise HTTPException(status_code=400, detail="pagesが空です")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and not ctx.get("is_admin"):
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    from api.core.html_analyzer import (
        analyze_html_page, get_cached_analysis,
        save_analysis_to_cache, apply_analysis_to_mapping,
    )

    results = []
    for page in req.pages:
        if not page.raw_html.strip():
            results.append({"page_url": page.page_url, "ok": False, "error": "HTMLが空です"})
            continue

        cached = get_cached_analysis(db, page.page_url) if page.page_url else None
        from_cache = False
        if cached:
            analysis = cached
            from_cache = True
        else:
            analysis = analyze_html_page(
                raw_html=page.raw_html,
                page_url=page.page_url,
                page_type_hint=page.page_type_hint,
            )
            if page.page_url and analysis.get("confidence", 0) >= 0.5:
                save_analysis_to_cache(db, page.page_url, analysis)

        apply_result = apply_analysis_to_mapping(db, mapping_id, analysis)
        results.append({
            "page_url":       page.page_url,
            "page_type":      analysis.get("page_type"),
            "confidence":     analysis.get("confidence"),
            "from_cache":     from_cache,
            "ok":             apply_result.get("ok", False),
            "updated_fields": apply_result.get("updated_fields") or [],
            "error":          apply_result.get("error"),
        })

    ready_ops = []
    try:
        latest = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        op_map = latest.get("operation_mappings") or {}
        ready_ops = [op for op, st in op_map.items() if isinstance(st, dict) and st.get("status") == "READY"]
    except Exception:
        pass

    return {
        "ok":        True,
        "mapping_id": mapping_id,
        "results":   results,
        "ready_ops": ready_ops,
        "summary": {
            "total":   len(results),
            "success": sum(1 for r in results if r.get("ok")),
            "cached":  sum(1 for r in results if r.get("from_cache")),
        },
    }


@router.get("/media/html_cache/check")
def check_html_cache(
    url: str,
    user: dict = Depends(verify_token),
):
    """
    URLのキャッシュ解析が存在するか確認する。
    存在する場合はanalysisのサマリを返す（実データは apply で取得）。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")
    if not url:
        raise HTTPException(status_code=400, detail="urlが空です")

    from api.core.html_analyzer import get_cached_analysis, url_to_hash
    db = get_db()

    url_hash = url_to_hash(url)
    cached = get_cached_analysis(db, url)
    if not cached:
        return {"exists": False, "url_hash": url_hash, "url": url}

    return {
        "exists":       True,
        "url_hash":     url_hash,
        "url":          url,
        "page_type":    cached.get("page_type"),
        "confidence":   cached.get("confidence"),
        "site_purpose": cached.get("site_purpose"),
        "capabilities": cached.get("capabilities") or {},
        "ready_ops":    [
            op for op, val in (cached.get("capabilities") or {}).items()
            if val and op.startswith("can_")
        ],
    }


@router.post("/media/map/{mapping_id}/ai_clone_from_cache")
def ai_clone_from_cache(
    mapping_id: str,
    user: dict = Depends(verify_token),
):
    """
    同じURLの他テナント解析済みキャッシュを、このmappingに適用する。
    ユーザーがHTMLを貼らなくてもよい状態にするための高速セットアップ。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and not ctx.get("is_admin"):
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    # 共有キャッシュ確認
    media_url = mapping.get("media_url", "") or mapping.get("login_url", "")
    cache_applied = False
    if media_url:
        try:
            _cache_q = db.collection("media_html_cache").where("url", "==", media_url).limit(1).stream()
            for _cd in _cache_q:
                _cdata = _cd.to_dict() or {}
                _op_maps = _cdata.get("operation_mappings") or {}
                import datetime as _dt_cl2
                _upd: dict = {"updated_at": _dt_cl2.datetime.utcnow()}
                for _op, _od in _op_maps.items():
                    if not isinstance(_od, dict):
                        continue
                    _upd[f"operation_mappings.{_op}"] = {
                        **_od,
                        "production_ready":    True,
                        "confirmation_status": "AI_CONFIRMED",
                        "source":              "AI_CONFIRMED",
                        "cloned_from_cache":   True,
                    }
                if _upd:
                    db.collection("media_mappings").document(mapping_id).update(_upd)
                    cache_applied = True
                break
        except Exception as _e_cl2:
            print(f"[AI_CLONE_ERROR] {type(_e_cl2).__name__}", flush=True)

    if not cache_applied:
        return {
            "ok":    False,
            "reason": "このURLの共有キャッシュはまだ存在しません。/auto_setup で解析してください。",
        }

    latest = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
    op_map = latest.get("operation_mappings") or {}
    ready_ops = [op for op, st in op_map.items() if isinstance(st, dict) and st.get("status") == "READY"]
    return {
        "ok":        True,
        "mapping_id": mapping_id,
        "ready_ops": ready_ops,
        "message":   "共有キャッシュからマッピングを適用しました。",
    }


@router.post("/media/map/{mapping_id}/auto_setup")
def auto_setup_mapping(
    mapping_id: str,
    background_tasks: BackgroundTasks,
    body: dict = Body(default={}),
    user: dict = Depends(verify_token),
):
    """
    新設計: Playwright クロール → Gemini 解析 → AI_CONFIRMED で保存。
    - ユーザー操作不要。登録済みの認証情報でバックグラウンド実行。
    - 同じURLを別テナントが登録した場合はキャッシュから即適用。
    - 5媒体の既存マッピングの再解析にも使用。
    run_in_bg=true (デフォルト) でバックグラウンド実行。
    run_in_bg=false で同期実行（デバッグ用）。
    """
    if not _check_agent_access(user):
        raise HTTPException(status_code=403, detail="権限がありません")

    db = get_db()
    ctx = _resolve_agent_user_context(user)
    tenant_id = ctx["tenant_id"]

    doc = db.collection("media_mappings").document(mapping_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="media_mappingが見つかりません")
    mapping = doc.to_dict() or {}
    if mapping.get("tenant_id") != tenant_id and not ctx.get("is_admin"):
        raise HTTPException(status_code=403, detail="このmappingへのアクセス権がありません")

    run_in_bg = body.get("run_in_bg", True)

    # 認証情報チェック（credential_secret_nameは作成時に常に生成されるため credential_registered で確認）
    if not mapping.get("credential_registered"):
        raise HTTPException(status_code=400, detail="認証情報が未登録です。先にID/PASSを登録してください。")

    def _do_setup(_mid: str, _mapping_doc: dict):
        try:
            from api.core.browser_executor import auto_setup_mapping_ai as _asa
            _mapping_doc["mapping_id"] = _mid
            result = _asa(_mapping_doc, db=db)
            print(
                f"[AUTO_SETUP_DONE] mapping_id={_mid} "
                f"ok={result.get('ok')} ready_ops={result.get('ready_ops')}",
                flush=True,
            )
        except Exception as _e_setup:
            print(f"[AUTO_SETUP_ERROR] mapping_id={_mid} {type(_e_setup).__name__}:{_e_setup}", flush=True)

    mapping["mapping_id"] = mapping_id

    if run_in_bg:
        background_tasks.add_task(_do_setup, mapping_id, mapping)
        return {
            "ok":      True,
            "status":  "RUNNING",
            "message": "バックグラウンドで解析を開始しました。数十秒後に media_mappings を確認してください。",
            "mapping_id": mapping_id,
        }
    else:
        # 同期実行（デバッグ・再解析用）
        from api.core.browser_executor import auto_setup_mapping_ai as _asa_sync
        result = _asa_sync(mapping, db=db)
        return {
            "ok":         result.get("ok", False),
            "status":     result.get("status", ""),
            "ready_ops":  result.get("ready_ops", []),
            "failed_ops": result.get("failed_ops", []),
            "cache_saved": result.get("cache_saved", False),
            "pages_scanned": result.get("pages_scanned", 0),
            "mapping_id": mapping_id,
        }
