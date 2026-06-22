# api/core/html_analyzer.py
"""
外部LLM (Gemini) によるHTML解析モジュール
- ユーザーがHTMLを貼り付け → LLM解析 → セレクタ+ステップ抽出
- 解析結果を Firestore media_html_cache/{url_hash} に保存（テナント横断共有）
- 同一URLの媒体を別テナントが登録した場合も即時適用可能
"""

import hashlib
import json
import re
import os
from typing import Optional

# ── 操作タイプ定義 ──────────────────────────────────────────────────────
OPERATION_TYPES = [
    "news_post", "text_update", "media_replace", "blog_post",
    "schedule_update", "price_update", "entity_register", "entity_update",
]
SCOUT_OPERATION_TYPES = ["offer_send", "recruit_inbox_scan", "recruit_reply"]

# ── HTMLプリプロセス ─────────────────────────────────────────────────────
_STRIP_TAGS = re.compile(
    r"<(script|style|svg|noscript|link|meta|head)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_BLANK_LINES = re.compile(r"\n{3,}")
_MAX_HTML_CHARS = 80_000  # Gemini token節約のため上限

def preprocess_html(raw_html: str) -> str:
    cleaned = _STRIP_TAGS.sub("", raw_html or "")
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    if len(cleaned) > _MAX_HTML_CHARS:
        cleaned = cleaned[:_MAX_HTML_CHARS] + "\n[HTML省略]"
    return cleaned.strip()


# ── URL正規化・ハッシュ ─────────────────────────────────────────────────
def normalize_url_for_cache(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    url = re.sub(r"[?#].*$", "", url)
    return url.lower()

def url_to_hash(url: str) -> str:
    norm = normalize_url_for_cache(url)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


# ── LLM解析プロンプト生成 ─────────────────────────────────────────────────
def _build_analysis_prompt(cleaned_html: str, page_url: str, page_type_hint: str) -> str:
    op_hints = ""
    if page_type_hint and page_type_hint not in ("auto", "login"):
        op_hints = f"このページは主に「{page_type_hint}」操作に使われる画面です。"

    return f"""あなたはWebシステムのHTML解析専門家です。
以下のHTMLを解析し、管理システムの自動操作に必要な情報をJSONで返してください。

解析対象URL: {page_url or "（URLなし）"}
ヒント: {op_hints or "自動判定してください。"}

## HTMLの内容:
```html
{cleaned_html}
```

## 返却するJSONの形式（他のテキスト一切不要、JSONのみ返却）:
{{
  "page_type": "login" | "news_post" | "blog_post" | "text_update" | "media_replace" | "schedule_update" | "price_update" | "entity_register" | "entity_update" | "offer_send" | "unknown",
  "login_selectors": {{
    "username": "CSSセレクタ（loginページのみ）",
    "password": "CSSセレクタ（loginページのみ）",
    "submit": "CSSセレクタ（loginページのみ）",
    "verify_selector": "ログイン成功後に存在するCSSセレクタ"
  }},
  "operation_selectors": {{
    "title":  "タイトル入力のCSSセレクタ（なければnull）",
    "body":   "本文テキストエリアのCSSセレクタ（なければnull）",
    "image":  "画像アップロードinputのCSSセレクタ（なければnull）",
    "save":   "保存・送信ボタンのCSSセレクタ（必須）",
    "date":   "日付inputのCSSセレクタ（なければnull）",
    "price":  "料金inputのCSSセレクタ（なければnull）",
    "name":   "名前・タイトルinputのCSSセレクタ（なければnull）",
    "status": "ステータス切替のCSSセレクタ（なければnull）"
  }},
  "operation_steps": [
    {{"step_id": "1", "step_type": "fill", "selector_key": "title", "value": "{{{{payload.title}}}}", "terminal": false}},
    {{"step_id": "2", "step_type": "fill", "selector_key": "body",  "value": "{{{{payload.body}}}}",  "terminal": false}},
    {{"step_id": "3", "step_type": "click","selector_key": "save",  "value": null,                  "terminal": true}}
  ],
  "capabilities": {{
    "can_login":           true,
    "can_post_news":       false,
    "can_update_text":     false,
    "can_upload_image":    false,
    "can_update_schedule": false,
    "can_update_price":    false,
    "can_register_entity": false,
    "can_update_entity":   false,
    "can_verify":          false
  }},
  "site_purpose": "post" | "scout" | "reply" | "monitor" | "other",
  "confidence": 0.0～1.0,
  "analysis_notes": "解析メモ（セレクタの根拠など）"
}}

## 注意事項:
- operation_stepsのstep_typeは fill/click/upload/select のいずれか
- 最後のステップは必ず terminal: true
- CSSセレクタは可能な限り具体的に（id>name属性>class の優先順）
- ログインページならlogin_selectors、操作ページならoperation_selectorsを埋める
- confidence < 0.5 の場合は page_type を "unknown" にする
"""


# ── Gemini API呼び出し ──────────────────────────────────────────────────
def _call_gemini(prompt: str) -> str:
    from api.core.llm_client import call_llm
    return call_llm(
        system_prompt="あなたはWebHTML解析専門家です。JSONのみ返してください。",
        messages=[{"role": "user", "content": prompt}],
        ai_tier="ultra",
        max_tokens=4096,
    )


# ── JSON応答パース ──────────────────────────────────────────────────────
def _parse_llm_response(raw: str) -> dict:
    text = raw.strip()
    # ```json ... ``` or ``` ... ``` を除去
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.rstrip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSONブロックを探す
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ── 解析結果バリデーション・正規化 ──────────────────────────────────────
def _normalize_analysis(raw: dict, page_type_hint: str) -> dict:
    page_type = str(raw.get("page_type") or page_type_hint or "unknown").lower()
    if page_type not in (
        ["login", "unknown"] + OPERATION_TYPES + SCOUT_OPERATION_TYPES
    ):
        page_type = "unknown"

    login_sel = raw.get("login_selectors") or {}
    op_sel    = raw.get("operation_selectors") or {}
    steps     = raw.get("operation_steps") or []
    caps      = raw.get("capabilities") or {}
    confidence = float(raw.get("confidence") or 0.0)

    # セレクタのNoneを除去
    login_sel = {k: v for k, v in login_sel.items() if v}
    op_sel    = {k: v for k, v in op_sel.items() if v}

    # capabilities のデフォルト
    default_caps = {
        "can_login":           bool(login_sel.get("username") and login_sel.get("password")),
        "can_post_news":       page_type in ("news_post",),
        "can_update_text":     page_type in ("text_update",),
        "can_upload_image":    bool(op_sel.get("image")) or page_type == "media_replace",
        "can_update_schedule": page_type == "schedule_update",
        "can_update_price":    page_type == "price_update",
        "can_register_entity": page_type == "entity_register",
        "can_update_entity":   page_type == "entity_update",
        "can_verify":          bool(login_sel.get("verify_selector")),
    }
    merged_caps = {**default_caps, **{k: bool(v) for k, v in caps.items()}}

    site_purpose = str(raw.get("site_purpose") or "other").lower()
    if site_purpose not in ("post", "scout", "reply", "monitor", "other"):
        site_purpose = "other"

    return {
        "page_type":           page_type,
        "login_selectors":     login_sel,
        "operation_selectors": op_sel,
        "operation_steps":     steps,
        "capabilities":        merged_caps,
        "site_purpose":        site_purpose,
        "confidence":          confidence,
        "analysis_notes":      str(raw.get("analysis_notes") or ""),
    }


# ── メイン解析関数 ───────────────────────────────────────────────────────
def analyze_html_page(
    raw_html: str,
    page_url: str = "",
    page_type_hint: str = "auto",
    max_retries: int = 2,
) -> dict:
    """
    HTMLを解析してセレクタ・ステップ・capabilityを返す。
    エラー時は confidence=0 の空結果を返す（例外は投げない）。
    """
    cleaned = preprocess_html(raw_html)
    prompt  = _build_analysis_prompt(cleaned, page_url, page_type_hint)
    last_err = ""
    for attempt in range(max_retries):
        try:
            raw_text = _call_gemini(prompt)
            parsed   = _parse_llm_response(raw_text)
            if not parsed:
                last_err = "LLM応答がJSONとして解析できませんでした"
                continue
            result = _normalize_analysis(parsed, page_type_hint)
            result["raw_response"] = raw_text[:500]
            print(
                f"[HTML_ANALYZER] page_type={result['page_type']} "
                f"confidence={result['confidence']:.2f} "
                f"url={page_url[:60]}",
                flush=True,
            )
            return result
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[HTML_ANALYZER_ERROR] attempt={attempt+1} {last_err}", flush=True)

    return {
        "page_type": "unknown",
        "login_selectors": {},
        "operation_selectors": {},
        "operation_steps": [],
        "capabilities": {},
        "site_purpose": "other",
        "confidence": 0.0,
        "analysis_notes": f"解析失敗: {last_err}",
        "error": last_err,
    }


# ── 共有キャッシュ CRUD ──────────────────────────────────────────────────
CACHE_COLLECTION = "media_html_cache"

def get_cached_analysis(db, page_url: str) -> Optional[dict]:
    """URLハッシュで共有キャッシュを検索。なければNone。"""
    if not page_url:
        return None
    url_hash = url_to_hash(page_url)
    try:
        doc = db.collection(CACHE_COLLECTION).document(url_hash).get()
        if doc.exists:
            data = doc.to_dict() or {}
            print(f"[HTML_CACHE_HIT] url_hash={url_hash[:8]} url={page_url[:60]}", flush=True)
            return data.get("analysis")
    except Exception as e:
        print(f"[HTML_CACHE_READ_ERROR] {type(e).__name__}: {e}", flush=True)
    return None


def save_analysis_to_cache(db, page_url: str, analysis: dict) -> str:
    """解析結果を共有キャッシュに保存。url_hashを返す。"""
    if not page_url:
        return ""
    import datetime
    url_hash = url_to_hash(page_url)
    try:
        db.collection(CACHE_COLLECTION).document(url_hash).set({
            "url_hash":    url_hash,
            "url":         page_url,
            "analysis":    analysis,
            "analyzed_at": datetime.datetime.utcnow(),
            "model":       "gemini",
            "usage_count": 0,
        }, merge=True)
        # usage_countはインクリメント
        db.collection(CACHE_COLLECTION).document(url_hash).update({
            "usage_count": db.collection(CACHE_COLLECTION).document(url_hash).get().to_dict().get("usage_count", 0) + 1
        })
        print(f"[HTML_CACHE_SAVE] url_hash={url_hash[:8]} page_type={analysis.get('page_type')}", flush=True)
    except Exception as e:
        print(f"[HTML_CACHE_SAVE_ERROR] {type(e).__name__}: {e}", flush=True)
    return url_hash


# ── mapping への適用 ─────────────────────────────────────────────────────
def apply_analysis_to_mapping(db, mapping_id: str, analysis: dict) -> dict:
    """
    解析結果を media_mappings/{mapping_id} に書き込む。
    - ログインページ → dom_selectors, login_selectors 更新
    - 操作ページ → operation_mappings[op_type] を READY 状態で保存
    - capabilities 上書きマージ
    - business_conditions.site_purpose 更新
    返り値: {ok, updated_fields}
    """
    import datetime
    page_type = analysis.get("page_type", "unknown")
    now = datetime.datetime.utcnow().isoformat()
    updates: dict = {"updated_at": datetime.datetime.utcnow()}
    updated_fields = []

    # ── ログインページ ──
    if page_type == "login":
        ls = analysis.get("login_selectors") or {}
        if ls:
            dom_upd = {}
            if ls.get("username"):
                dom_upd["username"] = ls["username"]
            if ls.get("password"):
                dom_upd["password"] = ls["password"]
            if ls.get("submit"):
                dom_upd["login_submit"] = ls["submit"]
            if ls.get("verify_selector"):
                dom_upd["verify_selector"] = ls["verify_selector"]
                updates["verify_selector"] = ls["verify_selector"]
            # dom_selectors はネストされているため個別にセット
            for k, v in dom_upd.items():
                updates[f"dom_selectors.{k}"] = v
            updates["login_selectors"] = ls
            updates["login_health"] = "UNKNOWN"
            updated_fields.append("login_selectors")

    # ── 操作ページ ──
    elif page_type in OPERATION_TYPES:
        op_sel   = analysis.get("operation_selectors") or {}
        steps    = analysis.get("operation_steps") or []
        conf     = float(analysis.get("confidence") or 0.0)

        selectors_for_op: dict = {}
        for role, sel_val in op_sel.items():
            if sel_val:
                selectors_for_op[role] = sel_val

        # operation_mappings[op_type] 構造
        op_mapping_doc = {
            "status":               "READY",
            "executable":           True,
            "production_ready":     True,
            "confirmation_status":  "AI_CONFIRMED",
            "source":               "AI_HTML_ANALYZE",
            "selectors":            selectors_for_op,
            "fields":               _selectors_to_fields(op_sel, page_type),
            "step_count":           len(steps),
            "last_scanned_at":      now,
            "ai_confidence":        conf,
        }
        # operation_steps_by_type にも保存
        if steps:
            updates[f"operation_steps_by_type.{page_type}"] = steps

        updates[f"operation_mappings.{page_type}"] = op_mapping_doc
        updated_fields.append(f"operation_mappings.{page_type}")

    # ── capabilities マージ ──
    caps = analysis.get("capabilities") or {}
    if caps:
        try:
            existing = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            merged_caps = {**({} if not existing.get("capabilities") else existing["capabilities"]), **caps}
            updates["capabilities"] = merged_caps
            updated_fields.append("capabilities")
        except Exception:
            updates["capabilities"] = caps
            updated_fields.append("capabilities")

    # ── site_purpose ──
    sp = analysis.get("site_purpose") or ""
    if sp and sp != "other":
        updates["business_conditions.site_purpose"] = sp
        updated_fields.append("site_purpose")

    try:
        db.collection("media_mappings").document(mapping_id).update(updates)
        print(
            f"[APPLY_ANALYSIS] mapping_id={mapping_id} "
            f"page_type={page_type} fields={updated_fields}",
            flush=True,
        )
        return {"ok": True, "updated_fields": updated_fields, "page_type": page_type}
    except Exception as e:
        print(f"[APPLY_ANALYSIS_ERROR] mapping_id={mapping_id} {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "error": str(e), "updated_fields": []}


def _selectors_to_fields(op_sel: dict, operation_type: str) -> list:
    """セレクタ辞書をfieldsリスト形式に変換（browser_executor互換）"""
    _SELECTOR_ROLE_TO_PAYLOAD = {
        "title":  ("title",       "text"),
        "body":   ("body",        "text"),
        "image":  ("file_path",   "file"),
        "date":   ("date_value",  "text"),
        "price":  ("price_value", "text"),
        "name":   ("name",        "text"),
        "status": ("status",      "text"),
    }
    fields = []
    for role, sel_val in op_sel.items():
        if not sel_val or role == "save":
            continue
        payload_key, input_type = _SELECTOR_ROLE_TO_PAYLOAD.get(role, (role, "text"))
        fields.append({
            "selector_key": role,
            "selector":     sel_val,
            "payload_key":  payload_key,
            "input_type":   input_type,
        })
    return fields


# ── 媒体登録時の自動キャッシュ適用 ───────────────────────────────────────
def try_clone_from_cache(db, mapping_id: str, media_url: str, login_url: str) -> bool:
    """
    媒体登録直後に呼び出す。
    キャッシュが存在する場合は即座に解析を適用。
    戻り値: True=適用成功, False=キャッシュなし
    """
    for url in [login_url, media_url]:
        cached = get_cached_analysis(db, url)
        if cached:
            result = apply_analysis_to_mapping(db, mapping_id, cached)
            if result.get("ok"):
                print(
                    f"[CACHE_CLONE] mapping_id={mapping_id} from_url={url[:60]}",
                    flush=True,
                )
                return True
    return False
