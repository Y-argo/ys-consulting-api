# api/core/llm_client.py
import os
import base64 as _b64
from google import genai
from google.genai import types
import time as _time

# ── models.list() キャッシュ（300秒TTL） ──
_model_cache: dict = {"models": set(), "ts": 0}
_CACHE_TTL = 300

# ── LLMレスポンスキャッシュ（同一クエリのRPM節約） ──
import hashlib as _hashlib
_llm_cache: dict = {}
_LLM_CACHE_TTL = 300  # 5分

def _make_cache_key(system_prompt: str, messages: list, ai_tier: str, max_tokens: int) -> str:
    last_msg = messages[-1].get("content", "") if messages else ""
    raw = f"{ai_tier}:{max_tokens}:{system_prompt[:200]}:{last_msg[:500]}"
    return _hashlib.md5(raw.encode()).hexdigest()

def _get_llm_cache(key: str):
    import time as _t
    entry = _llm_cache.get(key)
    if entry and (_t.time() - entry["ts"]) < _LLM_CACHE_TTL:
        print(f"[LLM_CACHE_HIT] key={key[:8]}", flush=True)
        return entry["val"]
    return None

def _set_llm_cache(key: str, val: str):
    import time as _t
    _llm_cache[key] = {"val": val, "ts": _t.time()}
    # 古いキャッシュを削除（最大100件）
    if len(_llm_cache) > 100:
        oldest = sorted(_llm_cache.items(), key=lambda x: x[1]["ts"])[:20]
        for k, _ in oldest:
            del _llm_cache[k]

def _list_available_models_cached(client) -> set:
    now = _time.time()
    if _model_cache["models"] and (now - _model_cache["ts"]) < _CACHE_TTL:
        return _model_cache["models"]
    try:
        result = {m.name.split("/")[-1] for m in client.models.list()}
        _model_cache["models"] = result
        _model_cache["ts"] = now
        return result
    except Exception:
        return _model_cache["models"] or set()

# モデル優先リスト（旧pick_model_candidatesと同仕様）
_CORE_PREFERRED = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]
# 画像解析専用モデル（visionサポート確認済み）
_VISION_PREFERRED = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
_ULTRA_PREFERRED = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]
_APEX_PREFERRED = [
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro",
]
_PRO_PREFERRED = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
]

MODEL_TIERS = {
    "core":  _CORE_PREFERRED,
    "ultra": _ULTRA_PREFERRED,
    "apex":  _APEX_PREFERRED,
    "pro":   _PRO_PREFERRED,
}

def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key) if api_key else genai.Client()

def pick_model(ai_tier: str = "core") -> str:
    """利用可能なモデルからtiereに合ったものを選択。フォールバックあり"""
    preferred = MODEL_TIERS.get(ai_tier, _CORE_PREFERRED)
    try:
        client = _get_client()
        available = _list_available_models_cached(client)
        if available:
            for m in preferred:
                if m in available:
                    return m
            # フォールバック: coreから探す
            for m in _CORE_PREFERRED:
                if m in available:
                    return m
    except Exception:
        pass
    return preferred[0] if preferred else "gemini-2.0-flash-001"

def _load_tenant_temperature(tenant_id: str) -> float:
    """テナント別temperatureをFirestoreから取得"""
    try:
        from api.core.firestore_client import get_db
        db = get_db()
        doc = db.collection("tenant_settings").document(tenant_id).get()
        if doc.exists:
            v = (doc.to_dict() or {}).get("temperature")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return 0.7

def call_llm(
    system_prompt: str,
    messages: list,
    ai_tier: str = "core",
    temperature: float = None,
    max_tokens: int = 8192,
    image_b64: str = None,
    image_mime: str = "image/png",
    tenant_id: str = None,
) -> str:
    # キャッシュチェック（secondary LLM呼び出しのRPM節約）
    _ck = _make_cache_key(system_prompt, messages, ai_tier, max_tokens)
    _cached = _get_llm_cache(_ck)
    if _cached:
        return _cached
    client = _get_client()
    model_name = pick_model(ai_tier)

    # テナント別temperature適用
    if temperature is None:
        if tenant_id:
            temperature = _load_tenant_temperature(tenant_id)
        else:
            temperature = 0.7

    sdk_messages = []
    for i, m in enumerate(messages):
        role = "user" if m["role"] == "user" else "model"
        if role == "user" and i == len(messages) - 1 and image_b64:
            try:
                img_bytes = _b64.b64decode(image_b64)
                sdk_messages.append(types.Content(role=role, parts=[
                    types.Part(text=m["content"]),
                    types.Part(inline_data=types.Blob(mime_type=image_mime, data=img_bytes)),
                ]))
            except Exception:
                sdk_messages.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        else:
            sdk_messages.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    # モデル候補を順番に試す（pick_modelで選定済みを先頭に）
    preferred = MODEL_TIERS.get(ai_tier, _CORE_PREFERRED)
    try:
        available = _list_available_models_cached(client)
        candidates = [m for m in preferred if m in available] if available else preferred[:2]
        if not candidates:
            candidates = [model_name]
    except Exception:
        candidates = [model_name]

    import time as _time, threading as _th_llm
    last_err = None
    _TIMEOUT = 50
    for candidate in candidates[:5]:
        try:
            _res_box = {}
            _err_box = {}
            def _do_call(c=candidate, r=_res_box, er=_err_box):
                try:
                    resp = client.models.generate_content(
                        model=c,
                        contents=sdk_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                            ],
                        ),
                    )
                    _t2 = ""
                    _finish_reason = ""
                    try:
                        _t2 = resp.text or ""
                    except Exception:
                        pass
                    if not _t2 and resp.candidates:
                        try:
                            _finish_reason = str(resp.candidates[0].finish_reason)
                            _t2 = resp.candidates[0].content.parts[0].text or ""
                        except Exception:
                            pass
                    if not _t2:
                        try:
                            _cands = resp.candidates
                            _pf = resp.prompt_feedback if hasattr(resp, 'prompt_feedback') else None
                            _pf_str = str(_pf) if _pf else ""
                            print(f"[LLM_EMPTY] model={c} finish_reason={_finish_reason} candidates={len(_cands) if _cands else 0} prompt_feedback={_pf_str}", flush=True)
                            if "PROHIBITED_CONTENT" in _pf_str or "SAFETY" in _pf_str:
                                last_err = Exception(f"{c}: コンテンツポリシーによりスキップ")
                                print(f"[LLM_POLICY_SKIP] {c} → 次候補へ", flush=True)
                        except Exception as _de:
                            print(f"[LLM_EMPTY] model={c} debug_err={_de}", flush=True)
                    r["v"] = _t2
                except Exception as _ce:
                    er["v"] = _ce
            _t = _th_llm.Thread(target=_do_call, daemon=True)
            _t.start()
            _t.join(timeout=_TIMEOUT)
            if _t.is_alive():
                last_err = Exception(f"{candidate}: {_TIMEOUT}秒タイムアウト")
                print(f"[LLM_TIMEOUT] {candidate}", flush=True)
                continue
            if "v" in _err_box:
                raise _err_box["v"]
            _text = _res_box.get("v", "")
            if not _text or not _text.strip():
                last_err = Exception(f"{candidate}: 空レスポンス")
                continue
            _set_llm_cache(_ck, _text)
            return _text
        except Exception as e:
            last_err = e
            _e_str = str(e)
            if '429' in _e_str or 'RESOURCE_EXHAUSTED' in _e_str:
                print(f"[LLM_429] {candidate} 429 - skip to next", flush=True)
            continue
    raise Exception(f"全モデル失敗: {last_err}")

def call_llm_pro(
    system_prompt: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tenant_id: str = None,
) -> str:
    """Pro tier専用呼び出し（脳内カルテ・QueryPlan等の高精度判定用）"""
    return call_llm(
        system_prompt=system_prompt,
        messages=messages,
        ai_tier="pro",
        temperature=temperature,
        max_tokens=max_tokens,
        tenant_id=tenant_id,
    )
