# api/core/llm_client.py
import os
import base64 as _b64
from google import genai
from google.genai import types

# モデル優先リスト（旧pick_model_candidatesと同仕様）
_CORE_PREFERRED = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
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
    "gemini-2.5-pro-latest",
    "gemini-2.5-flash",
    "gemini-2.0-pro",
]
_APEX_PREFERRED = [
    "gemini-3.0-ultra",
    "gemini-3.0-pro",
    "gemini-3.0-flash",
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

_model_cache: dict = {"models": set(), "expires": 0}

def _list_available_models(client) -> set:
    import time
    global _model_cache
    if _model_cache["models"] and time.time() < _model_cache["expires"]:
        return _model_cache["models"]
    try:
        models = {m.name.split("/")[-1] for m in client.models.list()}
        _model_cache = {"models": models, "expires": time.time() + 600}
        return models
    except Exception:
        return _model_cache["models"] if _model_cache["models"] else set()

def pick_model(ai_tier: str = "core") -> str:
    """利用可能なモデルからtiereに合ったものを選択。フォールバックあり"""
    preferred = MODEL_TIERS.get(ai_tier, _CORE_PREFERRED)
    try:
        client = _get_client()
        available = _list_available_models(client)
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

    # モデル候補を順番に試す（フォールバック付き）
    preferred = MODEL_TIERS.get(ai_tier, _CORE_PREFERRED)
    try:
        available = _list_available_models(client)
        candidates = [m for m in preferred if m in available] if available else preferred[:2]
        if not candidates:
            candidates = [model_name]
    except Exception:
        candidates = [model_name]

    last_err = None
    for candidate in candidates[:3]:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=sdk_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            _text = response.text
            if not _text or not _text.strip():
                last_err = Exception(f"{candidate}: 空レスポンス（モデルが無応答）")
                continue
            return _text
        except Exception as e:
            last_err = e
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
