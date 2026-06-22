# api/core/browser_executor.py
# ブラウザ操作実行層
# Playwrightはトップレベルimportしない（Cloud Run未導入時の起動クラッシュ防止）
# ID/PASSはログ・例外メッセージに絶対出さない

import os
from typing import Optional
from api.core.secret_manager import get_secret_json
import time as _time


class OperationStepRuntimeError(RuntimeError):
    """operation_steps失敗時に完了済みstep情報を呼び元へ渡す例外。"""

    def __init__(self, message: str, step_results: list | None = None, failed_step_id: str = ""):
        super().__init__(message)
        self.step_results = step_results or []
        self.failed_step_id = failed_step_id


def _is_terminal_operation_step(step: dict) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("terminal") is True or step.get("once") is True:
        return True
    step_type = str(step.get("step_type") or "").lower()
    selector_key = str(step.get("selector_key") or "").lower()
    step_id = str(step.get("step_id") or "").lower()
    if step_type in ("click", "search") and selector_key in ("save", "submit"):
        return True
    return step_type in ("click", "search") and any(k in step_id for k in ("save", "submit", "post", "register"))


def _terminal_step_done(operation_steps: list | None, step_results: list | None) -> bool:
    if not operation_steps or not step_results:
        return False
    terminal_ids = {str(s.get("step_id") or "") for s in operation_steps if _is_terminal_operation_step(s)}
    return any(r.get("status") == "DONE" and str(r.get("step_id") or "") in terminal_ids for r in step_results or [])

_BUSINESS_FIELD_CONTROL_TYPES = {"hidden", "password", "submit", "button", "reset", "image"}
_BUSINESS_FIELD_CONTROL_KEYWORDS = (
    "login", "signin", "sign_in", "ログイン", "管理者id", "管理者", "admin",
    "password", "passwd", "pass", "パスワード",
    "hidden", "(hidden)", "csrf", "_token", "token", "nonce",
    "submit", "send", "送信", "保存する", "ログインid/パスワードを保存する",
    "remember", "button", "open_field", "select_girl_review",
    "sort", "ソート", "削除", "delete", "戻る", "back", "確認",
)

def _is_business_form_field(field: dict) -> bool:
    if not isinstance(field, dict):
        return False
    ftype = str(field.get("type") or "text").lower().strip()
    if ftype in _BUSINESS_FIELD_CONTROL_TYPES:
        return False
    blob = " ".join(str(field.get(k) or "") for k in ("label", "name", "id", "selector", "placeholder", "canonical")).lower()
    if not blob:
        return False
    if "input[type=password]" in blob or "type='password'" in blob or 'type="password"' in blob:
        return False
    return not any(k in blob for k in _BUSINESS_FIELD_CONTROL_KEYWORDS)


def _mark_irreversible_rollback(rollback: dict | None) -> dict:
    data = dict(rollback or {})
    data["external_side_effect_reversible"] = False
    data["success"] = False
    reason = data.get("reason") or ""
    suffix = "submit後の外部反映は自動rollback不可"
    data["reason"] = (reason + " / " + suffix).strip(" /") if reason else suffix
    return data


# ── P16.7: Selector Learning Cache (TTL 60秒) ────────────────────────────
# Firestore selector_learning_stats を毎回直読みしない
# { cache_key: {"data": dict, "expires_at": float} }
import threading as _threading
_SELECTOR_LEARNING_CACHE: dict = {}
_SELECTOR_LEARNING_CACHE_TTL = 60  # seconds
_SELECTOR_LEARNING_LOCK = _threading.Lock()

def _get_selector_learning_cache(key: str):
    with _SELECTOR_LEARNING_LOCK:
        entry = _SELECTOR_LEARNING_CACHE.get(key)
        if entry and _time.time() < entry["expires_at"]:
            return entry["data"]
    return None

def _set_selector_learning_cache(key: str, data: dict):
    with _SELECTOR_LEARNING_LOCK:
        _SELECTOR_LEARNING_CACHE[key] = {
            "data":       data,
            "expires_at": _time.time() + _SELECTOR_LEARNING_CACHE_TTL,
        }

def _invalidate_selector_learning_cache(key: str):
    with _SELECTOR_LEARNING_LOCK:
        _SELECTOR_LEARNING_CACHE.pop(key, None)
# ─────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# P25 Session Management
# ログイン済みCookieを mapping_id 単位でメモリにキャッシュ
# ID/PASS・Cookie中身は絶対にログ出力しない（件数のみ）
# ══════════════════════════════════════════════════════════════════════════════
import datetime as _dt_p25

_session_cache: dict = {}
# 構造: { "{tenant_id}:{mapping_id}": { "cookies": [], "tenant_id": str, "logged_in_at": datetime, "expires_at": datetime } }
_SESSION_TTL_MINUTES = 30
_SESSION_LOCK = _threading.Lock()

def _session_key(mapping_id: str, tenant_id: str = "") -> str:
    """テナント分離のためtenant_idをキーに含める。"""
    return f"{tenant_id}:{mapping_id}" if tenant_id else mapping_id


def _get_cached_session(mapping_id: str, tenant_id: str = "") -> dict | None:
    """メモリ優先、次にFirestoreからセッションを取得する。"""
    import datetime as _dt_check
    _skey = _session_key(mapping_id, tenant_id)
    # ① メモリキャッシュ確認（ロック保護）
    with _SESSION_LOCK:
        entry = _session_cache.get(_skey)
    if entry:
        # C-1: tenant_idが記録されている場合は一致を確認
        if tenant_id and entry.get("tenant_id") and entry["tenant_id"] != tenant_id:
            print(f"[P25_SESSION_TENANT_MISMATCH] mapping_id={mapping_id} expected={tenant_id} stored={entry.get('tenant_id')}", flush=True)
            return None
        now = _dt_check.datetime.utcnow()
        _exp = entry.get("expires_at")
        if hasattr(_exp, "tzinfo") and _exp.tzinfo is not None:
            now = _dt_check.datetime.now(_dt_check.timezone.utc)
        expired = (now >= _exp) if _exp else True
        has_cookies = bool(entry.get("cookies"))
        if has_cookies and not expired:
            print(f"[P25_SESSION_CHECK] mapping_id={mapping_id} source=memory hit=True expired=False", flush=True)
            return entry
    # ② Firestore確認
    try:
        from api.core.firestore_client import get_db as _get_db_p25
        _db_p25 = _get_db_p25()
        _snap = _db_p25.collection("agent_sessions").document(str(mapping_id)).get()
        print(f"[P25_SESSION_FIRESTORE_READ] mapping_id={mapping_id} exists={_snap.exists}", flush=True)
        if _snap.exists:
            _data = _snap.to_dict() or {}
            # P25: 論理削除済みセッションは無効扱い
            if _data.get("is_deleted", False):
                print(f"[P25_SESSION_DELETED] mapping_id={mapping_id} reason={_data.get('delete_reason','unknown')}", flush=True)
                return None
            # C-1: テナント不一致チェック
            if tenant_id and _data.get("tenant_id") and _data["tenant_id"] != tenant_id:
                print(f"[P25_SESSION_TENANT_MISMATCH_FS] mapping_id={mapping_id} expected={tenant_id} stored={_data.get('tenant_id')}", flush=True)
                return None
            _cookies = _data.get("cookies") or []
            _expires_at = _data.get("expires_at")
            print(f"[P25_SESSION_FIRESTORE_READ] mapping_id={mapping_id} exists=True cookies_count={len(_cookies)} expires_at={_expires_at}", flush=True)
            now_naive = _dt_check.datetime.utcnow()
            now_aware = _dt_check.datetime.now(_dt_check.timezone.utc)
            if _expires_at is None:
                expired = True
            else:
                try:
                    _exp_dt = _expires_at.ToDatetime() if hasattr(_expires_at, "ToDatetime") else _expires_at
                except Exception:
                    _exp_dt = _expires_at
                if hasattr(_exp_dt, "tzinfo") and _exp_dt.tzinfo is not None:
                    expired = now_aware >= _exp_dt
                else:
                    expired = now_naive >= _exp_dt
            if _cookies and not expired:
                with _SESSION_LOCK:
                    _session_cache[_skey] = _data
                print(f"[P25_SESSION_REUSE] mapping_id={mapping_id} source=firestore cookies_count={len(_cookies)}", flush=True)
                return _data
            else:
                print(f"[P25_SESSION_CHECK] mapping_id={mapping_id} source=firestore hit=False expired={expired}", flush=True)
                return None
        else:
            print(f"[P25_SESSION_CHECK] mapping_id={mapping_id} source=none hit=False expired=False", flush=True)
            return None
    except Exception as _e_get:
        print(f"[P25_SESSION_FIRESTORE_ERROR] mapping_id={mapping_id} error={_e_get}", flush=True)
        return None

def _is_authenticated_page(page) -> bool:
    """ログイン画面でないことを確認する。Falseならセッション保存禁止。"""
    try:
        url = page.url.lower()
        title = page.title().lower()
        login_url_signals = ["login", "signin", "sign_in", "sign-in"]
        login_title_signals = ["ログイン", "login", "signin"]
        _url_has_login_signal = any(s in url for s in login_url_signals)
        if _url_has_login_signal:
            # URLにloginシグナルがあってもパスワード欄がなければログイン後ページの可能性
            try:
                _auth_pw = page.locator("input[type=password]").count()
            except Exception:
                _auth_pw = 0
            if _auth_pw > 0:
                print(f"[P25_AUTH_CHECK] result=False reason=url_and_pw_field url={page.url}", flush=True)
                return False
            else:
                print(f"[P25_AUTH_CHECK] url_has_login_signal=True but no_pw_field url={page.url} continuing", flush=True)
        if any(s in title for s in login_title_signals):
            print(f"[P25_AUTH_CHECK] result=False reason=title_contains_login title={page.title()}", flush=True)
            return False
        try:
            pw_count = page.locator("input[type=password]").count()
            if pw_count > 0:
                print(f"[P25_AUTH_CHECK] result=False reason=password_field_exists count={pw_count}", flush=True)
                return False
        except Exception:
            pass
        try:
            login_btn = page.locator("input[name='login_req'], input[name*='login'], button[name*='login']").count()
            if login_btn > 0:
                print(f"[P25_AUTH_CHECK] result=False reason=login_button_exists count={login_btn}", flush=True)
                return False
        except Exception:
            pass
        print(f"[P25_AUTH_CHECK] result=True url={page.url}", flush=True)
        return True
    except Exception as _e:
        print(f"[P25_AUTH_CHECK] error={_e} result=False", flush=True)
        return False

def _save_cached_session(mapping_id: str, context, page, tenant_id: str = "") -> None:
    """ログイン成功後にstorage_state全体をメモリ+Firestoreに保存する。"""
    if page and not _is_authenticated_page(page):
        print(f"[P25_SESSION_SAVE_SKIP] mapping_id={mapping_id} reason=not_authenticated url={page.url}", flush=True)
        return
    try:
        cookies = context.cookies()
        try:
            _storage = context.storage_state()
        except Exception as _ss_err:
            _storage = {"cookies": cookies, "origins": []}
            print(f"[P25_STORAGE_STATE_ERROR] mapping_id={mapping_id} error={_ss_err}", flush=True)
        _origins = _storage.get("origins", [])
        # F-2: Cookie名のみログ（値は出力しない）、件数のみに変更
        print(f"[P25_STORAGE_STATE_SAVE] mapping_id={mapping_id} cookies={len(cookies)} origins={len(_origins)}", flush=True)
        import datetime as _dt_p25s
        now = _dt_p25s.datetime.utcnow()
        expires_at = now + _dt_p25s.timedelta(minutes=_SESSION_TTL_MINUTES)
        _skey = _session_key(mapping_id, tenant_id)
        entry = {
            "mapping_id":    mapping_id,
            "tenant_id":     tenant_id,
            "cookies":       cookies,
            "storage_state": _storage,
            "logged_in_at":  now,
            "expires_at":    expires_at,
            "current_url":   page.url if page else "",
            "title":         page.title() if page else "",
            "updated_at":    now,
        }
        with _SESSION_LOCK:
            _session_cache[_skey] = entry
        try:
            from api.core.firestore_client import get_db as _get_db_p25s
            _db_p25s = _get_db_p25s()
            _db_p25s.collection("agent_sessions").document(mapping_id).set(entry)
        except Exception as _e_fs:
            print(f"[P25_SESSION_SAVE_FS_ERROR] mapping_id={mapping_id} error={_e_fs}", flush=True)
        print(f"[P25_SESSION_SAVE] mapping_id={mapping_id} store=memory+firestore cookies_count={len(cookies)} expires_at={expires_at.isoformat()}", flush=True)
    except Exception as _e_save:
        print(f"[P25_SESSION_SAVE_ERROR] mapping_id={mapping_id} error={_e_save}", flush=True)

def _clear_cached_session(mapping_id: str, reason: str = "unknown", tenant_id: str = "") -> None:
    """メモリ+Firestoreからセッションを削除する。"""
    _skey = _session_key(mapping_id, tenant_id)
    with _SESSION_LOCK:
        _session_cache.pop(_skey, None)
        # テナントなしキーも念のため削除（旧形式互換）
        if _skey != mapping_id:
            _session_cache.pop(mapping_id, None)
    try:
        from api.core.firestore_client import get_db as _get_db_p25c
        _get_db_p25c().collection("agent_sessions").document(mapping_id).set({
            "is_deleted": True,
            "deleted_at": __import__("datetime").datetime.utcnow(),
            "delete_reason": reason,
            "cookies": [],
            "storage_state": {},
        }, merge=True)
    except Exception as _e_clr:
        print(f"[P25_SESSION_CLEAR_FS_ERROR] mapping_id={mapping_id} error={_e_clr}", flush=True)
    print(f"[P25_SESSION_CLEAR] mapping_id={mapping_id} reason={reason}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
def is_playwright_enabled() -> bool:
    return os.environ.get("PLAYWRIGHT_ENABLED", "false").lower() == "true"


def is_login_check_crawl_enabled() -> bool:
    return os.environ.get("LOGIN_CHECK_CRAWL_ENABLED", "false").lower() == "true"


def _login_check_crawl_max_pages(default_max: int = 200) -> int:
    raw = os.environ.get("LOGIN_CHECK_CRAWL_MAX_PAGES")
    try:
        value = int(raw) if raw is not None else int(default_max)
    except Exception:
        value = int(default_max)
    return max(1, min(value, 50))


def run_browser_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
    operation_steps: list = None,
    prior_step_results: list = None,
    task_id: str = "",
    db=None,
) -> dict:
    """
    browser_executorの入口。auth_typeに応じて処理を振り分ける。
    execute_task / agent_executorから呼ばれる。
    """
    auth_type = media_mapping.get("auth_type", "login_form")

    if auth_type == "login_form":
        return _run_login_form_operation(media_mapping, operation_type, payload, operation_steps=operation_steps,
                                         prior_step_results=prior_step_results, task_id=task_id, db=db)

    if auth_type == "api_key":
        return _run_api_key_operation(media_mapping, operation_type, payload)

    if auth_type == "manual":
        return {
            "status":   "BLOCKED",
            "executed": False,
            "message":  "auth_type=manualは手動操作が必要です。自動実行できません。",
            "blocked_reason": "auth_type_manual",
        }

    if auth_type == "none":
        return _run_no_auth_operation(media_mapping, operation_type, payload)

    return {
        "status":   "WAITING_EXECUTOR",
        "executed": False,
        "message":  f"auth_type '{auth_type}' は未対応です。",
    }


def _run_login_form_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
    operation_steps: list = None,
    prior_step_results: list = None,
    task_id: str = "",
    db=None,
) -> dict:
    """ログインフォーム型媒体への操作。Secret Manager取得→Playwright実行。"""

    # PLAYWRIGHT_ENABLED=false の場合はここで止まる
    if not is_playwright_enabled():
        return {
            "status":   "WAITING_EXECUTOR",
            "executed": False,
            "message":  "PLAYWRIGHT_ENABLED=false のためブラウザ実行は無効です。",
        }

    # Secret Managerから認証情報取得（ID/PASSはログに出さない）
    secret_name = media_mapping.get("credential_secret_name")
    creds = get_secret_json(secret_name)
    if not creds or creds.get("blocked"):
        return {
            "status":   "BLOCKED",
            "executed": False,
            "message":  creds.get("error", "認証情報の取得に失敗しました") if creds else "認証情報が取得できませんでした",
        }

    return _run_login_form_with_operation(media_mapping, creds, operation_type, payload, operation_steps=operation_steps,
                                          prior_step_results=prior_step_results, task_id=task_id, db=db)



def _build_selector_candidates(media_mapping: dict, role: str) -> list:
    """
    P0-1: username/password/login_submit の候補リストを生成する。
    dom_selectors → detected_summary → hardcoded fallback の順。
    """
    dom = media_mapping.get("dom_selectors") or {}
    ds  = (media_mapping.get("detected_summary") or {})
    if role == "username":
        return [c for c in [
            dom.get("username"),
            dom.get("login_id"),
            ds.get("login_id"),
            'input[name="txt_account"]',
            "#id",
            'input[name="id"]',
            'input[name="login_id"]',
            'input[name="loginId"]',
            'input[name="email"]',
            'input[name="user"]',
            'input[type="email"]',
            'input[type="text"]',
        ] if c]
    elif role == "password":
        return [c for c in [
            dom.get("password"),
            dom.get("login_password"),
            ds.get("password"),
            'input[name="txt_password"]',
            "#pass",
            'input[name="password"]',
            'input[name="pass"]',
            'input[name="passwd"]',
            'input[type="password"]',
            'input[placeholder="パスワード"]',
            'input[placeholder="password"]',
            'input[placeholder="Password"]',
        ] if c]
    elif role == "login_submit":
        return [c for c in [
            dom.get("login_submit"),
            dom.get("submit"),  # 互換fallback
            ds.get("login_button"),
            'button[name="login"]',
            'input[name="login"]',
            'button[type="submit"]',
            'input[type="submit"]',
            'button[name="submit"]',
        ] if c]
    return []



def _raw_scan_page(page) -> list:
    """
    P0-1: ページ＋全frameでinput/button/textarea/select/a/formのraw attributesを収集しログ出力する。
    value はsubmit/button/reset typeのみ取得。password/hiddenは絶対取得しない。
    要素数が多いframeを優先してソート。
    """
    frames = [page] + list(page.frames)
    all_results = []
    frame_data = []
    for ctx in frames:
        ctx_url = getattr(ctx, "url", "")
        try:
            elements = ctx.evaluate("""() => {
                const results = [];
                const safeValue = (el) => {
                    const t = (el.getAttribute('type') || '').toLowerCase();
                    if (t === 'password' || t === 'hidden') return null;
                    const tag = el.tagName.toLowerCase();
                    if (tag === 'input' && (t === 'submit' || t === 'button' || t === 'reset' || t === 'checkbox' || t === 'radio')) {
                        return el.value || null;
                    }
                    if (tag === 'button') return (el.textContent || '').trim().slice(0, 30) || null;
                    return null;
                };
                document.querySelectorAll('input, button, textarea, select, a[href], form').forEach(el => {
                    const tag = el.tagName.toLowerCase();
                    results.push({
                        tag: tag,
                        id: el.id || null,
                        name: el.getAttribute('name') || null,
                        type: el.getAttribute('type') || null,
                        placeholder: el.getAttribute('placeholder') || null,
                        autocomplete: el.getAttribute('autocomplete') || null,
                        class_name: el.className ? el.className.trim().slice(0, 60) : null,
                        text: (tag === 'button' || tag === 'a') ? (el.textContent || '').trim().slice(0, 50) : null,
                        value: safeValue(el),
                        href: tag === 'a' ? (el.getAttribute('href') || null) : null,
                        action: tag === 'form' ? (el.getAttribute('action') || null) : null,
                        method: tag === 'form' ? (el.getAttribute('method') || null) : null,
                        aria_label: el.getAttribute('aria-label') || null,
                        role: el.getAttribute('role') || null,
                    });
                });
                return results;
            }""")
        except Exception as ex:
            print(f"[RAW_SCAN_PAGE] frame={ctx_url} scan error: {type(ex).__name__}", flush=True)
            continue
        inputs    = [e for e in elements if e.get("tag") == "input"]
        buttons   = [e for e in elements if e.get("tag") == "button"]
        textareas = [e for e in elements if e.get("tag") == "textarea"]
        selects   = [e for e in elements if e.get("tag") == "select"]
        links     = [e for e in elements if e.get("tag") == "a"]
        forms     = [e for e in elements if e.get("tag") == "form"]
        print(f"[LOGIN_RAW_SCAN] frame={ctx_url} inputs={len(inputs)} buttons={len(buttons)}", flush=True)
        print(f"[RAW_SCAN_PAGE] frame={ctx_url} inputs={len(inputs)} buttons={len(buttons)} textareas={len(textareas)} selects={len(selects)} links={len(links)} forms={len(forms)}", flush=True)
        for idx, e in enumerate(inputs):
            print(f"[LOGIN_RAW_SCAN_INPUT] frame={ctx_url} index={idx} id={e.get('id')} name={e.get('name')} type={e.get('type')} placeholder={e.get('placeholder')} autocomplete={e.get('autocomplete')} class={e.get('class_name')}", flush=True)
        for idx, e in enumerate(buttons):
            print(f"[LOGIN_RAW_SCAN_BUTTON] frame={ctx_url} index={idx} id={e.get('id')} name={e.get('name')} type={e.get('type')} text={e.get('text')} value={e.get('value')} class={e.get('class_name')}", flush=True)
        for idx, e in enumerate(textareas):
            print(f"[RAW_SCAN_TEXTAREA] frame={ctx_url} index={idx} id={e.get('id')} name={e.get('name')} placeholder={e.get('placeholder')} class={e.get('class_name')}", flush=True)
        for idx, e in enumerate(selects):
            print(f"[RAW_SCAN_SELECT] frame={ctx_url} index={idx} id={e.get('id')} name={e.get('name')} class={e.get('class_name')}", flush=True)
        for idx, e in enumerate(links):
            print(f"[RAW_SCAN_LINK] frame={ctx_url} index={idx} id={e.get('id')} text={e.get('text')} href={e.get('href')} class={e.get('class_name')}", flush=True)
        for idx, e in enumerate(forms):
            print(f"[RAW_SCAN_FORM] frame={ctx_url} index={idx} id={e.get('id')} action={e.get('action')} method={e.get('method')} class={e.get('class_name')}", flush=True)
        for e in elements:
            e["_frame_url"] = ctx_url
            e["_ctx"] = ctx
        frame_data.append((len(inputs) + len(buttons) + len(textareas) + len(selects) + len(links) + len(forms), elements))
    # 要素数が多いframeを先頭にソート
    frame_data.sort(key=lambda x: x[0], reverse=True)
    for _, elements in frame_data:
        all_results.extend(elements)
    return all_results
def _infer_from_raw(elements: list, role: str):
    """
    P0-1: raw element attributesからlogin欄を推定し (locator, selector, frame_url) を返す。
    見つからなければ None を返す。
    """
    for e in elements:
        tag  = e.get("tag", "")
        eid  = e.get("id") or ""
        name = e.get("name") or ""
        typ  = (e.get("type") or "").lower()
        text = (e.get("text") or "").lower()
        ctx  = e.get("_ctx")
        furl = e.get("_frame_url", "")

        if role == "username":
            if tag == "input" and typ != "password":
                if eid.lower() in ("id", "loginid", "userid", "email", "username", "user"):
                    sel = f"#{eid}"
                elif name.lower() in ("id", "login_id", "loginid", "userid", "email", "username", "user"):
                    sel = f'input[name="{name}"]'
                elif typ in ("email", "text") and "pass" not in name.lower():
                    sel = f'input[name="{name}"]' if name else f'input[type="{typ}"]'
                else:
                    continue
                try:
                    loc = ctx.locator(sel).first
                    loc.wait_for(timeout=2000)
                    print(f"[LOGIN_RAW_INFER] username found: {sel} (frame={furl})", flush=True)
                    return loc, sel, furl
                except Exception:
                    pass

        elif role == "password":
            placeholder = (e.get("placeholder") or "").lower()
            pw_signals_id   = tag == "input" and ("password" in eid.lower() or "passwd" in eid.lower() or eid.lower() == "pass")
            pw_signals_name = tag == "input" and ("password" in name.lower() or "passwd" in name.lower() or name.lower() in ("pass", "password", "passwd"))
            pw_signals_ph   = tag == "input" and any(k in placeholder for k in ("password", "passwd", "パスワード"))
            is_std_pw   = tag == "input" and typ == "password"
            is_text_pw  = tag == "input" and typ == "text" and (pw_signals_id or pw_signals_name or pw_signals_ph)
            if not (is_std_pw or is_text_pw):
                continue
            if is_std_pw and eid.lower() in ("pass", "password", "passwd"):
                sel = f"#{eid}"
            elif is_std_pw and name.lower() in ("pass", "password", "passwd"):
                sel = f'input[name="{name}"]'
            elif pw_signals_id:
                sel = f"#{eid}"
            elif pw_signals_name:
                sel = f'input[name="{name}"]'
            elif is_std_pw:
                sel = 'input[type="password"]'
            elif pw_signals_ph:
                sel = f'input[placeholder="{e.get("placeholder")}"]'
            else:
                continue
            try:
                loc = ctx.locator(sel).first
                loc.wait_for(timeout=2000)
                print(f"[LOGIN_RAW_INFER] password found: {sel} (is_std={is_std_pw} is_text={is_text_pw} frame={furl})", flush=True)
                return loc, sel, furl
            except Exception:
                pass

        elif role in ("submit", "login_submit"):
            if tag in ("button", "input") and (
                typ == "submit" or
                name.lower() in ("login", "submit", "send") or
                any(k in text for k in ("ログイン", "login", "sign in", "submit"))
            ):
                if name:
                    sel = f'{tag}[name="{name}"]'
                elif eid:
                    sel = f"#{eid}"
                elif typ == "submit":
                    sel = f'{tag}[type="submit"]'
                else:
                    continue
                try:
                    loc = ctx.locator(sel).first
                    loc.wait_for(timeout=2000)
                    print(f"[LOGIN_RAW_INFER] submit found: {sel} (frame={furl})", flush=True)
                    return loc, sel, furl
                except Exception:
                    pass
    return None


def _find_with_fallback(page, candidates: list, label: str, raw_elements: list):
    """
    P0-1: 候補リスト探索 → 全滅時にraw element再推定。
    raw_elementsは要素数が多いframe順にソート済み。そのframe順で探索する。
    見つかったら (locator, selector, frame_url) を返す。
    """
    # raw_elementsのframe順序を保持（要素数多い順）
    seen_frames = []
    ordered_frames = []
    for e in raw_elements:
        ctx = e.get("_ctx")
        if ctx and ctx not in seen_frames:
            seen_frames.append(ctx)
            ordered_frames.append(ctx)
    # raw_elementsにないframeも末尾に追加
    all_frames = [page] + list(page.frames)
    for f in all_frames:
        if f not in ordered_frames:
            ordered_frames.append(f)

    tried = []
    for ctx in ordered_frames:
        ctx_url = getattr(ctx, "url", "")
        for sel in candidates:
            if not sel:
                continue
            try:
                all_locs = ctx.locator(sel)
                cnt = all_locs.count()
                if cnt == 0:
                    tried.append({"sel": sel, "frame": ctx_url, "err": "count=0"})
                    continue
                best_loc = None
                for i in range(cnt):
                    loc = all_locs.nth(i)
                    try:
                        visible  = loc.is_visible()
                        enabled  = loc.is_enabled()
                        bbox     = loc.bounding_box()
                        el_type  = loc.get_attribute("type") or ""
                        el_name  = loc.get_attribute("name") or ""
                        if el_type == "hidden":
                            tried.append({"sel": sel, "frame": ctx_url, "err": f"nth={i} type=hidden"})
                            continue
                        if visible and enabled and bbox:
                            print(f"[LOGIN_SELECTED] {label} found: {sel} nth={i} name={el_name} type={el_type} (frame={ctx_url})", flush=True)
                            best_loc = loc
                            break
                        tried.append({"sel": sel, "frame": ctx_url, "err": f"nth={i} visible={visible} enabled={enabled} bbox={bbox is not None}"})
                    except Exception as ie:
                        tried.append({"sel": sel, "frame": ctx_url, "err": f"nth={i} {type(ie).__name__}"})
                if best_loc is not None:
                    return best_loc, sel, ctx_url
            except Exception as e:
                tried.append({"sel": sel, "frame": ctx_url, "err": type(e).__name__})

    # 候補全滅 → raw再推定
    print(f"[_find_with_fallback] {label} candidates exhausted tried_count={len(tried)}", flush=True)
    role_map = {"username": "username", "password": "password", "login_submit": "login_submit"}
    role = role_map.get(label, label)
    result = _infer_from_raw(raw_elements, role)
    if result:
        return result
    raise RuntimeError(f"ログイン入力欄を検出できませんでした（{label}）")

def _find_first_locator(page, candidates: list, label: str):
    """
    P0-1: 候補selectorをページ＋全iframeで順番に探索する。
    見つかったら (locator, selector, frame_url) を返す。
    見つからなければ RuntimeError を raise し tried一覧をログに出す。
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    frames = [page] + list(page.frames)
    tried = []
    for ctx in frames:
        ctx_url = getattr(ctx, "url", "")
        for sel in candidates:
            if not sel:
                continue
            try:
                loc = ctx.locator(sel).first
                loc.wait_for(timeout=3000)
                print(f"[_find_first_locator] {label} found: {sel} (frame={ctx_url})", flush=True)
                return loc, sel, ctx_url
            except Exception as e:
                tried.append({"selector": sel, "frame": ctx_url, "error": type(e).__name__})
    print(f"[_find_first_locator] {label} not found. tried={tried}", flush=True)
    raise RuntimeError(f"{label} not found tried={tried}")


def run_login_form_check(media_mapping: dict, creds: dict, max_pages: int = 200) -> dict:
    """
    P0-1: Playwrightでログイン試行。候補リスト＋iframe探索。
    ID/PASSは絶対にログ・戻り値に含めない。
    """
    login_url = media_mapping.get("login_url")
    dom = media_mapping.get("dom_selectors") or {}
    verify_sel = media_mapping.get("verify_selector")

    username_candidates = _build_selector_candidates(media_mapping, "username")
    password_candidates = _build_selector_candidates(media_mapping, "password")
    submit_candidates   = _build_selector_candidates(media_mapping, "login_submit")

    dom = media_mapping.get("dom_selectors") or {}
    print(f"[LOGIN_SELECTOR_SOURCE] username={dom.get('username')} password={dom.get('password')} login_submit={dom.get('login_submit')} resolved_from={'login_submit' if dom.get('login_submit') else 'submit_fallback' if dom.get('submit') else 'hardcoded'}", flush=True)

    print("[AGENT_LOGIN_CHECK_INPUT]", {
        "login_url": login_url,
        "dom_keys": list(dom.keys()),
        "username_candidates": username_candidates,
        "password_candidates": password_candidates,
        "submit_candidates":   submit_candidates,
        "verify_sel": verify_sel,
    }, flush=True)

    if not login_url:
        return {"status": "BLOCKED", "executed": False, "message": "login_url が未設定です"}
    if not username_candidates or not password_candidates or not submit_candidates:
        return {"status": "BLOCKED", "executed": False, "message": "selector候補が生成できません"}
    if not is_playwright_enabled():
        return {
            "status": "WAITING_EXECUTOR",
            "executed": False,
            "login_checked": False,
            "login_success": False,
            "error_type": "PLAYWRIGHT_DISABLED",
            "message": "PLAYWRIGHT_ENABLED=false のためログイン確認は実行しません。",
        }

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        with sync_playwright() as p:
            browser = None
            login_success = False
            _error_result = None
            try:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
                page = browser.new_page()
                try:
                    page.goto(login_url, timeout=10000)
                except PlaywrightTimeout:
                    raise RuntimeError("page_load_timeout")

                # domcontentloaded待機 + 2秒安定待ち
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                # raw element scan（ログ出力）
                raw_elements = _raw_scan_page(page)

                # --- fallback探索（候補リスト → raw再推定） ---
                username_loc, username_sel_used, username_frame = _find_with_fallback(page, username_candidates, "username", raw_elements)
                password_loc, password_sel_used, password_frame = _find_with_fallback(page, password_candidates, "password", raw_elements)
                submit_loc,   submit_sel_used,   submit_frame   = _find_with_fallback(page, submit_candidates,   "login_submit", raw_elements)

                print(f"[LOGIN_SELECTED] username={username_sel_used}({username_frame}) password={password_sel_used} submit={submit_sel_used}", flush=True)

                try:
                    print(f"[LOGIN_FILL] role=username selector={username_sel_used} frame={username_frame}", flush=True)
                    username_loc.fill(creds["username"])
                except Exception as _fe:
                    try:
                        _vis = username_loc.is_visible()
                        _ena = username_loc.is_enabled()
                        _nm  = username_loc.get_attribute("name") or ""
                        _tp  = username_loc.get_attribute("type") or ""
                        print(f"[LOGIN_FILL_ERROR] role=username selector={username_sel_used} frame={username_frame} visible={_vis} enabled={_ena} name={_nm} type={_tp} error={_fe}", flush=True)
                    except Exception:
                        print(f"[LOGIN_FILL_ERROR] role=username selector={username_sel_used} error={_fe}", flush=True)
                    raise RuntimeError("username fill failed")
                try:
                    print(f"[LOGIN_FILL] role=password selector={password_sel_used} frame={password_frame}", flush=True)
                    password_loc.fill(creds["password"])
                except Exception:
                    raise RuntimeError("password fill failed")
                try:
                    print(f"[LOGIN_FILL] role=submit selector={submit_sel_used} frame={submit_frame}", flush=True)
                    # CSRFトークン確認（hiddenフィールドの値が入っているかログ）
                    try:
                        _hidden_vals = submit_loc.evaluate("""
                            el => {
                                const form = el.closest("form");
                                if (!form) return {};
                                const hidden = Array.from(form.querySelectorAll("input[type=hidden]"));
                                return Object.fromEntries(hidden.map(h => [h.name || h.id, h.value ? "SET" : "EMPTY"]));
                            }
                        """)
                        print(f"[LOGIN_HIDDEN_FIELDS] hidden_fields={_hidden_vals}", flush=True)
                    except Exception as _hve:
                        print(f"[LOGIN_HIDDEN_FIELDS] check_error={_hve}", flush=True)
                    submit_loc.scroll_into_view_if_needed()
                    submit_loc.click()
                except Exception:
                    raise RuntimeError("submit click failed")

                before_url = login_url
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                after_url = page.url
                after_title = page.title()
                url_changed = (after_url != before_url)
                try:
                    _pw_count_after = page.locator("input[type=password]").count()
                except Exception:
                    _pw_count_after = 0
                _pw_gone = (_pw_count_after == 0)
                print(f"[LOGIN_AFTER_SUBMIT] before_url={before_url} after_url={after_url} url_changed={url_changed} pw_gone={_pw_gone} verify_sel={verify_sel} title={after_title}", flush=True)
                # ログイン成功判定: URL変化 or パスワード欄消滅 or verify_selector
                _is_login_url = any(k in after_url for k in ["login", "signin", "Login", "SignIn", "auth", "Auth"])
                if (url_changed and not _is_login_url) or _pw_gone:
                    login_success = True
                elif verify_sel:
                    try:
                        page.wait_for_selector(verify_sel, timeout=3000)
                        login_success = True
                    except Exception:
                        login_success = False
                print(f"[LOGIN_SUCCESS_JUDGE] url_changed={url_changed} is_login_url={_is_login_url} pw_gone={_pw_gone} verify_sel={verify_sel} login_success={login_success}", flush=True)

            except RuntimeError as re:
                print(f"[browser_executor] login step error: {re}", flush=True)
                _error_result = {
                    "status": "BLOCKED", "executed": False,
                    "login_checked": True, "login_success": False,
                    "error_type": "STEP_ERROR",
                    "message": "ログイン入力欄を検出できませんでした。サイト構造がiframeまたは外部部品で構成されている可能性があります。再解析してください。" if "検出できませんでした" in str(re) else str(re),
                }
            except Exception as e:
                import traceback
                print(f"[browser_executor] login error: {type(e).__name__}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                _error_result = {
                    "status": "BLOCKED", "executed": False,
                    "login_checked": True, "login_success": False,
                    "error_type": type(e).__name__,
                    "message": f"ログイン中にエラーが発生しました: {type(e).__name__}",
                }
            finally:
                pass
            if _error_result:
                if browser:
                    try:
                        browser.close()
                    except Exception:
                        pass
                return _error_result

            # ── P21/P25/selector: browser生存中に実行し、finally で必ずclose ──
            try:
                # ── selector保存 ─────────────────────────────────────
                try:
                    from api.core.firestore_client import get_db as _get_db
                    _mapping_id = media_mapping.get("mapping_id") or media_mapping.get("id")
                    if _mapping_id and username_sel_used and password_sel_used and submit_sel_used:
                        import datetime as _dt_ls
                        _get_db().collection("media_mappings").document(_mapping_id).update({
                            "dom_selectors.username":     username_sel_used,
                            "dom_selectors.password":     password_sel_used,
                            "dom_selectors.login_submit": submit_sel_used,
                            "last_verified_at":           _dt_ls.datetime.utcnow(),
                        })
                        print(f"[LOGIN_SELECTED] saved username={username_sel_used} password={password_sel_used} submit={submit_sel_used}", flush=True)
                except Exception as _se:
                    print(f"[LOGIN_SELECTED] save error: {_se}", flush=True)

                # ── P21: ログイン後管理画面クローラー実行 ─────────────────
                _crawl_result = {"status": "SKIPPED", "reason": "LOGIN_CHECK_CRAWL_DISABLED", "pages_crawled": 0}
                if login_success and is_login_check_crawl_enabled():
                    try:
                        _crawl_mapping_id = (media_mapping.get("mapping_id") or media_mapping.get("id")) if media_mapping else None
                        if _crawl_mapping_id:
                            try:
                                _pre_url    = page.url
                                _pre_title  = page.title()
                                _pre_frames = [f.url for f in page.frames]
                                _browser_alive  = browser is not None
                                print(f"[P21_BROWSER_STATE] browser_alive={_browser_alive} page_url={_pre_url} title={_pre_title} frame_count={len(_pre_frames)} frames={_pre_frames}", flush=True)
                            except Exception as _pre_err:
                                print(f"[P21_BROWSER_STATE] read error: {_pre_err}", flush=True)
                            try:
                                page.wait_for_load_state("domcontentloaded", timeout=5000)
                            except Exception:
                                pass
                            from api.core.firestore_client import get_db as _get_db_crawl
                            _crawl_result = post_login_admin_crawl(
                                page,
                                _crawl_mapping_id,
                                _get_db_crawl(),
                                max_pages=_login_check_crawl_max_pages(max_pages),
                            )
                    except Exception as _crawl_err:
                        print(f"[P21] crawl error: {type(_crawl_err).__name__}: {_crawl_err}", flush=True)
                        _crawl_result = {"status": "ERROR", "error": type(_crawl_err).__name__}
                elif login_success:
                    print("[P21] login_check crawl skipped: LOGIN_CHECK_CRAWL_ENABLED=false", flush=True)

                # ── P25: login_check経路でセッション保存 ─────────────────
                if login_success:
                    _lc_mapping_id = media_mapping.get("mapping_id") or media_mapping.get("id") or ""
                    _lc_tenant_id  = str(media_mapping.get("tenant_id") or "")
                    if _lc_mapping_id:
                        try:
                            _save_cached_session(_lc_mapping_id, page.context, page, tenant_id=_lc_tenant_id)
                            print(f"[P25_SESSION_SAVE_LOGIN_CHECK] mapping_id={_lc_mapping_id}", flush=True)
                        except Exception as _lc_se:
                            print(f"[P25_SESSION_SAVE_LOGIN_CHECK_ERROR] mapping_id={_lc_mapping_id} error={_lc_se}", flush=True)
            finally:
                # D-2: 例外が発生しても必ずブラウザをclose
                if browser:
                    try:
                        browser.close()
                    except Exception:
                        pass

            # login結果に応じてreturn
            if login_success:
                _crawl_ok = _crawl_result.get("status") == "OK"
                return {
                    "status": "WAITING_EXECUTOR", "executed": False,
                    "login_checked": True, "login_success": True,
                    "message": "ログイン確認に成功しました。",
                    "crawl_result": _crawl_result,
                    "admin_crawl_completed":      _crawl_ok,
                    "pages_crawled":              _crawl_result.get("pages_crawled", 0),
                }
            else:
                return {
                    "status": "BLOCKED", "executed": False,
                    "login_checked": True, "login_success": False,
                    "message": "ログイン確認に失敗しました。selectorまたは認証情報を確認してください。",
                }

    except ImportError:
        return {"status": "WAITING_EXECUTOR", "executed": False, "message": "Playwrightがインストールされていません。"}
    except Exception as e:
        print(f"[browser_executor] unexpected error: {type(e).__name__}", flush=True)
        return {"status": "BLOCKED", "executed": False, "message": f"予期しないエラー: {type(e).__name__}"}



def _login_and_get_page(p, media_mapping: dict, creds: dict):
    """
    P0-1: Playwrightでログインし (browser, page) を返す。候補リスト＋iframe探索。
    失敗時は RuntimeError を raise。ID/PASSはログに出さない。
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    login_url  = media_mapping.get("login_url") or media_mapping.get("media_url") or ""
    verify_sel = media_mapping.get("verify_selector")

    if not login_url:
        raise RuntimeError("login_url / media_url が未設定です")
    print(f"[LOGIN_URL] resolved={login_url}", flush=True)

    username_candidates = _build_selector_candidates(media_mapping, "username")
    password_candidates = _build_selector_candidates(media_mapping, "password")
    submit_candidates   = _build_selector_candidates(media_mapping, "login_submit")

    browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
    page = browser.new_page()
    try:
        page.goto(login_url, timeout=10000)
    except PlaywrightTimeout:
        browser.close()
        raise RuntimeError("page_load_timeout")

    # domcontentloaded待機 + 2秒安定待ち
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass
    page.wait_for_timeout(2000)

    # raw element scan
    raw_elements = _raw_scan_page(page)

    try:
        username_loc, username_sel_used, _username_frame = _find_with_fallback(page, username_candidates, "username", raw_elements)
        password_loc, password_sel_used, _password_frame = _find_with_fallback(page, password_candidates, "password", raw_elements)
        submit_loc,   submit_sel_used,   _submit_frame   = _find_with_fallback(page, submit_candidates,   "login_submit", raw_elements)
    except RuntimeError:
        browser.close()
        raise

    print(f"[_LOGIN_AND_GET_PAGE] username={username_sel_used} password={password_sel_used} submit={submit_sel_used}", flush=True)

    try:
        print(f"[LOGIN_FILL] role=username selector={username_sel_used} frame={_username_frame}", flush=True)
        username_loc.fill(creds["username"])
    except Exception:
        browser.close()
        raise RuntimeError("username fill failed")
    try:
        print(f"[LOGIN_FILL] role=password selector={password_sel_used} frame={_password_frame}", flush=True)
        password_loc.fill(creds["password"])
    except Exception:
        browser.close()
        raise RuntimeError("password fill failed")
    try:
        print(f"[LOGIN_FILL] role=submit selector={submit_sel_used} frame={_submit_frame}", flush=True)
        # CSRFトークン確認
        try:
            _hidden_vals2 = submit_loc.evaluate("""
                el => {
                    const form = el.closest("form");
                    if (!form) return {};
                    const hidden = Array.from(form.querySelectorAll("input[type=hidden]"));
                    return Object.fromEntries(hidden.map(h => [h.name || h.id, h.value ? "SET" : "EMPTY"]));
                }
            """)
            print(f"[LOGIN_HIDDEN_FIELDS2] hidden_fields={_hidden_vals2}", flush=True)
        except Exception as _hve2:
            print(f"[LOGIN_HIDDEN_FIELDS2] check_error={_hve2}", flush=True)
        submit_loc.scroll_into_view_if_needed()
        submit_loc.click()
    except Exception:
        browser.close()
        raise RuntimeError("submit click failed")

    page.wait_for_load_state("domcontentloaded", timeout=35000)
    # ── 複合ログイン成功判定（verify_selは補助のみ） ──
    _lp_fail_signals = ["signin", "login", "sign_in"]
    _lp_after_url = page.url
    _lp_url_changed = (_lp_after_url != login_url)
    try:
        _lp_pw_count = page.locator("input[type=password]").count()
    except Exception:
        _lp_pw_count = 0
    _lp_url_is_login = (
        any(s in _lp_after_url.lower() for s in _lp_fail_signals)
        and _lp_pw_count > 0
    )
    _lp_pw_gone = (_lp_pw_count == 0)
    # ── P25: ログインページ残留チェック（優先判定） ──
    # /admin/を除外：Club華のようにログイン後で1ボURLに/admin/を含む場合に誤判定しないため
    # signin/loginはパスワード欄存在と合わせて判定する
    _lp_on_login_url = (
        any(s in _lp_after_url.lower() for s in ["signin", "login", "sign_in", "C1Login"])
        and _lp_pw_count > 0
    )
    _lp_verify_ok = False
    if verify_sel:
        try:
            page.wait_for_selector(verify_sel, timeout=3000)
            _lp_verify_ok = True
        except Exception:
            pass
    print(f"[AUTH_FINAL_STATE] url={_lp_after_url} pw_count={_lp_pw_count} verify_ok={_lp_verify_ok} url_changed={_lp_url_changed} on_login_url={_lp_on_login_url} pw_gone={_lp_pw_gone}", flush=True)
    if _lp_on_login_url and not _lp_verify_ok:
        browser.close()
        raise RuntimeError(f"login failed: still on login page url={_lp_after_url}")
    # パスワード欄が消えていればURLが変わらなくてもログイン成功とみなす
    # （/admin/ のようにログイン前後でURLが変わらないサイト対応）
    if not _lp_verify_ok and not (_lp_url_changed and not _lp_url_is_login) and not _lp_pw_gone:
        browser.close()
        raise RuntimeError("login failed: url did not change or still on login page")
    return browser, page

# ---------------------------------------------------------------------------
# ADMIN_URL_OPERATION_RULES
# URL・リンクテキスト・titleから page_type / operation候補 を機械的に分類する辞書
# LLM禁止。ヒューリスティック・スコアリングで完結。
# ---------------------------------------------------------------------------
ADMIN_URL_OPERATION_RULES = [
    {"keywords": ["cast_edit", "girl_edit", "cast_regist", "girl_regist"], "page_type": "entity_edit", "operations": ["entity_register", "entity_update", "media_replace"], "score": 95},
    {"keywords": ["cast_list", "girl_list", "standbygirl", "girlmypage"], "page_type": "entity_list", "operations": ["entity_update", "media_replace"], "score": 90},
    {"keywords": ["cast_sch", "shukkin", "shift", "schedule"], "page_type": "schedule_edit", "operations": ["schedule_update"], "score": 95},
    {"keywords": ["topics", "news", "notice", "realtime", "marquee"], "page_type": "text_edit", "operations": ["news_post", "text_update"], "score": 90},
    {"keywords": ["system", "fee", "price", "coupon"], "page_type": "price_edit", "operations": ["price_update"], "score": 90},
    {"keywords": ["freetext", "contents", "seo", "navi", "basic", "concept", "event"], "page_type": "text_edit", "operations": ["text_update", "status_update"], "score": 80},
    {"keywords": ["banner", "image", "back_image", "file", "upload", "logo"], "page_type": "media_edit", "operations": ["media_replace"], "score": 90},
    {"keywords": ["review", "inquiry", "readlog", "message"], "page_type": "monitoring", "operations": ["audit"], "score": 70},
]


def _normalize_admin_url_for_classification(url: str) -> str:
    """
    管理画面URLを分類用に正規化する。
    token/api_key/ckey/accessToken等の値はログ・分類キーに残さない。
    """
    import re
    u = (url or "").lower()
    u = re.sub(r"([?&])(token|_token__|api_key|accessToken|ckey|auth|password|pass|txt_password)=[^&]+", r"\1\2=***", u, flags=re.I)
    return u


def classify_admin_url_by_rules(url: str, link_text: str = "", title: str = "") -> dict:
    """
    URL・リンクテキスト・titleのみで管理画面ページを分類する。
    LLM禁止。
    """
    hay = " ".join([
        _normalize_admin_url_for_classification(url),
        (link_text or "").lower(),
        (title or "").lower(),
    ])
    best = {"page_type": "unknown", "operations": [], "score": 0, "reason": "no_rule_match", "matched_keywords": []}
    for rule in ADMIN_URL_OPERATION_RULES:
        matched = [kw for kw in rule["keywords"] if kw.lower() in hay]
        if not matched:
            continue
        score = rule.get("score", 0) + min(len(matched) * 3, 10)
        if score > best["score"]:
            best = {"page_type": rule["page_type"], "operations": rule["operations"], "score": score, "reason": "url_keyword_rule", "matched_keywords": matched}
    return best


def infer_operations_from_dom_structure(page_summary: dict) -> dict:
    """
    page_summary の forms/inputs/buttons/file_inputs/tables 等から operation候補を補強する。
    LLM禁止。
    """
    inputs      = page_summary.get("inputs", []) or []
    buttons     = page_summary.get("buttons", []) or []
    links       = page_summary.get("links", []) or []
    textareas   = page_summary.get("textareas", []) or []
    selects     = page_summary.get("selects", []) or []
    file_inputs = page_summary.get("file_inputs", []) or []
    ops     = set()
    signals = []
    if file_inputs:
        ops.add("media_replace")
        signals.append("file_input")
    if textareas:
        ops.add("text_update")
        signals.append("textarea")
    if selects:
        signals.append("select")
    joined = " ".join([
        str(x.get("name", "")) + " " + str(x.get("id", "")) + " " + str(x.get("text", "")) + " " + str(x.get("value", ""))
        for x in (inputs + buttons + links) if isinstance(x, dict)
    ]).lower()
    if any(k in joined for k in ["保存", "登録", "更新", "submit", "save", "regist"]):
        signals.append("save_button")
    if any(k in joined for k in ["price", "fee", "料金", "金額", "yen", "円"]):
        ops.add("price_update")
        signals.append("price_signal")
    if any(k in joined for k in ["date", "time", "shift", "schedule", "出勤", "開始", "終了"]):
        ops.add("schedule_update")
        signals.append("schedule_signal")
    if any(k in joined for k in ["title", "subject", "本文", "body", "content", "news", "topic"]):
        ops.add("news_post")
        ops.add("text_update")
        signals.append("news_text_signal")
    if page_summary.get("forms") and inputs:
        signals.append("form_with_inputs")
    score = 0
    if ops:                        score += 50
    if "save_button" in signals:   score += 25
    if "form_with_inputs" in signals: score += 15
    if file_inputs:                score += 10
    return {"operations": list(ops), "score": min(score, 100), "signals": signals}



# ---------------------------------------------------------------------------
# GENERIC_OPERATION_CONFIG
# operation_type → 必要なselector名・payload名・input_typeを定義
# 媒体別ifを廃止し、設定側で吸収する（P4: capability完全抽象化の前段）
# fields: [ {selector_key, payload_key, input_type} ]
#   input_type: "text" | "image" | "select" | "checkbox" | "file"
# submit_selector_key: dom_selectorsのどのキーをsubmitに使うか
# required_selector_keys: 不足チェック対象のキー一覧
# ---------------------------------------------------------------------------
GENERIC_OPERATION_CONFIG = {
    "text_update": {
        "capability_key": "can_update_text",
        "fields": [
            {"selector_key": "body", "payload_key": "text", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["body", "save"],
    },
    "news_post": {
        "capability_key": "can_post_news",
        "fields": [
            {"selector_key": "title", "payload_key": "title", "input_type": "text"},
            {"selector_key": "body",  "payload_key": "body",  "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["body", "save"],
    },
    # 店長ブログ（求人サイト）: news_post同等の入力構造。求人サイトはnews capabilityを
    # 持たないことが多いため capability ゲートは設けず、URL/セレクタ解決で制御する。
    "blog_post": {
        "capability_key": "",
        "fields": [
            {"selector_key": "title", "payload_key": "title", "input_type": "text"},
            {"selector_key": "body",  "payload_key": "body",  "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["body", "save"],
    },
    "status_update": {
        "capability_key": "can_update_text",
        "fields": [
            {"selector_key": "body", "payload_key": "body", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["body", "save"],
    },
    "media_replace": {
        "capability_key": "can_upload_image",
        "fields": [
            {"selector_key": "file", "payload_key": "file_path", "input_type": "file"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["file", "save"],
    },
    "schedule_update": {
        "capability_key": "can_update_schedule",
        "fields": [
            {"selector_key": "date_input", "payload_key": "schedule_value", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["save"],
    },
    "price_update": {
        "capability_key": "can_update_price",
        "fields": [
            {"selector_key": "price", "payload_key": "price_value", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["price", "save"],
    },
    "entity_register": {
        "capability_key": "can_register_entity",
        "fields": [
            {"selector_key": "required_inputs", "payload_key": "name", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["required_inputs", "save"],
    },
    "entity_update": {
        "capability_key": "can_update_entity",
        "fields": [
            {"selector_key": "editable_inputs", "payload_key": "value", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["editable_inputs", "save"],
    },
}



def create_authenticated_page(p, media_mapping: dict, creds: dict) -> dict:
    """
    P25: 認証済みページを統一生成して返す。
    セッションキャッシュ(メモリ+Firestore)を優先し、ミス時はフルログイン。
    返却: {"browser": browser, "context": context, "page": page}
    失敗時は RuntimeError を raise。
    """
    mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
    _tenant_id  = str(media_mapping.get("tenant_id") or "")
    # P25: キャッシュ確認
    _cached = _get_cached_session(mapping_id, tenant_id=_tenant_id) if mapping_id else None
    if _cached and _cached.get("cookies"):
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            context = browser.new_context()
            context.add_cookies(_cached["cookies"])
            page = context.new_page()
            # P25: Cookie復元後の遷移先をlogin_urlではなく保存済みURLを優先
            # current_url → admin_url → login_success_redirect_url → media_url → login_url
            _restore_url = (
                (_cached.get("current_url") or "")
                or media_mapping.get("admin_url", "")
                or media_mapping.get("login_success_redirect_url", "")
                or media_mapping.get("media_url", "")
                or media_mapping.get("login_url", "")
            )
            if _restore_url:
                print(f"[P25_SESSION_RESTORE_URL] mapping_id={mapping_id} url={_restore_url}", flush=True)
                page.goto(_restore_url, timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=5000)
                page.wait_for_timeout(1000)
            if _is_authenticated_page(page):
                print(f"[P25_SESSION_REUSE] mapping_id={mapping_id} source=cache cookies_count={len(_cached['cookies'])}", flush=True)
                return {"browser": browser, "context": context, "page": page}
            else:
                print(f"[P25_SESSION_EXPIRED] mapping_id={mapping_id} reason=auth_check_failed", flush=True)
                _clear_cached_session(mapping_id, reason="auth_check_failed", tenant_id=_tenant_id)
                browser.close()
        except Exception as _ce:
            print(f"[P25_SESSION_RESTORE_ERROR] mapping_id={mapping_id} error={_ce}", flush=True)
            try:
                browser.close()
            except Exception:
                pass
    # キャッシュミス or 復元失敗 → フルログイン
    browser, page = _login_and_get_page(p, media_mapping, creds)
    context = page.context
    if mapping_id:
        try:
            _save_cached_session(mapping_id, context, page, tenant_id=_tenant_id)
        except Exception as _se:
            print(f"[P25_SESSION_SAVE_ERROR] mapping_id={mapping_id} error={_se}", flush=True)
    return {"browser": browser, "context": context, "page": page}


def _capture_before_values(page, media_mapping: dict, operation_type: str) -> dict:
    """
    P7-rollback: _execute_operation実行前に各フィールドの現在値を取得。
    file inputはrollback不可のため記録のみ（rollbackable=False）。
    """
    config = GENERIC_OPERATION_CONFIG.get(operation_type, {})
    dom    = media_mapping.get("dom_selectors", {})
    before_values = {}

    for field in config.get("fields", []):
        sel        = dom.get(field["selector_key"])
        input_type = field.get("input_type", "text")
        key        = field["selector_key"]

        if not sel:
            continue

        if input_type == "file":
            before_values[key] = {"value": None, "input_type": "file", "rollbackable": False}
            continue

        try:
            el = page.query_selector(sel)
            if el is None:
                continue
            if input_type == "checkbox":
                val = el.is_checked()
            elif input_type == "select":
                val = page.eval_on_selector(sel, "el => el.value")
            else:
                val = el.input_value() if el.get_attribute("type") != "hidden" else el.inner_text()
            before_values[key] = {"value": val, "input_type": input_type, "rollbackable": True}
        except Exception as _bv_e:
            print(f"[P14_BEFORE_VALUE_ERROR] key={key} error={type(_bv_e).__name__}", flush=True)
            before_values[key] = {"value": None, "input_type": input_type, "rollbackable": False}

    return before_values
def _p27_anomaly_check(
    mapping_id: str,
    event_type: str,
    operation_type: str = "",
    selector_key: str = "",
    consecutive_fail_threshold: int = 3,
    posting_gap_days: int = 7,
) -> None:
    """
    P27: 異常検知。以下の3種を検知してアラートログを出力しFirestoreに記録。
    1. consecutive_failures カウントアップ + しきい値超過アラート
    2. 投稿ギャップ検知（last_success_atから現在までの日数）
    3. selector破損集計（SELECTOR_BROKEN発生時）
    event_type: "login_failed" | "operation_failed" | "selector_broken" | "success"
    """
    if not mapping_id:
        return
    try:
        import datetime as _dt_p27
        from api.core.firestore_client import get_db as _get_db_p27
        _db_p27 = _get_db_p27()
        if _db_p27 is None:
            return
        _ref = _db_p27.collection("media_mappings").document(mapping_id)
        _snap = _ref.get()
        if not _snap.exists:
            return
        _data = _snap.to_dict() or {}
        now = _dt_p27.datetime.utcnow()

        # 1. consecutive_failures カウントアップ
        # E-3: Increment で原子的に更新しread-modify-writeレースを排除
        if event_type in ("login_failed", "operation_failed"):
            try:
                from google.cloud.firestore import Increment as _FSInc27
                _ref.update({
                    "consecutive_failures": _FSInc27(1),
                    "last_failure_at": now,
                    "last_failure_type": event_type,
                })
            except Exception:
                _cf_fallback = (_data.get("consecutive_failures") or 0) + 1
                _ref.update({
                    "consecutive_failures": _cf_fallback,
                    "last_failure_at": now,
                    "last_failure_type": event_type,
                })
            _cf = (_data.get("consecutive_failures") or 0) + 1
            print(f"[P27_CONSECUTIVE_FAILURE] mapping_id={mapping_id} count~={_cf} event={event_type}", flush=True)
            if _cf >= consecutive_fail_threshold:
                print(f"[P27_ALERT] mapping_id={mapping_id} CONSECUTIVE_FAILURES>={consecutive_fail_threshold} event={event_type}", flush=True)
        elif event_type == "success":
            _ref.update({"consecutive_failures": 0})

        # 2. 投稿ギャップ検知
        if event_type == "success":
            _last_success = _data.get("last_success_at")
            if _last_success:
                try:
                    _ls_dt = _last_success.ToDatetime() if hasattr(_last_success, "ToDatetime") else _last_success
                    if hasattr(_ls_dt, "tzinfo") and _ls_dt.tzinfo is not None:
                        _ls_dt = _ls_dt.replace(tzinfo=None)
                    _gap_days = (now - _ls_dt).days
                    if _gap_days >= posting_gap_days:
                        print(f"[P27_ALERT] mapping_id={mapping_id} POSTING_GAP={_gap_days}days threshold={posting_gap_days}", flush=True)
                except Exception as _gap_e:
                    print(f"[P27_GAP_CHECK_ERROR] mapping_id={mapping_id} error={_gap_e}", flush=True)

        # 3. selector破損集計
        if event_type == "selector_broken" and selector_key:
            _broken_key = f"selector_broken_counts.{selector_key}"
            try:
                from google.cloud.firestore import Increment as _FSIncrement
                _ref.update({_broken_key: _FSIncrement(1)})
            except Exception:
                _cur = (_data.get("selector_broken_counts") or {}).get(selector_key, 0)
                _ref.update({f"selector_broken_counts.{selector_key}": _cur + 1})
            print(f"[P27_SELECTOR_BROKEN] mapping_id={mapping_id} selector_key={selector_key} op={operation_type}", flush=True)

    except Exception as _p27_e:
        print(f"[P27_ANOMALY_CHECK_ERROR] mapping_id={mapping_id} event={event_type} error={_p27_e}", flush=True)

def _capture_after_values(page, media_mapping: dict, operation_type: str) -> dict:
    """
    P28: 操作実行後に各フィールドの値を取得する。
    _capture_before_valuesと同じ構造で取得し、diffに使用する。
    """
    config = GENERIC_OPERATION_CONFIG.get(operation_type, {})
    dom    = media_mapping.get("dom_selectors", {})
    after_values = {}
    for field in config.get("fields", []):
        sel        = dom.get(field["selector_key"])
        input_type = field.get("input_type", "text")
        key        = field["selector_key"]
        if not sel:
            continue
        if input_type == "file":
            after_values[key] = {"value": None, "input_type": "file"}
            continue
        try:
            el = page.query_selector(sel)
            if el is None:
                continue
            if input_type == "checkbox":
                val = el.is_checked()
            elif input_type == "select":
                val = page.eval_on_selector(sel, "el => el.value")
            else:
                val = el.input_value() if el.get_attribute("type") != "hidden" else el.inner_text()
            after_values[key] = {"value": val, "input_type": input_type}
        except Exception as _av_e:
            print(f"[P28_AFTER_VALUE_ERROR] key={key} error={type(_av_e).__name__}", flush=True)
            after_values[key] = {"value": None, "input_type": input_type}
    return after_values

def _build_diff(before_values: dict, after_values: dict) -> dict:
    """
    P28: before_valuesとafter_valuesを比較してフィールドごとの差分dictを返す。
    戻り値: { field_key: {"before": val, "after": val, "changed": bool} }
    """
    diff = {}
    all_keys = set(list(before_values.keys()) + list(after_values.keys()))
    for key in all_keys:
        bval = (before_values.get(key) or {}).get("value")
        aval = (after_values.get(key) or {}).get("value")
        diff[key] = {
            "before":  bval,
            "after":   aval,
            "changed": bval != aval,
        }
    changed_count = sum(1 for v in diff.values() if v["changed"])
    print(f"[P28_DIFF] fields={len(diff)} changed={changed_count}", flush=True)
    return diff



def _rollback_fields(page, media_mapping: dict, operation_type: str, before_values: dict) -> dict:
    """
    P7-rollback: 更新失敗時に before_values へ戻す。
    rollback不可フィールドはskip。
    戻り値: { attempted, success, restored_fields, failed_fields, reason }
    """
    config = GENERIC_OPERATION_CONFIG.get(operation_type, {})
    dom    = media_mapping.get("dom_selectors", {})

    restored_fields = []
    failed_fields   = []
    skip_reason     = ""

    rollbackable_keys = [k for k, v in before_values.items() if v.get("rollbackable")]
    non_rollbackable  = [k for k, v in before_values.items() if not v.get("rollbackable")]

    if non_rollbackable:
        skip_reason = f"file input rollback unsupported: {non_rollbackable}"

    if not rollbackable_keys:
        return {
            "attempted":        False,
            "success":          False,
            "restored_fields":  [],
            "failed_fields":    [],
            "reason":           skip_reason or "rollback対象フィールドなし",
        }

    for key, info in before_values.items():
        if not info.get("rollbackable"):
            continue
        sel        = dom.get(key)
        val        = info.get("value", "")
        input_type = info.get("input_type", "text")

        if not sel:
            failed_fields.append(key)
            continue

        try:
            page.wait_for_selector(sel, timeout=3000)
            if input_type == "checkbox":
                if val:
                    page.check(sel)
                else:
                    page.uncheck(sel)
            elif input_type == "select":
                page.select_option(sel, str(val))
            else:
                page.fill(sel, str(val) if val is not None else "")
            restored_fields.append(key)
        except Exception:
            failed_fields.append(key)

    return {
        "attempted":       True,
        "success":         len(failed_fields) == 0,
        "restored_fields": restored_fields,
        "failed_fields":   failed_fields,
        "reason":          skip_reason if skip_reason else ("" if not failed_fields else f"復元失敗: {failed_fields}"),
    }



def _execute_operation_steps(
    page,
    media_mapping: dict,
    operation_steps: list,
    payload: dict,
    prior_step_results: list = None,
    task_id: str = "",
    db=None,
) -> list:
    """
    P14: multi-step operation runner。
    operation_stepsを順番に実行し、step_results(list)を返す。
    required=TrueのstepでRuntimeErrorが発生した場合は即停止。
    P26: SELECTOR_BROKEN/LOGIN_EXPIRED/SERVER_ERROR/TIMEOUTの4分類でリトライ実装。
    media_mappings本体は変更しない。Secretは出力しない。
    """
    import time as _time_mod
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    dom  = media_mapping.get("dom_selectors", {})
    urls = media_mapping.get("urls", {})

    # Checkpoint/resume: pre-populate from prior completed steps
    _prior_done_ids = {str(r.get("step_id") or "") for r in (prior_step_results or []) if r.get("status") == "DONE"}
    step_results = [r for r in (prior_step_results or []) if r.get("status") == "DONE"]
    if _prior_done_ids:
        print(f"[CHECKPOINT_RESUME] task_id={task_id} prior_done={len(_prior_done_ids)}", flush=True)

    sorted_steps = sorted(operation_steps, key=lambda s: s.get("order", 99))

    # P26: リトライ設定
    _SELECTOR_BROKEN_KEYWORDS = ["selector", "fill", "click", "wait_for_selector", "locator", "select", "upload"]
    _MAX_RETRY_TIMEOUT   = 3
    _MAX_RETRY_SERVER    = 1
    _MAX_RETRY_SELECTOR  = 1

    def _classify_error(e, step_type):
        """P26: エラー分類"""
        msg = str(e).lower()
        if isinstance(e, PlaywrightTimeout):
            return "TIMEOUT"
        if any(k in msg for k in ["selector", "locator", "element", "fill", "click"]):
            return "SELECTOR_BROKEN"
        if any(k in msg for k in ["net::", "connection", "socket", "econnrefused", "502", "503", "504"]):
            return "SERVER_ERROR"
        if any(k in msg for k in ["login", "auth", "session", "expired", "unauthorized"]):
            return "LOGIN_EXPIRED"
        return "STEP_ERROR"

    def _selector_from_step(step: dict, key_name: str = "selector_key") -> str:
        sel = dom.get(step.get(key_name, ""), "")
        if sel:
            return sel
        raw = step.get("selector")
        if isinstance(raw, dict):
            return raw.get("selector") or ""
        return raw or ""

    def _select_option_fuzzy(sel: str, value) -> None:
        raw_value = str(value or "")
        try:
            page.select_option(sel, raw_value)
            return
        except Exception as first_error:
            options = []
            try:
                options = page.eval_on_selector(
                    sel,
                    """el => Array.from(el.options || []).map(o => ({
                        value: o.value || "",
                        text: (o.textContent || "").trim()
                    }))"""
                )
            except Exception:
                raise first_error
            target = raw_value.lower()
            matched = None
            for opt in options:
                ov = str(opt.get("value") or "")
                ot = str(opt.get("text") or "")
                blob = f"{ov} {ot}".lower()
                if target and (target in blob or blob in target):
                    matched = ov
                    break
            if matched is None and options:
                matched = str(options[0].get("value") or "")
            if matched is None:
                raise first_error
            page.select_option(sel, matched)

    def _selector_kind(sel: str, step: dict) -> dict:
        raw = step.get("selector")
        if isinstance(raw, dict):
            tag = (raw.get("tag") or "").lower()
            typ = (raw.get("type") or "").lower()
            if tag or typ:
                return {"tag": tag, "type": typ}
        try:
            kind = page.eval_on_selector(
                sel,
                """el => ({
                    tag: (el.tagName || "").toLowerCase(),
                    type: (el.getAttribute("type") || "").toLowerCase()
                })"""
            )
            if isinstance(kind, dict):
                return {
                    "tag": (kind.get("tag") or "").lower(),
                    "type": (kind.get("type") or "").lower(),
                }
        except Exception:
            pass
        sel_l = str(sel).lower()
        if sel_l.startswith("select"):
            return {"tag": "select", "type": ""}
        if "type='checkbox'" in sel_l or 'type="checkbox"' in sel_l:
            return {"tag": "input", "type": "checkbox"}
        if "type='radio'" in sel_l or 'type="radio"' in sel_l:
            return {"tag": "input", "type": "radio"}
        return {"tag": "", "type": ""}

    for step in sorted_steps:
        step_id   = step.get("step_id", "")
        step_type = step.get("step_type", "")
        required  = step.get("required", True)
        started   = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
        error_msg = ""
        success   = False

        # Checkpoint/resume: skip already-completed steps
        if step_id and step_id in _prior_done_ids:
            continue

        # P26: ログイン切れチェック（各step実行前）
        # cross_media_ai_fill は自前でナビゲーションするためチェックをスキップ
        if step_type not in ("login", "sleep", "cross_media_ai_fill") and not _is_authenticated_page(page):
            print(f"[P26_LOGIN_EXPIRED] step_id={step_id} reason=auth_check_failed_before_step", flush=True)
            mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
            if mapping_id:
                _clear_cached_session(mapping_id, reason="login_expired_mid_operation",
                                      tenant_id=str(media_mapping.get("tenant_id") or ""))
            raise RuntimeError(f"[P26_LOGIN_EXPIRED] step:{step_id} ログインセッション切れ")

        # P26: リトライループ
        _retry_count   = 0
        _error_category = None
        _last_exception = None
        _terminal_side_effect_done = False

        while True:
            try:
                if step_type == "navigate":
                    url_key = step.get("url_key", "")
                    # P24/P14貫通修正:
                    # rebuild_operation_steps が生成する target_url を最優先で使う。
                    # 旧仕様の urls[url_key] / payload[url_key] も後方互換で維持。
                    url = step.get("target_url") or step.get("url") or urls.get(url_key) or payload.get(url_key, "")
                    if not url:
                        raise RuntimeError(f"navigate: target_url/url_key '{url_key}' が未設定です")
                    page.goto(url, timeout=15000)
                    page.wait_for_load_state("domcontentloaded", timeout=35000)
                    # P26: ナビゲーション後のセッション切れチェック（ログインページへリダイレクトされていた場合）
                    if not _is_authenticated_page(page):
                        _nav_mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
                        if _nav_mapping_id:
                            _clear_cached_session(_nav_mapping_id, reason="login_expired_after_navigate",
                                                  tenant_id=str(media_mapping.get("tenant_id") or ""))
                        print(f"[P26_LOGIN_EXPIRED] step_id={step_id} reason=auth_check_failed_after_navigate url={url}", flush=True)
                        raise RuntimeError(f"[P26_LOGIN_EXPIRED] step:{step_id} navigate後セッション切れ url={url}")
                    success = True

                elif step_type == "wait_for_selector":
                    sel = _selector_from_step(step)
                    if not sel:
                        raise RuntimeError(f"wait_for_selector: selector未設定")
                    page.wait_for_selector(sel, timeout=step.get("timeout", 5000))
                    success = True

                elif step_type == "fill":
                    sel   = _selector_from_step(step)
                    value = payload.get(step.get("payload_key", ""), "")
                    if not sel:
                        raise RuntimeError(f"fill: selector未設定 ({step.get('selector_key')})")
                    page.wait_for_selector(sel, timeout=5000)
                    _kind = _selector_kind(sel, step)
                    if _kind.get("tag") == "select":
                        _select_option_fuzzy(sel, value)
                    elif _kind.get("type") == "checkbox":
                        if value:
                            page.check(sel)
                        else:
                            page.uncheck(sel)
                    elif _kind.get("type") == "radio":
                        page.check(sel)
                    else:
                        page.fill(sel, str(value))
                    success = True

                elif step_type == "click":
                    sel = _selector_from_step(step)
                    if not sel:
                        raise RuntimeError(f"click: selector未設定 ({step.get('selector_key')})")
                    page.wait_for_selector(sel, timeout=5000)
                    page.click(sel)
                    if _is_terminal_operation_step(step):
                        _terminal_side_effect_done = True
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=35000)
                    except Exception:
                        pass
                    success = True

                elif step_type == "select":
                    sel   = _selector_from_step(step)
                    value = payload.get(step.get("payload_key", ""), "")
                    if not sel:
                        raise RuntimeError(f"select: selector未設定")
                    page.wait_for_selector(sel, timeout=5000)
                    _select_option_fuzzy(sel, value)
                    success = True

                elif step_type == "upload_file":
                    sel   = _selector_from_step(step)
                    value = payload.get(step.get("payload_key", ""), "")
                    if not sel:
                        raise RuntimeError(f"upload_file: selector未設定")
                    # G-2: ファイル存在チェックなしでset_input_filesを呼ぶと誤分類リトライが発生する
                    import os as _os_g2
                    if not value or not _os_g2.path.exists(str(value)):
                        step_result["status"] = "BLOCKED"
                        step_result["error"] = "FILE_NOT_FOUND"
                        step_result["message"] = f"upload_file: ファイルが見つかりません: {value}"
                        print(f"[UPLOAD_FILE_NOT_FOUND] selector={sel} path={value}", flush=True)
                        success = False
                    else:
                        page.set_input_files(sel, str(value))
                        success = True

                elif step_type == "search":
                    sel_input  = _selector_from_step(step)
                    sel_submit = dom.get(step.get("submit_selector_key", ""), "")
                    value      = payload.get(step.get("payload_key", ""), "")
                    if not sel_input:
                        raise RuntimeError(f"search: input selector未設定")
                    page.wait_for_selector(sel_input, timeout=5000)
                    page.fill(sel_input, str(value))
                    if sel_submit:
                        page.click(sel_submit)
                        if _is_terminal_operation_step(step):
                            _terminal_side_effect_done = True
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=35000)
                        except Exception:
                            pass
                    success = True

                elif step_type == "verify":
                    v = _verify_operation_detail(page, media_mapping, before_hash="", after_html=None)
                    if not v.get("verified"):
                        raise RuntimeError(f"verify: 検証失敗 method={v.get('method')}")
                    success = True

                elif step_type == "sleep":
                    _time_mod.sleep(step.get("duration", 1))
                    success = True

                elif step_type == "login":
                    success = True

                elif step_type == "cross_media_ai_fill":
                    # クロスメディア AI自動フォーム入力: Gemini でラベルベースマッピングして全フィールド入力
                    from api.core.llm_client import call_llm_json
                    import json as _json

                    _cm_target_url = step.get("target_url") or media_mapping.get("admin_url") or media_mapping.get("login_success_redirect_url") or media_mapping.get("media_url") or ""
                    if _cm_target_url:
                        print(f"[CROSS_MEDIA_AI_FILL] navigating to {_cm_target_url}", flush=True)
                        page.goto(_cm_target_url, timeout=20000)
                        page.wait_for_load_state("domcontentloaded", timeout=15000)
                        page.wait_for_timeout(1500)

                    # ペイロードから入力データを収集（内部管理キーを除外）
                    _SKIP_KEYS = {"media_mapping_id", "media_name", "cross_media_task_id",
                                  "source_mode", "source_url", "source_mapping_id",
                                  "cross_media_target_index", "cross_media_item_index",
                                  "cross_media_entity_label", "cross_media_entity_url",
                                  "cross_media_selected_fields", "cross_media_instruction",
                                  "cross_media_query", "operation_type"}
                    _fill_data = {k: v for k, v in payload.items()
                                  if isinstance(v, str) and v.strip() and k not in _SKIP_KEYS}
                    if isinstance(payload.get("structured_fields"), (dict, list)) and payload.get("structured_fields"):
                        _fill_data["structured_fields"] = payload["structured_fields"]
                    if isinstance(payload.get("source_media_schema"), dict) and payload.get("source_media_schema"):
                        _fill_data["source_media_schema"] = payload["source_media_schema"]

                    # 更新範囲（何を）: ユーザーが選択した宛先フィールドのラベル一覧（空=全項目）
                    _cm_selected = [str(s).strip() for s in (payload.get("cross_media_selected_fields") or []) if str(s).strip()]
                    _op_type = str(step.get("operation_type") or payload.get("operation_type") or "情報登録")
                    _known_fields_for_ai = []
                    try:
                        _seen_known = set()
                        def _add_known_fields(_fields, _source):
                            for _field in (_fields or [])[:300]:
                                if not isinstance(_field, dict) or not _is_business_form_field(_field):
                                    continue
                                _key = (
                                    str(_field.get("selector") or "").strip(),
                                    str(_field.get("canonical") or "").strip(),
                                    str(_field.get("name") or "").strip(),
                                    str(_field.get("id") or "").strip(),
                                    str(_field.get("label") or "").strip(),
                                )
                                if _key in _seen_known:
                                    continue
                                _seen_known.add(_key)
                                _known_fields_for_ai.append({
                                    "label": _field.get("label", ""),
                                    "name": _field.get("name", ""),
                                    "id": _field.get("id", ""),
                                    "canonical": _field.get("canonical", ""),
                                    "selector": _field.get("selector", ""),
                                    "type": _field.get("type", "text"),
                                    "source": _field.get("source", _source),
                                })
                        _op_map = ((media_mapping.get("operation_mappings") or {}).get(_op_type) or {})
                        _add_known_fields(_op_map.get("fields") or ((_op_map.get("form_schema") or {}).get("fields") or []), "operation_mappings")
                        for _page in (media_mapping.get("manual_form_pages") or []):
                            if isinstance(_page, dict) and _page.get("op_type") == _op_type:
                                _add_known_fields(_page.get("fields") or [], "manual_form_pages")
                    except Exception as _known_e:
                        print(f"[CROSS_MEDIA_AI_FILL] known-field load error: {_known_e}", flush=True)

                    # フォームフィールドをラベルも含めて収集（JS実行）
                    _cm_fields = []
                    try:
                        _cm_fields = page.evaluate("""
                        () => {
                            const inputs = Array.from(document.querySelectorAll(
                                'input[type="text"], input[type="number"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea, select'
                            )).filter(el => !['hidden','submit','button','reset','checkbox','radio','file'].includes(el.type));
                            return inputs.slice(0, 200).map((inp, i) => {
                                const id = inp.id || '';
                                let label = '';
                                if (id) {
                                    const lEl = document.querySelector('label[for="' + id + '"]');
                                    if (lEl) label = lEl.textContent.trim().replace(/\\s+/g, ' ');
                                }
                                if (!label) {
                                    let p = inp.parentElement;
                                    for (let j = 0; j < 4 && p; j++) {
                                        const prev = p.previousElementSibling;
                                        if (prev && ['TD','TH','DT','LABEL','DIV','SPAN','P'].includes(prev.tagName)) {
                                            const t = prev.textContent.trim().replace(/\\s+/g, ' ');
                                            if (t && t.length < 40) { label = t; break; }
                                        }
                                        const lEl2 = p.querySelector('label');
                                        if (lEl2 && lEl2 !== inp) {
                                            const t = lEl2.textContent.trim().replace(/\\s+/g, ' ');
                                            if (t && t.length < 40) { label = t; break; }
                                        }
                                        p = p.parentElement;
                                    }
                                }
                                return {
                                    index: i,
                                    name: inp.name || '',
                                    id: id,
                                    type: inp.type || inp.tagName.toLowerCase(),
                                    placeholder: inp.placeholder || '',
                                    label: label,
                                    tag: inp.tagName.toLowerCase()
                                };
                            });
                        }
                        """)
                    except Exception as _fe_js:
                        print(f"[CROSS_MEDIA_AI_FILL] JS field extraction error: {_fe_js}", flush=True)

                    _cm_fields_raw_count = len(_cm_fields)
                    _cm_fields = [f for f in _cm_fields if _is_business_form_field(f)]
                    print(f"[CROSS_MEDIA_AI_FILL] detected {len(_cm_fields)}/{_cm_fields_raw_count} business fields, payload keys={list(_fill_data.keys())}", flush=True)

                    # Gemini にフィールドマッピングを依頼
                    _cm_ai_mapping = {}
                    _cm_ai_error = False
                    if _cm_fields and _fill_data:
                        _instruction = payload.get("cross_media_instruction") or payload.get("cross_media_query") or ""
                        _has_structured = isinstance(_fill_data.get("structured_fields"), (dict, list)) and bool(_fill_data.get("structured_fields"))
                        _src_hint = (
                            "※ structured_fields には取得元ページのフォーム/テーブルから抽出した実データが入っています。これを最優先で参照してください。"
                            if _has_structured else
                            "※ body/text/value には取得元ページ本文が入っています。ラベルに対応する値をそこから読み取ってください。"
                        )
                        _prompt = f"""あなたはWebフォーム自動入力AIです。
操作タイプ: {_op_type}
追加指示: {_instruction or 'なし'}
{_src_hint}

【取得元データ】
{_json.dumps(_fill_data, ensure_ascii=False, indent=2)}

【出力先フォームフィールド一覧】
{_json.dumps(_cm_fields, ensure_ascii=False, indent=2)}

【保存済みマッピング構造】
{_json.dumps(_known_fields_for_ai, ensure_ascii=False, indent=2)}

上記フィールド一覧の各フィールドについて、取得元データから適切な値を選んでください。
- labelやname・placeholderを手がかりに意味的にマッピングすること
- 保存済みマッピング構造(canonical/label/name)がある場合は意味合わせに必ず使うこと
- 日本語ラベルにも対応すること（例：名前→name, 年齢→age, キャッチコピー→catchcopy等）
- データが存在しない・不明なフィールドはスキップ（含めない）
- selectフィールドはラベル（表示テキスト）ではなくvalue値で指定すること

次の形式のJSONのみ返してください（説明文不要）:
{{"0": "値", "1": "値", ...}}  ← keyはフィールドのindex（文字列）
"""
                        try:
                            _cm_ai_mapping = call_llm_json(
                                prompt=_prompt,
                                system_prompt="JSONのみ出力。```json等のMarkdownブロック禁止。",
                                ai_tier="core",
                                max_tokens=2048,
                            )
                            print(f"[CROSS_MEDIA_AI_FILL] AI mapping result: {list(_cm_ai_mapping.keys())}", flush=True)
                        except Exception as _ai_e:
                            print(f"[CROSS_MEDIA_AI_FILL] AI mapping error: {_ai_e}", flush=True)
                            _cm_ai_error = True

                    # H-4: Geminiエラー時はフィールド充填なし → G-1の_fill_count==0チェックがFAILEDを返す
                    if _cm_ai_error and not _cm_ai_mapping:
                        print(f"[CROSS_MEDIA_AI_FILL] Gemini failed, will not submit form", flush=True)

                    # 更新範囲フィルタ: 選択フィールドのみ残す（labelで突合、indexでフォールバック）
                    if _cm_selected and _cm_ai_mapping:
                        _label_by_idx = {str(f.get("index")): (f.get("label") or f.get("name") or f.get("placeholder") or "")
                                         for f in _cm_fields}
                        def _is_selected(_i):
                            _lab = (_label_by_idx.get(str(_i)) or "").strip()
                            if _lab and any(_sel == _lab or _sel in _lab or _lab in _sel for _sel in _cm_selected):
                                return True
                            return str(_i) in _cm_selected  # index指定にも対応
                        _filtered = {k: v for k, v in _cm_ai_mapping.items() if _is_selected(k)}
                        print(f"[CROSS_MEDIA_AI_FILL] field filter: {len(_cm_ai_mapping)} -> {len(_filtered)} (selected={len(_cm_selected)})", flush=True)
                        _cm_ai_mapping = _filtered

                    # AI マッピング結果でフィールドを入力
                    _fill_count = 0
                    _cm_inputs_live = page.query_selector_all(
                        "input[type='text'], input[type='number'], input[type='email'], input[type='tel'], input[type='url'], input:not([type]), textarea, select"
                    )
                    _cm_inputs_live = [el for el in _cm_inputs_live
                                       if (el.get_attribute("type") or "") not in ("hidden","submit","button","reset","checkbox","radio","file")]
                    for _idx_str, _val in _cm_ai_mapping.items():
                        try:
                            _idx = int(_idx_str)
                            if _idx >= len(_cm_inputs_live):
                                continue
                            _inp = _cm_inputs_live[_idx]
                            _tag = _inp.evaluate("el => el.tagName.toLowerCase()")
                            if _tag == "select":
                                try:
                                    _inp.select_option(value=str(_val), timeout=3000)
                                except Exception:
                                    try:
                                        _inp.select_option(label=str(_val), timeout=3000)
                                    except Exception:
                                        pass
                            else:
                                _inp.fill(str(_val), timeout=3000)
                            _fill_count += 1
                        except Exception as _ff:
                            print(f"[CROSS_MEDIA_AI_FILL] fill idx={_idx_str} error: {_ff}", flush=True)

                    print(f"[CROSS_MEDIA_AI_FILL] filled {_fill_count}/{len(_cm_ai_mapping)} fields", flush=True)

                    # G-1: 1フィールドも入力できていない場合は空フォームsubmitを防ぐ
                    if _fill_count == 0:
                        print(f"[CROSS_MEDIA_AI_FILL] skipping submit: no fields filled (payload_keys={list(_fill_data.keys())} fields={len(_cm_fields)})", flush=True)
                        step_result["status"] = "FAILED"
                        step_result["error"] = "NO_FIELDS_FILLED"
                        step_result["message"] = "入力可能なフィールドが見つかりませんでした。payload またはフォーム構造を確認してください。"
                        success = False
                    else:
                        # サブミットボタンを探してクリック
                        _submit_btn = None
                        for _sel in ["button[type='submit']", "input[type='submit']"]:
                            try:
                                _c = page.query_selector(_sel)
                                if _c and _c.is_visible():
                                    _submit_btn = _c
                                    break
                            except Exception:
                                pass
                        if not _submit_btn:
                            _submit_texts = ["登録", "保存", "送信", "確認", "完了", "ok", "save", "submit", "register", "update"]
                            try:
                                for _b in page.query_selector_all("button, input[type='button']"):
                                    try:
                                        _bt = (_b.inner_text() or "").strip().lower()
                                        if any(t in _bt for t in _submit_texts) and _b.is_visible():
                                            _submit_btn = _b
                                            break
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        if _submit_btn:
                            try:
                                _submit_btn.click(timeout=5000)
                                page.wait_for_load_state("domcontentloaded", timeout=10000)
                                page.wait_for_timeout(1500)
                                print(f"[CROSS_MEDIA_AI_FILL] form submitted", flush=True)
                            except Exception as _se:
                                print(f"[CROSS_MEDIA_AI_FILL] submit error: {_se}", flush=True)
                        else:
                            print(f"[CROSS_MEDIA_AI_FILL] no submit button found", flush=True)
                        success = True

                    # スナップショット保存（差分更新の基準データ）
                    # 取得元データ（_fill_data）と実際にマッピングしたフィールド（ラベル→値）を記録
                    try:
                        from api.core.firestore_client import get_db as _get_snap_db
                        from google.cloud import firestore as _fs_snap
                        import hashlib as _hl
                        _snap_src_mid = str(payload.get("source_mapping_id") or "")
                        _snap_dst_mid = str(payload.get("media_mapping_id") or "")
                        _snap_ent_url = str(payload.get("cross_media_entity_url") or "")
                        if _snap_src_mid and _snap_ent_url:
                            _snap_db = _get_snap_db()
                            _snap_id = _hl.sha256(
                                f"{_snap_src_mid}_{_snap_dst_mid}_{_snap_ent_url}".encode()
                            ).hexdigest()[:32]
                            # AI マッピング結果をラベル→値に変換
                            _snap_mapped = {}
                            for _si, _sv in (_cm_ai_mapping or {}).items():
                                try:
                                    _si_int = int(_si)
                                    if _si_int < len(_cm_fields):
                                        _slabel = (
                                            _cm_fields[_si_int].get("label")
                                            or _cm_fields[_si_int].get("name")
                                            or _cm_fields[_si_int].get("placeholder")
                                            or f"field_{_si}"
                                        )
                                        _snap_mapped[_slabel] = str(_sv)
                                except Exception:
                                    pass
                            _snap_db.collection("cross_media_snapshots").document(_snap_id).set({
                                "tenant_id":         str(payload.get("tenant_id") or ""),
                                "source_mapping_id": _snap_src_mid,
                                "dest_mapping_id":   _snap_dst_mid,
                                "entity_url":        _snap_ent_url,
                                "entity_label":      str(payload.get("cross_media_entity_label") or ""),
                                "industry":          str(payload.get("industry") or "generic"),
                                "synced_at":         _fs_snap.SERVER_TIMESTAMP,
                                # 取得元の生データ（次回差分検出に使う）
                                "source_data":  {k: str(v) for k, v in _fill_data.items() if k not in {"structured_fields", "source_media_schema"}},
                                "source_structured_fields": payload.get("structured_fields") if isinstance(payload.get("structured_fields"), dict) else {},
                                # 宛先フォームに実際に書き込んだラベル→値
                                "mapped_fields": _snap_mapped,
                            })
                            print(f"[SNAPSHOT_SAVED] id={_snap_id} entity={_snap_ent_url[:60]}", flush=True)
                    except Exception as _snap_e:
                        print(f"[SNAPSHOT_SAVE_ERROR] {type(_snap_e).__name__}: {_snap_e}", flush=True)

                else:
                    raise RuntimeError(f"step_type '{step_type}' は未対応です")

                # 成功 → ループ脱出
                break

            except (RuntimeError, PlaywrightTimeout, Exception) as e:
                _last_exception = e
                _error_category = _classify_error(e, step_type)
                print(f"[P26_RETRY_CHECK] step_id={step_id} category={_error_category} retry={_retry_count} error={e}", flush=True)
                if _is_terminal_operation_step(step) and _terminal_side_effect_done:
                    ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
                    step_results.append({
                        "step_id":    step_id,
                        "step_type":  step_type,
                        "status":     "DONE",
                        "started_at": started,
                        "ended_at":   ended,
                        "error":      "terminal step clicked; post-click confirmation failed",
                        "error_category": "TERMINAL_STEP_UNVERIFIED",
                        "retry_count": _retry_count,
                        "terminal": True,
                        "retry_safe": False,
                    })
                    raise OperationStepRuntimeError(
                        f"[step:{step_id}][TERMINAL_STEP_UNVERIFIED] {str(e)}",
                        step_results=step_results,
                        failed_step_id=step_id,
                    )

                # LOGIN_EXPIRED → 即リトライ不可・呼び元へRaise
                if _error_category == "LOGIN_EXPIRED":
                    mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
                    if mapping_id:
                        _clear_cached_session(mapping_id, reason="login_expired_in_step",
                                              tenant_id=str(media_mapping.get("tenant_id") or ""))
                    error_msg = str(e)
                    ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
                    step_results.append({
                        "step_id":    step_id,
                        "step_type":  step_type,
                        "status":     "FAILED",
                        "started_at": started,
                        "ended_at":   ended,
                        "error":      error_msg,
                        "error_category": _error_category,
                        "retry_count": _retry_count,
                    })
                    raise OperationStepRuntimeError(
                        f"[P26_LOGIN_EXPIRED] step:{step_id} {error_msg}",
                        step_results=step_results,
                        failed_step_id=step_id,
                    )

                # TIMEOUT → 最大3回リトライ・5秒待機
                elif _error_category == "TIMEOUT":
                    if _retry_count < _MAX_RETRY_TIMEOUT:
                        _retry_count += 1
                        print(f"[P26_RETRY] TIMEOUT step_id={step_id} retry={_retry_count}/{_MAX_RETRY_TIMEOUT} waiting=5s", flush=True)
                        _time_mod.sleep(5)
                        continue
                    error_msg = str(e)

                # SELECTOR_BROKEN → 1回リトライ（即時）
                elif _error_category == "SELECTOR_BROKEN":
                    if _retry_count < _MAX_RETRY_SELECTOR:
                        _retry_count += 1
                        print(f"[P26_RETRY] SELECTOR_BROKEN step_id={step_id} retry={_retry_count}/{_MAX_RETRY_SELECTOR}", flush=True)
                        _time_mod.sleep(1)
                        continue
                    error_msg = str(e)
                    _p27_anomaly_check(
                        mapping_id=str(media_mapping.get("id") or media_mapping.get("mapping_id") or ""),
                        event_type="selector_broken",
                        operation_type=step_type,
                        selector_key=step.get("selector_key", ""),
                    )

                # SERVER_ERROR → 1回リトライ・3秒待機
                elif _error_category == "SERVER_ERROR":
                    if _retry_count < _MAX_RETRY_SERVER:
                        _retry_count += 1
                        print(f"[P26_RETRY] SERVER_ERROR step_id={step_id} retry={_retry_count}/{_MAX_RETRY_SERVER} waiting=3s", flush=True)
                        _time_mod.sleep(3)
                        continue
                    error_msg = str(e)

                else:
                    error_msg = str(e)

                # リトライ上限超過 or 非リトライ → FAILED記録
                success = False
                ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
                step_results.append({
                    "step_id":    step_id,
                    "step_type":  step_type,
                    "status":     "FAILED",
                    "started_at": started,
                    "ended_at":   ended,
                    "error":      error_msg,
                    "error_category": _error_category,
                    "retry_count": _retry_count,
                })
                if required:
                    raise OperationStepRuntimeError(
                        f"[step:{step_id}][{_error_category}] {error_msg}",
                        step_results=step_results,
                        failed_step_id=step_id,
                    )
                else:
                    break

        if success:
            ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
            step_results.append({
                "step_id":    step_id,
                "step_type":  step_type,
                "status":     "DONE",
                "started_at": started,
                "ended_at":   ended,
                "error":      "",
                "error_category": None,
                "retry_count": _retry_count,
            })
            # Checkpoint: persist step completion to Firestore immediately
            if task_id and db is not None:
                try:
                    import datetime as _dt_ckpt
                    db.collection("agent_tasks").document(task_id).update({
                        "checkpoint_step_results": step_results,
                        "checkpoint_step_id":      step_id,
                        "updated_at":              _dt_ckpt.datetime.utcnow(),
                    })
                except Exception as _ckpt_e:
                    print(f"[CHECKPOINT_WRITE_ERROR] task_id={task_id} step_id={step_id} err={type(_ckpt_e).__name__}", flush=True)

    return step_results


def _ai_native_fill(page, operation_type: str, payload: dict, config: dict,
                    media_mapping: dict, db=None) -> dict:
    """
    セレクター未設定時のAI自動フォーム検出・入力。
    1. ページのフォーム要素をスキャン
    2. LLMにpayloadフィールドとのマッピングを依頼
    3. 入力実行 → Firestoreへキャッシュ（次回高速化）
    """
    import json as _json
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    try:
        elements = _raw_scan_page(page)
    except Exception as e:
        return {"success": False, "error": f"page scan failed: {e}"}

    # _ctx/_frame_url はJSON非シリアライザブルなので除外
    elem_summary = [
        {k: v for k, v in e.items() if k not in ("_ctx", "_frame_url")}
        for e in elements[:200]
    ]

    fields_desc = []
    for f in config.get("fields", []):
        val = payload.get(f["payload_key"], "")
        if val:
            fields_desc.append(
                f"  - payload_key={f['payload_key']} selector_key={f['selector_key']} "
                f"input_type={f.get('input_type','text')} value_preview={str(val)[:60]}"
            )

    system_prompt = (
        "あなたはWebフォームのDOM解析専門AIです。"
        "ページのフォーム要素リストとpayloadフィールドを照合し、最適なCSSセレクターを特定してください。"
        "回答はJSONのみ（```json不要）。形式:\n"
        '{"mappings":[{"payload_key":"xxx","selector_key":"yyy","selector":"input#id","input_type":"text"}],'
        '"submit_selector":"button[type=submit]"}'
    )
    user_msg = (
        f"operation_type: {operation_type}\n\n"
        f"入力すべきフィールド:\n{chr(10).join(fields_desc) if fields_desc else '(なし)'}\n"
        f"submit_selector_key: {config.get('submit_selector_key','submit')}\n\n"
        f"ページのフォーム要素（最大200件）:\n{_json.dumps(elem_summary, ensure_ascii=False)}"
    )

    try:
        from api.core.llm_client import call_llm
        raw = call_llm(system_prompt, [{"role": "user", "content": user_msg}],
                       ai_tier="core", temperature=0.1)
    except Exception as e:
        return {"success": False, "error": f"LLM call failed: {e}"}

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        data = _json.loads(cleaned)
    except Exception as e:
        print(f"[AI_NATIVE_PARSE_ERROR] op={operation_type} err={e} raw={raw[:200]}", flush=True)
        return {"success": False, "error": f"LLM response parse failed: {e}"}

    mappings = data.get("mappings", [])
    submit_sel = data.get("submit_selector", "")
    filled_count = 0
    discovered: dict = {}

    for m in mappings:
        sel = m.get("selector", "")
        pkey = m.get("payload_key", "")
        skey = m.get("selector_key", "")
        itype = m.get("input_type", "text")
        val = payload.get(pkey, "")
        if not sel or not val:
            continue
        try:
            page.wait_for_selector(sel, timeout=5000)
            if itype == "file":
                page.set_input_files(sel, str(val))
            elif itype == "select":
                page.select_option(sel, str(val))
            elif itype == "checkbox":
                if val:
                    page.check(sel)
                else:
                    page.uncheck(sel)
            else:
                page.fill(sel, str(val))
            if skey:
                discovered[skey] = sel
            filled_count += 1
            print(f"[AI_NATIVE_FILL] op={operation_type} payload_key={pkey} sel={sel[:60]}", flush=True)
        except PlaywrightTimeout:
            print(f"[AI_NATIVE_FILL_TIMEOUT] op={operation_type} pkey={pkey} sel={sel[:60]}", flush=True)
        except Exception as fe:
            print(f"[AI_NATIVE_FILL_ERROR] op={operation_type} pkey={pkey} sel={sel[:60]} err={fe}", flush=True)

    if submit_sel:
        try:
            page.wait_for_selector(submit_sel, timeout=5000)
            page.click(submit_sel)
            page.wait_for_load_state("domcontentloaded", timeout=35000)
            submit_skey = config.get("submit_selector_key", "submit")
            discovered[submit_skey] = submit_sel
            print(f"[AI_NATIVE_SUBMIT] op={operation_type} sel={submit_sel[:60]}", flush=True)
        except Exception as se:
            print(f"[AI_NATIVE_SUBMIT_ERROR] op={operation_type} sel={submit_sel[:60]} err={se}", flush=True)

    if filled_count == 0 and not submit_sel:
        return {"success": False, "error": "AI found no matching selectors"}

    # 発見済みセレクターをメモリ内 + Firestoreにキャッシュ
    if discovered:
        dom = media_mapping.setdefault("dom_selectors", {})
        for role, sel in discovered.items():
            if role not in dom:
                dom[role] = sel

        op_mappings = media_mapping.setdefault("operation_mappings", {})
        op_entry = op_mappings.setdefault(operation_type, {})
        op_selectors = op_entry.setdefault("selectors", {})
        for role, sel in discovered.items():
            op_selectors[role] = {"selector": sel, "source": "ai_native"}

        mapping_id = media_mapping.get("mapping_id", "")
        if db and mapping_id:
            try:
                db.collection("media_mappings").document(mapping_id).set(
                    {f"operation_mappings.{operation_type}.selectors": op_selectors},
                    merge=True,
                )
                print(f"[AI_NATIVE_CACHE] op={operation_type} cached={list(discovered.keys())} id={mapping_id}", flush=True)
            except Exception as ce:
                print(f"[AI_NATIVE_CACHE_ERROR] {ce}", flush=True)

    return {"success": True, "filled": filled_count, "discovered": list(discovered.keys())}


def _execute_operation(page, media_mapping: dict, operation_type: str, payload: dict, operation_steps: list = None,
                       prior_step_results: list = None, task_id: str = "", db=None) -> list:
    """
    ログイン済みpageに対してoperation_typeに応じた更新操作を実行する。
    operation_stepsがあればP14 multi-step runnerに委譲しstep_resultsを返す。
    なければGENERIC_OPERATION_CONFIGを参照してgeneric runnerで実行しNoneを返す。
    operation別if分岐なし。selector詳細・payload本文はログに出さない。
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    # P14: operation_stepsがある場合はmulti-step runnerに委譲
    if operation_steps:
        return _execute_operation_steps(page, media_mapping, operation_steps, payload,
                                        prior_step_results=prior_step_results, task_id=task_id, db=db)

    # P29: 実行前バリデーション（validation_rulesチェック）
    _vr = media_mapping.get("validation_rules") or {}
    if _vr and payload:
        _vr_errors = []
        _title_val  = payload.get("title", "") or ""
        _body_val   = payload.get("body", "") or payload.get("content", "") or ""
        _max_title  = _vr.get("max_title_length")
        _max_body   = _vr.get("max_body_length")
        _forbidden  = _vr.get("forbidden_words") or []
        _img_max_mb = _vr.get("image_max_size_mb")
        if _max_title and _title_val and len(_title_val) > _max_title:
            _vr_errors.append(f"タイトルが上限を超えています ({len(_title_val)}/{_max_title}文字)")
        if _max_body and _body_val and len(_body_val) > _max_body:
            _vr_errors.append(f"本文が上限を超えています ({len(_body_val)}/{_max_body}文字)")
        for _fw in _forbidden:
            if _fw and (_fw in _title_val or _fw in _body_val):
                _vr_errors.append(f"禁止ワード検出: '{_fw}'")
        if _img_max_mb:
            _img_path = payload.get("image") or payload.get("image_path") or ""
            if _img_path:
                try:
                    import os as _os_vr
                    _img_size_mb = _os_vr.path.getsize(str(_img_path)) / (1024 * 1024)
                    if _img_size_mb > _img_max_mb:
                        _vr_errors.append(f"画像サイズが上限を超えています ({_img_size_mb:.1f}MB/{_img_max_mb}MB)")
                except Exception as _vr_img_e:
                    print(f"[P29_VALIDATION_IMG_ERROR] {_vr_img_e}", flush=True)
        if _vr_errors:
            print(f"[P29_VALIDATION_FAILED] op={operation_type} errors={_vr_errors}", flush=True)
            raise RuntimeError(f"[P29_VALIDATION_FAILED] {'; '.join(_vr_errors)}")
        print(f"[P29_VALIDATION_OK] op={operation_type}", flush=True)

    config = GENERIC_OPERATION_CONFIG.get(operation_type)
    if config is None:
        raise RuntimeError(f"operation_type '{operation_type}' は未実装です")

    dom = media_mapping.get("dom_selectors", {})

    # セレクターが一切設定されていない場合はAI自動検出モードへ
    _has_any_selector = (
        any(dom.get(f["selector_key"]) for f in config.get("fields", []))
        or dom.get(config.get("submit_selector_key", "submit"))
    )
    if not _has_any_selector:
        print(f"[AI_NATIVE_MODE] op={operation_type} no selectors → AI field discovery", flush=True)
        _ai_r = _ai_native_fill(page, operation_type, payload, config, media_mapping, db=db)
        if _ai_r.get("success"):
            print(f"[AI_NATIVE_OK] op={operation_type} filled={_ai_r.get('filled')} discovered={_ai_r.get('discovered')}", flush=True)
            return []
        print(f"[AI_NATIVE_FAILED] op={operation_type} error={_ai_r.get('error')} — falling through", flush=True)

    submit_key = config.get("submit_selector_key", "submit")
    submit_sel = dom.get(submit_key)

    # --- 各フィールドへの入力 ---
    for field in config.get("fields", []):
        sel         = dom.get(field["selector_key"])
        value       = payload.get(field["payload_key"], "")
        input_type  = field.get("input_type", "text")

        if not sel and operation_type == "news_post" and field["selector_key"] == "title" and payload.get("title"):
            print(f"[TITLE_SELECTOR_OPTIONAL_MISSING] op=news_post reason=payload_title_present_but_selector_missing", flush=True)
        if not sel:
            continue

        page.wait_for_selector(sel, timeout=5000)

        if input_type == "file":
            page.set_input_files(sel, str(value))
        elif input_type == "select":
            page.select_option(sel, str(value))
        elif input_type == "checkbox":
            if value:
                page.check(sel)
            else:
                page.uncheck(sel)
        else:
            # text / その他
            page.fill(sel, str(value))

    # --- submit ---
    if submit_sel:
        page.wait_for_selector(submit_sel, timeout=5000)
        page.click(submit_sel)
        page.wait_for_load_state("domcontentloaded", timeout=35000)


    return []
def _verify_operation(page, media_mapping: dict) -> bool:
    """後方互換用。_verify_operation_detail のラッパー。"""
    result = _verify_operation_detail(page, media_mapping, before_hash="", after_html=None)
    return result.get("verified", False)


def _screen_candidate_profile(page, business_conditions: dict) -> dict:
    """
    Step 2: 候補者プロフィールページをGemini Visionでスクリーニング。
    Returns: {"pass": bool, "reason": str, "profile_data": dict, "images_checked": int}
    """
    import base64 as _b64
    import re as _re

    screening = business_conditions.get("screening") or {}
    if not screening:
        return {"pass": True, "reason": "条件未設定", "profile_data": {}, "images_checked": 0}

    # 候補者名をページから抽出（h1/h2/名前系要素）
    candidate_name = ""
    try:
        for _ns in ["h1", "h2", ".cast-name", ".girl-name", "[class*='name']", "title"]:
            _nel = page.locator(_ns).first
            if _nel.is_visible():
                _nt = (_nel.text_content() or "").strip()
                # 2〜15文字、英数字のみでない、サイトタイトルではない
                if 2 <= len(_nt) <= 15 and not _nt.isdigit():
                    candidate_name = _nt
                    break
    except Exception:
        pass

    # image_check=True（デフォルト）の時のみスクリーンショットを送信
    do_image = screening.get("image_check", True)
    screenshot_b64 = None
    if do_image:
        try:
            shot = page.screenshot(type="jpeg", quality=60, full_page=False)
            screenshot_b64 = _b64.b64encode(shot).decode()
        except Exception as _se:
            print(f"[SCREEN_PROF] screenshot error: {_se}", flush=True)

    try:
        page_text = (page.evaluate("() => document.body.innerText") or "")[:3000]
    except Exception:
        page_text = ""

    cond_parts: list = []
    if screening.get("height_min"):  cond_parts.append(f"身長{screening['height_min']}cm以上")
    if screening.get("height_max"):  cond_parts.append(f"身長{screening['height_max']}cm以下")
    if screening.get("weight_max"):  cond_parts.append(f"体重{screening['weight_max']}kg以下")
    if screening.get("cup_min"):     cond_parts.append(f"カップ{screening['cup_min']}以上")
    if screening.get("age_min"):     cond_parts.append(f"年齢{screening['age_min']}歳以上")
    if screening.get("age_max"):     cond_parts.append(f"年齢{screening['age_max']}歳以下")
    tattoo_ok = screening.get("tattoo_ok", False)
    cond_parts.append(f"タトゥー{'許可' if tattoo_ok else '不可'}")
    if screening.get("custom_conditions"):
        cond_parts.append(f"追加条件: {screening['custom_conditions']}")
    cond_text = "\n".join(f"・{c}" for c in cond_parts)

    system_p = (
        "あなたは求人サイト候補者スクリーニングAIです。"
        "プロフィールページ（テキスト＋スクリーンショット）を分析し採用条件に合致するか判定してください。"
        "必ずJSON形式のみで回答してください。他の文字は一切出力禁止。"
    )
    user_p = (
        f"【採用条件】\n{cond_text}\n\n"
        f"【プロフィールページテキスト】\n{page_text}\n\n"
        "以下のJSONのみで回答（理由は日本語1文）:\n"
        '{"pass":true,"extracted":{"age":null,"height":null,"weight":null,"cup":null,"tattoo":null,"notes":""},"reason":"理由"}'
    )

    try:
        from api.core.llm_client import call_llm as _call_llm_sp
        import json as _json_sp
        raw = _call_llm_sp(
            system_prompt=system_p,
            messages=[{"role": "user", "content": user_p}],
            ai_tier="core",
            temperature=0.1,
            image_b64=screenshot_b64,
            image_mime="image/jpeg",
        )
        m = _re.search(r'\{[\s\S]*\}', raw or "")
        if m:
            res = _json_sp.loads(m.group())
            extracted = res.get("extracted") or {}
            if candidate_name and not extracted.get("notes"):
                extracted["notes"] = candidate_name
            return {
                "pass": bool(res.get("pass", False)),
                "reason": str(res.get("reason", "")),
                "profile_data": extracted,
                "images_checked": 1 if screenshot_b64 else 0,
                "candidate_name": candidate_name,
            }
    except Exception as _le:
        print(f"[SCREEN_PROF] LLM error: {type(_le).__name__}: {_le}", flush=True)

    # LLM失敗 → 通過扱い（スクリーニング不能）
    return {"pass": True, "reason": "スクリーニングLLM失敗（通過扱い）", "profile_data": {}, "images_checked": 0, "candidate_name": candidate_name}


def _execute_offer_send(page, media_mapping: dict, payload: dict) -> dict:
    """
    offer_send: 候補者検索ページでフィルター適用→結果一覧→各候補者にオファー定型文を送信。
    Step 2: business_conditions.screeningで候補者を精査し、PASSのみ送信。
    payload keys:
      filter_fields: dict {selector: value}  絞り込み条件（省略時はデフォルト適用）
      body: str  ひな形文(オファー本文) ※business_conditions.offer_templateにfallback
      title: str (optional)
      max_send: int (default=5)  最大送信件数
    """
    import re as _re
    import time as _time

    # business_conditionsからひな形文fallbackとスクリーニング条件を取得
    bc = media_mapping.get("business_conditions") or {}
    _screening = bc.get("screening") or {}
    _do_screening = bool(_screening)  # image_checkはscreening内で画像送信をON/OFFするだけ
    bc_template = (bc.get("offer_template") or "").strip()
    body = payload.get("body", "").strip() or bc_template
    if not body:
        raise RuntimeError("offer_send: ひな形文(body)が空です。🏢業務条件パネルでオファーひな形文を設定してください")

    max_send = int(payload.get("max_send", 5))
    filter_fields = dict(payload.get("filter_fields") or {})

    # fi_* キー → filter_intent に変換（Create タブからのセマンティック絞り込み指定）
    filter_intent = dict(payload.get("filter_intent") or {})
    if payload.get("fi_scout_only"):
        filter_intent.setdefault("scout_only", True)
    if payload.get("fi_offer_unset_only"):
        filter_intent.setdefault("offer_unset_only", True)
    _fi_free = str(payload.get("fi_free_text") or "").strip()
    if _fi_free:
        filter_intent.setdefault("free_text", _fi_free)

    # mapping の offer_send フィールドをフィルタ条件に自動マッピング
    # ユーザーが name/id/canonical_tail キーで値を指定していれば selector に変換して適用
    _offer_op_fields = (media_mapping.get("operation_mappings") or {}).get("offer_send", {}).get("fields") or []
    for _omf in _offer_op_fields:
        _omf_sel = str(_omf.get("selector") or "").strip()
        if not _omf_sel:
            continue
        _omf_name = str(_omf.get("name") or _omf.get("id") or "").strip()
        _omf_tail = (_omf.get("canonical") or "").split(".")[-1]
        for _try_key in filter(None, [_omf_name, _omf_tail]):
            _val = payload.get(_try_key)
            if _val is not None and str(_val).strip() and _omf_sel not in filter_fields:
                filter_fields[_omf_sel] = str(_val).strip()
                break

    # 検索ページURL取得: manual_form_pages → operation_mappings → media_url
    search_url = payload.get("search_url", "")
    if not search_url:
        for pg in (media_mapping.get("manual_form_pages") or []):
            pg_op = str(pg.get("op_type") or "")
            pg_url = pg.get("url") or ""
            if pg_op == "offer_send" or "mypage_girl" in pg_url or "/girl" in pg_url:
                search_url = pg_url
                break
    if not search_url:
        op_map = (media_mapping.get("operation_mappings") or {}).get("offer_send", {})
        search_url = op_map.get("target_url", "")

    if not search_url:
        raise RuntimeError(
            "offer_send: 検索ページURLが見つかりません。"
            "手動ページ登録で候補者検索ページ（女の子を探す等）を登録してください"
        )

    # サイトルート
    login_url = media_mapping.get("login_url", "") or ""
    site_root = "/".join(login_url.split("/")[:3]) if login_url else ""

    print(f"[OFFER_SEND] navigate search_url={search_url}", flush=True)
    page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(1500)

    # filter_intent → 汎用キーワードマッチでフォームフィールドに自動マッピング
    # （fi_* → filter_intent 変換済みのため再代入しない）

    def _try_checkbox(candidates: list, want_on: bool):
        for sel in candidates:
            try:
                els = page.locator(sel).all()
                for loc in els:
                    if loc.is_visible():
                        checked = loc.is_checked()
                        if want_on and not checked:
                            loc.click()
                        elif not want_on and checked:
                            loc.click()
                        print(f"[OFFER_INTENT] checkbox {sel} -> {want_on}", flush=True)
                        return
            except Exception:
                pass

    def _try_select(candidates: list, value_keywords: list):
        for sel in candidates:
            try:
                loc = page.locator(sel).first
                if not loc.is_visible():
                    continue
                # 選択肢のtextで value_keywords を含むものを選ぶ
                opts = loc.locator("option").all()
                for opt in opts:
                    opt_text = (opt.text_content() or "").strip()
                    opt_val  = opt.get_attribute("value") or ""
                    if any(k in opt_text or k in opt_val for k in value_keywords):
                        loc.select_option(opt_val)
                        print(f"[OFFER_INTENT] select {sel} -> val={opt_val} ({opt_text})", flush=True)
                        return
            except Exception:
                pass

    def _try_fill(candidates: list, text: str):
        for sel in candidates:
            try:
                loc = page.locator(sel).first
                if loc.is_visible():
                    loc.fill(text)
                    print(f"[OFFER_INTENT] fill {sel} -> {text[:30]}", flush=True)
                    return
            except Exception:
                pass

    if filter_intent.get("scout_only"):
        _try_checkbox([
            'input[name*="scout_type"]', 'input[name*="scout"]',
            'input[name*="job_seeking"]', 'input[name*="kyushoku"]',
        ], True)

    if filter_intent.get("offer_unset_only"):
        _try_select([
            'select[name*="shop_message"]', 'select[name*="offer_status"]',
            'select[name*="message_flag"]', 'select[name*="offer"]',
        ], ["未送信", "未", "0", "none", "unset"])

    if filter_intent.get("free_text", "").strip():
        _try_fill([
            'input[name*="free_text"]', 'input[name*="keyword"]',
            'input[name*="search"]', 'input[placeholder*="フリー"]',
            'input[placeholder*="キーワード"]',
        ], filter_intent["free_text"].strip())

    # 直接 selector:value 指定がある場合はそちらも適用
    for selector, value in filter_fields.items():
        try:
            loc = page.locator(selector).first
            tag = loc.evaluate("el => el.tagName.toLowerCase()")
            typ = (loc.get_attribute("type") or "").lower()
            if tag == "select":
                loc.select_option(str(value))
            elif tag == "input" and typ == "checkbox":
                checked = loc.is_checked()
                want_on = str(value).lower() in ("1", "true", "on", "yes")
                if want_on and not checked:
                    loc.click()
                elif not want_on and checked:
                    loc.click()
            else:
                loc.fill(str(value))
            print(f"[OFFER_SEND_FILTER] {selector}={value}", flush=True)
        except Exception as _fe:
            print(f"[OFFER_SEND_FILTER] skip {selector}: {type(_fe).__name__}", flush=True)

    # 検索実行（保存ボタン以外のsubmit）
    search_btn = None
    for btn_sel in [
        'button:text-matches("絞り込", "i")',
        'button:text-matches("検索", "i")',
        'input[value*="絞り込"]',
        'input[value*="検索"]',
        'input[type="submit"]',
        'button[type="submit"]',
    ]:
        try:
            b = page.locator(btn_sel).first
            if b.is_visible() and b.is_enabled():
                txt = (b.get_attribute("onclick") or "") + (b.text_content() or "") + (b.get_attribute("value") or "")
                if "save" in txt.lower() or "保存" in txt:
                    continue
                search_btn = b
                print(f"[OFFER_SEND] search button found: {btn_sel}", flush=True)
                break
        except Exception:
            pass

    if search_btn:
        search_btn.click()
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1500)
    else:
        print(f"[OFFER_SEND] no search button; using page as-is", flush=True)

    results_url = page.url
    print(f"[OFFER_SEND] results_url={results_url}", flush=True)

    # 結果ページでオファー候補リンクを収集
    # 【重要】プロフィールURL（/girl/123/）を優先収集。なければオファーリンクにfallback。
    # プロフィールページでスクリーニング後にオファーボタンをクリックするのが正しいフロー。
    profile_urls: list = []
    offer_direct_urls: list = []
    seen: set = set()

    def _add_profile(href: str):
        full = href if href.startswith("http") else f"{site_root}{href}"
        if full not in seen:
            seen.add(full)
            profile_urls.append(full)

    def _add_offer(href: str):
        full = href if href.startswith("http") else f"{site_root}{href}"
        if full not in seen:
            seen.add(full)
            offer_direct_urls.append(full)

    # パターンA: プロフィールURL（末尾が数値IDで、/offer/を含まない）
    try:
        for el in page.locator("a[href]").all():
            h = el.get_attribute("href") or ""
            # /mypage_girl/12345/ や /girl/12345/ （/offer が続かないもの）
            if _re.search(r'/(?:mypage_girl|girl|scout)/(\d+)/?$', h):
                _add_profile(h)
    except Exception:
        pass

    # パターンB: オファー直接リンク（fallback。プロフィールURLが見つからない時のみ使用）
    for sel in ['a[href*="offer"]', 'a:text-matches("オファー", "i")']:
        try:
            for el in page.locator(sel).all():
                h = el.get_attribute("href") or ""
                if h:
                    _add_offer(h)
        except Exception:
            pass
    # /offer/123/ パターン
    try:
        for el in page.locator("a[href]").all():
            h = el.get_attribute("href") or ""
            if _re.search(r'/offer/\d+', h):
                _add_offer(h)
    except Exception:
        pass

    # プロフィールURLを優先。なければオファーURL。
    _using_profile_urls = bool(profile_urls)
    target_urls = profile_urls if _using_profile_urls else offer_direct_urls
    print(f"[OFFER_SEND] profile_urls={len(profile_urls)} offer_urls={len(offer_direct_urls)} → using={'profile' if _using_profile_urls else 'offer'} max_send={max_send}", flush=True)

    if not target_urls:
        return {
            "status": "NO_TARGETS",
            "sent_count": 0,
            "candidates_found": 0,
            "results_url": results_url,
            "message": "結果ページにオファー候補リンクが見つかりませんでした。検索条件・ページ登録を確認してください。",
        }

    screened_out: list = []
    conversations: list = []
    sent = []
    failed = []

    for i, turl in enumerate(target_urls[:max_send]):
        print(f"[OFFER_SEND] [{i+1}/{min(len(target_urls),max_send)}] {turl}", flush=True)
        try:
            page.goto(turl, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1200)

            # Step 2: スクリーニング（プロフィールURLの場合のみ実行）
            # オファーフォームページではプロフィール情報がないためスクリーニング不可
            _screen_r: dict = {"pass": True, "reason": "条件未設定", "profile_data": {}, "images_checked": 0, "candidate_name": ""}
            if _do_screening and _using_profile_urls:
                _screen_r = _screen_candidate_profile(page, bc)
                print(f"[OFFER_SEND] screen: pass={_screen_r['pass']} reason={_screen_r['reason']}", flush=True)
            elif _do_screening and not _using_profile_urls:
                print(f"[OFFER_SEND] screening skipped (offer URL, not profile URL)", flush=True)
            if not _screen_r["pass"]:
                screened_out.append({"url": turl, "reason": _screen_r["reason"], "profile_data": _screen_r["profile_data"]})
                _time.sleep(0.5)
                continue

            offer_ta = None
            offer_submit = None

            # テキストエリア探索
            for ta_sel in ["textarea", 'textarea[name*="message"]', 'textarea[name*="offer"]', 'textarea[name*="body"]']:
                try:
                    loc = page.locator(ta_sel).first
                    if loc.is_visible():
                        offer_ta = loc
                        break
                except Exception:
                    pass

            # テキストエリアが無い → オファーボタンをクリックして別フォームへ
            if offer_ta is None:
                for btn_sel in ['a:text-matches("オファー", "i")', 'button:text-matches("オファー", "i")', 'a[href*="offer"]']:
                    try:
                        b = page.locator(btn_sel).first
                        if b.is_visible():
                            b.click()
                            try:
                                page.wait_for_load_state("networkidle", timeout=10000)
                            except Exception:
                                pass
                            page.wait_for_timeout(1000)
                            for ta_sel in ["textarea", 'textarea[name*="message"]']:
                                try:
                                    loc = page.locator(ta_sel).first
                                    if loc.is_visible():
                                        offer_ta = loc
                                except Exception:
                                    pass
                            break
                    except Exception:
                        pass

            # 送信ボタン探索
            for btn_sel in ['button[type="submit"]', 'input[type="submit"]', 'button:text-matches("送信|送る|オファー", "i")']:
                try:
                    b = page.locator(btn_sel).first
                    if b.is_visible():
                        offer_submit = b
                        break
                except Exception:
                    pass

            if offer_ta and offer_submit:
                offer_ta.fill(body)
                page.wait_for_timeout(400)
                offer_submit.click()
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                page.wait_for_timeout(800)
                # オファー送信後のページURL = 会話スレッドURLになることが多い
                _reply_url = page.url
                _reply_url = _reply_url if _reply_url != turl else ""
                print(f"[OFFER_SEND] ✓ sent to {turl} reply_url={_reply_url}", flush=True)
                sent.append({"url": turl, "status": "sent", "reply_url": _reply_url})
                # Step 3: 会話スレッド記録
                import hashlib as _hl
                _cid = (
                    f"{(media_mapping.get('tenant_id') or 'unknown')}_"
                    f"{(media_mapping.get('mapping_id') or media_mapping.get('id') or '')}_"
                    f"{_hl.md5(turl.encode()).hexdigest()[:12]}"
                )
                # 候補者名: スクリーニング結果 → profile_data.notes → URL末尾ID
                _cand_name = (
                    _screen_r.get("candidate_name")
                    or (_screen_r.get("profile_data") or {}).get("notes", "")
                    or "候補者"
                )
                conversations.append({
                    "id": _cid,
                    "tenant_id": media_mapping.get("tenant_id") or "",
                    "mapping_id": media_mapping.get("mapping_id") or media_mapping.get("id") or "",
                    "candidate_url": turl,
                    "reply_url": _reply_url,
                    "candidate_name": _cand_name,
                    "phase": "offer_sent",
                    "messages": [{"role": "shop", "content": body}],
                    "profile_data": _screen_r.get("profile_data") or {},
                    "screening_pass": _screen_r.get("pass", True),
                    "screening_reason": _screen_r.get("reason", ""),
                })
            else:
                reason = f"form not found (textarea={'✓' if offer_ta else '✗'} submit={'✓' if offer_submit else '✗'})"
                print(f"[OFFER_SEND] ✗ {reason} url={turl}", flush=True)
                failed.append({"url": turl, "reason": reason})
        except Exception as _e:
            print(f"[OFFER_SEND] error {type(_e).__name__}: {_e} url={turl}", flush=True)
            failed.append({"url": turl, "reason": f"{type(_e).__name__}: {_e}"})

        _time.sleep(1)

    return {
        "status": "DONE" if sent else ("PARTIAL" if (failed or screened_out) else "NO_TARGETS"),
        "sent_count": len(sent),
        "failed_count": len(failed),
        "screened_out_count": len(screened_out),
        "candidates_found": len(target_urls),
        "results_url": results_url,
        "sent": sent,
        "failed": failed,
        "screened_out": screened_out,
        "conversations": conversations,
    }


def _execute_recruit_reply(page, media_mapping: dict, payload: dict) -> dict:
    """
    recruit_reply: 候補者との会話スレッドに返信を送信する（Step 3 送信フェーズ）。
    payload keys:
      body: str       返信本文 ※必須
      reply_url: str  会話スレッドURL ※必須
    """
    import time as _rr_time

    body = (payload.get("body") or "").strip()
    if not body:
        raise RuntimeError("recruit_reply: 返信本文(body)が空です")

    reply_url = (payload.get("reply_url") or "").strip()
    if not reply_url:
        raise RuntimeError("recruit_reply: reply_urlが設定されていません。会話スレッドURLを指定してください")

    print(f"[RECRUIT_REPLY] navigate to {reply_url}", flush=True)
    page.goto(reply_url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(1500)

    # テキストエリア探索
    reply_ta = None
    for ta_sel in ["textarea", 'textarea[name*="message"]', 'textarea[name*="body"]', 'textarea[name*="reply"]', 'textarea[name*="comment"]']:
        try:
            loc = page.locator(ta_sel).first
            if loc.is_visible():
                reply_ta = loc
                break
        except Exception:
            pass

    # 送信ボタン探索
    reply_submit = None
    for btn_sel in ['button[type="submit"]', 'input[type="submit"]', 'button:text-matches("送信|返信|送る", "i")']:
        try:
            b = page.locator(btn_sel).first
            if b.is_visible():
                reply_submit = b
                break
        except Exception:
            pass

    if not reply_ta:
        raise RuntimeError(f"recruit_reply: テキストエリアが見つかりません (url={reply_url})")
    if not reply_submit:
        raise RuntimeError(f"recruit_reply: 送信ボタンが見つかりません (url={reply_url})")

    reply_ta.fill(body)
    page.wait_for_timeout(400)
    reply_submit.click()
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(800)
    print(f"[RECRUIT_REPLY] ✓ reply sent", flush=True)
    return {"status": "DONE", "reply_url": reply_url, "body_length": len(body)}


def _execute_recruit_inbox_scan(page, media_mapping: dict, payload: dict, db=None) -> dict:
    """
    recruit_inbox_scan: 受信ボックスをスキャンして候補者からの返信を検知する（Step 3 監視フェーズ）。
    - Geminiでメッセージ一覧を抽出
    - URLのIDで recruit_conversations と照合
    - 新着メッセージをスレッドに追記、phase→replied
    """
    import re as _re_inbox
    import base64 as _b64_inbox
    from api.core.llm_client import call_llm_json as _cljson

    # inbox URL解決
    inbox_url = (payload.get("inbox_url") or "").strip()
    if not inbox_url:
        for pg in (media_mapping.get("manual_form_pages") or []):
            _pg_op = str(pg.get("op_type") or "")
            if _pg_op in ("recruit_inbox_scan", "inbox", "message", "messages"):
                inbox_url = pg.get("url") or ""
                break
    if not inbox_url:
        op_map = (media_mapping.get("operation_mappings") or {}).get("recruit_inbox_scan", {})
        inbox_url = op_map.get("target_url") or ""
    if not inbox_url:
        raise RuntimeError(
            "recruit_inbox_scan: 受信ボックスURLが設定されていません。"
            "手動ページ登録で受信ボックスページ（op_type=recruit_inbox_scan）を登録してください"
        )

    login_url = media_mapping.get("login_url") or ""
    site_root = "/".join(login_url.split("/")[:3]) if login_url else ""
    tenant_id = media_mapping.get("tenant_id") or ""
    mapping_id = media_mapping.get("mapping_id") or media_mapping.get("id") or ""

    print(f"[RECRUIT_INBOX] navigate to {inbox_url}", flush=True)
    page.goto(inbox_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(2000)

    screenshot_b64 = ""
    html = ""
    try:
        screenshot_b64 = _b64_inbox.b64encode(page.screenshot(type="jpeg", quality=55, full_page=False)).decode()
    except Exception:
        pass
    try:
        html = page.content()
    except Exception:
        pass

    # Geminiでメッセージ一覧抽出
    inbox_messages: list = []
    if html:
        _inbox_prompt = (
            "以下は求人サイトの受信ボックス（メッセージ一覧）ページのHTMLです。\n"
            "候補者からの受信メッセージ一覧を全件抽出してください。\n\n"
            "出力JSON:\n"
            '{"messages":[\n'
            '  {"sender_name":"候補者名","thread_url":"/messages/123/","preview":"メッセージ冒頭","is_unread":true,"date":"2025-06-20"}\n'
            "]}\n\n"
            "【ルール】\n"
            "- thread_urlは相対URLのまま（/で始まる形式）\n"
            "- 最大50件を返す\n"
            "- is_unreadはページ表示から判断（不明ならtrue）\n\n"
            f"HTML（先頭12000文字）:\n{html[:12000]}"
        )
        try:
            _ib_data = _cljson(
                system_prompt="求人サイト受信ボックスHTMLからメッセージ一覧を抽出するAI。JSONのみ出力。",
                user_prompt=_inbox_prompt,
                max_tokens=4000,
                temperature=0.05,
            ) or {}
            inbox_messages = _ib_data.get("messages") or []
        except Exception as _ije:
            print(f"[RECRUIT_INBOX] LLM error: {_ije}", flush=True)

    print(f"[RECRUIT_INBOX] inbox messages found={len(inbox_messages)}", flush=True)

    # Firestoreから既存の会話スレッドを取得
    existing_convs: list = []
    if db and tenant_id:
        try:
            _q = db.collection("recruit_conversations").where("tenant_id", "==", tenant_id)
            if mapping_id:
                _q = _q.where("mapping_id", "==", mapping_id)
            for _d in _q.limit(500).stream():
                _item = _d.to_dict() or {}
                _item["_conv_doc_id"] = _d.id
                existing_convs.append(_item)
        except Exception as _fce:
            print(f"[RECRUIT_INBOX] conv load error: {_fce}", flush=True)

    import datetime as _dt_ib
    updated_conversations: list = []
    unmatched: list = []

    for msg in inbox_messages:
        thread_url = (msg.get("thread_url") or "").strip()
        sender_name = (msg.get("sender_name") or "").strip()
        preview = (msg.get("preview") or "").strip()

        if not thread_url:
            continue

        full_thread_url = thread_url if thread_url.startswith("http") else f"{site_root}{thread_url}"

        # URLからIDを抽出して照合（thread_url=/messages/12345/ → IDs={12345}）
        _thread_ids = set(_re_inbox.findall(r'/(\d{3,})', thread_url))

        matched_conv = None
        for conv in existing_convs:
            _cand_url = conv.get("candidate_url") or ""
            _reply_url_c = conv.get("reply_url") or ""
            _conv_ids = set(
                _re_inbox.findall(r'/(\d{3,})', _cand_url)
                + _re_inbox.findall(r'/(\d{3,})', _reply_url_c)
            )
            if _thread_ids & _conv_ids:
                matched_conv = conv
                break
            # 名前マッチ（フォールバック）
            if sender_name and sender_name == (conv.get("candidate_name") or ""):
                matched_conv = conv
                break

        if not matched_conv:
            unmatched.append({"sender_name": sender_name, "thread_url": full_thread_url, "preview": preview})
            continue

        # スレッドページを開いて最新メッセージ内容を取得
        thread_content = preview
        try:
            page.goto(full_thread_url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1500)
            _thread_text = (page.evaluate("() => document.body.innerText") or "")
            if _thread_text:
                thread_content = _thread_text[:500]
        except Exception as _te:
            print(f"[RECRUIT_INBOX] thread page error: {_te}", flush=True)

        _now_ib = _dt_ib.datetime.utcnow()
        _conv_doc_id = matched_conv.get("_conv_doc_id")
        _cur_msgs = list(matched_conv.get("messages") or [])
        _cur_msgs.append({"role": "candidate", "content": preview or thread_content[:200], "sent_at": _now_ib})

        if db and _conv_doc_id:
            try:
                db.collection("recruit_conversations").document(_conv_doc_id).update({
                    "messages": _cur_msgs,
                    "phase": "replied",
                    "updated_at": _now_ib,
                    "last_candidate_message": preview or thread_content[:200],
                    "reply_url": full_thread_url,  # reply_urlをスレッドURLで更新
                })
                updated_conversations.append({
                    "conversation_id": _conv_doc_id,
                    "candidate_name": matched_conv.get("candidate_name"),
                    "thread_url": full_thread_url,
                    "preview": preview,
                })
                print(f"[RECRUIT_INBOX] ✓ updated conv {_conv_doc_id} ({matched_conv.get('candidate_name')})", flush=True)
            except Exception as _ue:
                print(f"[RECRUIT_INBOX] update error: {_ue}", flush=True)

    return {
        "status": "DONE",
        "inbox_url": inbox_url,
        "messages_found": len(inbox_messages),
        "updated_count": len(updated_conversations),
        "updated_conversations": updated_conversations,
        "unmatched_count": len(unmatched),
        "unmatched": unmatched[:10],
        "screenshot_b64": screenshot_b64,
    }


def _execute_page_monitor(page, media_mapping: dict, payload: dict = None) -> dict:
    """
    page_monitor: ログイン済みpageで一覧ページを開き、投稿データを抽出・分析して返す。
    フォーム送信なし・読み取り専用。
    分析内容:
      1. 投稿一覧（投稿日時・投稿者・本文抜粋）
      2. 投稿者別集計（誰が何件、最終投稿日）
      3. キャラ・売りとの一致評価（Gemini判定）
    """
    import base64
    from api.core.llm_client import call_llm_json

    payload = payload or {}
    check_points = payload.get("check_points") or ""

    # target_url解決: payload.monitor_url → operation_mappings → manual_form_pages の順で探す
    target_url = payload.get("monitor_url") or ""
    if not target_url:
        op_map = (media_mapping.get("operation_mappings") or {}).get("page_monitor", {})
        target_url = op_map.get("target_url") or ""
    if not target_url:
        for _p in (media_mapping.get("manual_form_pages") or []):
            if _p.get("op_type") == "page_monitor" and _p.get("url"):
                target_url = _p["url"]
                break
    if not target_url:
        raise RuntimeError("page_monitor: target_urlが設定されていません（監視ページを登録してください）")

    print(f"[PAGE_MONITOR] navigating to {target_url[:80]}", flush=True)
    page.goto(target_url, timeout=35000, wait_until="domcontentloaded")
    page.wait_for_timeout(2500)

    # スクリーンショット
    screenshot_b64 = ""
    try:
        screenshot_b64 = base64.b64encode(
            page.screenshot(type="jpeg", quality=55, full_page=False)
        ).decode()
    except Exception as _ss_e:
        print(f"[PAGE_MONITOR_SS_ERROR] {type(_ss_e).__name__}", flush=True)

    html = ""
    try:
        html = page.content()
    except Exception:
        pass

    # ── STEP1: 投稿一覧をGeminiで抽出 ──────────────────────────────────────
    items_data: dict = {}
    if html:
        try:
            _p1 = (
                "以下は管理画面の投稿・日記一覧ページのHTMLです。\n"
                "テーブルやリストに含まれる全投稿データを抽出してください。\n\n"
                "出力JSON（fieldsを先頭に置くこと）:\n"
                "{\n"
                '  "items": [\n'
                '    {\n'
                '      "投稿日時": "2025-06-19 14:30",\n'
                '      "投稿者": "れい",\n'
                '      "本文抜粋": "今日は...",\n'
                '      "画像": "あり or なし"\n'
                '    }\n'
                "  ],\n"
                '  "total_count": 10,\n'
                '  "page_title": "写メ日記一覧"\n'
                "}\n\n"
                "【ルール】\n"
                "- ナビ・メニューは除外、メインテーブルのみ\n"
                "- 投稿日時・投稿者（キャスト名）・本文の冒頭50文字・画像有無を抽出\n"
                "- 全件（最大30件）を返す\n\n"
                f"HTML（先頭12000文字）:\n{html[:12000]}"
            )
            items_data = call_llm_json(
                system_prompt="管理画面HTMLから投稿一覧データをJSON形式で抽出するアシスタント。",
                user_prompt=_p1,
                max_tokens=6000,
                temperature=0.05,
            ) or {}
            print(f"[PAGE_MONITOR_STEP1] items={len(items_data.get('items') or [])}", flush=True)
        except Exception as _e1:
            print(f"[PAGE_MONITOR_STEP1_ERROR] {type(_e1).__name__}", flush=True)
            items_data = {}

    # html.parserフォールバック
    if not items_data.get("items") and html:
        try:
            from html.parser import HTMLParser as _HP2

            class _TP2(_HP2):
                def __init__(self):
                    super().__init__()
                    self._in_t = 0; self._in_r = False; self._in_c = False
                    self._cell = ""; self._row: list = []
                    self.headers: list = []; self.rows: list = []; self._hd = False

                def handle_starttag(self, tag, attrs):
                    if tag == "table": self._in_t += 1
                    elif tag == "tr" and self._in_t: self._in_r = True; self._row = []
                    elif tag in ("td", "th") and self._in_t: self._in_c = True; self._cell = ""

                def handle_endtag(self, tag):
                    if tag == "table": self._in_t -= 1
                    elif tag == "tr" and self._in_r:
                        self._in_r = False
                        if self._row:
                            if not self._hd: self.headers = self._row[:]; self._hd = True
                            else: self.rows.append(self._row[:])
                    elif tag in ("td", "th") and self._in_c:
                        self._in_c = False; self._row.append(self._cell.strip())

                def handle_data(self, data):
                    if self._in_c: self._cell += data

            _tp2 = _TP2()
            _tp2.feed(html[:40000])
            if _tp2.headers and _tp2.rows:
                _its = []
                for _r in _tp2.rows[:30]:
                    _o = {(_tp2.headers[i] if i < len(_tp2.headers) else f"col{i}"): (_r[i] if i < len(_r) else "") for i in range(len(_tp2.headers))}
                    _its.append(_o)
                items_data = {"items": _its, "total_count": len(_tp2.rows), "page_title": ""}
                print(f"[PAGE_MONITOR_FB] rows={len(_tp2.rows)}", flush=True)
        except Exception as _fb_e:
            print(f"[PAGE_MONITOR_FB_ERROR] {type(_fb_e).__name__}", flush=True)

    # ── STEP2: 投稿者別集計 + キャラ一致評価 (Gemini) ──────────────────────
    analysis: dict = {}
    _items = items_data.get("items") or []
    if _items:
        try:
            import json as _json
            _check = f"\n確認ポイント: {check_points}" if check_points else ""
            _p2 = (
                "以下は管理画面から抽出した投稿一覧データです。\n"
                "このデータを分析して経営者向けレポートをJSON形式で返してください。" + _check + "\n\n"
                "出力JSON:\n"
                "{\n"
                '  "cast_summary": [\n'
                '    {\n'
                '      "名前": "れい",\n'
                '      "投稿数": 5,\n'
                '      "最終投稿": "2025-06-19",\n'
                '      "評価": "積極的 / 普通 / 少ない / 未投稿",\n'
                '      "コメント": "週1以上投稿しており良好。内容は体験談が多く共感を呼びやすい。"\n'
                '    }\n'
                "  ],\n"
                '  "alert_casts": ["さくら", "みお"],\n'
                '  "top_poster": "れい",\n'
                '  "overall_comment": "全体として投稿頻度は高い。ただしさくら・みおが7日以上未投稿のため要確認。"\n'
                "}\n\n"
                "【分析基準】\n"
                "- 投稿数0: 未投稿（要対応）\n"
                "- 最終投稿が7日以上前: alert_castsに追加\n"
                "- 投稿内容がキャラ・売りに合っているか判定（本文抜粋から推測）\n"
                "- overall_commentで経営者視点のサマリーを100文字以内で\n\n"
                f"投稿データ:\n{_json.dumps(_items[:30], ensure_ascii=False)}"
            )
            analysis = call_llm_json(
                system_prompt="投稿データを分析して経営者向けレポートをJSON形式で返すアシスタント。",
                user_prompt=_p2,
                max_tokens=4096,
                temperature=0.2,
            ) or {}
            print(f"[PAGE_MONITOR_STEP2] alert_casts={analysis.get('alert_casts')} top={analysis.get('top_poster')}", flush=True)
        except Exception as _e2:
            print(f"[PAGE_MONITOR_STEP2_ERROR] {type(_e2).__name__}", flush=True)

    monitor_data = {
        "items": _items,
        "total_count": items_data.get("total_count") or len(_items),
        "page_title": items_data.get("page_title") or "",
        "analysis": analysis,
        "summary": analysis.get("overall_comment") or f"投稿{len(_items)}件取得",
    }

    return {
        "monitor_data": monitor_data,
        "screenshot_b64": screenshot_b64,
        "target_url": target_url,
        "html_length": len(html),
    }


def _verify_operation_detail(
    page,
    media_mapping: dict,
    before_hash: str,
    after_html=None,
) -> dict:
    """
    P6: 実行後検証強化。
    verify_selector確認 + before/after差分判定 + 検証メソッド記録。
    戻り値: { verified, method, before_hash, after_hash, diff_detected }
    """
    import hashlib
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    verify_sel = media_mapping.get("verify_selector")
    login_url  = media_mapping.get("login_url", "")

    try:
        current_html = after_html if after_html is not None else page.content()
    except Exception:
        current_html = ""

    after_hash = hashlib.md5(current_html.encode("utf-8", errors="replace")).hexdigest() if current_html else ""
    diff_detected = (before_hash != after_hash) if before_hash else False

    if verify_sel:
        try:
            page.wait_for_selector(verify_sel, timeout=3000)
            return {"verified": True,  "method": "selector",        "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}
        except PlaywrightTimeout:
            return {"verified": False, "method": "selector",        "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}

    # verify_sel未設定の場合→完了テキストをDOMから自動検索
    _VERIFY_SUCCESS_TEXTS = [
        "保存しました", "投稿しました", "登録しました", "更新しました", "完了しました",
        "送信しました", "反映しました", "完了", "success", "saved", "posted", "registered", "updated", "submitted",
    ]
    # [VERIFY_NEGATIVE] error keywords - if found, do not verify as success
    _VERIFY_ERROR_TEXTS = [
        "エラー", "失敗", "できませんでした", "ができません",
        "error", "failed", "failure", "invalid", "unauthorized", "forbidden",
        "exception", "traceback", "500", "403", "404",
    ]
    try:
        _page_text = page.inner_text("body") if page else ""
        # negative check first
        _page_text_lower = _page_text.lower()
        for _et in _VERIFY_ERROR_TEXTS:
            if _et.lower() in _page_text_lower:
                print(f"[VERIFY_ERROR_TEXT] matched={_et} url={page.url if page else ''}", flush=True)
                return {"verified": False, "method": "error_text_detected", "matched_error": _et, "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}
        for _st in _VERIFY_SUCCESS_TEXTS:
            if _st.lower() in _page_text_lower:
                print(f"[VERIFY_SUCCESS_TEXT] matched={_st} url={page.url if page else ''}", flush=True)
                return {"verified": True, "method": "success_text", "matched_text": _st, "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}
    except Exception as _vt_e:
        print(f"[VERIFY_SUCCESS_TEXT_ERROR] {_vt_e}", flush=True)
    # submitボタンdisabledチェック
    dom = media_mapping.get("dom_selectors", {})
    submit_sel = dom.get("submit")
    if submit_sel:
        try:
            btn = page.query_selector(submit_sel)
            if btn and btn.get_attribute("disabled") is not None:
                return {"verified": True, "method": "submit_disabled", "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}
        except Exception:
            pass
    # verify_sel未設定かつ完了テキスト未検出→verified=False
    print(f"[VERIFY_FAILED] verify_sel未設定かつ完了テキスト未検出 url={page.url if page else ''}", flush=True)
    return {"verified": False, "method": "none", "before_hash": before_hash, "after_hash": after_hash, "diff_detected": diff_detected}


def _attempt_self_heal_execution(
    page,
    media_mapping: dict,
    operation_type: str,
    payload: dict,
    before_values: dict,
    before_hash: str,
    operation_steps: list = None,
    prior_step_results: list = None,
) -> dict:
    """
    P13: self-heal executor。
    通常execute失敗後、DOM_SCAN → temporary_selectors merge → 再execute(1回のみ)。
    media_mappings本体は変更しない。
    """
    import hashlib
    self_heal_result = {
        "attempted": True,
        "success": False,
        "retry_succeeded": False,
        "temporary_selectors": {},
        "failed_selectors": [],
        "suggested_selectors": [],
        "verification": None,
        "resubmit_blocked": False,
        "terminal_step_done": False,
    }
    try:
        def _retry_steps_after_failure() -> list | None:
            if not operation_steps:
                return operation_steps
            sorted_steps = sorted(operation_steps, key=lambda s: s.get("order", 99))
            results = prior_step_results or []
            done_ids = {str(r.get("step_id") or "") for r in results if r.get("status") == "DONE"}
            failed_ids = [str(r.get("step_id") or "") for r in results if r.get("status") == "FAILED"]
            terminal_done_ids = {
                str(s.get("step_id") or "")
                for s in sorted_steps
                if _is_terminal_operation_step(s) and str(s.get("step_id") or "") in done_ids
            }
            if terminal_done_ids:
                terminal_orders = [
                    int(s.get("order", 99))
                    for s in sorted_steps
                    if str(s.get("step_id") or "") in terminal_done_ids
                ]
                boundary = max(terminal_orders) if terminal_orders else 99
                self_heal_result["terminal_step_done"] = True
                self_heal_result["resubmit_blocked"] = True
                retry = [
                    s for s in sorted_steps
                    if int(s.get("order", 99)) > boundary and not _is_terminal_operation_step(s)
                ]
                return retry
            if failed_ids:
                failed_orders = [
                    int(s.get("order", 99))
                    for s in sorted_steps
                    if str(s.get("step_id") or "") in failed_ids
                ]
                boundary = min(failed_orders) if failed_orders else 0
                return [s for s in sorted_steps if int(s.get("order", 99)) >= boundary]
            return operation_steps

        _scan = run_dom_scan(media_mapping)
        _candidates = _scan.get("selectors", [])
        _suggested = [c for c in _candidates if c.get("suggested_selector")]
        self_heal_result["suggested_selectors"] = _suggested
        self_heal_result["failed_selectors"] = list(before_values.keys())

        if not _suggested:
            self_heal_result["success"] = False
            return self_heal_result

        # 一時コピーにmerge（本体変更禁止）
        import copy
        temp_mapping = copy.deepcopy(media_mapping)
        temp_dom = temp_mapping.get("dom_selectors", {})
        temporary_selectors = {}
        for cand in _suggested:
            key = cand.get("selector_key")
            sel = cand.get("suggested_selector")
            if key and sel:
                temp_dom[key] = sel
                temporary_selectors[key] = sel
        # P16-5: selector_rankings high scoreをtemporary候補に最優先追加（保存禁止）
        _rankings = media_mapping.get("selector_rankings", {})
        _ranked_list = _rankings.get("ranked_selectors", [])
        for _r in _ranked_list:
            if _r.get("score", 0) >= 0.75 and _r.get("selector"):
                _rkey = _r.get("label") or ""
                if _rkey and _rkey not in temp_dom:
                    temp_dom[_rkey] = _r["selector"]
                    temporary_selectors[_rkey] = _r["selector"]
        # P15-6: semantic high confidence selectorをtemporary候補に優先追加
        _sem = _scan.get("semantic_selector_candidates", {})
        _sem_labels = _sem.get("labels", {})
        _sem_conf = _sem.get("confidence", {})
        for label_key, sel in _sem_labels.items():
            if _sem_conf.get(label_key) == "high" and sel:
                temp_dom[label_key] = sel
                temporary_selectors[label_key] = sel
        temp_mapping["dom_selectors"] = temp_dom
        self_heal_result["temporary_selectors"] = temporary_selectors

        # 再execute（1回のみ）。submit済みなら再submitせずverify以降だけ確認する。
        if not operation_steps:
            self_heal_result["resubmit_blocked"] = True
            self_heal_result["message"] = "operation_stepsが無いため自動再実行を禁止しました"
            return self_heal_result
        retry_steps = _retry_steps_after_failure()
        if operation_steps and not retry_steps:
            self_heal_result["success"] = False
            self_heal_result["retry_succeeded"] = False
            self_heal_result["message"] = "submit済みのためself-heal再送信を禁止しました"
            return self_heal_result
        try:
            _execute_operation(page, temp_mapping, operation_type, payload, operation_steps=retry_steps)
        except RuntimeError as re:
            self_heal_result["success"] = False
            self_heal_result["retry_succeeded"] = False
            return self_heal_result

        # verification
        try:
            after_html = page.content()
        except Exception:
            after_html = ""
        verification = _verify_operation_detail(page, temp_mapping, before_hash, after_html)
        self_heal_result["verification"] = verification

        if verification.get("verified"):
            self_heal_result["success"] = True
            self_heal_result["retry_succeeded"] = True
            # P16-7: selector_rank使用ログ付与
            _rank_used = bool(temporary_selectors)
            _rank_score = None
            _rank_source = ""
            _rankings = media_mapping.get("selector_rankings", {})
            _ranked_list = _rankings.get("ranked_selectors", [])
            for _r in _ranked_list:
                if _r.get("selector") in temporary_selectors.values():
                    _rank_score = _r.get("score")
                    _rank_source = _r.get("source", "")
                    break
            self_heal_result["selector_rank_used"] = _rank_used
            self_heal_result["selector_rank_score"] = _rank_score
            self_heal_result["selector_rank_source"] = _rank_source
        else:
            self_heal_result["success"] = False
            self_heal_result["retry_succeeded"] = False

    except Exception as e:
        self_heal_result["success"] = False
        print(f"[self_heal] unexpected error: {type(e).__name__}", flush=True)

    return self_heal_result

    return {"verified": False, "method": "none", "before_hash": before_hash, "after_hash": after_hash, "diff_detected": False}

def _run_login_form_with_operation(
    media_mapping: dict,
    creds: dict,
    operation_type: str,
    payload: dict,
    operation_steps: list = None,
    prior_step_results: list = None,
    task_id: str = "",
    db=None,
) -> dict:
    """
    ログイン後に実際の操作を実行するフル実行モード。
    run_login_form_check と分離。
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        with sync_playwright() as p:
            browser = None
            try:
                _auth = create_authenticated_page(p, media_mapping, creds)
                browser, page = _auth["browser"], _auth["page"]
                print(f"[browser_executor] login OK, executing: {operation_type}", flush=True)

                # P6: before_html取得
                import hashlib
                try:
                    before_html = page.content()
                    before_hash = hashlib.md5(before_html.encode("utf-8", errors="replace")).hexdigest()
                except Exception:
                    before_html = ""
                    before_hash = ""

                # offer_send: 候補者検索→スクリーニング→オファー一括送信
                if operation_type == "offer_send":
                    try:
                        _offer_r = _execute_offer_send(page, media_mapping, payload)
                    except RuntimeError as _offer_e:
                        return {
                            "status": "FAILED", "executed": False,
                            "login_success": True, "operation_type": "offer_send",
                            "error": str(_offer_e),
                            "message": str(_offer_e),
                        }
                    # Step 3: 送信成功分の会話スレッドをFirestoreに保存
                    if _offer_r.get("conversations"):
                        try:
                            from api.core.firestore_client import get_db as _get_db_conv
                            import datetime as _dt_conv
                            _conv_db = _get_db_conv()
                            _conv_now = _dt_conv.datetime.utcnow()
                            for _conv in _offer_r["conversations"]:
                                _cid = _conv.get("id")
                                if not _cid:
                                    continue
                                _msgs = _conv.get("messages") or []
                                for _m in _msgs:
                                    if "sent_at" not in _m:
                                        _m["sent_at"] = _conv_now
                                _conv_db.collection("recruit_conversations").document(_cid).set(
                                    {**_conv, "created_at": _conv_now, "updated_at": _conv_now, "offer_sent_at": _conv_now},
                                    merge=True,
                                )
                            print(f"[OFFER_SEND] {len(_offer_r['conversations'])}件の会話スレッドをFirestoreに保存", flush=True)
                        except Exception as _conv_e:
                            print(f"[OFFER_SEND] conv save error: {type(_conv_e).__name__}: {_conv_e}", flush=True)
                    _sc = _offer_r.get("screened_out_count", 0)
                    _sc_msg = f" / 精査除外{_sc}件" if _sc else ""
                    return {
                        "status": _offer_r.get("status", "DONE"),
                        "executed": _offer_r.get("sent_count", 0) > 0,
                        "login_success": True,
                        "operation_type": "offer_send",
                        "offer_result": _offer_r,
                        "verification": {"verified": _offer_r.get("sent_count", 0) > 0, "method": "offer_send"},
                        "rollback": None,
                        "message": (
                            f"オファー送信完了: {_offer_r.get('sent_count',0)}件送信"
                            f" / {_offer_r.get('failed_count',0)}件失敗{_sc_msg}"
                            f" / 候補者{_offer_r.get('candidates_found',0)}件"
                        ),
                    }

                # recruit_reply: 会話スレッドへの返信送信
                if operation_type == "recruit_reply":
                    try:
                        _rr_result = _execute_recruit_reply(page, media_mapping, payload)
                    except RuntimeError as _rr_e:
                        return {
                            "status": "FAILED", "executed": False,
                            "login_success": True, "operation_type": "recruit_reply",
                            "error": str(_rr_e), "message": str(_rr_e),
                        }
                    return {
                        "status": "DONE", "executed": True, "login_success": True,
                        "operation_type": "recruit_reply",
                        "verification": {"verified": True, "method": "recruit_reply"},
                        "rollback": None,
                        "message": f"返信を送信しました: {_rr_result.get('reply_url', '')}",
                    }

                # recruit_inbox_scan: 受信ボックス監視 → 会話スレッド自動更新
                if operation_type == "recruit_inbox_scan":
                    try:
                        from api.core.firestore_client import get_db as _get_db_ib
                        _ib_result = _execute_recruit_inbox_scan(
                            page, media_mapping, payload, db=_get_db_ib()
                        )
                    except RuntimeError as _ib_e:
                        return {
                            "status": "FAILED", "executed": False,
                            "login_success": True, "operation_type": "recruit_inbox_scan",
                            "error": str(_ib_e), "message": str(_ib_e),
                        }
                    return {
                        "status": "DONE", "executed": True, "login_success": True,
                        "operation_type": "recruit_inbox_scan",
                        "inbox_result": _ib_result,
                        "verification": {"verified": True, "method": "recruit_inbox_scan"},
                        "rollback": None,
                        "message": (
                            f"受信ボックス確認完了: {_ib_result.get('messages_found',0)}件取得 / "
                            f"{_ib_result.get('updated_count',0)}件の会話を更新"
                        ),
                    }

                # page_monitor: フォーム操作なし・読み取り専用パス
                if operation_type == "page_monitor":
                    try:
                        _mon = _execute_page_monitor(page, media_mapping, payload=payload)
                    except RuntimeError as _mon_e:
                        return {
                            "status": "FAILED",
                            "executed": False,
                            "login_success": True,
                            "operation_type": "page_monitor",
                            "error": str(_mon_e),
                        }
                    return {
                        "status": "DONE",
                        "executed": True,
                        "login_success": True,
                        "operation_type": "page_monitor",
                        "monitor_data": _mon.get("monitor_data"),
                        "screenshot_b64": _mon.get("screenshot_b64"),
                        "target_url": _mon.get("target_url"),
                        "html_length": _mon.get("html_length", 0),
                        "verification": {"verified": True, "method": "read_only"},
                        "rollback": None,
                        "message": "監視ページの読み取りが完了しました",
                    }

                # operation_mappings[op].target_url が設定されていれば操作前にナビゲート
                _op_target_url = (
                    (media_mapping.get("operation_mappings") or {})
                    .get(operation_type, {})
                    .get("target_url", "")
                )
                if _op_target_url and _op_target_url != page.url:
                    try:
                        print(f"[OP_TARGET_NAV] op={operation_type} url={_op_target_url}", flush=True)
                        page.goto(_op_target_url, timeout=30000, wait_until="domcontentloaded")
                    except Exception as _nav_e:
                        print(f"[OP_TARGET_NAV_ERROR] op={operation_type} err={type(_nav_e).__name__}", flush=True)

                # P7-rollback: before_values取得
                try:
                    before_values = _capture_before_values(page, media_mapping, operation_type)
                except Exception as _bvc_e:
                    print(f"[P14_BEFORE_CAPTURE_ERROR] error={type(_bvc_e).__name__}", flush=True)
                    before_values = {}

                try:
                    _step_results_buf = _execute_operation(page, media_mapping, operation_type, payload, operation_steps=operation_steps,
                                                           prior_step_results=prior_step_results, task_id=task_id, db=db) or []
                except RuntimeError as oe:
                    _failed_step_results = getattr(oe, "step_results", []) or []
                    _terminal_done = _terminal_step_done(operation_steps, _failed_step_results)
                    # P19: self_heal前にfailure clusterから類似repair参照（提案のみ・自動適用禁止）
                    _p19_similar_repairs = []
                    try:
                        from api.core.firestore_client import get_db as _get_db_p19
                        _p19_db = _get_db_p19()
                        if _p19_db is not None:
                            _p19_tenant = media_mapping.get("tenant_id", "")
                            _p19_family = media_mapping.get("media_family", "")
                            _p19_similar_repairs = find_similar_failures(
                                db=_p19_db,
                                tenant_id=_p19_tenant,
                                error_type=type(oe).__name__,
                                operation_type=operation_type,
                                media_family=_p19_family,
                                top_n=3,
                            )
                    except Exception as _p19_e:
                        print(f"[P19 find_similar_failures] エラー: {type(_p19_e).__name__}", flush=True)
                    # P13: self-heal試行
                    _self_heal = _attempt_self_heal_execution(
                        page, media_mapping, operation_type, payload,
                        before_values, before_hash,
                        operation_steps=operation_steps,
                        prior_step_results=_failed_step_results,
                    )
                    # P19: similar_repairsをself_heal結果に付与（参照用）
                    if _p19_similar_repairs:
                        _self_heal["similar_failure_repairs"] = _p19_similar_repairs
                    if _self_heal.get("retry_succeeded") and _self_heal.get("verification", {}).get("verified"):
                        return {
                            "status":         "DONE",
                            "executed":       True,
                            "login_success":  True,
                            "operation_type": operation_type,
                            "verification":   _self_heal.get("verification"),
                            "rollback":       None,
                            "selector_repair": {
                                "suggested": True,
                                "failed_selectors": _self_heal.get("failed_selectors", []),
                                "suggested_selectors": _self_heal.get("suggested_selectors", []),
                            },
                            "self_heal": _self_heal,
                            "step_results": _failed_step_results,
                            "terminal_step_done": _terminal_done,
                            "retry_safe": not _terminal_done,
                        }
                    # self-heal失敗 → rollback
                    try:
                        rollback = _rollback_fields(page, media_mapping, operation_type, before_values)
                    except Exception as _rb_e0:
                        print(f"[P30_ROLLBACK_ERROR] error={type(_rb_e0).__name__}", flush=True)
                        rollback = {
                            "attempted": False, "success": False,
                            "restored_fields": [], "failed_fields": [],
                            "reason": "rollback中に予期しないエラー",
                        }
                    if _terminal_done:
                        rollback = _mark_irreversible_rollback(rollback)
                    return {
                        "status":         "FAILED",
                        "executed":       False,
                        "login_success":  True,
                        "operation_type": operation_type,
                        "message":        str(oe),
                        "verification":   None,
                        "rollback":       rollback,
                        "selector_repair": {
                            "suggested": bool(_self_heal.get("suggested_selectors")),
                            "failed_selectors": _self_heal.get("failed_selectors", []),
                            "suggested_selectors": _self_heal.get("suggested_selectors", []),
                        },
                        "self_heal": _self_heal,
                        "step_results": _failed_step_results,
                        "terminal_step_done": _terminal_done,
                        "retry_safe": not _terminal_done,
                    }

                # P6: after_html取得 + 検証
                try:
                    after_html = page.content()
                except Exception:
                    after_html = ""
                # P28: after_values取得 + diff生成
                try:
                    _after_values = _capture_after_values(page, media_mapping, operation_type)
                    _diff = _build_diff(before_values, _after_values)
                except Exception as _p28_e:
                    _after_values = {}
                    _diff = {}
                    print(f"[P28_DIFF_ERROR] {_p28_e}", flush=True)
                verification = _verify_operation_detail(page, media_mapping, before_hash, after_html)
                if verification.get("verified"):
                    _done = {
                        "status":         "DONE",
                        "executed":       True,
                        "login_success":  True,
                        "operation_type": operation_type,
                        "verification":   verification,
                        "rollback":       None,
                        "diff":           _diff,
                    }
                    # P14: operation_stepsがある場合はstep_resultsを付加
                    if operation_steps:
                        _done["operation_graph"] = True
                        _done["step_results"]    = _step_results_buf
                        _done["current_step"]    = "verify"
                    return _done
                else:
                    # P19: self_heal前にfailure clusterから類似repair参照（提案のみ・自動適用禁止）
                    _p19_similar_repairs2 = []
                    try:
                        from api.core.firestore_client import get_db as _get_db_p19b
                        _p19_db2 = _get_db_p19b()
                        if _p19_db2 is not None:
                            _p19_tenant2 = media_mapping.get("tenant_id", "")
                            _p19_family2 = media_mapping.get("media_family", "")
                            _p19_similar_repairs2 = find_similar_failures(
                                db=_p19_db2,
                                tenant_id=_p19_tenant2,
                                error_type="verify_failure",
                                operation_type=operation_type,
                                media_family=_p19_family2,
                                top_n=3,
                            )
                    except Exception as _p19_e2:
                        print(f"[P19 find_similar_failures2] エラー: {type(_p19_e2).__name__}", flush=True)
                    # P13: 検証失敗 → self-heal試行
                    _self_heal = _attempt_self_heal_execution(
                        page, media_mapping, operation_type, payload,
                        before_values, before_hash,
                        operation_steps=operation_steps,
                        prior_step_results=_step_results_buf,
                    )
                    _terminal_done = _terminal_step_done(operation_steps, _step_results_buf)
                    if _p19_similar_repairs2:
                        _self_heal["similar_failure_repairs"] = _p19_similar_repairs2
                    if _self_heal.get("retry_succeeded") and _self_heal.get("verification", {}).get("verified"):
                        return {
                            "status":         "DONE",
                            "executed":       True,
                            "login_success":  True,
                            "operation_type": operation_type,
                            "verification":   _self_heal.get("verification"),
                            "rollback":       None,
                            "selector_repair": {
                                "suggested": True,
                                "failed_selectors": _self_heal.get("failed_selectors", []),
                                "suggested_selectors": _self_heal.get("suggested_selectors", []),
                            },
                            "self_heal": _self_heal,
                            "step_results": _step_results_buf,
                            "terminal_step_done": _terminal_done,
                            "retry_safe": not _terminal_done,
                        }
                    # self-heal失敗 → rollback
                    try:
                        rollback = _rollback_fields(page, media_mapping, operation_type, before_values)
                    except Exception as _rb_e1:
                        print(f"[P30_ROLLBACK_ERROR] error={type(_rb_e1).__name__}", flush=True)
                        rollback = {
                            "attempted": False, "success": False,
                            "restored_fields": [], "failed_fields": [],
                            "reason": "rollback中に予期しないエラー",
                        }
                    if _terminal_done:
                        rollback = _mark_irreversible_rollback(rollback)
                    return {
                        "status":             "FAILED",
                        "executed":           False,
                        "login_success":      True,
                        "operation_verified": False,
                        "operation_type":     operation_type,
                        "message":            "操作は実行しましたが、完了確認に失敗しました",
                        "verification":       verification,
                        "rollback":           rollback,
                        "selector_repair": {
                            "suggested": bool(_self_heal.get("suggested_selectors")),
                            "failed_selectors": _self_heal.get("failed_selectors", []),
                            "suggested_selectors": _self_heal.get("suggested_selectors", []),
                        },
                        "self_heal": _self_heal,
                        "step_results": _step_results_buf,
                        "terminal_step_done": _terminal_done,
                        "retry_safe": not _terminal_done,
                    }
            except RuntimeError as re:
                print(f"[browser_executor] operation error: {re}", flush=True)
                return {
                    "status":   "FAILED",
                    "executed": False,
                    "message":  str(re),
                }
            except Exception as e:
                print(f"[browser_executor] unexpected: {type(e).__name__}", flush=True)
                return {
                    "status":   "FAILED",
                    "executed": False,
                    "message":  f"予期しないエラー: {type(e).__name__}",
                }
            finally:
                if browser:
                    try:
                        browser.close()
                    except Exception as e:
                        print(f"[browser_executor] browser close error: {type(e).__name__}", flush=True)

    except ImportError:
        return {
            "status":   "WAITING_EXECUTOR",
            "executed": False,
            "message":  "Playwrightがインストールされていません。",
        }


def _run_api_key_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
) -> dict:
    """APIキー認証型媒体への操作。将来実装。"""
    return {
        "status":         "BLOCKED",
        "executed":       False,
        "auth_type":      "api_key",
        "operation_type": operation_type,
        "message":        "API連携実行層は現在未対応です（開発中）。",
        "blocked_reason": "auth_type_api_key_not_implemented",
    }


def _run_no_auth_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
) -> dict:
    """認証不要型媒体への操作。将来実装。"""
    return {
        "status":         "BLOCKED",
        "executed":       False,
        "auth_type":      "none",
        "operation_type": operation_type,
        "message":        "認証不要型ブラウザ実行層は現在未対応です（開発中）。",
        "blocked_reason": "auth_type_none_not_implemented",
    }



def normalize_css_selector(sel: str) -> str:
    """
    P0-1: CSSセレクターを正規化する。
    - 壊れたスペースを修正 (input [name=...] -> input[name=...])
    - <> タグ記法を除去
    - id属性がある場合は #id 形式を優先
    """
    if not sel:
        return ""
    s = str(sel).strip()
    s = s.replace("<", "").replace(">", "")
    s = s.replace("input [", "input[")
    s = s.replace("button [", "button[")
    s = s.replace("textarea [", "textarea[")
    s = s.replace("select [", "select[")
    return s


def selector_from_element(el: dict) -> str:
    """
    P0-1: DOM要素から最適なCSSセレクターを生成する。
    id属性がある場合は #id を優先する。
    """
    el_id = el.get("id")
    if el_id:
        return "#" + el_id
    name = el.get("name") or el.get("aria_label")
    tag = el.get("tag", "input")
    if name:
        return f'{tag}[name="{name}"]'
    return el.get("suggested_selector") or ""


def infer_dom_semantics(elements: list) -> dict:
    """
    P15-2: DOM要素リストから意味ラベルを推定する。
    LLMは使用せず、heuristicのみで判定。
    返り値: { "labels": {label: selector}, "confidence": {label: "high"|"medium"|"low"} }
    """
    labels = {}
    confidence = {}

    def _score(el, keywords):
        text = " ".join([
            el.get("name") or "",
            el.get("placeholder") or "",
            el.get("aria_label") or "",
            el.get("id") or "",
            " ".join(el.get("class_list") or []),
            el.get("text_content") or "",
            el.get("parent_id") or "",
            " ".join(el.get("parent_class") or []),
        ]).lower()
        return sum(1 for k in keywords if k in text)

    for el in elements:
        tag = el.get("tag", "")
        typ = (el.get("type") or "").lower()
        sel = normalize_css_selector(selector_from_element(el))
        if not sel:
            continue

        # login_id
        if "login_id" not in labels and tag == "input" and typ in ("text", "email", ""):
            score = _score(el, ["user", "mail", "login", "id", "account", "email", "username"])
            if score >= 2:
                labels["login_id"] = sel
                confidence["login_id"] = "high"
            elif score == 1:
                labels.setdefault("login_id", sel)
                confidence.setdefault("login_id", "medium")

        # login_password
        if "login_password" not in labels and tag == "input" and typ == "password":
            labels["login_password"] = sel
            confidence["login_password"] = "high"

        # title
        if "title" not in labels and tag == "input" and typ in ("text", ""):
            score = _score(el, ["title", "subject", "headline", "name", "topic"])
            if score >= 1:
                labels["title"] = sel
                confidence["title"] = "high" if score >= 2 else "medium"

        # body
        if "body" not in labels and tag == "textarea":
            score = _score(el, ["body", "content", "text", "message", "detail", "description"])
            if score >= 1:
                labels["body"] = sel
                confidence["body"] = "high"
            else:
                labels.setdefault("body", sel)
                confidence.setdefault("body", "low")

        # search
        if "search" not in labels and tag == "input" and typ in ("text", "search", ""):
            score = _score(el, ["search", "query", "keyword", "find"])
            if score >= 1:
                labels["search"] = sel
                confidence["search"] = "high" if score >= 2 else "medium"

        # price
        if "price" not in labels and tag == "input":
            score = _score(el, ["price", "fee", "cost", "amount", "charge"])
            if score >= 1:
                labels["price"] = sel
                confidence["price"] = "high" if score >= 2 else "medium"

        # image_upload
        if "image_upload" not in labels and tag == "input" and typ == "file":
            labels["image_upload"] = sel
            confidence["image_upload"] = "high"

        # submit
        if "submit" not in labels and (tag == "button" or (tag == "input" and typ == "submit")):
            _neg_keys = ("search", "filter", "find", "sort", "order", "preview", "back", "cancel", "絞込", "検索")
            _el_text = ((el.get("id") or "") + (el.get("name") or "") + (el.get("value") or "") + (el.get("text") or "") + (el.get("placeholder") or "")).lower()
            _is_negative = any(k in _el_text for k in _neg_keys)
            if _is_negative:
                print(f"[P24_SAVE_REJECTED] selector={sel} reason=negative_keyword_matched text={_el_text[:80]}", flush=True)
            else:
                score = _score(el, ["submit", "send", "post", "save", "login", "sign", "register", "confirm"])
                if score >= 1:
                    labels["submit"] = sel
                    confidence["submit"] = "high"
                else:
                    labels.setdefault("submit", sel)
                    confidence.setdefault("submit", "low")

    return {"labels": labels, "confidence": confidence}


# ── 対話型マッピング ────────────────────────────────────────────────

def scan_page_for_mapping(media_mapping: dict, creds: dict, page_url: str, page_name: str, intent: str = "") -> dict:
    """
    対話型マッピング用。指定ページに遷移→DOM解析→LLMでステップと候補を動的生成して返す。
    固定operation_typeを持たず、サイト固有のページ構造に適応する。
    intent: 操作意図ヒント（例: "新規登録ページ", "編集ページ"）
    返り値: {ok, page_name, page_url, steps, discovered_tabs}
    """
    import json as _json
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            auth = create_authenticated_page(p, media_mapping, creds)
            browser, page = auth["browser"], auth["page"]
            try:
                # 指定ページに遷移
                page.goto(page_url, timeout=15000, wait_until="domcontentloaded")
                page.wait_for_timeout(800)
                current_url = page.url
                page_title  = page.title()

                # ページ内リンクスキャン（サブページ・個別編集リンク候補用）
                try:
                    links = page.evaluate("""() => {
                        return Array.from(document.querySelectorAll('a[href]')).slice(0, 100).map(a => ({
                            href: a.href,
                            text: (a.textContent || '').trim().slice(0, 60),
                        })).filter(a => a.href && !a.href.startsWith('javascript:') && a.text.length > 0);
                    }""")
                except Exception:
                    links = []

                # タブメニュー・ページ内ナビリンク抽出（サイドバーに含まれないタブを検出）
                try:
                    discovered_tabs = page.evaluate("""() => {
                        const base = document.baseURI;
                        function makeAbsolute(href) {
                            try { return new URL(href, base).href; } catch { return href; }
                        }
                        const navSelectors = [
                            'ul.girl-menu a', 'ul.pagemenu a', 'ul.tab-menu a', '.page-tab a',
                            '.sub-menu a', 'ul.subMenu a', '.nav-tabs a', '.tabs a',
                            '.page-menu a', '[class*="tab"] a', '[class*="menu"] a'
                        ];
                        let found = [];
                        const seen = new Set();
                        for (const sel of navSelectors) {
                            try {
                                Array.from(document.querySelectorAll(sel)).forEach(a => {
                                    const href = a.getAttribute('href') || '';
                                    const absUrl = makeAbsolute(href);
                                    if (href && !href.startsWith('#') && !href.startsWith('javascript:')
                                        && absUrl.startsWith('http') && !seen.has(absUrl)) {
                                        seen.add(absUrl);
                                        found.push({
                                            href: href,
                                            absolute_url: absUrl,
                                            text: (a.textContent || '').trim().slice(0, 60),
                                        });
                                    }
                                });
                            } catch(e) {}
                        }
                        return found.slice(0, 30);
                    }""")
                except Exception:
                    discovered_tabs = []

                # フォーム要素スキャン
                elements = _raw_scan_page(page)
                elem_summary = [
                    {k: v for k, v in e.items() if k not in ("_ctx", "_frame_url")}
                    for e in elements[:200]
                ]

                from api.core.llm_client import call_llm

                system_prompt = (
                    "あなたはWebサイト管理画面のDOM解析専門AIです。\n"
                    "指定ページの目的（一覧・新規作成・編集・設定など）を分析し、\n"
                    "ASCENDエージェントが自動操作するために必要なセレクターのステップを動的に生成してください。\n\n"
                    "ルール:\n"
                    "- 一覧ページ: 新規作成ボタン・個別編集リンクパターンを質問\n"
                    "- フォームページ: 主要入力欄・ファイルアップロード・保存ボタンを質問\n"
                    "- 質問は最小限（2〜5個）。必須でないものは optional: true\n"
                    "- 候補には必ず tag / text / placeholder を含める\n\n"
                    "回答はJSONのみ（```不要）。形式:\n"
                    '{"page_type":"list|form|unknown","steps":['
                    '{"role":"new_button","question":"新規作成ボタンはどれですか？","type":"selector","optional":false,'
                    '"candidates":[{"selector":"a.btn-add","description":"追加ボタン","confidence":"high","tag":"a","text":"新規追加","placeholder":null}]}'
                    "]}"
                )
                intent_hint = f"\n操作意図: {intent}" if intent else ""
                user_msg = (
                    f"ページ名: {page_name}\n"
                    f"URL: {current_url}\n"
                    f"タイトル: {page_title}{intent_hint}\n\n"
                    f"ページ内リンク:\n{_json.dumps(links[:50], ensure_ascii=False)}\n\n"
                    f"フォーム要素:\n{_json.dumps(elem_summary, ensure_ascii=False)}"
                )

                raw = call_llm(system_prompt, [{"role": "user", "content": user_msg}],
                               ai_tier="core", temperature=0.1)

                try:
                    cleaned = raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                        cleaned = cleaned.rsplit("```", 1)[0].strip()
                    data = _json.loads(cleaned)
                except Exception as pe:
                    print(f"[PAGE_SCAN_PARSE_ERR] {pe} raw={raw[:200]}", flush=True)
                    data = {"page_type": "unknown", "steps": []}

                # target_url ステップを先頭に固定追加（このページURLで確定）
                steps_out = [
                    {
                        "role":       "target_url",
                        "question":   f"「{page_name}」のページURLを確認してください",
                        "type":       "url",
                        "optional":   False,
                        "candidates": [{"value": current_url, "description": page_title or current_url, "confidence": "high"}],
                    }
                ] + [
                    {
                        "role":       s.get("role", ""),
                        "question":   s.get("question", ""),
                        "type":       s.get("type", "selector"),
                        "optional":   s.get("optional", False),
                        "candidates": s.get("candidates", []),
                    }
                    for s in data.get("steps", [])
                    if s.get("role") and s.get("question")
                ]

                return {"ok": True, "page_name": page_name, "page_url": current_url, "steps": steps_out, "discovered_tabs": discovered_tabs}

            finally:
                try: browser.close()
                except Exception: pass

    except Exception as e:
        print(f"[PAGE_SCAN_ERROR] page={page_url} name={page_name} err={type(e).__name__}:{e}", flush=True)
        return {"ok": False, "error": str(e)}


def preview_selector_element(media_mapping: dict, creds: dict, selector: str, navigate_url: str) -> dict:
    """
    指定セレクターに該当する要素をハイライトしてスクリーンショットを返す。
    ID/PASSはログ・戻り値に絶対含めない。
    返却: {ok, screenshot_b64, element_found, element_tag, element_text}
    """
    import base64 as _b64
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            auth = create_authenticated_page(p, media_mapping, creds)
            browser, page = auth["browser"], auth["page"]
            try:
                if navigate_url:
                    page.goto(navigate_url, timeout=15000, wait_until="domcontentloaded")
                    page.wait_for_timeout(800)

                # 要素を探してハイライト
                element_info = page.evaluate(f"""(sel) => {{
                    try {{
                        const el = document.querySelector(sel);
                        if (!el) return null;
                        el.style.outline = '3px solid #ef4444';
                        el.style.outlineOffset = '2px';
                        el.style.backgroundColor = 'rgba(239,68,68,0.08)';
                        el.scrollIntoView({{block:'center', inline:'center'}});
                        return {{
                            tag: el.tagName.toLowerCase(),
                            text: (el.textContent || '').trim().slice(0, 80),
                            placeholder: el.getAttribute('placeholder') || null,
                            found: true,
                        }};
                    }} catch(e) {{ return null; }}
                }}""", selector)

                page.wait_for_timeout(300)
                screenshot_bytes = page.screenshot(type="jpeg", quality=70, full_page=False)
                b64 = _b64.b64encode(screenshot_bytes).decode("utf-8")

                if element_info:
                    return {
                        "ok": True,
                        "screenshot_b64": b64,
                        "element_found": True,
                        "element_tag":  element_info.get("tag", ""),
                        "element_text": element_info.get("text", ""),
                    }
                else:
                    return {
                        "ok": True,
                        "screenshot_b64": b64,
                        "element_found": False,
                        "element_tag": "",
                        "element_text": "",
                    }
            finally:
                try: browser.close()
                except Exception: pass
    except Exception as e:
        print(f"[ELEMENT_PREVIEW_ERROR] selector={selector} err={type(e).__name__}:{e}", flush=True)
        return {"ok": False, "error": str(e)}


def run_dom_scan(
    media_mapping: dict,
    max_pages: int = 200,
    start_url: str = "",
    include_patterns: list = None,
    exclude_patterns: list = None,
    reset_resume: bool = False,
) -> dict:
    """
    P5: DOM自動候補抽出。
    Playwrightで対象URLを開き、input/textarea/button/select/file inputを収集。
    ログインが必要な場合はlogin_urlへアクセスしてから対象URLへ遷移。
    ID/PASSはログ・戻り値に絶対含めない。
    """
    if not is_playwright_enabled():
        return {
            "status": "WAITING_EXECUTOR",
            "executed": False,
            "message": "PLAYWRIGHT_ENABLED=false のためDOM自動解析は無効です。",
        }

    target_url = media_mapping.get("media_url") or media_mapping.get("login_url")
    if not target_url:
        return {
            "status": "BLOCKED",
            "executed": False,
            "message": "media_url が未設定です。",
        }

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    if secret_name:
        creds = get_secret_json(secret_name)
        if creds and creds.get("blocked"):
            creds = None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        with sync_playwright() as p:
            browser = None
            _login_browser = None
            try:
                browser = None
                page = None

                # ログイン試行（credential存在時）
                _login_browser = None
                _login_page = None
                _login_success_dom = False
                if creds:
                    try:
                        _auth_dom = create_authenticated_page(p, media_mapping, creds)
                        _login_browser, _login_page = _auth_dom["browser"], _auth_dom["page"]
                        _login_success_dom = True
                        print(f"[P5_LOGIN_SUCCESS] url={_login_page.url}", flush=True)
                        # 最初のbrowserは不要なのでclose
                        try:
                            if browser:
                                browser.close()
                        except Exception:
                            pass
                        browser = None
                    except RuntimeError as _le:
                        print(f"[P5_LOGIN_FAILED] reason={_le}", flush=True)
                        _p27_anomaly_check(
                            mapping_id=str(media_mapping.get("id") or media_mapping.get("mapping_id") or ""),
                            event_type="login_failed",
                        )

                # ログイン済みpageがあればそれを使う
                if _login_success_dom and _login_page:
                    page = _login_page

                # 対象URLへ遷移（ログイン成功時はスキップ・ログイン後URLから継続）
                if not _login_success_dom:
                    # ログイン不要 or ログイン失敗時: browserを新規起動してpageを生成
                    if page is None:
                        browser = p.chromium.launch(headless=True)
                        context = browser.new_context()
                        page = context.new_page()
                    try:
                        page.goto(target_url, timeout=15000)
                        page.wait_for_load_state("domcontentloaded", timeout=35000)
                    except PlaywrightTimeout:
                        return {
                            "status": "BLOCKED",
                            "executed": False,
                            "message": "対象URLの読み込みタイムアウト。",
                        }


                # DOM収集
                selectors = page.evaluate("""() => {
                    const results = [];
                    const els = document.querySelectorAll('input, textarea, button, select');
                    els.forEach((el, idx) => {
                        const parent = el.parentElement;
                        const info = {
                            tag: el.tagName.toLowerCase(),
                            type: el.getAttribute('type') || null,
                            id: el.id || null,
                            name: el.getAttribute('name') || null,
                            placeholder: el.getAttribute('placeholder') || null,
                            aria_label: el.getAttribute('aria-label') || null,
                            text_content: (el.textContent || '').trim().slice(0, 50) || null,
                            parent_tag: parent ? parent.tagName.toLowerCase() : null,
                            parent_id: parent ? (parent.id || null) : null,
                            parent_class: parent ? (parent.className ? parent.className.trim().split(/\\s+/).slice(0, 3) : []) : [],
                            class_list: el.className ? el.className.trim().split(/\\s+/).slice(0, 3) : [],
                            index: idx,
                        };
                        // CSSセレクター候補を生成
                        if (el.id) {
                            info.suggested_selector = '#' + el.id;
                        } else if (el.getAttribute('name')) {
                            info.suggested_selector = el.tagName.toLowerCase() + '[name="' + el.getAttribute('name') + '"]';
                        } else if (el.getAttribute('aria-label')) {
                            info.suggested_selector = el.tagName.toLowerCase() + '[aria-label="' + el.getAttribute('aria-label') + '"]';
                        } else {
                            info.suggested_selector = null;
                        }
                        results.push(info);
                    });
                    return results;
                }""")

                # suggested_structure生成（用途推定）P0-1: selector_from_element + normalize適用
                suggested_structure = {}
                suggested_verify_selector = None
                for el in selectors:
                    sel = normalize_css_selector(selector_from_element(el))
                    if not sel:
                        continue
                    tag  = el.get("tag", "")
                    typ  = (el.get("type") or "").lower()
                    name = (el.get("name") or "").lower()
                    ph   = (el.get("placeholder") or "").lower()

                    if tag == "input" and typ in ("text", "email", "") and any(k in name + ph for k in ("user", "mail", "login", "id", "account")):
                        suggested_structure.setdefault("username", sel)
                    elif tag == "input" and typ == "password":
                        suggested_structure.setdefault("password", sel)
                    elif tag == "input" and typ == "file":
                        suggested_structure.setdefault("file_input", sel)
                    elif tag == "textarea":
                        suggested_structure.setdefault("body_field", sel)
                    elif tag == "input" and typ in ("text", ""):
                        suggested_structure.setdefault("text_field", sel)
                    elif tag == "select":
                        suggested_structure.setdefault("select_field", sel)
                    elif tag == "button" or (tag == "input" and typ == "submit"):
                        if any(k in name + ph + (el.get("id") or "").lower() for k in ("submit", "send", "post", "save", "login", "sign")):
                            suggested_structure.setdefault("submit", sel)
                            suggested_verify_selector = suggested_verify_selector or sel

                # capabilities_candidate推定
                _caps_candidate = {}
                _caps_reason = {}
                _has_username = "username" in suggested_structure
                _has_password = "password" in suggested_structure
                _has_submit   = "submit" in suggested_structure
                _has_file     = "file_input" in suggested_structure
                _has_textarea = "body_field" in suggested_structure
                _has_verify   = bool(suggested_verify_selector)
                if _has_username and _has_password and _has_submit:
                    _caps_candidate["can_login"] = True
                    _caps_reason["can_login"] = "username/password/submitが検出されました"
                if _has_file:
                    _caps_candidate["can_upload_image"] = True
                    _caps_reason["can_upload_image"] = "file inputが検出されました"
                if _has_textarea and _has_submit:
                    _caps_candidate["can_post_news"] = True
                    _caps_candidate["can_update_text"] = True
                    _caps_reason["can_post_news"] = "textarea+submitが検出されました"
                    _caps_reason["can_update_text"] = "textarea+submitが検出されました"
                if _has_verify:
                    _caps_candidate["can_verify"] = True
                    _caps_reason["can_verify"] = "verify_selector候補が検出されました"
                capabilities_candidate = {
                    "capabilities": _caps_candidate,
                    "reason": _caps_reason,
                }
                # P15-2: semantic labeling
                import datetime as _dt15
                _semantic = infer_dom_semantics(selectors)
                semantic_selector_candidates = {
                    "created_at": _dt15.datetime.utcnow(),
                    "labels": _semantic.get("labels", {}),
                    "confidence": _semantic.get("confidence", {}),
                }
                # P0-1: detected_summary 生成
                _sem_labels = _semantic.get("labels", {})
                _sem_conf   = _semantic.get("confidence", {})
                _label_map  = {
                    "login_id": "username", "id": "username", "user_id": "username",
                    "username": "username", "account": "username", "email": "username",
                    "mail": "username", "loginId": "username",
                    "login_password": "password", "pass": "password",
                    "password": "password", "pwd": "password",
                    "submit": "login_submit", "login_submit": "login_submit",
                    "login_button": "login_submit", "button": "login_submit",
                    "send": "login_submit",
                }
                _norm_for_summary = {}
                for _lk, _lv in _sem_labels.items():
                    _nk = _label_map.get(_lk, _lk)
                    if _nk not in _norm_for_summary:
                        _norm_for_summary[_nk] = normalize_css_selector(_lv)

                detected_summary = {
                    "login_id":       _norm_for_summary.get("username"),
                    "password":       _norm_for_summary.get("password"),
                    "login_button":   _norm_for_summary.get("login_submit"),
                    "verify_candidates": [suggested_verify_selector] if suggested_verify_selector else [],
                    "inputs":         [normalize_css_selector(selector_from_element(e)) for e in selectors if e.get("tag") == "input" and (e.get("type") or "") not in ("file", "hidden", "submit")],
                    "buttons":        [normalize_css_selector(selector_from_element(e)) for e in selectors if e.get("tag") == "button" or (e.get("tag") == "input" and e.get("type") == "submit")],
                    "forms":          [],
                    "file_inputs":    [normalize_css_selector(selector_from_element(e)) for e in selectors if e.get("tag") == "input" and e.get("type") == "file"],
                    "textareas":      [normalize_css_selector(selector_from_element(e)) for e in selectors if e.get("tag") == "textarea"],
                    "capabilities":   list(_caps_candidate.keys()),
                }

                # ── P21: Post Login Admin Crawler ────────────────────────
                _p21_result = {}
                _p21_mapping_id = media_mapping.get("mapping_id") or media_mapping.get("id")
                if _p21_mapping_id and _login_success_dom:
                    print(f"[P21_START] mapping_id={_p21_mapping_id}", flush=True)
                    try:
                        from api.core.firestore_client import get_db as _get_db_p21
                        _p21_result = post_login_admin_crawl(page, _p21_mapping_id, _get_db_p21(), max_pages=max_pages, start_url=start_url, include_patterns=include_patterns, exclude_patterns=exclude_patterns, reset_resume=reset_resume)
                        print(f"[P21_COMPLETE] mapping_id={_p21_mapping_id} pages={_p21_result.get('pages_crawled',0)} status={_p21_result.get('status')}", flush=True)
                        # P21成功時にlogin_health=HEALTHYを保存
                        try:
                            import datetime as _dt_lh
                            _get_db_p21().collection("media_mappings").document(_p21_mapping_id).update({
                                "login_health": "HEALTHY",
                                "selector_health": "HEALTHY",
                                "health_score": 100,
                                "consecutive_failures": 0,
                                "last_success_at": _dt_lh.datetime.utcnow(),
                                "last_verified_at": _dt_lh.datetime.utcnow(),
                            })
                            print(f"[P21_LOGIN_HEALTH_SAVED] mapping_id={_p21_mapping_id} login_health=HEALTHY", flush=True)
                        except Exception as _lh_err:
                            print(f"[P21_LOGIN_HEALTH_ERROR] {_lh_err}", flush=True)
                    except Exception as _p21_err:
                        print(f"[P21_ERROR] mapping_id={_p21_mapping_id} error={type(_p21_err).__name__}: {_p21_err}", flush=True)
                else:
                    print(f"[P21_SKIP] mapping_id未設定のためP21スキップ", flush=True)

                return {
                    "status": "DONE",
                    "executed": True,
                    "target_url": target_url,
                    "selectors": selectors,
                    "suggested_structure": suggested_structure,
                    "suggested_verify_selector": suggested_verify_selector,
                    "capabilities_candidate": capabilities_candidate,
                    "semantic_selector_candidates": semantic_selector_candidates,
                    "detected_summary": detected_summary,
                    "admin_crawl_completed":      _p21_result.get("status") == "OK",
                    "pages_crawled":              _p21_result.get("pages_crawled", 0),
                    "crawl_state":                _p21_result.get("crawl_state", {}),
                    "crawl_resume_queue_count":   _p21_result.get("crawl_resume_queue_count", 0),
                    "crawl_paused":               _p21_result.get("crawl_paused", False),
                    "last_crawled_url":           _p21_result.get("last_crawled_url", ""),
                }

            except Exception as e:
                import traceback
                print(f"[browser_executor] dom_scan error: {type(e).__name__}", flush=True)
                print(traceback.format_exc(), flush=True)
                return {
                    "status": "FAILED",
                    "executed": False,
                    "message": f"DOM解析中にエラーが発生しました: {type(e).__name__}",
                }
            finally:
                try:
                    if browser:
                        browser.close()
                except Exception:
                    pass
                try:
                    if _login_browser and _login_browser is not browser:
                        _login_browser.close()
                except Exception:
                    pass

    except ImportError:
        return {
            "status": "WAITING_EXECUTOR",
            "executed": False,
            "message": "Playwrightがインストールされていません。",
        }


# ── P16.7: 分離された scoring 関数群 ──────────────────────────────────────

def compute_heuristic_score(
    selector: str,
    source: str,
    conf: str,
    label: str,
    semantic_label: str,
    cand: dict,
    success_selectors: dict,
    failed_selectors: dict,
    timeout_selectors: set,
) -> tuple:
    """
    P16.7: heuristic scoreをrank_selector_candidatesから分離。
    返り値: (h_score: float, h_reasons: list)
    """
    sel_lower = selector.lower()
    h_score   = 0.5
    h_reasons = []
    if source == "semantic" and conf == "high":
        h_score += 0.25; h_reasons.append("semantic confidence high")
    elif source == "semantic" and conf == "medium":
        h_score += 0.10; h_reasons.append("semantic confidence medium")
    if selector in success_selectors:
        bonus = min(0.20, success_selectors[selector] * 0.05)
        h_score += bonus; h_reasons.append(f"過去成功{success_selectors[selector]}回")
    if cand.get("selector_success_rate", 0) > 0.8:
        h_score += 0.10; h_reasons.append("selector_success_rate高")
    if cand.get("verify_success"):
        h_score += 0.05; h_reasons.append("verify成功履歴あり")
    if semantic_label and label and semantic_label.lower() in label.lower():
        h_score += 0.10; h_reasons.append("exact label match")
    if selector.startswith("#") or "[name=" in sel_lower or "[aria-label=" in sel_lower:
        h_score += 0.08; h_reasons.append("id/name/aria-label由来")
    if selector in failed_selectors:
        penalty = min(0.30, failed_selectors[selector] * 0.08)
        h_score -= penalty; h_reasons.append(f"過去失敗{failed_selectors[selector]}回")
    if selector in timeout_selectors:
        h_score -= 0.15; h_reasons.append("timeout履歴あり")
    if source == "semantic" and conf == "low":
        h_score -= 0.10; h_reasons.append("confidence low")
    if selector.startswith(".") and " " not in selector:
        h_score -= 0.08; h_reasons.append("classのみの脆いselector")
    if "nth-child" in sel_lower or "nth-of-type" in sel_lower:
        h_score -= 0.12; h_reasons.append("nth-child系selector")
    h_score = max(0.0, min(1.0, h_score))
    return h_score, h_reasons


def compute_learning_score(
    selector: str,
    conf: str,
    stats: dict,
) -> tuple:
    """
    P16.7: learning scoreをrank_selector_candidatesから分離。
    返り値: (l_score: float, fv: dict)
    """
    fv = build_selector_feature_vector(selector, conf, stats)
    l_score = compute_selector_learning_score(fv)
    return l_score, fv


def compute_model_score(
    feature_vector: dict,
) -> dict:
    """
    P16.7: model scoreをselector_rank_trainerへブリッジ。
    将来LightGBM/XGBoost/embedding rerankに差し替え可能。
    返り値: {"model_score": float, "confidence": float, "model_version": str, "is_stub": bool}
    """
    from api.core.selector_rank_trainer import predict_selector_rank
    return predict_selector_rank(feature_vector)


# ── P16.7: rank_selector_candidates (分離版) ──────────────────────────────

def rank_selector_candidates(
    media_mapping: dict,
    operation_type: str,
    step_type: str,
    semantic_label: str,
    candidates: list,
    agent_logs: list,
    learning_stats: dict = None,
) -> list:
    """
    P16昇格: selector候補をheuristic + learning hybridでスコアリング。
    P16.7: compute_heuristic_score / compute_learning_score / compute_model_score に分離済み。
    learning_weight=0.6 / heuristic_weight=0.4
    自動保存・自動適用禁止。media_mapping本体は変更しない。
    """
    import hashlib
    LEARNING_WEIGHT  = 0.6
    HEURISTIC_WEIGHT = 0.4
    if learning_stats is None:
        learning_stats = {}

    success_selectors = {}
    failed_selectors  = {}
    timeout_selectors = set()
    for log in agent_logs:
        sel_results = log.get("selector_results") or {}
        for sel, result in sel_results.items():
            if result.get("success"):
                success_selectors[sel] = success_selectors.get(sel, 0) + 1
            else:
                failed_selectors[sel] = failed_selectors.get(sel, 0) + 1
            if result.get("timeout"):
                timeout_selectors.add(sel)
        failed_list = log.get("failed_selectors") or []
        for sel in failed_list:
            failed_selectors[sel] = failed_selectors.get(sel, 0) + 1

    ranked = []

    for cand in candidates:
        selector = cand.get("selector") or cand.get("suggested_selector")
        if not selector:
            continue
        source = cand.get("source", "dom_scan")
        label  = cand.get("label") or cand.get("name") or ""
        conf   = cand.get("confidence", "low")

        # --- heuristic score (分離) ---
        h_score, h_reasons = compute_heuristic_score(
            selector, source, conf, label, semantic_label,
            cand, success_selectors, failed_selectors, timeout_selectors,
        )

        # --- learning score (分離) ---
        _hash = hashlib.md5(selector.encode()).hexdigest()[:12]
        _media_name = media_mapping.get("media_name", "")
        _stats_key = f"{_media_name}__{operation_type}__{_hash}"
        _stats = learning_stats.get(_stats_key) or {}
        l_score, fv = compute_learning_score(selector, conf, _stats)

        # --- model score (分離・stub) ---
        model_result = compute_model_score(fv)
        m_score      = model_result.get("model_score", l_score)
        confidence   = model_result.get("confidence", 0.5)

        # --- hybrid ---
        final_score = round(LEARNING_WEIGHT * l_score + HEURISTIC_WEIGHT * h_score, 3)
        final_score = max(0.0, min(1.0, final_score))

        reasons = h_reasons + ([f"learning:{l_score:.2f}"] if _stats else [])
        ranked.append({
            "selector":        selector,
            "score":           final_score,
            "heuristic_score": round(h_score, 3),
            "learning_score":  l_score,
            "model_score":     m_score,
            "confidence":      confidence,
            "feature_vector":  fv,
            "reasons":         reasons,
            "source":          source,
            "label":           label,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # --- P16.7: selector_training_logs 保存（selected + negative sample） ---
    try:
        from api.core.firestore_client import get_db
        _db = get_db()
        if _db is not None:
            save_selector_training_logs(
                db=_db,
                media_mapping=media_mapping,
                operation_type=operation_type,
                ranked_selectors=ranked,
                selected_selector=ranked[0].get("selector") if ranked else None,
            )
    except Exception as _e:
        print(f"[selector_training_logs] 保存エラー: {type(_e).__name__}", flush=True)

    return ranked
def build_selector_feature_vector(
    selector: str,
    semantic_confidence: str = "low",
    stats: dict = None,
) -> dict:
    """
    P16昇格: selector feature vector生成。
    将来のLGBM化に対応できる構造で分離。
    """
    if stats is None:
        stats = {}
    sel_lower = selector.lower()
    total = (stats.get("success_count", 0) + stats.get("failure_count", 0)) or 1
    usage = stats.get("usage_count", 0) or 1
    verify_total = (stats.get("verify_success_count", 0) + stats.get("failure_count", 0)) or 1

    # 時間減衰（recent_success_decay）
    import math, datetime as _dt
    last_success = stats.get("last_success_at")
    if last_success:
        try:
            if hasattr(last_success, "timestamp"):
                days = (_dt.datetime.utcnow() - last_success.replace(tzinfo=None)).days
            else:
                days = 0
            decay = math.exp(-days / 30.0)
        except Exception:
            decay = 0.5
    else:
        decay = 0.0

    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
    return {
        "is_id_selector":          int(selector.startswith("#")),
        "is_name_selector":        int("[name=" in sel_lower),
        "is_aria_selector":        int("[aria-label=" in sel_lower),
        "is_class_only":           int(selector.startswith(".") and " " not in selector),
        "has_nth_child":           int("nth-child" in sel_lower or "nth-of-type" in sel_lower),
        "selector_depth":          selector.count(" ") + selector.count(">"),
        "selector_length":         len(selector),
        "semantic_confidence":     conf_map.get(semantic_confidence, 0.2),
        "historical_success_rate": stats.get("success_count", 0) / total,
        "historical_timeout_rate": stats.get("timeout_count", 0) / total,
        "verify_success_rate":     stats.get("verify_success_count", 0) / verify_total,
        "repair_rate":             stats.get("repair_applied_count", 0) / usage,
        "usage_frequency":         min(1.0, stats.get("usage_count", 0) / 100.0),
        "recent_success_decay":    decay,
    }


def compute_selector_learning_score(feature_vector: dict) -> float:
    """
    P16昇格: learning scoreをfeature vectorから算出。
    現在はrule-based。将来はLightGBMに差し替え可能。
    """
    score = 0.5
    fv = feature_vector

    # 加点
    score += fv.get("is_id_selector", 0) * 0.15
    score += fv.get("is_name_selector", 0) * 0.12
    score += fv.get("is_aria_selector", 0) * 0.10
    score += fv.get("semantic_confidence", 0.2) * 0.20
    score += fv.get("historical_success_rate", 0) * 0.25
    score += fv.get("verify_success_rate", 0) * 0.10
    score += fv.get("recent_success_decay", 0) * 0.10
    score += fv.get("usage_frequency", 0) * 0.05

    # 減点
    score -= fv.get("is_class_only", 0) * 0.10
    score -= fv.get("has_nth_child", 0) * 0.15
    score -= fv.get("historical_timeout_rate", 0) * 0.20
    score -= fv.get("repair_rate", 0) * 0.05
    depth = fv.get("selector_depth", 0)
    if depth > 3:
        score -= (depth - 3) * 0.03

    return max(0.0, min(1.0, round(score, 3)))


def update_selector_learning_stats(
    db,
    media_name: str,
    operation_type: str,
    selector: str,
    success: bool,
    timeout: bool = False,
    verify_success: bool = False,
    latency_ms: float = 0.0,
    semantic_match_score: float = 0.0,
):
    """
    P16昇格: execution結果をselector_learning_statsへ反映。
    media_name + operation_type + selector_hash をkeyとして保存。
    Firestore自動生成IDではなく固定keyで管理。
    """
    import hashlib, datetime as _dt
    from google.cloud import firestore as _fs_sls
    selector_hash = hashlib.md5(selector.encode()).hexdigest()[:12]
    doc_id = f"{media_name}__{operation_type}__{selector_hash}"
    ref = db.collection("selector_learning_stats").document(doc_id)
    # E-2: read-modify-writeレースをトランザクションで解消
    try:
        @_fs_sls.transactional
        def _sls_txn(txn, _ref):
            snap = _ref.get(transaction=txn)
            now = _dt.datetime.utcnow()
            if snap.exists:
                d = snap.to_dict() or {}
                success_count = d.get("success_count", 0) + (1 if success else 0)
                failure_count = d.get("failure_count", 0) + (0 if success else 1)
                timeout_count = d.get("timeout_count", 0) + (1 if timeout else 0)
                verify_count  = d.get("verify_success_count", 0) + (1 if verify_success else 0)
                usage_count   = d.get("usage_count", 0) + 1
                prev_avg      = d.get("avg_latency_ms", 0.0)
                avg_latency   = prev_avg + (latency_ms - prev_avg) / usage_count
                total         = success_count + failure_count or 1
                stability     = round(success_count / total, 4)
                txn.update(_ref, {
                    "success_count":        success_count,
                    "failure_count":        failure_count,
                    "timeout_count":        timeout_count,
                    "verify_success_count": verify_count,
                    "usage_count":          usage_count,
                    "avg_latency_ms":       round(avg_latency, 2),
                    "stability_score":      stability,
                    "semantic_match_score": semantic_match_score,
                    "last_success_at":      now if success else d.get("last_success_at"),
                    "last_failure_at":      now if not success else d.get("last_failure_at"),
                    "last_seen_at":         now,
                })
            else:
                txn.set(_ref, {
                    "selector":              selector,
                    "selector_hash":         selector_hash,
                    "media_name":            media_name,
                    "operation_type":        operation_type,
                    "success_count":         1 if success else 0,
                    "failure_count":         0 if success else 1,
                    "timeout_count":         1 if timeout else 0,
                    "verify_success_count":  1 if verify_success else 0,
                    "avg_latency_ms":        latency_ms,
                    "last_success_at":       now if success else None,
                    "last_failure_at":       now if not success else None,
                    "last_seen_at":          now,
                    "stability_score":       1.0 if success else 0.0,
                    "semantic_match_score":  semantic_match_score,
                    "repair_generated_count": 0,
                    "repair_applied_count":   0,
                    "usage_count":           1,
                    "score":                 0.0,
                })
        _sls_txn(db.transaction(), ref)
    except Exception as e:
        print(f"[selector_learning_stats] 保存エラー: {type(e).__name__}", flush=True)


# ── P16.7: selector_transition_graph (将来P17用) ─────────────────────────

def update_selector_transition_graph(
    db,
    media_name: str,
    operation_type: str,
    prev_selector: str,
    next_selector: str,
    success: bool,
) -> None:
    """
    P16.7 (将来P17用): selectorの成功遷移を selector_transition_graph へ記録。
    どのselectorの後にどのselectorが成功しやすいかを蓄積する。
    自動適用禁止。参照専用データとして保存のみ行う。
    ID/PASS/secretは保存しない。

    Args:
        db:             Firestoreクライアント
        media_name:     媒体名
        operation_type: 操作種別
        prev_selector:  直前に試みたselector
        next_selector:  次に試みたselector
        success:        next_selectorが成功したか
    """
    import hashlib
    import datetime as _dt
    if not prev_selector or not next_selector:
        return
    try:
        prev_hash = hashlib.md5(prev_selector.encode()).hexdigest()[:12]
        next_hash = hashlib.md5(next_selector.encode()).hexdigest()[:12]
        doc_id = f"{media_name}__{operation_type}__{prev_hash}__{next_hash}"
        ref = db.collection("selector_transition_graph").document(doc_id)
        now = _dt.datetime.utcnow()
        doc = ref.get()
        if doc.exists:
            d = doc.to_dict()
            total       = d.get("transition_count", 0) + 1
            succ_count  = d.get("success_count", 0) + (1 if success else 0)
            ref.update({
                "transition_count": total,
                "success_count":    succ_count,
                "success_rate":     round(succ_count / total, 4),
                "last_seen_at":     now,
            })
        else:
            ref.set({
                "media_name":       media_name,
                "operation_type":   operation_type,
                "prev_selector":    prev_selector,
                "prev_hash":        prev_hash,
                "next_selector":    next_selector,
                "next_hash":        next_hash,
                "transition_count": 1,
                "success_count":    1 if success else 0,
                "success_rate":     1.0 if success else 0.0,
                "created_at":       now,
                "last_seen_at":     now,
            })
    except Exception as e:
        print(f"[selector_transition_graph] 保存エラー: {type(e).__name__}", flush=True)


# ── P16.7: ranking_model_version 管理 ────────────────────────────────────

def save_selector_ranking_result(
    db,
    media_name: str,
    operation_type: str,
    step_type: str,
    ranked: list,
    model_version: str = None,
) -> None:
    """
    P16.7: ranking結果を selector_rankings へ保存。
    ranking_model_version を明示し、旧versionとの比較を可能にする。
    自動適用禁止。保存のみ。dom_selectors更新禁止。
    ID/PASS/secretは保存しない。

    Args:
        db:             Firestoreクライアント
        media_name:     媒体名
        operation_type: 操作種別
        step_type:      ステップ種別
        ranked:         rank_selector_candidates()の返り値
        model_version:  使用したモデルバージョン（省略時はtrainerから取得）
    """
    import datetime as _dt
    if not ranked:
        return
    try:
        if model_version is None:
            from api.core.selector_rank_trainer import RANKING_MODEL_VERSION
            model_version = RANKING_MODEL_VERSION
        now = _dt.datetime.utcnow()
        top = ranked[0] if ranked else {}
        doc = {
            "media_name":            media_name,
            "operation_type":        operation_type,
            "step_type":             step_type,
            "ranking_model_version": model_version,
            "top_selector":          top.get("selector", ""),
            "top_score":             top.get("score", 0.0),
            "top_confidence":        top.get("confidence", 0.0),
            "candidate_count":       len(ranked),
            "ranked_summary": [
                {
                    "selector":        r.get("selector", ""),
                    "score":           r.get("score", 0.0),
                    "heuristic_score": r.get("heuristic_score", 0.0),
                    "learning_score":  r.get("learning_score", 0.0),
                    "model_score":     r.get("model_score", 0.0),
                    "confidence":      r.get("confidence", 0.0),
                    "reasons":         r.get("reasons", []),
                }
                for r in ranked[:5]  # 上位5件のみ保存
            ],
            "created_at": now,
        }
        db.collection("selector_rankings").add(doc)
    except Exception as e:
        print(f"[selector_rankings] 保存エラー: {type(e).__name__}", flush=True)


# ── P16.7: selector_training_logs 保存関数 ───────────────────────────────

def save_selector_training_logs(
    db,
    media_mapping: dict,
    operation_type: str,
    ranked_selectors: list,
    selected_selector: str = None,
) -> None:
    """
    P16.7: selector_training_logs へ候補全件を保存。
    selected=True が1件、残りはnegative_sample=True。
    自動適用禁止。dom_selectors変更禁止。学習用ログのみ。
    ID/PASS/secretは保存しない。
    """
    import hashlib
    import datetime as _dt
    from google.cloud import firestore
    try:
        from api.core.selector_rank_trainer import RANKING_MODEL_VERSION
    except Exception:
        RANKING_MODEL_VERSION = "p16_v1"

    if not ranked_selectors:
        return

    if selected_selector is None and ranked_selectors:
        selected_selector = ranked_selectors[0].get("selector", "")

    tenant_id  = media_mapping.get("tenant_id", "")
    mapping_id = media_mapping.get("mapping_id") or media_mapping.get("id", "")
    media_name = media_mapping.get("media_name", "")
    now        = _dt.datetime.utcnow()
    col        = db.collection("selector_training_logs")

    for item in ranked_selectors:
        selector = item.get("selector", "")
        if not selector:
            continue
        selector_hash = hashlib.md5(selector.encode()).hexdigest()[:12]
        is_selected   = (selector == selected_selector)
        try:
            col.add({
                "tenant_id":              tenant_id,
                "mapping_id":             mapping_id,
                "media_name":             media_name,
                "operation_type":         operation_type,
                "selector":               selector,
                "selector_hash":          selector_hash,
                "feature_vector":         item.get("feature_vector") or {},
                "heuristic_score":        item.get("heuristic_score", 0.0),
                "learning_score":         item.get("learning_score", 0.0),
                "model_score":            item.get("model_score", 0.0),
                "final_score":            item.get("score", 0.0),
                "confidence":             item.get("confidence", 0.5),
                "selected":               is_selected,
                "negative_sample":        not is_selected,
                "executed":               is_selected,
                "success":                None,
                "timeout":                False,
                "verify_success":         None,
                "latency_ms":             0,
                "ranking_model_version":  RANKING_MODEL_VERSION,
                "created_at":             now,
            })
        except Exception as e:
            print(f"[selector_training_logs] 保存エラー: {type(e).__name__}", flush=True)


# ── P17: operation chain memory ───────────────────────────────────────────

def build_chain_signature(operation_steps: list, operation_type: str = "") -> str:
    """
    P17: operation_stepsからchain signatureを生成。
    例: login->navigate->upload_image->submit
    operation_stepsが空の場合はoperation_typeのみ返す。
    """
    if not operation_steps:
        return operation_type or "unknown"
    sorted_steps = sorted(operation_steps, key=lambda s: s.get("order", 0))
    parts = []
    for s in sorted_steps:
        step_type = s.get("step_type") or s.get("type") or ""
        if step_type:
            parts.append(step_type)
    if not parts:
        return operation_type or "unknown"
    return "->".join(parts)


def update_operation_chain_memory(
    db,
    tenant_id: str,
    workflow_id: str,
    media_name: str,
    operation_type: str,
    operation_steps: list,
    success: bool,
    duration_ms: float = 0.0,
    retry_count: int = 0,
    selector_repair_count: int = 0,
) -> None:
    """
    P17: 成功/失敗したoperation chainをoperation_chain_memoryへ保存・更新。
    自動適用禁止。学習用メモリのみ。
    ID/PASS/secretは保存しない。

    document key: {tenant_id}__{workflow_hash}
    """
    import hashlib
    import datetime as _dt
    if not tenant_id or not operation_type:
        return
    try:
        chain_signature = build_chain_signature(operation_steps, operation_type)
        workflow_hash   = hashlib.md5(
            f"{tenant_id}__{media_name}__{chain_signature}".encode()
        ).hexdigest()[:16]
        doc_id = f"{tenant_id}__{workflow_hash}"
        now    = _dt.datetime.utcnow()
        ref    = db.collection("operation_chain_memory").document(doc_id)
        doc    = ref.get()

        # transition_stats更新（step間の遷移確率）
        transition_stats = {}
        if operation_steps and len(operation_steps) >= 2:
            sorted_steps = sorted(operation_steps, key=lambda s: s.get("order", 0))
            for i in range(len(sorted_steps) - 1):
                a = sorted_steps[i].get("step_type") or sorted_steps[i].get("type") or ""
                b = sorted_steps[i+1].get("step_type") or sorted_steps[i+1].get("type") or ""
                if a and b:
                    key = f"{a}->{b}"
                    transition_stats[key] = {"success": 0, "failed": 0, "score": 0.0}

        if doc.exists:
            d = doc.to_dict()
            succ  = d.get("success_count", 0) + (1 if success else 0)
            fail  = d.get("failure_count", 0) + (0 if success else 1)
            total = succ + fail or 1
            usage = d.get("avg_step_count", 0)
            new_avg_dur = (
                d.get("avg_duration_ms", 0.0) * (total - 1) + duration_ms
            ) / total
            new_avg_steps = (
                usage * (total - 1) + len(operation_steps or [])
            ) / total

            # transition_stats マージ
            existing_ts = d.get("transition_stats") or {}
            for k, v in transition_stats.items():
                if k not in existing_ts:
                    existing_ts[k] = {"success": 0, "failed": 0, "score": 0.0}
                existing_ts[k]["success"] += (1 if success else 0)
                existing_ts[k]["failed"]  += (0 if success else 1)
                _ts_total = existing_ts[k]["success"] + existing_ts[k]["failed"] or 1
                existing_ts[k]["score"] = round(
                    existing_ts[k]["success"] / _ts_total, 4
                )

            ref.update({
                "success_count":      succ,
                "failure_count":      fail,
                "avg_duration_ms":    round(new_avg_dur, 2),
                "avg_step_count":     round(new_avg_steps, 2),
                "transition_stats":   existing_ts,
                "last_success_at":    now if success else d.get("last_success_at"),
                "last_failure_at":    now if not success else d.get("last_failure_at"),
                "updated_at":         now,
            })
        else:
            # 新規作成
            ts_init = {}
            for k in transition_stats:
                ts_init[k] = {
                    "success": 1 if success else 0,
                    "failed":  0 if success else 1,
                    "score":   1.0 if success else 0.0,
                }
            ref.set({
                "tenant_id":         tenant_id,
                "workflow_id":       workflow_id,
                "workflow_hash":     workflow_hash,
                "media_name":        media_name,
                "operation_type":    operation_type,
                "chain_signature":   chain_signature,
                "steps":             [
                    {"step_type": s.get("step_type") or s.get("type", ""),
                     "order":     s.get("order", i)}
                    for i, s in enumerate(operation_steps or [])
                ],
                "success_count":     1 if success else 0,
                "failure_count":     0 if success else 1,
                "avg_duration_ms":   duration_ms,
                "avg_step_count":    float(len(operation_steps or [])),
                "last_success_at":   now if success else None,
                "last_failure_at":   now if not success else None,
                "transition_stats":  ts_init,
                "retry_count_total": retry_count,
                "repair_count_total": selector_repair_count,
                "created_at":        now,
                "updated_at":        now,
            })
    except Exception as e:
        print(f"[operation_chain_memory] 保存エラー: {type(e).__name__}", flush=True)


def find_similar_workflows(
    db,
    tenant_id: str,
    operation_type: str,
    media_name: str = "",
    top_n: int = 5,
) -> list:
    """
    P17: 類似workflow検索。過去成功率の高いchainを返す。
    自動実行禁止。提案のみ。
    Firestore制約: whereはtenant_idのみ、ソートはPython側。

    Returns:
        [
            {
                "workflow_hash": str,
                "chain_signature": str,
                "success_rate": float,
                "avg_duration_ms": float,
                "avg_step_count": float,
                "success_count": int,
                "score": float,
                "steps": list,
                "transition_stats": dict,
            },
            ...
        ]
    """
    import datetime as _dt
    import math
    try:
        docs = (
            db.collection("operation_chain_memory")
            .where("tenant_id", "==", tenant_id)
            .stream()
        )
        candidates = []
        now = _dt.datetime.utcnow()
        for d in docs:
            w = d.to_dict()
            # operation_typeフィルタ（Python側）
            if w.get("operation_type") != operation_type:
                continue
            total = (w.get("success_count", 0) + w.get("failure_count", 0)) or 1
            succ  = w.get("success_count", 0)
            success_rate = succ / total

            # recency score（最終成功から経過日数で減衰）
            last_succ = w.get("last_success_at")
            if last_succ:
                try:
                    days = (now - last_succ.replace(tzinfo=None)).days
                    recency = math.exp(-days / 30.0)
                except Exception:
                    recency = 0.5
            else:
                recency = 0.0

            # duration score（短いほど高い）
            avg_dur = w.get("avg_duration_ms", 5000.0)
            duration_score = max(0.0, 1.0 - avg_dur / 30000.0)

            # retry/repair penalty
            retry_penalty  = min(0.20, w.get("retry_count_total", 0) * 0.02)
            repair_penalty = min(0.20, w.get("repair_count_total", 0) * 0.02)

            # 総合score
            score = (
                success_rate  * 0.50
                + recency     * 0.25
                + duration_score * 0.15
                - retry_penalty
                - repair_penalty
            )
            score = max(0.0, min(1.0, round(score, 4)))

            candidates.append({
                "workflow_hash":    w.get("workflow_hash", ""),
                "chain_signature":  w.get("chain_signature", ""),
                "media_name":       w.get("media_name", ""),
                "success_rate":     round(success_rate, 4),
                "avg_duration_ms":  w.get("avg_duration_ms", 0.0),
                "avg_step_count":   w.get("avg_step_count", 0.0),
                "success_count":    succ,
                "score":            score,
                "steps":            w.get("steps", []),
                "transition_stats": w.get("transition_stats", {}),
            })

        # Python側ソート（Firestore複合index不使用）
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
    except Exception as e:
        print(f"[find_similar_workflows] 検索エラー: {type(e).__name__}", flush=True)
        return []


# ── P18: cross-media reusable templates ──────────────────────────────────

def build_template_signature(capabilities: dict, operation_steps: list = None) -> str:
    """
    P18: template signatureを生成。
    例: upload_image+update_text+verify
    capabilities keysとoperation_steps step_typeを合成。
    """
    parts = set()
    if capabilities:
        for k, v in capabilities.items():
            if v is True or v == "true":
                parts.add(k)
    if operation_steps:
        for s in operation_steps:
            st = s.get("step_type") or s.get("type") or ""
            if st:
                parts.add(st)
    if not parts:
        return "unknown"
    return "+".join(sorted(parts))


def update_cross_media_template(
    db,
    tenant_id: str,
    media_name: str,
    media_family: str,
    industry: str,
    operation_type: str,
    capabilities: dict,
    operation_steps: list,
    dom_selectors: dict,
    success: bool,
    duration_ms: float = 0.0,
    repair_count: int = 0,
) -> None:
    """
    P18: P17成功workflow完了後にcross_media_templatesへ保存・更新。
    自動適用禁止。保存・提案のみ。dom_selectors直接更新禁止。
    ID/PASS/secretは保存しない。
    """
    import hashlib
    import datetime as _dt
    if not operation_type:
        return
    try:
        template_signature = build_template_signature(capabilities, operation_steps)
        sig_src = f"{industry}__{media_family}__{operation_type}__{template_signature}"
        template_id = hashlib.md5(sig_src.encode()).hexdigest()[:16]
        now = _dt.datetime.utcnow()

        # selector pattern抽象化（固定selectorではなくtype/pattern/semantic_labelで保存）
        selector_patterns = []
        for label, selector in (dom_selectors or {}).items():
            if not selector:
                continue
            sel_lower = selector.lower()
            if selector.startswith("#"):
                pattern = "#id"
            elif "[name=" in sel_lower:
                pattern = "[name=*]"
            elif "[aria-label=" in sel_lower:
                pattern = "[aria-label=*]"
            elif "[type=" in sel_lower:
                import re
                m = re.search(r'\[type=["\']?(\w+)', sel_lower)
                pattern = f"[type={m.group(1)}]" if m else selector
            else:
                pattern = selector
            selector_patterns.append({
                "type":           label,
                "pattern":        pattern,
                "semantic_label": label,
                "example":        selector,
            })

        ref = db.collection("agent_templates").document(template_id)
        doc = ref.get()
        if doc.exists:
            d = doc.to_dict()
            succ  = d.get("success_count", 0) + (1 if success else 0)
            fail  = d.get("failure_count", 0) + (0 if success else 1)
            total = succ + fail or 1
            src_names = list(set(d.get("source_media_names", []) + ([media_name] if media_name else [])))
            ref.update({
                "success_count":      succ,
                "failure_count":      fail,
                "success_rate":       round(succ / total, 4),
                "source_media_names": src_names,
                "selector_patterns":  selector_patterns,
                "updated_at":         now,
            })
        else:
            ref.set({
                "template_id":        template_id,
                "template_signature": template_signature,
                "template_name":      f"{industry}_{operation_type}_{media_family}",
                "industry":           industry,
                "media_family":       media_family,
                "operation_type":     operation_type,
                "capabilities":       capabilities or {},
                "operation_steps":    [
                    {"step_type": s.get("step_type") or s.get("type", ""),
                     "order":     s.get("order", i)}
                    for i, s in enumerate(operation_steps or [])
                ],
                "selector_patterns":  selector_patterns,
                "payload_schema":     {},
                "success_count":      1 if success else 0,
                "failure_count":      0 if success else 1,
                "success_rate":       1.0 if success else 0.0,
                "source_media_names": [media_name] if media_name else [],
                "compatible_media":   [],
                "repair_rate":        round(repair_count / 1, 4) if repair_count else 0.0,
                "avg_duration_ms":    duration_ms,
                "created_at":         now,
                "updated_at":         now,
            })
    except Exception as e:
        print(f"[agent_templates] 保存エラー: {type(e).__name__}", flush=True)


def find_reusable_templates(
    db,
    tenant_id: str,
    operation_type: str,
    industry: str = "",
    media_family: str = "",
    capabilities: dict = None,
    top_n: int = 5,
) -> list:
    """
    P18: 新媒体mapping時に再利用可能なtemplateを提案。
    自動適用禁止。提案のみ。
    Firestore制約: whereはoperation_typeのみ、ソートはPython側。

    Returns:
        [
            {
                "template_id": str,
                "template_name": str,
                "template_signature": str,
                "industry": str,
                "media_family": str,
                "success_rate": float,
                "score": float,
                "selector_patterns": list,
                "operation_steps": list,
                "capabilities": dict,
                "source_media_names": list,
                "missing_selectors": list,
            },
            ...
        ]
    """
    import math
    import datetime as _dt
    try:
        # agent_templates優先読み込み、fallbackにcross_media_templates
        _at_docs = list(db.collection("agent_templates").where("operation_type", "==", operation_type).stream())
        _cm_docs = list(db.collection("cross_media_templates").where("operation_type", "==", operation_type).stream()) if not _at_docs else []
        docs = _at_docs if _at_docs else _cm_docs
        candidates = []
        now = _dt.datetime.utcnow()
        cap_keys = set((capabilities or {}).keys())

        for d in docs:
            t = d.to_dict()
            total = (t.get("success_count", 0) + t.get("failure_count", 0)) or 1
            succ  = t.get("success_count", 0)
            success_rate = succ / total

            # recency
            updated = t.get("updated_at")
            if updated:
                try:
                    days = (now - updated.replace(tzinfo=None)).days
                    recency = math.exp(-days / 60.0)
                except Exception:
                    recency = 0.5
            else:
                recency = 0.3

            # media_similarity（industry + media_family）
            media_sim = 0.0
            if industry and t.get("industry") == industry:
                media_sim += 0.5
            if media_family and t.get("media_family") == media_family:
                media_sim += 0.5

            # capability_match
            tmpl_caps = set((t.get("capabilities") or {}).keys())
            if cap_keys and tmpl_caps:
                cap_match = len(cap_keys & tmpl_caps) / max(len(cap_keys), len(tmpl_caps))
            else:
                cap_match = 0.5

            # repair/duration penalty
            repair_penalty  = min(0.20, t.get("repair_rate", 0.0) * 2)
            avg_dur         = t.get("avg_duration_ms", 5000.0)
            duration_score  = max(0.0, 1.0 - avg_dur / 30000.0)

            # 総合score
            score = (
                success_rate   * 0.35
                + recency      * 0.15
                + media_sim    * 0.25
                + cap_match    * 0.15
                + duration_score * 0.10
                - repair_penalty
            )
            score = max(0.0, min(1.0, round(score, 4)))

            # 不足selector算出（template inheritanceのため）
            tmpl_patterns  = t.get("selector_patterns") or []
            tmpl_types     = {p.get("type", "") for p in tmpl_patterns}
            missing_selectors = [
                p for p in tmpl_patterns
                if not p.get("example")
            ]

            candidates.append({
                "template_id":        t.get("template_id", ""),
                "template_name":      t.get("template_name", ""),
                "template_signature": t.get("template_signature", ""),
                "industry":           t.get("industry", ""),
                "media_family":       t.get("media_family", ""),
                "success_rate":       round(success_rate, 4),
                "score":              score,
                "selector_patterns":  tmpl_patterns,
                "operation_steps":    t.get("operation_steps", []),
                "capabilities":       t.get("capabilities", {}),
                "source_media_names": t.get("source_media_names", []),
                "missing_selectors":  missing_selectors,
                "media_similarity":   media_sim,
                "capability_match":   round(cap_match, 4),
            })

        # Python側ソート
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
    except Exception as e:
        print(f"[find_reusable_templates] 検索エラー: {type(e).__name__}", flush=True)
        return []


# ── P19: failure pattern clustering ──────────────────────────────────────

# failure taxonomy定義
FAILURE_TAXONOMY = {
    "timeout":          "selector_timeout",
    "TimeoutError":     "selector_timeout",
    "PlaywrightTimeout":"selector_timeout",
    "login":            "login_failure",
    "LOGIN_FAILED":     "login_failure",
    "verify":           "verify_failure",
    "verification":     "verify_failure",
    "upload":           "upload_failure",
    "file":             "upload_failure",
    "dom":              "dom_changed",
    "DOM":              "dom_changed",
    "permission":       "permission_denied",
    "403":              "permission_denied",
    "rate":             "rate_limited",
    "429":              "rate_limited",
    "RuntimeError":     "unknown_runtime",
    "Exception":        "unknown_runtime",
}

def _classify_failure_taxonomy(error_type: str, error_msg: str = "") -> str:
    """error_type / error_msgからfailure taxonomyを分類する。"""
    combined = f"{error_type} {error_msg}"
    for keyword, taxonomy in FAILURE_TAXONOMY.items():
        if keyword.lower() in combined.lower():
            return taxonomy
    return "unknown_runtime"


def build_failure_signature(
    error_type: str,
    operation_type: str,
    media_family: str = "",
    failed_selectors: list = None,
    rollback_reason: str = "",
    self_heal_status: str = "",
) -> str:
    """
    P19: failure signatureを生成。
    例: selector_timeout__upload_image__nightlife_portal
    """
    taxonomy = _classify_failure_taxonomy(error_type, rollback_reason)
    parts = [taxonomy, operation_type or "unknown"]
    if media_family:
        parts.append(media_family)
    return "__".join(parts)


def update_failure_pattern_cluster(
    db,
    tenant_id: str,
    error_type: str,
    error_msg: str,
    operation_type: str,
    media_name: str,
    media_family: str,
    failed_selectors: list,
    rollback_reason: str = "",
    self_heal_status: str = "",
    repair_type: str = "",
    repair_selector: str = "",
    repair_success: bool = False,
) -> None:
    """
    P19: FAILED時にfailure_pattern_clustersへ保存・更新。
    自動修復禁止。clustering + 提案のみ。
    ID/PASS/secretは保存しない。
    """
    import hashlib
    import datetime as _dt
    if not operation_type:
        return
    try:
        failure_signature = build_failure_signature(
            error_type, operation_type, media_family,
            failed_selectors, rollback_reason, self_heal_status,
        )
        taxonomy   = _classify_failure_taxonomy(error_type, error_msg)
        cluster_id = hashlib.md5(
            f"{tenant_id}__{failure_signature}".encode()
        ).hexdigest()[:16]
        now = _dt.datetime.utcnow()
        ref = db.collection("failure_pattern_clusters").document(cluster_id)
        doc = ref.get()

        # repair pattern
        repair_entry = {}
        if repair_type:
            repair_entry = {
                "repair_type":     repair_type,
                "repair_selector": repair_selector,
                "repair_success":  repair_success,
                "recorded_at":     now,
            }

        if doc.exists:
            d = doc.to_dict()
            recurring = d.get("recurring_count", 0) + 1
            is_recurring = recurring >= 3

            # sample_errors（最大10件）
            samples = d.get("sample_errors", [])
            if len(samples) < 10:
                samples.append({
                    "error_type": error_type,
                    "error_msg":  error_msg[:200] if error_msg else "",
                    "media_name": media_name,
                    "at":         now,
                })

            # repair_patterns蓄積
            repair_patterns = d.get("repair_patterns", [])
            if repair_entry:
                repair_patterns.append(repair_entry)
                repair_patterns = repair_patterns[-20:]  # 最大20件

            succ_repair = d.get("success_repair_count", 0) + (1 if repair_success else 0)
            fail_repair = d.get("failed_repair_count", 0) + (0 if repair_success else (1 if repair_type else 0))

            # selector_patterns蓄積
            sel_patterns = d.get("selector_patterns", [])
            for sel in (failed_selectors or []):
                if sel and sel not in sel_patterns:
                    sel_patterns.append(sel)
            sel_patterns = sel_patterns[-30:]

            # media spread
            media_spread = d.get("media_spread", [])
            if media_name and media_name not in media_spread:
                media_spread.append(media_name)

            ref.update({
                "recurring_count":      recurring,
                "recurring_failure":    is_recurring,
                "sample_errors":        samples,
                "repair_patterns":      repair_patterns,
                "success_repair_count": succ_repair,
                "failed_repair_count":  fail_repair,
                "selector_patterns":    sel_patterns,
                "media_spread":         media_spread,
                "last_seen_at":         now,
                "updated_at":           now,
            })
        else:
            ref.set({
                "cluster_id":           cluster_id,
                "failure_signature":    failure_signature,
                "failure_taxonomy":     taxonomy,
                "failure_group":        taxonomy.split("_")[0] if taxonomy else "unknown",
                "error_type":           error_type,
                "operation_type":       operation_type,
                "media_family":         media_family,
                "tenant_id":            tenant_id,
                "sample_errors":        [{
                    "error_type": error_type,
                    "error_msg":  error_msg[:200] if error_msg else "",
                    "media_name": media_name,
                    "at":         now,
                }],
                "selector_patterns":    failed_selectors or [],
                "repair_patterns":      [repair_entry] if repair_entry else [],
                "success_repair_count": 1 if repair_success else 0,
                "failed_repair_count":  0 if repair_success else (1 if repair_type else 0),
                "recurring_count":      1,
                "recurring_failure":    False,
                "media_spread":         [media_name] if media_name else [],
                "last_seen_at":         now,
                "created_at":           now,
                "updated_at":           now,
            })
    except Exception as e:
        print(f"[failure_pattern_clusters] 保存エラー: {type(e).__name__}", flush=True)


def find_similar_failures(
    db,
    tenant_id: str,
    error_type: str,
    operation_type: str,
    media_family: str = "",
    top_n: int = 5,
) -> list:
    """
    P19: 類似failure clusterを検索し、過去成功repairパターンを返す。
    self_heal前に参照し、repair候補へ加点するために使用。
    自動適用禁止。提案のみ。
    Firestore制約: whereはtenant_idのみ、ソートはPython側。

    Returns:
        [
            {
                "cluster_id": str,
                "failure_signature": str,
                "failure_taxonomy": str,
                "success_repair_count": int,
                "repair_success_rate": float,
                "recurring_count": int,
                "recurring_failure": bool,
                "repair_patterns": list,
                "selector_patterns": list,
                "score": float,
            },
            ...
        ]
    """
    import datetime as _dt
    import math
    try:
        docs = (
            db.collection("failure_pattern_clusters")
            .where("tenant_id", "==", tenant_id)
            .stream()
        )
        taxonomy = _classify_failure_taxonomy(error_type, "")
        candidates = []
        now = _dt.datetime.utcnow()

        for d in docs:
            c = d.to_dict()
            # Python側フィルタ
            if c.get("operation_type") != operation_type:
                continue
            if c.get("failure_taxonomy") != taxonomy:
                continue

            succ_r = c.get("success_repair_count", 0)
            fail_r = c.get("failed_repair_count", 0)
            total_r = succ_r + fail_r or 1
            repair_rate = succ_r / total_r

            # recency
            last_seen = c.get("last_seen_at")
            if last_seen:
                try:
                    days = (now - last_seen.replace(tzinfo=None)).days
                    recency = math.exp(-days / 30.0)
                except Exception:
                    recency = 0.5
            else:
                recency = 0.3

            # spread（複数媒体で発生していると深刻）
            spread = len(c.get("media_spread", []))
            spread_score = min(1.0, spread / 5.0)

            # severity（recurring_countが多いほど深刻）
            severity = min(1.0, c.get("recurring_count", 1) / 10.0)

            # cluster score（repair成功率が高いほど有用）
            score = (
                repair_rate  * 0.40
                + recency    * 0.20
                + severity   * 0.20
                + spread_score * 0.20
            )
            score = max(0.0, min(1.0, round(score, 4)))

            candidates.append({
                "cluster_id":           c.get("cluster_id", ""),
                "failure_signature":    c.get("failure_signature", ""),
                "failure_taxonomy":     c.get("failure_taxonomy", ""),
                "success_repair_count": succ_r,
                "repair_success_rate":  round(repair_rate, 4),
                "recurring_count":      c.get("recurring_count", 0),
                "recurring_failure":    c.get("recurring_failure", False),
                "repair_patterns":      c.get("repair_patterns", []),
                "selector_patterns":    c.get("selector_patterns", []),
                "score":                score,
            })

        # Python側ソート
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
    except Exception as e:
        print(f"[find_similar_failures] 検索エラー: {type(e).__name__}", flush=True)
        return []


# ── P20: semi-autonomous workflow orchestration ───────────────────────────

# risk level定義
RISK_LEVEL_LOW    = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH   = "HIGH"

def estimate_workflow_risk(
    db,
    tenant_id: str,
    operation_type: str,
    media_family: str,
    operation_steps: list,
    media_mapping: dict,
) -> dict:
    """
    P20: workflow実行前にrisk levelを推定する。
    risk_level: LOW / MEDIUM / HIGH
    HIGH riskは必ず承認必須。LOW/MEDIUMのみ半自律実行許可。
    自動実行禁止。

    Returns:
        {
            "risk_level": str,
            "risk_score": float,
            "risk_factors": list,
            "require_human_approval": bool,
        }
    """
    risk_score   = 0.0
    risk_factors = []

    # 1. operation_type固有リスク
    op_risk_map = {
        "media_replace":   0.7,
        "price_update":    0.6,
        "entity_register": 0.4,
        "entity_update":   0.4,
        "news_post":       0.3,
        "status_update":   0.3,
        "text_update":     0.2,
        "schedule_update": 0.2,
    }
    op_risk = op_risk_map.get(operation_type, 0.3)
    risk_score += op_risk
    if op_risk >= 0.6:
        risk_factors.append(f"高リスク操作: {operation_type}")

    # 2. workflow complexity（step数）
    step_count = len(operation_steps or [])
    if step_count >= 5:
        risk_score += 0.2
        risk_factors.append(f"複雑なworkflow: {step_count}ステップ")
    elif step_count >= 3:
        risk_score += 0.1

    # 3. selector instability（selector_learning_statsから）
    try:
        dom_selectors = (media_mapping or {}).get("dom_selectors") or {}
        media_name    = (media_mapping or {}).get("media_name", "")
        unstable_count = 0
        for label, selector in dom_selectors.items():
            if not selector:
                continue
            import hashlib as _hl
            sel_hash = _hl.md5(selector.encode()).hexdigest()[:12]
            doc_id   = f"{media_name}__{operation_type}__{sel_hash}"
            stats_doc = db.collection("selector_learning_stats").document(doc_id).get()
            if stats_doc.exists:
                s = stats_doc.to_dict()
                total = (s.get("success_count", 0) + s.get("failure_count", 0)) or 1
                stab  = s.get("success_count", 0) / total
                if stab < 0.5:
                    unstable_count += 1
        if unstable_count >= 2:
            risk_score += 0.3
            risk_factors.append(f"不安定なselector: {unstable_count}件")
        elif unstable_count == 1:
            risk_score += 0.1
            risk_factors.append("不安定なselector: 1件")
    except Exception:
        pass

    # 4. recurring failures
    try:
        cluster_docs = (
            db.collection("failure_pattern_clusters")
            .where("tenant_id", "==", tenant_id)
            .stream()
        )
        recurring = [
            d.to_dict() for d in cluster_docs
            if d.to_dict().get("operation_type") == operation_type
            and d.to_dict().get("recurring_failure") is True
        ]
        if recurring:
            risk_score += 0.2
            risk_factors.append(f"繰り返し失敗パターン検出: {len(recurring)}件")
    except Exception:
        pass

    # 5. cross-media uncertainty（media_familyが未設定）
    if not media_family:
        risk_score += 0.1
        risk_factors.append("media_family未設定（cross-media不確実性）")

    # risk_level判定
    risk_score = min(1.0, round(risk_score, 4))
    if risk_score >= 0.7:
        risk_level = RISK_LEVEL_HIGH
    elif risk_score >= 0.4:
        risk_level = RISK_LEVEL_MEDIUM
    else:
        risk_level = RISK_LEVEL_LOW

    return {
        "risk_level":              risk_level,
        "risk_score":              risk_score,
        "risk_factors":            risk_factors,
        "require_human_approval":  risk_level == RISK_LEVEL_HIGH,
    }


def build_execution_graph(
    operation_type: str,
    operation_steps: list,
    execution_policy: dict,
) -> dict:
    """
    P20: execution graphを生成。条件分岐を含む。
    on_failure / on_verify_failed / on_timeout を各stepに付与。
    """
    allow_self_heal    = (execution_policy or {}).get("allow_self_heal", True)
    allow_replan       = (execution_policy or {}).get("allow_replan", True)
    max_retry          = (execution_policy or {}).get("max_retry", 2)

    steps = []
    for i, s in enumerate(operation_steps or []):
        step = {
            "step_id":   s.get("step_id", f"step_{i}"),
            "step_type": s.get("step_type") or s.get("type", ""),
            "order":     s.get("order", i),
            "on_failure":      "self_heal" if allow_self_heal else "fail",
            "on_verify_failed":"self_heal" if allow_self_heal else "fail",
            "on_timeout":      "retry"     if max_retry > 0   else "fail",
            "on_blocked":      "wait_human",
        }
        steps.append(step)

    return {
        "steps":            steps,
        "allow_self_heal":  allow_self_heal,
        "allow_replan":     allow_replan,
        "max_retry":        max_retry,
        "on_critical_fail": "replan" if allow_replan else "abort",
    }


def rebuild_execution_plan(
    db,
    session_id: str,
    tenant_id: str,
    operation_type: str,
    operation_steps: list,
    failed_step: str,
    failure_reason: str,
    execution_policy: dict,
) -> dict:
    """
    P20: FAILED時にworkflow再構築(adaptive replanning)。
    失敗したstepをスキップ or 代替stepを挿入。
    自動実行禁止。再計画のみ提案。

    Returns:
        {
            "replanned": bool,
            "new_steps": list,
            "replan_reason": str,
            "branch_taken": str,
        }
    """
    import datetime as _dt
    new_steps    = []
    branch_taken = "none"
    replan_reason = f"step '{failed_step}' 失敗: {failure_reason}"

    for s in (operation_steps or []):
        st = s.get("step_type") or s.get("type", "")
        if st == failed_step:
            # timeout系は再試行stepを挿入
            if "timeout" in failure_reason.lower():
                retry_step = dict(s)
                retry_step["step_id"]   = f"{s.get('step_id','step')}_retry"
                retry_step["is_retry"]  = True
                new_steps.append(retry_step)
                branch_taken = "retry_inserted"
            else:
                # それ以外はskip（verify stepはkeep）
                if "verify" in st:
                    new_steps.append(s)
                else:
                    branch_taken = "step_skipped"
        else:
            new_steps.append(s)

    # Firestore: adaptive_branch_historyへ記録
    try:
        now = _dt.datetime.utcnow()
        ref = db.collection("workflow_execution_sessions").document(session_id)
        doc = ref.get()
        if doc.exists:
            history = doc.to_dict().get("adaptive_branch_history", [])
            history.append({
                "failed_step":    failed_step,
                "failure_reason": failure_reason,
                "branch_taken":   branch_taken,
                "replan_reason":  replan_reason,
                "at":             now,
            })
            ref.update({
                "adaptive_branch_history": history,
                "updated_at":              now,
            })
    except Exception as e:
        print(f"[rebuild_execution_plan] 履歴保存エラー: {type(e).__name__}", flush=True)

    return {
        "replanned":     len(new_steps) != len(operation_steps or []) or branch_taken != "none",
        "new_steps":     new_steps,
        "replan_reason": replan_reason,
        "branch_taken":  branch_taken,
    }


def create_workflow_session(
    db,
    tenant_id: str,
    workflow_id: str,
    goal: str,
    operation_type: str,
    operation_steps: list,
    execution_policy: dict,
    risk_estimation: dict,
) -> str:
    """
    P20: workflow_execution_sessionsに新規セッションを作成。
    approval_state: HIGH riskはWAITING_APPROVAL、それ以外はAPPROVED。
    返り値: session_id
    """
    import uuid as _uuid
    import datetime as _dt
    session_id = str(_uuid.uuid4())
    now        = _dt.datetime.utcnow()
    risk_level = risk_estimation.get("risk_level", RISK_LEVEL_MEDIUM)

    approval_state = (
        "WAITING_APPROVAL"
        if risk_level == RISK_LEVEL_HIGH
        else "APPROVED"
    )

    execution_graph = build_execution_graph(
        operation_type, operation_steps, execution_policy
    )

    policy = execution_policy or {}
    db.collection("workflow_execution_sessions").document(session_id).set({
        "session_id":              session_id,
        "workflow_id":             workflow_id,
        "tenant_id":               tenant_id,
        "goal":                    goal,
        "execution_policy":        policy,
        "approval_state":          approval_state,
        "current_phase":           "planning",
        "current_step":            "",
        "execution_graph":         execution_graph,
        "risk_estimation":         risk_estimation,
        "adaptive_branch_history": [],
        "interruptible":           policy.get("interruptible", True),
        "paused":                  False,
        "cancelled":               False,
        "status":                  "PENDING" if approval_state == "WAITING_APPROVAL" else "READY",
        "created_at":              now,
        "updated_at":              now,
    })
    return session_id


def check_workflow_approval(
    db,
    session_id: str,
    tenant_id: str = "",
) -> dict:
    """
    P20: browser_executor実行前にapproval_stateを確認。
    未承認ならBLOCKED。
    返り値: {"approved": bool, "approval_state": str, "paused": bool, "cancelled": bool}
    """
    try:
        doc = db.collection("workflow_execution_sessions").document(session_id).get()
        if not doc.exists:
            return {"approved": False, "approval_state": "NOT_FOUND", "paused": False, "cancelled": False}
        d = doc.to_dict()
        if tenant_id and d.get("tenant_id") != tenant_id:
            return {"approved": False, "approval_state": "TENANT_MISMATCH", "paused": False, "cancelled": False}
        if d.get("cancelled"):
            return {"approved": False, "approval_state": "CANCELLED", "paused": False, "cancelled": True}
        if d.get("paused"):
            return {"approved": False, "approval_state": "PAUSED", "paused": True, "cancelled": False}
        state = d.get("approval_state", "WAITING_APPROVAL")
        return {
            "approved":       state == "APPROVED",
            "approval_state": state,
            "paused":         False,
            "cancelled":      False,
        }
    except Exception as e:
        print(f"[check_workflow_approval] エラー: {type(e).__name__}", flush=True)
        return {"approved": False, "approval_state": "ERROR", "paused": False, "cancelled": False}


# ── P21: Post Login Admin Crawler ────────────────────────────────────────


def _score_operation_dom_evidence(operation_type: str, labels: dict, confidence: dict, target_url: str = "") -> dict:
    """
    修正3: DOM証拠スコア。URLではなくDOM要素の存在で判定する。
    """
    score = 0
    matched_roles = []
    negative_reasons = []
    url_lower = (target_url or "").lower()
    # ネガティブガード: price/fee/course/pricelist系URLはentity_updateから除外
    _price_url_keys = ("price", "fee", "course", "pricelist", "料金", "料金表", "systemlist", "multifee")
    _list_url_keys  = ("cast_list", "readlog", "review_list", "/list")
    if operation_type == "entity_update":
        if any(k in url_lower for k in _price_url_keys):
            negative_reasons.append("price_page_excluded_for_entity_update")
            return {"score": 0, "matched_roles": [], "negative_reasons": negative_reasons}
        if any(k in url_lower for k in _list_url_keys):
            negative_reasons.append("list_page_excluded_for_entity_update")
            return {"score": 0, "matched_roles": [], "negative_reasons": negative_reasons}
    # Operation別必須DOM証拠
    _required_evidence = {
        "news_post":       ["title", "body", "submit"],
        "text_update":     ["body", "submit"],
        "media_replace":   ["image_upload", "submit"],
        "schedule_update": ["date_input", "submit"],
        "price_update":    ["price", "submit"],
        "entity_register": ["submit"],
        "entity_update":   ["editable_inputs", "submit"],
        "status_update":   ["submit"],
    }
    required = _required_evidence.get(operation_type, ["submit"])
    for role in required:
        if labels.get(role):
            score += 2 if confidence.get(role) == "high" else 1
            matched_roles.append(role)
        else:
            negative_reasons.append(f"missing_{role}")
    # price_update: price_inputなし・saveのみはスコア0
    if operation_type == "price_update" and "price" not in matched_roles:
        negative_reasons.append("price_input_missing_price_update_rejected")
        score = 0
    print(
        f"[P24_DOM_EVIDENCE] op={operation_type} target_url={target_url}"
        f" score={score} matched_roles={matched_roles} negative_reasons={negative_reasons}",
        flush=True
    )
    return {"score": score, "matched_roles": matched_roles, "negative_reasons": negative_reasons}

def _schema_safe_key(value: str) -> str:
    import re as _re_schema_key
    key = _re_schema_key.sub(r"[^0-9a-zA-Z_]+", "_", str(value or "").strip().lower())
    return key.strip("_")[:48] or "field"


def _infer_schema_entity(page: dict, form_schema: dict, fields: list[dict]) -> str:
    blob = " ".join([
        str(page.get("url") or ""),
        str(page.get("title") or page.get("html_title") or ""),
        str(page.get("page_purpose") or ""),
        str(form_schema.get("title") or ""),
        " ".join(str(f.get("canonical") or "") for f in fields[:40]),
        " ".join(str(f.get("label") or f.get("section") or f.get("name") or "") for f in fields[:40]),
    ]).lower()
    if any(k in blob for k in ("profile", "cast", "girl", "staff", "member", "プロフィール", "キャスト", "女の子", "スタッフ")):
        return "profile"
    if any(k in blob for k in ("schedule", "shift", "calendar", "出勤", "予定", "シフト")):
        return "schedule"
    if any(k in blob for k in ("price", "fee", "course", "料金", "コース", "金額")):
        return "price"
    if any(k in blob for k in ("image", "photo", "media", "upload", "画像", "写真", "動画", "ファイル")):
        return "media"
    if any(k in blob for k in ("news", "blog", "diary", "post", "topic", "ニュース", "お知らせ", "投稿", "日記")):
        return "content"
    if any(k in blob for k in ("status", "public", "private", "visible", "hidden", "公開", "非公開", "表示", "状態")):
        return "status"
    return "entity"


def _normalize_schema_canonical(field: dict, default_entity: str) -> str:
    raw = str(field.get("canonical") or "").strip()
    name = str(field.get("name") or field.get("id") or "").strip()
    label = str(field.get("label") or field.get("section") or "").strip()
    typ = str(field.get("type") or field.get("tag") or "").lower()
    blob = f"{raw} {name} {label}".lower()

    if raw and raw != "profile.custom":
        return raw
    if typ == "file" or any(k in blob for k in ("image", "photo", "画像", "写真", "ファイル", "upload")):
        return "media.file"
    if any(k in blob for k in ("title", "subject", "headline", "タイトル", "件名", "見出し")):
        return "content.title"
    if any(k in blob for k in ("body", "content", "text", "comment", "message", "本文", "内容", "紹介", "説明")):
        return "content.body"
    if any(k in blob for k in ("date", "day", "日付", "年月日")):
        return "schedule.date"
    if any(k in blob for k in ("start", "from", "開始", "出勤", "open")):
        return "schedule.start_time"
    if any(k in blob for k in ("end", "to", "終了", "退勤", "close")):
        return "schedule.end_time"
    if any(k in blob for k in ("price", "fee", "amount", "cost", "料金", "価格", "金額", "コース")):
        return "price.amount"
    if any(k in blob for k in ("status", "state", "public", "private", "display", "visible", "公開", "非公開", "表示", "状態")):
        return "status.visibility"
    if any(k in blob for k in ("name", "名前", "名称", "氏名", "源氏名")):
        return f"{default_entity}.name"
    custom_source = name or label or raw or "field"
    return f"{default_entity}.custom.{_schema_safe_key(custom_source)}"


def build_media_schema_from_pages(pages: list, operation_mappings: dict | None = None, menu_items: list | None = None) -> dict:
    """フォームDOM群から媒体構造を再構築する。

    ここで作るmedia_schemaは「実行selector」ではなく、媒体の意味構造を保存する層。
    以後のクロスメディア、項目対応、READY判定の土台にする。
    """
    import datetime as _dt_schema

    forms: list[dict] = []
    entities: dict[str, dict] = {}
    canonical_index: dict[str, list[dict]] = {}
    source_urls = set()

    def _add_field(entity: str, canonical: str, entry: dict) -> None:
        entity_row = entities.setdefault(entity, {
            "entity_type": entity,
            "fields": {},
            "forms": [],
            "field_count": 0,
        })
        field_row = entity_row["fields"].setdefault(canonical, {
            "canonical": canonical,
            "label": entry.get("label") or entry.get("name") or canonical,
            "type": entry.get("type") or entry.get("tag") or "text",
            "required": bool(entry.get("required")),
            "targets": [],
            "aliases": [],
        })
        alias = entry.get("name") or entry.get("label") or entry.get("id") or ""
        if alias and alias not in field_row["aliases"] and len(field_row["aliases"]) < 12:
            field_row["aliases"].append(alias)
        target = {
            "url": entry.get("url") or "",
            "selector": entry.get("selector") or "",
            "name": entry.get("name") or "",
            "id": entry.get("id") or "",
            "label": entry.get("label") or "",
            "section": entry.get("section") or "",
            "order": entry.get("order") or entry.get("index") or 0,
        }
        if target["url"] or target["selector"]:
            field_row["targets"].append(target)
            field_row["targets"] = field_row["targets"][:5]
        canonical_index.setdefault(canonical, []).append(target)

    for page in pages or []:
        if not isinstance(page, dict):
            continue
        form_schema = page.get("form_schema") or {}
        if not isinstance(form_schema, dict):
            continue
        fields = [f for f in (form_schema.get("fields") or []) if isinstance(f, dict)]
        if not fields:
            synthesized = []
            for group_name in ("inputs", "selects", "textareas", "file_inputs"):
                for el in (page.get(group_name) or [])[:160]:
                    if not isinstance(el, dict):
                        continue
                    typ = str(el.get("type") or ("textarea" if group_name == "textareas" else "select" if group_name == "selects" else "file" if group_name == "file_inputs" else "text"))
                    if typ.lower() in ("hidden", "password"):
                        continue
                    synthesized.append({
                        "tag": el.get("tag") or ("textarea" if group_name == "textareas" else "select" if group_name == "selects" else "input"),
                        "type": typ,
                        "name": el.get("name") or "",
                        "id": el.get("id") or "",
                        "label": el.get("label") or el.get("placeholder") or el.get("text") or "",
                        "section": page.get("title") or page.get("html_title") or "",
                        "selector": el.get("selector") or el.get("suggested_selector") or "",
                        "required": bool(el.get("required")),
                        "value": el.get("value") or "",
                        "options": el.get("options") or [],
                    })
            fields = synthesized
        if not fields:
            continue
        url = str(page.get("url") or form_schema.get("url") or "")
        if url:
            source_urls.add(url)
        entity = _infer_schema_entity(page, form_schema, fields)
        compact_fields = []
        for idx, raw_field in enumerate(fields[:160]):
            canonical = _normalize_schema_canonical(raw_field, entity)
            field_entity = canonical.split(".", 1)[0] if "." in canonical else entity
            entry = {
                "canonical": canonical,
                "entity_type": field_entity,
                "name": str(raw_field.get("name") or "")[:120],
                "id": str(raw_field.get("id") or "")[:120],
                "label": str(raw_field.get("label") or "")[:160],
                "section": str(raw_field.get("section") or "")[:160],
                "type": str(raw_field.get("type") or raw_field.get("tag") or "")[:40],
                "tag": str(raw_field.get("tag") or "")[:40],
                "selector": str(raw_field.get("selector") or "")[:220],
                "required": bool(raw_field.get("required")),
                "order": int(raw_field.get("order") or raw_field.get("index") or idx + 1),
                "url": url,
                "options": (raw_field.get("options") or [])[:20] if isinstance(raw_field.get("options"), list) else [],
            }
            compact_fields.append(entry)
            _add_field(field_entity, canonical, entry)
        form_record = {
            "url": url,
            "title": str(page.get("title") or page.get("html_title") or form_schema.get("title") or "")[:180],
            "page_purpose": page.get("page_purpose") or "",
            "page_purpose_source": page.get("page_purpose_source") or "",
            "entity_type": entity,
            "fields_count": int(form_schema.get("fields_count") or len(fields)),
            "profile_fields_count": int(form_schema.get("profile_fields_count") or 0),
            "is_profile_form": bool(form_schema.get("is_profile_form") or entity == "profile"),
            "fields": compact_fields[:45],
        }
        forms.append(form_record)
        ent = entities.setdefault(entity, {"entity_type": entity, "fields": {}, "forms": [], "field_count": 0})
        if url and url not in ent["forms"]:
            ent["forms"].append(url)
            ent["forms"] = ent["forms"][:30]

    for op, op_map in (operation_mappings or {}).items():
        if not isinstance(op_map, dict):
            continue
        fs = op_map.get("form_schema") or {}
        if not isinstance(fs, dict) or not fs.get("fields"):
            continue
        pseudo_page = {
            "url": op_map.get("target_url") or fs.get("url") or "",
            "title": fs.get("title") or "",
            "page_purpose": op_map.get("page_purpose") or "",
            "page_purpose_source": op_map.get("page_purpose_source") or "operation_mapping",
            "form_schema": fs,
        }
        nested = build_media_schema_from_pages([pseudo_page])
        for form in nested.get("forms") or []:
            if not any(x.get("url") == form.get("url") and x.get("fields_count") == form.get("fields_count") for x in forms):
                forms.append(form)
        for ent_key, ent_val in (nested.get("entities") or {}).items():
            ent = entities.setdefault(ent_key, {"entity_type": ent_key, "fields": {}, "forms": [], "field_count": 0})
            for canon, field in (ent_val.get("fields") or {}).items():
                existing = ent["fields"].setdefault(canon, field)
                if existing is not field:
                    existing["targets"] = (existing.get("targets") or []) + (field.get("targets") or [])
                    existing["targets"] = existing["targets"][:5]
                    existing["aliases"] = list(dict.fromkeys((existing.get("aliases") or []) + (field.get("aliases") or [])))[:8]

    for ent in entities.values():
        ent["field_count"] = len(ent.get("fields") or {})

    def _dedupe_schema_forms(form_rows: list[dict]) -> list[dict]:
        deduped = []
        seen = set()
        for form in form_rows or []:
            if not isinstance(form, dict):
                continue
            field_sig = tuple(sorted({
                str(f.get("canonical") or f.get("name") or f.get("label") or "")
                for f in (form.get("fields") or [])
                if isinstance(f, dict) and (f.get("canonical") or f.get("name") or f.get("label"))
            }))
            title_key = _schema_safe_key(form.get("title") or form.get("page_purpose") or form.get("url") or "")
            key = (
                form.get("entity_type") or "",
                title_key,
                int(form.get("fields_count") or len(form.get("fields") or []) or 0),
                field_sig,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(form)
        return deduped

    unique_forms = _dedupe_schema_forms(forms)

    operation_coverage = {}
    for op, op_map in (operation_mappings or {}).items():
        if not isinstance(op_map, dict):
            continue
        operation_coverage[op] = {
            "status": op_map.get("status") or "",
            "target_url": op_map.get("target_url") or "",
            "has_form_schema": bool((op_map.get("form_schema") or {}).get("fields")),
            "selector_count": len(op_map.get("selectors") or {}),
        }

    return {
        "schema_version": "schema_first_v1",
        "source": "form_schema_aggregator",
        "updated_at": _dt_schema.datetime.utcnow().isoformat(),
        "forms_count": len(unique_forms),
        "form_instances_count": len(forms),
        "source_urls_count": len(source_urls),
        "entities_count": len(entities),
        "canonical_fields_count": len(canonical_index),
        "entities": entities,
        "forms": unique_forms[:40],
        "canonical_fields": {
            canon: {"canonical": canon, "targets": targets[:5], "count": len(targets)}
            for canon, targets in list(canonical_index.items())[:180]
        },
        "operation_coverage": operation_coverage,
        "menu_items_count": len(menu_items or []),
    }


def _gemini_split_html_chunks(html: str, max_size: int = 50_000) -> list:
    """HTMLをブロック境界（</form>, </section>, </div>）で分割して最大3チャンクのリストを返す"""
    if len(html) <= max_size:
        return [html]
    chunks = []
    pos = 0
    while pos < len(html) and len(chunks) < 3:
        end = min(pos + max_size, len(html))
        if end >= len(html):
            chunks.append(html[pos:end])
            break
        cut = end
        for tag in ['</form>', '</section>', '</article>', '</div>']:
            idx = html.rfind(tag, pos + max_size // 2, end)
            if idx != -1:
                cut = idx + len(tag)
                break
        chunks.append(html[pos:cut])
        pos = cut
    return chunks


_GEMINI_INDUSTRY_HINTS: dict = {
    "nightlife": {
        "entity":       "キャスト/女の子/スタッフ/ホスト",
        "entity_fields":"名前・年齢・身長・バスト・カップ・ウエスト・ヒップ等",
        "schedule":     "出勤/シフト/在籍情報 (atwork/zaiseki/sokuhime/即ヒメ)",
        "news":         "ニュース/お知らせ/写メ日記/イベント/クーポン/速報",
        "media":        "プロフィール写真/サムネイル/写メ",
        "status":       "公開/非公開/出勤中/待機中/姫デコ/デコ設定",
        "price":        "料金/コース/システム料金/指名料/プラン",
        "url_kw":       "cast|girl|staff|hime|zaiseki|sokuhime|schedule|shift|atwork|hime_deco",
    },
    "beauty": {
        "entity":       "スタッフ/スタイリスト/セラピスト/施術者",
        "entity_fields":"名前・担当メニュー・経歴・資格等",
        "schedule":     "予約枠/施術スケジュール/空き状況/シフト",
        "news":         "キャンペーン/お知らせ/スタッフブログ/新メニュー",
        "media":        "スタッフ写真/施術写真/ビフォーアフター",
        "status":       "受付中/満席/休業/公開/非公開",
        "price":        "メニュー料金/施術コース/オプション料金",
        "url_kw":       "staff|stylist|therapist|reservation|schedule|menu|course|shift",
    },
    "retail": {
        "entity":       "商品/アイテム/SKU",
        "entity_fields":"商品名・価格・在庫数・カテゴリー・説明文等",
        "schedule":     "販売期間/在庫更新/営業時間",
        "news":         "新着商品/セール/お知らせ/コラム",
        "media":        "商品画像/メイン画像/サブ画像/サムネイル",
        "status":       "在庫あり/売り切れ/公開/非公開/販売中/販売停止",
        "price":        "販売価格/定価/割引価格/セール価格",
        "url_kw":       "product|item|goods|stock|inventory|category",
    },
    "realestate": {
        "entity":       "物件/マンション/一戸建て/土地/部屋",
        "entity_fields":"物件名・賃料・面積・間取り・所在地等",
        "schedule":     "内覧予定/空室状況/更新日/入居可能日",
        "news":         "新着物件/コラム/お知らせ/不動産ニュース",
        "media":        "外観写真/間取り図/室内写真/周辺環境写真",
        "status":       "募集中/成約済み/公開/非公開/掲載中/掲載停止",
        "price":        "賃料/管理費/礼金/敷金/販売価格",
        "url_kw":       "property|bukken|room|floor_plan|gallery|rent|sale",
    },
    "fitness": {
        "entity":       "講師/インストラクター/トレーナー",
        "entity_fields":"名前・担当レッスン・資格・プロフィール等",
        "schedule":     "レッスン/クラス/開催スケジュール/予約枠",
        "news":         "キャンペーン/イベント/ブログ/お知らせ",
        "media":        "講師写真/施設写真/レッスン動画",
        "status":       "受付中/満員/キャンセル待ち/公開/非公開",
        "price":        "レッスン料/月会費/都度利用料/コース料金",
        "url_kw":       "instructor|trainer|lesson|class|schedule|program",
    },
    "btob": {
        "entity":       "サービス/ソリューション/製品/事例",
        "entity_fields":"サービス名・説明・価格・カテゴリー等",
        "schedule":     "セミナー/ウェビナー/商談/配信スケジュール",
        "news":         "プレスリリース/導入事例/コラム/ニュース",
        "media":        "資料/事例PDF/製品画像/ホワイトペーパー",
        "status":       "公開/非公開/受付中/終了/掲載中",
        "price":        "料金/プラン/見積もり/ライセンス費用",
        "url_kw":       "service|product|seminar|webinar|case|document|news",
    },
}
_GEMINI_DEFAULT_HINT: dict = {
    "entity":       "登録対象（スタッフ/商品/物件等）",
    "entity_fields":"名前・基本情報等",
    "schedule":     "スケジュール/予定/日程/空き状況",
    "news":         "お知らせ/ニュース/ブログ/投稿",
    "media":        "画像/写真/ファイル",
    "status":       "公開/非公開/表示/非表示/有効/無効",
    "price":        "料金/価格/コース/プラン",
    "url_kw":       "edit|new|create|update|schedule|status|register|staff|product",
}


def _gemini_analyze_page_html(html: str, url: str, page_title: str = "", industry: str = "other", force_refresh: bool = False) -> dict:
    """
    HTML全文をGeminiで解析してフォーム構造JSONを返す。
    industry引数で業種別ヒントをプロンプトに注入する（全業種対応）。
    50KB超の場合はブロック境界で分割して最大3チャンク（合計最大150KB）を順次解析し fields をマージ。
    フォームがないページは fields=[] で返す。失敗時は {} を返す。
    Firestoreキャッシュ（TTL 7日）: 同一URL+industryは再分析しない。force_refresh=Trueでキャッシュを無視。
    """
    # Firestore URLキャッシュ確認（7日TTL）
    _gc_db = None
    _gc_url_key = None
    try:
        import hashlib as _hs_gc
        import datetime as _dt_gc
        from api.core.firestore_client import get_db as _get_db_gc
        _gc_db = _get_db_gc()
        if _gc_db is not None:
            _gc_url_key = _hs_gc.md5(f"{industry}:{url}".encode()).hexdigest()
            if not force_refresh:
                _gc_doc = _gc_db.collection("gemini_page_cache").document(_gc_url_key).get()
                if _gc_doc.exists:
                    _gc_data = _gc_doc.to_dict() or {}
                    _gc_ts = _gc_data.get("cached_at")
                    if _gc_ts:
                        _gc_age = (_dt_gc.datetime.utcnow() - _gc_ts.replace(tzinfo=None)).total_seconds()
                        if _gc_age < 86400 * 7 and _gc_data.get("result"):
                            print(f"[GEMINI_PAGE_CACHE_HIT] url={url[:60]} age={int(_gc_age//3600)}h", flush=True)
                            return _gc_data["result"]
            else:
                print(f"[GEMINI_PAGE_CACHE_SKIP] force_refresh url={url[:60]}", flush=True)
    except Exception as _gc_pre_e:
        print(f"[GEMINI_PAGE_CACHE_CHECK_ERROR] {type(_gc_pre_e).__name__}", flush=True)

    try:
        from api.core.llm_client import call_llm_json
        import re as _re_g

        _c = _re_g.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=_re_g.IGNORECASE | _re_g.DOTALL)
        _c = _re_g.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>',   '', _c,  flags=_re_g.IGNORECASE | _re_g.DOTALL)
        _c = _re_g.sub(r'<noscript[^>]*>.*?</noscript>', '', _c, flags=_re_g.IGNORECASE | _re_g.DOTALL)

        _chunks = _gemini_split_html_chunks(_c, max_size=50_000)
        _total_chunks = len(_chunks)

        _h = _GEMINI_INDUSTRY_HINTS.get(industry, _GEMINI_DEFAULT_HINT)

        _base_prompt_header = (
            "店舗・サービス管理システムのHTMLを解析してフォーム構造をJSONで返してください。\n\n"
            f"URL: {url}\nタイトル: {page_title}\n業種: {industry}\n\n"
            "出力JSON形式（fieldsを必ず先頭に出力すること）:\n"
            '{\n'
            '  "fields": [\n'
            '    {\n'
            '      "selector": "CSSセレクター例: #name または input[name=\'staff_name\']",\n'
            '      "label": "ラベル文字列",\n'
            '      "name": "name属性",\n'
            '      "id": "id属性",\n'
            '      "type": "text|select|textarea|file|checkbox|radio",\n'
            '      "required": true,\n'
            '      "canonical": "profile.name|profile.kana|profile.age|profile.height|profile.bust|profile.cup|profile.waist|profile.hip|profile.image|profile.status|profile.pickup|profile.custom|profile.type_tag|profile.default_start|profile.default_end|profile.room|profile.joined_at|profile.newface|schedule.date|schedule.start_time|schedule.end_time|schedule.day|schedule.status|status.visibility|status.state|status.waiting_time|content.title|content.body|price.amount|price.name|price.item|price.fee|media.file|image.file|contact.message|inquiry.body|reply.body|survey.body|form.field"\n'
            '    }\n'
            '  ],\n'
            '  "page_type": "entity_register|entity_update|schedule_update|news_post|text_update|media_replace|price_update|status_update|page_monitor|other",\n'
            '  "form_action": "フォームのPOST先URL（なければ空文字）",\n'
            '  "save_selector": "保存/登録ボタンのCSSセレクター",\n'
            '  "confidence": 0.9\n'
            '}\n\n'
            f"page_type判定基準（業種: {industry}）:\n"
            f"- entity_register: 新規{_h['entity']}の登録フォーム（{_h['entity_fields']}）\n"
            f"- entity_update: 既存{_h['entity']}の編集フォーム（{_h['entity_fields']}）\n"
            f"- schedule_update: {_h['schedule']}の入力/更新フォーム（日付・時刻・曜日等）\n"
            f"- news_post: {_h['news']}の投稿フォーム\n"
            "- text_update: 自己紹介文/説明テキスト/コメントの編集フォーム\n"
            f"- media_replace: {_h['media']}のアップロード/差し替え（file input必須）\n"
            f"- price_update: {_h['price']}の設定フォーム\n"
            f"- status_update: {_h['status']}の変更フォーム\n"
            "- page_monitor: 一覧・監視ページ（データを読み取るだけ、フォーム送信なし。URLに list/List/一覧 が含まれ入力フォームがない場合）\n"
            "- other: ログイン・ダッシュボード等（page_monitor以外の操作不能ページ）\n\n"
            "【最重要ルール】（上の番号が低いほど優先度高）\n"
            "0. タイトルに '追加'/'新規'/'新規登録'/'add'/'create'/'new'/'register' が含まれる → entity_register（URLルール1より優先）\n"
            "1. URLに 'edit'/'girledit'/'profile'/'update' が含まれる → entity_update（ただしルール0が適用済みなら entity_register を維持）\n"
            "2. URLに 'new'/'create'/'register'/'add'/'entry' が含まれる → entity_register\n"
            "3. URLに 'schedule'/'shift'/'work' が含まれる → schedule_update\n"
            "4. URLに 'news'/'blog'/'post'/'diary' が含まれる → news_post\n"
            "5. URLに 'price'/'fee'/'rate' が含まれる → price_update\n"
            "6. URLに 'status'/'visible'/'online' が含まれる → status_update\n"
            "7. URLに 'list'/'List'/'一覧' が含まれ、かつ送信ボタンが存在しない → page_monitor\n"
            "8. HTMLにフォームが見えない場合（JavaScript SPA等）でもURLとタイトルで必ずpage_typeを決定すること。other禁止。\n"
            "9. fieldsにはtype='hidden'を除く可視フィールドのみ含めること。ナビゲーション・検索バー・ページ外フォームは除外し、メインフォームの入力欄だけを抽出すること。\n"
            f"URLパターン参考: {_h['url_kw']}\n\n"
        )

        _merged_fields = []
        _merged_page_type = "other"
        _merged_form_action = ""
        _merged_save_selector = ""
        _merged_confidence = 0.0

        for _ci, _chunk in enumerate(_chunks):
            _chunk_note = f"（{_total_chunks}分割中 {_ci + 1}チャンク目）\n" if _total_chunks > 1 else ""
            _prompt = _base_prompt_header + _chunk_note + f"HTML:\n{_chunk}"
            try:
                _r = call_llm_json(
                    prompt=_prompt,
                    system_prompt="WebフォームHTML解析AI。JSONのみ出力。Markdownや説明文は禁止。fieldsを最初に出力すること。",
                    ai_tier="core",
                    max_tokens=16384,
                )
                _chunk_fields = _r.get("fields") or []
                _merged_fields.extend(_chunk_fields)
                if _ci == 0:
                    _merged_page_type    = _r.get("page_type", "other") or "other"
                    _merged_form_action  = _r.get("form_action", "") or ""
                    _merged_save_selector = _r.get("save_selector", "") or ""
                    _merged_confidence   = float(_r.get("confidence") or 0)
                elif not _merged_save_selector:
                    _merged_save_selector = _r.get("save_selector", "") or ""
                print(f"[GEMINI_PAGE_ANALYSIS] url={url[:60]} chunk={_ci+1}/{_total_chunks} page_type={_merged_page_type} chunk_fields={len(_chunk_fields)}", flush=True)
            except Exception as _e_chunk:
                print(f"[GEMINI_PAGE_ANALYSIS_CHUNK_ERROR] url={url[:60]} chunk={_ci+1} error={type(_e_chunk).__name__}:{_e_chunk}", flush=True)

        result = {
            "page_type":     _merged_page_type,
            "form_action":   _merged_form_action,
            "fields":        _merged_fields,
            "save_selector": _merged_save_selector,
            "confidence":    _merged_confidence,
        }
        print(f"[GEMINI_PAGE_ANALYSIS_DONE] url={url[:60]} total_fields={len(_merged_fields)} chunks={_total_chunks}", flush=True)
        # Firestoreキャッシュ書き込み（有意な結果のみ保存 — 空結果をキャッシュすると再トライしてもLLMが走らなくなる）
        _gc_has_content = bool(result.get("page_type")) or bool(result.get("fields")) or bool(result.get("form_action"))
        if _gc_db is not None and _gc_url_key and _gc_has_content:
            try:
                import datetime as _dt_gc_w
                _gc_db.collection("gemini_page_cache").document(_gc_url_key).set({
                    "url":       url,
                    "industry":  industry,
                    "result":    result,
                    "cached_at": _dt_gc_w.datetime.utcnow(),
                })
            except Exception as _gc_save_e:
                print(f"[GEMINI_PAGE_CACHE_SAVE_ERROR] {type(_gc_save_e).__name__}", flush=True)
        elif _gc_db is not None and _gc_url_key and not _gc_has_content:
            print(f"[GEMINI_PAGE_CACHE_SKIP_EMPTY] url={url[:60]} — 空結果はキャッシュしない", flush=True)
        return result
    except Exception as _e_gpa:
        print(f"[GEMINI_PAGE_ANALYSIS_ERROR] url={url[:60]} error={type(_e_gpa).__name__}:{_e_gpa}", flush=True)
        return {}


_GEMINI_PAGE_TYPE_TO_PURPOSE = {
    "entity_register": "create_page",
    "entity_update":   "entity_edit_page",
    "schedule_update": "schedule_page",
    "news_post":       "news_post_page",
    "text_update":     "text_edit_page",
    "media_replace":   "media_upload_page",
    "price_update":    "price_page",
    "status_update":   "status_page",
}


def fetch_dom_for_url(media_mapping: dict, target_url: str, max_follow_urls: int = 50) -> dict:
    """
    P21/P23貫通: ログイン済みセッションでtarget_urlのDOMだけを軽量取得して
    navigation_graph.pagesの該当ページを更新する。
    run_dom_scanより軽量（1ページのみ・全巡回なし）。
    """
    import datetime as _dt_fetch
    mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
    _industry   = str(media_mapping.get("industry") or "other")
    if not is_playwright_enabled():
        return {"status": "WAITING_EXECUTOR", "message": "PLAYWRIGHT_ENABLED=false"}
    if not target_url:
        return {"status": "BLOCKED", "message": "target_url empty"}

    def _compact_el_for_parent(el: dict) -> dict:
        if not isinstance(el, dict):
            return {}
        keep = {}
        for key in ("tag", "type", "name", "id", "placeholder", "aria_label", "label", "value", "text", "href", "class_name", "class", "onclick", "selector", "suggested_selector", "action", "accept"):
            val = el.get(key)
            if val is None or val == "":
                continue
            keep[key] = str(val)[:180]
        return keep

    def _compact_page_for_parent(pg: dict, ultra: bool = False) -> dict:
        if not isinstance(pg, dict):
            return {}
        base = {
            "url": pg.get("url", ""),
            "title": pg.get("title") or pg.get("html_title") or "",
            "html_title": pg.get("html_title", ""),
            "category": pg.get("category", ""),
            "manual_import": bool(pg.get("manual_import")),
            "followed_from": pg.get("followed_from", ""),
            "page_purpose": pg.get("page_purpose", ""),
            "page_purpose_source": pg.get("page_purpose_source", ""),
            "inputs_count": int(pg.get("inputs_count") or len(pg.get("inputs") or []) or 0),
            "buttons_count": int(pg.get("buttons_count") or len(pg.get("buttons") or []) or 0),
            "textareas_count": int(pg.get("textareas_count") or len(pg.get("textareas") or []) or 0),
            "forms_count": int(pg.get("forms_count") or len(pg.get("forms") or []) or 0),
            "file_inputs_count": int(pg.get("file_inputs_count") or len(pg.get("file_inputs") or []) or 0),
            "selects_count": int(pg.get("selects_count") or len(pg.get("selects") or []) or 0),
            "raw_snapshot": bool(pg.get("raw_snapshot")),
            "collected_at": pg.get("collected_at", ""),
            "dom_evidence": pg.get("dom_evidence") or {},
            "form_schema": pg.get("form_schema") or {},
        }
        if ultra:
            return base
        base.update({
            "inputs": [_compact_el_for_parent(e) for e in (pg.get("inputs") or [])[:50] if isinstance(e, dict)],
            "buttons": [_compact_el_for_parent(e) for e in (pg.get("buttons") or [])[:30] if isinstance(e, dict)],
            "textareas": [_compact_el_for_parent(e) for e in (pg.get("textareas") or [])[:20] if isinstance(e, dict)],
            "forms": [_compact_el_for_parent(e) for e in (pg.get("forms") or [])[:12] if isinstance(e, dict)],
            "file_inputs": [_compact_el_for_parent(e) for e in (pg.get("file_inputs") or [])[:15] if isinstance(e, dict)],
            "selects": [_compact_el_for_parent(e) for e in (pg.get("selects") or [])[:20] if isinstance(e, dict)],
            "links": [_compact_el_for_parent(e) for e in (pg.get("links") or [])[:25] if isinstance(e, dict)],
        })
        return base

    def _merge_compact_pages_for_parent(existing_pages: list, new_page: dict) -> list:
        pages = []
        seen = set()
        new_url = str((new_page or {}).get("url") or "")
        for pg in existing_pages or []:
            if not isinstance(pg, dict):
                continue
            url = str(pg.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            pages.append(_compact_page_for_parent(new_page if url == new_url else pg, ultra=(url != new_url)))
        if new_url and new_url not in seen:
            pages.append(_compact_page_for_parent(new_page))
        return pages[:300]

    def _size_safe_pages(pages: list) -> list:
        """Proactively ensure pages list fits within Firestore 800KB safety margin."""
        import json as _jsz
        _sz = len(_jsz.dumps(pages).encode("utf-8"))
        if _sz <= 800_000:
            return pages
        _ultra = [_compact_page_for_parent(p, ultra=True) for p in pages if isinstance(p, dict)]
        if len(_jsz.dumps(_ultra).encode("utf-8")) <= 800_000:
            return _ultra
        _trim = _ultra[-100:]
        if len(_jsz.dumps(_trim).encode("utf-8")) <= 800_000:
            return _trim
        return _ultra[-50:]

    def _canonical_profile_field(name: str, label: str, typ: str = "") -> str:
        blob = f"{name} {label}".lower()
        pairs = [
            ("content.title", ("title", "subject", "headline", "タイトル", "件名", "見出し")),
            ("content.body", ("body", "content", "comment", "message", "本文", "内容", "紹介文", "説明文")),
            ("schedule.date", ("date", "day", "日付", "対象日")),
            ("schedule.start_time", ("start", "from", "開始", "出勤開始", "open")),
            ("schedule.end_time", ("end", "to", "終了", "退勤", "close")),
            ("price.amount", ("price", "fee", "amount", "料金", "価格", "金額", "コース")),
            ("status.visibility", ("status", "state", "display", "visible", "public", "private", "表示", "公開", "非公開", "状態")),
            ("profile.status", ("status", "表示", "公開", "有効")),
            ("profile.pickup", ("pickup", "ピックアップ")),
            ("profile.name", ("name", "名前", "源氏名", "キャスト名", "女の子名")),
            ("profile.kana", ("kana", "ふりがな", "フリガナ")),
            ("profile.age", ("age", "年齢")),
            ("profile.height", ("height", "身長")),
            ("profile.bust", ("bust", "バスト", "b:")),
            ("profile.cup", ("cup", "カップ")),
            ("profile.waist", ("waist", "ウエスト", "w:")),
            ("profile.hip", ("hip", "ヒップ", "h:")),
            ("profile.default_start", ("default_start", "開始", "出勤時間")),
            ("profile.default_end", ("default_end", "終了", "退勤", "出勤時間")),
            ("profile.room", ("room", "ルーム")),
            ("profile.joined_at", ("on_store", "入店日")),
            ("profile.newface", ("newface", "新人")),
            ("profile.thumbnail_image", ("image_thumb", "サムネイル画像")),
            ("profile.thumbnail_movie_url", ("thumb", "サムネイル動画", "動画")),
            ("profile.thumbnail_movie_status", ("thumb_status", "動画")),
        ]
        if typ == "file" or "画像" in label or "写真" in label or "photo" in blob or "image" in blob or "upload" in blob:
            if any(k in blob for k in ("profile", "cast", "girl", "女の子", "キャスト", "サムネ")):
                return "profile.image"
            return "media.file"
        for canonical, keys in pairs:
            if any(k.lower() in blob for k in keys):
                return canonical
        if "type_" in blob or "タイプ" in label:
            return "profile.type_tag"
        return "profile.custom"

    def _extract_form_schema_from_page(pg, url: str, html_title: str = "") -> dict:
        try:
            fields = pg.evaluate(
                """() => {
                  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                  const cssPath = (el) => {
                    if (!el || !el.tagName) return '';
                    if (el.id) return '#' + CSS.escape(el.id);
                    const name = el.getAttribute('name');
                    const tag = el.tagName.toLowerCase();
                    if (name) return `${tag}[name="${String(name).replace(/"/g, '\\"')}"]`;
                    const cls = Array.from(el.classList || [])[0];
                    if (cls) return `${tag}.${CSS.escape(cls)}`;
                    return tag;
                  };
                  const sectionFor = (el) => {
                    let n = el;
                    for (let i = 0; n && i < 8; i++, n = n.parentElement) {
                      const h = n.querySelector && n.querySelector('h1,h2,h3,caption');
                      if (h && norm(h.textContent)) return norm(h.textContent);
                    }
                    let p = el;
                    while (p && p.previousElementSibling) {
                      p = p.previousElementSibling;
                      if (/^H[1-3]$/.test(p.tagName || '') && norm(p.textContent)) return norm(p.textContent);
                    }
                    const h = document.querySelector('h1');
                    return h ? norm(h.textContent) : '';
                  };
                  const labelFor = (el) => {
                    const id = el.id;
                    if (id) {
                      const lb = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                      if (lb && norm(lb.textContent)) return norm(lb.textContent);
                    }
                    const ownLabel = el.closest('label');
                    if (ownLabel && norm(ownLabel.textContent)) return norm(ownLabel.textContent);
                    const tr = el.closest('tr');
                    if (tr) {
                      const th = tr.querySelector('th');
                      if (th && norm(th.textContent)) return norm(th.textContent);
                    }
                    const td = el.closest('td');
                    if (td) {
                      const text = Array.from(td.childNodes || [])
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => norm(n.textContent))
                        .filter(Boolean)
                        .join(' ');
                      if (text) return text.slice(0, 160);
                    }
                    return norm(el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name') || '');
                  };
                  return Array.from(document.querySelectorAll('input, select, textarea'))
                    .filter(el => {
                      const type = (el.getAttribute('type') || '').toLowerCase();
                      const name = el.getAttribute('name') || '';
                      return !['hidden','password'].includes(type) && !/token|csrf/i.test(name);
                    })
                    .map((el, idx) => {
                      const tag = el.tagName.toLowerCase();
                      const type = (el.getAttribute('type') || tag).toLowerCase();
                      return {
                        index: idx + 1,
                        tag,
                        type,
                        name: el.getAttribute('name') || '',
                        id: el.id || '',
                        label: labelFor(el),
                        section: sectionFor(el),
                        selector: cssPath(el),
                        required: !!el.required,
                        value: (el.getAttribute('value') || '').slice(0, 120),
                        options: tag === 'select' ? Array.from(el.options || []).slice(0, 200).map(o => ({ value: o.value || '', label: norm(o.textContent) })) : [],
                      };
                    });
                }"""
            )
        except Exception as e:
            print(f"[FORM_SCHEMA_EXTRACT_ERROR] url={url[:60]} {e}", flush=True)
            fields = []
        enriched = []
        for idx, field in enumerate(fields or []):
            if not isinstance(field, dict):
                continue
            label = str(field.get("label") or "")
            name = str(field.get("name") or "")
            typ = str(field.get("type") or "")
            field["canonical"] = _canonical_profile_field(name, label, typ)
            field["order"] = idx + 1
            enriched.append(field)
        profile_count = len([f for f in enriched if str(f.get("canonical", "")).startswith("profile.") and f.get("canonical") != "profile.custom"])
        return {
            "source": "dom_form_schema",
            "url": url,
            "title": html_title,
            "fields": enriched[:220],
            "fields_count": len(enriched),
            "profile_fields_count": profile_count,
            "is_profile_form": profile_count >= 4 or any("プロフィール" in str(f.get("section") or f.get("label") or "") for f in enriched),
        }

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    if secret_name:
        creds = get_secret_json(secret_name)
        if creds and creds.get("blocked"):
            creds = None
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
        with sync_playwright() as p:
            try:
                if creds:
                    auth = create_authenticated_page(p, media_mapping, creds)
                    browser, page = auth["browser"], auth["page"]
                else:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                # target_urlへ遷移
                try:
                    page.goto(target_url, timeout=35000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                    # JS動的レンダリング待機: form要素が出現するまで最大5秒追加待機
                    try:
                        page.wait_for_selector("input:not([type=hidden]):not([type=password]), textarea, select, form", timeout=5000, state="attached")
                    except Exception:
                        pass
                    # ネットワーク静止確認（XHRでDOMを構築するSPAサイト対策）
                    try:
                        page.wait_for_load_state("networkidle", timeout=4000)
                    except Exception:
                        pass
                except PlaywrightTimeout:
                    print(f"[FETCH_DOM_TIMEOUT] url={target_url[:60]}", flush=True)
                    browser.close()
                    return {"status": "FAILED", "message": "goto timeout"}
                # DOM収集
                raw = _raw_scan_page(page)
                # [P21_DYNAMIC_FORM] 同一ページ内ボタンクリックで動的フォームを取得
                # <a href>はページ遷移するため除外。button/input[type=button]のみ対象。
                _NEW_POST_KW = ("新規", "追加", "作成", "投稿", "new", "add", "create")
                _click_candidates = []
                try:
                    # aタグを除外: button と input[type=button/submit] のみ
                    _all_btns = page.query_selector_all("button, input[type=button]")
                    for _btn in _all_btns:
                        try:
                            _btn_text = (_btn.inner_text() or "").strip()
                            _btn_val  = (_btn.get_attribute("value") or "")
                            _combined = _btn_text + _btn_val
                            if not any(k in _combined for k in _NEW_POST_KW):
                                continue
                            _danger = any(d in _combined for d in ("削除","delete","logout","signout","一括","確認","実行","検索","戻る","cancel"))
                            if _danger:
                                continue
                            # フォーム内ボタンは保存系のため除外
                            _is_in_form = False
                            try:
                                _is_in_form = page.evaluate("(el) => !!el.closest('form')", _btn)
                            except Exception:
                                pass
                            if not _is_in_form:
                                _click_candidates.append(_btn)
                        except Exception:
                            pass
                except Exception as _e_cand:
                    print(f"[P21_DYNAMIC_FORM_CAND_ERROR] {_e_cand}", flush=True)
                if _click_candidates:
                    _url_before = page.url
                    try:
                        _click_candidates[0].click(timeout=5000)
                        page.wait_for_timeout(2000)
                        # ページ遷移が発生した場合は元URLに戻して元のDOMを使用
                        if page.url != _url_before:
                            print(f"[P21_DYNAMIC_FORM_NAVIGATED] before={_url_before[:60]} after={page.url[:60]} -> revert", flush=True)
                            try:
                                page.goto(_url_before, timeout=35000, wait_until="domcontentloaded")
                                page.wait_for_timeout(1500)
                            except Exception:
                                pass
                        else:
                            _raw2 = _raw_scan_page(page)
                            for _e2 in _raw2:
                                _e2.pop("_ctx", None)
                                _e2.pop("_frame_url", None)
                            _ta2      = [e for e in _raw2 if e.get("tag") == "textarea"]
                            _fi2      = [e for e in _raw2 if e.get("tag") == "input" and e.get("type") == "file"]
                            _ta_orig  = [e for e in raw if e.get("tag") == "textarea"]
                            if len(_ta2) > len(_ta_orig) or len(_fi2) > 0:
                                raw = _raw2
                                print(f"[P21_DYNAMIC_FORM_CLICK_OK] url={target_url[:60]} textareas={len(_ta2)} file_inputs={len(_fi2)}", flush=True)
                            else:
                                print(f"[P21_DYNAMIC_FORM_NO_NEW_FIELDS] url={target_url[:60]} textareas={len(_ta2)}", flush=True)
                    except Exception as _e_click:
                        print(f"[P21_DYNAMIC_FORM_CLICK_ERROR] {_e_click}", flush=True)
                # Playwrightオブジェクト除去（Firestore保存エラー防止）
                for _e in raw:
                    _e.pop("_ctx", None)
                    _e.pop("_frame_url", None)
                inputs    = [
                    e for e in raw
                    if (
                        e.get("tag") == "input"
                        and e.get("type") not in ("hidden","password")
                    ) or e.get("tag") == "select"
                ]
                buttons   = [e for e in raw if e.get("tag") == "button"]
                textareas = [e for e in raw if e.get("tag") == "textarea"]
                forms     = [e for e in raw if e.get("tag") == "form"]
                links     = [e for e in raw if e.get("tag") == "a"]
                file_inputs = [e for e in raw if e.get("tag") == "input" and e.get("type") == "file"]
                selects   = [e for e in raw if e.get("tag") == "select"]
                _html_title = ""
                try:
                    _html_title = page.title() or ""
                except Exception:
                    _html_title = ""
                _form_schema = _extract_form_schema_from_page(page, target_url, _html_title)
                # ── Gemini HTML全文解析（フォーム構造の完全取得） ──
                _gemini_result = {}
                try:
                    _raw_html_g = page.content()
                    _gemini_result = _gemini_analyze_page_html(_raw_html_g, target_url, _html_title, industry=_industry)
                    if _gemini_result:
                        # page_type/confidence等はfields=[]でも必ず保存（フォールバック判定に必要）
                        _form_schema["gemini_page_type"]     = _gemini_result.get("page_type") or ""
                        _form_schema["gemini_form_action"]   = _gemini_result.get("form_action") or ""
                        _form_schema["gemini_save_selector"] = _gemini_result.get("save_selector") or ""
                        _form_schema["gemini_confidence"]    = float(_gemini_result.get("confidence") or 0)
                        if _gemini_result.get("fields"):
                            _form_schema["gemini_fields"] = _gemini_result.get("fields", [])
                except Exception as _e_g_main:
                    print(f"[GEMINI_MAIN_PAGE_ERROR] url={target_url[:60]} {_e_g_main}", flush=True)
                _gemini_page_type = (_gemini_result.get("page_type") or "")
                _gemini_purpose   = _GEMINI_PAGE_TYPE_TO_PURPOSE.get(_gemini_page_type, "")
                page_data = {
                    "url":              target_url,
                    "manual_import":    True,
                    "html_title":        _html_title,
                    "page_purpose":      _gemini_purpose,
                    "page_purpose_source": "gemini_html_analysis" if _gemini_purpose else "",
                    "inputs":           inputs[:120],
                    "buttons":          buttons[:80],
                    "textareas":        textareas[:30],
                    "forms":            forms[:20],
                    "file_inputs":      file_inputs[:20],
                    "selects":          selects[:30],
                    "links":            links[:100],
                    "inputs_count":     len(inputs),
                    "buttons_count":    len(buttons),
                    "textareas_count":  len(textareas),
                    "forms_count":      len(forms),
                    "file_inputs_count": len(file_inputs),
                    "selects_count":    len(selects),
                    "raw_snapshot":     True,
                    "collected_at":     _dt_fetch.datetime.utcnow().isoformat(),
                    "dom_evidence": {
                        "has_form":       len(forms) > 0,
                        "has_input":      len(inputs) > 0,
                        "has_file_input": len(file_inputs) > 0,
                        "has_button":     len(buttons) > 0,
                        "has_textarea":   len(textareas) > 0,
                        "has_select":     len(selects) > 0,
                    },
                    "form_schema":       _form_schema,
                }
                _followed_pages_data = []
                # ── リストページでtextarea/form未検出の場合: edit/new/add/create/register リンクを全件追跡 ──
                # 全URLを抽出→順番にスキャン→到達順にFirestore保存→途中再開対応
                _save_kw_for_follow = (
                    "save", "submit", "update", "register", "regist", "create", "add",
                    "commit", "apply", "publish",
                    "保存", "登録", "更新", "反映", "送信", "設定", "公開",
                )
                _search_kw_for_follow = (
                    "search", "filter", "sort", "preview", "back", "cancel",
                    "検索", "絞込", "並び替え", "戻る", "キャンセル",
                )
                _has_save_like_button = False
                for _b_follow in (buttons + inputs):
                    _bt_follow = " ".join([
                        _b_follow.get("text") or "",
                        _b_follow.get("value") or "",
                        _b_follow.get("name") or "",
                        _b_follow.get("id") or "",
                        _b_follow.get("class_name") or _b_follow.get("class") or "",
                        _b_follow.get("onclick") or "",
                    ]).lower()
                    if any(k in _bt_follow for k in _save_kw_for_follow) and not any(k in _bt_follow for k in _search_kw_for_follow):
                        _has_save_like_button = True
                        break
                # フォームタグ自体がある かつ 保存系ボタンもある場合のみ「実フォームページ」と判定してフォローをスキップ。
                # 一覧ページは「新規登録」ボタンがあってもフォームタグはないのでフォローが行われる。
                _needs_follow_edit_links = (
                    len(textareas) == 0
                    and len(file_inputs) == 0
                    and not (len(forms) > 0 and _has_save_like_button)
                )
                if _needs_follow_edit_links and mapping_id:
                    from urllib.parse import urljoin as _urljoin
                    from api.core.firestore_client import get_db as _get_db_follow
                    _db_follow = _get_db_follow()
                    _follow_kw = (
                        "/edit", "/new", "/add", "/create", "/register",
                        "edit", "update", "modify", "detail", "view", "form",
                        "regist", "entry", "input",
                        "?id=", "&id=",   # 個別アイテムページのIDパラメータ
                        "cast", "girl", "staff", "talent", "profile", "info",  # 風俗管理システム汎用
                        "編集", "変更", "修正", "詳細", "登録", "追加", "新規", "入力",
                    )
                    _skip_kw   = ("delete", "signout", "logout", "up?id=", "down?id=", "copy", "&c=on", "_list", "search", "csv")
                    # 全対象URLを重複除去して収集
                    _follow_urls = []
                    _seen_follow = set()
                    for _lk in links[:100]:
                        _lk_href = _lk.get("href") or ""
                        _lk_text = " ".join([
                            _lk_href,
                            _lk.get("text") or "",
                            _lk.get("class_name") or _lk.get("class") or "",
                            _lk.get("aria_label") or "",
                            _lk.get("onclick") or "",
                        ]).lower()
                        if not any(k in _lk_text for k in _follow_kw):
                            continue
                        if any(k in _lk_text for k in _skip_kw):
                            continue
                        _abs_url = _lk_href if _lk_href.startswith("http") else _urljoin(target_url, _lk_href)
                        if not str(_abs_url).startswith("http"):
                            continue
                        if _abs_url not in _seen_follow and _abs_url != target_url:
                            _follow_urls.append(_abs_url)
                            _seen_follow.add(_abs_url)
                    _follow_limit = max(0, int(max_follow_urls or 0))
                    _follow_urls = _follow_urls[:_follow_limit]
                    print(f"[FETCH_DOM_FOLLOW_URLS] list_url={target_url[:60]} found={len(_follow_urls)} urls={_follow_urls[:3]} (limited to {_follow_limit})", flush=True)
                    # resume対応: 既にスキャン済みのURLをスキップ
                    _doc_now = _db_follow.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                    _ng_now  = _doc_now.get("navigation_graph") or {}
                    _done_urls = set(_ng_now.keys())
                    _pages_now = _ng_now.get("pages") or []
                    _done_urls.update(p.get("url","") for p in _pages_now)
                    _total_follow = len(_follow_urls)
                    for _fi_idx, _followed_url in enumerate(_follow_urls):
                        # resume: スキャン済みならスキップ
                        if _followed_url in _done_urls:
                            print(f"[FETCH_DOM_FOLLOW_SKIP] already_scanned url={_followed_url[:60]}", flush=True)
                            continue
                        # 進捗をFirestoreに書き込み
                        try:
                            _db_follow.collection("media_mappings").document(mapping_id).update({
                                "scan_progress.follow_current": _followed_url,
                                "scan_progress.follow_done": _fi_idx,
                                "scan_progress.follow_total": _total_follow,
                                "scan_progress.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                            })
                        except Exception as _e_prog_follow:
                            print(f"[FETCH_DOM_FOLLOW_PROG_ERROR] {_e_prog_follow}", flush=True)
                        print(f"[FETCH_DOM_FOLLOW_LINK] ({_fi_idx+1}/{_total_follow}) {_followed_url[:80]}", flush=True)
                        try:
                            page.goto(_followed_url, timeout=35000, wait_until="domcontentloaded")
                            page.wait_for_timeout(1500)
                            # JS動的レンダリング待機（フォローページ）
                            try:
                                page.wait_for_selector("input:not([type=hidden]):not([type=password]), textarea, select, form", timeout=5000, state="attached")
                            except Exception:
                                pass
                            try:
                                page.wait_for_load_state("networkidle", timeout=4000)
                            except Exception:
                                pass
                            _raw_follow = _raw_scan_page(page)
                            for _ef in _raw_follow:
                                _ef.pop("_ctx", None)
                                _ef.pop("_frame_url", None)
                            _f_inputs      = [
                                e for e in _raw_follow
                                if (
                                    e.get("tag") == "input"
                                    and e.get("type") not in ("hidden","password")
                                ) or e.get("tag") == "select"
                            ]
                            _f_buttons     = [e for e in _raw_follow if e.get("tag") == "button"]
                            _f_textareas   = [e for e in _raw_follow if e.get("tag") == "textarea"]
                            _f_forms       = [e for e in _raw_follow if e.get("tag") == "form"]
                            _f_links       = [e for e in _raw_follow if e.get("tag") == "a"]
                            _f_file_inputs = [e for e in _raw_follow if e.get("tag") == "input" and e.get("type") == "file"]
                            _f_selects     = [e for e in _raw_follow if e.get("tag") == "select"]
                            _f_html_title = ""
                            try:
                                _f_html_title = page.title() or ""
                            except Exception:
                                _f_html_title = ""
                            _f_form_schema = _extract_form_schema_from_page(page, _followed_url, _f_html_title)
                            # ── フォローページもGemini HTML全文解析 ──
                            _f_gemini_result = {}
                            try:
                                _f_raw_html_g = page.content()
                                _f_gemini_result = _gemini_analyze_page_html(_f_raw_html_g, _followed_url, _f_html_title, industry=_industry)
                                if _f_gemini_result:
                                    # page_type等はfields=[]でも必ず保存
                                    _f_form_schema["gemini_page_type"]     = _f_gemini_result.get("page_type") or ""
                                    _f_form_schema["gemini_form_action"]   = _f_gemini_result.get("form_action") or ""
                                    _f_form_schema["gemini_save_selector"] = _f_gemini_result.get("save_selector") or ""
                                    _f_form_schema["gemini_confidence"]    = float(_f_gemini_result.get("confidence") or 0)
                                    if _f_gemini_result.get("fields"):
                                        _f_form_schema["gemini_fields"] = _f_gemini_result.get("fields", [])
                            except Exception as _e_g_follow:
                                print(f"[GEMINI_FOLLOW_PAGE_ERROR] url={_followed_url[:60]} {_e_g_follow}", flush=True)
                            _f_gemini_page_type = (_f_gemini_result.get("page_type") or "")
                            _f_gemini_purpose   = _GEMINI_PAGE_TYPE_TO_PURPOSE.get(_f_gemini_page_type, "")
                            _follow_page_data = {
                                "url":               _followed_url,
                                "manual_import":     True,
                                "followed_from":      target_url,
                                "html_title":         _f_html_title,
                                "page_purpose":       _f_gemini_purpose,
                                "page_purpose_source": "gemini_html_analysis" if _f_gemini_purpose else "",
                                "inputs":            _f_inputs[:120],
                                "buttons":           _f_buttons[:80],
                                "textareas":         _f_textareas[:30],
                                "forms":             _f_forms[:20],
                                "file_inputs":       _f_file_inputs[:20],
                                "selects":           _f_selects[:30],
                                "links":             _f_links[:100],
                                "inputs_count":      len(_f_inputs),
                                "buttons_count":     len(_f_buttons),
                                "textareas_count":   len(_f_textareas),
                                "forms_count":       len(_f_forms),
                                "file_inputs_count": len(_f_file_inputs),
                                "selects_count":     len(_f_selects),
                                "raw_snapshot":      True,
                                "collected_at":      _dt_fetch.datetime.utcnow().isoformat(),
                                "dom_evidence": {
                                    "has_form":       len(_f_forms) > 0,
                                    "has_input":      len(_f_inputs) > 0,
                                    "has_file_input": len(_f_file_inputs) > 0,
                                    "has_button":     len(_f_buttons) > 0,
                                    "has_textarea":   len(_f_textareas) > 0,
                                    "has_select":     len(_f_selects) > 0,
                                },
                                "form_schema":        _f_form_schema,
                            }
                            _followed_pages_data.append(_follow_page_data)
                            print(f"[FETCH_DOM_FOLLOW_DONE] ({_fi_idx+1}/{_total_follow}) url={_followed_url[:60]} inputs={len(_f_inputs)} textareas={len(_f_textareas)}", flush=True)
                            # 到達順にFirestoreへ即時保存
                            try:
                                _doc_save = _db_follow.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                                _pages_save = (_doc_save.get("navigation_graph") or {}).get("pages") or []
                                _pages_save = _merge_compact_pages_for_parent(_pages_save, _follow_page_data)
                                _pages_save = _size_safe_pages(_pages_save)
                                _db_follow.collection("media_mappings").document(mapping_id).update({
                                    "navigation_graph.pages": _pages_save,
                                    "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                                })
                                _done_urls.add(_followed_url)
                                print(f"[FETCH_DOM_FOLLOW_SAVED] url={_followed_url[:60]}", flush=True)
                            except Exception as _e_save_follow:
                                print(f"[FETCH_DOM_FOLLOW_SAVE_ERROR] {_e_save_follow}", flush=True)
                                if "maximum allowed size" in str(_e_save_follow):
                                    try:
                                        _db_follow.collection("media_mappings").document(mapping_id).update({
                                            "navigation_graph.pages": [_compact_page_for_parent(_follow_page_data)],
                                            "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                                            "navigation_graph.storage_mode": "latest_page_only",
                                        })
                                        print(f"[FETCH_DOM_FOLLOW_SAVE_COMPACTED] url={_followed_url[:60]}", flush=True)
                                    except Exception as _e_save_follow2:
                                        print(f"[FETCH_DOM_FOLLOW_SAVE_COMPACTED_ERROR] {_e_save_follow2}", flush=True)
                            # ── 2段階フォロー: 1段階目でフォームが見つからない場合、その先のリンクも追う ──
                            _f_needs_2nd_follow = (
                                len(_f_textareas) == 0
                                and len(_f_file_inputs) == 0
                                and len(_f_inputs) == 0
                                and not (len(_f_forms) > 0 and len(_f_buttons) > 0)
                                and len(_f_links) > 0
                                and _f_gemini_page_type in ("other", "")
                            )
                            if _f_needs_2nd_follow:
                                _L2_KW = (
                                    "schedule", "shift", "atwork", "zaiseki", "sokuhime", "hime",
                                    "cast", "girl", "staff", "talent",
                                    "?id=", "&id=", "/edit", "/input", "/form",
                                    "編集", "入力", "スケジュール", "出勤",
                                )
                                _l2_candidates = []
                                _seen_l2 = set()
                                for _lk2 in _f_links[:50]:
                                    _lk2_href = _lk2.get("href") or ""
                                    _lk2_text = (_lk2_href + " " + (_lk2.get("text") or "")).lower()
                                    if not any(k in _lk2_text for k in _L2_KW):
                                        continue
                                    if any(k in _lk2_text for k in ("delete", "signout", "logout", "search", "csv")):
                                        continue
                                    _abs_l2 = _lk2_href if _lk2_href.startswith("http") else _urljoin(_followed_url, _lk2_href)
                                    if not str(_abs_l2).startswith("http"):
                                        continue
                                    if _abs_l2 not in _done_urls and _abs_l2 not in _seen_l2 and _abs_l2 != _followed_url and _abs_l2 != target_url:
                                        _l2_candidates.append(_abs_l2)
                                        _seen_l2.add(_abs_l2)
                                _l2_candidates = _l2_candidates[:3]
                                print(f"[FETCH_DOM_L2_FOLLOW] from={_followed_url[:60]} l2_count={len(_l2_candidates)}", flush=True)
                                for _l2_url in _l2_candidates:
                                    try:
                                        page.goto(_l2_url, timeout=25000, wait_until="domcontentloaded")
                                        page.wait_for_timeout(1500)
                                        try:
                                            page.wait_for_selector("input:not([type=hidden]):not([type=password]), textarea, select, form", timeout=5000, state="attached")
                                        except Exception:
                                            pass
                                        _raw_l2 = _raw_scan_page(page)
                                        for _el2 in _raw_l2:
                                            _el2.pop("_ctx", None)
                                            _el2.pop("_frame_url", None)
                                        _l2_inputs     = [e for e in _raw_l2 if (e.get("tag") == "input" and e.get("type") not in ("hidden","password")) or e.get("tag") == "select"]
                                        _l2_buttons    = [e for e in _raw_l2 if e.get("tag") == "button"]
                                        _l2_textareas  = [e for e in _raw_l2 if e.get("tag") == "textarea"]
                                        _l2_forms      = [e for e in _raw_l2 if e.get("tag") == "form"]
                                        _l2_links      = [e for e in _raw_l2 if e.get("tag") == "a"]
                                        _l2_file_inputs= [e for e in _raw_l2 if e.get("tag") == "input" and e.get("type") == "file"]
                                        _l2_selects    = [e for e in _raw_l2 if e.get("tag") == "select"]
                                        _l2_html_title = ""
                                        try:
                                            _l2_html_title = page.title() or ""
                                        except Exception:
                                            pass
                                        _l2_form_schema = _extract_form_schema_from_page(page, _l2_url, _l2_html_title)
                                        _l2_gemini_result = {}
                                        try:
                                            _l2_raw_html = page.content()
                                            _l2_gemini_result = _gemini_analyze_page_html(_l2_raw_html, _l2_url, _l2_html_title, industry=_industry)
                                            if _l2_gemini_result:
                                                _l2_form_schema["gemini_page_type"]     = _l2_gemini_result.get("page_type") or ""
                                                _l2_form_schema["gemini_form_action"]   = _l2_gemini_result.get("form_action") or ""
                                                _l2_form_schema["gemini_save_selector"] = _l2_gemini_result.get("save_selector") or ""
                                                _l2_form_schema["gemini_confidence"]    = float(_l2_gemini_result.get("confidence") or 0)
                                                if _l2_gemini_result.get("fields"):
                                                    _l2_form_schema["gemini_fields"] = _l2_gemini_result.get("fields", [])
                                        except Exception as _e_g_l2:
                                            print(f"[GEMINI_L2_PAGE_ERROR] url={_l2_url[:60]} {_e_g_l2}", flush=True)
                                        _l2_gpt = (_l2_gemini_result.get("page_type") or "")
                                        _l2_purpose = _GEMINI_PAGE_TYPE_TO_PURPOSE.get(_l2_gpt, "")
                                        _l2_page_data = {
                                            "url":               _l2_url,
                                            "manual_import":     True,
                                            "followed_from":     _followed_url,
                                            "html_title":        _l2_html_title,
                                            "page_purpose":      _l2_purpose,
                                            "page_purpose_source": "gemini_html_analysis" if _l2_purpose else "",
                                            "inputs":            _l2_inputs[:120],
                                            "buttons":           _l2_buttons[:80],
                                            "textareas":         _l2_textareas[:30],
                                            "forms":             _l2_forms[:20],
                                            "file_inputs":       _l2_file_inputs[:20],
                                            "selects":           _l2_selects[:30],
                                            "links":             _l2_links[:100],
                                            "inputs_count":      len(_l2_inputs),
                                            "buttons_count":     len(_l2_buttons),
                                            "textareas_count":   len(_l2_textareas),
                                            "forms_count":       len(_l2_forms),
                                            "file_inputs_count": len(_l2_file_inputs),
                                            "selects_count":     len(_l2_selects),
                                            "raw_snapshot":      True,
                                            "collected_at":      _dt_fetch.datetime.utcnow().isoformat(),
                                            "dom_evidence": {
                                                "has_form":       len(_l2_forms) > 0,
                                                "has_input":      len(_l2_inputs) > 0,
                                                "has_file_input": len(_l2_file_inputs) > 0,
                                                "has_button":     len(_l2_buttons) > 0,
                                                "has_textarea":   len(_l2_textareas) > 0,
                                                "has_select":     len(_l2_selects) > 0,
                                            },
                                            "form_schema": _l2_form_schema,
                                        }
                                        _followed_pages_data.append(_l2_page_data)
                                        print(f"[FETCH_DOM_L2_DONE] url={_l2_url[:60]} inputs={len(_l2_inputs)} textareas={len(_l2_textareas)}", flush=True)
                                        try:
                                            _doc_l2 = _db_follow.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                                            _pages_l2 = (_doc_l2.get("navigation_graph") or {}).get("pages") or []
                                            _pages_l2 = _merge_compact_pages_for_parent(_pages_l2, _l2_page_data)
                                            _pages_l2 = _size_safe_pages(_pages_l2)
                                            _db_follow.collection("media_mappings").document(mapping_id).update({
                                                "navigation_graph.pages": _pages_l2,
                                                "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                                            })
                                            _done_urls.add(_l2_url)
                                        except Exception as _e_l2_save:
                                            print(f"[FETCH_DOM_L2_SAVE_ERROR] {_e_l2_save}", flush=True)
                                            if "maximum allowed size" in str(_e_l2_save):
                                                try:
                                                    _db_follow.collection("media_mappings").document(mapping_id).update({
                                                        "navigation_graph.pages": [_compact_page_for_parent(_l2_page_data, ultra=True)],
                                                        "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                                                        "navigation_graph.storage_mode": "latest_page_only",
                                                    })
                                                except Exception as _e_l2_save2:
                                                    print(f"[FETCH_DOM_L2_SAVE_COMPACTED_ERROR] {_e_l2_save2}", flush=True)
                                    except Exception as _e_l2:
                                        print(f"[FETCH_DOM_L2_ERROR] url={_l2_url[:60]} {_e_l2}", flush=True)
                        except Exception as _e_follow:
                            print(f"[FETCH_DOM_FOLLOW_ERROR] ({_fi_idx+1}/{_total_follow}) url={_followed_url[:60]} {_e_follow}", flush=True)
                    # 完了進捗書き込み
                    try:
                        _db_follow.collection("media_mappings").document(mapping_id).update({
                            "scan_progress.follow_done": _total_follow,
                            "scan_progress.follow_total": _total_follow,
                            "scan_progress.follow_current": "",
                            "scan_progress.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                        })
                    except Exception:
                        pass
                browser.close()
                print(f"[FETCH_DOM_DONE] url={target_url[:60]} inputs={len(inputs)} buttons={len(buttons)} textareas={len(textareas)}", flush=True)
                # navigation_graph.pagesの該当ページを更新
                if mapping_id:
                    try:
                        from api.core.firestore_client import get_db as _get_db_fetch
                        _db_fetch = _get_db_fetch()
                        _doc = _db_fetch.collection("media_mappings").document(mapping_id).get().to_dict() or {}
                        _pages = (_doc.get("navigation_graph") or {}).get("pages") or []
                        _pages = _merge_compact_pages_for_parent(_pages, page_data)
                        _pages = _size_safe_pages(_pages)
                        _db_fetch.collection("media_mappings").document(mapping_id).update({
                            "navigation_graph.pages": _pages,
                            "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                        })
                        print(f"[FETCH_DOM_SAVED] mapping_id={mapping_id} url={target_url[:60]}", flush=True)
                    except Exception as _e_save_fetch:
                        print(f"[FETCH_DOM_SAVE_ERROR] {_e_save_fetch}", flush=True)
                        if "maximum allowed size" in str(_e_save_fetch):
                            try:
                                _db_fetch.collection("media_mappings").document(mapping_id).update({
                                    "navigation_graph.pages": [_compact_page_for_parent(page_data)],
                                    "navigation_graph.updated_at": _dt_fetch.datetime.utcnow().isoformat(),
                                    "navigation_graph.storage_mode": "latest_page_only",
                                })
                                print(f"[FETCH_DOM_SAVE_COMPACTED] mapping_id={mapping_id} url={target_url[:60]}", flush=True)
                            except Exception as _e_save_fetch2:
                                print(f"[FETCH_DOM_SAVE_COMPACTED_ERROR] {_e_save_fetch2}", flush=True)
                return {"status": "OK", "page_data": page_data, "followed_pages": _followed_pages_data}
            except Exception as _e_fetch:
                import traceback as _tb_fetch
                print(f"[FETCH_DOM_ERROR] url={target_url[:60]} {_e_fetch}", flush=True)
                print(_tb_fetch.format_exc(), flush=True)
                try: browser.close()
                except Exception: pass
                return {"status": "FAILED", "message": str(_e_fetch)}
    except Exception as _e_outer_fetch:
        import traceback as _tb_outer_fetch
        print(f"[FETCH_DOM_OUTER_ERROR] {_e_outer_fetch}", flush=True)
        print(_tb_outer_fetch.format_exc(), flush=True)
        return {"status": "FAILED", "message": str(_e_outer_fetch)}


def fetch_content_snapshot_for_url(media_mapping: dict, target_url: str) -> dict:
    """
    Cross-media source extraction: read visible page content with the mapped
    authenticated session when credentials exist. This does not bypass auth;
    it only uses the user's stored media mapping.
    """
    if not is_playwright_enabled():
        return {"status": "WAITING_EXECUTOR", "ok": False, "message": "PLAYWRIGHT_ENABLED=false"}
    if not target_url or not str(target_url).startswith(("http://", "https://")):
        return {"status": "BLOCKED", "ok": False, "message": "target_url invalid"}

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    if secret_name:
        creds = get_secret_json(secret_name)
        if creds and creds.get("blocked"):
            creds = None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
        with sync_playwright() as p:
            browser = None
            try:
                if creds:
                    auth = create_authenticated_page(p, media_mapping, creds)
                    browser, page = auth["browser"], auth["page"]
                else:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                try:
                    page.goto(target_url, timeout=35000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                except PlaywrightTimeout:
                    try:
                        browser.close()
                    except Exception:
                        pass
                    return {"status": "FAILED", "ok": False, "message": "goto timeout", "source_url": target_url}

                title = ""
                description = ""
                body_text = ""
                image_urls = []
                try:
                    title = page.title() or ""
                except Exception:
                    pass
                try:
                    description = page.locator("meta[name='description']").first().get_attribute("content", timeout=1000) or ""
                except Exception:
                    description = ""
                try:
                    body_text = page.locator("body").inner_text(timeout=5000) or ""
                except Exception:
                    body_text = ""
                try:
                    image_urls = page.evaluate(
                        """() => Array.from(document.images || [])
                            .map(img => img.currentSrc || img.src || img.getAttribute('src') || '')
                            .filter(Boolean)
                            .slice(0, 30)"""
                    ) or []
                except Exception:
                    image_urls = []
                items = []
                try:
                    items = page.evaluate(
                        """() => {
                            const out = [];
                            const addObj = (obj) => {
                                if (!obj || typeof obj !== 'object') return;
                                const graph = Array.isArray(obj['@graph']) ? obj['@graph'] : [obj];
                                for (const item of graph) {
                                    if (!item || typeof item !== 'object') continue;
                                    const name = item.name || item.headline || item.title || '';
                                    const desc = item.description || item.text || '';
                                    const url = item.url || item['@id'] || location.href;
                                    let images = item.image || [];
                                    if (typeof images === 'string') images = [images];
                                    if (name || desc) {
                                        out.push({ name, title: name, body: desc, text: desc, value: desc, source_url: url, image_urls: Array.isArray(images) ? images.slice(0, 10) : [] });
                                    }
                                }
                            };
                            for (const script of Array.from(document.querySelectorAll('script[type="application/ld+json"]')).slice(0, 10)) {
                                try {
                                    const parsed = JSON.parse(script.textContent || 'null');
                                    if (Array.isArray(parsed)) parsed.forEach(addObj);
                                    else addObj(parsed);
                                } catch {}
                            }
                            return out.slice(0, 20);
                        }"""
                    ) or []
                except Exception:
                    items = []
                # フォームフィールド値・テーブル・定義リストから構造化データを抽出
                # （取得元がキャスト編集フォームや一覧テーブルの場合、実データが取れる）
                structured_fields = {}
                try:
                    structured_fields = page.evaluate("""
                    () => {
                        const result = {};
                        const seen = new Set();
                        const addField = (key, val) => {
                            if (!key || !val || key.length > 40 || val.length > 300) return;
                            const k = key.replace(/[\\s\\u3000]+/g, '').slice(0, 30);
                            if (k && !seen.has(k)) { seen.add(k); result[key.trim()] = val.trim(); }
                        };
                        // フォーム入力値
                        document.querySelectorAll('input[type="text"],input[type="number"],input[type="email"],input[type="tel"],textarea,select').forEach(inp => {
                            const v = inp.tagName === 'SELECT'
                                ? (inp.options[inp.selectedIndex] || {}).text || inp.value
                                : inp.value;
                            if (!v || !v.trim()) return;
                            let label = '';
                            if (inp.id) { const l = document.querySelector('label[for="' + inp.id + '"]'); if (l) label = l.textContent.trim(); }
                            if (!label) { const p = inp.closest('tr,td,li,p,div'); if (p) { const t = p.textContent.replace(v,'').trim(); if (t.length < 40) label = t; } }
                            if (!label && inp.placeholder) label = inp.placeholder;
                            if (!label && inp.name) label = inp.name;
                            addField(label, v);
                        });
                        // テーブル（th:td ペア）
                        document.querySelectorAll('table tr').forEach(row => {
                            const th = row.querySelector('th');
                            const td = row.querySelector('td');
                            if (th && td) addField(th.textContent.trim(), td.textContent.trim());
                        });
                        // 定義リスト（dt:dd ペア）
                        document.querySelectorAll('dl').forEach(dl => {
                            const dts = Array.from(dl.querySelectorAll('dt'));
                            const dds = Array.from(dl.querySelectorAll('dd'));
                            dts.forEach((dt, i) => { if (dds[i]) addField(dt.textContent.trim(), dds[i].textContent.trim()); });
                        });
                        return result;
                    }
                    """) or {}
                except Exception:
                    structured_fields = {}
                try:
                    browser.close()
                except Exception:
                    pass
                return {
                    "status": "OK",
                    "ok": True,
                    "source_url": page.url or target_url,
                    "requested_url": target_url,
                    "title": title.strip(),
                    "description": description.strip(),
                    "body_text": " ".join(str(body_text).split())[:12000],
                    "image_urls": image_urls[:30],
                    "items": items[:20],
                    "structured_fields": {k: v for k, v in list(structured_fields.items())[:50] if k and v},
                    "content_type": "browser/dom",
                    "authenticated": bool(creds),
                }
            except Exception as e:
                try:
                    if browser:
                        browser.close()
                except Exception:
                    pass
                print(f"[FETCH_CONTENT_SNAPSHOT_ERROR] url={target_url[:60]} {e}", flush=True)
                return {"status": "FAILED", "ok": False, "message": str(e), "source_url": target_url}
    except Exception as e:
        print(f"[FETCH_CONTENT_SNAPSHOT_OUTER_ERROR] {e}", flush=True)
        return {"status": "FAILED", "ok": False, "message": str(e), "source_url": target_url}


def extract_entity_list(media_mapping: dict, list_url: str, entity_label: str = "対象", max_items: int = 200) -> dict:
    """
    取得元のエンティティ一覧ページ（例: 女の子一覧）へ認証付きで遷移し、
    {name, url} のエンティティ候補リストを抽出する。
    認証回避はしない（保存済みマッピングのセッションを利用するのみ）。
    ID/PASS/Cookieはログ・戻り値に一切含めない。
    """
    if not is_playwright_enabled():
        return {"ok": False, "status": "WAITING_EXECUTOR", "message": "PLAYWRIGHT_ENABLED=false", "entities": []}
    if not list_url or not str(list_url).startswith(("http://", "https://")):
        return {"ok": False, "status": "BLOCKED", "message": "list_url invalid", "entities": []}

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    if secret_name:
        creds = get_secret_json(secret_name)
        if creds and creds.get("blocked"):
            creds = None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
        with sync_playwright() as p:
            browser = None
            try:
                if creds:
                    auth = create_authenticated_page(p, media_mapping, creds)
                    browser, page = auth["browser"], auth["page"]
                else:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()

                # ログイン後の実際のURLを確認し、list_urlがlogin_urlと同じなら上書き
                # （login_success_redirect_urlが未設定の場合への対処）
                _raw_login_url = (media_mapping.get("login_url") or "").rstrip("/")
                _post_login_url = (page.url or "").rstrip("/")
                if list_url.rstrip("/") == _raw_login_url and _post_login_url and _post_login_url != _raw_login_url:
                    list_url = page.url
                    print(f"[EXTRACT_ENTITY_LIST] list_url==login_url → post-login: {list_url}", flush=True)

                try:
                    page.goto(list_url, timeout=35000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                except PlaywrightTimeout:
                    try:
                        browser.close()
                    except Exception:
                        pass
                    return {"ok": False, "status": "FAILED", "message": "goto timeout", "entities": []}

                current_url = page.url
                import json as _json_sel

                # Step 1: hrefにエンティティ種別+listキーワードを含むリンクで一覧ページへ移動
                try:
                    _list_cands = page.evaluate(r"""() => {
                        const origin = location.origin;
                        const hKws = ['cast', 'girl', 'gal', 'member', 'staff', 'lady', 'talent', 'model'];
                        const lKws = ['list', '_list', 'index', 'ichiran'];
                        const out = [];
                        for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                            const h = a.href.toLowerCase();
                            if (!a.href.startsWith(origin)) continue;
                            for (const ek of hKws) {
                                if (h.includes(ek) && lKws.some(lk => h.includes(lk))) { out.push(a.href); break; }
                            }
                        }
                        return [...new Set(out)].slice(0, 5);
                    }""") or []
                    if _list_cands and _list_cands[0].rstrip("/") != page.url.rstrip("/"):
                        print(f"[EXTRACT_ENTITY_LIST] nav→list: {_list_cands[0]}", flush=True)
                        page.goto(_list_cands[0], timeout=25000, wait_until="domcontentloaded")
                        page.wait_for_timeout(1000)
                        current_url = page.url
                except Exception as _nav_e:
                    print(f"[EXTRACT_ENTITY_LIST_NAV] {type(_nav_e).__name__}", flush=True)

                # Step 2: シンプルJS抽出（table行 + em/strong/b + ?id=パラメータ）
                # HTMLを見て把握した構造に合わせた直接抽出
                _SIMPLE_JS = f"""() => {{
                    const origin = location.origin;
                    const out = [];
                    const seen = new Set();
                    const ID_PAT = /[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i;
                    const HIDDEN_CLASSES = ['disabled','hidden','inactive','gray','grey','muted','deleted','stop','hishow','non-public','closed','retire'];
                    const HIDDEN_TEXTS = ['非表示','非公開','停止中','無効','休止','退店','削除','未掲載'];
                    const VISIBLE_TEXTS = ['表示中','公開中','掲載中','有効','表示','公開'];
                    for (const row of Array.from(document.querySelectorAll('table tr, .list-item, .cast-row, .item-row'))) {{
                        const links = Array.from(row.querySelectorAll('a[href]'));
                        const idLink = links.find(a => a.href.startsWith(origin) && (ID_PAT.test(a.href) || /\\/\\d{{3,}}/.test(a.href)));
                        if (!idLink) continue;
                        const emEl = row.querySelector('em, strong, b');
                        let name = emEl ? (emEl.textContent || '').replace(/\\s+/g, ' ').trim() : '';
                        if (!name) {{
                            for (const td of Array.from(row.querySelectorAll('td')).slice(0, 6)) {{
                                const tok = (td.textContent || '').replace(/\\s+/g, ' ').trim().split(/[\\s（(]/)[0];
                                if (tok && tok.length >= 2 && tok.length <= 20 && !/^[\\d]+$/.test(tok)) {{ name = tok; break; }}
                            }}
                        }}
                        if (!name) continue;
                        const key = name + '|' + idLink.href;
                        if (seen.has(key)) continue;
                        seen.add(key);
                        // 非表示判定（exact class matching / 親伝播false positive回避）
                        const rowClasses = Array.from(row.classList).map(c => c.toLowerCase());
                        const rowStyle = (row.getAttribute('style') || '').toLowerCase();
                        const classHidden = HIDDEN_CLASSES.some(c => rowClasses.includes(c));
                        const styleHidden = rowStyle.includes('display:none') || rowStyle.includes('display: none') ||
                            rowStyle.includes('visibility:hidden') || /opacity\\s*:\\s*0\\./.test(rowStyle);
                        const statusTexts = Array.from(row.querySelectorAll('td, span, em, strong, b, label'))
                            .filter(el => !el.closest('a,button'))
                            .map(el => (el.textContent || '').replace(/\\s+/g,'').trim())
                            .filter(t => t && t.length <= 12);
                        const visibleText = statusTexts.some(t => VISIBLE_TEXTS.includes(t));
                        const hiddenText = statusTexts.some(t => HIDDEN_TEXTS.includes(t));
                        const textHidden = hiddenText && !visibleText;
                        // computedStyle.display===none は親伝播で全件ヒットするため除外
                        // visibility:hidden のみ（行自体に直接設定されている場合のみ有効）
                        const computedHidden = row.style.visibility === 'hidden';
                        const hidden = classHidden || styleHidden || textHidden || computedHidden;
                        out.push({{name: name.substring(0, 60), url: idLink.href, hidden: hidden}});
                    }}
                    // 表示中を先、非表示を後に並べて上限適用
                    out.sort((a, b) => (a.hidden ? 1 : 0) - (b.hidden ? 1 : 0));
                    return out.slice(0, {max_items});
                }}"""
                entities = []
                try:
                    entities = page.evaluate(_SIMPLE_JS) or []
                    print(f"[EXTRACT_ENTITY_LIST_SIMPLE] extracted={len(entities)}", flush=True)
                except Exception as _se:
                    print(f"[EXTRACT_ENTITY_LIST_SIMPLE] {type(_se).__name__}: {_se}", flush=True)

                # Step 3: 0件ならAIにテーブルHTMLのみ送ってセレクターを聞く
                if not entities:
                    try:
                        _tbl_html = page.evaluate("""() => {
                            const t = document.querySelector('table.basic, table, [class*="cast-list"], [class*="list-table"]');
                            return t ? t.outerHTML.substring(0, 12000) : '';
                        }""") or ""
                        if len(_tbl_html) > 200:
                            from api.core.llm_client import call_llm_json
                            _ai = call_llm_json(
                                prompt=f'HTMLテーブルから「{entity_label}」を抽出するCSSセレクターをJSONで返してください。\nHTML:\n{_tbl_html}\n\n{{"has_list":true,"row_selector":"...","name_selector":"...","url_selector":"..."}}',
                                system_prompt="JSONのみ出力。説明不要。",
                                ai_tier="core", max_tokens=300,
                            )
                            if isinstance(_ai, dict) and _ai.get("has_list") and _ai.get("row_selector"):
                                _rs = str(_ai.get("row_selector") or "")
                                _ns = str(_ai.get("name_selector") or "")
                                _us = str(_ai.get("url_selector") or "")
                                _nc2 = f'row.querySelector({_json_sel.dumps(_ns)})' if _ns else 'row.querySelector("em,strong,b")'
                                _uc2 = f'row.querySelector({_json_sel.dumps(_us)})' if _us else 'row.querySelector("a[href]")'
                                print(f"[EXTRACT_ENTITY_LIST_AI] row={_rs} name={_ns} url={_us}", flush=True)
                                entities = page.evaluate(f"""() => {{
                                    const origin = location.origin;
                                    const out = [];
                                    const seen = new Set();
                                    for (const row of Array.from(document.querySelectorAll({_json_sel.dumps(_rs)}))) {{
                                        const nameEl = {_nc2};
                                        const urlEl = {_uc2};
                                        const name = nameEl ? (nameEl.textContent || '').replace(/\\s+/g, ' ').trim() : '';
                                        const url = urlEl ? (urlEl.href || '') : '';
                                        if (!name || !url || !url.startsWith(origin)) continue;
                                        const key = name + '|' + url;
                                        if (seen.has(key)) continue;
                                        seen.add(key);
                                        out.push({{name: name.substring(0, 60), url}});
                                    }}
                                    return out.slice(0, {max_items});
                                }}""") or []
                                print(f"[EXTRACT_ENTITY_LIST_AI] extracted={len(entities)}", flush=True)
                    except Exception as _ae:
                        print(f"[EXTRACT_ENTITY_LIST_AI] {type(_ae).__name__}: {_ae}", flush=True)

                # Step 4: ページネーション
                if entities:
                    _pag_visited = {page.url.rstrip("/")}
                    for _pg in range(8):
                        try:
                            _nxt = page.evaluate("""() => {
                                const o = location.origin;
                                for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                                    const t=(a.textContent||'').replace(/\s+/g,' ').trim();
                                    if (!a.href.startsWith(o)) continue;
                                    if (/^(次へ|次|next|>>|›|▶)$/i.test(t)||a.rel==='next') return a.href;
                                }
                                return null;
                            }""")
                            if not _nxt or _nxt.rstrip("/") in _pag_visited:
                                break
                            _pag_visited.add(_nxt.rstrip("/"))
                            page.goto(_nxt, timeout=20000, wait_until="domcontentloaded")
                            page.wait_for_timeout(700)
                            _pg_ents = page.evaluate(_SIMPLE_JS) or []
                            _ex = {e["url"] for e in entities}
                            added = 0
                            for ne in _pg_ents:
                                if ne["url"] not in _ex:
                                    entities.append(ne)
                                    _ex.add(ne["url"])
                                    added += 1
                            print(f"[EXTRACT_ENTITY_LIST_PAGE] page {_pg+2}: +{added}, total={len(entities)}", flush=True)
                            if len(entities) >= max_items:
                                break
                        except Exception:
                            break

                try:
                    browser.close()
                except Exception:
                    pass

                if not entities:
                    return {"ok": True, "status": "EMPTY", "message": "エンティティ候補が見つかりませんでした", "entities": [], "current_url": current_url}

                return {"ok": True, "status": "OK", "entities": entities[:max_items],
                        "count": len(entities), "current_url": current_url}

                # ----- 以下は旧推測セレクター方式（参照のみ・実行されない） -----
                raw_links = []
                try:
                    raw_links = page.evaluate(
                        """() => {
                            const origin = location.origin;
                            const out = [];
                            const seen = new Set();
                            for (const a of Array.from(document.querySelectorAll('a[href]')).slice(0, 600)) {
                                let href = a.href || '';
                                if (!href.startsWith(origin)) continue;
                                const text = (a.textContent || '').replace(/\\s+/g,' ').trim();
                                if (!text || text.length < 1 || text.length > 30) continue;
                                // ナビ/汎用リンク/UIアクション除外
                                if (/ログアウト|ヘルプ|FAQ|お問い合わせ|TOP|戻る|次へ|前へ|一覧|新規|登録|設定|検索|編集|削除|保存|更新|キャンセル|確認|詳細|追加|変更|入力|送信|完了|閉じる/.test(text)) continue;
                                const key = text + '|' + href;
                                if (seen.has(key)) continue;
                                seen.add(key);
                                // id/数値パラメータを持つリンクを優先的にスコア付け
                                const hasId = /[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i.test(href) || /\\/\\d{2,}/.test(href);
                                out.push({ name: text, url: href, score: hasId ? 2 : 1 });
                            }
                            return out.slice(0, 200);
                        }"""
                    ) or []
                except Exception:
                    raw_links = []

                # テーブル行からも名前候補を補完（<em>タグ優先、次に最初トークン）
                try:
                    table_links = page.evaluate(
                        """() => {
                            const origin = location.origin;
                            const out = [];
                            const seen = new Set();
                            const ACTION = /編集|削除|保存|更新|キャンセル|確認|詳細|追加|変更|入力|送信|完了|一覧|設定|フォーム|ピックアップ|管理|配信|ログアウト|ヘルプ|FAQ|お問い合わせ|登録|新規|検索|TOP|トップ|メニュー|戻る|次へ|前へ|有効|無効|表示|非表示/;
                            for (const row of Array.from(document.querySelectorAll('table tr, ul li, .list-item, .cast-row, .staff-row, .item-row'))) {
                                const links = Array.from(row.querySelectorAll('a[href]'));
                                // 同行のIDパラメータ持ちリンクを優先して選ぶ
                                const idLink = links.find(a => {
                                    const h = a.href || '';
                                    return h.startsWith(origin) && (/[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i.test(h) || /\\/\\d{3,}/.test(h));
                                });
                                const anyLink = links.find(a => (a.href || '').startsWith(origin));
                                const rowLink = idLink || anyLink;
                                if (!rowLink) continue;
                                const href = rowLink.href;
                                let name = '';
                                // 優先1: <em>タグ（多くのCMSはキャスト名をemで囲む）
                                const emEl = row.querySelector('em');
                                if (emEl) {
                                    const t = (emEl.textContent || '').replace(/\\s+/g,' ').trim();
                                    if (t && t.length >= 1 && t.length <= 30 && !ACTION.test(t) && !/^[\\d\\s:/-]+$/.test(t)) name = t;
                                }
                                // 優先2: strong/bタグ
                                if (!name) {
                                    const boldEl = row.querySelector('strong, b');
                                    if (boldEl) {
                                        const t = (boldEl.textContent || '').replace(/\\s+/g,' ').trim();
                                        if (t && t.length >= 1 && t.length <= 30 && !ACTION.test(t) && !/^[\\d\\s:/-]+$/.test(t)) name = t;
                                    }
                                }
                                // 優先3: セルテキストの最初のトークン（括弧・スペース前）
                                if (!name) {
                                    const cells = Array.from(row.querySelectorAll('td, th')).slice(0, 8);
                                    for (const cell of cells) {
                                        const full = (cell.textContent || '').replace(/\\s+/g,' ').trim();
                                        const tok = full.split(/[\\s（(\\d]/)[0];
                                        if (tok && tok.length >= 2 && tok.length <= 20 && !ACTION.test(tok) && !/^[\\d\\s:/-]+$/.test(tok)) {
                                            name = tok; break;
                                        }
                                    }
                                }
                                if (!name) continue;
                                const key = name + '|' + href;
                                if (seen.has(key)) continue;
                                seen.add(key);
                                out.push({ name, url: href, score: 3 });
                            }
                            return out.slice(0, 200);
                        }"""
                    ) or []
                    for tl in table_links:
                        key = tl.get("name", "") + "|" + tl.get("url", "")
                        existing_keys = set(x.get("name", "") + "|" + x.get("url", "") for x in raw_links)
                        if key not in existing_keys:
                            raw_links.append(tl)
                except Exception:
                    pass

                # ページネーション対応（「次へ」系リンクを最大8ページまで追跡）
                _pag_visited = {page.url.rstrip("/")}
                for _pg in range(8):
                    try:
                        _next_url = page.evaluate("""() => {
                            const origin = location.origin;
                            for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                                const t = (a.textContent||'').replace(/\\s+/g,' ').trim();
                                const h = a.href||'';
                                if (!h.startsWith(origin)) continue;
                                if (/^(次へ|次|next|>>|›|‹›|▶)$/i.test(t) || a.rel==='next') return h;
                            }
                            return null;
                        }""")
                        if not _next_url or _next_url.rstrip("/") in _pag_visited:
                            break
                        _pag_visited.add(_next_url.rstrip("/"))
                        page.goto(_next_url, timeout=20000, wait_until="domcontentloaded")
                        page.wait_for_timeout(700)
                        _pg_ACTION = r"/編集|削除|保存|更新|キャンセル|確認|詳細|追加|変更|入力|送信|完了|一覧|設定|フォーム|ピックアップ|管理|配信|ログアウト|ヘルプ|FAQ|お問い合わせ|登録|新規|検索|TOP|トップ|メニュー|戻る|次へ|前へ/"
                        _pg_table = page.evaluate(f"""() => {{
                            const origin = location.origin;
                            const out = [];
                            const seen = new Set();
                            const ACTION = {_pg_ACTION};
                            for (const row of Array.from(document.querySelectorAll('table tr, ul li, .list-item, .cast-row, .staff-row, .item-row'))) {{
                                const links = Array.from(row.querySelectorAll('a[href]'));
                                const idLink = links.find(a => {{ const h=a.href||''; return h.startsWith(origin)&&(/[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i.test(h)||/\\/\\d{{3,}}/.test(h)); }});
                                const anyLink = links.find(a=>(a.href||'').startsWith(origin));
                                const rowLink = idLink||anyLink;
                                if (!rowLink) continue;
                                const href = rowLink.href;
                                const cells = Array.from(row.querySelectorAll('td, th, span, div')).slice(0,6);
                                let name='';
                                for (const cell of cells) {{
                                    const t=(cell.textContent||'').replace(/\\s+/g,' ').trim();
                                    if (t&&t.length>=2&&t.length<=30&&!ACTION.test(t)&&!/^[\\d\\s:/-]+$/.test(t)) {{ name=t; break; }}
                                }}
                                if (!name) continue;
                                const key=name+'|'+href;
                                if (seen.has(key)) continue;
                                seen.add(key);
                                out.push({{name,url:href,score:3}});
                            }}
                            return out.slice(0,100);
                        }}""") or []
                        _existing_keys = {x.get("name","") + "|" + x.get("url","") for x in raw_links}
                        for nl in _pg_table:
                            k = nl.get("name","") + "|" + nl.get("url","")
                            if k not in _existing_keys:
                                raw_links.append(nl)
                                _existing_keys.add(k)
                        print(f"[EXTRACT_ENTITY_LIST_PAGINATION] page {_pg+2}: +{len(_pg_table)} links, total={len(raw_links)}", flush=True)
                        if len(raw_links) >= max_items:
                            break
                    except Exception as _pe:
                        print(f"[EXTRACT_ENTITY_LIST_PAGINATION] page {_pg+2} error: {type(_pe).__name__}", flush=True)
                        break

                # score≥2のリンクが0件なら、ナビからエンティティ系リンクをたどる
                _has_id_links = any(l.get("score", 0) >= 2 for l in raw_links)
                if not _has_id_links:
                    try:
                        _entity_nav = page.evaluate(
                            r"""() => {
                                const origin = location.origin;
                                const hrefKws = ['cast', 'girl', 'gal', 'member', 'staff', 'lady', 'talent', 'model', 'worker'];
                                const listKws = ['list', '_list', 'index', 'ichiran'];
                                const textKws = ['キャスト', '女の子', 'ガール', 'メンバー', 'スタッフ', 'レディ', '嬢', '在籍', '求人', 'cast', 'girl', 'member'];
                                const links = Array.from(document.querySelectorAll('a[href]'));
                                // 優先1: hrefにエンティティ種別 + listキーワード含む
                                for (const a of links) {
                                    const h = (a.href || '').toLowerCase();
                                    if (!a.href.startsWith(origin)) continue;
                                    for (const ek of hrefKws) {
                                        if (h.includes(ek) && listKws.some(lk => h.includes(lk))) return a.href;
                                    }
                                }
                                // 優先2: テキストにエンティティ種別キーワード含む
                                for (const a of links) {
                                    const t = (a.textContent || '').replace(/\s+/g,' ').trim();
                                    const h = a.href || '';
                                    if (!h.startsWith(origin)) continue;
                                    for (const kw of textKws) {
                                        if (t.includes(kw)) return h;
                                    }
                                }
                                // 優先3: hrefにエンティティ種別キーワードのみ
                                for (const a of links) {
                                    const h = (a.href || '').toLowerCase();
                                    if (!a.href.startsWith(origin)) continue;
                                    for (const ek of hrefKws) {
                                        if (h.includes(ek)) return a.href;
                                    }
                                }
                                return null;
                            }"""
                        )
                        if _entity_nav:
                            print(f"[EXTRACT_ENTITY_LIST] nav→entity page: {_entity_nav}", flush=True)
                            page.goto(_entity_nav, timeout=25000, wait_until="domcontentloaded")
                            page.wait_for_timeout(1200)
                            current_url = page.url
                            # さらに「一覧」リンクがあれば1ホップ追加（サブメニューページ対策）
                            try:
                                _list_hop = page.evaluate(
                                    """() => {
                                        const origin = location.origin;
                                        for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                                            const text = (a.textContent || '').replace(/\\s+/g,' ').trim();
                                            const href = a.href || '';
                                            if (!href.startsWith(origin)) continue;
                                            if (/^(一覧|在籍一覧|キャスト一覧|スタッフ一覧|リスト|girl list|cast list)$/i.test(text)) return href;
                                        }
                                        return null;
                                    }"""
                                )
                                if _list_hop:
                                    print(f"[EXTRACT_ENTITY_LIST] nav→list hop: {_list_hop}", flush=True)
                                    page.goto(_list_hop, timeout=25000, wait_until="domcontentloaded")
                                    page.wait_for_timeout(1200)
                                    current_url = page.url
                            except Exception:
                                pass
                            _nav_links = page.evaluate(
                                """() => {
                                    const origin = location.origin;
                                    const out = [];
                                    const seen = new Set();
                                    const ACTION = /編集|削除|保存|更新|キャンセル|確認|詳細|追加|変更|入力|送信|完了|一覧|設定|フォーム|ピックアップ|管理|配信|ログアウト|ヘルプ|FAQ|お問い合わせ|登録|新規|検索|TOP|トップ|メニュー|戻る|次へ|前へ/;
                                    for (const row of Array.from(document.querySelectorAll('table tr, ul li, .list-item, .cast-row, .staff-row, .item-row'))) {
                                        const links = Array.from(row.querySelectorAll('a[href]'));
                                        const idLink = links.find(a => {
                                            const h = a.href || '';
                                            return h.startsWith(origin) && (/[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i.test(h) || /\\/\\d{3,}/.test(h));
                                        });
                                        const anyLink = links.find(a => (a.href || '').startsWith(origin));
                                        const rowLink = idLink || anyLink;
                                        if (!rowLink) continue;
                                        const href = rowLink.href;
                                        const cells = Array.from(row.querySelectorAll('td, th, span, div')).slice(0, 6);
                                        let name = '';
                                        for (const cell of cells) {
                                            const t = (cell.textContent || '').replace(/\\s+/g,' ').trim();
                                            if (t && t.length >= 2 && t.length <= 30 && !ACTION.test(t) && !/^[\\d\\s:/-]+$/.test(t)) {
                                                name = t; break;
                                            }
                                        }
                                        if (!name) continue;
                                        const key = name + '|' + href;
                                        if (seen.has(key)) continue;
                                        seen.add(key);
                                        out.push({ name, url: href, score: 3 });
                                    }
                                    // fallback: plain id-param links
                                    if (!out.length) {
                                        for (const a of Array.from(document.querySelectorAll('a[href]')).slice(0, 400)) {
                                            const h = a.href || '';
                                            if (!h.startsWith(origin)) continue;
                                            const hasId = /[?&](id|girl|cast|gid|cid|member|staff|no|seq)=/i.test(h) || /\\/\\d{3,}/.test(h);
                                            if (!hasId) continue;
                                            const text = (a.textContent || '').replace(/\\s+/g,' ').trim();
                                            if (!text || text.length < 1 || text.length > 30) continue;
                                            const key = text + '|' + h;
                                            if (seen.has(key)) continue;
                                            seen.add(key);
                                            out.push({ name: text, url: h, score: 2 });
                                        }
                                    }
                                    return out.slice(0, 200);
                                }"""
                            ) or []
                            for nl in _nav_links:
                                key = nl.get("name", "") + "|" + nl.get("url", "")
                                existing_keys = set(x.get("name", "") + "|" + x.get("url", "") for x in raw_links)
                                if key not in existing_keys:
                                    raw_links.append(nl)
                    except Exception as _nav_e:
                        print(f"[EXTRACT_ENTITY_LIST_NAV_ERROR] {type(_nav_e).__name__}: {_nav_e}", flush=True)

                try:
                    browser.close()
                except Exception:
                    pass

                if not raw_links:
                    return {"ok": True, "status": "EMPTY", "message": "エンティティ候補が見つかりませんでした", "entities": [], "current_url": current_url}

                # Geminiで「実在エンティティ（人名/商品名等）」だけに絞り込む
                entities = []
                try:
                    from api.core.llm_client import call_llm_json
                    import json as _json
                    _cand = sorted(raw_links, key=lambda x: -x.get("score", 0))[:120]
                    _prompt = f"""次は管理画面の「{entity_label}一覧」ページから抽出したリンク候補です。
この中から実在する{entity_label}（個別の人物名/商品名など、編集対象になる固有のエンティティ）だけを選び、
ナビゲーション・カテゴリ・汎用リンクは除外してください。

【リンク候補（name, url）】
{_json.dumps([{ 'name': c['name'], 'url': c['url'] } for c in _cand], ensure_ascii=False)}

[{{"name":"...","url":"..."}}] 形式のJSON配列のみ返してください。最大{max_items}件。"""
                    _res = call_llm_json(
                        prompt=_prompt,
                        system_prompt="JSON配列のみ出力。説明文やMarkdown禁止。",
                        ai_tier="core",
                        max_tokens=4096,
                    )
                    if isinstance(_res, list):
                        for it in _res[:max_items]:
                            if isinstance(it, dict) and it.get("name") and it.get("url"):
                                entities.append({"name": str(it["name"])[:60], "url": str(it["url"])})
                    elif isinstance(_res, dict) and isinstance(_res.get("entities"), list):
                        for it in _res["entities"][:max_items]:
                            if isinstance(it, dict) and it.get("name") and it.get("url"):
                                entities.append({"name": str(it["name"])[:60], "url": str(it["url"])})
                except Exception as _le:
                    print(f"[EXTRACT_ENTITY_LIST_LLM_ERROR] {type(_le).__name__}", flush=True)

                # LLM失敗時はid付きリンク上位をそのまま返す
                if not entities:
                    for c in sorted(raw_links, key=lambda x: -x.get("score", 0))[:max_items]:
                        if c.get("score", 0) >= 2:
                            entities.append({"name": c["name"], "url": c["url"]})

                return {"ok": True, "status": "OK", "entities": entities[:max_items],
                        "count": len(entities), "current_url": current_url}
            except Exception as e:
                try:
                    if browser:
                        browser.close()
                except Exception:
                    pass
                print(f"[EXTRACT_ENTITY_LIST_ERROR] url={list_url[:60]} {type(e).__name__}", flush=True)
                return {"ok": False, "status": "FAILED", "message": str(e), "entities": []}
    except Exception as e:
        print(f"[EXTRACT_ENTITY_LIST_OUTER_ERROR] {type(e).__name__}", flush=True)
        return {"ok": False, "status": "FAILED", "message": str(e), "entities": []}


def deep_scan_operation(media_mapping: dict, operation_type: str, hint_url: str = "") -> dict:
    """
    P23/P24 deep scan wrapper.
    正規経路:
    media_mappings.navigation_graph.pages
    → build_operation_mappings_from_dom_evidence
    → operation_mappings[operation_type]

    run_dom_scan の戻り値から pages/admin_pages は参照しない。
    Firestore保存は呼び出し側(agent.py)で行う。
    """
    import datetime as _dt

    mapping_id = str(
        media_mapping.get("id")
        or media_mapping.get("mapping_id")
        or ""
    )

    req = OPERATION_REQUIREMENTS.get(operation_type, {})
    required = req.get("required") or []

    def _empty(status: str, reason: str, error: str = "") -> dict:
        r = {
            "status": status,
            "selectors": {},
            "missing": required,
            "target_url": None,
            "validation_score": 0,
            "executable": False,
            "source": "deep_scan_operation",
            "error_reason": reason,
            "last_scanned_at": _dt.datetime.utcnow().isoformat(),
        }
        if error:
            r["error"] = error
            r["error_message"] = error
        return r

    if not media_mapping:
        return _empty("ERROR", "media_mapping_empty", "media_mapping is empty")

    if not mapping_id:
        return _empty("ERROR", "mapping_id_missing", "mapping_id missing")

    try:
        from api.core.firestore_client import get_db as _get_db_ds
        db = _get_db_ds()
    except Exception as e:
        return _empty("ERROR", "firestore_unavailable", type(e).__name__)

    def _load_pages() -> list:
        try:
            snap = db.collection("media_mappings").document(mapping_id).get()
            doc = snap.to_dict() or {} if snap.exists else {}
            nav = doc.get("navigation_graph") or {}
            pages = nav.get("pages") or []
            if pages:
                return pages
            # 後方互換: navigation_graph が url=>summary dict 形式の場合
            if isinstance(nav, dict):
                converted = []
                for k, v in nav.items():
                    if k in ("pages", "updated_at", "__meta__"):
                        continue
                    if isinstance(v, dict):
                        item = dict(v)
                        item.setdefault("url", item.get("url") or k)
                        converted.append(item)
                return converted
        except Exception as e:
            print(f"[P23_DEEP_SCAN_LOAD_PAGES_ERROR] mapping_id={mapping_id} error={type(e).__name__}", flush=True)
        return []

    pages = _load_pages()

    # navigation_graph.pages が無い場合のみ dom_scan を実行して保存済みnavigation_graphを再読込
    if not pages:
        try:
            print(f"[P23_DEEP_SCAN_DOM_SCAN_FALLBACK] mapping_id={mapping_id} op={operation_type}", flush=True)
            run_dom_scan(
                media_mapping,
                start_url=hint_url or "",
                reset_resume=True,
            )
            pages = _load_pages()
        except Exception as e:
            return _empty("ERROR", "dom_scan_failed", type(e).__name__)

    if not pages:
        return _empty("UNDISCOVERED", "navigation_graph_pages_empty")

    try:
        all_mappings = build_operation_mappings_from_dom_evidence(mapping_id, pages)
    except Exception as e:
        return _empty("ERROR", "build_operation_mappings_failed", type(e).__name__)

    result = all_mappings.get(operation_type)
    if not result:
        return _empty("UNDISCOVERED", "operation_mapping_not_found")

    # 保存互換のため必須キーを補完
    result = dict(result)
    result.setdefault("status", "UNDISCOVERED")
    result.setdefault("selectors", {})
    result.setdefault("missing", required)
    result.setdefault("target_url", None)
    result.setdefault("validation_score", 0)
    result.setdefault("source", "deep_scan_operation")
    result.setdefault("executable", result.get("status") == "READY")
    result["last_scanned_at"] = _dt.datetime.utcnow().isoformat()

    print(
        f"[P23_DEEP_SCAN_RESULT] mapping_id={mapping_id} op={operation_type} "
        f"status={result.get('status')} target_url={str(result.get('target_url'))[:80]} "
        f"score={result.get('validation_score')} missing={result.get('missing')}",
        flush=True,
    )
    return result

def rebuild_operation_steps(operation_types: list, nav_graph: dict, op_mappings: dict, detail_by_op: dict) -> dict:
    """
    P23/P24 → P14 実行貫通用 operation_steps 再生成。

    重要:
    生成する step_type は _execute_operation_steps が実行できる型だけに限定する。
    対応step_type:
      login / navigate / fill / click / upload_file / verify / sleep

    旧未対応step:
      navigate_to_news / input_title / input_body / save / verify_post 等は生成しない。
    """
    REQUIRED_SELECTORS = {
        "news_post":       ["body", "save"],
        "blog_post":       ["body", "save"],
        "text_update":     ["body", "save"],
        "status_update":   ["body", "save"],
        "media_replace":   ["file", "save"],
        "schedule_update": ["save"],
        "price_update":    ["price", "save"],
        "entity_register": ["save"],
        "entity_update":   ["save"],
    }

    NAV_KEYWORDS = {
        "news_post":       ["news", "post", "blog", "diary", "topic", "topics", "event", "coupon", "お知らせ", "ニュース", "投稿", "新規", "写メ", "freetext", "contents", "campaign", "realtime", "marquee", "速報"],
        "blog_post":       ["blog", "tencho", "tenchoblog", "店長", "ブログ", "求人", "recruit", "オーナー", "owner", "diary", "コラム", "記事"],
        "text_update":     ["profile", "about", "text", "プロフィール", "自己紹介", "説明", "freetext", "contents", "con_txt", "seo", "concept", "フリー", "編集", "ページ編集"],
        "status_update":   ["status", "public", "private", "表示", "非表示", "公開", "停止", "有効", "無効", "ステータス", "state", "visible", "hidden", "active", "inactive"],
        "media_replace":   ["photo", "image", "gallery", "写真", "画像", "メディア"],
        "schedule_update": ["schedule", "shift", "出勤", "予定", "calendar"],
        "price_update":    ["price", "course", "料金", "コース", "fee"],
        "entity_register": ["register", "new", "add", "登録", "新規追加"],
        "entity_update":   ["edit", "update", "編集", "更新"],
    }

    # operation_type → 実行field定義
    # selector_key は operation_mappings.selectors / dom_selectors に存在するrole名。
    # payload_key は task.payload のキー。
    OP_FIELD_STEPS = {
        "news_post": [
            {"selector_key": "title", "payload_key": "title", "step_type": "fill", "required": False},
            {"selector_key": "body",  "payload_key": "body",  "step_type": "fill", "required": True},
        ],
        "blog_post": [
            {"selector_key": "title", "payload_key": "title", "step_type": "fill", "required": False},
            {"selector_key": "body",  "payload_key": "body",  "step_type": "fill", "required": True},
        ],
        "text_update": [
            {"selector_key": "body", "payload_key": "text", "step_type": "fill", "required": True},
        ],
        "status_update": [
            {"selector_key": "body", "payload_key": "body", "step_type": "fill", "required": True},
        ],
        "media_replace": [
            {"selector_key": "file", "payload_key": "file_path", "step_type": "upload_file", "required": True},
        ],
        "schedule_update": [
            {"selector_key": "date_input", "payload_key": "schedule_value", "step_type": "fill", "required": False},
        ],
        "price_update": [
            {"selector_key": "price", "payload_key": "price_value", "step_type": "fill", "required": True},
        ],
        "entity_register": [
            {"selector_key": "required_inputs", "payload_key": "name", "step_type": "fill", "required": False},
        ],
        "entity_update": [
            {"selector_key": "editable_inputs", "payload_key": "value", "step_type": "fill", "required": False},
        ],
    }

    result = {}

    def _selector_exists(avail_sel: dict, key: str) -> bool:
        val = avail_sel.get(key)
        if isinstance(val, dict):
            return bool(val.get("selector"))
        return bool(val)

    def _abs_url(u, base):
        try:
            from urllib.parse import urljoin
            if not u:
                return u
            if str(u).startswith("http"):
                return u
            if base:
                return urljoin(base, u)
            return u
        except Exception:
            return u

    def _is_c1main(u):
        return bool(u and "C1Main.php" in str(u))

    for op_type in operation_types:
        op_map = op_mappings.get(op_type, {}) if isinstance(op_mappings, dict) else {}
        executable = op_map.get("executable", None)
        status = op_map.get("status", "")
        if status not in ("READY", "NEEDS_REVIEW"):
            print(f"[OP_STEPS_SKIPPED_NOT_TARGET] op_type={op_type} executable={executable} status={status}", flush=True)
            continue

        avail_sel = op_map.get("selectors", {}) or {}
        req_sels = REQUIRED_SELECTORS.get(op_type, ["save"])
        missing = [k for k in req_sels if not _selector_exists(avail_sel, k)]

        # target_url 優先順位:
        # ① nav_graph.__meta__.operation_entrypoints
        # ② detail_by_op.source_url
        # ③ op_map.p24_source_url
        # ④ op_map.target_url
        # ⑤ keyword fallback（ただしkeywordだけならUNDISCOVERED扱い）
        struct_map = {}
        try:
            struct_map = (nav_graph.get("__meta__", {}).get("operation_entrypoints") or {}) if isinstance(nav_graph, dict) else {}
        except Exception:
            struct_map = {}

        struct_url = (struct_map.get(op_type) or {}).get("url", "")
        detail_url = (detail_by_op or {}).get(op_type, {}).get("source_url", "") if isinstance(detail_by_op, dict) else ""
        p24_url = op_map.get("p24_source_url", "")
        map_url = op_map.get("target_url", "")

        media_base = ""
        if isinstance(nav_graph, dict):
            _all_pages_kw = list(nav_graph.get("pages") or [])
            for _k, _v in nav_graph.items():
                if _k in ("pages", "updated_at", "manual_import", "__meta__"):
                    continue
                if isinstance(_v, dict) and _v.get("url", "").startswith("http"):
                    _all_pages_kw.append(_v)
            for _pg_mb in _all_pages_kw:
                if isinstance(_pg_mb, dict) and _pg_mb.get("url", "").startswith("http"):
                    media_base = _pg_mb["url"]
                    break

        struct_url = _abs_url(struct_url, media_base)
        detail_url = _abs_url(detail_url, media_base)
        p24_url = _abs_url(p24_url, media_base)
        map_url = _abs_url(map_url, media_base)

        keywords = NAV_KEYWORDS.get(op_type, [])
        kw_url = None
        kw_url_fallback = None
        if isinstance(nav_graph, dict):
            for _pg_kw in _all_pages_kw:
                if not isinstance(_pg_kw, dict):
                    continue
                ul = str(_pg_kw.get("url", "")).lower()
                tl = str(_pg_kw.get("title", "")).lower()
                if any(kw in ul or kw in tl for kw in keywords):
                    if _is_c1main(_pg_kw.get("url", "")):
                        kw_url_fallback = _pg_kw.get("url", "")
                    else:
                        kw_url = _pg_kw.get("url", "")
                        break
            if kw_url is None and kw_url_fallback:
                kw_url = kw_url_fallback

        selected_src = "none"
        target_url = None

        for src_name, cand in (
            ("structure_entrypoint", struct_url),
            ("map_target_url", map_url),
            ("p24_source_url", p24_url),
            ("detail_source_url", detail_url),
        ):
            if cand and str(cand).startswith("http") and not _is_c1main(cand):
                target_url = cand
                selected_src = src_name
                break

        if not target_url and detail_url and str(detail_url).startswith("http"):
            target_url = detail_url
            selected_src = "detail_source_url(c1main_fallback)"
        elif not target_url and kw_url:
            target_url = kw_url
            selected_src = "keyword_match"

        print(
            f"[P24_TARGET_PRIORITY] op={op_type} "
            f"detail_source_url={detail_url} p24_source_url={p24_url} "
            f"mapping_target_url={map_url} selected_target_url={target_url} selected_src={selected_src}",
            flush=True,
        )

        if not target_url:
            print(f"[P24_TARGET_REJECTED] op={op_type} candidate_url=None reason=no_target_url", flush=True)
            result[op_type] = []
            continue

        if selected_src == "keyword_match":
            print(f"[P24_TARGET_REJECTED] op={op_type} candidate_url={target_url} reason=keyword_only_target_rejected", flush=True)
            result[op_type] = []
            continue

        computed_status = status or "UNDISCOVERED"
        eligible = (
            computed_status == "READY"
            or (computed_status == "NEEDS_REVIEW" and target_url)
        )

        print(
            f"[P24_STATUS_DECISION_FROM_MAPPING] op={op_type} status={computed_status} "
            f"score={op_map.get('validation_score', 0)} missing={missing} target_url={target_url}",
            flush=True,
        )

        if not eligible:
            result[op_type] = []
            continue

        steps = []
        order = 0

        # login: 認証済みpage生成後なので実処理なし。ただしgraph上の明示stepとして残す。
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
        })
        order += 1

        # entity_register / entity_update: expand all mapped fields dynamically
        # (entity forms have many fields: name, birthday, height, etc.)
        # required_inputs/editable_inputs are alias copies of "name" — exclude to avoid double-fill
        _ENTITY_CTRL_KEYS = {"save", "submit", "edit_trigger", "delete", "back", "cancel", "search", "filter",
                             "required_inputs", "editable_inputs"}
        if op_type in ("entity_register", "entity_update"):
            _dyn_fields = []
            for _esk, _esv in avail_sel.items():
                if _esk in _ENTITY_CTRL_KEYS:
                    continue
                if not _selector_exists(avail_sel, _esk):
                    continue
                _est = "upload_file" if (isinstance(_esv, dict) and _esv.get("type") == "file") else "fill"
                _dyn_fields.append({
                    "selector_key": _esk,
                    "payload_key": _esk,
                    "step_type": _est,
                    "required": _esk in ("name", "required_inputs", "editable_inputs"),
                })
            _field_steps_src = _dyn_fields if _dyn_fields else OP_FIELD_STEPS.get(op_type, [])
        else:
            # 静的ステップに加え、mapping取得フィールドのうち未カバー分を動的追加
            # （payloadにキーがあれば実行、なければスキップ — required=False 保証）
            _static_steps = OP_FIELD_STEPS.get(op_type, [])
            _covered_skeys = {s["selector_key"] for s in _static_steps}
            _extra_steps = []
            for _xsk, _xsv in avail_sel.items():
                if _xsk in _ENTITY_CTRL_KEYS or _xsk in _covered_skeys:
                    continue
                if not _selector_exists(avail_sel, _xsk):
                    continue
                _xst = "upload_file" if (isinstance(_xsv, dict) and _xsv.get("type") == "file") else "fill"
                _extra_steps.append({
                    "selector_key": _xsk,
                    "payload_key": _xsk,
                    "step_type": _xst,
                    "required": False,
                })
            _field_steps_src = _static_steps + _extra_steps

        for field in _field_steps_src:
            skey = field["selector_key"]
            # optional titleなど、selectorがない任意stepは生成しない
            if not _selector_exists(avail_sel, skey) and not field.get("required", True):
                print(f"[P24_OPTIONAL_STEP_SKIPPED] op={op_type} selector_key={skey} reason=selector_missing_optional", flush=True)
                continue

            step_status = computed_status if _selector_exists(avail_sel, skey) else "FAILED"
            step = {
                "order": order,
                "step_id": f"{op_type}_{field['step_type']}_{skey}",
                "step_type": field["step_type"],
                "display_name": "入力" if field["step_type"] == "fill" else "ファイルアップロード",
                "status": step_status,
                "required": bool(field.get("required", True)),
                "source_url": target_url,
                "target_url": target_url,
                "selector_key": skey,
                "payload_key": field["payload_key"],
                "selector": avail_sel.get(skey),
            }
            if missing:
                step["missing_required_fields"] = missing
            steps.append(step)
            order += 1

        save_key = "save" if _selector_exists(avail_sel, "save") else "submit"
        steps.append({
            "order": order,
            "step_id": f"{op_type}_click_save",
            "step_type": "click",
            "display_name": "保存",
            "required": True,
            "terminal": True,
            "once": True,
            "status": computed_status if _selector_exists(avail_sel, save_key) else "FAILED",
            "target_url": target_url,
            "selector_key": save_key,
            "selector": avail_sel.get(save_key),
            "missing_required_fields": missing if missing else [],
        })
        order += 1

        steps.append({
            "order": order,
            "step_id": f"{op_type}_verify",
            "step_type": "verify",
            "display_name": "反映確認",
            "status": computed_status if computed_status in ("READY", "NEEDS_REVIEW") else "UNDISCOVERED",
            "required": True,
            "source_url": target_url,
            "target_url": target_url,
            "selector_key": None,
            "selector": None,
        })

        _unsupported = [st.get("step_type") for st in steps if st.get("step_type") not in ("login", "navigate", "fill", "click", "select", "upload_file", "verify", "sleep", "search")]
        if _unsupported:
            print(f"[P24_STEP_UNSUPPORTED_GENERATED] op={op_type} unsupported={_unsupported}", flush=True)
            result[op_type] = []
            continue

        print(f"[P24_STEPS_RUNNER_ALIGNED] op={op_type} steps={[s.get('step_type') for s in steps]}", flush=True)
        result[op_type] = steps

    return result

def _crawl_url_allowed(url: str, include_patterns: list, exclude_patterns: list) -> bool:
    """修正5: include/exclude patternによるURLフィルタ"""
    u = (url or "").lower()
    inc = [str(x).lower() for x in (include_patterns or []) if str(x).strip()]
    exc = [str(x).lower() for x in (exclude_patterns or []) if str(x).strip()]
    if inc and not any(p in u for p in inc):
        return False
    if exc and any(p in u for p in exc):
        return False
    return True


def _is_static_or_blocked_url(url: str) -> tuple:
    """完全除外URL判定。(is_blocked, reason)を返す。"""
    u = (url or "").lower()
    for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".zip", ".exe", ".mp4", ".mp3"):
        if u.endswith(ext) or ext + "?" in u:
            return True, f"static_file_{ext}"
    for pat in ("/img/", "/css/", "/js/", "/fonts/", "/assets/", "/static/", "/download", "/manual", "file="):
        if pat in u:
            return True, f"static_path_{pat.strip('/')}"
    return False, ""


def _extract_followup_edit_links(page_summary: dict, base_url: str) -> list:
    """
    listページからedit/new/detail候補リンクを抽出する。
    links/buttons/menu_items/forms[].action を対象にする。
    """
    from urllib.parse import urlparse
    _edit_kw = (
        "edit", "update", "detail", "new", "add", "create", "regist", "register",
        "form", "input",
        "編集", "修正", "詳細", "新規", "追加", "登録", "作成", "入力",
    )
    _ng_kw = (
        "delete", "remove", "search", "filter", "sort", "back", "cancel",
        "削除", "検索", "絞込", "並び替え", "戻る", "キャンセル",
    )
    def _to_abs(raw_href):
        if not raw_href or raw_href.startswith("javascript") or raw_href.startswith("#"):
            return None
        if raw_href.startswith("//"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}:{raw_href}"
        elif raw_href.startswith("/"):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{raw_href}"
        elif raw_href.startswith("http"):
            return raw_href
        return None
    seen = set()
    result = []
    def _check_and_add(raw_href, text="", cls="", onclick="", aria=""):
        raw_href = str(raw_href or "")
        combined = " ".join([
            str(raw_href or "").lower(),
            str(text or "").lower(),
            str(cls or "").lower(),
            str(onclick or "").lower(),
            str(aria or "").lower(),
        ])
        if any(k in combined for k in _ng_kw):
            return
        if any(k in combined for k in _edit_kw):
            abs_href = _to_abs(raw_href)
            if abs_href and abs_href not in seen:
                seen.add(abs_href)
                result.append(abs_href)
    # links[]
    for lnk in page_summary.get("links", []):
        _check_and_add(
            lnk.get("href") or "",
            lnk.get("text") or "",
            lnk.get("class") or "",
            lnk.get("onclick") or "",
            lnk.get("aria_label") or "",
        )
    # buttons[]
    for btn in page_summary.get("buttons", []):
        _check_and_add(
            btn.get("href") or "",
            btn.get("text") or "",
            btn.get("class") or "",
            btn.get("onclick") or "",
            btn.get("aria_label") or "",
        )
    # menu_items[]
    for mi in page_summary.get("menu_items", []):
        _check_and_add(
            mi.get("href") or "",
            mi.get("text") or "",
        )
    # forms[].action
    for fm in page_summary.get("forms", []):
        action = fm.get("action") or ""
        if action:
            _check_and_add(action, action)
    return result


def _is_low_value_operation_page(url: str, title: str) -> tuple:
    """Operation候補化しない低価値ページ判定。(is_low, reason)を返す。"""
    u = (url or "").lower()
    t = (title or "").lower()
    _low_patterns = [
        ("chat",           "low_value_chat"),
        ("チャット",        "low_value_chat"),
        ("reservation",    "low_value_reservation"),
        ("受付台帳",        "low_value_reservation"),
        ("accept",         "low_value_reservation"),
        ("booking",        "low_value_reservation"),
        ("timechart",      "low_value_reservation"),
        ("customer",       "low_value_customer"),
        ("顧客",            "low_value_customer"),
        ("analytics",      "low_value_analytics"),
        ("sales",          "low_value_analytics"),
        ("readlog",        "low_value_log"),
        ("accesslog",      "low_value_log"),
        ("log/list",       "low_value_log"),
        ("message",        "low_value_message"),
        ("mailmagazine",   "low_value_message"),
        ("mail",           "low_value_message"),
        ("template",       "low_value_system"),
        ("platinum",       "low_value_system"),
        ("vip",            "low_value_system"),
        ("review",         "low_value_system"),
        ("ranking",        "low_value_system"),
        ("bookmark",       "low_value_system"),
        ("support",        "low_value_support"),
        ("inquiry",        "low_value_support"),
        ("contact",        "low_value_support"),
        ("help",           "low_value_help"),
        ("404",            "error_page"),
        ("not found",      "error_page"),
        ("index of",       "static_index"),
        ("/lib/img",       "low_value_system"),
        ("/lib/js",        "low_value_system"),
        ("/lib/css",       "low_value_system"),
        ("/lib/font",      "low_value_system"),
        ("media_library",  "low_value_system"),
        ("filebrowser",    "low_value_system"),
        ("elfinder",       "low_value_system"),
    ]
    for pat, reason in _low_patterns:
        if pat in u or pat in t:
            return True, reason
    return False, ""


def classify_admin_page_structure(page_summary: dict) -> dict:
    """
    P21.5 DOM役割証拠版: DOM量ではなく編集フォーム証拠で候補化する。
    除外優先: low_value → DOM評価 → unknown
    """
    url   = (page_summary.get("url") or "").lower()
    title = (page_summary.get("title") or "").lower()
    forms     = page_summary.get("forms_count", 0)
    inputs    = page_summary.get("inputs_count", 0)
    buttons   = page_summary.get("buttons_count", 0)
    files     = page_summary.get("file_inputs_count", 0)
    textareas = page_summary.get("textareas_count", 0)
    links     = page_summary.get("links_count", 0) or len(page_summary.get("links", []))

    page_type        = "unknown"
    domain_area      = "unknown"
    op_capability    = []
    negative_reasons = []
    evidence         = []
    value_score      = 0
    is_candidate     = False
    # [修正B-1] URLキーワードによる候補昇格（Club華等の独自管理画面対応）
    _url_raw = page_summary.get('url') or ''
    _url_kw_edit   = any(kw in url for kw in ('cast_edit','cast_list','girl_edit','girl_list','staff_edit','staff_list','profile','photo','image','media','shift','schedule'))
    _url_kw_exclude = any(kw in url for kw in ('review','mail','inquiry','questionnaire','logout','dashboard','global_nav','home'))
    _has_dom_kw = (inputs > 0 or buttons > 0 or forms > 0 or files > 0)
    _url_promoted = False
    if _url_kw_edit and not _url_kw_exclude and _has_dom_kw:
        _url_promoted = True
        if any(kw in url for kw in ('cast_edit','girl_edit','staff_edit','profile')):
            page_type   = 'entity_edit'
            domain_area = 'entity'
            if 'entity_register' not in op_capability: op_capability.append('entity_register')
            if 'entity_update'   not in op_capability: op_capability.append('entity_update')
            if 'media_replace'   not in op_capability: op_capability.append('media_replace')
        elif any(kw in url for kw in ('cast_list','girl_list','staff_list')):
            page_type   = 'entity_edit'
            domain_area = 'entity'
            if 'entity_update'   not in op_capability: op_capability.append('entity_update')
            if 'entity_register' not in op_capability: op_capability.append('entity_register')
        elif any(kw in url for kw in ('photo','image','media')):
            page_type   = 'media_edit'
            domain_area = 'media'
            if 'media_replace' not in op_capability: op_capability.append('media_replace')
        elif any(kw in url for kw in ('shift','schedule')):
            page_type   = 'schedule_edit'
            domain_area = 'schedule'
            if 'schedule_update' not in op_capability: op_capability.append('schedule_update')
        is_candidate = True
        evidence.append('url_keyword_promoted')
        print(f'[P21_URL_KEYWORD_PROMOTED] url={_url_raw[:80]} page_type={page_type} ops={op_capability}', flush=True)
    elif _url_kw_edit and _url_kw_exclude:
        negative_reasons.append('url_keyword_excluded')
        print(f'[P21_URL_KEYWORD_EXCLUDED] url={_url_raw[:80]}', flush=True)

    # A. 完全除外（static/error/pdf）
    _static, _static_reason = _is_static_or_blocked_url(url)
    if _static:
        return {"page_type": "static", "domain_area": "system", "operation_capability": [],
                "is_operation_candidate": False, "negative_reasons": [_static_reason],
                "value_score": 0, "evidence": []}
    if "404" in title or "not found" in title or "index of" in title:
        return {"page_type": "error", "domain_area": "system", "operation_capability": [],
                "is_operation_candidate": False, "negative_reasons": ["error_page"],
                "value_score": 0, "evidence": []}

    # B. low_value除外（DOM評価より先に実行）
    _low, _low_reason = _is_low_value_operation_page(url, title)
    if _low:
        _area_map = {
            "low_value_chat":        "customer",
            "low_value_reservation": "reservation",
            "low_value_customer":    "customer",
            "low_value_analytics":   "analytics",
            "low_value_log":         "analytics",
            "low_value_message":     "customer",
            "low_value_support":     "support",
            "low_value_system":      "system",
            "low_value_help":        "help",
            "error_page":            "system",
            "static_index":          "system",
        }
        _pt_map = {
            "low_value_chat":        "customer_page",
            "low_value_reservation": "reservation_page",
            "low_value_customer":    "customer_page",
            "low_value_analytics":   "log_list",
            "low_value_log":         "log_list",
            "low_value_message":     "customer_page",
            "low_value_support":     "support_page",
            "low_value_system":      "system_page",
            "low_value_help":        "help_page",
            "error_page":            "error",
            "static_index":          "static",
        }
        print(
            f"[P21_STRUCTURE_REJECTED] url={url[:80]} page_type={_pt_map.get(_low_reason,'low_value')}"
            f" reason={_low_reason}",
            flush=True
        )
        return {
            "page_type":              _pt_map.get(_low_reason, "low_value"),
            "domain_area":            _area_map.get(_low_reason, "unknown"),
            "operation_capability":   [],
            "is_operation_candidate": False,
            "negative_reasons":       [_low_reason],
            "value_score":            0,
            "evidence":               [],
            "ignored_for_operation":  True,
            "ignore_reason":          "low_value_page",
        }

    # B1.5 login_page判定
    _login_signals = 0
    _forms_data    = page_summary.get("forms", []) or []
    _inputs_data   = page_summary.get("inputs", []) or []
    _buttons_data  = page_summary.get("buttons", []) or []
    _login_form_kw = ("login","signin","auth","c1login")
    _login_btn_kw  = ("login","ログイン","signin","サインイン")
    _login_inp_kw  = ("account","user","userid","user_id","password","pass","txt_account","login_id")
    if any(p in url or p in title for p in ("login","signin","c1login","ログイン","サインイン")):
        _login_signals += 1
    for _fm in _forms_data:
        _fa = (_fm.get("action") or "").lower()
        if any(k in _fa for k in _login_form_kw):
            _login_signals += 1
            break
    for _btn in _buttons_data:
        _bt = ((_btn.get("text") or "") + " " + (_btn.get("value") or "") + " " + (_btn.get("name") or "") + " " + (_btn.get("id") or "")).lower()
        if any(k in _bt for k in _login_btn_kw):
            _login_signals += 1
            break
    for _inp in _inputs_data:
        _in = ((_inp.get("name") or "") + " " + (_inp.get("id") or "")).lower()
        if any(k in _in for k in _login_inp_kw):
            _login_signals += 1
            break
    if _login_signals >= 2:
        print(f"[P21_LOGIN_PAGE_REJECTED] url={url[:80]} reason=login_form_detected signals={_login_signals}", flush=True)
        return {
            "page_type":              "login_page",
            "domain_area":            "system",
            "operation_capability":   [],
            "is_operation_candidate": False,
            "negative_reasons":       ["login_page"],
            "value_score":            0,
            "evidence":               [],
        }

    # B2. home/dashboard判定（具体領域語がある場合は除外）
    _is_home = any(p in url or p in title for p in (
        "c1main", "main.php", "/home", "/dashboard", "/top", "管理画面トップ", "ホーム", "dashboard"
    ))
    _has_domain_word = any(p in url or p in title for p in (
        "list", "edit", "form", "new", "add", "regist", "topics", "cast", "banner",
        "schedule", "price", "staff", "member", "media", "news", "blog", "post",
        "一覧", "編集", "登録", "新規", "スケジュール", "料金",
    ))
    _strong_dom = (
        any(p in url for p in ("edit", "create", "form", "new", "regist"))
        or page_summary.get("file_inputs_count", 0) > 0
        or page_summary.get("textareas_count", 0) > 0
    )
    if _is_home and not _strong_dom and not _has_domain_word:
        print(f"[P21_HOME_REJECTED] url={url[:80]} reason=dashboard_without_operation_form", flush=True)
        return {
            "page_type":              "dashboard",
            "domain_area":            "system",
            "operation_capability":   [],
            "is_operation_candidate": False,
            "negative_reasons":       ["dashboard_without_operation_form"],
            "value_score":            0,
            "evidence":               [],
        }

    # C. listページ除外（編集フォーム証拠がなければ候補化しない）
    _is_list_url = any(p in url for p in ("/list", "_list", "list/", "=list", "一覧"))
    _is_list_title = any(p in title for p in ("一覧", "リスト", "list"))
    _has_edit_evidence = any(p in url or p in title for p in (
        "edit", "new", "create", "add", "regist", "update", "form", "編集", "登録", "新規"
    ))

    if (_is_list_url or _is_list_title) and not _has_edit_evidence:
        # list系はpage_typeを分けてcandidate=false
        if any(p in url or p in title for p in ("cast", "staff", "member", "スタッフ", "キャスト")):
            _pt = "entity_list"
            _da = "entity"
        elif any(p in url or p in title for p in ("banner", "image", "media", "photo")):
            _pt = "media_list"
            _da = "media"
        elif any(p in url or p in title for p in ("topics", "news", "blog", "post", "content")):
            _pt = "content_list"
            _da = "content"
        else:
            _pt = "list_or_nav"
            _da = "unknown"
        print(
            f"[P21_STRUCTURE_REJECTED] url={url[:80]} page_type={_pt} reason=list_page_no_edit_evidence",
            flush=True
        )
        return {
            "page_type":              _pt,
            "domain_area":            _da,
            "operation_capability":   [],
            "is_operation_candidate": False,
            "negative_reasons":       ["list_page_no_edit_evidence"],
            "value_score":            0,
            "evidence":               [],
        }

    # D. save系ボタン・edit証拠ヘルパー
    _price_url = any(p in url for p in ("price", "fee", "course", "pricelist", "料金", "multifee", "systemlist"))

    def _has_save_button():
        """save/submit/update/register系ボタンがあるか（search/filter/sort/cancel除外）"""
        raw_inputs  = page_summary.get("inputs", []) or []
        raw_buttons = page_summary.get("buttons", []) or []
        raw_links   = page_summary.get("links", []) or []
        _save_kw = (
            "save", "submit", "update", "register", "regist", "create", "add",
            "confirm", "commit", "apply",
            "保存", "登録", "更新", "反映", "確認", "送信", "設定",
        )
        _ng_kw = (
            "search", "filter", "sort", "preview", "back", "cancel",
            "delete", "remove", "reset", "clear",
            "検索", "絞込", "並び替え", "戻る", "キャンセル", "削除", "クリア",
        )
        save_evidence = []
        # input/button要素チェック
        for el in raw_inputs + raw_buttons:
            val     = (el.get("value") or "").lower()
            text    = (el.get("text") or "").lower()
            typ     = (el.get("type") or "").lower()
            cls     = (el.get("class") or "").lower()
            nm      = (el.get("name") or "").lower()
            onclick = (el.get("onclick") or "").lower()
            aria    = (el.get("aria_label") or "").lower()
            combined = " ".join([val, text, cls, nm, onclick, aria])
            if typ == "submit":
                if not any(k in combined for k in _ng_kw):
                    save_evidence.append(f"type=submit val={val[:20]}")
            elif any(k in combined for k in _save_kw):
                if not any(k in combined for k in _ng_kw):
                    save_evidence.append(f"kw_match val={val[:20]} text={text[:20]}")
        # a.btn / a[href*=save/update/regist] チェック
        for lnk in raw_links:
            href    = (lnk.get("href") or "").lower()
            text    = (lnk.get("text") or "").lower()
            cls     = (lnk.get("class") or "").lower()
            onclick = (lnk.get("onclick") or "").lower()
            aria    = (lnk.get("aria_label") or "").lower()
            combined = " ".join([href, text, cls, onclick, aria])
            if any(k in combined for k in _save_kw):
                if not any(k in combined for k in _ng_kw):
                    save_evidence.append(f"link_save href={href[:30]} text={text[:20]}")
        if save_evidence:
            print(f"[P21_SAVE_BUTTON_EVIDENCE] url={url[:80]} matched={save_evidence[:3]}", flush=True)
            return True
        # フォールバック: buttons_countあり + edit系URL
        if buttons > 0 and any(p in url or p in title for p in
                ("edit", "new", "create", "add", "regist", "update", "form", "編集", "登録", "新規")):
            print(f"[P21_SAVE_BUTTON_EVIDENCE] url={url[:80]} matched=[fallback_edit_url]", flush=True)
            return True
        return False

    _save_ok = _has_save_button()
    _has_dom = inputs > 0 or forms > 0 or files > 0 or textareas > 0

    if not _has_dom:
        negative_reasons.append("no_input_evidence")
        if links > 0:
            page_type = "list_or_nav"
        print(
            f"[P21_OPERATION_CANDIDATE_REJECTED] url={url[:80]} reason=no_input_evidence evidence=[]",
            flush=True
        )
        return {
            "page_type":              page_type,
            "domain_area":            domain_area,
            "operation_capability":   [],
            "is_operation_candidate": False,
            "negative_reasons":       negative_reasons,
            "value_score":            0,
            "evidence":               [],
        }

    # [P21_COLLECT_ONLY] Operation推定・Capability生成はP24で行う。ここでは収集データのみ返す。
    return {
        "page_type":              page_type,
        "domain_area":            domain_area,
        "operation_capability":   [],
        "is_operation_candidate": False,
        "negative_reasons":       negative_reasons,
        "value_score":            0,
        "evidence":               [],
    }


# ==============================================================
# P22 DOM Evidence Mapper
# ==============================================================

ROLE_DICTIONARY = {
    "title": [
        "title","subject","headline","heading","caption","topic_title","news_title",
        "post_title","article_title","blog_title","diary_title","event_title",
        "campaign_title","coupon_title","notice_title","information_title",
        "ttl","hd","head","name_title",
        "件名","題名","表題","タイトル","見出し","記事タイトル","投稿タイトル",
        "ニュースタイトル","トピックタイトル","日記タイトル","イベントタイトル",
        "キャンペーン名","お知らせタイトル","掲載タイトル","表示タイトル",
    ],
    "body": [
        "body","content","contents","description","desc","detail","details","message",
        "comment","comments","text","textarea","article","post_body","blog_body",
        "diary_body","news_body","topic_body","event_body","campaign_body",
        "coupon_body","notice_body","information_body","profile_text",
        "introduction","intro","about","summary","memo","note","remarks","remark",
        "free_text","main_text","html","editor","wysiwyg","ckeditor","tinymce",
        "quill","contenteditable",
        "本文","内容","詳細","説明","紹介","本文内容","記事本文","投稿本文",
        "ニュース本文","日記本文","トピック本文","イベント本文","キャンペーン本文",
        "お知らせ本文","紹介文","自己紹介","プロフィール文","説明文","備考",
        "メモ","コメント","自由記入","フリーテキスト","本文HTML","本文テキスト",
        "status","state","public","private","visible","hidden","active","inactive",
        "enabled","disabled","display","publish","release",
        "ステータス","状態","公開","非公開","表示","非表示","有効","無効","掲載",
    ],
    "name": [
        "name","full_name","display_name","user_name","username","staff_name",
        "cast_name","girl_name","member_name","employee_name","customer_name",
        "client_name","shop_name","store_name","company_name","person_name",
        "real_name","kana","furigana","first_name","last_name","nickname",
        "nick_name","alias",
        "名前","氏名","名称","表示名","スタッフ名","キャスト名","女の子名",
        "担当者名","顧客名","お客様名","店舗名","会社名","店名","源氏名",
        "ニックネーム","ふりがな","フリガナ","かな","カナ","名前かな",
    ],
    "price": [
        "price","fee","amount","cost","charge","rate","fare","payment","money",
        "course","plan","menu","option_price","base_price","regular_price",
        "sale_price","discount","tax","total","subtotal","yen","jpy","point",
        "min_price","max_price",
        "料金","金額","価格","費用","単価","値段","コース","プラン","メニュー",
        "オプション料金","基本料金","通常料金","割引","税込","税別","合計","小計",
        "円","ポイント","支払","会計","請求","料","入会金","指名料","延長料金",
    ],
    "date": [
        "date","day","datetime","time","start_date","end_date","start_time","end_time",
        "open_time","close_time","schedule","shift","calendar","reservation_date",
        "booking_date","work_date","attendance","available_date","published_at",
        "publish_date","posted_at","event_date","deadline","period","from","to",
        "hour","minute",
        "日付","日時","時間","開始日","終了日","開始時間","終了時間","公開日",
        "掲載日","投稿日","予約日","出勤","出勤日","出勤時間","勤務","勤務日",
        "予定","スケジュール","シフト","カレンダー","営業日","営業時間",
        "期間","期限","締切","時","分","曜日",
    ],
    "file": [
        "file","upload","image","photo","picture","pic","img","thumbnail","thumb",
        "avatar","icon","banner","main_image","sub_image","cover","gallery",
        "movie","video","media","attachment","document","pdf","csv","excel",
        "画像","写真","画像アップロード","写真アップロード","ファイル","添付",
        "サムネイル","アイコン","バナー","メイン画像","サブ画像","カバー画像",
        "ギャラリー","動画","ムービー","メディア","資料","PDF","CSV",
    ],
    "save": [
        "save","submit","update","regist","register","registration","create","add",
        "apply","commit","confirm","complete","finish","send","post","publish",
        "release","reflect","entry","insert","store","upload","execute","run",
        "done","ok","yes","set","setting","change","modify",
        "保存","登録","更新","追加","作成","反映","確定","確認","完了","送信",
        "投稿","公開","掲載","設定","変更","修正","実行","決定","OK",
        "アップロード","登録する","保存する","更新する","送信する","公開する",
        "反映する","設定する","追加する","作成する",
    ],
    "edit_trigger": [
        "edit","update","detail","details","show","view","open","modify","change",
        "setting","settings","config","configure","manage","maintenance",
        "profile","form","input","entry","select","choose",
        "編集","修正","詳細","表示","開く","変更","設定","管理","確認",
        "プロフィール","入力","選択","編集する","詳細を見る","設定する",
    ],
    "new_trigger": [
        "new","add","create","regist","register","insert","entry",
        "新規","追加","作成","登録","新規追加","新規登録","追加登録","作成する",
    ],
    "search": [
        "search","find","filter","query","keyword","keywords","condition","conditions",
        "narrow","refine","lookup","sort","order","display_change",
        "検索","探す","絞込","絞り込み","条件","検索条件","キーワード",
        "並び替え","ソート","表示変更","抽出",
    ],
    "delete": [
        "delete","remove","destroy","trash","clear","erase","drop","cancel",
        "削除","消去","破棄","クリア","取消","キャンセル","取り消し",
    ],
    "login": [
        "login","signin","sign_in","log_in","auth","authentication","password",
        "username","user_id","account",
        "ログイン","サインイン","認証","パスワード","ID","ユーザーID","アカウント",
    ],
}

NEGATIVE_ROLE_DICTIONARY = {
    "save": [
        "search","find","filter","sort","order","preview","back","cancel",
        "delete","remove","clear","close","copy","download","export",
        "display","undisplay","toggle","bulk_delete","trash",
        "login","signin","sign_in","auth","authenticate",
        "prev","next","current","back","forward","week","page","pagination",
        "検索","絞込","絞り込み","並び替え","プレビュー","戻る","キャンセル",
        "削除","一括削除","クリア","リセット","閉じる","コピー","ダウンロード",
        "エクスポート","表示","非表示","一括表示","一括非表示",
        "ログイン","サインイン","認証",
    ],
    "edit_trigger": [
        "delete","remove","destroy","search","filter","sort","back","cancel",
        "削除","検索","絞込","並び替え","戻る","キャンセル",
    ],
    "body": [
        "search","keyword","csrf","token","password","hidden",
        "検索","キーワード","パスワード",
    ],
    "price": [
        "sort","order","display","search","filter",
        "並び順","表示順","検索","絞込",
    ],
}

ROLE_ELEMENT_WEIGHTS = {
    "title":  {"input_text":45,"label":20,"name_attr":25,"id_attr":20,"placeholder":20},
    "body":   {"textarea":60,"contenteditable":60,"label":20,"name_attr":25,"id_attr":20,"placeholder":20},
    "name":   {"input_text":40,"label":20,"name_attr":25,"id_attr":20,"placeholder":20},
    "price":  {"input_text":45,"input_number":55,"label":25,"name_attr":25,"id_attr":20},
    "date":   {"input_date":60,"input_time":60,"input_datetime":60,"input_text":30,"label":25,"name_attr":25,"id_attr":20},
    "file":   {"input_file":80,"accept_image":20,"label":15,"name_attr":15},
    "save":   {"submit_button":70,"button_text":45,"button_value":45,"name_attr":25,"id_attr":20,"onclick":25,"form_action":15},
    "edit_trigger": {"link_text":45,"href":35,"button_text":35,"onclick":30,"class":20},
}

OPERATION_REQUIREMENTS = {
    "news_post":      {"required":["body","save"],"preferred":["title"],"page_types":["content_edit","content_form"],"areas":["content"],"min_score":75},
    "text_update":    {"required":["body","save"],"preferred":[],"page_types":["content_edit","content_form"],"areas":["content"],"min_score":70},
    "status_update":  {"required":["body","save"],"preferred":[],"page_types":["status_edit","status_page","content_edit","content_form","text_edit"],"areas":["content","entity"],"min_score":50},
    "entity_register":{"required":["name","save"],"preferred":[],"page_types":["entity_edit","entity_form"],"areas":["entity"],"min_score":70},
    "entity_update":  {"required":["name","save"],"preferred":["edit_trigger"],"page_types":["entity_edit","entity_form"],"areas":["entity"],"min_score":75},
    "schedule_update":{"required":["save"],"preferred":["date"],"page_types":["schedule_edit","schedule_form","schedule_page"],"areas":["schedule"],"min_score":60},
    "media_replace":  {"required":["file","save"],"preferred":[],"page_types":["media_edit","media_form"],"areas":["media","entity"],"min_score":80},
    "price_update":   {"required":["price","save"],"preferred":[],"page_types":["price_edit","price_form"],"areas":["price"],"min_score":75},
}


def build_dom_role_dictionary() -> dict:
    return ROLE_DICTIONARY


def score_dom_element_role(element: dict, role: str) -> dict:
    """
    element属性をROLE_DICTIONARY/NEGATIVE_ROLE_DICTIONARYと照合しscoreを返す。
    """
    pos_kw  = ROLE_DICTIONARY.get(role, [])
    neg_kw  = NEGATIVE_ROLE_DICTIONARY.get(role, [])
    weights = ROLE_ELEMENT_WEIGHTS.get(role, {})

    tag     = (element.get("tag") or element.get("type") or "").lower()
    typ     = (element.get("type") or "").lower()
    name    = (element.get("name") or "").lower()
    eid     = (element.get("id") or "").lower()
    ph      = (element.get("placeholder") or "").lower()
    aria    = (element.get("aria_label") or "").lower()
    label   = (element.get("label") or "").lower()
    text    = (element.get("text") or "").lower()
    value   = (element.get("value") or "").lower()
    cls     = (element.get("class_name") or element.get("class") or "").lower()
    title   = (element.get("title") or "").lower()
    onclick = (element.get("onclick") or "").lower()
    href    = (element.get("href") or "").lower()
    action  = (element.get("action") or "").lower()
    accept  = (element.get("accept") or "").lower()

    # hidden/password/csrf/tokenは除外
    if typ in ("hidden", "password") or "csrf" in name or "token" in name:
        return {"score": -50, "matched": [], "negative": ["hidden_or_credential"], "evidence": []}
    if role == "save" and typ == "reset":
        return {"score": -50, "matched": [], "negative": ["type=reset"], "evidence": []}

    attrs = {
        "name":        name,
        "id":          eid,
        "placeholder": ph,
        "aria_label":  aria,
        "label":       label,
        "text":        text,
        "value":       value,
        "class":       cls,
        "title":       title,
        "onclick":     onclick,
        "href":        href,
        "action":      action,
    }

    score   = 0
    matched = []
    negative= []
    evidence= []

    # type強加点
    if role == "file" and typ == "file":
        score += weights.get("input_file", 80)
        evidence.append("type=file")
        if "image" in accept or "photo" in accept:
            score += weights.get("accept_image", 20)
            evidence.append("accept=image")
    if role == "date" and typ in ("date", "time", "datetime-local", "datetime"):
        score += weights.get(f"input_{typ}", 60)
        evidence.append(f"type={typ}")
    if role == "price" and typ == "number":
        score += weights.get("input_number", 55)
        evidence.append("type=number")
    if role == "save" and typ == "submit":
        score += weights.get("submit_button", 70)
        evidence.append("type=submit")
    if role == "save" and tag == "button" and typ not in ("submit",) and any(k in text + value for k in ("設定", "保存", "登録", "更新", "save", "submit")):
        score += weights.get("submit_button", 50)
        evidence.append("tag=button_save_text")
    if role == "body" and tag == "textarea":
        score += weights.get("textarea", 60)
        evidence.append("tag=textarea")
    if role == "body" and tag == "select":
        score += 35
        evidence.append("tag=select")
    if role == "body" and typ in ("checkbox", "radio"):
        score += 25
        evidence.append(f"type={typ}")

    # 属性照合
    for attr_name, attr_val in attrs.items():
        if not attr_val:
            continue
        for kw in pos_kw:
            if kw in attr_val:
                w = 15
                if attr_name == "name":    w = weights.get("name_attr", 25)
                elif attr_name == "id":    w = weights.get("id_attr", 20)
                elif attr_name in ("placeholder","aria_label","label"): w = weights.get("label", 20)
                elif attr_name in ("text","value"):  w = weights.get("button_text", 45) if role == "save" else 15
                elif attr_name == "onclick": w = weights.get("onclick", 25)
                elif attr_name == "href":  w = weights.get("href", 35)
                elif attr_name == "class": w = 10
                score += w
                matched.append(f"{attr_name}={kw}")
                break

    # negative照合（大きく減点）
    for attr_name, attr_val in attrs.items():
        if not attr_val:
            continue
        for nkw in neg_kw:
            if nkw in attr_val:
                score -= 40
                negative.append(f"{attr_name}={nkw}")
                break

    return {"score": max(score, -100), "matched": matched, "negative": negative, "evidence": evidence}


def extract_operation_selectors_from_page(page_summary: dict, operation_type: str) -> dict:
    """
    page内のDOM証拠からoperation_typeに必要なrole selectorを抽出する。
    """
    req = OPERATION_REQUIREMENTS.get(operation_type)
    if not req:
        return {"status": "UNDISCOVERED", "selectors": {}, "missing": [operation_type], "validation_score": 0, "target_url": page_summary.get("url") or "", "evidence": [], "executable": False}

    required_roles  = req["required"]
    preferred_roles = req.get("preferred", [])
    all_roles = list(set(required_roles + preferred_roles))

    # 対象要素を収集
    elements = []
    for inp in (page_summary.get("inputs") or []):
        el = dict(inp); el.setdefault("tag","input"); elements.append(el)
    for ta in (page_summary.get("textareas") or []):
        el = dict(ta); el["tag"] = "textarea"; elements.append(el)
    for btn in (page_summary.get("buttons") or []):
        el = dict(btn); el.setdefault("tag","button"); elements.append(el)
    for fi in (page_summary.get("file_inputs") or []):
        el = dict(fi); el["tag"] = "input"; el["type"] = "file"; elements.append(el)
    for lnk in (page_summary.get("links") or []):
        el = dict(lnk); el["tag"] = "a"; elements.append(el)
    for fm in (page_summary.get("forms") or []):
        el = dict(fm); el["tag"] = "form"; elements.append(el)
    for se in (page_summary.get("selects") or []):
        el = dict(se); el["tag"] = "select"; elements.append(el)
    for fld in ((page_summary.get("form_schema") or {}).get("fields") or []):
        if not isinstance(fld, dict):
            continue
        el = {
            "tag": fld.get("tag") or ("select" if fld.get("options") else "input"),
            "type": fld.get("type") or "",
            "name": fld.get("name") or "",
            "id": fld.get("id") or "",
            "label": fld.get("label") or "",
            "placeholder": fld.get("label") or "",
            "text": fld.get("label") or fld.get("section") or "",
            "selector": fld.get("selector") or "",
            "suggested_selector": fld.get("selector") or "",
            "value": fld.get("value") or "",
            "class_name": "",
        }
        elements.append(el)

    tag_counts = {}
    type_counts = {}
    for el in elements:
        _tag = (el.get("tag") or "").lower()
        _typ = (el.get("type") or "").lower()
        if _tag:
            tag_counts[_tag] = tag_counts.get(_tag, 0) + 1
        if _tag and _typ:
            type_counts[(_tag, _typ)] = type_counts.get((_tag, _typ), 0) + 1

    def _quote_attr(v: str) -> str:
        return str(v or "").replace("\\", "\\\\").replace("'", "\\'")

    def _quote_text(v: str) -> str:
        return str(v or "").replace("\\", "\\\\").replace('"', '\\"')

    def _selector_for_element(el: dict) -> str:
        tag = (el.get("tag") or "input").lower()
        typ = (el.get("type") or "").lower()
        if el.get("selector"):
            return str(el.get("selector"))
        if el.get("suggested_selector"):
            return str(el.get("suggested_selector"))
        if el.get("id"):
            return f"#{el['id']}"
        if tag == "input" and typ in ("checkbox", "radio") and el.get("name") and el.get("value"):
            return f"input[name='{_quote_attr(el['name'])}'][value='{_quote_attr(el['value'])}']"
        if el.get("name"):
            return f"{tag}[name='{_quote_attr(el['name'])}']"
        if el.get("aria_label"):
            return f"{tag}[aria-label='{_quote_attr(el['aria_label'])}']"
        cls = (el.get("class_name") or el.get("class") or "").strip()
        if cls:
            first_cls = cls.split()[0]
            if first_cls:
                return f"{tag}.{first_cls}"
        value = (el.get("value") or "").strip()
        text = (el.get("text") or "").strip()
        if tag == "input" and typ:
            if value and typ in ("submit", "button", "reset"):
                return f"input[type='{_quote_attr(typ)}'][value='{_quote_attr(value)}']"
            if type_counts.get((tag, typ), 0) == 1:
                return f"input[type='{_quote_attr(typ)}']"
        if tag in ("button", "a") and text:
            return f'{tag}:has-text("{_quote_text(text[:40])}")'
        if tag_counts.get(tag, 0) == 1 and tag in ("textarea", "select", "button", "form"):
            return tag
        return tag

    # roleごとに最高scoreのelementを選ぶ
    best = {}
    for role in all_roles:
        best_score  = -999
        best_el     = None
        best_result = None
        for el in elements:
            r = score_dom_element_role(el, role)
            if r["score"] > best_score and r["score"] > 0 and not r["negative"]:
                best_score  = r["score"]
                best_el     = el
                best_result = r
        if best_el and best_result:
            # [ROLE_TAG_FILTER] name/title/body/price role: a tag is invalid
            _role_no_a = role in ("name", "title", "body", "price", "date")
            if _role_no_a and best_el.get("tag") == "a":
                print(f"[ROLE_TAG_REJECTED] url={page_summary.get('url','')[:60]} role={role} tag=a selector rejected", flush=True)
                best_el = None
                best_result = None
        if best_el and best_result:
            # selector生成
            sel = _selector_for_element(best_el)
            # [SELECTOR_QUALITY] reject tag-only selectors (no id/name/class)
            _sel_is_tag_only = sel in ("input", "button", "a", "textarea", "select", "form")
            _tag_only_unique = _sel_is_tag_only and tag_counts.get(sel, 0) == 1 and sel in ("button", "textarea", "select", "form")
            if _sel_is_tag_only and not _tag_only_unique:
                print(f"[SELECTOR_QUALITY_REJECTED] url={page_summary.get('url','')[:60]} role={role} selector={sel} score={best_score} reason=tag_only_selector", flush=True)
            else:
                best[role] = {
                    "selector":  sel,
                    "role":      role,
                    "tag":       (best_el.get("tag") or "").lower(),
                    "type":      (best_el.get("type") or "").lower(),
                    "score":     best_score,
                    "matched":   best_result["matched"],
                    "evidence":  best_result["evidence"],
                }
                print(f"[P22_DOM_ROLE_MATCH] url={page_summary.get('url','')[:60]} role={role} selector={sel} score={best_score} matched={best_result['matched'][:3]}", flush=True)

    if operation_type == "schedule_update" and "date" in best and "date_input" not in best:
        best["date_input"] = dict(best["date"], role="date_input")
    if operation_type == "entity_register" and "name" in best and "required_inputs" not in best:
        best["required_inputs"] = dict(best["name"], role="required_inputs")
    if operation_type == "entity_update" and "name" in best and "editable_inputs" not in best:
        best["editable_inputs"] = dict(best["name"], role="editable_inputs")

    missing = [r for r in required_roles if r not in best]
    validation_score = 0
    if best:
        scores = [v["score"] for v in best.values()]
        validation_score = int(sum(scores) / len(scores)) if scores else 0

    if not missing and validation_score >= req["min_score"]:
        status = "READY"
    elif best:
        status = "NEEDS_REVIEW"
    else:
        status = "UNDISCOVERED"

    return {
        "status":           status,
        "selectors":        best,
        "missing":          missing,
        "validation_score": validation_score,
        "target_url":       page_summary.get("url") or "",
        "form_schema":      page_summary.get("form_schema") or {},
        "evidence":         [v["matched"] for v in best.values()],
    }


def build_operation_mappings_from_dom_evidence(mapping_id: str, pages: list) -> dict:
    """
    pagesを走査しoperation_typeごとにDOM証拠からmappingを生成する。
    """
    result = {}
    # list/dashboard/unknown/non_op page除外リスト
    _NON_OP_PAGE_TYPES = (
        "dashboard", "home", "system_page", "entity_list", "content_list",
        "media_list", "log_list", "reservation_page", "customer_page",
        "support_page", "help_page", "list_or_nav", "unknown", "error", "static",
        "login_page",
    )

    def _page_has_editable_dom(pg: dict) -> bool:
        _fs = pg.get("form_schema") or {}
        _gemini_pt = _fs.get("gemini_page_type") or ""
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
            or bool(_fs.get("gemini_fields"))
            or (_gemini_pt and _gemini_pt != "other")  # page_typeが判定済みなら有効
        )

    # Gemini canonical → operation selector key マッピング
    _CANONICAL_TO_SEL_KEY = {
        "profile.name":               "name",
        "profile.kana":               "name",
        "profile.age":                "age",
        "profile.height":             "height",
        "profile.bust":               "bust",
        "profile.cup":                "cup",
        "profile.waist":              "waist",
        "profile.hip":                "hip",
        "profile.image":              "image",
        "profile.status":             "body",
        "profile.pickup":             "body",
        "profile.custom":             "body",
        "profile.type_tag":           "body",
        "profile.default_start":      "start_time",
        "profile.default_end":        "end_time",
        "profile.room":               "body",
        "profile.joined_at":          "body",
        "profile.newface":            "body",
        "profile.thumbnail_image":    "file",
        "profile.thumbnail_movie_url":"body",
        "profile.thumbnail_movie_status": "body",
        "schedule.date":              "date",
        "schedule.start_time":        "start_time",
        "schedule.end_time":          "end_time",
        "schedule.day":               "date",
        "schedule.status":            "body",
        "status.visibility":          "body",
        "status.state":               "body",
        "status.waiting_time":        "body",
        "content.title":              "title",
        "content.body":               "body",
        "price.amount":               "price",
        "price.name":                 "price",
        "price.item":                 "price",
        "price.fee":                  "price",
        "media.file":                 "file",
        "image.file":                 "file",
        "contact.message":            "body",
        "inquiry.body":               "body",
        "reply.body":                 "body",
        "survey.body":                "body",
        "form.field":                 "body",
    }

    def _build_mapping_from_gemini(pg: dict, op_type: str, req: dict) -> dict | None:
        """GeminiのHTMLフォーム解析結果からoperation_typeのmappingを直接構築する。"""
        _fs = pg.get("form_schema") or {}
        _gemini_fields   = _fs.get("gemini_fields") or []
        _gemini_pt       = _fs.get("gemini_page_type") or pg.get("page_purpose") or ""
        _gemini_save     = _fs.get("gemini_save_selector") or ""
        _gemini_action   = _fs.get("gemini_form_action") or ""
        _gemini_conf     = float(_fs.get("gemini_confidence") or 0)
        # Geminiが判定したpage_typeとop_typeが一致するか確認
        _PT_TO_OP = {
            "entity_register": "entity_register",
            "entity_update":   "entity_update",
            "schedule_update": "schedule_update",
            "news_post":       "news_post",
            "text_update":     "text_update",
            "media_replace":   "media_replace",
            "price_update":    "price_update",
            "status_update":   "status_update",
        }
        _expected_op = _PT_TO_OP.get(_gemini_pt)
        if _expected_op and _expected_op != op_type:
            return None  # このページは別のoperationのもの
        if not _gemini_fields:
            # fieldsなしでもpage_typeがop_typeと一致する場合はNEEDS_REVIEWフォールバック
            if _expected_op == op_type:
                _target_url = _gemini_action or pg.get("url", "")
                # save_selectorが得られていればselectorsに含めてREADY判定を狙う
                _fallback_selectors = {}
                if _gemini_save:
                    _fallback_selectors["save"] = {"selector": _gemini_save, "source": "gemini_save_only"}
                _required = req.get("required", [])
                _missing_fb = [r for r in _required if r not in _fallback_selectors]
                _fb_score = max(int(_gemini_conf * 50), 3) + (10 if _fallback_selectors else 0)
                _fb_status = "READY" if not _missing_fb and _fallback_selectors else "NEEDS_REVIEW"
                _fb_exec   = _fb_status == "READY"
                print(f"[GEMINI_FIELDS_EMPTY_FALLBACK] op={op_type} gemini_pt={_gemini_pt} url={_target_url[:60]} save={'yes' if _gemini_save else 'no'} status={_fb_status}", flush=True)
                return {
                    "status":            _fb_status,
                    "target_url":        _target_url,
                    "selectors":         _fallback_selectors,
                    "validation_score":  _fb_score,
                    "missing":           _missing_fb,
                    "source":            "gemini_page_type_only",
                    "executable":        _fb_exec,
                    "page_purpose":      _gemini_pt,
                    "page_purpose_source": "gemini_html_analysis",
                    "form_schema":       _fs,
                    "body_candidates":   [],
                    "textareas":         [],
                    "editor_candidates": [],
                }
            return None
        selectors: dict = {}
        for _f in _gemini_fields:
            _sel = str(_f.get("selector") or "").strip()
            _canonical = str(_f.get("canonical") or "").strip()
            if not _sel or not _canonical:
                continue
            _key = _CANONICAL_TO_SEL_KEY.get(_canonical)
            if _key and _key not in selectors:
                selectors[_key] = {"selector": _sel, "label": _f.get("label", ""), "source": "gemini"}
        if _gemini_save and "save" not in selectors:
            selectors["save"] = {"selector": _gemini_save, "source": "gemini"}
        if not selectors:
            # page_typeがop_typeと一致する場合: フィールド未マッチでもNEEDS_REVIEWとして返す
            if _expected_op == op_type and _gemini_pt:
                _target_url = _gemini_action or pg.get("url", "")
                _required = req.get("required", [])
                print(f"[GEMINI_PAGE_TYPE_FALLBACK] op={op_type} gemini_pt={_gemini_pt} url={_target_url[:60]} reason=no_canonical_match", flush=True)
                return {
                    "status":            "NEEDS_REVIEW",
                    "target_url":        _target_url,
                    "selectors":         {},
                    "validation_score":  max(int(_gemini_conf * 60), 5),
                    "missing":           _required,
                    "source":            "gemini_page_type_only",
                    "executable":        False,
                    "page_purpose":      _gemini_pt,
                    "page_purpose_source": "gemini_html_analysis",
                    "form_schema":       _fs,
                    "body_candidates":   [],
                    "textareas":         [],
                    "editor_candidates": [],
                }
            return None
        _required = req.get("required", [])
        _missing = [r for r in _required if r not in selectors]
        _score = int(_gemini_conf * 100) + len(selectors) * 5
        _target_url = _gemini_action or pg.get("url", "")
        return {
            "status":            "READY" if not _missing else "NEEDS_REVIEW",
            "target_url":        _target_url,
            "selectors":         selectors,
            "validation_score":  min(_score, 100),
            "missing":           _missing,
            "source":            "gemini_html_analysis",
            "executable":        not _missing,
            "page_purpose":      _gemini_pt,
            "page_purpose_source": "gemini_html_analysis",
            "form_schema":       _fs,
            "body_candidates":   [],
            "textareas":         [],
            "editor_candidates": [],
        }

    _PURPOSE_TO_OP = {
        "create_page":       "entity_register",
        "edit_page":         "entity_update",
        "entity_edit_page":  "entity_update",
        "news_post_page":    "news_post",
        "text_edit_page":    "text_update",
        "media_upload_page": "media_replace",
        "schedule_page":     "schedule_update",
        "price_page":        "price_update",
        "status_page":       "status_update",
    }

    def _selector_from_dom_element(el: dict, fallback_tag: str = "input") -> str:
        sel = str(el.get("selector") or el.get("css_selector") or "").strip()
        if sel:
            return sel
        eid = str(el.get("id") or "").strip()
        if eid:
            return "#" + eid.replace('"', '\\"')
        name = str(el.get("name") or "").strip()
        tag = str(el.get("tag") or fallback_tag or "input").strip().lower() or "input"
        if name:
            _esc = name.replace('"', '\\"')
            return f'{tag}[name="{_esc}"]'
        return ""

    def _dom_fields_for_schema(pg: dict) -> list[dict]:
        fields = []
        seen = set()
        source_lists = [
            ("inputs", pg.get("inputs") or []),
            ("textareas", pg.get("textareas") or []),
            ("selects", pg.get("selects") or []),
            ("file_inputs", pg.get("file_inputs") or []),
        ]
        for source, rows in source_lists:
            for idx, el in enumerate(rows or []):
                if not isinstance(el, dict):
                    continue
                tag = str(el.get("tag") or ("textarea" if source == "textareas" else "select" if source == "selects" else "input")).lower()
                typ = str(el.get("type") or ("textarea" if source == "textareas" else "select" if source == "selects" else "text")).lower()
                if typ in {"hidden", "password", "submit", "button", "reset"}:
                    continue
                sel = _selector_from_dom_element(el, tag)
                label = str(el.get("label") or el.get("aria_label") or el.get("placeholder") or el.get("name") or el.get("id") or f"{source}_{idx + 1}").strip()
                key = (sel, str(el.get("name") or ""), str(el.get("id") or ""), label)
                if key in seen:
                    continue
                seen.add(key)
                fields.append({
                    "selector": sel,
                    "label": label,
                    "name": el.get("name") or "",
                    "id": el.get("id") or "",
                    "type": "file" if typ == "file" else ("textarea" if tag == "textarea" else ("select" if tag == "select" else typ or "text")),
                    "required": bool(el.get("required")),
                    "source": "ai_dom_purpose_fallback",
                })
        return fields

    def _pick_field_selector(fields: list[dict], keywords: tuple[str, ...], fallback_kind: str = "text") -> str:
        for field in fields:
            hay = " ".join(str(field.get(k) or "") for k in ("label", "name", "id", "selector")).lower()
            if any(k.lower() in hay for k in keywords) and field.get("selector"):
                return str(field.get("selector"))
        if fallback_kind == "file":
            for field in fields:
                if field.get("type") == "file" and field.get("selector"):
                    return str(field.get("selector"))
        if fallback_kind == "textarea":
            for field in fields:
                if field.get("type") == "textarea" and field.get("selector"):
                    return str(field.get("selector"))
        for field in fields:
            if field.get("selector") and field.get("type") not in {"checkbox", "radio", "file"}:
                return str(field.get("selector"))
        return ""

    def _pick_save_selector(pg: dict) -> str:
        save_words = ("保存", "更新", "登録", "送信", "追加", "確認", "submit", "save", "update", "register", "add", "send")
        for el in (pg.get("buttons") or []) + (pg.get("inputs") or []):
            if not isinstance(el, dict):
                continue
            typ = str(el.get("type") or "").lower()
            txt = " ".join(str(el.get(k) or "") for k in ("text", "value", "label", "name", "id", "class", "onclick")).lower()
            if typ == "submit" or any(w.lower() in txt for w in save_words):
                sel = _selector_from_dom_element(el, "button" if str(el.get("tag") or "").lower() == "button" else "input")
                if sel:
                    return sel
        return "input[type=\"submit\"],button[type=\"submit\"]" if (pg.get("buttons") or pg.get("inputs")) else ""

    def _build_mapping_from_page_purpose(pg: dict, op_type: str, req: dict) -> dict | None:
        purpose = str(pg.get("page_purpose") or (pg.get("form_schema") or {}).get("gemini_page_type") or "")
        expected = _PURPOSE_TO_OP.get(purpose) or {
            "entity_register": "entity_register",
            "entity_update": "entity_update",
            "schedule_update": "schedule_update",
            "news_post": "news_post",
            "text_update": "text_update",
            "media_replace": "media_replace",
            "price_update": "price_update",
            "status_update": "status_update",
        }.get(purpose)
        if expected != op_type or not _page_has_editable_dom(pg):
            return None
        fields = _dom_fields_for_schema(pg)
        selectors: dict = {}
        save_sel = _pick_save_selector(pg)
        if save_sel:
            selectors["save"] = {"selector": save_sel, "source": "ai_dom_purpose_fallback", "label": "保存"}
        if op_type in {"entity_register", "entity_update"}:
            name_sel = _pick_field_selector(fields, ("名前", "氏名", "お名前", "name", "girl", "cast", "staff"), "text")
            if name_sel:
                selectors["name"] = {"selector": name_sel, "source": "ai_dom_purpose_fallback", "label": "名前"}
        elif op_type in {"news_post", "text_update", "status_update"}:
            body_sel = _pick_field_selector(fields, ("本文", "内容", "コメント", "紹介", "body", "text", "comment", "content", "message"), "textarea")
            if body_sel:
                selectors["body"] = {"selector": body_sel, "source": "ai_dom_purpose_fallback", "label": "本文"}
        elif op_type == "price_update":
            price_sel = _pick_field_selector(fields, ("料金", "金額", "価格", "price", "fee", "course"), "text")
            if price_sel:
                selectors["price"] = {"selector": price_sel, "source": "ai_dom_purpose_fallback", "label": "料金"}
        elif op_type == "media_replace":
            file_sel = _pick_field_selector(fields, ("画像", "写真", "ファイル", "image", "photo", "file"), "file")
            if file_sel:
                selectors["file"] = {"selector": file_sel, "source": "ai_dom_purpose_fallback", "label": "ファイル"}
        elif op_type == "schedule_update":
            date_sel = _pick_field_selector(fields, ("日付", "曜日", "date", "day"), "text")
            if date_sel:
                selectors["date"] = {"selector": date_sel, "source": "ai_dom_purpose_fallback", "label": "日付"}

        missing = [r for r in req.get("required", []) if r not in selectors]
        score = 92 if not missing else 45
        return {
            "status": "READY" if not missing else "NEEDS_REVIEW",
            "target_url": pg.get("url", ""),
            "selectors": selectors,
            "validation_score": score,
            "missing": missing,
            "source": "ai_dom_purpose_fallback",
            "executable": not missing,
            "page_purpose": purpose,
            "page_purpose_source": pg.get("page_purpose_source") or (pg.get("form_schema") or {}).get("page_purpose_source") or "ai_dom_purpose",
            "form_schema": {"fields": fields, "fields_count": len(fields), "source": "ai_dom_purpose_fallback"},
            "body_candidates": pg.get("body_candidates", []),
            "textareas": pg.get("textareas", []),
            "editor_candidates": pg.get("editor_candidates", []),
        }

    for op_type, req in OPERATION_REQUIREMENTS.items():
        allowed_areas      = req.get("areas", [])
        allowed_page_types = req.get("page_types", [])
        min_score          = req.get("min_score", 70)
        best_mapping       = None
        best_score         = -1
        list_only          = True  # edit formページが1つも見つからなかった場合
        for pg in pages:
            area      = pg.get("domain_area", "")
            page_type = pg.get("page_type", "")
            _has_editable_dom = _page_has_editable_dom(pg)
            # ── Gemini HTML解析結果が存在すれば最優先で mapping 構築 ──
            _gemini_mapping = _build_mapping_from_gemini(pg, op_type, req)
            if _gemini_mapping and _gemini_mapping["validation_score"] > best_score:
                best_score   = _gemini_mapping["validation_score"]
                best_mapping = _gemini_mapping
                print(f"[P22_GEMINI_MAPPING_BUILT] op={op_type} url={pg.get('url','')[:60]} score={best_score} missing={_gemini_mapping['missing']}", flush=True)
            # list/dashboard/non_op pagesはtarget除外
            _pg_llm_purpose = pg.get("page_purpose", "")
            _expected_from_purpose = _PURPOSE_TO_OP.get(str(_pg_llm_purpose or ""))
            if _expected_from_purpose and _expected_from_purpose != op_type:
                print(f"[P22_PURPOSE_OP_GUARD] op={op_type} expected={_expected_from_purpose} purpose={_pg_llm_purpose} url={pg.get('url','')[:60]}", flush=True)
                continue
            _llm_is_op_target = _pg_llm_purpose in (
                "edit_page", "create_page", "news_post_page", "text_edit_page",
                "entity_edit_page", "media_upload_page", "schedule_page", "price_page",
                "status_page",
            )
            # [HTML_MENU_IMPORT] manual_importページはis_operation_candidateフラグを持たないためバイパス
            _is_manual_import = pg.get("manual_import", False)
            if not pg.get("is_operation_candidate", False) and not _llm_is_op_target and not _is_manual_import:
                if not _has_editable_dom:
                    print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=not_operation_candidate page_type={page_type} url={pg.get('url','')[:60]}", flush=True)
                    continue
                print(f"[P22_DOM_EVIDENCE_OVERRIDE] op={op_type} reason=editable_dom page_type={page_type} url={pg.get('url','')[:60]}", flush=True)
            if not pg.get("is_operation_candidate", False) and _llm_is_op_target:
                print(f"[P22_LLM_PURPOSE_OVERRIDE] op={op_type} page_purpose={_pg_llm_purpose} url={pg.get('url','')[:60]}", flush=True)
            if page_type in _NON_OP_PAGE_TYPES and not _llm_is_op_target and not _is_manual_import and not _has_editable_dom:
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=non_op_page_rejected page_type={page_type} url={pg.get('url','')[:60]}", flush=True)
                continue
            if area not in allowed_areas and page_type not in allowed_page_types and not _llm_is_op_target and not _is_manual_import and not _has_editable_dom:
                continue
            # [HTML_MENU_IMPORT] manual_importページはDOM情報なし→URLキーワードスコアで代替
            # [HTML_MENU_IMPORT] manual_importページ: DOMあり→通常パス / DOMなし→URLキーワードフォールバック
            if _is_manual_import:
                _has_dom = _has_editable_dom
                if not _has_dom:
                    _mi_url   = (pg.get("url") or "").lower()
                    _mi_title = (pg.get("title") or pg.get("html_title") or "").lower()
                    _mi_cat   = (pg.get("category") or "").lower()
                    _mi_text  = _mi_url + " " + _mi_title + " " + _mi_cat
                    _MI_KW = {
                        "news_post":       ["news","post","blog","diary","topic","topics","event","coupon","お知らせ","ニュース","投稿","新規","写メ","日記","freetext","contents","campaign","realtime","marquee","速報"],
                        "text_update":     ["profile","about","text","プロフィール","自己紹介","説明","紹介文","freetext","contents","con_txt","seo","concept","フリー","編集"],
                        "status_update":   ["status","public","private","表示","非表示","公開","停止","有効","無効","ステータス","state","visible","hidden","active","inactive","standby","girl","cast"],
                        "media_replace":   ["photo","image","gallery","写真","画像","メディア","upload","アップロード"],
                        "schedule_update": ["schedule","shift","出勤","予定","calendar","シフト","カレンダー"],
                        "price_update":    ["price","course","料金","コース","fee","システム"],
                        "entity_register": ["register","new","add","登録","新規","追加","cast","staff","キャスト","スタッフ"],
                        "entity_update":   ["edit","update","編集","更新","cast","staff","キャスト","スタッフ","profile","プロフィール"],
                    }
                    _mi_kws = _MI_KW.get(op_type, [])
                    _mi_score = sum(1 for k in _mi_kws if k in _mi_text) * 20
                    if _mi_score >= 20:
                        _mi_mapping = {
                            "status": "NEEDS_REVIEW",
                            "target_url": pg.get("url", ""),
                            "selectors": {},
                            "validation_score": _mi_score,
                            "missing": req.get("required", []),
                            "source": "manual_menu_import",
                            "executable": False,
                            "page_purpose": "",
                            "page_purpose_source": "manual_import",
                            "body_candidates": [],
                            "textareas": [],
                            "editor_candidates": [],
                        }
                        if _mi_score > best_score:
                            best_score   = _mi_score
                            best_mapping = _mi_mapping
                            print(f"[HTML_MENU_IMPORT_MATCH] op={op_type} url={str(pg.get('url',''))[:60]} score={_mi_score}", flush=True)
                    continue
                # DOMあり → is_operation_candidateをTrueとして通常パスへ
                print(f"[HTML_MENU_IMPORT_DOM_PATH] op={op_type} url={str(pg.get('url',''))[:60]} forms={len(pg.get('forms',[]))} inputs={len(pg.get('inputs',[]))}", flush=True)

            # C: operation intent URL/DOM制限
            _pg_url   = (pg.get("url") or "").lower()
            _pg_title = (pg.get("title") or pg.get("html_title") or "").lower()
            _INTENT_REQUIRED = {
                "schedule_update": ("shift","schedule","calendar","attendance","出勤","予定","シフト","カレンダー","勤務","在籍","atwork","sokuhime","速報"),
                "entity_register": ("cast","staff","girl","member","profile","user","item","product","キャスト","スタッフ","女の子","会員","プロフィール","商品","在籍","add","new","register","追加","登録","新規","zaiseki","hime","姫"),
                "entity_update":   ("cast","staff","girl","member","profile","user","item","product","キャスト","スタッフ","女の子","会員","プロフィール","商品","在籍","edit","update","編集","更新","zaiseki","hime","姫"),
                "media_replace":   ("photo","image","media","file","upload","img","写真","画像","素材","アップロード","写メ","動画","video","photo_diary","memlog","diary"),
                "price_update":    ("price","course","fee","料金","コース","pricelist","multifee","system","システム","割引","coupon","クーポン","option"),
                "status_update":   ("status","public","private","表示","非表示","公開","停止","有効","無効","ステータス","state","active","inactive","standby","girl","cast","list","sokuhime","速報","即ヒメ","hime"),
            }
            _intent_kws = _INTENT_REQUIRED.get(op_type)
            _intent_missing = False
            if _intent_kws and not any(k in _pg_url or k in _pg_title for k in _intent_kws):
                _intent_missing = True
                print("[OP_INTENT_GUARD_SOFT] op=" + op_type + " url=" + _pg_url[:60] + " reason=intent_keyword_missing", flush=True)
            list_only = False
            mapping = extract_operation_selectors_from_page(pg, op_type)
            _purpose_mapping = _build_mapping_from_page_purpose(pg, op_type, req)
            if _purpose_mapping and _purpose_mapping["validation_score"] > best_score:
                best_score = _purpose_mapping["validation_score"]
                best_mapping = _purpose_mapping
                print(f"[P22_PURPOSE_MAPPING_BUILT] op={op_type} purpose={_purpose_mapping.get('page_purpose')} url={str(pg.get('url',''))[:60]} score={best_score} missing={_purpose_mapping['missing']}", flush=True)
            if _is_manual_import:
                mapping["source"] = "manual_menu_import"
                mapping["page_purpose"] = pg.get("page_purpose", "")
                mapping["page_purpose_source"] = pg.get("page_purpose_source") or "manual_import"
            if op_type in ("news_post", "text_update"):
                _body_sel_data = (mapping.get("selectors") or {}).get("body") or {}
                _body_selector = (
                    _body_sel_data.get("selector")
                    if isinstance(_body_sel_data, dict)
                    else str(_body_sel_data or "")
                )
                _body_selector = str(_body_selector or "").lower()
                _body_evidence = " ".join(str(x or "") for x in (_body_sel_data.get("evidence", []) if isinstance(_body_sel_data, dict) else [])).lower()
                _body_tag = (_body_sel_data.get("tag") if isinstance(_body_sel_data, dict) else "") or ""
                _body_type = (_body_sel_data.get("type") if isinstance(_body_sel_data, dict) else "") or ""
                if (
                    _body_selector.startswith("select")
                    or _body_tag == "select"
                    or _body_type in ("checkbox", "radio")
                    or "type='checkbox'" in _body_selector
                    or "type='radio'" in _body_selector
                    or "tag=select" in _body_evidence
                ):
                    print(f"[P22_TEXT_BODY_REJECTED] op={op_type} url={_pg_url[:60]} reason=non_text_body_selector selector={_body_selector[:60]}", flush=True)
                    continue
            if _intent_missing:
                _specific_roles = {
                    "schedule_update": ("date", "date_input", "start_time", "end_time", "save"),
                    "status_update": ("body",),
                    "entity_register": ("name", "required_inputs", "save"),
                    "entity_update": ("name", "editable_inputs", "edit_trigger", "save"),
                    "media_replace": ("file",),
                    "price_update": ("price", "save"),
                }.get(op_type, ())
                _sel_keys = set((mapping.get("selectors") or {}).keys())
                _has_specific_role = bool(_sel_keys.intersection(_specific_roles))
                if op_type == "status_update":
                    _body_sel_data = (mapping.get("selectors") or {}).get("body") or {}
                    _status_signal_words = (
                        "status", "state", "public", "private", "visible", "hidden",
                        "active", "inactive", "enabled", "disabled", "display",
                        "公開", "非公開", "表示", "非表示", "有効", "無効", "状態", "ステータス",
                    )
                    _body_signal_text = ""
                    if isinstance(_body_sel_data, dict):
                        _body_signal_parts = (
                            [_body_sel_data.get("selector")]
                            + (_body_sel_data.get("matched") or [])
                            + (_body_sel_data.get("evidence") or [])
                        )
                        _body_signal_text = " ".join(str(x or "") for x in _body_signal_parts).lower()
                    _has_specific_role = any(k in _body_signal_text for k in _status_signal_words)
                    if not _has_specific_role:
                        print(f"[OP_INTENT_GUARD_HARD] op={op_type} url={_pg_url[:60]} reason=status_signal_missing", flush=True)
                        continue
                if not _has_specific_role and _sel_keys and _sel_keys.issubset({"save"}):
                    print(f"[OP_INTENT_GUARD_HARD] op={op_type} url={_pg_url[:60]} reason=save_only_without_intent", flush=True)
                    continue
                if mapping.get("status") == "READY" and not _has_specific_role:
                    mapping["status"] = "NEEDS_REVIEW"
                    mapping["executable"] = False
                    mapping["validation_score"] = min(int(mapping.get("validation_score") or 0), max(1, min_score - 1))
                    mapping["intent_keyword_missing"] = True
            if mapping["validation_score"] > best_score:
                best_score   = mapping["validation_score"]
                best_mapping = mapping
                best_mapping["page_purpose"] = pg.get("page_purpose", "")
                best_mapping["page_purpose_source"] = mapping.get("page_purpose_source") or pg.get("page_purpose_source", "")
                best_mapping["body_candidates"] = pg.get("body_candidates", [])
                best_mapping["textareas"] = pg.get("textareas", [])
                best_mapping["editor_candidates"] = pg.get("editor_candidates", [])
        if best_mapping and best_score >= min_score and not best_mapping["missing"]:
            _ready_target_url = str(best_mapping.get("target_url") or "")
            result[op_type] = {
                "status":           "READY",
                "target_url":       _ready_target_url,
                "selectors":        best_mapping["selectors"],
                "validation_score": best_score,
                "missing":          [],
                "source":           best_mapping.get("source", "dom_evidence_mapper"),
                "page_purpose":     best_mapping.get("page_purpose", ""),
                "page_purpose_source": best_mapping.get("page_purpose_source", ""),
                "form_schema":      best_mapping.get("form_schema", {}),
                "executable":       True,
            }
            print(f"[P22_OPERATION_MAPPING_BUILT] op={op_type} status=READY target_url={_ready_target_url[:60]} validation_score={best_score}", flush=True)
            if op_type == "schedule_update":
                print("[P22_SCHEDULE_TRACE] phase=save status=READY target_url=" + _ready_target_url[:80] + " validation_score=" + str(best_score) + " missing=" + str(best_mapping["missing"]), flush=True)
        elif best_mapping and best_score > 0:
            # NEEDS_REVIEW厳格化: saveだけ取れてもNEEDS_REVIEW不可
            _partial_url   = str(best_mapping.get("target_url") or "").lower()
            try:
                from urllib.parse import urlparse as _urlparse
                _parsed  = _urlparse(_partial_url)
                _is_root = (_parsed.path or "/") in ("/", "")
            except Exception:
                _is_root = False
            _partial_ng = (
                any(p in _partial_url for p in ("login","signin","root","dashboard","c1main"))
                or _is_root
            )
            # 指示書13番: news_post/text_updateはpage_purposeとbody系selector必須
            _body_ops = ("news_post", "text_update")
            if op_type in _body_ops:
                # [MANUAL_IMPORT_EXEMPT] manual_menu_import由来はpage_purpose/bodyチェック免除
                if best_mapping.get("source") == "manual_menu_import" or best_mapping.get("page_purpose_source") in ("manual_import_keyword", "manual_import"):
                    pass
                else:
                    _pg_purpose = best_mapping.get("page_purpose", "")
                    _pg_source = best_mapping.get("page_purpose_source", "")
                    _allowed_purposes = ("news_post_page", "text_edit_page", "edit_page", "create_page")
                    _has_body_sel = any(k in best_mapping.get("selectors", {}) for k in ("body", "content", "text", "textarea", "editor"))
                    _has_body_cand = bool(best_mapping.get("body_candidates") or best_mapping.get("textareas") or best_mapping.get("editor_candidates"))
                    _purpose_ok = not _pg_purpose or _pg_purpose in _allowed_purposes  # purpose=None(未分類)はOK
                    _has_title_sel = any(k in best_mapping.get("selectors", {}) for k in ("title", "subject", "headline"))
                    _has_save_sel = any(k in best_mapping.get("selectors", {}) for k in ("save", "submit", "send"))
                    _body_ok = _has_body_sel or _has_body_cand or (_has_title_sel and _has_save_sel)  # title+saveでbodyなしもOK
                    if not _purpose_ok or not _body_ok:
                        result[op_type] = {
                            "status": "UNDISCOVERED",
                            "target_url": "",
                            "selectors": {},
                            "validation_score": 0,
                            "missing": best_mapping.get("missing", req["required"]),
                            "error_reason": f"news_post/text_update rejected: purpose={_pg_purpose} body_ok={_body_ok}",
                            "source": "dom_evidence_mapper",
                            "executable": False,
                            "human_review_required": False,
                        }
                        print(f"[P22_BODY_OP_REJECTED] op={op_type} purpose={_pg_purpose} body_ok={_body_ok} url={str(best_mapping.get('target_url') or '')[:60]}", flush=True)
                        continue
            if best_mapping and best_score > 0 and best_mapping.get("target_url") and not _partial_ng:
                _review_target_url = str(best_mapping.get("target_url") or "")
                result[op_type] = {
                    "status":           "NEEDS_REVIEW",
                    "target_url":       _review_target_url,
                    "selectors":        best_mapping["selectors"],
                    "validation_score": best_score,
                    "missing":          best_mapping["missing"],
                    "source":           best_mapping.get("source", "dom_evidence_mapper"),
                    "page_purpose":     best_mapping.get("page_purpose", ""),
                    "page_purpose_source": best_mapping.get("page_purpose_source", ""),
                    "form_schema":      best_mapping.get("form_schema", {}),
                    "executable":       False,
                    "human_review_required": True,
                }
                print(f"[P22_OPERATION_MAPPING_BUILT] op={op_type} status=NEEDS_REVIEW target_url={_review_target_url[:60]} missing={best_mapping['missing']} validation_score={best_score}", flush=True)
                if op_type == "schedule_update":
                    print("[P22_SCHEDULE_TRACE] phase=save status=NEEDS_REVIEW target_url=" + _review_target_url[:80] + " validation_score=" + str(best_score) + " missing=" + str(best_mapping["missing"]), flush=True)
            else:
                result[op_type] = {
                    "status":           "UNDISCOVERED",
                    "target_url":       "",
                    "selectors":        {},
                    "validation_score": 0,
                    "missing":          req["required"],
                    "error_reason":     "partial_candidate_rejected",
                    "source":           "dom_evidence_mapper",
                    "executable":       False,
                }
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=partial_candidate_rejected url={str(best_mapping.get('target_url') or '')[:60]}", flush=True)
        else:
            # page_typeフォールバック: best_mappingが取れなくてもGeminiがop_typeと一致するpage_typeを返したページがあればNEEDS_REVIEW
            _PT_FALLBACK = {
                "entity_register": "entity_register", "entity_update": "entity_update",
                "schedule_update": "schedule_update", "news_post": "news_post",
                "text_update": "text_update", "media_replace": "media_replace",
                "price_update": "price_update", "status_update": "status_update",
            }
            _fallback_pg = None
            for _fp in pages:
                _fpt = str((_fp.get("form_schema") or {}).get("gemini_page_type") or "")
                if _PT_FALLBACK.get(_fpt) == op_type:
                    _fallback_pg = _fp
                    break
            if _fallback_pg:
                _fb_url = str(_fallback_pg.get("url") or "")
                _fb_fs  = _fallback_pg.get("form_schema") or {}
                _fb_pt  = str(_fb_fs.get("gemini_page_type") or "")
                result[op_type] = {
                    "status":            "NEEDS_REVIEW",
                    "target_url":        _fb_url,
                    "selectors":         {},
                    "validation_score":  5,
                    "missing":           req["required"],
                    "source":            "gemini_page_type_fallback",
                    "page_purpose":      _fb_pt,
                    "page_purpose_source": "gemini_html_analysis",
                    "form_schema":       _fb_fs,
                    "executable":        False,
                    "human_review_required": True,
                }
                print(f"[P22_PAGETYPE_FALLBACK] op={op_type} gemini_pt={_fb_pt} url={_fb_url[:60]}", flush=True)
            else:
                _nm_reason = "list_page_requires_followup_edit_form" if list_only else "no_qualifying_page"
                result[op_type] = {
                    "status":           "UNDISCOVERED",
                    "target_url":       "",
                    "selectors":        {},
                    "validation_score": 0,
                    "missing":          req["required"],
                    "error_reason":     _nm_reason,
                    "source":           "dom_evidence_mapper",
                    "executable":       False,
                }
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason={_nm_reason}", flush=True)
    return result


def build_media_structure_map(pages: list) -> dict:
    """
    P21.5: navigation_graph.pages から媒体構造マップを生成する。
    """
    import datetime as _dt_msm
    _area_names = ["content", "entity", "media", "schedule", "price",
                   "reservation", "customer", "analytics", "system", "help", "unknown"]
    areas = {a: {"pages": [], "operations": []} for a in _area_names}
    _ignored_reasons = {
        "reservation": "not supported operation area",
        "customer":    "not supported operation area",
        "analytics":   "not supported operation area",
        "system":      "not direct operation target",
        "help":        "not operation page",
    }
    for a, reason in _ignored_reasons.items():
        areas[a]["ignored_reason"] = reason

    entrypoints = {}
    total = 0
    candidate_count = 0
    ignored_count = 0

    for pg in pages:
        if not isinstance(pg, dict):
            continue
        total += 1
        classification = classify_admin_page_structure(pg)
        pg["page_type"]             = classification["page_type"]
        pg["domain_area"]           = classification["domain_area"]
        pg["operation_capability"]  = classification["operation_capability"]
        pg["is_operation_candidate"] = classification["is_operation_candidate"]
        pg["negative_reasons"]      = classification["negative_reasons"]
        pg["value_score"]           = classification["value_score"]

        area = classification["domain_area"]
        if area not in areas:
            area = "unknown"
        areas[area]["pages"].append(pg.get("url", ""))

        if classification["is_operation_candidate"]:
            candidate_count += 1
            for op in classification["operation_capability"]:
                if op not in areas[area]["operations"]:
                    areas[area]["operations"].append(op)
                # entrypoint: value_scoreが高いものを優先採用
                if (op not in entrypoints or classification["value_score"] > entrypoints[op].get("confidence", 0)) \
                        and classification["value_score"] >= 70:
                    entrypoints[op] = {
                        "url":           pg.get("url", ""),
                        "confidence":    classification["value_score"],
                        "evidence":      classification["evidence"],
                        "domain_area":   area,
                        "page_type":     classification["page_type"],
                        "required_roles": [],
                        "missing_roles":  [],
                    }
        else:
            ignored_count += 1
            if classification.get("ignored_for_operation"):
                pg["dom_evidence"] = pg.get("dom_evidence", {})
                pg["dom_evidence"]["ignored_for_operation"] = True
                pg["dom_evidence"]["ignore_reason"] = classification.get("ignore_reason", "low_value_page")

    areas_detected = [a for a in _area_names if areas[a]["pages"]]
    result = {
        "generated_at": _dt_msm.datetime.utcnow().isoformat(),
        "summary": {
            "total_pages":              total,
            "operation_candidate_pages": candidate_count,
            "ignored_pages":            ignored_count,
            "areas_detected":           areas_detected,
        },
        "areas":              areas,
        "operation_entrypoints": entrypoints,
    }
    print(
        f"[P21_STRUCTURE_MAP_SAVED] areas={areas_detected} entrypoints={list(entrypoints.keys())}",
        flush=True
    )
    return result


def _save_crawl_snapshot(
    db, mapping_id, nav_graph, visited, crawl_queue,
    status, max_pages, start_url, include_patterns, exclude_patterns,
    last_url="", timeout_reason=""
):
    """全経路共通のcrawl保存関数。navigation_graph.pages・crawl_state・crawl_resume_queueを保存。"""
    import datetime as _dt_snap
    _remaining = [u for u in crawl_queue if u not in visited]
    # status補正: resume_queueが残っているのにDONEにしない
    if _remaining and status == "DONE":
        status = "PAUSED_REMAINING"
    # [P21_COLLECT_ONLY] DOM/HTML収集専用。分類・推定・Operation生成は一切しない。
    # [P21_COLLECT_ONLY] DOM/HTML収集専用。分類・推定・Operation生成は一切しない。
    # [MANUAL_IMPORT_PRESERVE] Firestore既存のmanual_importページを保持してマージ
    _existing_manual_pages = []
    try:
        _existing_doc = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
        _existing_pages = (_existing_doc.get("navigation_graph") or {}).get("pages") or []
        _existing_manual_pages = [p for p in _existing_pages if p.get("manual_import")]
        # manual_importページが消えていてもmanual_menu_itemsから再生成
        if not _existing_manual_pages:
            _menu_items = _existing_doc.get("manual_menu_items") or []
            _seen_urls = set()
            for _mi in _menu_items:
                _abs = _mi.get("absolute_url") or _mi.get("href") or ""
                if _abs and _abs not in _seen_urls and _abs.startswith("http"):
                    _seen_urls.add(_abs)
                    _existing_manual_pages.append({
                        "url":           _abs,
                        "title":         _mi.get("title") or _mi.get("href") or "",
                        "category":      _mi.get("category") or "",
                        "manual_import": True,
                        "collected_at":  _existing_doc.get("manual_menu_imported_at", ""),
                    })
            if _existing_manual_pages:
                print(f"[MANUAL_IMPORT_REGENERATED] mapping_id={mapping_id} regenerated={len(_existing_manual_pages)}", flush=True)
        if _existing_manual_pages:
            print(f"[MANUAL_IMPORT_PRESERVE] mapping_id={mapping_id} preserved={len(_existing_manual_pages)}", flush=True)
    except Exception as _mipreserve_err:
        print(f"[MANUAL_IMPORT_PRESERVE_ERROR] {_mipreserve_err}", flush=True)
    _pages_list = []
    for _u, _v in nav_graph.items():
        if not isinstance(_v, dict):
            continue
        _pages_list.append({
            "url":               _v.get("url", _u),
            "title":             _v.get("title", ""),
            "manual_import":       _v.get("manual_import", False),
            "forms_count":       _v.get("forms_count", 0),
            "inputs_count":      _v.get("inputs_count", 0),
            "buttons_count":     _v.get("buttons_count", 0),
            "file_inputs_count": _v.get("file_inputs_count", 0),
            "textareas_count":   _v.get("textareas_count", 0),
            "selects_count":     _v.get("selects_count", 0),
            "links_count":       len(_v.get("links", [])),
            "tables_count":      _v.get("tables", 0),
            "inputs":            _v.get("inputs", [])[:120],
            "textareas":         _v.get("textareas", [])[:30],
            "buttons":           _v.get("buttons", [])[:80],
            "links":             _v.get("links", [])[:200],
            "forms":             _v.get("forms", [])[:20],
            "file_inputs":       _v.get("file_inputs", [])[:20],
            "menu_items":        _v.get("menu_items", [])[:50],
            "followup_links":    _v.get("followup_links", [])[:20],
            "source_chain":      _v.get("source_chain", []),
            "link_text":         _v.get("link_text", ""),
            "raw_snapshot":      True,
            "collected_at":      _dt_snap.datetime.utcnow().isoformat(),
            "dom_evidence": {
                "has_form":       _v.get("forms_count", 0) > 0,
                "has_input":      _v.get("inputs_count", 0) > 0,
                "has_file_input": _v.get("file_inputs_count", 0) > 0,
                "has_button":     _v.get("buttons_count", 0) > 0,
                "has_textarea":   _v.get("textareas_count", 0) > 0,
            },
        })
        print(f"[P21_PAGE_COLLECTED] url={_v.get('url',_u)[:80]} forms={_v.get('forms_count',0)} inputs={_v.get('inputs_count',0)} buttons={_v.get('buttons_count',0)} links={len(_v.get('links',[]))}", flush=True)
    # [MANUAL_IMPORT_MERGE] manual_importページをcrawl済みpagesにマージ（上書き防止）
    _crawled_urls = {p.get("url") for p in _pages_list}
    for _mp in _existing_manual_pages:
        if _mp.get("url") not in _crawled_urls:
            _pages_list.append(_mp)
    _crawl_state = {
        "status":           status,
        "updated_at":       _dt_snap.datetime.utcnow().isoformat(),
        "last_url":         last_url,
        "pages_crawled":    len(visited),
        "remaining_count":  len(_remaining),
        "resume_queue_count": len(_remaining),
        "max_pages":        max_pages,
        "start_url":        start_url,
        "include_patterns": include_patterns or [],
        "exclude_patterns": exclude_patterns or [],
        "timeout_reason":   timeout_reason,
    }
    _rq_save = _remaining[:200] if _remaining else []
    # [P21_COLLECT_ONLY] media_structure_map生成はP24で行う
    # E-1: トランザクションで書き込み — 並列クロールによるページデータ消失を防止
    from google.cloud import firestore as _fs_e1
    _ref_snap = db.collection("media_mappings").document(mapping_id)
    _txn_snap = db.transaction()

    @_fs_e1.transactional
    def _atomic_crawl_save(_txn, _ref, _pages, _state, _rq, _dt):
        _snap_doc = _ref.get(transaction=_txn)
        _existing = (_snap_doc.to_dict() or {}).get("navigation_graph", {}).get("pages") or []
        # manual_importページが並列クロールで消えないようマージ
        _existing_by_url = {p.get("url"): p for p in _existing if p.get("manual_import")}
        _new_by_url      = {p.get("url"): p for p in _pages}
        _merged = {**_existing_by_url, **_new_by_url}
        _final  = list(_merged.values())[:300]
        _txn.update(_ref, {
            "navigation_graph.pages":  _final,
            "navigation_graph.updated_at": _dt,
            "crawl_state":             _state,
            "crawl_status":            _state["status"],
            "crawl_resume_queue":      _rq,
        })
        return len(_final)

    try:
        _saved_count = _atomic_crawl_save(_txn_snap, _ref_snap, _pages_list, _crawl_state, _rq_save,
                                          _dt_snap.datetime.utcnow().isoformat())
        print(
            f"[P21_SNAPSHOT_SAVED] mapping_id={mapping_id} status={status} pages={_saved_count} remaining={len(_remaining)}",
            flush=True
        )
    except Exception as _snap_err:
        print(f"[P21_SNAPSHOT_SAVE_ERROR] {_snap_err}", flush=True)
    return _crawl_state


def _navigate_admin_link_or_goto(page, target_url):
    """
    P21巡回遷移: 現在ページ内にtarget_urlと一致するa[href]があればクリック優先。
    失敗時はpage.goto fallback。媒体専用if禁止・URL語句判定禁止。
    """
    try:
        hrefs = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({href: e.getAttribute('href'), text: (e.innerText||'').trim()}))"
        )
    except Exception as _e:
        print(f"[P21_NAV_CLICK] href走査失敗 reason={_e} fallback to goto", flush=True)
        hrefs = []

    matched_el_index = None
    for i, item in enumerate(hrefs):
        raw_href = item.get("href", "") or ""
        from urllib.parse import urljoin
        normalized = urljoin(page.url, raw_href)
        if normalized == target_url:
            matched_el_index = i
            break

    if matched_el_index is not None:
        print(f"[P21_NAV_CLICK] target={target_url}", flush=True)
        try:
            els = page.query_selector_all("a[href]")
            el = els[matched_el_index]
            el.click(timeout=8000)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            actual_url = page.url
            print(f"[P21_NAV_CLICK_DONE] requested={target_url} actual={actual_url}", flush=True)
            return
        except Exception as _click_err:
            print(f"[P21_NAV_GOTO_FALLBACK] target={target_url} reason=click_failed:{_click_err}", flush=True)
    else:
        print(f"[P21_NAV_GOTO_FALLBACK] target={target_url} reason=link_not_found_in_page", flush=True)

    try:
        page.goto(target_url, wait_until="domcontentloaded", timeout=10000)
        actual_url = page.url
        print(f"[P21_NAV_GOTO_DONE] requested={target_url} actual={actual_url}", flush=True)
    except Exception as _goto_err:
        print(f"[P21_NAV_GOTO_DONE] requested={target_url} actual=ERROR reason={_goto_err}", flush=True)


def post_login_admin_crawl(
    page,
    mapping_id: str,
    db,
    max_pages: int = 200,
    start_url: str = "",
    include_patterns: list = None,
    exclude_patterns: list = None,
    reset_resume: bool = False,
) -> dict:
    """
    P21: ログイン後の管理画面を自動解析する（完全同期版）。
    Playwright sync APIのみ使用。asyncio/await禁止。
    GET遷移のみ。POST/保存/削除/更新ボタンは押さない。
    ID/PASSはログ出力しない。
    結果を media_mappings/{mapping_id} に保存する。
    resume_queue対応: タイムアウト時に未巡回URLを保存し次回続きから再開。
    """
    import datetime as _dt
    from urllib.parse import urlparse, parse_qs

    print(f"[P21_CRAWL_START] mapping_id={mapping_id}", flush=True)

    def _is_blocked_url(url: str) -> bool:
        u = str(url or "").lower()
        # [BLOCKED_STATIC] static file block before queue insert
        _static_exts = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
                        ".zip", ".exe", ".mp4", ".mp3", ".mov", ".avi", ".doc", ".docx",
                        ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt", ".xml", ".json")
        _u_path = u.split("?")[0]
        if any(_u_path.endswith(ext) for ext in _static_exts):
            return True
        if any(p in u for p in ["/pdf/", "/manual/", "/help/", "/img/cms/", "manual.pdf", "/download/"]):
            return True
        if any(p in u for p in ["logout", "signout", "sign-out", "log-out", "delete", "remove", "destroy"]):
            return True
        try:
            _p = urlparse(url)
            _qs = parse_qs(_p.query)
            if len(_qs) >= 3:
                return True
            _id_keys = [k for k in _qs if k.lower().endswith("_id") or k.lower() == "id"]
            if _id_keys and len(_qs) <= 2:
                return True
        except Exception:
            pass
        return False

    def _is_same_origin(base: str, target: str) -> bool:
        try:
            b = urlparse(base)
            t = urlparse(target)
            return b.scheme == t.scheme and b.netloc == t.netloc
        except Exception:
            return False

    def _normalize_href(href: str, base_url: str) -> str:
        from urllib.parse import urlparse as _up, urljoin as _uj
        if not href:
            return ""
        if href.startswith("//"):
            parsed = _up(base_url)
            return f"{parsed.scheme}:{href}"
        elif href.startswith("/"):
            parsed = _up(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{href}"
        return _uj(base_url, href)

    def _analyze_page_full(pg) -> dict:
        result = {}
        try:
            result["url"]   = pg.url
            result["title"] = pg.title()
        except Exception:
            result["url"]   = ""
            result["title"] = ""

        # frame対応: pg本体 + 全frameを走査して合算
        try:
            all_frames = [pg]
            for _f in pg.frames:
                if _f not in all_frames:
                    all_frames.append(_f)
        except Exception:
            all_frames = [pg]
        print(f"[P21_FRAME_SCAN] page_url={pg.url} frames={len(all_frames)}", flush=True)

        # 合算バケット
        links = []
        menu_items = []
        inputs_list = []
        textareas_list = []
        forms_list = []
        fi_list = []
        buttons_list = []
        tables_count = 0
        selects_count = 0

        for ctx in all_frames:
            ctx_url = ""
            try:
                ctx_url = ctx.url
            except Exception:
                pass
            try:
                anchors = ctx.query_selector_all("a[href]")
                for a in anchors[:300]:
                    try:
                        href = a.get_attribute("href") or ""
                        text = a.inner_text().strip()[:80]
                        if href and not href.startswith("javascript"):
                            links.append({
                                "href":       href,
                                "text":       text,
                                "class":      (a.get_attribute("class") or "")[:80],
                                "title":      (a.get_attribute("title") or "")[:60],
                                "aria_label": (a.get_attribute("aria-label") or "")[:60],
                                "onclick":    (a.get_attribute("onclick") or "")[:80],
                                "rel":        (a.get_attribute("rel") or "")[:40],
                                "frame_url":  ctx_url,
                            })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                nav_els = ctx.query_selector_all("nav a, .sidebar a, .menu a, #menu a, #nav a, [class*=menu] a, [class*=nav] a, [class*=sidebar] a")
                for el in nav_els[:100]:
                    try:
                        href = el.get_attribute("href") or ""
                        text = el.inner_text().strip()[:60]
                        if text:
                            menu_items.append({"href": href, "text": text, "frame_url": ctx_url})
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                inp_els = ctx.query_selector_all("input:not([type=hidden]):not([type=password]):not([type=file]), select")
                for el in inp_els[:120]:
                    try:
                        inputs_list.append({
                            "type":        (el.get_attribute("type") or "")[:20],
                            "name":        (el.get_attribute("name") or "")[:60],
                            "id":          (el.get_attribute("id") or "")[:60],
                            "placeholder": (el.get_attribute("placeholder") or "")[:80],
                            "aria_label":  (el.get_attribute("aria-label") or "")[:60],
                            "class":       (el.get_attribute("class") or "")[:80],
                            "frame_url":   ctx_url,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                ta_els = ctx.query_selector_all("textarea")
                for el in ta_els[:30]:
                    try:
                        textareas_list.append({
                            "name":        (el.get_attribute("name") or "")[:60],
                            "id":          (el.get_attribute("id") or "")[:60],
                            "placeholder": (el.get_attribute("placeholder") or "")[:80],
                            "aria_label":  (el.get_attribute("aria-label") or "")[:60],
                            "class":       (el.get_attribute("class") or "")[:80],
                            "frame_url":   ctx_url,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                form_els = ctx.query_selector_all("form")
                for fm in form_els[:20]:
                    try:
                        fm_inputs = fm.query_selector_all("input:not([type=hidden]):not([type=password])")
                        fm_btns   = fm.query_selector_all("button, input[type=submit]")
                        forms_list.append({
                            "action":        (fm.get_attribute("action") or "")[:100],
                            "method":        (fm.get_attribute("method") or "")[:10],
                            "class":         (fm.get_attribute("class") or "")[:80],
                            "id":            (fm.get_attribute("id") or "")[:60],
                            "name":          (fm.get_attribute("name") or "")[:60],
                            "inputs_count":  len(fm_inputs),
                            "buttons_count": len(fm_btns),
                            "frame_url":     ctx_url,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                fi_els = ctx.query_selector_all("input[type=file]")
                for el in fi_els[:20]:
                    try:
                        fi_list.append({
                            "name":       (el.get_attribute("name") or "")[:60],
                            "id":         (el.get_attribute("id") or "")[:60],
                            "class":      (el.get_attribute("class") or "")[:80],
                            "accept":     (el.get_attribute("accept") or "")[:60],
                            "aria_label": (el.get_attribute("aria-label") or "")[:60],
                            "frame_url":  ctx_url,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                btn_els = ctx.query_selector_all("button, input[type=submit], input[type=button], a.btn, a[class*=btn]")
                for b in btn_els[:100]:
                    try:
                        buttons_list.append({
                            "text":       b.inner_text().strip()[:60],
                            "value":      (b.get_attribute("value") or "")[:60],
                            "type":       (b.get_attribute("type") or "")[:20],
                            "class":      (b.get_attribute("class") or "")[:80],
                            "id":         (b.get_attribute("id") or "")[:40],
                            "name":       (b.get_attribute("name") or "")[:40],
                            "onclick":    (b.get_attribute("onclick") or "")[:80],
                            "aria_label": (b.get_attribute("aria-label") or "")[:60],
                            "href":       (b.get_attribute("href") or "")[:80],
                            "frame_url":  ctx_url,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                tables_count += len(ctx.query_selector_all("table"))
            except Exception:
                pass
            try:
                selects_count += len(ctx.query_selector_all("select"))
            except Exception:
                pass
            print(f"[P21_FRAME_DOM_SUMMARY] frame_url={ctx_url} links={len(links)} inputs={len(inputs_list)} forms={len(forms_list)} buttons={len(buttons_list)} file_inputs={len(fi_list)}", flush=True)

        result["links"]             = links
        result["menu_items"]        = menu_items
        result["inputs"]            = inputs_list
        result["inputs_count"]      = len(inputs_list)
        result["textareas"]         = textareas_list
        result["textareas_count"]   = len(textareas_list)
        result["forms"]             = forms_list
        result["forms_count"]       = len(forms_list)
        result["file_inputs"]       = fi_list
        result["file_inputs_count"] = len(fi_list)
        result["buttons"]           = buttons_list
        result["buttons_count"]     = len(buttons_list)
        result["tables"]            = tables_count
        result["selects_count"]     = selects_count
        return result

    try:
        import time as _time

        base_url  = page.url
        visited   = set()
        nav_graph = {}
        _source_chain_map = {}  # url -> source_chain記録（設計図4章）

        # JS描画完了待機
        try:
            page.wait_for_load_state("domcontentloaded", timeout=35000)
        except Exception:
            pass

        # 引数初期化
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []
        # reset_resume=Trueなら保存済みresume_queueを破棄
        if reset_resume:
            try:
                db.collection("media_mappings").document(mapping_id).update({"crawl_resume_queue": []})
                print(f"[P21_RESET_RESUME] mapping_id={mapping_id}", flush=True)
            except Exception as _rre:
                print(f"[P21_RESET_RESUME_ERROR] {_rre}", flush=True)
        # start_urlが指定されていればそこへ移動してbase_urlを更新
        if start_url and start_url.startswith("http"):
            try:
                page.goto(start_url, timeout=35000)
                page.wait_for_load_state("domcontentloaded", timeout=35000)
                base_url = start_url
                print(f"[P21_START_URL] moved to start_url={start_url}", flush=True)
            except Exception as _su_err:
                print(f"[P21_START_URL_ERROR] {_su_err}", flush=True)
        # resume_queue確認（前回の続きがあれば再開）
        resume_queue = []
        try:
            _mm = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _rq = _mm.get("crawl_resume_queue") or []
            if _rq and not reset_resume:
                resume_queue = _rq
                print(f"[P21_RESUME] resume_queue={len(resume_queue)}", flush=True)
                # 既存nav_graphを引き継ぐ
                _existing_nav = _mm.get("navigation_graph") or {}
                for k, v in _existing_nav.items():
                    nav_graph[k] = v
                    visited.add(k)
        except Exception as _re:
            print(f"[P21_RESUME_ERROR] {_re}", flush=True)

        # トップページ解析
        # [LOGIN_REDIRECT_DETECT] if still on login page after login, try to find admin top
        _cur_url = str(page.url or "").lower()
        _login_kws = ('signin', 'login', 'auth', 'c1login')
        if any(k in _cur_url for k in _login_kws) and not start_url:
            # try clicking first non-login link
            try:
                _admin_links = page.eval_on_selector_all('a[href]', 'els => els.map(e => e.getAttribute("href")).filter(h => h && !h.includes("login") && !h.includes("signin") && !h.startsWith("#") && !h.startsWith("javascript"))')
                _admin_top = next((h for h in _admin_links if 'admin' in str(h or '').lower() or 'manage' in str(h or '').lower()), None) or next((h for h in _admin_links if str(h or '').startswith('/')), None)
                if _admin_top:
                    from urllib.parse import urljoin as _uj_rdr
                    _admin_top_abs = _uj_rdr(page.url, _admin_top)
                    page.goto(_admin_top_abs, wait_until='domcontentloaded', timeout=15000)
                    page.wait_for_timeout(2000)
                    base_url = page.url
                    print(f'[LOGIN_REDIRECT_NAVIGATED] redirected to admin top: {base_url[:80]}', flush=True)
            except Exception as _rdr_err:
                print(f'[LOGIN_REDIRECT_ERROR] {_rdr_err}', flush=True)
        top_structure = _analyze_page_full(page)
        nav_graph[base_url] = top_structure
        visited.add(base_url)
        print(f"[P21_PAGE] url={top_structure.get('url','')} title={top_structure.get('title','')}", flush=True)

        # 巡回対象リンク収集
        # [ADMIN_CRAWL_START] if start_url specified, navigate there first
        if start_url and start_url != base_url:
            try:
                _navigate_admin_link_or_goto(page, start_url)
                page.wait_for_load_state('domcontentloaded', timeout=15000)
                page.wait_for_timeout(2000)
                _start_structure = _analyze_page_full(page)
                nav_graph[start_url] = _start_structure
                visited.add(start_url)
                print(f'[P21_START_URL_LOADED] url={start_url[:80]}', flush=True)
                top_structure = _start_structure
                base_url = start_url
            except Exception as _su_err:
                print(f'[P21_START_URL_ERROR] {_su_err}', flush=True)
        crawl_queue = list(resume_queue) if resume_queue else []
        # [修正C] リンク優先キーワード定数
        _KW_CANDIDATE = ('cast','staff','girl','profile','photo','image','media','shift','schedule','edit','list','regist','new',
                         'topics','news','freetext','event','content','coupon','banner','diary','topic',
                         'キャスト','スタッフ','写真','画像','出勤','登録','編集','ニュース','投稿','イベント')
        _KW_DANGER     = ('logout','delete','remove','destroy')
        _candidate_links1 = []
        _ignored_links1   = []
        for link in top_structure.get("links", []):
            href = _normalize_href(link.get("href", ""), base_url)
            _ltext = (link.get("text") or "").lower()
            _lhref_low = str(href or "").lower()
            _is_kw  = any(k in _lhref_low or k in _ltext for k in _KW_CANDIDATE)
            _is_dng = any(k in _lhref_low for k in _KW_DANGER)
            if (href and href not in visited and _is_same_origin(base_url, href)
                    and not _is_blocked_url(href) and href not in crawl_queue
                    and _crawl_url_allowed(href, include_patterns, exclude_patterns)):
                if _is_kw and not _is_dng:
                    crawl_queue.insert(0, href)
                    _candidate_links1.append(href)
                elif not _is_dng:
                    crawl_queue.append(href)
                else:
                    _ignored_links1.append(href)
            elif _is_dng:
                _ignored_links1.append(href)
            if len(crawl_queue) >= max_pages:
                break
        print(f"[P21_LINK_DISCOVERY] url={base_url[:80]} links_count={len(top_structure.get('links',[]))} candidate_links={_candidate_links1[:5]} ignored_links={_ignored_links1[:5]}", flush=True)
        _crawl_start = _time.time()
        _CRAWL_TIME_LIMIT = 90
        while crawl_queue:
            url = crawl_queue.pop(0)
            if url in visited:
                continue
            if _time.time() - _crawl_start > _CRAWL_TIME_LIMIT:
                print(f"[P21_CRAWL_TIMEOUT] 巡回時間上限{_CRAWL_TIME_LIMIT}秒到達 visited={len(visited)}", flush=True)
                # 未巡回URLをFirestoreに保存（次回再開用）
                _save_crawl_snapshot(
                    db, mapping_id, nav_graph, visited, crawl_queue,
                    status="PAUSED_TIMEOUT", max_pages=max_pages,
                    start_url=start_url, include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns, last_url=url,
                    timeout_reason="crawl_time_limit_reached"
                )
                break
            try:
                _navigate_admin_link_or_goto(page, url)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
                try:
                    page.wait_for_timeout(2500)
                except Exception:
                    pass
                try:
                    page.wait_for_selector("body, frameset, frame, iframe, form, input, textarea, button, table, a", timeout=8000)
                except Exception:
                    pass
                try:
                    structure = _analyze_page_full(page)
                except Exception as _dom_err:
                    print(f"[P21_DOM_EVIDENCE_EMPTY] url={page.url} reason={type(_dom_err).__name__}: {_dom_err}", flush=True)
                    structure = {
                        "url": page.url,
                        "title": page.title(),
                        "links": [],
                        "inputs_count": 0,
                        "forms_count": 0,
                        "file_inputs_count": 0,
                        "buttons_count": 0,
                        "textareas_count": 0,
                        "selects_count": 0,
                        "tables": 0,
                        "menu_items": [],
                    }
                print(f"[P21_DOM_SUMMARY] url={structure.get('url','')[:80]} forms={structure.get('forms_count',0)} inputs={structure.get('inputs_count',0)} buttons={structure.get('buttons_count',0)} file_inputs={structure.get('file_inputs_count',0)} links={len(structure.get('links',[]))}", flush=True)
                nav_graph[url] = structure
                # source_chain付与（設計図4章）
                if url in _source_chain_map:
                    _sc = _source_chain_map[url]
                    # edit_page_detected を末尾に追加
                    if not any(s.get("type") == "edit_page_detected" for s in _sc):
                        _sc.append({"type": "edit_page_detected", "url": url})
                    structure["source_chain"] = _sc
                    print(f"[P21_SOURCE_CHAIN_ATTACHED] url={url[:60]} chain_len={len(_sc)}", flush=True)
                else:
                    structure["source_chain"] = []
                visited.add(url)
                print(f"[P21_PAGE] url={structure.get('url','')} title={structure.get('title','')}", flush=True)
                # [修正C-2] 新リンクをキューに追加（キーワード優先）
                _candidate_links2 = []
                _ignored_links2   = []
                for _lnk in structure.get("links", []):
                    _href = _normalize_href(_lnk.get("href", ""), url)
                    _ltxt2 = (_lnk.get("text") or "").lower()
                    _lhref2 = str(_href or "").lower()
                    _is_kw2  = any(k in _lhref2 or k in _ltxt2 for k in _KW_CANDIDATE)
                    _is_dng2 = any(k in _lhref2 for k in _KW_DANGER)
                    if (_href and _href not in visited and _href not in crawl_queue
                            and _is_same_origin(base_url, _href)
                            and not _is_blocked_url(_href)
                            and _crawl_url_allowed(_href, include_patterns, exclude_patterns)
                            and len(crawl_queue) < max_pages):
                        if _is_kw2 and not _is_dng2:
                            crawl_queue.insert(0, _href)
                            _candidate_links2.append(_href)
                            # [ADMIN_URL_PRE_CLASSIFY] link_textをnav_graphに事前記録
                            if _href not in nav_graph:
                                nav_graph[_href] = {"url": _href, "link_text": (_lnk.get("text") or "")}
                            elif not nav_graph[_href].get("link_text"):
                                nav_graph[_href]["link_text"] = (_lnk.get("text") or "")
                            _pre_cls = classify_admin_url_by_rules(
                                url=_href,
                                link_text=(_lnk.get("text") or ""),
                            )
                            if _pre_cls["score"] > 0:
                                print(
                                    f"[ADMIN_URL_RULE_MATCH] url={_normalize_admin_url_for_classification(_href)[:120]}"
                                    f" page_type={_pre_cls['page_type']} ops={_pre_cls['operations']}"
                                    f" score={_pre_cls['score']} matched={_pre_cls['matched_keywords']}",
                                    flush=True
                                )
                        elif not _is_dng2:
                            crawl_queue.append(_href)
                        else:
                            _ignored_links2.append(_href)
                    elif _is_dng2:
                        _ignored_links2.append(_href)
                print(f"[P21_LINK_DISCOVERY] url={url[:80]} links_count={len(structure.get('links',[]))} candidate_links={_candidate_links2[:5]} ignored_links={_ignored_links2[:5]}", flush=True)
                # followupリンク優先追加（listページからedit/new/detail候補を探す）
                _pg_type = structure.get("page_type", "")
                if _pg_type in ("entity_list", "content_list", "media_list", "list_or_nav"):
                    _followup = _extract_followup_edit_links(structure, base_url)
                    structure["followup_links"] = _followup
                    for _fu in _followup:
                        if (_fu not in visited and _fu not in crawl_queue
                                and _is_same_origin(base_url, _fu)
                                and not _is_blocked_url(_fu)
                                and _crawl_url_allowed(_fu, include_patterns, exclude_patterns)
                                and len(crawl_queue) < max_pages):
                            crawl_queue.insert(0, _fu)
                            _source_chain_map[_fu] = [
                                {"type": "list_page", "url": url},
                                {"type": "follow_edit_link", "url": _fu},
                            ]
                            print(f"[P21_SOURCE_CHAIN_RECORDED] from={url[:60]} to={_fu[:60]}", flush=True)
                            print(f"[P21_FOLLOWUP_ENQUEUED] from_url={url[:60]} to_url={_fu[:60]}", flush=True)
                    if _followup:
                        print(f"[P21_FOLLOWUP_LINK_FOUND] url={url[:60]} followup={_followup[:3]} reason={_pg_type}", flush=True)
            except Exception as _nav_err:
                print(f"[P21] navigate error: {url} {type(_nav_err).__name__}", flush=True)
        else:
            # 全ページ巡回完了
            try:
                _save_crawl_snapshot(
                    db, mapping_id, nav_graph, visited, crawl_queue,
                    status="DONE", max_pages=max_pages,
                    start_url=start_url, include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns, last_url="",
                    timeout_reason=""
                )
                print(f"[P21_CRAWL_COMPLETE] visited={len(visited)}", flush=True)
            except Exception:
                pass

        now = _dt.datetime.utcnow()

        # capabilities推定
        all_file_inputs = sum(v.get("file_inputs_count", 0) for v in nav_graph.values())
        all_inputs      = sum(v.get("inputs_count", 0) for v in nav_graph.values())
        all_forms       = sum(v.get("forms_count", 0) for v in nav_graph.values())
        all_tables      = sum(v.get("tables", 0) for v in nav_graph.values())

        capabilities = {
            "can_login":           True,
            "can_verify":          True,
            "can_navigate_admin":  len(nav_graph) > 1,
            "can_upload_image":    all_file_inputs > 0,
            "can_update_text":     all_inputs > 5,
            "can_post_news":       all_forms > 0,
            "can_update_schedule": any("schedule" in str(u or "").lower() or "sch" in str(u or "").lower() or "出勤" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_update_price":    any("price" in str(u or "").lower() or "course" in str(u or "").lower() or "料金" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_register_entity": any("register" in str(u or "").lower() or "new" in str(u or "").lower() or "add" in str(u or "").lower() or "edit" in str(u or "").lower() or "追加" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_update_entity":   any("edit" in str(u or "").lower() or "list" in str(u or "").lower() or "一覧" in (v.get("title","")) for u, v in nav_graph.items()),
        }

        # [P21_COLLECT_ONLY] operation_candidates生成はP23で行う

        # Firestore保存
        _nav_save = {}
        for k, v in nav_graph.items():
            _nav_save[k] = {
                "url":               v.get("url", ""),
                "title":             v.get("title", ""),
                "inputs_count":      v.get("inputs_count", 0),
                "forms_count":       v.get("forms_count", 0),
                "file_inputs_count": v.get("file_inputs_count", 0),
                "buttons_count":     v.get("buttons_count", 0),
                "links_count":       len(v.get("links", [])),
                "tables":            v.get("tables", 0),
                "inputs":            v.get("inputs", [])[:120],
                "textareas":         v.get("textareas", [])[:30],
                "forms":             v.get("forms", [])[:20],
                "buttons":           v.get("buttons", [])[:80],
                "file_inputs":       v.get("file_inputs", [])[:20],
                "menu_items":        v.get("menu_items", [])[:50],
                "links":             v.get("links", [])[:200],
                "dom_evidence":      {
                    "has_form":        v.get("forms_count", 0) > 0,
                    "has_input":       v.get("inputs_count", 0) > 0,
                    "has_file_input":  v.get("file_inputs_count", 0) > 0,
                    "has_button":      v.get("buttons_count", 0) > 0,
                },
            }

        db.collection("media_mappings").document(mapping_id).update({
            "navigation_graph":        _nav_save,
            "capabilities":            capabilities,
            "pages_crawled":           len(visited),
            "crawler_last_run_at":     now,
            "login_health":            "HEALTHY",
        })
        # navigation_graph.pagesを確実に保存（修正7整合）
        _save_crawl_snapshot(
            db, mapping_id, nav_graph, visited, [],
            status="DONE", max_pages=max_pages,
            start_url=start_url, include_patterns=include_patterns,
            exclude_patterns=exclude_patterns, last_url="",
            timeout_reason=""
        )

        # [P21_COLLECT_ONLY] operation_mappings生成はP24で行う。P21巡回完了後は収集データ保存のみ。
        # verify_selector自動保存
        try:
            _existing = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _svs = _existing.get("verify_selector")
            if not _svs:
                # タイトルベースでverify_selector候補を探す
                for _url, _vd in nav_graph.items():
                    if "signin" not in str(_url or "").lower() and "login" not in str(_url or "").lower():
                        _vs_cand = "title"
                        db.collection("media_mappings").document(mapping_id).update({
                            "verify_selector": _vs_cand
                        })
                        print(f"[P21_VERIFY_SELECTOR_SAVED] verify_selector={_vs_cand}", flush=True)
                        break
        except Exception as _vs_err:
            print(f"[P21_VERIFY_SELECTOR_SAVE_ERROR] {_vs_err}", flush=True)

        total_inputs  = sum(v.get("inputs_count", 0) for v in nav_graph.values())
        total_forms   = sum(v.get("forms_count", 0) for v in nav_graph.values())
        total_files   = sum(v.get("file_inputs_count", 0) for v in nav_graph.values())
        total_buttons = sum(v.get("buttons_count", 0) for v in nav_graph.values())
        total_tables  = sum(v.get("tables", 0) for v in nav_graph.values())

        print(f"[P21_COUNTS] mapping_id={mapping_id} url={base_url} pages={len(visited)} links=0 forms={total_forms} inputs={total_inputs} textareas=0 buttons={total_buttons} file_inputs={total_files} tables={total_tables}", flush=True)

        # crawl_stateをFirestoreから取得して戻り値に含める
        _final_cs = {}
        try:
            _final_mm = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _final_cs = _final_mm.get("crawl_state") or {}
        except Exception:
            pass
        _remaining_count = len([u for u in (db.collection("media_mappings").document(mapping_id).get().to_dict() or {}).get("crawl_resume_queue", [])])
        return {
            "status":                  "OK",
            "pages_crawled":           len(visited),
            "capabilities":            capabilities,
            "crawl_state":             _final_cs,
            "crawl_resume_queue_count": _remaining_count,
            "crawl_paused":            _final_cs.get("status") == "PAUSED_TIMEOUT",
            "last_crawled_url":        _final_cs.get("last_url", ""),
        }

    except Exception as e:
        import traceback
        print(f"[P21_CRAWL_ERROR] {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {"status": "ERROR", "error": type(e).__name__}


# ==============================================================
# HTML Menu Import（初期媒体構造把握の正式ルート）
# ==============================================================

def parse_menu_html(raw_html: str, source_url: str = "") -> list:
    """
    管理メニューHTMLから汎用的にカテゴリ・タイトル・URLを抽出する。
    <a>タグを直接取得し、親要素・見出し・URLキーワードからカテゴリを推定。
    """
    try:
        from html.parser import HTMLParser
        from urllib.parse import urljoin, urlsplit, urlunsplit
        max_items = max(20, min(int(os.environ.get("HTML_MENU_IMPORT_MAX_ITEMS", "300")), 1000))

        URL_CATEGORY_MAP = [
            (["news", "info", "notice", "topics", "blog", "post", "article", "osira", "oshira", "topics2"], "ニュース・お知らせ"),
            (["cast", "staff", "member", "girl", "lady", "talent", "actor", "player"], "キャスト・スタッフ"),
            (["schedule", "shift", "calendar", "attend", "syukkin", "attend"], "スケジュール"),
            (["price", "fee", "charge", "ryokin", "course", "plan"], "料金・コース"),
            (["photo", "image", "gallery", "picture", "media", "upload"], "写真・メディア"),
            (["profile", "about", "shop", "store", "company", "base", "gaiyou"], "店舗・会社情報"),
            (["recruit", "apply", "entry", "job", "work", "boshuu"], "採用・応募"),
            (["login", "logout", "auth", "session", "signin"], "認証"),
            (["setting", "config", "admin", "manage", "dashboard", "kanri"], "管理・設定"),
            (["contact", "inquiry", "message", "mail", "toiawase"], "お問い合わせ"),
            (["edit", "update", "modify", "change", "henshu"], "編集・更新"),
            (["list", "index", "search", "find", "ichiran"], "一覧・検索"),
            (["register", "create", "add", "new", "touroku"], "登録・追加"),
            (["delete", "remove", "trash"], "削除"),
            (["report", "stat", "analyze", "access", "log"], "レポート・統計"),
            (["marquee", "banner", "proto", "pickup", "special", "feature"], "コンテンツ管理"),
            (["reserve", "yoyaku", "booking"], "予約管理"),
            (["area", "region", "local"], "エリア"),
            (["mail", "magazine", "mag", "melma"], "メールマガジン"),
            (["vote", "ranking", "rank", "hyouka"], "投票・ランキング"),
            (["faq", "help", "support", "question"], "サポート"),
        ]

        def guess_category_from_url(url: str) -> str:
            u = str(url or "").lower()
            for keywords, cat in URL_CATEGORY_MAP:
                if any(k in u for k in keywords):
                    return cat
            return ""

        def make_absolute(href: str, base: str) -> str:
            if not href or href.startswith("#") or href.startswith("javascript"):
                return ""
            if href.startswith("http"):
                return href
            if base:
                return urljoin(base, href)
            return href

        def canonicalize_href_for_dedupe(href: str) -> str:
            try:
                from urllib.parse import parse_qsl, urlencode
                parsed = urlsplit(href or "")
                kept = []
                for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                    key_l = str(key or "").lower()
                    if key_l in {"sid", "z", "token", "nonce", "_token", "csrf", "entry_sid", "entry_time", "phpsessid", "session", "sessionid"}:
                        continue
                    if key_l.startswith("utm_") or key_l in {"gclid", "fbclid", "yclid"}:
                        continue
                    kept.append((key, value))
                kept.sort(key=lambda row: (str(row[0]), str(row[1])))
                return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), (parsed.path or "").rstrip("/"), urlencode(kept, doseq=True), ""))
            except Exception:
                return str(href or "").strip()

        def fallback_title_from_href(href: str) -> str:
            try:
                parsed = urlsplit(href or "")
                raw_path = parsed.path or ""
                last = raw_path.split("/")[-1] if raw_path else ""
                last = last.rsplit(".", 1)[0] if "." in last else last
                last = last.replace("_", " ").replace("-", " ").strip()
                if last:
                    return last
            except Exception:
                pass
            href = str(href or "").strip()
            return href if len(href) <= 80 else "..." + href[-80:]

        class _GenericMenuParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = []
                self.current_category = ""
                self._tag_stack = []
                self._in_a = False
                self._current_href = ""
                self._current_text = ""
                self._order = 0
                self._in_heading = False
                self._heading_text = ""
                self._heading_tag = ""
                self._in_category_tag = False
                self._category_text = ""
                self._category_stack = []  # カテゴリ候補スタック

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                self._tag_stack.append((tag, attrs_dict))

                # 見出しタグ・定義タグ
                if tag in ("h1","h2","h3","h4","h5","h6","dt","th","caption"):
                    self._in_heading = True
                    self._heading_tag = tag
                    self._heading_text = ""

                # カテゴリ候補となるクラス名パターン（広めに取る）
                cls = str(attrs_dict.get("class") or "").lower()
                id_ = str(attrs_dict.get("id") or "").lower()
                combined = cls + " " + id_
                category_keywords = [
                    "subheading","heading","category","group","section","label",
                    "title","nav-header","menu-header","nav_header","menu_title",
                    "sidebar","chapter","block-title","list-header","item-header",
                    "tab-label","accordion","panel-title","widget-title","box-title",
                ]
                if tag in ("li","div","span","p","strong","b","td","th") and any(k in combined for k in category_keywords):
                    self._in_category_tag = True
                    self._category_text = ""

                if tag == "a":
                    href = attrs_dict.get("href","")
                    self._in_a = True
                    self._current_href = href
                    self._current_text = ""

            def handle_endtag(self, tag):
                if self._tag_stack:
                    self._tag_stack.pop()

                if tag in ("h1","h2","h3","h4","h5","h6","dt","th","caption") and self._in_heading:
                    text = self._heading_text.strip()
                    if text and len(text) < 40:
                        self.current_category = text
                        self._category_stack.append(text)
                    self._in_heading = False
                    self._heading_text = ""

                if self._in_category_tag and tag in ("li","div","span","p","strong","b","td","th"):
                    text = self._category_text.strip()
                    if text and len(text) < 40 and not text.startswith("http"):
                        self.current_category = text
                        self._category_stack.append(text)
                    self._in_category_tag = False
                    self._category_text = ""

                if tag == "a" and self._in_a:
                    href = self._current_href
                    text = self._current_text.strip()
                    # imgタグのみのリンクはalt属性から取得試行済み → スキップ
                    abs_url = make_absolute(href, source_url)
                    if abs_url and not href.startswith("#") and not href.startswith("javascript"):
                        # テキストなし（画像リンク等）はhrefからタイトル推定
                        if not text:
                            text = fallback_title_from_href(href)
                        if text:
                            # カテゴリ決定: 現在カテゴリ → URLから推定 → スタック末尾 → その他
                            cat = self.current_category
                            if not cat and self._category_stack:
                                cat = self._category_stack[-1]
                            if not cat:
                                cat = guess_category_from_url(abs_url)
                            if not cat:
                                cat = "その他"
                            self._order += 1
                            self.items.append({
                                "type":         "menu_link",
                                "category":     cat,
                                "title":        text,
                                "href":         href,
                                "absolute_url": abs_url,
                                "canonical_url": canonicalize_href_for_dedupe(abs_url),
                                "visible":      True,
                                "target_blank": False,
                                "order":        self._order,
                                "source":       "manual_html_import",
                            })
                    self._in_a = False
                    self._current_href = ""
                    self._current_text = ""

            def handle_data(self, data):
                if self._in_a:
                    self._current_text += data
                if self._in_heading:
                    self._heading_text += data
                if self._in_category_tag:
                    self._category_text += data

        parser = _GenericMenuParser()
        parser.feed(raw_html)

        # 重複URLを除去（同じhrefは1件のみ）
        seen = set()
        unique_items = []
        for item in parser.items:
            key = item.get("canonical_url") or item.get("absolute_url") or item.get("href")
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
            if len(unique_items) >= max_items:
                break
        truncated = len(parser.items) > len(unique_items)
        if truncated:
            print(f"[HTML_MENU_IMPORT_TRUNCATED] raw_items={len(parser.items)} returned={len(unique_items)} max_items={max_items}", flush=True)
        print(f"[HTML_MENU_IMPORT] source_url={source_url[:80] if source_url else ''} items={len(unique_items)}", flush=True)
        return unique_items
    except Exception as e:
        print(f"[HTML_MENU_IMPORT_ERROR] {e}", flush=True)
        return []


def expand_menu_links_from_seed_urls(
    media_mapping: dict,
    creds: dict,
    seed_items: list[dict],
    start_url: str = "",
    max_pages: int = 25,
    max_links_per_page: int = 160,
) -> dict:
    """
    HTML importなどで得た入口URLを起点に、認証済みブラウザで配下ページを辿り、
    タブ・編集導線・iframe内リンクを追加発見する。
    Operationの確定までは行わず、manual_menu_items を増やすための補助探索だけを担当する。
    """
    if not is_playwright_enabled():
        return {"ok": False, "error": "PLAYWRIGHT_ENABLED=false", "status": "SKIPPED"}

    from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

    seed_items = [it for it in (seed_items or []) if isinstance(it, dict)]
    if not seed_items and not start_url:
        return {"ok": False, "error": "seed_items empty", "status": "SKIPPED"}

    _scope_url = (
        str(start_url or "").strip()
        or str(media_mapping.get("login_url") or "").strip()
        or str(media_mapping.get("media_url") or "").strip()
    )
    _scope_parsed = urlparse(_scope_url) if _scope_url else None
    _allowed_host = (_scope_parsed.netloc or "").lower() if _scope_parsed else ""
    _skip_kw = (
        "logout", "signout", "logoff", "help", "support", "faq", "manual", "disclaimer",
        "download", "csv", "print", "tel:", "mailto:", "javascript:",
        "pcmode=sp", "?op=newc", "/newc", "preview", "公開ページ", "スマホ",
    )

    def _same_host(abs_url: str) -> bool:
        if not _allowed_host:
            return True
        try:
            return urlparse(abs_url).netloc.lower() == _allowed_host
        except Exception:
            return False

    def _should_skip(abs_url: str, text: str = "") -> bool:
        blocked, _ = _is_static_or_blocked_url(abs_url)
        if blocked:
            return True
        blob = f"{abs_url} {text}".lower()
        return any(tok in blob for tok in _skip_kw)

    def _canonical_menu_url(abs_url: str) -> str:
        try:
            parsed = urlsplit(str(abs_url or "").strip())
            kept = []
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                key_l = str(key or "").lower()
                if key_l in {"sid", "z", "token", "nonce", "_token", "csrf", "entry_sid", "entry_time", "phpsessid", "session", "sessionid"}:
                    continue
                if key_l.startswith("utm_") or key_l in {"gclid", "fbclid", "yclid"}:
                    continue
                kept.append((key, value))
            kept.sort(key=lambda row: (str(row[0]), str(row[1])))
            return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), (parsed.path or "").rstrip("/"), urlencode(kept, doseq=True), ""))
        except Exception:
            return str(abs_url or "").strip()

    def _collect_candidates(page) -> tuple[list[dict], list[str]]:
        frames = []
        try:
            frames = [page] + list(page.frames)
        except Exception:
            frames = [page]
        out: list[dict] = []
        frame_urls: list[str] = []
        for ctx in frames:
            ctx_url = str(getattr(ctx, "url", "") or "")
            if ctx_url:
                frame_urls.append(ctx_url)
            try:
                rows = ctx.evaluate(
                    """() => {
                      const base = document.baseURI || window.location.href;
                      const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                      const makeAbsolute = (href) => {
                        try { return new URL(href, base).href; } catch { return ''; }
                      };
                      const seen = new Set();
                      const out = [];
                      const push = (href, text, kind) => {
                        if (!href) return;
                        const abs = makeAbsolute(href);
                        if (!abs || seen.has(abs)) return;
                        seen.add(abs);
                        out.push({
                          href,
                          absolute_url: abs,
                          text: norm(text).slice(0, 80),
                          kind,
                        });
                      };
                      Array.from(document.querySelectorAll('a[href]')).forEach((el) => {
                        push(el.getAttribute('href') || '', el.textContent || el.title || el.getAttribute('aria-label') || '', 'anchor');
                      });
                      Array.from(document.querySelectorAll('[onclick]')).forEach((el) => {
                        const onclick = el.getAttribute('onclick') || '';
                        const m = onclick.match(/(?:location\\.href|window\\.location\\.href|window\\.open|open)\\s*\\(?\\s*['"]([^'"]+)['"]/);
                        if (m && m[1]) {
                          push(m[1], el.textContent || el.getAttribute('value') || el.title || el.getAttribute('aria-label') || '', 'onclick');
                        }
                      });
                      Array.from(document.querySelectorAll('form[action]')).forEach((el) => {
                        const submit = el.querySelector('button, input[type=submit], input[type=button]');
                        const label = submit ? (submit.textContent || submit.value || '') : '';
                        push(el.getAttribute('action') || '', label, 'form_action');
                      });
                      Array.from(document.querySelectorAll('iframe[src], frame[src]')).forEach((el) => {
                        push(el.getAttribute('src') || '', el.getAttribute('title') || el.getAttribute('name') || el.id || '', 'frame');
                      });
                      return out;
                    }"""
                ) or []
                for row in rows:
                    if isinstance(row, dict):
                        row["frame_url"] = ctx_url
                        out.append(row)
            except Exception as exc:
                print(f"[MENU_EXPAND_COLLECT_ERROR] frame={ctx_url[:80]} err={type(exc).__name__}", flush=True)
        return out, frame_urls

    def _page_summary_from_raw(raw_rows: list[dict]) -> dict:
        return {
            "links": [r for r in raw_rows if r.get("tag") == "a"],
            "buttons": [r for r in raw_rows if r.get("tag") == "button"],
            "forms": [r for r in raw_rows if r.get("tag") == "form"],
            "menu_items": [
                {"href": r.get("absolute_url") or r.get("href") or "", "text": r.get("text") or ""}
                for r in raw_rows
                if isinstance(r, dict) and (r.get("absolute_url") or r.get("href"))
            ],
        }

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            auth = create_authenticated_page(p, media_mapping, creds)
            browser, page = auth["browser"], auth["page"]
            try:
                if start_url and start_url != page.url:
                    try:
                        page.goto(start_url, timeout=20000, wait_until="domcontentloaded")
                        page.wait_for_timeout(800)
                    except Exception as exc:
                        print(f"[MENU_EXPAND_START_GOTO_ERROR] url={start_url[:80]} err={type(exc).__name__}", flush=True)

                current_url = str(page.url or "")
                source_url = start_url or current_url or _scope_url
                if not _allowed_host and source_url:
                    try:
                        _allowed_host = urlparse(source_url).netloc.lower()
                    except Exception:
                        _allowed_host = ""

                link_map: dict[str, dict] = {}
                queue: list[str] = []
                queued: set[str] = set()
                visited: set[str] = set()
                frame_sources: set[str] = set()

                def _add_link(abs_url: str, text: str = "", source_kind: str = "", raw_href: str = "") -> bool:
                    abs_url = str(abs_url or "").strip()
                    text = str(text or "").strip()
                    if not abs_url or not abs_url.startswith("http"):
                        return False
                    key = _canonical_menu_url(abs_url)
                    if not key:
                        return False
                    if not _same_host(abs_url):
                        return False
                    if _should_skip(abs_url, text):
                        return False
                    existing = link_map.get(key)
                    if existing:
                        if text and (not existing.get("text") or len(text) < len(str(existing.get("text") or ""))):
                            existing["text"] = text[:80]
                        return False
                    link_map[key] = {
                        "href": raw_href or abs_url,
                        "absolute_url": abs_url,
                        "canonical_url": key,
                        "text": text[:80] or abs_url,
                        "source_kind": source_kind or "expanded",
                    }
                    return True

                def _enqueue(abs_url: str, text: str = "") -> None:
                    key = _canonical_menu_url(abs_url)
                    if not abs_url or not key or key in visited or key in queued:
                        return
                    if not _same_host(abs_url) or _should_skip(abs_url, text):
                        return
                    queued.add(key)
                    queue.append(abs_url)

                for item in seed_items:
                    _abs = str(item.get("absolute_url") or item.get("href") or "").strip()
                    _txt = str(item.get("title") or item.get("text") or item.get("label") or "").strip()
                    if _add_link(_abs, _txt, "seed", item.get("href") or _abs):
                        _enqueue(_abs, _txt)

                if source_url and source_url.startswith("http"):
                    _enqueue(source_url, "source")

                while queue and len(visited) < max(1, int(max_pages or 25)):
                    target_url = queue.pop(0)
                    target_key = _canonical_menu_url(target_url)
                    queued.discard(target_key)
                    if target_key in visited:
                        continue
                    visited.add(target_key)
                    try:
                        page.goto(target_url, timeout=25000, wait_until="domcontentloaded")
                        page.wait_for_timeout(700)
                        try:
                            page.wait_for_load_state("networkidle", timeout=2500)
                        except Exception:
                            pass
                    except Exception as exc:
                        print(f"[MENU_EXPAND_GOTO_ERROR] url={target_url[:80]} err={type(exc).__name__}", flush=True)
                        continue

                    try:
                        rows, frame_urls = _collect_candidates(page)
                        for fr_url in frame_urls:
                            if fr_url and fr_url.startswith("http") and _same_host(fr_url) and not _should_skip(fr_url):
                                frame_sources.add(fr_url)
                        for row in rows[: max(20, int(max_links_per_page or 160))]:
                            abs_url = str(row.get("absolute_url") or row.get("href") or "").strip()
                            text = str(row.get("text") or "").strip()
                            if _add_link(abs_url, text, row.get("kind") or "page", row.get("href") or abs_url):
                                _enqueue(abs_url, text)
                    except Exception as exc:
                        print(f"[MENU_EXPAND_DISCOVERY_ERROR] url={target_url[:80]} err={type(exc).__name__}", flush=True)

                    try:
                        raw_scan = _raw_scan_page(page)
                        followups = _extract_followup_edit_links(_page_summary_from_raw(raw_scan), target_url)
                        for f_url in followups[:20]:
                            if _add_link(f_url, "followup", "followup", f_url):
                                _enqueue(f_url, "followup")
                    except Exception as exc:
                        print(f"[MENU_EXPAND_FOLLOWUP_ERROR] url={target_url[:80]} err={type(exc).__name__}", flush=True)

                for fr_url in list(frame_sources)[:50]:
                    _add_link(fr_url, "frame", "frame", fr_url)

                if not link_map:
                    return {
                        "ok": False,
                        "status": "EMPTY",
                        "error": "expanded_links_empty",
                        "source_url": source_url,
                        "visited_pages": len(visited),
                    }

                html_lines = [f"<!-- expanded_menu_discovery from {source_url} -->"]
                for rec in list(link_map.values())[:1000]:
                    text = str(rec.get("text") or rec.get("absolute_url") or "").replace("<", "&lt;").replace(">", "&gt;")
                    href = str(rec.get("absolute_url") or rec.get("href") or "")
                    html_lines.append(f'<a href="{href}">{text}</a>')
                merged_items = parse_menu_html("\n".join(html_lines), source_url=source_url)

                _seed_meta = {}
                for item in seed_items:
                    _abs = str(item.get("absolute_url") or item.get("href") or "").strip()
                    if not _abs:
                        continue
                    _seed_meta[_canonical_menu_url(_abs)] = {
                        "title": str(item.get("title") or item.get("text") or item.get("label") or "").strip(),
                        "category": str(item.get("category") or "").strip(),
                    }
                for item in merged_items:
                    _abs = str(item.get("absolute_url") or item.get("href") or "").strip()
                    _meta = _seed_meta.get(_canonical_menu_url(_abs)) or {}
                    if _meta.get("title") and (not item.get("title") or item.get("title") == _abs):
                        item["title"] = _meta["title"]
                    if _meta.get("category") and item.get("category") in ("", "その他"):
                        item["category"] = _meta["category"]

                print(
                    f"[MENU_EXPAND_DONE] mapping={media_mapping.get('mapping_id', '?')} "
                    f"seed={len(seed_items)} expanded={len(merged_items)} visited={len(visited)}",
                    flush=True,
                )
                return {
                    "ok": True,
                    "status": "OK",
                    "items": merged_items,
                    "source_url": source_url,
                    "visited_pages": len(visited),
                    "raw_count": len(link_map),
                }
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as exc:
        import traceback
        print(f"[MENU_EXPAND_ERROR] {exc}\n{traceback.format_exc()[:700]}", flush=True)
        return {"ok": False, "status": "ERROR", "error": str(exc)}


def auto_discover_menu_links(media_mapping: dict, creds: dict, start_url: str = "") -> dict:
    """
    Playwrightでログイン後、管理トップページの全ナビリンク（<a>+onclickボタン）を自動抽出する。
    HTMLを手動で貼らなくても管理メニュー構造を取得できる。
    返り値: {ok, items, source_url, raw_count}  items は parse_menu_html と同形式
    """
    if not is_playwright_enabled():
        return {"ok": False, "error": "PLAYWRIGHT_ENABLED=false のため自動取得は無効です"}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            auth = create_authenticated_page(p, media_mapping, creds)
            browser, page = auth["browser"], auth["page"]
            try:
                # ログイン後の現在URLを優先（ログイン完了後はダッシュボードにいる）
                post_login_url = page.url
                nav_url = start_url if start_url else post_login_url
                if not nav_url or nav_url == "about:blank":
                    nav_url = media_mapping.get("login_url", "")
                if not nav_url:
                    return {"ok": False, "error": "start_urlまたはlogin_urlが必要です"}

                # start_urlが指定された場合のみ遷移（そうでなければログイン後のページをそのまま使用）
                if start_url and start_url != post_login_url:
                    try:
                        page.goto(start_url, timeout=15000, wait_until="networkidle")
                    except Exception:
                        page.goto(start_url, timeout=15000, wait_until="domcontentloaded")
                else:
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                page.wait_for_timeout(1000)
                current_url = page.url

                # <a>タグ全取得
                raw_links = page.evaluate("""() => {
                    const base = document.baseURI;
                    function makeAbsolute(href) {
                        try { return new URL(href, base).href; } catch { return href; }
                    }
                    return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        href: a.getAttribute('href') || '',
                        absolute_url: makeAbsolute(a.getAttribute('href') || ''),
                        text: (a.textContent || a.title || '').trim().slice(0, 80),
                    })).filter(a =>
                        a.href &&
                        !a.href.startsWith('#') &&
                        !a.href.startsWith('javascript:') &&
                        a.absolute_url.startsWith('http')
                    );
                }""")

                # onclickベースのボタン・input[type=button]も取得
                onclick_links = page.evaluate("""() => {
                    const base = document.baseURI;
                    function extractHref(onclick) {
                        const m = onclick.match(/location\\.href\\s*=\\s*['"]([^'"]+)['"]/);
                        return m ? m[1] : null;
                    }
                    function makeAbsolute(href) {
                        try { return new URL(href, base).href; } catch { return href; }
                    }
                    return Array.from(document.querySelectorAll('[onclick]')).map(el => {
                        const onclick = el.getAttribute('onclick') || '';
                        const href = extractHref(onclick);
                        if (!href) return null;
                        const absUrl = makeAbsolute(href);
                        if (!absUrl.startsWith('http')) return null;
                        return {
                            href: href,
                            absolute_url: absUrl,
                            text: (el.textContent || el.getAttribute('value') || el.title || '').trim().slice(0, 80),
                        };
                    }).filter(x => x !== null);
                }""")

                all_links = raw_links + [x for x in onclick_links if x]

                # Level 2: 各サブページに入ってタブメニュー・ページ内ナビを追加取得
                # サイドバーには無いが各ページ内に存在するタブ（新規登録・編集など）を発見するため
                visited_l1 = set(lk.get("absolute_url") for lk in all_links if lk.get("absolute_url"))
                extra_links = []
                uniq_sub_urls = list(dict.fromkeys(
                    lk.get("absolute_url") for lk in all_links if lk.get("absolute_url")
                ))
                for sub_url in uniq_sub_urls[:15]:  # 最大15ページ巡回
                    try:
                        page.goto(sub_url, timeout=7000, wait_until="domcontentloaded")
                        page.wait_for_timeout(400)
                        tabs = page.evaluate("""() => {
                            const base = document.baseURI;
                            function makeAbsolute(href) {
                                try { return new URL(href, base).href; } catch { return href; }
                            }
                            const selectors = [
                                'ul.girl-menu a', 'ul.pagemenu a', 'ul.tab-menu a', 'ul.subMenu a',
                                '.page-menu a', '.page-tab a', '.tabs a', '.sub-menu a',
                                '[class*="tab"] a', '[class*="submenu"] a', '[class*="page-menu"] a'
                            ];
                            let found = [];
                            const seen = new Set();
                            for (const sel of selectors) {
                                try {
                                    Array.from(document.querySelectorAll(sel)).forEach(a => {
                                        const href = a.getAttribute('href') || '';
                                        const absUrl = makeAbsolute(href);
                                        if (href && !href.startsWith('#') && !href.startsWith('javascript:')
                                            && absUrl.startsWith('http') && !seen.has(absUrl)) {
                                            seen.add(absUrl);
                                            found.push({
                                                href: href,
                                                absolute_url: absUrl,
                                                text: (a.textContent || '').trim().slice(0, 60),
                                            });
                                        }
                                    });
                                } catch(e) {}
                            }
                            return found;
                        }""")
                        for t in tabs:
                            abs_url = t.get("absolute_url", "")
                            if abs_url and abs_url not in visited_l1:
                                visited_l1.add(abs_url)
                                extra_links.append(t)
                    except Exception as e2:
                        print(f"[AUTO_DISCOVER_L2] skip {sub_url[:60]}: {e2}", flush=True)
                        continue

                if extra_links:
                    all_links = all_links + extra_links
                    print(f"[AUTO_DISCOVER_L2] found {len(extra_links)} tab links from sub-pages", flush=True)

                # parse_menu_htmlに渡すHTML生成（既存のカテゴリ推定ロジックを再利用）
                html_lines = [f"<!-- auto_discover from {current_url} -->"]
                for lk in all_links[:400]:
                    text = (lk.get("text") or "").replace("<", "&lt;").replace(">", "&gt;")
                    href = lk.get("href") or ""
                    html_lines.append(f'<a href="{href}">{text}</a>')

                menu_items = parse_menu_html("\n".join(html_lines), source_url=current_url)

                print(f"[AUTO_DISCOVER_MENU] mapping={media_mapping.get('mapping_id', '?')} raw={len(all_links)} items={len(menu_items)} url={current_url}", flush=True)
                return {"ok": True, "items": menu_items, "source_url": current_url, "raw_count": len(all_links)}
            finally:
                try: browser.close()
                except Exception: pass
    except Exception as e:
        import traceback
        print(f"[AUTO_DISCOVER_MENU_ERROR] {e}\n{traceback.format_exc()[:500]}", flush=True)
        return {"ok": False, "error": str(e)}


# ── サイトプレビュー（スクリーンショット取得）────────────────────────────────

def take_site_preview(media_mapping: dict, target_url: str = "", extract_all_form_elements: bool = False) -> dict:
    """
    ログイン後のサイトをスクリーンショットしてbase64で返す。
    マッピング済みセレクターのbounding_boxも付与する。
    extract_all_form_elements=True のとき、ページ上の全インタラクティブ要素を form_elements として返す。
    ID/PASS・Cookieはログ・戻り値に絶対含めない。
    """
    if not is_playwright_enabled():
        return {
            "status": "WAITING_EXECUTOR",
            "executed": False,
            "message": "PLAYWRIGHT_ENABLED=false のためプレビュー取得は無効です。",
        }

    url = target_url or media_mapping.get("media_url") or media_mapping.get("login_url") or ""
    if not url:
        return {
            "status": "BLOCKED",
            "executed": False,
            "message": "media_url / login_url が未設定です。",
        }

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    _creds_reason = "no_secret_name"
    if secret_name:
        try:
            creds = get_secret_json(secret_name)
            if creds and creds.get("blocked"):
                _creds_reason = f"blocked: {creds.get('error', '?')}"
                creds = None
            elif creds:
                _creds_reason = "ok"
        except Exception as _se:
            _creds_reason = f"exception: {type(_se).__name__}"
            creds = None
    print(f"[SITE_PREVIEW_CREDS] secret_name={secret_name!r} result={_creds_reason}", flush=True)

    try:
        import base64
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        VIEWPORT_W, VIEWPORT_H = 1280, 800

        with sync_playwright() as p:
            browser = None
            page = None
            try:
                # ログイン試行
                _login_ok = False
                if creds:
                    try:
                        auth = create_authenticated_page(p, media_mapping, creds)
                        browser = auth["browser"]
                        page = auth["page"]
                        _login_ok = True
                        print(f"[SITE_PREVIEW_LOGIN_OK] url={page.url}", flush=True)
                        # ターゲットURLへ移動（ログイン後URLと異なる場合）
                        if target_url and target_url != page.url:
                            from urllib.parse import urlparse, urlencode, parse_qs
                            _auth_qs2 = parse_qs(urlparse(page.url).query, keep_blank_values=True)
                            _tgt_prs2 = urlparse(target_url)
                            _tgt_base2 = f"{_tgt_prs2.scheme}://{_tgt_prs2.netloc}{_tgt_prs2.path}"
                            _orig_qs2 = parse_qs(_tgt_prs2.query, keep_blank_values=True)
                            _NEVER2 = {"z", "nonce", "csrf", "_token"}
                            _REFRESH2 = {"sid", "token", "session", "sess"}
                            def _bld2(orig, fresh):
                                r = {}
                                for k, v in orig.items():
                                    if k.lower() in _NEVER2: continue
                                    if k in fresh: r[k] = fresh[k]
                                    elif k.lower() in _REFRESH2: continue
                                    else: r[k] = v
                                return r
                            try:
                                # Strategy1: 認証済みページ上のリンクをクリック
                                _clicked2 = False
                                try:
                                    if page.locator(f"a[href*='{_tgt_prs2.path}']").count() > 0:
                                        page.locator(f"a[href*='{_tgt_prs2.path}']").first.click(timeout=5000)
                                        page.wait_for_load_state("domcontentloaded", timeout=15000)
                                        page.wait_for_timeout(1000)
                                        _clicked2 = True
                                        print(f"[SITE_PREVIEW] Clicked nav link to {_tgt_prs2.path}", flush=True)
                                except Exception:
                                    pass
                                # Strategy2: 汎用パラメータ処理（CSRF除去・ページパラメータ保持）
                                if not _clicked2 or not _is_authenticated_page(page):
                                    _nqs2 = _bld2(_orig_qs2, _auth_qs2)
                                    _nav2 = _tgt_base2 + ("?" + urlencode(_nqs2, doseq=True) if _nqs2 else "")
                                    print(f"[SITE_PREVIEW] Strategy2 universal nav: {_nav2[:120]}", flush=True)
                                    page.goto(_nav2, timeout=15000)
                                    page.wait_for_load_state("domcontentloaded", timeout=30000)
                                # Strategy3: パスのみ
                                if not _is_authenticated_page(page):
                                    page.goto(_tgt_base2, timeout=15000)
                                    page.wait_for_load_state("domcontentloaded", timeout=30000)
                            except PlaywrightTimeout:
                                pass
                    except Exception as _le:
                        print(f"[SITE_PREVIEW_LOGIN_FAILED] {type(_le).__name__}: {_le}", flush=True)
                        _login_ok = False
                        if browser:
                            try:
                                browser.close()
                            except Exception:
                                pass
                            browser = None
                        page = None

                # ログイン不要 or 失敗時: 公開ページを開く
                if not _login_ok:
                    browser = p.chromium.launch(headless=True)
                    ctx = browser.new_context(
                        viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                    )
                    page = ctx.new_page()
                    try:
                        page.goto(url, timeout=20000)
                        page.wait_for_load_state("domcontentloaded", timeout=30000)
                    except PlaywrightTimeout:
                        return {"status": "BLOCKED", "executed": False, "message": "URLの読み込みがタイムアウトしました。"}

                # フルページスクリーンショット（スクロール全体）
                try:
                    page.set_viewport_size({"width": VIEWPORT_W, "height": VIEWPORT_H})
                except Exception:
                    pass
                try:
                    screenshot_bytes = page.screenshot(type="png", full_page=True)
                except Exception:
                    screenshot_bytes = page.screenshot(type="png", clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": VIEWPORT_H})
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                try:
                    VIEWPORT_H = page.evaluate("() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, 800)")
                except Exception:
                    pass

                current_url = page.url
                try:
                    title = page.title()
                except Exception:
                    title = ""

                # マッピング済みセレクターのbounding_box取得（ログイン後の操作ページで取得できるもの）
                dom_selectors = media_mapping.get("dom_selectors") or {}
                field_boxes = []
                SKIP_KEYS = {"username", "password", "login_submit", "verify"}
                for key, selector in dom_selectors.items():
                    if not selector or key in SKIP_KEYS:
                        continue
                    try:
                        el = page.locator(str(selector)).first
                        bbox = el.bounding_box(timeout=2000)
                        if bbox and bbox["width"] > 0 and bbox["height"] > 0:
                            field_boxes.append({
                                "key": key,
                                "selector": str(selector),
                                "x": round(bbox["x"], 1),
                                "y": round(bbox["y"], 1),
                                "w": round(bbox["width"], 1),
                                "h": round(bbox["height"], 1),
                            })
                    except Exception:
                        pass

                # ページHTML取得 + サニタイズ（script除去 + base href注入）
                page_html = ""
                try:
                    import re as _re
                    _raw_html = page.content()
                    # scriptタグを除去（セキュリティ）
                    _safe = _re.sub(
                        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
                        '', _raw_html, flags=_re.IGNORECASE | _re.DOTALL
                    )
                    # noscript除去
                    _safe = _re.sub(r'<noscript[^>]*>.*?</noscript>', '', _safe, flags=_re.IGNORECASE | _re.DOTALL)
                    # base href注入（相対URLを元サイトで解決）
                    _base = f'<base href="{current_url}" target="_blank">'
                    if _re.search(r'<head[^>]*>', _safe, _re.IGNORECASE):
                        _safe = _re.sub(r'(<head[^>]*>)', r'\1' + _base, _safe, count=1, flags=_re.IGNORECASE)
                    else:
                        _safe = _base + _safe
                    # 400KB上限（大規模サイト対策）
                    if len(_safe) > 400_000:
                        _safe = _safe[:400_000]
                    page_html = _safe
                except Exception as _he:
                    print(f"[SITE_PREVIEW_HTML_ERROR] {_he}", flush=True)
                    page_html = ""

                # 全フォーム要素抽出（extract_all_form_elements=True のとき）
                form_elements = []
                if extract_all_form_elements:
                    try:
                        form_elements = page.evaluate("""() => {
                            const results = [];
                            const els = document.querySelectorAll(
                                'input:not([type=hidden]):not([type=password]),' +
                                'textarea, select, button[type=submit], input[type=submit]'
                            );
                            els.forEach((el, idx) => {
                                const rect = el.getBoundingClientRect();
                                const absY = rect.top + window.scrollY;
                                if (rect.width <= 0 || rect.height <= 0) return;
                                let label = '';
                                const id = el.getAttribute('id');
                                if (id) {
                                    const lbl = document.querySelector('label[for="' + id + '"]');
                                    if (lbl) label = lbl.textContent.trim();
                                }
                                if (!label) {
                                    const prev = el.previousElementSibling;
                                    if (prev && prev.tagName === 'LABEL') label = prev.textContent.trim();
                                }
                                if (!label) {
                                    label = el.getAttribute('placeholder') ||
                                            el.getAttribute('aria-label') ||
                                            el.getAttribute('name') ||
                                            el.getAttribute('value') ||
                                            el.textContent.trim().slice(0, 40) ||
                                            el.tagName.toLowerCase();
                                }
                                let selector = null;
                                if (el.id) selector = '#' + el.id;
                                else if (el.name) selector = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                                let currentValue = '';
                                const tag = el.tagName.toLowerCase();
                                const elType = (el.getAttribute('type') || '').toLowerCase();
                                if (tag === 'select') {
                                    currentValue = el.value || '';
                                } else if (elType === 'checkbox' || elType === 'radio') {
                                    currentValue = el.checked ? 'true' : 'false';
                                } else if (tag === 'textarea' || tag === 'input') {
                                    currentValue = el.value || '';
                                }
                                results.push({
                                    idx: idx,
                                    tag: tag,
                                    type: el.getAttribute('type') || tag,
                                    label: label.slice(0, 120),
                                    name: el.getAttribute('name') || '',
                                    selector: selector,
                                    current_value: currentValue.slice(0, 500),
                                    x: Math.round(rect.left),
                                    y: Math.round(absY),
                                    w: Math.round(rect.width),
                                    h: Math.round(Math.max(rect.height, 20)),
                                });
                            });
                            return results.slice(0, 200);
                        }""")
                    except Exception as _fe:
                        print(f"[SITE_PREVIEW_FORM_ELEMENTS_ERROR] {_fe}", flush=True)

                print(f"[SITE_PREVIEW_DONE] url={current_url[:80]} fields={len(field_boxes)} form_elements={len(form_elements)} html_len={len(page_html)} login={_login_ok}", flush=True)
                return {
                    "status": "DONE",
                    "executed": True,
                    "screenshot_b64": screenshot_b64,
                    "page_html": page_html,
                    "current_url": current_url,
                    "title": title,
                    "field_boxes": field_boxes,
                    "form_elements": form_elements,
                    "login_used": _login_ok,
                    "viewport": {"width": VIEWPORT_W, "height": VIEWPORT_H},
                }

            finally:
                try:
                    if browser:
                        browser.close()
                except Exception:
                    pass

    except Exception as e:
        import traceback
        print(f"[SITE_PREVIEW_ERROR] {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "status": "FAILED",
            "executed": False,
            "message": f"スクリーンショット取得に失敗しました: {type(e).__name__}",
        }


def fill_and_submit_form(media_mapping: dict, target_url: str, field_values: dict) -> dict:
    """
    ログイン後に target_url へ遷移し、field_values (CSSセレクタ→値) でフォームを埋めて更新ボタンをクリックする。
    ID/PASS・Cookieはレスポンスに絶対含めない。
    """
    if not is_playwright_enabled():
        return {"status": "WAITING_EXECUTOR", "executed": False, "message": "PLAYWRIGHT_ENABLED=false"}

    secret_name = media_mapping.get("credential_secret_name")
    creds = None
    if secret_name:
        try:
            creds = get_secret_json(secret_name)
            if creds and creds.get("blocked"):
                creds = None
        except Exception:
            creds = None

    if not creds:
        return {"status": "FAILED", "executed": False, "message": "認証情報が取得できません"}

    try:
        import base64
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        VIEWPORT_W = 1280
        VIEWPORT_H = 800

        with sync_playwright() as p:
            browser = None
            try:
                auth = create_authenticated_page(p, media_mapping, creds)
                browser = auth["browser"]
                page = auth["page"]

                if target_url and target_url != page.url:
                    try:
                        page.goto(target_url, timeout=20000)
                        page.wait_for_load_state("domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1000)
                    except PlaywrightTimeout:
                        pass

                # フィールドを埋める
                field_errors = []
                for selector, value in (field_values or {}).items():
                    if not selector:
                        continue
                    try:
                        el = page.locator(selector).first
                        cnt = el.count()
                        if cnt == 0:
                            field_errors.append(f"not_found: {selector}")
                            continue
                        tag = el.evaluate("el => el.tagName.toLowerCase()")
                        el_type = (el.get_attribute("type") or "").lower()
                        if tag == "select":
                            try:
                                el.select_option(value=str(value))
                            except Exception:
                                el.select_option(label=str(value))
                        elif el_type in ("checkbox", "radio"):
                            if str(value).lower() in ("true", "1", "on", "checked"):
                                el.check()
                            else:
                                el.uncheck()
                        else:
                            el.fill(str(value))
                        print(f"[FORM_FILL] filled selector={selector} value_len={len(str(value))}", flush=True)
                    except Exception as _fe:
                        field_errors.append(f"{selector}: {type(_fe).__name__}")
                        print(f"[FORM_FILL_FIELD_ERROR] selector={selector} error={_fe}", flush=True)

                # 更新ボタンをクリック
                submit_clicked = False
                submit_candidates = [
                    "input[type='submit']",
                    "button[type='submit']",
                    "input[value='更新']",
                    "input[value='保存']",
                    "input[value='登録']",
                    "button:has-text('更新')",
                    "button:has-text('保存')",
                    "button:has-text('登録')",
                ]
                for sc in submit_candidates:
                    try:
                        btn = page.locator(sc).first
                        if btn.count() > 0:
                            btn.scroll_into_view_if_needed()
                            btn.click()
                            page.wait_for_load_state("domcontentloaded", timeout=30000)
                            page.wait_for_timeout(1000)
                            submit_clicked = True
                            print(f"[FORM_FILL_SUBMIT] clicked selector={sc}", flush=True)
                            break
                    except Exception:
                        continue

                # 結果スクリーンショット（フルページ）
                try:
                    screenshot_bytes = page.screenshot(type="png", full_page=True)
                except Exception:
                    screenshot_bytes = page.screenshot(type="png")
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                try:
                    VIEWPORT_H = page.evaluate("() => Math.max(document.body.scrollHeight, 800)")
                except Exception:
                    pass

                return {
                    "status": "DONE",
                    "executed": True,
                    "submit_clicked": submit_clicked,
                    "screenshot_b64": screenshot_b64,
                    "current_url": page.url,
                    "field_errors": field_errors,
                    "viewport": {"width": VIEWPORT_W, "height": VIEWPORT_H},
                    "message": "保存しました" if submit_clicked else "フィールドを埋めましたが更新ボタンが見つかりませんでした",
                }
            finally:
                if browser:
                    try:
                        browser.close()
                    except Exception:
                        pass

    except Exception as e:
        import traceback
        print(f"[FORM_FILL_ERROR] {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {"status": "FAILED", "executed": False, "message": f"フォーム送信に失敗: {type(e).__name__}"}


# ─────────────────────────────────────────────────────────────────────────────
# C検証: operation_url_verification
# deep_scan後に各operation_typeのtarget_urlを実ブラウザで確認。
# 正しいフォームページなら VERIFIED / URLが違えば自動追跡して修正 / ダメならNEEDS_REVIEW+SS
# ─────────────────────────────────────────────────────────────────────────────

def _dom_passes_op_check(raw_elements: list, operation_type: str) -> bool:
    """対象ページのDOM要素がoperation_typeの要件を満たすか簡易判定。"""
    has_submit    = any(
        (e.get("tag") == "button" and e.get("type") not in ("reset",))
        or (e.get("tag") == "input" and e.get("type") in ("submit", "button"))
        for e in raw_elements
    )
    has_textarea  = any(e.get("tag") == "textarea" for e in raw_elements)
    has_file      = any(e.get("tag") == "input" and e.get("type") == "file" for e in raw_elements)
    has_text_inp  = any(
        e.get("tag") == "input"
        and e.get("type") not in ("hidden", "submit", "button", "reset", "checkbox", "radio", "file", "password")
        for e in raw_elements
    )
    has_any_inp   = has_text_inp or has_textarea or has_file
    checks = {
        "news_post":       has_submit and has_textarea,
        "text_update":     has_submit and (has_textarea or has_text_inp),
        "media_replace":   has_submit and has_file,
        "schedule_update": has_submit and has_text_inp,
        "price_update":    has_submit and has_text_inp,
        "entity_register": has_submit and has_any_inp,
        "entity_update":   has_submit and has_any_inp,
        "status_update":   has_submit,
    }
    return bool(checks.get(operation_type, has_submit and has_any_inp))


def _upload_verification_screenshot(page, mapping_id: str, operation_type: str, suffix: str = "") -> str:
    """PlaywrightページのスクリーンショットをGCSにアップロードしてURLを返す。失敗時は空文字。"""
    try:
        import os as _os_ss, datetime as _dt_ss
        _bucket_name = _os_ss.environ.get("CENTRAL_BLOB_BUCKET", "my-consulting-ai-central-blob")
        _ss_bytes = page.screenshot(full_page=False, timeout=8000)
        from google.cloud import storage as _gcs_ss
        _gc_ss  = _gcs_ss.Client()
        _bkt_ss = _gc_ss.bucket(_bucket_name)
        _ts_ss  = _dt_ss.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _sfx    = f"_{suffix}" if suffix else ""
        _path   = f"agent_verification/{mapping_id}/{operation_type}{_sfx}_{_ts_ss}.png"
        _blob_ss = _bkt_ss.blob(_path)
        _blob_ss.upload_from_string(_ss_bytes, content_type="image/png")
        _url_ss = f"https://storage.googleapis.com/{_bucket_name}/{_path}"
        print(f"[VERIFY_SS_UPLOADED] op={operation_type} url={_url_ss}", flush=True)
        return _url_ss
    except Exception as _sse:
        print(f"[VERIFY_SS_ERROR] op={operation_type} err={type(_sse).__name__}", flush=True)
        return ""


def _follow_links_for_form(page, operation_type: str, visited: set, base_url: str,
                            max_depth: int = 2, current_depth: int = 0):
    """
    現在ページのリンクを辿ってoperation_typeのフォームを探す。
    見つかったら (url, raw_elements) を返す。見つからなければ (None, None)。
    """
    if current_depth >= max_depth:
        return None, None
    from urllib.parse import urljoin as _urljoin_v
    _FORM_KW = (
        "/edit", "/new", "/add", "/create", "/register", "/input", "/form",
        "edit", "regist", "entry", "new", "add", "create", "form",
        "?id=", "&id=",
        "cast", "girl", "staff", "talent", "profile",
        "編集", "登録", "追加", "新規", "入力",
    )
    _SKIP_KW = ("delete", "signout", "logout", "search", "csv", "download", "print", "back")
    try:
        _raw_cur = _raw_scan_page(page)
        _links = [e for e in _raw_cur if e.get("tag") == "a" and e.get("href")]
        candidates = []
        for _lk in _links[:80]:
            _href = _lk.get("href") or ""
            _text = (_href + " " + (_lk.get("text") or "") + " " + (_lk.get("aria_label") or "")).lower()
            if any(k in _text for k in _SKIP_KW):
                continue
            if not any(k in _text for k in _FORM_KW):
                continue
            _abs = _href if _href.startswith("http") else _urljoin_v(base_url, _href)
            if not _abs.startswith("http") or _abs in visited:
                continue
            candidates.append(_abs)
        for _cand_url in candidates[:5]:
            visited.add(_cand_url)
            try:
                page.goto(_cand_url, timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(1000)
                _raw2 = _raw_scan_page(page)
                if _dom_passes_op_check(_raw2, operation_type):
                    return _cand_url, _raw2
                # 再帰で1段深く
                _found_url, _found_raw = _follow_links_for_form(
                    page, operation_type, visited, _cand_url,
                    max_depth=max_depth, current_depth=current_depth + 1,
                )
                if _found_url:
                    return _found_url, _found_raw
            except Exception as _fe:
                print(f"[VERIFY_FOLLOW_ERROR] url={_cand_url[:60]} err={type(_fe).__name__}", flush=True)
    except Exception as _lfe:
        print(f"[VERIFY_LINK_EXTRACT_ERROR] err={type(_lfe).__name__}", flush=True)
    return None, None


def run_operation_url_verification(media_mapping: dict, operation_types: list = None) -> dict:
    """
    C検証: operation_mappingsの各target_urlを実ブラウザで確認。
    - フォームDOMあり → VERIFIED + スクリーンショット
    - フォームDOMなし → リンク自動追跡(最大2段) → 正しいURL発見時はURLを更新してVERIFIED
    - それでもなし   → NEEDS_REVIEW + スクリーンショット
    Firestoreのoperation_mappings.{op_type}.verification フィールドを更新。
    """
    import datetime as _dt_v, os as _os_v
    if not is_playwright_enabled():
        return {"status": "SKIPPED", "reason": "PLAYWRIGHT_DISABLED"}

    mapping_id = str(media_mapping.get("id") or media_mapping.get("mapping_id") or "")
    if not mapping_id:
        return {"status": "FAILED", "reason": "mapping_id_empty"}

    op_maps = media_mapping.get("operation_mappings") or {}
    if operation_types:
        op_maps = {k: v for k, v in op_maps.items() if k in operation_types}
    if not op_maps:
        return {"status": "SKIPPED", "reason": "no_operation_mappings"}

    try:
        from api.core.firestore_client import get_db as _get_db_v
        _db_v = _get_db_v()
    except Exception as _dbe:
        return {"status": "FAILED", "reason": f"db_error: {type(_dbe).__name__}"}

    results = {}

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as _PlaywrightTimeout
        with sync_playwright() as _p_v:
            _browser_v = _p_v.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            try:
                # ログイン済みセッションを復元
                _context_v = None
                try:
                    _cached = _load_cached_session(mapping_id)
                    if _cached:
                        _context_v = _browser_v.new_context(storage_state=_cached)
                        print(f"[VERIFY_SESSION_LOADED] mapping_id={mapping_id}", flush=True)
                except Exception as _se_v:
                    print(f"[VERIFY_SESSION_LOAD_ERROR] err={type(_se_v).__name__}", flush=True)
                if _context_v is None:
                    _context_v = _browser_v.new_context()

                _page_v = _context_v.new_page()

                for op_type, op_data in op_maps.items():
                    target_url = str(op_data.get("target_url") or op_data.get("url") or "")
                    if not target_url or not target_url.startswith("http"):
                        results[op_type] = {"status": "SKIPPED", "reason": "no_target_url"}
                        continue

                    print(f"[VERIFY_OP_START] op={op_type} url={target_url[:80]}", flush=True)
                    _now_v = _dt_v.datetime.utcnow()
                    _visited = {target_url}

                    try:
                        _page_v.goto(target_url, timeout=25000, wait_until="domcontentloaded")
                        _page_v.wait_for_timeout(1500)
                        try:
                            _page_v.wait_for_selector(
                                "input:not([type=hidden]):not([type=password]), textarea, select, form",
                                timeout=5000, state="attached",
                            )
                        except Exception:
                            pass
                        _raw_v = _raw_scan_page(_page_v)

                        if _dom_passes_op_check(_raw_v, op_type):
                            # ─ VERIFIED: 正しいURL ─
                            _ss_url = _upload_verification_screenshot(_page_v, mapping_id, op_type, "verified")
                            _vresult = {
                                "status":        "VERIFIED",
                                "verified_url":  target_url,
                                "original_url":  target_url,
                                "screenshot_url": _ss_url,
                                "verified_at":   _now_v.isoformat(),
                            }
                            print(f"[VERIFY_OP_VERIFIED] op={op_type} url={target_url[:60]}", flush=True)
                        else:
                            # ─ フォームなし: リンク追跡で修正を試みる ─
                            print(f"[VERIFY_OP_NO_FORM] op={op_type} url={target_url[:60]} → link follow", flush=True)
                            _found_url, _found_raw = _follow_links_for_form(
                                _page_v, op_type, _visited, target_url, max_depth=2,
                            )
                            if _found_url:
                                _ss_url = _upload_verification_screenshot(_page_v, mapping_id, op_type, "corrected")
                                _vresult = {
                                    "status":        "URL_CORRECTED",
                                    "verified_url":  _found_url,
                                    "original_url":  target_url,
                                    "screenshot_url": _ss_url,
                                    "verified_at":   _now_v.isoformat(),
                                }
                                # target_url を正しいURLに上書き
                                try:
                                    _db_v.collection("media_mappings").document(mapping_id).update({
                                        f"operation_mappings.{op_type}.target_url": _found_url,
                                        f"operation_mappings.{op_type}.target_url_corrected_from": target_url,
                                    })
                                except Exception as _uue:
                                    print(f"[VERIFY_URL_UPDATE_ERROR] op={op_type} err={type(_uue).__name__}", flush=True)
                                print(f"[VERIFY_OP_CORRECTED] op={op_type} {target_url[:50]}→{_found_url[:50]}", flush=True)
                            else:
                                # ─ NEEDS_REVIEW: 手動確認が必要 ─
                                # 元のURLに戻ってスクリーンショット
                                try:
                                    _page_v.goto(target_url, timeout=15000, wait_until="domcontentloaded")
                                    _page_v.wait_for_timeout(800)
                                except Exception:
                                    pass
                                _ss_url = _upload_verification_screenshot(_page_v, mapping_id, op_type, "needs_review")
                                _vresult = {
                                    "status":        "NEEDS_REVIEW",
                                    "verified_url":  target_url,
                                    "original_url":  target_url,
                                    "screenshot_url": _ss_url,
                                    "verified_at":   _now_v.isoformat(),
                                    "reason":        "フォームDOM未検出・リンク追跡でも発見できず",
                                }
                                print(f"[VERIFY_OP_NEEDS_REVIEW] op={op_type} url={target_url[:60]}", flush=True)

                        results[op_type] = _vresult
                        # Firestoreに検証結果を保存
                        try:
                            _db_v.collection("media_mappings").document(mapping_id).update({
                                f"operation_mappings.{op_type}.verification": _vresult,
                                f"operation_mappings.{op_type}.verified_at":  _now_v,
                            })
                        except Exception as _fse:
                            print(f"[VERIFY_FS_SAVE_ERROR] op={op_type} err={type(_fse).__name__}", flush=True)

                    except Exception as _op_e:
                        print(f"[VERIFY_OP_ERROR] op={op_type} err={type(_op_e).__name__}:{_op_e}", flush=True)
                        results[op_type] = {"status": "ERROR", "reason": type(_op_e).__name__}

            finally:
                try:
                    _browser_v.close()
                except Exception:
                    pass

    except ImportError:
        return {"status": "SKIPPED", "reason": "playwright_not_installed"}
    except Exception as _e_v:
        print(f"[VERIFY_GLOBAL_ERROR] err={type(_e_v).__name__}:{_e_v}", flush=True)
        return {"status": "FAILED", "reason": type(_e_v).__name__, "results": results}

    verified   = sum(1 for r in results.values() if r.get("status") == "VERIFIED")
    corrected  = sum(1 for r in results.values() if r.get("status") == "URL_CORRECTED")
    needs_rev  = sum(1 for r in results.values() if r.get("status") == "NEEDS_REVIEW")
    print(f"[VERIFY_SUMMARY] mapping_id={mapping_id} verified={verified} corrected={corrected} needs_review={needs_rev}", flush=True)
    return {
        "status":       "DONE",
        "verified":     verified,
        "corrected":    corrected,
        "needs_review": needs_rev,
        "results":      results,
    }


# ==============================================================
# auto_setup_mapping_ai
# 新設計: Playwright クロール → Gemini 解析 → AI_CONFIRMED で保存
# 旧 deep_scan_operation との違い:
#   confirmation_status:"AI_CONFIRMED" / production_ready:True / source:"AI_CONFIRMED"
#   の3フラグを必ず付与してFirestoreに書き込む
#   → _operation_mapping_is_production_ready() が True を返すようになる
#   → 7ツール全て連動可能になる
# ==============================================================

def auto_setup_mapping_ai(media_mapping: dict, db=None) -> dict:
    """
    1. run_dom_scan で Playwright クロール + Gemini 解析（既存処理を流用）
    2. build_operation_mappings_from_dom_evidence でセレクタ抽出（既存処理を流用）
    3. 結果に AI_CONFIRMED フラグを付与して Firestore に保存
    4. media_html_cache に共有キャッシュとして保存

    戻り値:
      ok:         bool
      ready_ops:  解析成功した operation_type リスト
      failed_ops: 解析失敗した operation_type リスト
      status:     "DONE" / "WAITING_EXECUTOR" / "FAILED" / "NO_PAGES"
      cache_saved: bool
    """
    import datetime as _dt_as

    mapping_id = str(media_mapping.get("mapping_id") or media_mapping.get("id") or "")
    if not mapping_id:
        return {"ok": False, "status": "FAILED", "reason": "mapping_id_missing"}

    if not is_playwright_enabled():
        return {"ok": False, "status": "WAITING_EXECUTOR", "reason": "PLAYWRIGHT_ENABLED=false"}

    if db is None:
        from api.core.firestore_client import get_db as _get_db_as
        db = _get_db_as()

    print(f"[AUTO_SETUP_AI] START mapping_id={mapping_id}", flush=True)

    # ── Step1: Playwright クロール + Gemini 解析 ──────────────────────────
    # run_dom_scan が navigation_graph.pages に form_schema.gemini_* を保存する
    scan_result = run_dom_scan(
        media_mapping,
        max_pages=60,
        reset_resume=True,
    )
    scan_status = scan_result.get("status", "")
    if scan_status in ("WAITING_EXECUTOR", "BLOCKED"):
        return {"ok": False, "status": scan_status, "reason": scan_result.get("message", "")}

    print(f"[AUTO_SETUP_AI] run_dom_scan done status={scan_status}", flush=True)

    # ── Step2: Firestore から Gemini 解析済みページを取得 ─────────────────
    try:
        snap = db.collection("media_mappings").document(mapping_id).get()
        doc = snap.to_dict() or {} if snap.exists else {}
        pages = (doc.get("navigation_graph") or {}).get("pages") or []
    except Exception as _e_load:
        print(f"[AUTO_SETUP_AI] pages_load_error {type(_e_load).__name__}", flush=True)
        pages = []

    if not pages:
        return {"ok": False, "status": "NO_PAGES", "reason": "navigation_graph.pages が空です。ログインURLと認証情報を確認してください。"}

    # ── Step3: セレクタ抽出（既存関数を流用） ────────────────────────────
    try:
        all_op_mappings = build_operation_mappings_from_dom_evidence(mapping_id, pages)
    except Exception as _e_build:
        print(f"[AUTO_SETUP_AI] build_error {type(_e_build).__name__}:{_e_build}", flush=True)
        return {"ok": False, "status": "FAILED", "reason": f"build_error: {type(_e_build).__name__}"}

    if not all_op_mappings:
        return {"ok": False, "status": "NO_PAGES", "reason": "operation_mappings を構築できませんでした。"}

    # ── Step4: AI_CONFIRMED フラグを付与して Firestore に保存 ─────────────
    now = _dt_as.datetime.utcnow()
    ready_ops   = []
    failed_ops  = []
    # capability_view をリセット: 古いデータが残っていると task 作成時に cap_op チェックでブロックされる
    fs_updates: dict = {"updated_at": now, "capability_view": {}}

    for op_type, op_data in all_op_mappings.items():
        if not isinstance(op_data, dict):
            continue

        selectors = op_data.get("selectors") or {}
        has_save  = "save" in selectors

        if op_data.get("status") in ("READY", "NEEDS_REVIEW") and (selectors or op_data.get("target_url")):
            confirmed_data = {
                **op_data,
                "status":              "READY" if has_save else "NEEDS_REVIEW",
                "executable":          has_save,
                "production_ready":    has_save,          # ← 旧設計で欠けていた
                "confirmation_status": "AI_CONFIRMED",    # ← 旧設計で欠けていた
                "source":              "AI_CONFIRMED",    # ← 旧設計で欠けていた
                "auto_setup_at":       now.isoformat(),
            }
            fs_updates[f"operation_mappings.{op_type}"] = confirmed_data

            # operation_steps_by_type も更新（stepsがあれば）
            steps = op_data.get("steps") or []
            if steps:
                fs_updates[f"operation_steps_by_type.{op_type}"] = steps

            if has_save:
                ready_ops.append(op_type)
            else:
                failed_ops.append(op_type)
        else:
            failed_ops.append(op_type)

    try:
        db.collection("media_mappings").document(mapping_id).update(fs_updates)
        print(f"[AUTO_SETUP_AI] saved ready_ops={ready_ops} failed_ops={failed_ops}", flush=True)
    except Exception as _e_save:
        print(f"[AUTO_SETUP_AI] save_error {type(_e_save).__name__}:{_e_save}", flush=True)
        return {"ok": False, "status": "FAILED", "reason": f"save_error: {type(_e_save).__name__}"}

    # ── Step5: capabilities を更新 ────────────────────────────────────────
    _CAP_MAP = {
        "news_post":       "can_post_news",
        "blog_post":       "can_post_news",
        "text_update":     "can_update_text",
        "media_replace":   "can_upload_image",
        "schedule_update": "can_update_schedule",
        "price_update":    "can_update_price",
        "entity_register": "can_register_entity",
        "entity_update":   "can_update_entity",
    }
    cap_updates: dict = {}
    for op in ready_ops:
        cap_key = _CAP_MAP.get(op)
        if cap_key:
            cap_updates[f"capabilities.{cap_key}"] = True
    if cap_updates:
        try:
            db.collection("media_mappings").document(mapping_id).update(cap_updates)
        except Exception:
            pass

    # ── Step6: 共有キャッシュに保存 ───────────────────────────────────────
    cache_saved = False
    media_url = str(media_mapping.get("media_url") or media_mapping.get("login_url") or "")
    if media_url and ready_ops:
        try:
            import hashlib as _hs_as
            _url_hash = _hs_as.sha256(media_url.lower().rstrip("/").encode()).hexdigest()[:32]
            db.collection("media_html_cache").document(_url_hash).set({
                "url_hash":        _url_hash,
                "url":             media_url,
                "ready_ops":       ready_ops,
                "operation_mappings": {
                    op: all_op_mappings[op]
                    for op in ready_ops
                    if op in all_op_mappings
                },
                "analyzed_at":     now,
                "model":           "gemini",
                "source":          "auto_setup_mapping_ai",
            }, merge=True)
            cache_saved = True
            print(f"[AUTO_SETUP_AI] cache_saved url_hash={_url_hash[:8]}", flush=True)
        except Exception as _e_cache:
            print(f"[AUTO_SETUP_AI] cache_save_error {type(_e_cache).__name__}", flush=True)

    return {
        "ok":          len(ready_ops) > 0,
        "status":      "DONE",
        "ready_ops":   ready_ops,
        "failed_ops":  failed_ops,
        "cache_saved": cache_saved,
        "pages_scanned": len(pages),
    }
