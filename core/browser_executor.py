# api/core/browser_executor.py
# ブラウザ操作実行層
# Playwrightはトップレベルimportしない（Cloud Run未導入時の起動クラッシュ防止）
# ID/PASSはログ・例外メッセージに絶対出さない

import os
from typing import Optional
from api.core.secret_manager import get_secret_json
import time as _time

# ── P16.7: Selector Learning Cache (TTL 60秒) ────────────────────────────
# Firestore selector_learning_stats を毎回直読みしない
# { cache_key: {"data": dict, "expires_at": float} }
_SELECTOR_LEARNING_CACHE: dict = {}
_SELECTOR_LEARNING_CACHE_TTL = 60  # seconds

def _get_selector_learning_cache(key: str):
    entry = _SELECTOR_LEARNING_CACHE.get(key)
    if entry and _time.time() < entry["expires_at"]:
        return entry["data"]
    return None

def _set_selector_learning_cache(key: str, data: dict):
    _SELECTOR_LEARNING_CACHE[key] = {
        "data":       data,
        "expires_at": _time.time() + _SELECTOR_LEARNING_CACHE_TTL,
    }

def _invalidate_selector_learning_cache(key: str):
    _SELECTOR_LEARNING_CACHE.pop(key, None)
# ─────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# P25 Session Management
# ログイン済みCookieを mapping_id 単位でメモリにキャッシュ
# ID/PASS・Cookie中身は絶対にログ出力しない（件数のみ）
# ══════════════════════════════════════════════════════════════════════════════
import datetime as _dt_p25

_session_cache: dict = {}
# 構造: { mapping_id: { "cookies": [], "logged_in_at": datetime, "expires_at": datetime, "current_url": "", "title": "" } }
_SESSION_TTL_MINUTES = 30


def _get_cached_session(mapping_id: str) -> dict | None:
    """メモリ優先、次にFirestoreからセッションを取得する。"""
    import datetime as _dt_check
    # ① メモリキャッシュ確認
    entry = _session_cache.get(mapping_id)
    if entry:
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
            _expires_at = _data.get("expires_at")
            _cookies = _data.get("cookies") or []
            print(f"[P25_SESSION_FIRESTORE_READ] mapping_id={mapping_id} exists=True cookies_count={len(_cookies)} expires_at={_expires_at}", flush=True)
            # expires_at timezone統一
            now_naive = _dt_check.datetime.utcnow()
            now_aware = _dt_check.datetime.now(_dt_check.timezone.utc)
            if _expires_at is None:
                expired = True
            else:
                # Firestore Timestampはdatetimeに変換済みのことが多いが念のため
                try:
                    _exp_dt = _expires_at.ToDatetime() if hasattr(_expires_at, "ToDatetime") else _expires_at
                except Exception:
                    _exp_dt = _expires_at
                if hasattr(_exp_dt, "tzinfo") and _exp_dt.tzinfo is not None:
                    expired = now_aware >= _exp_dt
                else:
                    expired = now_naive >= _exp_dt
            if _cookies and not expired:
                _session_cache[mapping_id] = _data
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

def _save_cached_session(mapping_id: str, context, page) -> None:
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
        for _ck in cookies:
            print(f"[P25_COOKIE_DETAIL] name={_ck.get('name')} domain={_ck.get('domain')} path={_ck.get('path')} expires={_ck.get('expires')}", flush=True)
        print(f"[P25_STORAGE_STATE_SAVE] mapping_id={mapping_id} cookies={len(cookies)} origins={len(_origins)}", flush=True)
        import datetime as _dt_p25s
        now = _dt_p25s.datetime.utcnow()
        expires_at = now + _dt_p25s.timedelta(minutes=_SESSION_TTL_MINUTES)
        entry = {
            "mapping_id":    mapping_id,
            "cookies":       cookies,
            "storage_state": _storage,
            "logged_in_at":  now,
            "expires_at":    expires_at,
            "current_url":   page.url if page else "",
            "title":         page.title() if page else "",
            "updated_at":    now,
        }
        _session_cache[mapping_id] = entry
        try:
            from api.core.firestore_client import get_db as _get_db_p25s
            _db_p25s = _get_db_p25s()
            _db_p25s.collection("agent_sessions").document(mapping_id).set(entry)
        except Exception as _e_fs:
            print(f"[P25_SESSION_SAVE_FS_ERROR] mapping_id={mapping_id} error={_e_fs}", flush=True)
        print(f"[P25_SESSION_SAVE] mapping_id={mapping_id} store=memory+firestore cookies_count={len(cookies)} expires_at={expires_at.isoformat()}", flush=True)
    except Exception as _e_save:
        print(f"[P25_SESSION_SAVE_ERROR] mapping_id={mapping_id} error={_e_save}", flush=True)
def _clear_cached_session(mapping_id: str, reason: str = "unknown") -> None:
    """メモリ+Firestoreからセッションを削除する。"""
    _session_cache.pop(mapping_id, None)
    try:
        from api.core.firestore_client import get_db as _get_db_p25c
        _get_db_p25c().collection("agent_sessions").document(mapping_id).delete()
    except Exception as _e_clr:
        print(f"[P25_SESSION_CLEAR_FS_ERROR] mapping_id={mapping_id} error={_e_clr}", flush=True)
    print(f"[P25_SESSION_CLEAR] mapping_id={mapping_id} reason={reason}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
def is_playwright_enabled() -> bool:
    return os.environ.get("PLAYWRIGHT_ENABLED", "false").lower() == "true"


def run_browser_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
    operation_steps: list = None,
) -> dict:
    """
    browser_executorの入口。auth_typeに応じて処理を振り分ける。
    execute_task / agent_executorから呼ばれる。
    """
    auth_type = media_mapping.get("auth_type", "login_form")

    if auth_type == "login_form":
        return _run_login_form_operation(media_mapping, operation_type, payload, operation_steps=operation_steps)

    if auth_type == "api_key":
        return _run_api_key_operation(media_mapping, operation_type, payload)

    if auth_type == "manual":
        return {
            "status":   "WAITING_EXECUTOR",
            "executed": False,
            "message":  "auth_type=manualは手動操作が必要です。自動実行できません。",
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

    return _run_login_form_with_operation(media_mapping, creds, operation_type, payload, operation_steps=operation_steps)



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
                    if (tag === 'input' && (t === 'submit' || t === 'button' || t === 'reset')) {
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
            if tag == "input" and typ == "password":
                if eid.lower() in ("pass", "password", "passwd"):
                    sel = f"#{eid}"
                elif name.lower() in ("pass", "password", "passwd"):
                    sel = f'input[name="{name}"]'
                else:
                    sel = 'input[type="password"]'
                try:
                    loc = ctx.locator(sel).first
                    loc.wait_for(timeout=2000)
                    print(f"[LOGIN_RAW_INFER] password found: {sel} (frame={furl})", flush=True)
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


def run_login_form_check(media_mapping: dict, creds: dict) -> dict:
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
                print(f"[LOGIN_AFTER_SUBMIT] before_url={before_url} after_url={after_url} url_changed={url_changed} verify_sel={verify_sel} title={after_title}", flush=True)
                # ログイン成功判定
                _is_login_url = any(k in after_url for k in ["login", "signin", "Login", "SignIn", "auth", "Auth"])
                if url_changed and not _is_login_url:
                    login_success = True
                elif verify_sel:
                    try:
                        page.wait_for_selector(verify_sel, timeout=3000)
                        login_success = True
                    except Exception:
                        login_success = False
                print(f"[LOGIN_SUCCESS_JUDGE] url_changed={url_changed} is_login_url={_is_login_url} verify_sel={verify_sel} login_success={login_success}", flush=True)

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
                pass  # browser.close() はP21 crawl完了後に実行する
            if _error_result:
                if browser:
                    try:
                        browser.close()
                    except Exception:
                        pass
                return _error_result

            # ── selector保存（with内・browser生存中）──────────────────
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

            # ── P21: ログイン後管理画面クローラー実行（with内・browser生存中）──
            _crawl_result = {}
            if login_success:
                try:
                    _crawl_mapping_id = (media_mapping.get("mapping_id") or media_mapping.get("id")) if media_mapping else None
                    if _crawl_mapping_id:
                        # crawl開始前ページ状態ログ
                        try:
                            _pre_url    = page.url
                            _pre_title  = page.title()
                            _pre_frames = [f.url for f in page.frames]
                            _browser_alive  = browser is not None
                            print(f"[P21_BROWSER_STATE] browser_alive={_browser_alive} page_url={_pre_url} title={_pre_title} frame_count={len(_pre_frames)} frames={_pre_frames}", flush=True)
                        except Exception as _pre_err:
                            print(f"[P21_BROWSER_STATE] read error: {_pre_err}", flush=True)
                        # submit後の遷移完了を待つ
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            pass
                        from api.core.firestore_client import get_db as _get_db_crawl
                        _crawl_result = post_login_admin_crawl(page, _crawl_mapping_id, _get_db_crawl(), max_pages=max_pages)
                except Exception as _crawl_err:
                    print(f"[P21] crawl error: {type(_crawl_err).__name__}: {_crawl_err}", flush=True)
                    _crawl_result = {"status": "ERROR", "error": type(_crawl_err).__name__}

            # ── browser close（P21完了後）────────────────────────────
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
                    "operation_candidates_count": len(_crawl_result.get("operation_candidates", [])),
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
    login_url  = media_mapping.get("login_url")
    verify_sel = media_mapping.get("verify_selector")

    if not login_url:
        raise RuntimeError("login_url が未設定です")

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
    print(f"[AUTH_FINAL_STATE] url={_lp_after_url} pw_count={_lp_pw_count} verify_ok={_lp_verify_ok} url_changed={_lp_url_changed} on_login_url={_lp_on_login_url}", flush=True)
    if _lp_on_login_url and not _lp_verify_ok:
        browser.close()
        raise RuntimeError(f"login failed: still on login page url={_lp_after_url}")
    if not _lp_verify_ok and not (_lp_url_changed and not _lp_url_is_login) and not (_lp_pw_gone and _lp_url_changed):
        browser.close()
        raise RuntimeError("login failed: url did not change or still on login page")
    return browser, page


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
    "status_update": {
        "capability_key": "can_update_text",
        "fields": [
            {"selector_key": "body", "payload_key": "status", "input_type": "text"},
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
        "required_selector_keys": ["date_input", "save"],
    },
    "price_update": {
        "capability_key": "can_update_price",
        "fields": [
            {"selector_key": "price_input", "payload_key": "price_value", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["price_input", "save"],
    },
    "entity_register": {
        "capability_key": "can_register_entity",
        "fields": [
            {"selector_key": "required_inputs", "payload_key": "name", "input_type": "text"},
        ],
        "submit_selector_key": "save",
        "required_selector_keys": ["save"],
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
    返却: {"browser": browser, "context": None, "page": page}
    失敗時は RuntimeError を raise。
    """
    browser, page = _login_and_get_page(p, media_mapping, creds)
    return {"browser": browser, "context": None, "page": page}


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
        except Exception:
            before_values[key] = {"value": None, "input_type": input_type, "rollbackable": False}

    return before_values


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
) -> list:
    """
    P14: multi-step operation runner。
    operation_stepsを順番に実行し、step_results(list)を返す。
    required=TrueのstepでRuntimeErrorが発生した場合は即停止。
    media_mappings本体は変更しない。Secretは出力しない。
    """
    import time as _time_mod
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    dom = media_mapping.get("dom_selectors", {})
    urls = media_mapping.get("urls", {})
    step_results = []

    sorted_steps = sorted(operation_steps, key=lambda s: s.get("order", 99))

    for step in sorted_steps:
        step_id   = step.get("step_id", "")
        step_type = step.get("step_type", "")
        required  = step.get("required", True)
        started   = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
        error_msg = ""
        success   = False

        try:
            if step_type == "navigate":
                url_key = step.get("url_key", "")
                url = urls.get(url_key) or payload.get(url_key, "")
                if not url:
                    raise RuntimeError(f"navigate: url_key '{url_key}' が未設定です")
                page.goto(url, timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=35000)
                success = True

            elif step_type == "wait_for_selector":
                sel = dom.get(step.get("selector_key", ""), "")
                if not sel:
                    raise RuntimeError(f"wait_for_selector: selector未設定")
                page.wait_for_selector(sel, timeout=step.get("timeout", 5000))
                success = True

            elif step_type == "fill":
                sel   = dom.get(step.get("selector_key", ""), "")
                value = payload.get(step.get("payload_key", ""), "")
                if not sel:
                    raise RuntimeError(f"fill: selector未設定 ({step.get('selector_key')})")
                page.wait_for_selector(sel, timeout=5000)
                page.fill(sel, str(value))
                success = True

            elif step_type == "click":
                sel = dom.get(step.get("selector_key", ""), "")
                if not sel:
                    raise RuntimeError(f"click: selector未設定 ({step.get('selector_key')})")
                page.wait_for_selector(sel, timeout=5000)
                page.click(sel)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=35000)
                except Exception:
                    pass
                success = True

            elif step_type == "select":
                sel   = dom.get(step.get("selector_key", ""), "")
                value = payload.get(step.get("payload_key", ""), "")
                if not sel:
                    raise RuntimeError(f"select: selector未設定")
                page.wait_for_selector(sel, timeout=5000)
                page.select_option(sel, str(value))
                success = True

            elif step_type == "upload_file":
                sel   = dom.get(step.get("selector_key", ""), "")
                value = payload.get(step.get("payload_key", ""), "")
                if not sel:
                    raise RuntimeError(f"upload_file: selector未設定")
                page.set_input_files(sel, str(value))
                success = True

            elif step_type == "search":
                sel_input  = dom.get(step.get("selector_key", ""), "")
                sel_submit = dom.get(step.get("submit_selector_key", ""), "")
                value      = payload.get(step.get("payload_key", ""), "")
                if not sel_input:
                    raise RuntimeError(f"search: input selector未設定")
                page.wait_for_selector(sel_input, timeout=5000)
                page.fill(sel_input, str(value))
                if sel_submit:
                    page.click(sel_submit)
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=35000)
                    except Exception:
                        pass
                success = True

            elif step_type == "verify":
                import hashlib
                v = _verify_operation_detail(page, media_mapping, before_hash="", after_html=None)
                if not v.get("verified"):
                    raise RuntimeError(f"verify: 検証失敗 method={v.get('method')}")
                success = True

            elif step_type == "sleep":
                _time_mod.sleep(step.get("duration", 1))
                success = True

            elif step_type == "login":
                # loginはrun_login_form_with_operation内で済み。stepとして記録のみ。
                success = True

            else:
                raise RuntimeError(f"step_type '{step_type}' は未対応です")

        except (RuntimeError, PlaywrightTimeout, Exception) as e:
            error_msg = str(e)
            success   = False
            ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
            step_results.append({
                "step_id":    step_id,
                "step_type":  step_type,
                "status":     "FAILED",
                "started_at": started,
                "ended_at":   ended,
                "error":      error_msg,
            })
            if required:
                raise RuntimeError(f"[step:{step_id}] {error_msg}")
            else:
                continue

        ended = _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime())
        step_results.append({
            "step_id":    step_id,
            "step_type":  step_type,
            "status":     "DONE",
            "started_at": started,
            "ended_at":   ended,
            "error":      "",
        })

    return step_results

def _execute_operation(page, media_mapping: dict, operation_type: str, payload: dict, operation_steps: list = None) -> list:
    """
    ログイン済みpageに対してoperation_typeに応じた更新操作を実行する。
    operation_stepsがあればP14 multi-step runnerに委譲しstep_resultsを返す。
    なければGENERIC_OPERATION_CONFIGを参照してgeneric runnerで実行しNoneを返す。
    operation別if分岐なし。selector詳細・payload本文はログに出さない。
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    # P14: operation_stepsがある場合はmulti-step runnerに委譲
    if operation_steps:
        return _execute_operation_steps(page, media_mapping, operation_steps, payload)

    config = GENERIC_OPERATION_CONFIG.get(operation_type)
    if config is None:
        raise RuntimeError(f"operation_type '{operation_type}' は未実装です")

    dom = media_mapping.get("dom_selectors", {})

    # --- required selector チェック ---
    for key in config.get("required_selector_keys", []):
        if not dom.get(key):
            raise RuntimeError(f"{operation_type}: selector不足 ({key})")

    submit_key = config.get("submit_selector_key", "submit")
    submit_sel = dom.get(submit_key)

    # --- 各フィールドへの入力 ---
    for field in config.get("fields", []):
        sel         = dom.get(field["selector_key"])
        value       = payload.get(field["payload_key"], "")
        input_type  = field.get("input_type", "text")

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


def _verify_operation(page, media_mapping: dict) -> bool:
    """後方互換用。_verify_operation_detail のラッパー。"""
    result = _verify_operation_detail(page, media_mapping, before_hash="", after_html=None)
    return result.get("verified", False)


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
    }
    try:
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

        # 再execute（1回のみ）
        try:
            _execute_operation(page, temp_mapping, operation_type, payload, operation_steps=operation_steps)
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

                # P7-rollback: before_values取得
                try:
                    before_values = _capture_before_values(page, media_mapping, operation_type)
                except Exception:
                    before_values = {}

                try:
                    _step_results_buf = _execute_operation(page, media_mapping, operation_type, payload, operation_steps=operation_steps) or []
                except RuntimeError as oe:
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
                        before_values, before_hash, operation_steps=operation_steps
                    )
                    # P19: similar_repairsをself_heal結果に付与（参照用）
                    if _p19_similar_repairs:
                        _self_heal["similar_failure_repairs"] = _p19_similar_repairs
                    if _self_heal.get("retry_succeeded"):
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
                        }
                    # self-heal失敗 → rollback
                    try:
                        rollback = _rollback_fields(page, media_mapping, operation_type, before_values)
                    except Exception:
                        rollback = {
                            "attempted": False, "success": False,
                            "restored_fields": [], "failed_fields": [],
                            "reason": "rollback中に予期しないエラー",
                        }
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
                    }

                # P6: after_html取得 + 検証
                try:
                    after_html = page.content()
                except Exception:
                    after_html = ""

                verification = _verify_operation_detail(page, media_mapping, before_hash, after_html)

                if verification.get("verified"):
                    _done = {
                        "status":         "DONE",
                        "executed":       True,
                        "login_success":  True,
                        "operation_type": operation_type,
                        "verification":   verification,
                        "rollback":       None,
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
                        before_values, before_hash, operation_steps=operation_steps
                    )
                    if _p19_similar_repairs2:
                        _self_heal["similar_failure_repairs"] = _p19_similar_repairs2
                    if _self_heal.get("retry_succeeded"):
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
                        }
                    # self-heal失敗 → rollback
                    try:
                        rollback = _rollback_fields(page, media_mapping, operation_type, before_values)
                    except Exception:
                        rollback = {
                            "attempted": False, "success": False,
                            "restored_fields": [], "failed_fields": [],
                            "reason": "rollback中に予期しないエラー",
                        }
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
        "status":         "WAITING_EXECUTOR",
        "executed":       False,
        "auth_type":      "api_key",
        "operation_type": operation_type,
        "message":        "API連携実行層は現在開発中です。",
    }


def _run_no_auth_operation(
    media_mapping: dict,
    operation_type: str,
    payload: dict,
) -> dict:
    """認証不要型媒体への操作。将来実装。"""
    return {
        "status":         "WAITING_EXECUTOR",
        "executed":       False,
        "auth_type":      "none",
        "operation_type": operation_type,
        "message":        "認証不要型ブラウザ実行層は現在開発中です。",
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
                            browser.close()
                        except Exception:
                            pass
                        browser = None
                    except RuntimeError as _le:
                        print(f"[P5_LOGIN_FAILED] reason={_le}", flush=True)

                # ログイン済みpageがあればそれを使う
                if _login_success_dom and _login_page:
                    page = _login_page

                # 対象URLへ遷移（ログイン成功時はスキップ・ログイン後URLから継続）
                if not _login_success_dom:
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
                    "operation_candidates_count": len(_p21_result.get("operation_candidates", [])),
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
    selector_hash = hashlib.md5(selector.encode()).hexdigest()[:12]
    doc_id = f"{media_name}__{operation_type}__{selector_hash}"
    ref = db.collection("selector_learning_stats").document(doc_id)
    try:
        doc = ref.get()
        now = _dt.datetime.utcnow()
        if doc.exists:
            d = doc.to_dict()
            success_count  = d.get("success_count", 0)  + (1 if success else 0)
            failure_count  = d.get("failure_count", 0)  + (0 if success else 1)
            timeout_count  = d.get("timeout_count", 0)  + (1 if timeout else 0)
            verify_count   = d.get("verify_success_count", 0) + (1 if verify_success else 0)
            usage_count    = d.get("usage_count", 0) + 1
            # moving average latency
            prev_avg = d.get("avg_latency_ms", 0.0)
            avg_latency = prev_avg + (latency_ms - prev_avg) / usage_count
            total = success_count + failure_count or 1
            stability = round(success_count / total, 4)
            ref.update({
                "success_count":         success_count,
                "failure_count":         failure_count,
                "timeout_count":         timeout_count,
                "verify_success_count":  verify_count,
                "usage_count":           usage_count,
                "avg_latency_ms":        round(avg_latency, 2),
                "stability_score":       stability,
                "semantic_match_score":  semantic_match_score,
                "last_success_at":       now if success else d.get("last_success_at"),
                "last_failure_at":       now if not success else d.get("last_failure_at"),
                "last_seen_at":          now,
            })
        else:
            ref.set({
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
                "repair_generated_count":0,
                "repair_applied_count":  0,
                "usage_count":           1,
                "score":                 0.0,
            })
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

        ref = db.collection("cross_media_templates").document(template_id)
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
        print(f"[cross_media_templates] 保存エラー: {type(e).__name__}", flush=True)


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
        docs = (
            db.collection("cross_media_templates")
            .where("operation_type", "==", operation_type)
            .stream()
        )
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
) -> dict:
    """
    P20: browser_executor実行前にapproval_stateを確認。
    未承認ならBLOCKED。
    返り値: {"approved": bool, "approval_state": str, "paused": bool, "cancelled": bool}
    """
    try:
        doc = db.collection("workflow_execution_sessions").document(session_id).get()
        if not doc.exists:
            return {"approved": True, "approval_state": "NOT_FOUND", "paused": False, "cancelled": False}
        d = doc.to_dict()
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
        return {"approved": True, "approval_state": "ERROR", "paused": False, "cancelled": False}


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


def rebuild_operation_steps(operation_candidates: list, nav_graph: dict, op_mappings: dict, detail_by_op: dict) -> dict:
    """
    P23完了後にoperation_steps_by_typeを再生成する。
    detail_by_opのsource_urlをtarget_urlに優先使用。
    """
    REQUIRED_SELECTORS = {
        "news_post":       ["title", "body", "save"],
        "text_update":     ["body", "save"],
        "media_replace":   ["file", "save"],
        "schedule_update": ["date_input", "save"],
        "price_update":    ["price_input", "save"],
        "entity_register": ["save"],
        "entity_update":   ["save"],
    }
    NAV_KEYWORDS = {
        "news_post":       ["news", "post", "blog", "diary", "topic", "event", "coupon", "お知らせ", "ニュース", "投稿", "新規", "写メ"],
        "text_update":     ["profile", "about", "text", "プロフィール", "自己紹介", "説明"],
        "media_replace":   ["photo", "image", "gallery", "写真", "画像", "メディア"],
        "schedule_update": ["schedule", "shift", "出勤", "予定", "calendar"],
        "price_update":    ["price", "course", "料金", "コース", "fee"],
        "entity_register": ["register", "new", "add", "登録", "新規追加"],
        "entity_update":   ["edit", "update", "編集", "更新"],
    }
    STEP_TEMPLATES = {
        "news_post":       ["login", "navigate_to_news",     "input_title", "input_body",       "save", "verify_post"],
        "text_update":     ["login", "navigate_to_edit",     "input_text",                      "save", "verify_update"],
        "media_replace":   ["login", "navigate_to_media",    "upload_file",                     "save", "verify_upload"],
        "schedule_update": ["login", "navigate_to_schedule", "input_schedule",                  "save", "verify_schedule"],
        "price_update":    ["login", "navigate_to_price",    "input_price",                     "save", "verify_price"],
        "entity_register": ["login", "navigate_to_register", "input_entity_fields",             "save", "verify_register"],
        "entity_update":   ["login", "navigate_to_entity",   "edit_entity_fields",              "save", "verify_update"],
    }
    result = {}
    for op_type in operation_candidates:
        # OP_STEPS_SKIPPED: executable=Falseの候補はステップ生成しない
        _op_map_check = op_mappings.get(op_type, {})
        _executable_check = _op_map_check.get("executable", None)
        _status_check = _op_map_check.get("status", "")
        if _executable_check is False or (_executable_check is None and _status_check not in ("READY", "PARTIAL")):
            print(f"[OP_STEPS_SKIPPED_INCOMPLETE] op_type={op_type} executable={_executable_check} status={_status_check}", flush=True)
            continue
        print(f"[OP_STEPS_GENERATED] op_type={op_type} executable={_executable_check} status={_status_check}", flush=True)
        template  = STEP_TEMPLATES.get(op_type, ["login", "navigate", "input", "save", "verify"])
        keywords  = NAV_KEYWORDS.get(op_type, [])
        op_map    = op_mappings.get(op_type, {})
        req_sels  = REQUIRED_SELECTORS.get(op_type, ["save"])
        avail_sel = op_map.get("selectors", {})
        missing   = [k for k in req_sels if not avail_sel.get(k)]
        # ── P24.5 target_url優先順位 ──────────────────────────────────────
        # ① media_structure_map.operation_entrypoints[op].url  ← P21.5追加
        # ② operation_candidates_detail[op].source_url
        # ③ operation_mappings[op].p24_source_url
        # ④ operation_mappings[op].target_url
        # ⑤ keyword match (C1Main.php は他候補がある場合スキップ)
        # ⑥ None
        # media_structure_map entrypoint取得
        _struct_map_ep = {}
        try:
            _struct_map_ep = (nav_graph.get("__meta__", {}).get("operation_entrypoints") or {})
        except Exception:
            pass
        _struct_ep_url = (_struct_map_ep.get(op_type) or {}).get("url", "")
        if _struct_ep_url:
            print(f"[P24_STRUCTURE_ENTRYPOINT] op={op_type} url={_struct_ep_url} confidence={(_struct_map_ep.get(op_type) or {}).get('confidence',0)} evidence={(_struct_map_ep.get(op_type) or {}).get('evidence',[])}", flush=True)
        else:
            print(f"[P24_STRUCTURE_MISSING] op={op_type} reason=no_structure_entrypoint", flush=True)
        from urllib.parse import urlparse as _urlparse_p245, urljoin as _urljoin_p245
        _detail_url   = (detail_by_op or {}).get(op_type, {}).get("source_url", "")
        _p24_src_url  = op_map.get("p24_source_url", "")
        _map_url      = op_map.get("target_url", "")
        _media_base   = nav_graph and next((v.get("url","") for v in nav_graph.values() if isinstance(v,dict) and v.get("url","").startswith("http")), "")
        def _abs_url(u, base):
            if not u:
                return u
            if u.startswith("http"):
                return u
            if base:
                return _urljoin_p245(base, u)
            return u
        _detail_url  = _abs_url(_detail_url, _media_base)
        _p24_src_url = _abs_url(_p24_src_url, _media_base)
        _map_url     = _abs_url(_map_url, _media_base)
        def _is_c1main(u):
            return u and "C1Main.php" in u
        _kw_url = None
        _kw_url_fallback = None
        for url, pg in nav_graph.items():
            ul = url.lower()
            tl = str(pg.get("title","")).lower() if isinstance(pg, dict) else ""
            if any(kw in ul or kw in tl for kw in keywords):
                if _is_c1main(url):
                    _kw_url_fallback = url
                else:
                    _kw_url = url
                    break
        if _kw_url is None and _kw_url_fallback:
            _kw_url = _kw_url_fallback
        # 選定
        _selected_src = "none"
        # P21.5: media_structure_map entrypointを第一候補に
        if _struct_ep_url and _struct_ep_url.startswith("http") and not _is_c1main(_struct_ep_url):
            target_url    = _struct_ep_url
            _selected_src = "structure_entrypoint"
        elif _detail_url and _detail_url.startswith("http") and not _is_c1main(_detail_url):
            target_url   = _detail_url
            _selected_src = "detail_source_url"
        elif _detail_url and _detail_url.startswith("http"):
            # C1Mainでも他候補がなければ採用
            _alt = next((u for u in [_p24_src_url, _map_url, _kw_url] if u and u.startswith("http") and not _is_c1main(u)), None)
            if _alt:
                target_url   = _alt
                _selected_src = "p24_source_url" if _alt == _p24_src_url else ("map_target_url" if _alt == _map_url else "keyword_match")
            else:
                target_url   = _detail_url
                _selected_src = "detail_source_url(c1main_fallback)"
        elif _p24_src_url and _p24_src_url.startswith("http") and not _is_c1main(_p24_src_url):
            target_url   = _p24_src_url
            _selected_src = "p24_source_url"
        elif _map_url and _map_url.startswith("http") and not _is_c1main(_map_url):
            target_url   = _map_url
            _selected_src = "map_target_url"
        elif _kw_url:
            target_url   = _kw_url
            _selected_src = "keyword_match"
        else:
            target_url = None
        print(
            f"[P24_TARGET_PRIORITY]"
            f" op={op_type}"
            f" detail_source_url={_detail_url}"
            f" p24_source_url={_p24_src_url}"
            f" mapping_target_url={_map_url}"
            f" selected_target_url={target_url}"
            f" selected_src={_selected_src}",
            flush=True
        )
        if not target_url:
            computed_status = "NEEDS_MAPPING"
            print(f"[P24_TARGET_REJECTED] op={op_type} candidate_url=None reason=no_target_url", flush=True)
        elif _selected_src == "keyword_match":
            computed_status = "NEEDS_MAPPING"
            print(f"[P24_TARGET_REJECTED] op={op_type} candidate_url={target_url} reason=keyword_only_target_rejected", flush=True)
            target_url = None
        else:
            computed_status = _status_check or op_map.get("status") or "NEEDS_MAPPING"
            _dom_score = op_map.get("validation_score", 0)
            print(
                f"[P24_STATUS_DECISION_FROM_MAPPING] op={op_type} status={computed_status}"
                f" score={_dom_score} missing={missing} target_url={target_url}",
                flush=True
            )
        # 修正9: READYのみsteps生成。PARTIAL/NEEDS_MAPPING/FAILEDは空で返す
        # READY or PARTIAL(target_url確定・score>0)をsteps生成対象にする
        _steps_eligible = (
            computed_status == "READY"
            or (computed_status == "PARTIAL" and target_url and _dom_score > 0)
        )
        if not _steps_eligible:
            result[op_type] = []
            continue
        steps = []
        for order, step_type in enumerate(template):
            step = {"order": order, "step_type": step_type, "display_name": step_type, "status": "UNKNOWN", "source_url": None, "target_url": None, "selector_role": None, "selector": None}
            if step_type == "login":
                step["status"] = "READY"
                step["display_name"] = "ログイン"
            elif step_type.startswith("navigate_"):
                step["target_url"] = target_url
                step["display_name"] = "画面へ移動"
                step["status"] = "READY" if target_url else "NEEDS_MAPPING"
            elif step_type in ("input_title","input_body","input_text","input_entity_fields","edit_entity_fields","input_schedule","input_price"):
                role = "body" if "body" in step_type or "text" in step_type else step_type.replace("input_","").replace("edit_","")
                step["source_url"] = (avail_sel.get(role) or {}).get("source_url", target_url) if isinstance(avail_sel.get(role), dict) else target_url
                step["target_url"] = target_url
                step["selector_role"] = role
                step["selector"] = avail_sel.get(role) or avail_sel.get("body") or avail_sel.get("input")
                step["display_name"] = "入力"
                step["status"] = computed_status
                if missing: step["missing_required_fields"] = missing
            elif step_type == "upload_file":
                step["source_url"] = target_url
                step["selector_role"] = "file"
                step["selector"] = avail_sel.get("file")
                step["display_name"] = "ファイルアップロード"
                step["status"] = computed_status
                if missing: step["missing_required_fields"] = missing
            elif step_type == "save":
                step["source_url"] = (avail_sel.get("save") or avail_sel.get("submit") or {}).get("source_url", target_url) if isinstance(avail_sel.get("save") or avail_sel.get("submit"), dict) else target_url
                step["target_url"] = target_url
                step["selector"] = avail_sel.get("save") or avail_sel.get("submit")
                step["display_name"] = "保存"
                step["status"] = computed_status
                if missing: step["missing_required_fields"] = missing
            elif step_type.startswith("verify_"):
                step["display_name"] = "反映確認"
                step["status"] = computed_status if computed_status in ("READY", "PARTIAL") else "NEEDS_MAPPING"
            steps.append(step)
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
        combined = " ".join([raw_href.lower(), text.lower(), cls.lower(), onclick.lower(), aria.lower()])
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

    # E. Operation判定（編集フォーム証拠必須）

    # E-1. media_replace: file input + save系ボタン
    if files > 0 and _save_ok:
        op_capability.append("media_replace")
        evidence.append("file_input_with_save")
        page_type   = "media_edit"
        domain_area = "media"
        value_score += 30

    # E-2. schedule_update: schedule系URL/title + input + save系ボタン
    _is_schedule = any(p in url or p in title for p in (
        "schedule", "shift", "calendar", "予定", "出勤", "スケジュール", "sch"
    ))
    if _is_schedule and inputs > 0 and _save_ok:
        op_capability = [o for o in op_capability if o != "entity_update"]  # entity_updateと併記禁止
        if "schedule_update" not in op_capability:
            op_capability.append("schedule_update")
        evidence.append("schedule_url_with_input_and_save")
        page_type   = "schedule_edit"
        domain_area = "schedule"
        value_score += 30

    # E-3. price_update: price系URL + price系input + save系ボタン
    if _price_url and inputs > 0 and _save_ok:
        if "price_update" not in op_capability:
            op_capability.append("price_update")
        evidence.append("price_url_with_input_and_save")
        page_type   = "price_edit"
        domain_area = "price"
        value_score += 30

    # E-4. news_post: news系URL/title + textarea/body系input + save系ボタン
    _is_news_url = any(p in url or p in title for p in (
        "news", "post", "blog", "diary", "投稿", "ニュース", "日記", "topics", "topic", "content", "article"
    ))
    _is_edit_url = any(p in url or p in title for p in (
        "edit", "new", "create", "add", "regist", "form", "編集", "登録", "新規"
    ))
    if _is_news_url and _is_edit_url and (textareas > 0 or inputs > 0) and _save_ok:
        if "news_post" not in op_capability:
            op_capability.append("news_post")
        if "text_update" not in op_capability:
            op_capability.append("text_update")
        evidence.append("news_edit_url_with_textarea_and_save")
        if page_type == "unknown":
            page_type   = "content_edit"
            domain_area = "content"
        value_score += 30

    # E-5. text_update: edit系URL + textarea + save系ボタン（searchフォームのみは不可）
    _is_text_edit = any(p in url or p in title for p in (
        "edit", "text", "profile", "info", "update", "編集", "テキスト", "banner", "description"
    ))
    if _is_text_edit and (textareas > 0 or inputs > 0) and _save_ok and not _is_news_url:
        if "text_update" not in op_capability:
            op_capability.append("text_update")
        evidence.append("text_edit_url_with_input_and_save")
        if page_type == "unknown":
            page_type   = "content_edit"
            domain_area = "content"
        value_score += 20

    # E-6. entity_register: register/new系URL + input + save系ボタン（listページ除外済）
    _is_register = any(p in url or p in title for p in (
        "register", "regist", "new", "add", "create", "登録", "新規", "entry"
    ))
    if _is_register and inputs > 0 and _save_ok and not _price_url and not _is_schedule:
        if "entity_register" not in op_capability:
            op_capability.append("entity_register")
        evidence.append("register_url_with_input_and_save")
        if page_type == "unknown":
            page_type   = "entity_edit"
            domain_area = "entity"
        value_score += 20

    # E-7. entity_update: cast_edit/staff_edit系URL + input + save系ボタン（listページ除外済）
    _is_entity_edit = any(p in url or p in title for p in (
        "cast_edit", "staff_edit", "member_edit", "cast/edit", "staff/edit"
    ))
    if _is_entity_edit and inputs > 0 and _save_ok and not _price_url and not _is_schedule:
        if "entity_update" not in op_capability:
            op_capability.append("entity_update")
        evidence.append("entity_edit_url_with_input_and_save")
        if page_type == "unknown":
            page_type   = "entity_edit"
            domain_area = "entity"
        value_score += 20

    # E-8. DOM密度フォールバック\uff08URL語彙非依存\uff09
    # formあり + save_ok + inputs>2 ならcontent_edit候補として扱う
    if page_type == "unknown" and forms > 0 and _save_ok and inputs > 2:
        if "text_update" not in op_capability:
            op_capability.append("text_update")
        evidence.append("dom_density_fallback")
        page_type   = "content_edit"
        domain_area = "content"
        value_score += 20
        print(f"[P21_DOM_DENSITY_FALLBACK] url={url[:80]} forms={forms} inputs={inputs} save_ok={_save_ok}", flush=True)

    # F. 共通条件チェック
    print(
        f"[P21_STRUCTURE_EVIDENCE] url={url[:80]} inputs={inputs} textareas={textareas}"
        f" file_inputs={files} buttons={buttons} save_ok={_save_ok} evidence={evidence} ops={op_capability}",
        flush=True
    )

    # 修正3: operation_capability競合排除
    if page_type == "entity_edit":
        op_capability = [o for o in op_capability if o not in ("text_update",)]
    if page_type == "schedule_edit":
        op_capability = [o for o in op_capability if o not in ("entity_update", "entity_register", "text_update")]
    if page_type == "media_edit":
        op_capability = [o for o in op_capability if o not in ("text_update", "entity_update", "entity_register")]
    # スコア内訳ログ
    print(
        f"[P21_SCORE_BREAKDOWN] url={url[:80]} page_type={page_type}"
        f" base_score={value_score} save_ok={_save_ok}"
        f" ops={op_capability} negative={negative_reasons}",
        flush=True
    )
    _edit_page_types = (
        "media_edit", "schedule_edit", "price_edit", "content_edit", "entity_edit",
        "entity_form", "schedule_form", "content_form", "media_form",
        "news_edit", "text_edit", "price_form", "news_form",
    )
    # edit/form系ページでDOMがあればscore/save_button条件を免除してcandidate許可
    _has_dom = (forms > 0 or inputs > 0 or buttons > 0)
    _is_edit_form_type = page_type in _edit_page_types
    if 0 < value_score < 70 and op_capability and not (_is_edit_form_type and _has_dom):
        negative_reasons.append("score_below_candidate_threshold")
    if page_type not in _edit_page_types and op_capability:
        negative_reasons.append("page_type_not_edit_or_form")
    if _is_edit_form_type and _has_dom and len(op_capability) > 0:
        # edit/form系: DOM存在のみでcandidate=True（save_button・score免除）
        is_candidate = True
    else:
        is_candidate = (
            len(op_capability) > 0
            and value_score >= 70
            and _save_ok
            and page_type in _edit_page_types
            and not negative_reasons
        )

    if is_candidate:
        print(
            f"[P21_OPERATION_CANDIDATE_ACCEPTED] url={url[:80]} ops={op_capability}"
            f" evidence={evidence} score={value_score}",
            flush=True
        )
    else:
        if op_capability:
            negative_reasons.append("save_button_missing_or_low_score")
        print(
            f"[P21_OPERATION_CANDIDATE_REJECTED] url={url[:80]} reason={negative_reasons}"
            f" evidence={evidence}",
            flush=True
        )

    # [修正B-2] P21_PAGE_CLASSIFYログ（全ページ詳細出力）
    print(
        f'[P21_PAGE_CLASSIFY] url={page_summary.get("url","")[:80]}'
        f' title={page_summary.get("title","")[:40]}'
        f' page_type={page_type} domain_area={domain_area}'
        f' is_operation_candidate={is_candidate}'
        f' operation_candidates={op_capability}'
        f' inputs={inputs} textareas={textareas} file_inputs={files}'
        f' buttons={buttons} forms={forms} links={links}'
        f' negative_reasons={negative_reasons}',
        flush=True
    )
    return {
        "page_type":              page_type,
        "domain_area":            domain_area,
        "operation_capability":   op_capability,
        "is_operation_candidate": is_candidate,
        "negative_reasons":       negative_reasons,
        "value_score":            value_score,
        "evidence":               evidence,
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
        "delete","remove","clear","reset","close","copy","download","export",
        "display","undisplay","toggle","bulk_delete","trash",
        "login","signin","sign_in","auth","authenticate",
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
    cls     = (element.get("class") or "").lower()
    title   = (element.get("title") or "").lower()
    onclick = (element.get("onclick") or "").lower()
    href    = (element.get("href") or "").lower()
    action  = (element.get("action") or "").lower()
    accept  = (element.get("accept") or "").lower()

    # hidden/password/csrf/tokenは除外
    if typ in ("hidden", "password") or "csrf" in name or "token" in name:
        return {"score": -50, "matched": [], "negative": ["hidden_or_credential"], "evidence": []}

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
    if role == "body" and tag == "textarea":
        score += weights.get("textarea", 60)
        evidence.append("tag=textarea")

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
        return {"status": "NEEDS_MAPPING", "selectors": {}, "missing": [operation_type], "validation_score": 0, "target_url": page_summary.get("url",""), "evidence": [], "executable": False}

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
            sel = ""
            if best_el.get("id"):
                sel = f"#{best_el['id']}"
            elif best_el.get("name"):
                tag = best_el.get("tag","input")
                sel = f"{tag}[name='{best_el['name']}']"
            elif best_el.get("class"):
                tag = best_el.get("tag","input")
                first_cls = best_el["class"].split()[0]
                sel = f"{tag}.{first_cls}"
            else:
                sel = best_el.get("tag","input")
            # [SELECTOR_QUALITY] reject tag-only selectors (no id/name/class)
            _sel_is_tag_only = sel in ("input", "button", "a", "textarea", "select", "form")
            if _sel_is_tag_only:
                print(f"[SELECTOR_QUALITY_REJECTED] url={page_summary.get('url','')[:60]} role={role} selector={sel} score={best_score} reason=tag_only_selector", flush=True)
            else:
                best[role] = {
                    "selector":  sel,
                    "role":      role,
                    "score":     best_score,
                    "matched":   best_result["matched"],
                    "evidence":  best_result["evidence"],
                }
                print(f"[P22_DOM_ROLE_MATCH] url={page_summary.get('url','')[:60]} role={role} selector={sel} score={best_score} matched={best_result['matched'][:3]}", flush=True)

    missing = [r for r in required_roles if r not in best]
    validation_score = 0
    if best:
        scores = [v["score"] for v in best.values()]
        validation_score = int(sum(scores) / len(scores)) if scores else 0

    if not missing and validation_score >= req["min_score"]:
        status = "READY"
    elif best:
        status = "PARTIAL"
    else:
        status = "NEEDS_MAPPING"

    return {
        "status":           status,
        "selectors":        best,
        "missing":          missing,
        "validation_score": validation_score,
        "target_url":       page_summary.get("url", ""),
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
            # list/dashboard/non_op pagesはtarget除外
            if not pg.get("is_operation_candidate", False):
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=not_operation_candidate page_type={page_type} url={pg.get('url','')[:60]}", flush=True)
                continue
            if page_type in _NON_OP_PAGE_TYPES:
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=non_op_page_rejected page_type={page_type} url={pg.get('url','')[:60]}", flush=True)
                continue
            if area not in allowed_areas and page_type not in allowed_page_types:
                continue
            list_only = False
            mapping = extract_operation_selectors_from_page(pg, op_type)
            if mapping["validation_score"] > best_score:
                best_score   = mapping["validation_score"]
                best_mapping = mapping
        if best_mapping and best_score >= min_score and not best_mapping["missing"]:
            result[op_type] = {
                "status":           "READY",
                "target_url":       best_mapping["target_url"],
                "selectors":        best_mapping["selectors"],
                "validation_score": best_score,
                "missing":          [],
                "source":           "dom_evidence_mapper",
                "executable":       True,
            }
            print(f"[P22_OPERATION_MAPPING_BUILT] op={op_type} status=READY target_url={best_mapping['target_url'][:60]} validation_score={best_score}", flush=True)
        elif best_mapping and best_score > 0:
            # PARTIAL厳格化: saveだけ取れてもPARTIAL不可
            _partial_url   = best_mapping["target_url"].lower()
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
            if best_mapping and best_score > 0 and best_mapping.get("target_url") and not _partial_ng:
                result[op_type] = {
                    "status":           "PARTIAL",
                    "target_url":       best_mapping["target_url"],
                    "selectors":        best_mapping["selectors"],
                    "validation_score": best_score,
                    "missing":          best_mapping["missing"],
                    "source":           "dom_evidence_mapper",
                "executable":       False,
                }
                print(f"[P22_OPERATION_MAPPING_BUILT] op={op_type} status=PARTIAL target_url={best_mapping['target_url'][:60]} missing={best_mapping['missing']} validation_score={best_score}", flush=True)
            else:
                result[op_type] = {
                    "status":           "NEEDS_MAPPING",
                    "target_url":       "",
                    "selectors":        {},
                    "validation_score": 0,
                    "missing":          req["required"],
                    "error_reason":     "partial_candidate_rejected",
                    "source":           "dom_evidence_mapper",
                    "executable":       False,
                }
                print(f"[P22_OPERATION_MAPPING_SKIPPED] op={op_type} reason=partial_candidate_rejected url={best_mapping['target_url'][:60]}", flush=True)
        else:
            _nm_reason = "list_page_requires_followup_edit_form" if list_only else "no_qualifying_page"
            _nm_missing = req["required"] + (["edit_form"] if list_only else [])
            result[op_type] = {
                "status":           "NEEDS_MAPPING",
                "target_url":       "",
                "selectors":        {},
                "validation_score": 0,
                "missing":          _nm_missing,
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
    # navigation_graph.pages リスト生成（P21.5: classify_admin_page_structure適用）
    _pages_list = []
    for _u, _v in nav_graph.items():
        if not isinstance(_v, dict):
            continue
        _pg_summary = {
            "url":               _v.get("url", _u),
            "title":             _v.get("title", ""),
            "forms_count":       _v.get("forms_count", 0),
            "inputs_count":      _v.get("inputs_count", 0),
            "buttons_count":     _v.get("buttons_count", 0),
            "file_inputs_count": _v.get("file_inputs_count", 0),
            "links_count":       len(_v.get("links", [])),
            "textareas_count":   _v.get("textareas_count", 0),
            "selects_count":     _v.get("selects_count", 0),
            "tables_count":      _v.get("tables", 0),
            "inputs":            _v.get("inputs", []),
            "textareas":         _v.get("textareas", []),
            "buttons":           _v.get("buttons", []),
            "links":             _v.get("links", []),
            "forms":             _v.get("forms", []),
            "file_inputs":       _v.get("file_inputs", []),
            "menu_items":        _v.get("menu_items", []),
            "followup_links":    _v.get("followup_links", []),
        }
        _cls = classify_admin_page_structure(_pg_summary)
        # 完全除外(static/error)はpages保存しない
        if _cls["page_type"] in ("static", "error"):
            print(f"[P21_URL_REJECTED] url={_pg_summary['url'][:80]} reason={_cls['negative_reasons']}", flush=True)
            continue
        _pages_list.append({
            "url":                    _pg_summary["url"],
            "title":                  _pg_summary["title"],
            "forms_count":            _pg_summary["forms_count"],
            "inputs_count":           _pg_summary["inputs_count"],
            "buttons_count":          _pg_summary["buttons_count"],
            "file_inputs_count":      _pg_summary["file_inputs_count"],
            "textareas_count":        _pg_summary["textareas_count"],
            "selects_count":          _pg_summary["selects_count"],
            "links_count":            _pg_summary["links_count"],
            "inputs":                 _pg_summary["inputs"][:120],
            "textareas":              _pg_summary["textareas"][:30],
            "buttons":                _pg_summary["buttons"][:80],
            "links":                  _pg_summary["links"][:200],
            "forms":                  _pg_summary["forms"][:20],
            "file_inputs":            _pg_summary["file_inputs"][:20],
            "menu_items":             _pg_summary["menu_items"][:50],
            "followup_links":         _pg_summary["followup_links"][:20],
            "operation_hints":        _cls["operation_capability"],
            "page_type":              _cls["page_type"],
            "domain_area":            _cls["domain_area"],
            "is_operation_candidate": _cls["is_operation_candidate"],
            "value_score":            _cls["value_score"],
            "negative_reasons":       _cls["negative_reasons"],
            "evidence":               _cls.get("evidence", []),
            "dom_evidence": {
                "has_form":              _pg_summary["forms_count"] > 0,
                "has_input":             _pg_summary["inputs_count"] > 0,
                "has_file_input":        _pg_summary["file_inputs_count"] > 0,
                "has_button":            _pg_summary["buttons_count"] > 0,
                "has_textarea":          _pg_summary["textareas_count"] > 0,
                "ignored_for_operation": _cls.get("ignored_for_operation", False),
                "ignore_reason":         _cls.get("ignore_reason", ""),
            },
        })
        if not _cls["is_operation_candidate"] and _cls.get("ignored_for_operation"):
            print(f"[P21_OPERATION_HINT_REJECTED] url={_pg_summary['url'][:80]} title={_pg_summary['title'][:40]} reason={_cls['negative_reasons']}", flush=True)
        print(f"[P21_PAGE_VALUE] url={_pg_summary['url'][:80]} forms={_pg_summary['forms_count']} inputs={_pg_summary['inputs_count']} buttons={_pg_summary['buttons_count']} links={_pg_summary['links_count']} value_score={_cls['value_score']} ignored_for_operation={_cls.get('ignored_for_operation',False)}", flush=True)
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
    # media_structure_map生成（P21.5）
    _struct_map = {}
    try:
        _struct_map = build_media_structure_map(_pages_list)
    except Exception as _sm_err:
        print(f"[P21_STRUCTURE_MAP_ERROR] {_sm_err}", flush=True)
    try:
        db.collection("media_mappings").document(mapping_id).update({
            "navigation_graph.pages":  _pages_list,
            "navigation_graph.updated_at": _dt_snap.datetime.utcnow().isoformat(),
            "crawl_state":             _crawl_state,
            "crawl_status":            status,
            "crawl_resume_queue":      _rq_save,
            "crawl_status":            status,
            "media_structure_map":     _struct_map,
        })
        print(
            f"[P21_SNAPSHOT_SAVED] mapping_id={mapping_id} status={status}"
            f" pages={len(_pages_list)} remaining={len(_remaining)}"
            f" structure_areas={_struct_map.get('summary',{}).get('areas_detected',[])} "
            f"entrypoints={list(_struct_map.get('operation_entrypoints',{}).keys())}",
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
        u = url.lower()
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
        _cur_url = page.url.lower()
        _login_kws = ('signin', 'login', 'auth', 'c1login')
        if any(k in _cur_url for k in _login_kws) and not start_url:
            # try clicking first non-login link
            try:
                _admin_links = page.eval_on_selector_all('a[href]', 'els => els.map(e => e.getAttribute("href")).filter(h => h && !h.includes("login") && !h.includes("signin") && !h.startsWith("#") && !h.startsWith("javascript"))')
                _admin_top = next((h for h in _admin_links if 'admin' in h.lower() or 'manage' in h.lower()), None) or next((h for h in _admin_links if h.startswith('/')), None)
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
                         'キャスト','スタッフ','写真','画像','出勤','登録','編集')
        _KW_DANGER     = ('logout','delete','remove','destroy')
        _candidate_links1 = []
        _ignored_links1   = []
        for link in top_structure.get("links", []):
            href = _normalize_href(link.get("href", ""), base_url)
            _ltext = (link.get("text") or "").lower()
            _lhref_low = href.lower()
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
                visited.add(url)
                print(f"[P21_PAGE] url={structure.get('url','')} title={structure.get('title','')}", flush=True)
                # [修正C-2] 新リンクをキューに追加（キーワード優先）
                _candidate_links2 = []
                _ignored_links2   = []
                for _lnk in structure.get("links", []):
                    _href = _normalize_href(_lnk.get("href", ""), url)
                    _ltxt2 = (_lnk.get("text") or "").lower()
                    _lhref2 = _href.lower()
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
            "can_update_schedule": any("schedule" in u.lower() or "sch" in u.lower() or "出勤" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_update_price":    any("price" in u.lower() or "course" in u.lower() or "料金" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_register_entity": any("register" in u.lower() or "new" in u.lower() or "add" in u.lower() or "edit" in u.lower() or "追加" in (v.get("title","")) for u, v in nav_graph.items()),
            "can_update_entity":   any("edit" in u.lower() or "list" in u.lower() or "一覧" in (v.get("title","")) for u, v in nav_graph.items()),
        }

        # operation_candidates推定
        operation_candidates = []
        if capabilities.get("can_update_schedule"): operation_candidates.append("schedule_update")
        if capabilities.get("can_post_news"):        operation_candidates.append("news_post")
        if capabilities.get("can_update_text"):      operation_candidates.append("text_update")
        if capabilities.get("can_upload_image"):     operation_candidates.append("media_replace")
        if capabilities.get("can_update_price"):     operation_candidates.append("price_update")
        if capabilities.get("can_register_entity"):  operation_candidates.append("entity_register")
        if capabilities.get("can_update_entity"):    operation_candidates.append("entity_update")
        if capabilities.get("can_navigate_admin"):   operation_candidates.append("admin_crawl")

        # Firestore保存
        _nav_save = {}
        for k, v in nav_graph.items():
            _op_hints = []
            _title_lower = (v.get("title") or "").lower()
            _url_lower   = (v.get("url") or k or "").lower()
            for _op_hint, _op_kws in {
                "news_post":       ["news", "post", "blog", "diary", "投稿", "ニュース", "日記"],
                "text_update":     ["edit", "text", "profile", "info", "update", "編集", "テキスト"],
                "media_replace":   ["image", "photo", "media", "upload", "写真", "画像"],
                "schedule_update": ["schedule", "shift", "calendar", "予定", "スケジュール", "出勤"],
                "price_update":    ["price", "fee", "course", "料金", "コース", "pricelist"],
                "entity_register": ["register", "new", "add", "create", "登録", "新規"],
                "entity_update":   ["edit", "update", "修正", "変更", "編集"],
            }.items():
                if any(kw in _url_lower or kw in _title_lower for kw in _op_kws):
                    _op_hints.append(_op_hint)
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
                "operation_hints":   _op_hints,
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
            "operation_candidates":    operation_candidates,
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

        # P22: operation_mappings生成（DOM証拠から）
        try:
            _p22_pages = (db.collection('media_mappings').document(mapping_id).get().to_dict() or {}).get('navigation_graph', {}).get('pages', [])
            if not _p22_pages:
                # _save_crawl_snapshotで保存した_pages_listを再構築
                _p22_pages = []
                for _pu, _pv in nav_graph.items():
                    if isinstance(_pv, dict):
                        _p22_pages.append(_pv)
            _op_mappings = build_operation_mappings_from_dom_evidence(mapping_id, _p22_pages)
            # operation_steps_by_typeをREADYのみ再生成
            _ready_ops = [
                op for op, m in _op_mappings.items()
                if m.get('status') == 'READY'
                or (m.get('status') == 'PARTIAL' and m.get('target_url') and m.get('validation_score', 0) > 0)
            ]
            _op_steps = {}
            if _ready_ops:
                try:
                    # DOM evidence由来のdetail_by_opとmeta付きnav_graphを構築
                    _nav_for_steps = dict(nav_graph or {})
                    _nav_for_steps["__meta__"] = {
                        "operation_entrypoints": {
                            op: {
                                "url":        (_op_mappings.get(op) or {}).get("target_url", ""),
                                "confidence": (_op_mappings.get(op) or {}).get("validation_score", 0) / 100,
                                "evidence":   (_op_mappings.get(op) or {}).get("evidence", []),
                            }
                            for op in _ready_ops
                            if (_op_mappings.get(op) or {}).get("target_url")
                        }
                    }
                    _detail_by_op = {
                        op: {
                            "source_url":  (_op_mappings.get(op) or {}).get("target_url", ""),
                            "confidence":  (_op_mappings.get(op) or {}).get("validation_score", 0),
                            "source":      "dom_evidence_mapper",
                        }
                        for op in _ready_ops
                    }
                    print(f'[P24_REBUILD_INPUT] ready_ops={_ready_ops} detail_by_op_keys={list(_detail_by_op.keys())} meta_entrypoints={list(_nav_for_steps["__meta__"]["operation_entrypoints"].keys())}', flush=True)
                    _op_steps = rebuild_operation_steps(
                        _ready_ops,
                        _nav_for_steps,
                        {op: _op_mappings[op] for op in _ready_ops},
                        _detail_by_op,
                    )
                    print(f'[P24_STEPS_REBUILT_FROM_DOM] ops={_ready_ops} steps_keys={list(_op_steps.keys())}', flush=True)
                    # READYがあるのにstepsが空ならエラーログ
                    for _chk_op in _ready_ops:
                        if _chk_op not in _op_steps:
                            _chk_m = _op_mappings.get(_chk_op, {})
                            print(f'[P24_STEPS_MISSING_FOR_READY] op={_chk_op} status={_chk_m.get("status")} target_url={_chk_m.get("target_url","")[:60]} selectors_keys={list(_chk_m.get("selectors",{}).keys())}', flush=True)
                except Exception as _rs_err:
                    print(f'[P22_REBUILD_STEPS_ERROR] {_rs_err}', flush=True)
            # READY以外のstepsは削除
            _existing_steps = (db.collection("media_mappings").document(mapping_id).get().to_dict() or {}).get("operation_steps_by_type", {})
            # [READY_PROTECT] existing READY not overwritten by new result
            _existing_op_mappings = (db.collection("media_mappings").document(mapping_id).get().to_dict() or {}).get("operation_mappings", {})
            _protected_op_mappings = {}
            for _op_k, _op_v in _op_mappings.items():
                _existing_op = _existing_op_mappings.get(_op_k, {})
                if _existing_op.get("status") == "READY" and _op_v.get("status") != "READY":
                    _protected_op_mappings[_op_k] = _existing_op
                    print(f'[READY_PROTECT] op={_op_k} kept_existing_READY new_status={_op_v.get("status")}', flush=True)
                else:
                    _protected_op_mappings[_op_k] = _op_v
            # [READY_OPS_SYNC] recalc _ready_ops after READY_PROTECT to protect steps
            _ready_ops_protected = [
                op for op, m in _protected_op_mappings.items()
                if m.get('status') == 'READY'
                or (m.get('status') == 'PARTIAL' and m.get('target_url') and m.get('validation_score', 0) > 0)
            ]
            _cleaned_steps = {k: v for k, v in _existing_steps.items() if k in _ready_ops_protected}
            _cleaned_steps.update(_op_steps)
            db.collection('media_mappings').document(mapping_id).update({
                'operation_mappings':      _protected_op_mappings,
                'operation_steps_by_type': _cleaned_steps,
            })
            print(f'[P22_OPERATION_MAPPINGS_SAVED] mapping_id={mapping_id} ops={list(_op_mappings.keys())} step_ops={_ready_ops} steps_count={len(_cleaned_steps)}', flush=True)
            print(f'[P24_STEPS_SAVE_DONE] count={len(_cleaned_steps)}', flush=True)
            # [OP_CANDIDATES_SYNC] P22 READY/PARTIALをoperation_candidatesに同期
            _synced_candidates = list(operation_candidates)
            for _sc_op, _sc_m in _protected_op_mappings.items():
                if _sc_m.get('status') in ('READY', 'PARTIAL') and _sc_op not in _synced_candidates:
                    _synced_candidates.append(_sc_op)
                    print(f'[OP_CANDIDATES_SYNC] op={_sc_op} status={_sc_m.get("status")} added_to_candidates', flush=True)
            if set(_synced_candidates) != set(operation_candidates):
                db.collection('media_mappings').document(mapping_id).update({'operation_candidates': _synced_candidates})
                print(f'[OP_CANDIDATES_SYNC_SAVED] before={operation_candidates} after={_synced_candidates}', flush=True)
        except Exception as _p22_err:
            print(f'[P22_OPERATION_MAPPINGS_ERROR] {_p22_err}', flush=True)

        # verify_selector自動保存
        try:
            _existing = db.collection("media_mappings").document(mapping_id).get().to_dict() or {}
            _svs = _existing.get("verify_selector")
            if not _svs:
                # タイトルベースでverify_selector候補を探す
                for _url, _vd in nav_graph.items():
                    if "signin" not in _url.lower() and "login" not in _url.lower():
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
            "operation_candidates":    operation_candidates,
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
