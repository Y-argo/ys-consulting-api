# api/core/agent_executor.py
# Executor層：execute_taskから委譲される実処理の振り分け
# LLMは使わない。実処理・媒体操作・条件分岐はここで行う。

from typing import Optional
from api.core.browser_executor import run_browser_operation, GENERIC_OPERATION_CONFIG


WAITING_MAPPING  = "WAITING_MAPPING"
WAITING_EXECUTOR = "WAITING_EXECUTOR"
BLOCKED          = "BLOCKED"
DONE             = "DONE"

# P8: OPERATION_CAPABILITY_MAP → GENERIC_OPERATION_CONFIGから動的生成
# browser_executor.GENERIC_OPERATION_CONFIGのcapability_keyを参照
# operation_type → capability_key のマッピングはGENERIC_OPERATION_CONFIGに統合済み
# ここではrequired_selectors取得もGENERIC_OPERATION_CONFIGから行う

def _get_required_capability(operation_type: str) -> str:
    """GENERIC_OPERATION_CONFIGからoperation_typeの必要capabilityを返す"""
    cfg = GENERIC_OPERATION_CONFIG.get(operation_type, {})
    return cfg.get("capability_key", "")



def execute_agent_task(task: dict, media_mapping: Optional[dict] = None) -> dict:
    """
    agent_taskを受け取り、agent_typeに応じて処理を振り分ける。
    返り値: {"status": str, "executed": bool, "message": str, ...}
    """
    agent_type = task.get("agent_type", "")

    if agent_type == "hp_update":
        return _execute_hp_update(task, media_mapping)

    if agent_type == "audit":
        return _execute_audit(task, media_mapping)

    if agent_type == "interview":
        return _execute_interview(task, media_mapping)

    return {
        "status":   WAITING_EXECUTOR,
        "executed": False,
        "message":  f"agent_type '{agent_type}' は未対応です",
    }


# ── HP・媒体更新 ────────────────────────────────────────────────

def _execute_hp_update(task: dict, media_mapping: Optional[dict]) -> dict:
    operation_type = task.get("operation_type", "")

    # 1. 媒体マッピング未設定チェック
    if not media_mapping:
        return {
            "status":           WAITING_MAPPING,
            "executed":         False,
            "requires_mapping": True,
            "message":          "媒体マッピングが未設定のため実行できません。媒体マッピングタブで設定してください。",
        }

    # 2. credential_secret_name未設定チェック（ID/PASSをFirestoreに平文保存しない）
    if not media_mapping.get("credential_secret_name"):
        return {
            "status":               BLOCKED,
            "executed":             False,
            "message":              "credential_secret_nameが未設定です。Secret Managerへの媒体ログイン情報登録が必要です。",
            "missing_capabilities": [],
            "missing_selectors":    [],
            "missing_credentials":  True,
            "blocked_reason":       "credential_secret_name未設定",
            "executor_reason":      "",
        }

    # 2.5. P1-2: capability enforcement
    _caps = media_mapping.get("capabilities") or {}
    _required_cap = _get_required_capability(operation_type)
    if _required_cap and not _caps.get(_required_cap, False):
        return {
            "status":               WAITING_EXECUTOR,
            "executed":             False,
            "message":              f"この媒体は '{operation_type}' に必要な capability '{_required_cap}' が未対応です。媒体マッピングのcapabilitiesを更新してください。",
            "missing_capabilities": [_required_cap],
            "missing_selectors":    [],
            "missing_credentials":  False,
            "blocked_reason":       "",
            "executor_reason":      f"capability '{_required_cap}' not enabled",
        }

    # 3. login_url未設定チェック
    if not media_mapping.get("login_url"):
        return {
            "status":               BLOCKED,
            "executed":             False,
            "message":              "login_urlが未設定です。媒体マッピングを更新してください。",
            "missing_capabilities": [],
            "missing_selectors":    [],
            "missing_credentials":  False,
            "blocked_reason":       "login_url未設定",
            "executor_reason":      "",
        }

    # 4. ログイン用セレクター必須チェック
    dom = media_mapping.get("dom_selectors", {})
    login_required = ["username", "password", "login_submit"]
    missing_login  = [k for k in login_required if not dom.get(k)]
    if missing_login:
        return {
            "status":               WAITING_MAPPING,
            "executed":             False,
            "message":              f"ログイン用DOMセレクターが不足しています: {', '.join(missing_login)}",
            "missing_capabilities": [],
            "missing_selectors":    missing_login,
            "missing_credentials":  False,
            "blocked_reason":       "",
            "executor_reason":      f"missing login selectors: {missing_login}",
        }

    # 5. operation_typeごとの必須セレクターチェック（P14: operation_stepsがある場合はスキップ）
    if not task.get("operation_steps"):
        required = _required_selectors(operation_type)
        missing  = [k for k in required if not dom.get(k)]
        if missing:
            return {
                "status":               WAITING_MAPPING,
                "executed":             False,
                "message":              f"操作用DOMセレクターが不足しています: {', '.join(missing)}",
                "missing_capabilities": [],
                "missing_selectors":    missing,
                "missing_credentials":  False,
                "blocked_reason":       "",
                "executor_reason":      f"missing op selectors: {missing}",
            }

    # 6. 全条件クリア → browser_executorへ委譲
    # P14: operation_stepsがあれば引き渡す
    _operation_steps = task.get("operation_steps") or None
    # [P22_SELECTOR_MERGE] merge P22 detected selectors into dom_selectors
    _merged_mapping = dict(media_mapping)
    _op_mappings = media_mapping.get("operation_mappings", {})
    _op_selectors = (_op_mappings.get(operation_type) or {}).get("selectors", {})
    if _op_selectors:
        _merged_dom = dict(media_mapping.get("dom_selectors", {}))
        for _role, _sel_info in _op_selectors.items():
            if isinstance(_sel_info, dict):
                _sel_str = _sel_info.get("selector", "")
            else:
                _sel_str = str(_sel_info)
            if _sel_str and _role not in _merged_dom:
                _merged_dom[_role] = _sel_str
        _merged_mapping["dom_selectors"] = _merged_dom
        print(f"[P22_SELECTOR_MERGE] op={operation_type} merged_roles={list(_op_selectors.keys())}", flush=True)
    return run_browser_operation(_merged_mapping, operation_type, task.get("payload", {}), operation_steps=_operation_steps)


def _required_selectors(operation_type: str) -> list:
    """P8: GENERIC_OPERATION_CONFIGからrequired_selector_keysを取得。重複定義を排除。"""
    cfg = GENERIC_OPERATION_CONFIG.get(operation_type, {})
    return cfg.get("required_selector_keys", ["submit"])



# ── 投稿・更新監査 ───────────────────────────────────────────────

def _execute_audit(task: dict, media_mapping: Optional[dict]) -> dict:
    payload     = task.get("payload", {})
    check_items = payload.get("check_items", [])

    if not check_items:
        return {
            "status":   BLOCKED,
            "executed": False,
            "message":  "監査対象のcheck_itemsがpayloadに含まれていません。",
        }

    return {
        "status":      WAITING_EXECUTOR,
        "executed":    False,
        "message":     f"監査対象: {', '.join(check_items)}。監査実行層は現在開発中です。",
        "check_items": check_items,
    }


# ── 面接・ヒアリング補佐 ─────────────────────────────────────────

def _execute_interview(task: dict, media_mapping: Optional[dict]) -> dict:
    payload  = task.get("payload", {})
    use_case = payload.get("use_case", "")

    if not use_case:
        return {
            "status":   BLOCKED,
            "executed": False,
            "message":  "ヒアリング用途(use_case)がpayloadに含まれていません。",
        }

    return {
        "status":   WAITING_EXECUTOR,
        "executed": False,
        "message":  f"ヒアリング補佐: {use_case}。実行層は現在開発中です。",
        "use_case": use_case,
    }
