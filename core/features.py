# api/core/features.py
from api.core.firestore_client import get_db

FEATURE_REGISTRY = {
    "image_generation":        {"label": "画像生成",           "category": "生成",      "default_enabled": True},
    "personal_consulting":     {"label": "個人相談",           "category": "相談",      "default_enabled": True},
    "current_issue_diagnosis": {"label": "現状課題診断",       "category": "診断",      "default_enabled": True},
    "decision_metrics":        {"label": "Decision Metrics",  "category": "診断",      "default_enabled": True},
    "fixed_concept_report":    {"label": "固定概念レポート",   "category": "分析",      "default_enabled": True},
    "ascend_ultra":            {"label": "ASCEND Ultra",      "category": "AIエンジン", "default_enabled": False},
    "ascend_apex":             {"label": "ASCEND Apex",       "category": "AIエンジン", "default_enabled": False},
    "image_gallery":           {"label": "生成画像ギャラリー", "category": "ギャラリー", "default_enabled": False},
    "diag_structure":          {"label": "構造診断タブ",       "category": "診断タブ",  "default_enabled": True},
    "diag_issue":              {"label": "課題仮説タブ",       "category": "診断タブ",  "default_enabled": True},
    "diag_comparison":         {"label": "比較分析タブ",       "category": "診断タブ",  "default_enabled": True},
    "diag_contradiction":      {"label": "矛盾検知タブ",       "category": "診断タブ",  "default_enabled": True},
    "diag_execution":          {"label": "実行計画タブ",       "category": "診断タブ",  "default_enabled": True},
    "diag_investment":         {"label": "投資シグナルタブ",   "category": "診断タブ",  "default_enabled": False},
    "diag_graph":              {"label": "思考マップタブ",     "category": "診断タブ",  "default_enabled": True},
    "diag_file":               {"label": "ファイル診断タブ",   "category": "診断タブ",  "default_enabled": True},
}

# STARTER:0円 / STANDARD:9800円 / PRO:39800円 / APEX:89800円
PLAN_PRICES: dict[str, int] = {
    "starter":  0,
    "standard": 9800,
    "pro":      39800,
    "apex":     89800,
    "ultra_member": 300000,
    "ultra_admin":  300000,
}

PLAN_FEATURE_MAP: dict[str, dict[str, bool]] = {
    "starter": {
        "image_generation":        False,
        "personal_consulting":     False,
        "current_issue_diagnosis": False,
        "decision_metrics":        False,
        "fixed_concept_report":    False,
        "ascend_ultra":            False,
        "ascend_apex":             False,
        "image_gallery":           False,
        "diag_structure":          False,
        "diag_issue":              False,
        "diag_comparison":         False,
        "diag_contradiction":      False,
        "diag_execution":          False,
        "diag_investment":         False,
        "diag_graph":              False,
        "diag_file":               False,
    },
    "standard": {
        "image_generation":        True,
        "personal_consulting":     False,
        "current_issue_diagnosis": True,
        "decision_metrics":        True,
        "fixed_concept_report":    False,
        "ascend_ultra":            False,
        "ascend_apex":             False,
        "image_gallery":           False,
        "diag_structure":          True,
        "diag_issue":              True,
        "diag_comparison":         True,
        "diag_contradiction":      True,
        "diag_execution":          True,
        "diag_investment":         False,
        "diag_graph":              False,
        "diag_file":               False,
    },
    "pro": {
        "image_generation":        True,
        "personal_consulting":     True,
        "current_issue_diagnosis": True,
        "decision_metrics":        True,
        "fixed_concept_report":    True,
        "ascend_ultra":            True,
        "ascend_apex":             False,
        "image_gallery":           True,
        "diag_structure":          True,
        "diag_issue":              True,
        "diag_comparison":         True,
        "diag_contradiction":      True,
        "diag_execution":          True,
        "diag_investment":         False,
        "diag_graph":              True,
        "diag_file":               True,
    },
    "apex": {
        "image_generation":        True,
        "personal_consulting":     True,
        "current_issue_diagnosis": True,
        "decision_metrics":        True,
        "fixed_concept_report":    True,
        "ascend_ultra":            True,
        "ascend_apex":             True,
        "image_gallery":           True,
        "diag_structure":          True,
        "diag_issue":              True,
        "diag_comparison":         True,
        "diag_contradiction":      True,
        "diag_execution":          True,
        "diag_investment":         True,
        "diag_graph":              True,
        "diag_file":               True,
    },
    # ULTRA企業契約: メンバー=PRO相当、管理者=APEX相当
    "ultra_member": {
        "image_generation":        True,
        "personal_consulting":     True,
        "current_issue_diagnosis": True,
        "decision_metrics":        True,
        "fixed_concept_report":    True,
        "ascend_ultra":            True,
        "ascend_apex":             False,
        "image_gallery":           True,
        "diag_structure":          True,
        "diag_issue":              True,
        "diag_comparison":         True,
        "diag_contradiction":      True,
        "diag_execution":          True,
        "diag_investment":         False,
        "diag_graph":              True,
        "diag_file":               True,
    },
    "ultra_admin": {
        "image_generation":        True,
        "personal_consulting":     True,
        "current_issue_diagnosis": True,
        "decision_metrics":        True,
        "fixed_concept_report":    True,
        "ascend_ultra":            True,
        "ascend_apex":             True,
        "image_gallery":           True,
        "diag_structure":          True,
        "diag_issue":              True,
        "diag_comparison":         True,
        "diag_contradiction":      True,
        "diag_execution":          True,
        "diag_investment":         True,
        "diag_graph":              True,
        "diag_file":               True,
    },
}

PLAN_ALLOWED_MODES: dict[str, list[str]] = {
    "starter":  ["auto"],
    "standard": ["auto", "numeric", "growth", "control", "analysis", "planning", "risk"],
    "pro":           [],
    "apex":          [],
    "ultra_member":  [],
    "ultra_admin":   [],
}


def load_user_plan(uid: str) -> str:
    if not uid:
        return "starter"
    try:
        db = get_db()
        snap = db.collection("users").document(uid).get()
        d = (snap.to_dict() or {}) if snap.exists else {}
        plan = d.get("plan") or "starter"
        if plan not in PLAN_FEATURE_MAP:  # ultra_member/ultra_admin も含む
            plan = "starter"
        return plan
    except Exception:
        return "starter"


def get_plan_allowed_modes(uid: str) -> list[str]:
    plan = load_user_plan(uid)
    return list(PLAN_ALLOWED_MODES.get(plan, []))


def load_user_feature_overrides(uid: str) -> dict:
    if not uid:
        return {}
    try:
        db = get_db()
        snap = db.collection("users").document(uid).get()
        d = (snap.to_dict() or {}) if snap.exists else {}
        return d.get("feature_overrides") or {}
    except Exception:
        return {}


def get_effective_feature_flags(uid: str) -> dict:
    plan = load_user_plan(uid)
    plan_flags = PLAN_FEATURE_MAP.get(plan, {})
    overrides = load_user_feature_overrides(uid)
    result = {}
    for fid, reg in FEATURE_REGISTRY.items():
        if fid in overrides:
            result[fid] = bool(overrides[fid])
        elif fid in plan_flags:
            result[fid] = bool(plan_flags[fid])
        else:
            result[fid] = bool(reg.get("default_enabled", True))
    return result


def is_feature_enabled(uid: str, feature_id: str, role: str = "user") -> bool:
    if role == "admin":
        return True
    if not uid or not feature_id:
        return bool(uid is None or feature_id is None)
    if feature_id not in FEATURE_REGISTRY:
        return True
    flags = get_effective_feature_flags(uid)
    return bool(flags.get(feature_id, True))
