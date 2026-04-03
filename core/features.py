# api/core/features.py
# app.py の FEATURE_REGISTRY / get_effective_feature_flags を FastAPI 用に移植

from api.core.firestore_client import get_db

FEATURE_REGISTRY = {
    "image_generation": {
        "label": "画像生成",
        "category": "生成",
        "default_enabled": True,
    },
    "personal_consulting": {
        "label": "個人相談",
        "category": "相談",
        "default_enabled": True,
    },
    "current_issue_diagnosis": {
        "label": "現状課題診断",
        "category": "診断",
        "default_enabled": True,
    },
    "decision_metrics": {
        "label": "Decision Metrics",
        "category": "診断",
        "default_enabled": True,
    },
    "fixed_concept_report": {
        "label": "固定概念レポート",
        "category": "分析",
        "default_enabled": True,
    },
    "ascend_ultra": {
        "label": "ASCEND Ultra",
        "category": "AIエンジン",
        "default_enabled": False,
    },
    "ascend_apex": {
        "label": "ASCEND Apex",
        "category": "AIエンジン",
        "default_enabled": False,
    },
    "diag_structure": {
        "label": "構造診断タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_issue": {
        "label": "課題仮説タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_comparison": {
        "label": "比較分析タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_contradiction": {
        "label": "矛盾検知タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_execution": {
        "label": "実行計画タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_investment": {
        "label": "投資シグナルタブ",
        "category": "診断タブ",
        "default_enabled": False,
    },
    "diag_graph": {
        "label": "思考マップタブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
    "diag_file": {
        "label": "ファイル診断タブ",
        "category": "診断タブ",
        "default_enabled": True,
    },
}


def load_user_feature_overrides(uid: str) -> dict:
    """Firestore users/{uid}.feature_overrides を返す。失敗時は {}"""
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
    """全 feature_id について有効/無効を解決して返す dict。
    優先順位: user override → registry default。
    """
    overrides = load_user_feature_overrides(uid)
    result = {}
    for fid, reg in FEATURE_REGISTRY.items():
        if fid in overrides:
            result[fid] = bool(overrides[fid])
        else:
            result[fid] = bool(reg.get("default_enabled", True))
    return result


def is_feature_enabled(uid: str, feature_id: str, role: str = "user") -> bool:
    """feature_id が uid に対して有効かを返す。admin は常に True。"""
    if role == "admin":
        return True
    if not uid or not feature_id:
        return bool(uid is None or feature_id is None)
    if feature_id not in FEATURE_REGISTRY:
        return True  # 未定義機能は許可（後方互換）
    flags = get_effective_feature_flags(uid)
    return bool(flags.get(feature_id, True))
