"""
selector_rank_trainer.py
P16.7: Selector ranking専用学習データ生成・将来LightGBM差し替え可能interface。
stub実装。train/predict interfaceを固定する。
自動適用禁止・承認制維持。
"""
from __future__ import annotations
import datetime
from typing import Any

# ── ranking model version管理 ──────────────────────────────────────────────
RANKING_MODEL_VERSION = "p16_v1"


def train_selector_rank_model(
    training_logs: list[dict],
    model_version: str = RANKING_MODEL_VERSION,
) -> dict:
    """
    selector_training_logs から学習データを受け取りモデルを訓練する。
    現在はstub（rule-basedをそのまま返す）。
    将来: LightGBM / XGBoost / embedding rerank に差し替え可能。

    Args:
        training_logs: selector_training_logs ドキュメントのリスト
        model_version:  保存するバージョン文字列

    Returns:
        {
            "model_version": str,
            "trained_at": ISO8601 str,
            "sample_count": int,
            "positive_count": int,
            "negative_count": int,
            "status": "stub" | "trained",
            "metrics": dict,
        }
    """
    positive = [r for r in training_logs if r.get("success") and r.get("executed")]
    negative = [r for r in training_logs if not r.get("success") or not r.get("executed")]

    return {
        "model_version":  model_version,
        "trained_at":     datetime.datetime.utcnow().isoformat(),
        "sample_count":   len(training_logs),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "status":         "stub",
        "metrics":        {},
    }


def predict_selector_rank(
    feature_vector: dict,
    model_version: str = RANKING_MODEL_VERSION,
) -> dict:
    """
    feature_vectorからmodel_scoreを予測する。
    現在はstub（0.5固定）。
    将来: 学習済みLightGBMモデルをロードして推論。

    Args:
        feature_vector: build_selector_feature_vector() の出力
        model_version:  使用するモデルバージョン

    Returns:
        {
            "model_score": float,       # 0.0〜1.0
            "confidence":  float,       # 0.0〜1.0（データ不足時は低め）
            "model_version": str,
            "is_stub": bool,
        }
    """
    # stub: rule-based score をmodel_scoreとして返す
    # 将来はここでpickle/joblib loadしてpredict()を呼ぶ
    score = _stub_model_score(feature_vector)
    confidence = _stub_confidence(feature_vector)

    return {
        "model_score":    round(score, 3),
        "confidence":     round(confidence, 3),
        "model_version":  model_version,
        "is_stub":        True,
    }


# ── internal stub helpers ──────────────────────────────────────────────────

def _stub_model_score(fv: dict) -> float:
    """stub: feature vectorから簡易スコアを算出。"""
    score = 0.5
    score += fv.get("is_id_selector",       0) * 0.15
    score += fv.get("is_name_selector",     0) * 0.12
    score += fv.get("is_aria_selector",     0) * 0.10
    score += fv.get("semantic_confidence",  0.2) * 0.20
    score += fv.get("historical_success_rate", 0) * 0.25
    score += fv.get("verify_success_rate",  0) * 0.10
    score += fv.get("recent_success_decay", 0) * 0.10
    score -= fv.get("is_class_only",        0) * 0.10
    score -= fv.get("has_nth_child",        0) * 0.15
    score -= fv.get("historical_timeout_rate", 0) * 0.20
    return max(0.0, min(1.0, score))


def _stub_confidence(fv: dict) -> float:
    """
    統計件数が少ない場合はconfidenceを下げる。
    usage_frequency が低いほど confidence も低い。
    """
    usage_freq = fv.get("usage_frequency", 0.0)
    hist_rate  = fv.get("historical_success_rate", 0.0)

    if usage_freq < 0.02:   # usage_count < 2件相当
        return 0.30
    if usage_freq < 0.05:   # usage_count < 5件相当
        return 0.50
    if usage_freq < 0.10:
        return 0.65
    # データが十分ある場合は historical_success_rate で補正
    return min(0.95, 0.70 + hist_rate * 0.25)
