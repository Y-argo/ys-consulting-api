# api/core/intent.py - 脳内カルテ・QueryPlan・SummaryLens
import os, re, json, datetime
from google.cloud import firestore as fs
from api.core.firestore_client import get_db, DEFAULT_TENANT
from api.core.llm_client import call_llm

SUMMARY_PRESETS = ["expert","executor","mentor","general"]

def update_user_intent_state(uid: str, tenant_id: str, history: list, current_p: str) -> dict:
    """脳内カルテ（深層プロファイル）を診断・更新してFirestoreに保存"""
    try:
        db = get_db()
        cfg_doc = None
        for tid in [tenant_id, DEFAULT_TENANT]:
            snap = db.collection("system_settings").document(f"rank_config_{tid}").get()
            if snap.exists:
                cfg_doc = snap.to_dict() or {}
                break
        r1 = (cfg_doc or {}).get("rank_1_name","追従者")
        r2 = (cfg_doc or {}).get("rank_2_name","実行者")
        r3 = (cfg_doc or {}).get("rank_3_name","戦略家")
        r4 = (cfg_doc or {}).get("rank_4_name","設計者")

        q = (current_p or "").lower()
        if any(w in q for w in ["構造","設計","アーキテクチャ","最適化","支配","力学"]):
            stage = r4
        elif any(w in q for w in ["戦略","差別化","競合","ポジション","kpi","roi"]):
            stage = r3
        elif any(w in q for w in ["実行","手順","タスク","改善","運用","効率"]):
            stage = r2
        else:
            stage = r1
        if any(w in q for w in ["売上","収益","利益","稼ぎ","収入"]):
            desire = "収益を最大化したい"
        elif any(w in q for w in ["競合","勝ち","差別化","優位"]):
            desire = "競争優位を確立したい"
        elif any(w in q for w in ["効率","時間","コスト","削減"]):
            desire = "業務を効率化したい"
        else:
            desire = "課題を解決したい"
        if any(w in q for w in ["絶対","必ず","確実","間違いない"]):
            bias = "確証バイアス"
        elif any(w in q for w in ["不安","怖い","リスク","失敗"]):
            bias = "損失回避バイアス"
        else:
            bias = "現状維持バイアス"
        state = {
            "current_stage": stage,
            "true_desire": desire,
            "bias": bias,
            "missing_piece": f"{r4}視点での構造把握",
            "confidence": 0.7,
        }
        if state:
            db.collection("users").document(uid).set({"intent_state": state}, merge=True)
        return state
    except Exception:
        return {}

def generate_query_plan(user_prompt: str, tenant_id: str, level: str) -> dict:
    """入力から最適な検索・要約計画をキーワードベースで生成（LLM不使用）"""
    q = (user_prompt or "").lower()
    # intent判定
    intent = "相談"
    if any(w in q for w in ["分析","解析","なぜ","原因","要因","調べ"]):
        intent = "分析"
    elif any(w in q for w in ["どうすべき","どちら","選択","判断","決め","べきか"]):
        intent = "意思決定"
    elif any(w in q for w in ["まとめ","要約","整理","概要"]):
        intent = "要約"
    elif any(w in q for w in ["作成","書いて","作って","生成","作り"]):
        intent = "作成"
    elif any(w in q for w in ["こんにちは","ありがとう","おはよう","こんばん","やあ","どうも"]):
        intent = "雑談"
    # summary_lens判定
    preset = "expert"
    if any(w in q for w in ["手順","実装","チェック","運用","具体","todo"]):
        preset = "executor"
    elif any(w in q for w in ["習慣","訓練","メンタル","マインド","継続","成長"]):
        preset = "mentor"
    elif any(w in q for w in ["概要","まとめ","全体","要点","要約"]):
        preset = "general"
    return {
        "intent": intent,
        "why": "",
        "retrieval": {"top_k_total": 20, "recency_bias": "med"},
        "summary_lens": {"preset": preset, "chars": 900},
        "output_style": {"format": "結論→根拠→打ち手→KPI", "tone": "断定"},
    }

def lgbm_select_summary_lens(query: str, chosen_mode: str, default_preset: str = "expert") -> tuple:
    """summary preset と hierarchy をキーワードベースで選択"""
    q = (query or "").lower()
    preset = default_preset or "expert"

    exec_kw = ["手順","実装","チェック","運用","具体","todo"]
    mentor_kw = ["習慣","訓練","メンタル","マインド","継続","成長"]
    expert_kw = ["構造","戦略","最適化","原因","支配","力学"]
    general_kw = ["概要","まとめ","全体","要点","要約"]

    scores = {
        "executor": sum(1 for w in exec_kw if w in q),
        "mentor":   sum(1 for w in mentor_kw if w in q),
        "expert":   sum(1 for w in expert_kw if w in q),
        "general":  sum(1 for w in general_kw if w in q),
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        preset = best

    # hierarchy
    if any(w in q for w in ["要約","まとめ","サマリー"]):
        hier = "prefer_summary"
    else:
        hier = "raw"

    return preset, hier
