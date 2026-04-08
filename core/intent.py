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

        hist_str = str(history[-3:]) if history else "[]"
        diag_prompt = f"""以下の履歴と最新発言から、ユーザーの精神状態と真の意図をJSONでプロファイリングせよ。
【履歴】{hist_str}
【最新】{current_p}
【出力形式（JSONのみ）】
{{"current_stage":"{r1}/{r2}/{r3}/{r4}のいずれか","true_desire":"本当に求めていること15文字以内","bias":"かかっているバイアス","missing_piece":"{r4}になるために足りない観点","confidence":0.8}}"""

        res = call_llm(
            system_prompt="あなたは深層心理と戦略に精通したプロファイラー。余計な前置き禁止。JSONのみ出力。",
            messages=[{"role":"user","content":diag_prompt}],
            ai_tier="core", max_tokens=512
        )
        m = re.search(r"\{.*\}", res, re.DOTALL)
        state = json.loads(m.group(0)) if m else {}
        if state:
            db.collection("users").document(uid).set({"intent_state": state}, merge=True)
        return state
    except Exception:
        return {}

def generate_query_plan(user_prompt: str, tenant_id: str, level: str) -> dict:
    """入力から最適な検索・要約計画(Query Plan)をJSONで生成"""
    schema = '''{
  "intent":"相談/意思決定/分析/要約/作成/雑談",
  "why":"なぜ今それを聞くのか（1行）",
  "retrieval":{"top_k_total":20,"recency_bias":"high/med/low"},
  "summary_lens":{"preset":"expert/executor/mentor/general","chars":900},
  "output_style":{"format":"結論→根拠→打ち手→KPI","tone":"断定"}
}'''
    prompt = f"【業種】{tenant_id}\n【格】{level}\n【入力】{user_prompt}\n\nJSONスキーマに従い検索計画を生成:\n{schema}"
    try:
        out = call_llm(
            system_prompt="優秀な検索プランナー。JSONのみ出力。余計なテキスト禁止。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=512
        )
        m = re.search(r"\{.*\}", out, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

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
