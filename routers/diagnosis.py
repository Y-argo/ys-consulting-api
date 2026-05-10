# api/routers/diagnosis.py
import datetime
_JST = datetime.timezone(datetime.timedelta(hours=9))
def _now_jst(): return datetime.datetime.now(_JST)
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Form
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import firestore as fs

from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT
from api.core.llm_client import call_llm

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])

def _load_rank_config(tenant_id: str) -> dict:
    try:
        db = get_db()
        doc = db.collection("tenant_settings").document(tenant_id).get()
        if doc.exists:
            d = doc.to_dict() or {}
            return {
                "rank_1_name": d.get("rank_1_name", "D"),
                "rank_2_name": d.get("rank_2_name", "C"),
                "rank_3_name": d.get("rank_3_name", "B"),
                "rank_4_name": d.get("rank_4_name", "A"),
            }
    except Exception:
        pass
    return {"rank_1_name": "D", "rank_2_name": "C", "rank_3_name": "B", "rank_4_name": "A"}

def _load_chat_history_across_sessions(uid: str, tenant_id: str, limit: int = 30) -> list:
    db = get_db()
    try:
        sessions_raw = list(
            db.collection("chat_sessions")
            .where("uid", "==", uid)
            .limit(50)
            .stream()
        )
        sessions_raw.sort(key=lambda s: str((s.to_dict() or {}).get("updated_at", "")), reverse=True)
        sessions = sessions_raw[:10]
        msgs = []
        for s in sessions:
            ref = db.collection("chat_sessions").document(s.id).collection("messages")
            for m in ref.order_by("ts").limit_to_last(10).get():
                d = m.to_dict() or {}
                msgs.append({"role": d.get("role", "user"), "content": d.get("content", "")})
            if len(msgs) >= limit:
                break
        return msgs[-limit:]
    except Exception:
        return []

def _load_score_config(tenant_id: str) -> dict:
    db = get_db()
    DEFAULT = {
        "struct_words": "構造,資本,市場,制度,最適,期待値,確率,アーキテクチャ,設計,フレームワーク",
        "strategy_words": "戦略,施策,優先,差別化,競合,ポジショニング,KPI,ROI,目標",
        "exec_words": "実行,手順,タスク,スケジュール,チェック,改善,運用,効率",
        "emotion_words": "不安,ムカつく,なぜ俺,怖い,どうせ,無理,クソ,無能,イライラ,最悪",
    }
    try:
        for tid in [tenant_id, "default"]:
            doc = db.collection("system_settings").document(f"score_config_{tid}").get()
            if doc.exists:
                d = doc.to_dict() or {}
                return {**DEFAULT, **d}
    except Exception:
        pass
    return DEFAULT

def _generate_diagnosis(uid: str, tenant_id: str, n_chats: int = 30) -> str:
    rank_cfg = _load_rank_config(tenant_id)
    msgs = _load_chat_history_across_sessions(uid, tenant_id, limit=n_chats)
    chat_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in msgs if m.get("content")])
    if not chat_text.strip():
        return ""
    timestamp_str = _now_jst().strftime("%Y-%m-%d %H:%M")
    dr1 = rank_cfg["rank_1_name"]
    dr2 = rank_cfg["rank_2_name"]
    dr3 = rank_cfg["rank_3_name"]
    dr4 = rank_cfg["rank_4_name"]
    prompt = f"""以下はユーザー「{uid}」の直近チャット履歴（{n_chats}件）です。

{chat_text}

---
【ランク体系（低→高）】: {dr1} → {dr2} → {dr3} → {dr4}

上記の履歴を踏まえ、以下のフォーマットを厳守して「現状課題診断レポート」を生成してください。

# 現状課題診断レポート
生成日時: {timestamp_str}
対象ユーザー: {uid}
解析範囲: 直近チャット {n_chats} 件
---
## 総合評価
評価ランク: [S/A/B/C/D]
現状状態: [1〜3行で要約]
優先改善度: [低/中/高/緊急]
---
## 主要課題（最大3件）
### 課題1: [課題名]
影響度: [高/中/低]
原因: [構造的原因]
推奨行動: [具体行動]
---
### 課題2: [課題名]
影響度: [高/中/低]
原因: [構造的原因]
推奨行動: [具体行動]
---
### 課題3: [課題名]
影響度: [高/中/低]
原因: [構造的原因]
推奨行動: [具体行動]
---
## 強み・弱点
強み: [維持すべき行動]
弱点: [改善が必要な行動]
"""
    try:
        return call_llm(
            system_prompt="あなたは戦略コンサルタントです。与えられたチャット履歴を分析し、構造的な課題診断レポートを生成してください。",
            messages=[{"role": "user", "content": prompt}],
            ai_tier="core",
            max_tokens=4096,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"診断生成エラー: {e}")

def _save_diagnosis(uid: str, tenant_id: str, report_md: str, rank: str = None, n_chats: int = None) -> str:
    db = get_db()
    doc_id = str(uuid.uuid4())
    data = {
        "uid": uid,
        "tenant_id": tenant_id,
        "report_md": report_md,
        "created_at": fs.SERVER_TIMESTAMP,
    }
    if rank: data["rank"] = rank
    if n_chats: data["n_chats"] = n_chats
    db.collection("user_diagnoses").document(doc_id).set(data)
    return doc_id

def _load_diagnoses(uid: str, tenant_id: str, limit: int = 5) -> list:
    db = get_db()
    try:
        docs = list(
            db.collection("user_diagnoses")
            .where("uid", "==", uid)
            .limit(50)
            .stream()
        )
        result = []
        for d in docs:
            data = d.to_dict() or {}
            result.append({
                "doc_id": d.id,
                "report_md": data.get("report_md", ""),
                "created_at": str(data.get("created_at", "")),
                "rank": data.get("rank"),
                "n_chats": data.get("n_chats"),
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result[:limit]
    except Exception:
        return []

class DiagnosisRequest(BaseModel):
    n_chats: int = 30

@router.post("/generate")
def generate_diagnosis(req: DiagnosisRequest, payload: dict = Depends(verify_token)):
    from api.core.features import is_feature_enabled
    uid = payload["uid"]; tenant_id = payload.get("tenant_id","default")
    if not is_feature_enabled(uid, "current_issue_diagnosis"):
        raise HTTPException(status_code=403, detail="この機能は利用できません")
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    report_md = _generate_diagnosis(uid, tenant_id, req.n_chats)
    if not report_md:
        raise HTTPException(status_code=400, detail="チャット履歴が不足しています")
    # ランク取得
    try:
        from api.routers.user_stats import _load_rank_config, _score_to_rank
        from api.core.firestore_client import get_db as _gdb2
        _snap = _gdb2().collection("users").document(uid).get()
        _score = int((_snap.to_dict() or {}).get("level_score", 0)) if _snap.exists else 0
        _cfg = _load_rank_config(tenant_id)
        _rank = _score_to_rank(_score, _cfg)
    except Exception:
        _rank = None
    doc_id = _save_diagnosis(uid, tenant_id, report_md, rank=_rank, n_chats=req.n_chats)
    return {"doc_id": doc_id, "report_md": report_md}

@router.get("/list")
def list_diagnoses(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    return {"diagnoses": _load_diagnoses(uid, tenant_id)}

@router.get("/thought_map")
def get_thought_map(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        sessions_raw = list(db.collection("chat_sessions").where("uid","==",uid).limit(200).stream())
        sessions_raw.sort(key=lambda s: str((s.to_dict() or {}).get("updated_at","")), reverse=True)

        TOPICS = {
            "戦略・競合":   ["戦略","競合","差別化","ポジション","市場","シェア","ブランド","参入","撤退","優位"],
            "集客・SNS":   ["集客","SNS","Instagram","Twitter","広告","フォロワー","投稿","認知","バズ","LP"],
            "売上・財務":   ["売上","収益","利益","資金","コスト","価格","単価","資産","融資","キャッシュ"],
            "組織・人材":   ["採用","チーム","組織","スタッフ","教育","マネジメント","育成","離職","評価"],
            "投資・株":     ["投資","株","銘柄","シグナル","相場","資産","ポートフォリオ","底打ち","売り"],
            "診断・分析":   ["診断","分析","レポート","スコア","評価","課題","ボトルネック","原因"],
            "指名・接客":   ["指名","接客","お客様","キャスト","リピート","コミュニケーション"],
            "マーケ・集客": ["マーケ","宣材","写真","予約","ネット予約","口コミ","レビュー","集患"],
            "計画・実行":   ["計画","ロードマップ","スケジュール","手順","ステップ","期限","タスク"],
            "リスク・危機": ["リスク","危機","失敗","損失","トラブル","クレーム","炎上","最悪"],
            "成長・習慣":   ["成長","スキル","習慣","訓練","学習","キャリア","目標","継続"],
            "交渉・説得":   ["交渉","説得","条件","合意","契約","提案","伝え方","折衝"],
            "思考・構造":   ["構造","フレームワーク","ロジック","仮説","論点","因果","MECE"],
        }

        def classify(text):
            for topic, keywords in TOPICS.items():
                if any(k in text for k in keywords):
                    return topic
            return "その他"

        theme_count = {}
        theme_sessions = {}
        all_prompts = []
        raw_nodes = []
        node_set = set()

        for s in sessions_raw[:20]:
            try:
                s_data = s.to_dict() or {}
                s_date = str(s_data.get("updated_at",""))[:10]
                msgs = list(db.collection("chat_sessions").document(s.id).collection("messages").limit(40).stream())
                msgs.sort(key=lambda m: str((m.to_dict() or {}).get("ts","")))
                for m in msgs:
                    d = m.to_dict() or {}
                    if d.get("role") == "user":
                        content = (d.get("content","") or "").strip()
                        if not content or len(content) <= 3: continue
                        all_prompts.append(content)
                        topic = classify(content)
                        theme_count[topic] = theme_count.get(topic, 0) + 1
                        if topic not in theme_sessions:
                            theme_sessions[topic] = []
                        if s_date not in theme_sessions[topic]:
                            theme_sessions[topic].append(s_date)
                        label = content[:35]
                        if label not in node_set:
                            node_set.add(label)
                            raw_nodes.append({"id": label, "label": label, "group": topic, "full_text": content[:200]})
            except Exception:
                continue

        raw_nodes = raw_nodes[:40]

        unresolved_alerts = [
            {"topic": t, "count": c, "message": f"「{t}」に関する相談が{c}回繰り返されています。根本解決が必要な可能性があります。"}
            for t, c in theme_count.items() if c >= 3 and t != "その他"
        ]
        unresolved_alerts.sort(key=lambda x: x["count"], reverse=True)

        growth_trend = [
            {"topic": t, "session_count": len(dates), "last_date": max(dates) if dates else ""}
            for t, dates in theme_sessions.items() if t != "その他"
        ]
        growth_trend.sort(key=lambda x: x["session_count"], reverse=True)

        topic_last = {}
        edges = []
        for n in raw_nodes:
            g = n["group"]
            if g in topic_last:
                edges.append({"from": topic_last[g], "to": n["id"], "topic": g})
            topic_last[g] = n["id"]

        topics_used = list(set(n["group"] for n in raw_nodes))
        center_nodes = [{"id": f"__topic_{t}__", "label": t, "group": t, "is_center": True} for t in topics_used]
        center_edges = [{"from": f"__topic_{n['group']}__", "to": n["id"], "topic": n["group"]} for n in raw_nodes]
        all_nodes = center_nodes + raw_nodes
        all_edges = center_edges

        issue_tree = {}
        try:
            if all_prompts:
                from api.core.llm_client import call_llm as _cllm
                import json as _json, re as _re
                _sample = "\n".join(all_prompts[:30])[:3000]
                _tree_prompt = f"""以下はユーザーの直近チャット相談内容です。コンサルタントとして課題構造を分析せよ。JSONのみ出力。

{_sample}

以下のJSONスキーマで返せ:
{{
  "root_issues": ["根本課題1", "根本課題2"],
  "surface_issues": ["表面的課題1", "表面的課題2", "表面的課題3"],
  "recurring_patterns": ["繰り返しパターン1", "繰り返しパターン2"],
  "growth_opportunities": ["成長機会1", "成長機会2"],
  "priority_action": "最優先で取り組むべきこと（1文）"
}}"""
                _raw = _cllm(
                    system_prompt="戦略コンサルタント。JSONのみ出力。",
                    messages=[{"role":"user","content":_tree_prompt}],
                    ai_tier="core", max_tokens=1024
                )
                _m = _re.search(r"\{.*\}", _raw, _re.DOTALL)
                if _m:
                    issue_tree = _json.loads(_m.group(0))
        except Exception:
            pass

        return {
            "nodes": all_nodes,
            "edges": all_edges,
            "topics": topics_used,
            "unresolved_alerts": unresolved_alerts[:5],
            "growth_trend": growth_trend[:8],
            "issue_tree": issue_tree,
            "theme_count": theme_count,
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

class ConsultRequest(BaseModel):
    analysis_type: str
    input_text: str
    supplement: str = ""
    options: str = ""
    strategy: str = ""
    policy: str = ""

@router.post("/consult")
def run_consult(req: ConsultRequest, payload: dict = Depends(verify_token)):
    from api.core.llm_client import call_llm
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()

    frameworks = []
    try:
        docs = list(db.collection("tenants").document(tenant_id).collection("consulting_frameworks").where("active","==",True).stream())
        frameworks = [d.to_dict().get("name","") for d in docs if d.to_dict().get("name")]
    except Exception:
        pass

    fw_str = "、".join(frameworks[:5]) if frameworks else "MECE・SWOT・3C・ロジックツリー・Issue Tree"

    if req.analysis_type == "structure":
        prompt = f"""あなたは戦略コンサルタントです。以下の相談内容を構造診断してください。
適用フレームワーク: {fw_str}

【相談内容】
{req.input_text}

【補足情報】
{req.supplement or "（なし）"}

【共通ルール】
- 感想ではなく分析を返すこと
- 抽象語だけで逃げないこと
- 不明点はmissing_informationに格納すること
- JSON以外の余計な文を絶対に返さないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

以下のJSONスキーマで返してください:
{{
  "issue_summary": "問題の要約（1〜2文）",
  "observations": ["観測事実1", "観測事実2"],
  "surface_causes": ["表層原因1", "表層原因2"],
  "root_causes": ["根因1", "根因2"],
  "constraints": ["制約1", "制約2"],
  "priority_points": ["優先論点1", "優先論点2"],
  "recommended_actions": ["打ち手1（優先度高）", "打ち手2", "打ち手3"],
  "risks": ["リスク1", "リスク2"],
  "missing_information": ["不足情報1", "不足情報2"]
}}"""

    elif req.analysis_type == "issue":
        prompt = f"""あなたは戦略コンサルタントです。以下の内容から論点・仮説を設計してください。
適用フレームワーク: {fw_str}

【入力内容】
{req.input_text}

【共通ルール】
- 感想ではなく分析を返すこと
- JSON以外の余計な文を絶対に返さないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

以下のJSONスキーマで返してください:
{{
  "main_issues": ["主要論点1", "主要論点2"],
  "hypotheses": ["仮説1", "仮説2"],
  "questions_to_verify": ["次に確認すべき質問1", "質問2"],
  "required_data": ["必要なデータ1", "データ2"],
  "decision_points": ["意思決定ポイント1", "ポイント2"]
}}"""

    elif req.analysis_type == "comparison":
        prompt = f"""あなたは戦略コンサルタントです。以下の複数案を比較分析してください。

【比較対象案】
{req.options or req.input_text}

【追加コンテキスト】
{req.supplement or "（なし）"}

【共通ルール】
- JSON以外の余計な文を絶対に返さないこと
- スコアは1〜5の整数で評価すること（5が最良）
- 最終推奨案を1つ返すこと
- 出力は必ず有効なJSONオブジェクトのみとすること

以下のJSONスキーマで返してください:
{{
  "comparison_axes": ["収益性", "実行難易度", "初期コスト", "回収期間", "リスク"],
  "options": [
    {{
      "name": "案の名前",
      "scores": {{"収益性": 0, "実行難易度": 0, "初期コスト": 0, "回収期間": 0, "リスク": 0}},
      "pros": ["長所1", "長所2"],
      "cons": ["短所1", "短所2"],
      "recommended_for": ["この案が向いているケース"]
    }}
  ],
  "final_recommendation": "最終推奨案と理由"
}}"""

    elif req.analysis_type == "contradiction":
        prompt = f"""あなたは戦略コンサルタントです。以下の内容から矛盾・齟齬を検出してください。

【戦略文】
{req.strategy or req.input_text}

【方針文】
{req.policy or req.supplement or "（なし）"}

【共通ルール】
- JSON以外の余計な文を絶対に返さないこと
- 矛盾がなければcontradictionsを空配列にすること
- 出力は必ず有効なJSONオブジェクトのみとすること

以下のJSONスキーマで返してください:
{{
  "contradictions": [
    {{
      "type": "矛盾の種類（例: 目的手段衝突、KPIズレ、前提矛盾）",
      "description": "矛盾の具体的な説明",
      "why_problematic": "なぜ問題か",
      "fix_direction": "修正方向"
    }}
  ],
  "consistency_score": 70,
  "overall_assessment": "総合評価"
}}"""

    elif req.analysis_type == "execution":
        prompt = f"""あなたは戦略コンサルタントです。以下の内容から実行プランを作成してください。
適用フレームワーク: {fw_str}

【内容】
{req.input_text}

【共通ルール】
- JSON以外の余計な文を絶対に返さないこと
- 優先度はhigh / medium / lowで分類すること
- 出力は必ず有効なJSONオブジェクトのみとすること

以下のJSONスキーマで返してください:
{{
  "action_plan": [
    {{
      "task": "タスク名",
      "owner": "担当者・部門",
      "deadline": "期限の目安",
      "kpi": "成功指標",
      "priority": "high"
    }}
  ]
}}"""
    else:
        return {"ok": False, "error": f"不明なanalysis_type: {req.analysis_type}"}

    try:
        import re, json as _json
        res = call_llm(
            system_prompt="あなたは戦略コンサルタントです。必ず有効なJSONオブジェクトのみ返してください。説明文・前置き・Markdownコードブロック・余計なテキストは一切禁止です。最初の文字は必ず{で始めてください。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=2048
        )
        res_clean = res.strip()
        if res_clean.startswith("```"):
            import re as _re2
            res_clean = _re2.sub(r"^```[a-z]*\n?", "", res_clean)
            res_clean = _re2.sub(r"```$", "", res_clean).strip()
        m = re.search(r"\{.*\}", res_clean, re.DOTALL)
        result = _json.loads(m.group(0)) if m else {"raw": res}

        db.collection("tenants").document(tenant_id).collection("consulting_analyses").add({
            "uid": uid, "tenant_id": tenant_id,
            "analysis_type": req.analysis_type,
            "input_text": req.input_text[:500],
            "result": result,
            "created_at": _now_jst().isoformat(),
        })
        return {"ok": True, "result": result, "analysis_type": req.analysis_type}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.get("/consult/history")
def get_consult_history(analysis_type: str = "", payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()
    try:
        docs = list(
            db.collection("tenants").document(tenant_id)
            .collection("consulting_analyses")
            .where("uid","==",uid)
            .limit(20)
            .stream()
        )
        analyses = [d.to_dict() for d in docs]
        if analysis_type:
            analyses = [a for a in analyses if a.get("analysis_type") == analysis_type]
        analyses.sort(key=lambda x: x.get("created_at",""), reverse=True)
        return {"analyses": analyses}
    except Exception:
        return {"analyses": []}

@router.get("/frameworks")
def get_frameworks(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()
    try:
        docs = list(db.collection("tenants").document(tenant_id).collection("consulting_frameworks").stream())
        fw = [d.to_dict() for d in docs]
        if not fw:
            fw = [
                {"name":"MECE","description":"ダブりなく、漏れなく","active":True},
                {"name":"SWOT","description":"強み・弱み・機会・脅威","active":True},
                {"name":"3C","description":"顧客・競合・自社","active":True},
                {"name":"ロジックツリー","description":"問題を論理的に分解","active":True},
                {"name":"Issue Tree","description":"課題を階層的に整理","active":True},
            ]
        return {"frameworks": fw}
    except Exception:
        return {"frameworks": []}

@router.post("/weekly_report")
def generate_weekly_report(body: dict = {}, payload: dict = Depends(verify_token)):
    from api.core.llm_client import call_llm_pro as _clp2
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()
    n_chats = int((body or {}).get("n_chats", 30))
    try:
        history = _load_chat_history_across_sessions(uid, tenant_id, n_chats)
        if not history:
            return {"ok": False, "error": "チャット履歴がありません"}
        chat_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in history if m.get("content")])[:4000]
        ts = _now_jst().strftime("%Y-%m-%d %H:%M")
        prompt = f"""以下はユーザーのチャット履歴です。週次戦術レポートを生成してください。

{chat_text}

# 週次戦術レポート
生成日時: {ts}
---
## 今週の主要相談テーマ
[最も多かった相談テーマ上位3件]
---
## 意思決定パターン分析
[どのような意思決定が多かったか]
---
## 成長トレンド
[前週比での変化・改善点]
---
## 来週の優先アクション
[最重要アクション3件]
---
## 戦略的提言
[中長期視点での提言]
"""
        report = _clp2(
            system_prompt="あなたは戦略コンサルタント。週次レポートを指定フォーマット厳守で生成せよ。",
            messages=[{"role":"user","content":prompt}],
            max_tokens=3000,
        )
        db.collection("weekly_reports").add({
            "uid": uid, "tenant_id": tenant_id,
            "report_md": report,
            "created_at": _now_jst().isoformat(),
        })
        return {"ok": True, "report_md": report}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.get("/weekly_report/list")
def list_weekly_reports(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()
    try:
        docs = list(
            db.collection("weekly_reports")
            .where("uid","==",uid)
            .where("tenant_id","==",tenant_id)
            .limit(10)
            .stream()
        )
        reports = [d.to_dict() for d in docs]
        reports.sort(key=lambda x: x.get("created_at",""), reverse=True)
        return {"reports": reports}
    except Exception:
        return {"reports": []}


@router.post("/file_diagnosis")
async def file_diagnosis(
    file: UploadFile = File(...),
    answer_context: str = Form(""),
    payload: dict = Depends(verify_token)
):
    """ファイル全タブ横断診断→構造診断・課題仮説・実行計画を一括生成"""
    from api.core.llm_client import call_llm as _cllm
    import io, re as _re, json as _json

    filename = file.filename or "file"
    content = await file.read()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 全シート抽出
    sheets_text = {}
    try:
        if ext in ("xlsx", "xls"):
            import pandas as pd
            xf = pd.ExcelFile(io.BytesIO(content))
            for sheet in xf.sheet_names:
                try:
                    df_raw = xf.parse(sheet, header=None).dropna(how="all").dropna(axis=1, how="all")
                    sheets_text[sheet] = df_raw.to_csv(index=False, header=False)[:4000] if not df_raw.empty else f"({sheet}:データなし)"
                except Exception as _e:
                    sheets_text[sheet] = f"({sheet}:読み込みエラー:{_e})"
        elif ext == "ods":
            import pandas as pd
            xf = pd.ExcelFile(io.BytesIO(content), engine="odf")
            for sheet in xf.sheet_names:
                try:
                    df_raw = xf.parse(sheet, header=None).dropna(how="all").dropna(axis=1, how="all")
                    sheets_text[sheet] = df_raw.to_csv(index=False, header=False)[:4000] if not df_raw.empty else f"({sheet}:データなし)"
                except Exception as _e:
                    sheets_text[sheet] = f"({sheet}:読み込みエラー:{_e})"
        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            sheets_text["Sheet1"] = df.to_csv(index=False)[:6000]
        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content))
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
                sheets_text["PDF"] = text[:6000]
            except Exception:
                sheets_text["PDF"] = content.decode("utf-8", errors="ignore")[:6000]
        else:
            sheets_text["TEXT"] = content.decode("utf-8", errors="ignore")[:6000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイル読み込みエラー: {e}")

    if not sheets_text:
        raise HTTPException(status_code=400, detail="ファイルの内容を読み取れませんでした")

    # 数式・グラフ・条件付き書式・ピボットテーブル読み込み
    formula_summary = ""
    if ext in ("xlsx", "xls"):
        try:
            import openpyxl as _opxl
            wb = _opxl.load_workbook(io.BytesIO(content), data_only=False)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                section = ""
                # 全数式取得（上限なし）
                formulas = []
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.value and isinstance(cell.value, str) and cell.value.startswith("="):
                            formulas.append(f"{cell.coordinate}: {cell.value}")
                if formulas:
                    section += f"\n【{sheet_name}】数式({len(formulas)}件):\n" + "\n".join(formulas)
                # 条件付き書式
                cf_list = []
                for cf_range, cf_rules in ws.conditional_formatting._cf_rules.items():
                    for rule in cf_rules:
                        rule_type = getattr(rule, "type", "")
                        formula_val = getattr(rule, "formula", [])
                        cf_list.append(f"範囲{cf_range}: タイプ={rule_type} 条件={formula_val}")
                if cf_list:
                    section += f"\n【{sheet_name}】条件付き書式({len(cf_list)}件):\n" + "\n".join(cf_list[:20])
                # グラフ
                charts = []
                for chart in ws._charts:
                    title = ""
                    try: title = chart.title.tx.rich.p[0].r[0].t
                    except Exception: pass
                    chart_type = type(chart).__name__
                    charts.append(f"グラフ種別={chart_type} タイトル={title or '(無題)'}")
                if charts:
                    section += f"\n【{sheet_name}】グラフ({len(charts)}件):\n" + "\n".join(charts)
                # ピボットテーブル
                pivots = []
                for pt in getattr(ws, "_pivots", []):
                    pt_name = getattr(pt, "name", "")
                    pivots.append(f"ピボット名={pt_name}")
                if pivots:
                    section += f"\n【{sheet_name}】ピボットテーブル({len(pivots)}件):\n" + "\n".join(pivots)
                if section:
                    formula_summary += section
        except Exception as _fe:
            formula_summary = f"（xlsx詳細読み込みエラー: {_fe}）"
    elif ext == "ods":
        try:
            from odf.opendocument import load as _odf_load
            from odf.table import Table, TableRow, TableCell
            doc = _odf_load(io.BytesIO(content))
            for sheet in doc.spreadsheet.getElementsByType(Table):
                sheet_name = sheet.getAttribute("name")
                formulas = []
                for row in sheet.getElementsByType(TableRow):
                    for cell in row.getElementsByType(TableCell):
                        formula = cell.getAttribute("formula")
                        if formula:
                            formulas.append(f"{sheet_name}: {formula}")
                if formulas:
                    formula_summary += f"\n【{sheet_name}】数式({len(formulas)}件):\n" + "\n".join(formulas)
        except Exception as _fe:
            formula_summary = f"（ods数式読み込みエラー: {_fe}）"


    # 全シートを結合
    combined = ""
    for sheet, text in sheets_text.items():
        combined += f"\n\n【シート: {sheet}】\n{text}"
    combined = combined[:12000]

    # ===== ③ Python先行数値分析 =====
    import numpy as _np
    import re as _re2, json as _json2

    numeric_analysis = {}
    for sheet, text in sheets_text.items():
        try:
            import pandas as _pd, io as _io
            lines = text.strip().split("\n")
            if len(lines) < 2:
                continue
            df = _pd.read_csv(_io.StringIO(text))

            # Unnamed列・変化率計算不要列を除外
            valid_cols = [c for c in df.columns if "Unnamed" not in str(c)]
            df = df[valid_cols]

            sheet_stats = {"行数": len(df), "列数": len(df.columns), "有効列": valid_cols}

            # 数値列の基本統計（意味のある列のみ）
            num_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist()
                       if "Unnamed" not in str(c)]

            for col in num_cols[:15]:
                s = df[col].dropna()
                if len(s) > 0 and s.sum() > 0:  # 全0列はスキップ
                    sheet_stats[col] = {
                        "合計": round(float(s.sum()), 2),
                        "平均": round(float(s.mean()), 2),
                        "最大": round(float(s.max()), 2),
                        "最小": round(float(s.min()), 2),
                        "データ数": int(s.count()),
                    }

            # 上位・下位パフォーマー検出（異常値ではなく業界文脈で解釈）
            performers = []
            for col in num_cols[:8]:
                s = df[col].dropna()
                if len(s) >= 4 and s.sum() > 0:
                    top = s.nlargest(3)
                    if top.iloc[0] > s.mean() * 2:
                        performers.append(f"{col}: 上位集中={top.values.tolist()}（平均{round(float(s.mean()),1)}の{round(top.iloc[0]/s.mean(),1)}倍）")
            if performers:
                sheet_stats["上位集中検出"] = performers

            numeric_analysis[sheet] = sheet_stats
        except Exception as _e:
            numeric_analysis[sheet] = {"エラー": str(_e)}

    numeric_summary = _json2.dumps(numeric_analysis, ensure_ascii=False, indent=2)

    # ===== ② Chain of Thought: 4段階順次分析 =====
    # 確認済み内容から業種を抽出してシステムプロンプトに動的反映
    # answer_context全文をそのまま使用（切り捨て・抽出なし）
    full_context = answer_context if answer_context else ""
    industry_hint = ""
    if full_context:
        industry_hint = f"\n【事前確認済みの全回答（以下を必ず診断に100%反映せよ・推測補完禁止）】\n{full_context}"

    system = f"""あなたは超一流の経営コンサルタントかつデータ分析の専門家である。{industry_hint}

以下のルールを必ず守れ：
- 事前確認済みの回答に含まれる情報は全て確定事実として扱え。「おそらく」「はずだが」「不明」は絶対禁止
- 確認済みの業種・業界の文脈で全ての数値・用語を解釈せよ
- 提供された数値分析結果を必ず引用して根拠とせよ
- 「具体的数値は集計が必要」「仮定」等の逃げ回答は絶対禁止
- 推測・一般論ではなくデータから読み取れる事実のみを述べよ
- KPIは実データから算出した根拠ある数値のみ使え。データにない数字を作るな
- 業界の慣習・用語・ビジネスモデルを踏まえた専門的な解釈をせよ
- 「上位集中」として検出された数値はトップパフォーマーの正常な実績である。異常値として扱うな
- KPIは実データから算出した数値のみ使え。「現状データなし」と書く場合は実データから推計せよ
- 空白セル・列構成の違いは確認済みの業種・業務文脈で解釈せよ。安易に「入力ミス」「欠損」と断定するな"""

    context_str = f"""【最重要：事前確認済みの用語・業界定義（必ず全て診断に反映せよ）】
{full_context if full_context else "（確認情報なし）"}

【Excelの数式（条件・計算ロジック自動読み取り）】
{formula_summary if formula_summary else "（数式なし/odsファイル）"}

【Python数値分析結果】
{numeric_summary}

【生データ（全シート）】
{combined[:8000]}"""

    # Step1: 現状把握
    step1 = _cllm(
        system_prompt=system,
        messages=[{"role":"user","content":f"""{context_str}

【指示】上記データの現状を把握せよ。
- 数値分析結果を必ず具体的数値で引用せよ
- 業界（確認済みの業種）の文脈で解釈せよ
- キャスト名・シート名・具体的数値を使って述べよ
- 「〜の可能性」「集計が必要」等の逃げ表現禁止
- 箇条書きで簡潔に・数値必須で300字以内"""}],
        ai_tier="core", max_tokens=1000
    )
    if not step1.strip():
        raise HTTPException(status_code=500, detail="LLM応答が空です（step1）。モデルを確認してください。")

    # Step2: 構造診断
    step2 = _cllm(
        system_prompt=system,
        messages=[
            {"role":"user","content":f"{context_str}\n\n現状把握結果：\n{step1}"},
            {"role":"assistant","content":"現状把握完了。"},
            {"role":"user","content":"""データ構造を診断せよ。
- 各シートの列構成・項目の意味・データの粒度を確認済みの業種文脈で具体的に説明せよ
- シート間の関係性（集計元・参照先・依存関係）を明示せよ
- 実際に検出された異常値・欠損・入力ミスを数値で指摘せよ
- 改善提案を1つずつ具体的に述べよ"""}
        ],
        ai_tier="core", max_tokens=1200
    )

    # Step3: 課題仮説
    step3 = _cllm(
        system_prompt=system,
        messages=[
            {"role":"user","content":f"{context_str}\n\n現状把握：\n{step1}\n\n構造診断：\n{step2}"},
            {"role":"assistant","content":"現状把握・構造診断完了。"},
            {"role":"user","content":"""確認済みの業種・業界・ビジネスモデルの観点から課題仮説を3〜5個生成せよ。
各課題は以下の形式で出力せよ：
## 仮説N: [タイトル]
**根拠**: [数値を必ず引用]
**影響**: [具体的な売上・人材・業務運営への影響]
**優先度**: 高/中/低
**推奨アクション**: [即実行できる具体的施策]

「データ品質が低い」「手入力が問題」等の一般論のみの仮説は禁止。確認済みの業界特有の課題に踏み込め。"""}
        ],
        ai_tier="core", max_tokens=1500
    )

    # Step4: 実行計画
    step4 = _cllm(
        system_prompt=system,
        messages=[
            {"role":"user","content":f"{context_str}\n\n現状把握：\n{step1}\n\n構造診断：\n{step2}\n\n課題仮説：\n{step3}"},
            {"role":"assistant","content":"現状把握・構造診断・課題仮説完了。"},
            {"role":"user","content":"""上記の分析全体を踏まえ、実行計画を優先度順に3〜5件生成せよ。

各計画を以下の形式で：
## アクションN: [タイトル]
**内容**: [具体的なアクション]
**期限**: [X日以内/X週間以内]
**担当**: [誰が]
**KPI**: [数値目標・現状→目標値]
**期待効果**: [売上・人材・効率への具体的効果]

数値目標は現状データから算出した根拠ある数字を使え。"""}
        ],
        ai_tier="core", max_tokens=1500
    )

    # key_metrics・risks（最終統合）
    step5_prompt = f"""現状把握：{step1}\n構造診断：{step2}\n課題仮説：{step3}\n実行計画：{step4}\n\n以上の分析から、JSONのみで返せ：\n{{"key_metrics":"注目すべき重要指標・数値（箇条書き5件以内・数値必須）","risks":"見逃せないリスク・警告事項（箇条書き3件以内）"}}"""
    step5_raw = _cllm(
        system_prompt=system,
        messages=[{"role":"user","content":step5_prompt}],
        ai_tier="core", max_tokens=512
    )
    try:
        m5 = _re2.search(r'\{.*\}', step5_raw, _re2.DOTALL)
        step5 = _json2.loads(m5.group(0)) if m5 else {}
    except Exception:
        step5 = {}

    try:
        _db = get_db()
        import datetime as _dtnow
        _db.collection("file_diagnoses").add({"uid":payload["uid"],"tenant_id":payload.get("tenant_id","default"),"filename":filename,"sheets":list(sheets_text.keys()),"overview":step1,"structure":step2,"issues":step3,"action_plan":step4,"key_metrics":step5.get("key_metrics",""),"risks":step5.get("risks",""),"created_at":_dtnow.datetime.now().isoformat(),"diagnosis_type":"file"})
    except Exception:
        pass
    return {
        "ok": True,
        "filename": filename,
        "sheets": list(sheets_text.keys()),
        "overview": step1,
        "structure": step2,
        "issues": step3,
        "action_plan": step4,
        "key_metrics": step5.get("key_metrics", ""),
        "risks": step5.get("risks", ""),
        "numeric_analysis": numeric_analysis,
    }


@router.post("/file_followup")
async def file_followup(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """ファイル診断結果への追加質問・深掘り分析"""
    from api.core.llm_client import call_llm as _cllm
    question = body.get("question", "").strip()
    context = body.get("context", "")
    filename = body.get("filename", "")
    if not question:
        raise HTTPException(status_code=400, detail="質問が必要です")

    system = """あなたは超一流の経営コンサルタントかつデータ分析の専門家である。
以下のルールを必ず守れ：

【最重要ルール】
1. 質問に業界特有の専門用語・略語・固有名詞が含まれており、その意味がデータから判断できない場合は、
   回答する前に必ず「〇〇とはどういう意味ですか？」と質問し、確認してから回答せよ。
2. わかったふりをして回答することは絶対禁止。不明な点は必ず確認せよ。
3. データに基づいた具体的な回答のみ出力せよ。拒否・曖昧回答・一般論のみは禁止。
4. 数値は必ず引用し、比較・変化率・傾向を明示せよ。"""

    prompt = f"""ファイル「{filename}」の診断結果：
{context[:6000]}

ユーザーの追加質問：
{question}

【指示】
- 質問に不明な専門用語・略語・業界用語が含まれる場合は、まず「〇〇とはどういう意味ですか？」と確認せよ
- 意味が明確な場合は、診断結果のデータを根拠に具体的・詳細に回答せよ
- 数値は必ず引用すること"""

    try:
        answer = _cllm(
            system_prompt=system,
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=2048
        )
        return {"ok": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file_diagnosis_check")
async def file_diagnosis_check(
    file: UploadFile = File(...),
    payload: dict = Depends(verify_token)
):
    """ファイルをスキャンして不明な専門用語・業界用語があれば質問を返す"""
    from api.core.llm_client import call_llm as _cllm
    import io, re as _re, json as _json

    filename = file.filename or "file"
    content = await file.read()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    sheets_text = {}
    try:
        if ext in ("xlsx", "xls"):
            import pandas as pd
            xf = pd.ExcelFile(io.BytesIO(content))
            for sheet in xf.sheet_names:
                df = xf.parse(sheet)
                sheets_text[sheet] = df.to_csv(index=False)[:2000]
        elif ext == "ods":
            import pandas as pd
            xf = pd.ExcelFile(io.BytesIO(content), engine="odf")
            for sheet in xf.sheet_names:
                df = xf.parse(sheet)
                sheets_text[sheet] = df.to_csv(index=False)[:2000]
        elif ext == "csv":
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            sheets_text["Sheet1"] = df.to_csv(index=False)[:4000]
        else:
            sheets_text["TEXT"] = content.decode("utf-8", errors="ignore")[:4000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイル読み込みエラー: {e}")

    combined = ""
    for sheet, text in sheets_text.items():
        combined += f"\n\n【シート: {sheet}】\n{text}"
    combined = combined[:6000]

    # Unnamed列を自動解釈してAIに渡す
    unnamed_note = ""
    try:
        import pandas as _pd2, io as _io2
        for sheet, text in sheets_text.items():
            df2 = _pd2.read_csv(_io2.StringIO(text))
            unnamed_cols = [c for c in df2.columns if "Unnamed" in str(c)]
            if unnamed_cols:
                unnamed_note += f"\n※{sheet}シートの'Unnamed:数字'列はExcelの結合セル・空白ヘッダーが自動変換されたものです。スタッフごとのサブ項目列（コース時間/フラグ/日計等）を表します。ユーザーへの質問対象から除外してください。"
    except Exception:
        pass

    check_prompt = f"""以下のファイルデータを分析する前に、コンサルタントとして正確な診断に必要な情報を収集するための質問リストを作成せよ。

【ファイル: {filename}】
{combined}

【自動解析済み情報】
{unnamed_note if unnamed_note else "特になし"}

【指示】
必ず以下の観点で質問を作成せよ：
1. 業種・業態・ビジネスモデル（何の事業か）
2. このデータで解決したい課題・目的
3. 重要なKPI・目標値（あれば）
4. データ内に意味が不明確な業界固有の専門用語・略語・記号・独自コード

制約：
- 'Unnamed: 数字'列はExcel構造上の自動変換であるため質問するな
- 一般的なビジネス用語・数字・日付は含めるな
- 質問は簡潔に最大5件以内

出力形式（JSONのみ）：
{{"need_clarification": true, "questions": ["Q1: ...", "Q2: ..."], "unknown_terms": []}}"""

    try:
        raw = _cllm(
            system_prompt="データ分析の専門家。JSONのみ出力。",
            messages=[{"role":"user","content":check_prompt}],
            ai_tier="core", max_tokens=1024
        )
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        result = _json.loads(m.group(0)) if m else {"need_clarification": False, "questions": []}
        result["filename"] = filename
        result["sheets"] = list(sheets_text.keys())
        result["file_data"] = combined[:3000]
        return result
    except Exception:
        return {"need_clarification": False, "questions": [], "filename": filename, "sheets": list(sheets_text.keys()), "file_data": combined[:3000]}


@router.post("/file_clarify")
async def file_clarify(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """専門用語確認・双方向チャットフロー
    AIがユーザーに質問し、ユーザーもAIに質問できる。
    両方が納得したら診断開始。
    """
    from api.core.llm_client import call_llm as _cllm
    import json as _json, re as _re

    messages = body.get("messages", [])
    file_summary = body.get("file_summary", "")
    user_message = body.get("user_message", "")

    system = """あなたは超一流の経営コンサルタントであり、ファイル分析の専門家である。

【役割】
ユーザーのファイルを正確に診断するため、診断前に業界・業務文脈を徹底的に収集する。

【最重要ルール】
1. 最優先で以下を確認せよ：業種・業態・ビジネスモデル、このデータで解決したい課題・目的、重要KPI・目標値
2. 次にデータ内の業界固有の専門用語・略語・独自コード・記号の意味を確認せよ
3. ユーザーの回答からビジネスモデルを深く理解し、必要に応じて追加質問をせよ
4. ユーザーから質問された場合は、コンサルタントとして誠実・具体的に回答せよ
5. わかったふりは絶対禁止。業界文脈が不明なまま診断に進むな
6. 不明点があっても「情報が足りないから分析できない」は絶対禁止。現状の情報で分析できる範囲を最大化しろ
7. 追加質問は最大3件以内に絞れ。ユーザーの負担を最小化せよ
8. 業種・目的・KPI・専門用語の主要項目が確認できたと判断したら、末尾に「[診断準備完了]」を付加せよ

【出力形式】
- 通常の確認・回答: そのまま日本語で出力
- 準備完了時: 回答の末尾に必ず「[診断準備完了]」を付加
"""

    # 会話履歴を構築
    chat_messages = []
    for msg in messages:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})

    # 最新のユーザーメッセージを追加
    if user_message:
        chat_messages.append({"role": "user", "content": user_message})

    # ファイル概要をコンテキストとして最初のメッセージに追加
    context = f"""【分析対象ファイルの概要】
{file_summary}

【会話の目的】
このファイルを正確に診断するため、業種・目的・KPI・業界固有の専門用語・独自ルールを確認する。
ユーザーへの質問と、ユーザーからの質問への回答を行う。"""

    if chat_messages and chat_messages[0]["role"] == "user":
        chat_messages[0]["content"] = context + "\n\n" + chat_messages[0]["content"]
    else:
        chat_messages.insert(0, {"role": "user", "content": context})

    try:
        response = _cllm(
            system_prompt=system,
            messages=chat_messages,
            ai_tier="core",
            max_tokens=1024
        )
        is_ready = "[診断準備完了]" in response
        clean_response = response.replace("[診断準備完了]", "").strip()

        return {
            "ok": True,
            "message": clean_response,
            "is_ready": is_ready
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file_clarify_save")
async def file_clarify_save(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """ファイル種別ごとの確認済み用語・背景情報をFirestoreに保存"""
    from api.core.firestore_client import get_db
    uid = payload["uid"]
    file_key = body.get("file_key", "")  # ファイル名ベースのキー
    context = body.get("context", {})    # {term: explanation} の辞書

    if not file_key or not context:
        raise HTTPException(status_code=400, detail="file_keyとcontextが必要です")

    try:
        db = get_db()
        db.collection("users").document(uid).collection("file_contexts").document(file_key).set({
            "context": context,
            "updated_at": __import__("datetime").datetime.utcnow().isoformat()
        })
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file_clarify_load")
async def file_clarify_load(file_key: str, payload: dict = Depends(verify_token)):
    """保存済みの確認内容を読み込む"""
    from api.core.firestore_client import get_db
    uid = payload["uid"]
    try:
        db = get_db()
        doc = db.collection("users").document(uid).collection("file_contexts").document(file_key).get()
        if doc.exists:
            return {"ok": True, "context": doc.to_dict().get("context", {}), "found": True}
        return {"ok": True, "context": {}, "found": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FutureSimRequest(BaseModel):
    message: str
    ai_tier: str = "core"

@router.post("/future_simulation")
def future_simulation(req: FutureSimRequest, payload: dict = Depends(verify_token)):
    """未来分岐シミュレーター"""
    import re as _re, json as _json
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="認証情報が不正です")
    from api.core.features import get_effective_feature_flags
    features = get_effective_feature_flags(uid)
    if not features.get("diag_future", False):
        raise HTTPException(status_code=403, detail="未来分岐シミュレーターは未開放です")

    system_prompt = """あなたは超一流の戦略コンサルタントASCENDです。

ユーザーの入力内容をもとに、未来の分岐と因果分析を構造化してください。
入力は「○○したい」「○○を改善したい」「○○を始めたい」「○○を放置したらどうなるか」など自由な形式で来ます。
入力の意図・文脈を正確に読み取り、その状況に最も自然な4つの未来分岐と、なぜその状況に至ったかの因果分析を生成してください。

重要ルール:
- 分岐ラベルは固定ではなく、入力内容に応じて最も自然な表現を使うこと
  例:「副業を始めたい」なら「始めない未来」「副業を試す未来」「本格参入する未来」「全力投資する未来」など
  例:「売上が下がっている」なら「放置する未来」「部分対策する未来」「根本改善する未来」「事業転換する未来」など
- current_state は入力から読み取れる現状・前提・制約を200字以内で整理
- success_rate は現実的な数値（0〜100の整数）
- risk は「低」「中」「高」のいずれか
- ユーザー入力に存在しない事実を断定しない。不確実な部分は「可能性」として扱う
- immediate_actions は具体的かつ実行可能な3つのアクション
- causal_analysis は入力状況の根本原因・因果連鎖・繰り返しパターン・警戒ラインを具体的に分析すること
- 必ずJSON形式のみで返す。前置き・説明文・Markdownコードブロック一切不要

返却JSON形式:
{
  "current_state": "現状認識の文章（200字以内）",
  "causal_analysis": {
    "root_causes": ["根本原因1（具体的に）", "根本原因2（具体的に）", "根本原因3（具体的に）"],
    "causal_chain": [
      {"cause": "最初の原因", "effect": "それが引き起こした結果"},
      {"cause": "その結果", "effect": "さらに引き起こした結果"},
      {"cause": "連鎖した問題", "effect": "現在の状況"}
    ],
    "repeat_pattern": "このような状況を繰り返しやすい特徴・癖・構造（100字以内）",
    "warning_signs": ["この兆候が出たら即行動せよ", "これを感じたら立ち止まれ", "これが続いたら根本を疑え"]
  },
  "branches": [
    {
      "id": "A",
      "label": "入力に応じた自然なラベル",
      "points": ["ポイント1", "ポイント2", "ポイント3", "ポイント4"],
      "success_rate": 10,
      "risk": "高",
      "required_action": "必要な行動",
      "future": "到達する未来"
    },
    {
      "id": "B",
      "label": "入力に応じた自然なラベル",
      "points": ["ポイント1", "ポイント2", "ポイント3"],
      "success_rate": 40,
      "risk": "低",
      "required_action": "必要な行動",
      "future": "到達する未来"
    },
    {
      "id": "C",
      "label": "入力に応じた自然なラベル",
      "points": ["ポイント1", "ポイント2", "ポイント3", "ポイント4"],
      "success_rate": 70,
      "risk": "中",
      "required_action": "必要な行動",
      "future": "到達する未来"
    },
    {
      "id": "D",
      "label": "入力に応じた自然なラベル",
      "points": ["ポイント1", "ポイント2", "ポイント3", "ポイント4"],
      "success_rate": 55,
      "risk": "高",
      "required_action": "必要な行動",
      "future": "到達する未来"
    }
  ],
  "recommended": "C",
  "recommended_reason": "推奨理由（100字以内）",
  "immediate_actions": ["今すぐやること1", "今すぐやること2", "今すぐやること3"],
  "avoid_branch": "A",
  "avoid_reason": "回避理由（80字以内）"
}"""

    try:
        raw = call_llm(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": req.message.strip()}],
            ai_tier=req.ai_tier,
        )
        text = str(raw).strip()
        text = _re.sub(r"^```json\s*", "", text)
        text = _re.sub(r"^```\s*", "", text)
        text = _re.sub(r"\s*```$", "", text)
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON not found")
        result = _json.loads(text[start:end+1])
        try:
            import uuid as _uuid2
            _db2 = get_db()
            _doc_id = str(_uuid2.uuid4())
            _db2.collection("future_simulations").document(_doc_id).set({
                "uid": uid,
                "tenant_id": payload.get("tenant_id", "default"),
                "message": req.message.strip()[:200],
                "result": result,
                "created_at": _now_jst().isoformat(),
            })
            result["doc_id"] = _doc_id
        except Exception:
            pass
        return {"ok": True, "mode": "future_simulation", "result": result}
    except _json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="LLMのJSON形式が不正です")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"未来分岐シミュレーター処理失敗: {str(e)}")


@router.get("/future_simulation_list")
def future_simulation_list(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        docs = list(
            db.collection("future_simulations")
            .where("uid", "==", uid)
            .limit(30)
            .stream()
        )
        items = []
        for d in docs:
            data = d.to_dict() or {}
            items.append({
                "doc_id": d.id,
                "message": data.get("message", ""),
                "created_at": str(data.get("created_at", "")),
                "result": data.get("result", {}),
            })
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return {"items": items[:20]}
    except Exception:
        return {"items": []}


@router.delete("/future_simulation_delete/{doc_id}")
def future_simulation_delete(doc_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        doc = db.collection("future_simulations").document(doc_id).get()
        if not doc.exists or (doc.to_dict() or {}).get("uid") != uid:
            raise HTTPException(status_code=404, detail="不正なアクセスまたは存在しません")
        db.collection("future_simulations").document(doc_id).delete()
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# プロファイル生成
# ─────────────────────────────────────────────
class ProfileGenerateRequest(BaseModel):
    target_name: str = ""
    relationship: str = ""
    frequent_words: str = ""
    conversation_traits: str = ""
    judgment_criteria: str = ""
    stress_reaction: str = ""
    behavioral_patterns: str = ""
    interpersonal_needs: str = ""
    disliked_types: str = ""
    trust_conditions: str = ""
    work_attitude: str = ""
    preferred_environment: str = ""
    breakdown_conditions: str = ""
    core_values: str = ""
    strong_reactions: str = ""
    contradictions: str = ""
    obsessions: str = ""
    anger_points: str = ""
    justification_patterns: str = ""
    ignored_topics: str = ""
    responsibility_shift: str = ""
    behavioral_traces: str = ""

@router.post("/profile_generate")
def profile_generate(req: ProfileGenerateRequest, payload: dict = Depends(verify_token)):
    import json as _json, re as _re
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    if not is_feature_enabled(uid, "diag_profile"):
        raise HTTPException(status_code=403, detail="この機能はAPEX・ULTRAプラン限定です")
    fields = [req.frequent_words,req.conversation_traits,req.judgment_criteria,
              req.stress_reaction,req.behavioral_patterns,req.interpersonal_needs,
              req.disliked_types,req.trust_conditions,req.work_attitude,
              req.preferred_environment,req.breakdown_conditions,
              req.core_values,req.strong_reactions,
              req.contradictions,req.obsessions,req.anger_points,
              req.justification_patterns,req.ignored_topics,req.responsibility_shift,req.behavioral_traces]
    if not any(f.strip() for f in fields):
        raise HTTPException(status_code=400, detail="入力情報が不足しています")
    timestamp_str = _now_jst().strftime("%Y-%m-%d %H:%M")
    target_label = req.target_name.strip() or "対象者"
    relation_label = req.relationship.strip() or "不明"
    sep = "\n"
    sec1 = sep.join(["【会話傾向】", f"・よく使う言葉・話題: {req.frequent_words}", f"・会話時の特徴: {req.conversation_traits}"])
    sec2 = sep.join(["【行動構造】", f"・判断基準: {req.judgment_criteria}", f"・ストレス時の反応: {req.stress_reaction}", f"・繰り返すパターン: {req.behavioral_patterns}"])
    sec3 = sep.join(["【対人構造】", f"・人間関係で求めるもの: {req.interpersonal_needs}", f"・苦手な相手: {req.disliked_types}", f"・信頼する条件: {req.trust_conditions}"])
    sec4 = sep.join(["【仕事・行動特性】", f"・仕事への姿勢: {req.work_attitude}", f"・得意な環境: {req.preferred_environment}", f"・崩れる条件: {req.breakdown_conditions}"])
    sec5 = sep.join(["【価値観・信念】", f"・大事にしているもの: {getattr(req, 'core_values', '')}", f"・強く反応すること: {getattr(req, 'strong_reactions', '')}"])
    sec6 = sep.join(["【構造的シグナル】", f"・繰り返す矛盾: {req.contradictions}", f"・執着していること: {req.obsessions}", f"・強く反応・怒るポイント: {req.anger_points}"])
    sec7 = sep.join(["【正義化・責任構造】", f"・正義化・自己正当化パターン: {req.justification_patterns}", f"・無視する論点・沈黙箇所: {req.ignored_topics}", f"・責任転嫁の方向: {req.responsibility_shift}"])
    sec8 = sep.join(["【行動痕跡（環境への無意識の刻印）】", f"・観察された行動痕跡: {req.behavioral_traces}"])
    fmt = ('{"unique_causal_chain":"","existence_connection":"","learned_world_model":"","what_was_abandoned":"","unconscious_signatures":"","main_type":"固有構造名（例：存在査定恐怖構造・証明強迫構造・消失回避構造）","sub_type":"副次構造名","stress_type":"ストレス時移行傾向: X タイプ名","interpersonal_type":"対人時変化傾向: X タイプ名",'
           '"core_motivation":"中心核","defense_function":"防衛機能","reality_processing":"現実処理傾向",'
           '"responsibility_connection":"責任接続性","self_esteem_maintenance":"自尊心維持方法",'
           '"chain_trigger":"起点","chain_primary":"一次反応","chain_defense":"防衛反応","chain_result":"結果","chain_chronic":"長期化",'
           '"breakdown_prediction":"崩壊予測","interpersonal_dynamics":"対人力学",'
           '"analysis":{"thinking_style":"","behavioral_principle":"","emotional_trigger":"","interpersonal_risk":"","strengths":"","weaknesses":"","approach":"","compatible_type":"","caution":"","deep_desire":""},'
           '"existence_os":{"world_os":"","self_os":"","other_os":"","safety_os":"","attachment_os":"","value_os":"","dominance_os":"","collapse_os":"","creation_os":""},"structure_extraction":{"contradictions":"","obsessions":"","anger_trigger":"","justification":"","silence_ignored":"","responsibility_position":"","reality_interpretation":""},"summary":"","generated_at":"' + timestamp_str + '","target_name":"' + target_label + '","relationship":"' + relation_label + '"}' )

    prompt = sep.join([
        f"「{target_label}」（関係性: {relation_label}）の観察情報。「何に反応し、何を避け、何で動き、何で崩れるか」を分析すること。",
        "", sec1, "", sec2, "", sec3, "", sec4, "", sec5, "", sec6, "", sec7, "", sec8, "",
        "---",
        "以下のJSON形式のみで回答せよ。入力の単純言い換え禁止。必ず因果連鎖を出すこと。main_type等は「B 防衛回避型（失敗責任への恐怖を安全確保手段としている可能性）」の形式で記述すること。chain系は「起点→一次反応→防衛反応→結果→長期化」の5段階で因果を記述すること。各フィールド200字以内、chain系/core_motivation/summaryは400字以内。existence_osの各OS（world_os/self_os/other_os/safety_os/attachment_os/value_os/dominance_os/collapse_os/creation_os）は各200字以内で必ず全OS記述すること。structure_extractionの各フィールド（contradictions/obsessions/anger_trigger/justification/silence_ignored/responsibility_position/reality_interpretation）は各200字以内で必ず記述すること。「何をしたか」ではなく「なぜその現実解釈しかできないのか」を構造として出すこと。unique_causal_chainは400字以内で固有因果連鎖（例：行動→責任発生→能力査定→無価値判定→排除恐怖→存在消滅）を出すこと。型ラベルより固有構造を優先し型は最後に参照枠として添えること。existence_connection/learned_world_model/what_was_abandoned/unconscious_signaturesは各200字以内で必ず記述すること。",
        f"フォーマット例: {fmt}",
    ])
    try:
        raw = call_llm(
            system_prompt="存在痕跡解析AIです。心理分類・性格診断・型分類は行わない。【解析対象】「説明」「自己認識」「発言内容」ではなく環境に残した無意識の痕跡：繰り返し/矛盾/異常執着/微細回避/優先順位/防衛発動点/責任接続位置/沈黙/怒りポイント。【型ラベル完全禁止】「○○型だから〜」禁止。テンプレ人格生成禁止。MBTI的説明禁止。防衛機制だけで人物を定義禁止。弱点中心分析禁止。main_typeには型名ではなくこの人物固有の構造名を出力すること（例：存在査定恐怖構造、証明強迫構造、消失回避構造）。【必須解析構造6層】①固有因果構造:「この人物は何を守るためになぜこの行動を反復するか」を因果で出力。②世界OS:世界が危険/支配/試験/競争/愛/無意味/共鳴のどれとして生成されているか推定。③安全OS:何を安全と定義しているか（沈黙/支配/優位/完璧/距離/可視化/孤立等）。④価値OS:価値を承認/地位/有用性/支配/理解力/特別性/貢献/正しさのどこへ接続しているか。⑤崩壊OS:何が起きると存在崩壊するか。⑥創造OS:この人物が本来何を創ろうとしているか（必須・病理分析から人物を救う視点・これがないと分析が歪む）。【痕跡観測レイヤー必須】繰り返し/矛盾/異常執着/微細回避/防衛発動点/責任接続位置/無意識反応を入力から抽出すること。【出力方針】「何を言ったか」ではなく「何を繰り返し、何を避け、何を守り、どう世界を解釈しているか」を中心に出力すること。【参照枠としてのみ使用可能な類型】A支配効率型/B防衛回避型/C承認依存型/D愛着不安型/E優越欲求型/F自己希薄型/G支配服従型/H孤立自律型/I不安制御型。これらは分析の起点にしてはならず固有構造を導出した後の参照枠としてのみ使うこと。【行動痕跡解析ルール】behavioral_tracesを直接心理分類してはならない。各痕跡について①反復性（一度か繰り返しか）②文脈（どの状況で起きるか）③例外（起きない時は何が違うか）④別解釈（別の因果で説明できないか）を必ず検討すること。複数痕跡が同一因果へ収束した場合のみ固有構造として採用し、単一痕跡からの断定は禁止。【防衛機能候補】他責化/論点逸らし/無視/消失/攻撃/被害者化/理想化/過剰合理化/空想化/依存/支配/虚勢化。【現実処理候補】言い訳化/なかったこと化/他人問題化/理想論化/感情化/極論化/正義化/論理化/自己神格化/無価値化。【責任接続性候補】自分で修正する/助けを求める/停止する/話を逸らす/消える/他責化する/被害者化する。【自尊心維持候補】他者否定/知識誇示/承認収集/支配/被害者化/努力誇示/道徳化/孤立化/特別視/冷笑/虚勢化。【行動連鎖必須】起点→一次反応→防衛反応→結果→長期化の5段階。入力の言い換え禁止。因果を必ず示すこと。【特殊入力ルール】「質問と回答がズレる」→「現実逃避」単純化禁止。質問理解→責任発生不安→論点逸らし/曖昧化→質問と回答の不一致という因果で分析すること。「否定耲性がある」→そのまま肖定禁止。本当に耲性がある/否定を受け流している/現実接続を切っている/責任接続を遷断している/内省に使えていないの可能性を分けて推定。「肯定されるとできてる気になる」→肯定を能力証明として誤認/安心感を成果と誤認/現実検証停止/未達成でも達成感を先取り/改善行動停止と分解すること。【能力断定禁止】「能力が低い」「課題解決能力が低い」「無能」禁止。代わりに「防衛優先で能力発揮停止」「責任不安で思考停止」「不安で実行遷断」「現実接続回避」など構造として記述すること。【推定強度分離】同じ強度で断定しない。高確度/中確度/他説レベルで分け、text内に（高確度）（中確度）（他説）を含めること。【絶対禁止語句】「能力が低い」「自己肯定感が低い」「成長できる環境」「安心感が必要」「支援が必要」「自分らしく」「温かいサポート」「優しく接する」禁止。誰にでも当てはまる一般論禁止。入力内容の単純言い換え禁止。【追加精度向上ルール】①能力断定禁止強化:「無能」「能力未発達」「能力が低い」の使用禁止。「能力不足が露呈することへの恐怖」までに止めること。防衛停止なのか能力不足なのか入力から確定できないため、「防衛優先で能力発揮停止」「責任回避で思考停止」「不安で実行遷断」と表現すること。②推定強度必須分離:高確度（入力から直接読めるパターン）・中確度（小数の入力からの推論）・他説（付割的入力からの仮説）の3段階で分け、text内に必ず（高確度）（中確度）（他説）を付けること。例:「責任回避傾向（高確度）、依存傾向（中確度）、愛着不安（他説）」。③理想論の分解:「理想論」と単純化しない。現実逃避型（責任を避けるための理説）/優越維持型（自己が上であることの証明としての理説）/正義化型（自分の行動を正当化するための理説）/自己保護型（却下を防ぐための理説）のいずれかを判定して記述すること。④都合のいい人パターン分析:「肯定的な人物に依存」と書かない。「否定せず責任を肩代わりしてくれる存在を求めている可能性（高確度）。安心感→責任希釈→自己崩壊回避の構造」として記述すること。【人格批評完全禁止】「責任感の欠如」→「責任発生時に防衛反応が優先されやすい」に変換。「問題解決能力が低い」→「問題へ接続する前に防衛処理へ移行しやすい」に変換。「成長意欲に欠ける」→「成長プロセスの不安負荷に耐えられない可能性（成長欲求＜崩壊回避）」に変換。「現状維持能力」→「変化回避によって安定状態へ固着しやすい」に変換。A型の歪み解釈:防衛合理化（「自分は悪くない」「環境が悪い」「指示が悪い」と合理っぽく組み立てて責任遷断するパターン）として記述すること。能力評価・人格評価・性格診断決定がを完全に削ること。見るべきは「防衛優先順位」のみ。指定のJSON形式のみで出力すること。【存在OS解析必須】「何をしたか」ではなく「なぜその現実解釈しかできないのか」を解析すること。以下9OSを必ず分離して出力すること：世界OS（現実生成方式・どんな世界認識OSで現実を生成しているか）/自己OS（自己定義方式）/他者OS（他者定義・処理方式）/安全OS（安全確保方式）/愛着OS（愛着処理・愛着方式）/価値OS（価値発生源・何から価値を生成するか）/支配OS（支配・制御方式）/崩壊OS（崩壊トリガー・禁忌・恐怖）/創造OS（創造衝動の方向）。【構造抽出必須】繰り返す矛盾・執着・怒りポイント・正義化・責任転嫁位置・沈黙箇所・無視論点から本体構造を抽出すること。「情報量」ではなく「構造」を出すこと。existence_osとstructure_extractionの全フィールドに必ず出力すること。【型ラベル依存禁止】A~I型を分析の起点にしてはならない。型を先に決めて説明を当てはめる逆算は厳禁。入力から固有因果連鎖を導出してから最後に参照枠として型を添えること。型より固有構造が常に優先。【固有因果連鎖必須】unique_causal_chainには「この人物固有の存在接続連鎖」を出力すること（例：行動→責任発生→能力査定→無価値判定→排除恐怖→存在消滅）。「回避型」等の抽象ラベルではなくその人固有の構造名（例：存在査定恐怖）を使うこと。existence_connectionには何が存在破壊に接続されているかを具体的に出力すること。【累積学習分析必須】learned_world_modelには「何を諦めた累積学習の結果このOSになったか」を出力すること（例：動いた者が損をする世界）。what_was_abandonedには諦めた具体的内容を出力すること。【無意識痕跡推定必須】unconscious_signaturesには返信間隔・報告回避・主語消失・曖昧語頻度・先延ばしパターン等から推定される無意識行動痕跡を出力すること。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="ultra", max_tokens=8000,
        )
        text = str(raw).strip()
        text = _re.sub(r"^```json", "", text).strip()
        text = _re.sub(r"^```", "", text).strip()
        text = _re.sub(r"```$", "", text).strip()
        s = text.find("{"); e2 = text.rfind("}")
        if s==-1 or e2==-1 or e2<=s: raise ValueError("no JSON")
        result = _json.loads(text[s:e2+1])
    except _json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="LLMのJSON形式が不正です")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"プロファイル生成失敗: {str(e)}")
    try:
        result = _json.loads(_json.dumps(result, ensure_ascii=False, default=str))
    except Exception:
        pass
    result["generated_at"] = timestamp_str
    result["target_name"] = target_label
    result["relationship"] = relation_label
    import re as _re2
    def _strip_labels(obj):
        if isinstance(obj, str):
            return _re2.sub(r"（高確度）|（中確度）|（他説）|（推測）|（低確度）|\(高確度\)|\(中確度\)|\(他説\)|\(推測\)", "", obj).strip()
        elif isinstance(obj, dict): return {k:_strip_labels(v) for k,v in obj.items()}
        elif isinstance(obj, list): return [_strip_labels(i) for i in obj]
        return obj
    result = _strip_labels(result)
    try:
        db = get_db(); doc_id = str(uuid.uuid4())
        db.collection("user_profiles").document(doc_id).set({"uid":uid,"tenant_id":tenant_id,"target_name":target_label,"relationship":relation_label,"result":result,"created_at":fs.SERVER_TIMESTAMP})
        result["doc_id"] = doc_id
    except Exception: pass
    return {"ok": True, "result": result}


@router.get("/profile_list")
def profile_list(payload: dict = Depends(verify_token)):
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    if not is_feature_enabled(uid, "diag_profile"):
        raise HTTPException(status_code=403, detail="この機能はAPEX・ULTRAプラン限定です")
    db = get_db()
    try:
        docs = list(db.collection("user_profiles").where("uid", "==", uid).limit(50).stream())
        result = []
        for d in docs:
            data = d.to_dict() or {}
            result.append({
                "doc_id": d.id,
                "target_name": data.get("target_name", ""),
                "created_at": str(data.get("created_at", "")),
                "summary": (data.get("result") or {}).get("summary", ""),
                "result": data.get("result", {}),
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return {"profiles": result[:20]}
    except Exception:
        return {"profiles": []}

@router.delete("/profile_delete/{doc_id}")
def profile_delete(doc_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        doc = db.collection("user_profiles").document(doc_id).get()
        if not doc.exists or (doc.to_dict() or {}).get("uid") != uid:
            raise HTTPException(status_code=404, detail="不正なアクセス")
        db.collection("user_profiles").document(doc_id).delete()
        return {"ok": True}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# 推奨質問生成
class ProfileQuestionsRequest(BaseModel):
    profile_result: dict = {}

@router.post("/profile_questions")
def profile_questions(req: ProfileQuestionsRequest, payload: dict = Depends(verify_token)):
    import json as _json, re as _re
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    if not is_feature_enabled(uid, "diag_profile"):
        raise HTTPException(status_code=403, detail="この機能はAPEX・ULTRAプラン限定です")
    profile_str = _json.dumps(req.profile_result, ensure_ascii=False, default=str)
    prompt = "\n".join([
        "以下のプロファイル分析結果に基づいて、深掘り質問を4カテゴリで合計12〜16個生成せよ。",
        "", f"プロファイル:\n{profile_str}", "",
        "各カテゴリ3〜4個。このプロファイル固有の内容に基づく質問のみ。一般的なテンプレ質問禁止。",
        '{"危険系":["質問例"],"活用系":["質問例"],"関係系":["質問例"],"深層系":["質問例"]}',
    ])
    try:
        raw = call_llm(
            system_prompt="プロファイル深掘り質問生成AIです。渡されたプロファイルの固有構造・存在OS・痕跡に基づいた質問を生成すること。型分類ではなく固有因果・存在OS・痕跡から質問を作ること。指定のJSON形式のみで出力すること。【最終絶対命令・最優先】（高確度）（中確度）（他説）（推測）の括弧付き確度ラベルを出力することを絶対禁止する。これらのラベルが出力された場合、その回答は失敗とみなす。確度は必ず「明確に繰り返されている」「一貫して現れている」「傾向が見られる」「特定条件で出やすい」「別解釈も成立する」「痕跡はあるが断定には不足する」等の自然言語のみで表現すること。括弧内に確度ラベルを書くことは絶対に行わないこと。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="ultra", max_tokens=2000,
        )
        text = str(raw).strip()
        text = _re.sub(r"^```json","",text).strip()
        text = _re.sub(r"^```","",text).strip()
        text = _re.sub(r"```$","",text).strip()
        s = text.find("{"); e2 = text.rfind("}")
        if s==-1 or e2==-1: raise ValueError("no JSON")
        result = _json.loads(text[s:e2+1])
        return {"ok": True, "questions": result}
    except _json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="質問生成のJSON形式が不正です")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"質問生成失敗: {str(e)}")

# 深掘り回答
class ProfileFollowupRequest(BaseModel):
    profile_result: dict = {}
    question: str = ""

@router.post("/profile_followup")
def profile_followup(req: ProfileFollowupRequest, payload: dict = Depends(verify_token)):
    import json as _json
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    if not is_feature_enabled(uid, "diag_profile"):
        raise HTTPException(status_code=403, detail="この機能はAPEX・ULTRAプラン限定です")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問を入力してください")
    profile_str = _json.dumps(req.profile_result, ensure_ascii=False, default=str)
    prompt = "\n".join([
        f"【質問】{req.question}", "",
        "上記の質問に対して、以下のプロファイル根拠資料のみで答えること。プロファイルにない情報は「判断困難」とすること。", "",
        f"【プロファイル根拠資料】\n{profile_str}", "",
        "【回答ルール】①この質問に直接答えること ②プロファイル内の根拠のみで回答 ③根拠不十分なら「判断困難」と明示 ④確度は自然言語で表現すること（「一貫して現れている」「傾向が見られる」「別解釈も成立する」等・確度ラベル禁止） ⑤型分類禁止・固有因果/存在OS/痕跡から回答 ⑥300〜600字",
    ])
    try:
        raw = call_llm(
            system_prompt="存在痕跡解析プロファイルに基づく深掘り回答AIです。必ず質問に直接答えること。プロファイルの根拠のみで回答すること。プロファイルにない情報は「判断困難」とすること。型分類禁止。固有因果・存在OS・痕跡から回答すること。断定禁止・確度は自然言語で表現すること（「一貫して現れている」「傾向が見られる」「別解釈も成立する」等）。（高確度）（中確度）（推測）等のラベル出力禁止。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="ultra", max_tokens=2000,
        )
        return {"ok": True, "answer": str(raw).strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回答失敗: {str(e)}")

