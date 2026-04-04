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

def _extract_decision_metrics_from_report(report_md: str, tenant_id: str = "default") -> dict:
    """診断レポートMDからDMスコアを抽出して返す（実際のフォーマットに対応）"""
    import re as _re
    text = report_md or ""

    # Q: 評価ランク → decision_quality_score
    rank_map = {"S":95,"A+":88,"A":83,"A-":78,"B+":73,"B":68,"B-":63,"C+":58,"C":53,"D":40}
    rank_m = _re.search(r"評価ランク[:：]\s*([SABCD][+\-]?)", text)
    rank_raw = rank_m.group(1).strip().upper() if rank_m else "C"
    decision_quality_score = float(rank_map.get(rank_raw, 55))

    # R: 優先改善度 → risk_tolerance（緊急ほど低い）
    risk_map = {"緊急": 40, "高": 52, "中": 65, "低": 85}
    risk_m = _re.search(r"優先改善度[:：]\s*(低|中|高|緊急)", text)
    risk_tolerance = float(risk_map.get(risk_m.group(1) if risk_m else "中", 65))

    # S: 主要課題セクションの記述量 → structural_intelligence
    struct_m = _re.search(r"## 主要課題(.+?)(?=## |$)", text, _re.DOTALL)
    struct_len = len((struct_m.group(1) if struct_m else "").strip())
    structural_intelligence = 85.0 if struct_len >= 300 else 65.0 if struct_len >= 120 else 45.0

    # V: 推奨行動の数 → decision_velocity
    actions = _re.findall(r"推奨行動[:：]", text)
    n_actions = len(actions)
    decision_velocity = min(90.0, 50.0 + n_actions * 13.0)

    # P: 強み記述の具体性 → prediction_accuracy
    strength_m = _re.search(r"強み[:：]\s*(.+)", text)
    strength_len = len(strength_m.group(1).strip()) if strength_m else 0
    prediction_accuracy = 80.0 if strength_len >= 20 else 65.0 if strength_len >= 5 else 45.0

    # E: 弱点記述の具体性 → execution_consistency
    weak_m = _re.search(r"弱点[:：]\s*(.+)", text)
    weak_len = len(weak_m.group(1).strip()) if weak_m else 0
    execution_consistency = 75.0 if weak_len >= 20 else 60.0 if weak_len >= 5 else 45.0

    # ウェイト（app.pyと同一）
    _w_raw = {"Q": 30, "R": 20, "S": 15, "V": 15, "P": 10, "E": 10}
    _w_sum = sum(_w_raw.values())
    w = {k: v / _w_sum for k, v in _w_raw.items()}

    diagnosis_total_score = round(
        w["Q"] * decision_quality_score +
        w["R"] * risk_tolerance +
        w["S"] * structural_intelligence +
        w["V"] * decision_velocity +
        w["P"] * prediction_accuracy +
        w["E"] * execution_consistency, 1
    )

    rank_thresholds = [("S",90),("A+",82),("A",78),("A-",74),("B+",70),("B",65),("B-",60),("C+",55),("C",50)]
    diagnosis_rank = "D"
    for rk, t in rank_thresholds:
        if diagnosis_total_score >= t:
            diagnosis_rank = rk
            break

    return {
        "decision_quality_score":  decision_quality_score,
        "risk_tolerance":          risk_tolerance,
        "structural_intelligence": structural_intelligence,
        "decision_velocity":       decision_velocity,
        "prediction_accuracy":     prediction_accuracy,
        "execution_consistency":   execution_consistency,
        "diagnosis_total_score":   diagnosis_total_score,
        "diagnosis_rank":          diagnosis_rank,
    }


def _save_decision_metrics(uid: str, metrics: dict, source_diagnosis_id: str = "") -> None:
    """decision_metrics/{uid}/records に保存する"""
    db = get_db()
    try:
        col = db.collection("decision_metrics").document(uid).collection("records")
        col.add({
            "user_id":                 uid,
            "created_at":              fs.SERVER_TIMESTAMP,
            "decision_quality_score":  float(metrics.get("decision_quality_score", 60)),
            "risk_tolerance":          float(metrics.get("risk_tolerance", 65)),
            "decision_velocity":       float(metrics.get("decision_velocity", 70)),
            "structural_intelligence": float(metrics.get("structural_intelligence", 65)),
            "prediction_accuracy":     float(metrics.get("prediction_accuracy", 65)),
            "execution_consistency":   float(metrics.get("execution_consistency", 65)),
            "diagnosis_total_score":   float(metrics.get("diagnosis_total_score", 0)),
            "diagnosis_rank":          str(metrics.get("diagnosis_rank", "C")),
            "source_diagnosis_id":     str(source_diagnosis_id),
        })
    except Exception as e:
        print(f"[SAVE_DM_ERROR] {e}", flush=True)


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

    # DMスコア計算・保存
    try:
        dm_metrics = _extract_decision_metrics_from_report(report_md, tenant_id)
        _save_decision_metrics(uid, dm_metrics, doc_id)
    except Exception as _dme:
        print(f"[DM_ERROR] {_dme}", flush=True)

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
        all_sessions = list(db.collection("chat_sessions").limit(200).stream())
        sessions_raw = [s for s in all_sessions if f"__{uid}__" in s.id]
        sessions_raw.sort(key=lambda s: str((s.to_dict() or {}).get("updated_at","")), reverse=True)

        # トピック分類キーワード
        TOPICS = {
            "戦略・競合": ["戦略","競合","差別化","ポジション","市場","シェア","ブランド"],
            "集客・SNS": ["集客","SNS","Instagram","Twitter","広告","フォロワー","投稿"],
            "売上・財務": ["売上","収益","利益","資金","コスト","価格","単価"],
            "組織・人材": ["採用","チーム","組織","スタッフ","教育","マネジメント"],
            "投資・株": ["投資","株","銘柄","シグナル","相場","資産","ポートフォリオ"],
            "診断・分析": ["診断","分析","レポート","スコア","評価","課題"],
            "指名・接客": ["指名","接客","お客様","キャスト","リピート","コミュニケーション"],
        }

        def classify(text):
            for topic, keywords in TOPICS.items():
                if any(k in text for k in keywords):
                    return topic
            return "その他"

        # ノード収集
        raw_nodes = []
        node_set = set()
        for s in sessions_raw[:15]:
            try:
                msgs = list(db.collection("chat_sessions").document(s.id).collection("messages").limit(40).stream())
                msgs.sort(key=lambda m: str((m.to_dict() or {}).get("ts","")))
                for m in msgs:
                    d = m.to_dict() or {}
                    if d.get("role") == "user":
                        content = (d.get("content","") or "")[:35]
                        if content and content not in node_set and len(content) > 3:
                            node_set.add(content)
                            topic = classify(content)
                            raw_nodes.append({"id": content, "label": content, "group": topic})
            except Exception:
                continue

        raw_nodes = raw_nodes[:30]

        # トピック間のエッジ（同トピック内で繋ぐ）
        edges = []
        topic_last = {}
        for n in raw_nodes:
            g = n["group"]
            if g in topic_last:
                edges.append({"from": topic_last[g], "to": n["id"], "topic": g})
            topic_last[g] = n["id"]

        # トピック中心ノードを追加
        topics_used = list(set(n["group"] for n in raw_nodes))
        center_nodes = [{"id": f"__topic_{t}__", "label": t, "group": t, "is_center": True} for t in topics_used]

        # 各ノードをトピック中心に繋ぐ
        center_edges = [{"from": f"__topic_{n['group']}__", "to": n["id"], "topic": n["group"]} for n in raw_nodes]

        all_nodes = center_nodes + raw_nodes
        all_edges = center_edges

        return {"nodes": all_nodes, "edges": all_edges, "topics": topics_used}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

from fastapi import Body as _Body

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
- 推測は推測として扱うこと
- JSON以外の余計な文を絶対に返さないこと
- 目的と手段を混同しないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

【追加ルール（構造診断）】
- 現象・表層原因・根因を必ず分離すること
- 制約条件を必ず抽出すること
- 打ち手は優先順位順に返すこと

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
- 抽象語だけで逃げないこと
- 不明点はmissing_informationに格納すること
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
- 感想ではなく分析を返すこと
- JSON以外の余計な文を絶対に返さないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

【追加ルール（比較表）】
- すべて同じ比較軸で比較すること
- 感覚論ではなく軸差で比較すること
- スコアは1〜5の整数で評価すること（5が最良）
- 最終推奨案を1つ返すこと

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
- 感想ではなく分析を返すこと
- JSON以外の余計な文を絶対に返さないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

【追加ルール（矛盾検出）】
- 目的と手段の衝突を優先検出すること
- KPIと戦略のズレも検出対象とすること
- 矛盾がなければcontradictionsを空配列にすること

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
- 感想ではなく分析を返すこと
- JSON以外の余計な文を絶対に返さないこと
- 出力は必ず有効なJSONオブジェクトのみとすること

【追加ルール（実行プラン）】
- タスクは実行可能な粒度で分割すること
- 優先度はhigh / medium / lowで分類すること
- KPIは可能な限り数値目標を含めること
- deadlineは相対的な目安で構わない（例: 2週間以内）

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
