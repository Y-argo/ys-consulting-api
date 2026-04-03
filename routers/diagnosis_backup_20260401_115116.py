# api/routers/diagnosis.py
import datetime
_JST = datetime.timezone(datetime.timedelta(hours=9))
def _now_jst(): return datetime.datetime.now(_JST)
import uuid
from fastapi import APIRouter, Depends, HTTPException
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
        prompt = f"""以下の入力を構造診断せよ。JSONで返せ。
入力: {req.input_text}
補足: {req.supplement}
フレームワーク候補: {fw_str}
出力形式: {{"summary":"全体要約","structure_layers":[{{"layer":"層名","content":"内容","strength":0.8}}],"key_bottleneck":"主要ボトルネック","recommended_framework":"推奨フレームワーク","next_actions":["アクション1"]}}"""

    elif req.analysis_type == "issue":
        prompt = f"""以下の状況から課題仮説を生成せよ。JSONで返せ。
入力: {req.input_text}
フレームワーク: {fw_str}
出力形式: {{"hypotheses":[{{"hypothesis":"仮説","priority":"high/mid/low","evidence":"根拠","verification":"検証方法"}}],"root_cause":"根本原因","quick_wins":["即効策"]}}"""

    elif req.analysis_type == "comparison":
        prompt = f"""以下の選択肢を多軸比較せよ。JSONで返せ。
選択肢: {req.options or req.input_text}
評価軸: コスト・リスク・効果・実現性・速度
出力形式: {{"options":[{{"name":"選択肢名","scores":{{"cost":80,"risk":60,"effect":90,"feasibility":70,"speed":75}},"summary":"評価コメント"}}],"recommendation":"推奨選択肢","rationale":"理由"}}"""

    elif req.analysis_type == "contradiction":
        prompt = f"""以下の戦略と方針の矛盾を検知せよ。JSONで返せ。
戦略: {req.strategy or req.input_text}
方針: {req.policy or req.supplement}
出力形式: {{"contradictions":[{{"point":"矛盾点","severity":"high/mid/low","resolution":"解決策"}}],"consistency_score":70,"overall_assessment":"総合評価"}}"""

    elif req.analysis_type == "execution":
        prompt = f"""以下の目標に対する実行計画を生成せよ。JSONで返せ。
目標・背景: {req.input_text}
フレームワーク: {fw_str}
出力形式: {{"phases":[{{"phase":"フェーズ名","duration":"期間","actions":["アクション"],"kpi":"KPI","risks":["リスク"]}}],"critical_path":"クリティカルパス","success_criteria":["成功条件"]}}"""
    else:
        return {"ok": False, "error": f"不明なanalysis_type: {req.analysis_type}"}

    try:
        import re, json as _json
        res = call_llm(
            system_prompt="戦略コンサルタント。JSONのみ出力。余計なテキスト禁止。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=2048
        )
        m = re.search(r"\{.*\}", res, re.DOTALL)
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
