# api/routers/user_stats.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from google.cloud import firestore as fs
from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT

router = APIRouter(prefix="/api/user", tags=["user"])

RANK_CFG_DEFAULT = {
    "rank_1_name": "追従者", "rank_1_threshold": 80,
    "rank_2_name": "実行者", "rank_2_threshold": 200,
    "rank_3_name": "戦略家", "rank_3_threshold": 450,
    "rank_4_name": "設計者",
}

ALL_PURPOSE_MODES = {
    "auto":        "AUTO（自動判別）",
    "numeric":     "NUMERIC（数字/指標）",
    "growth":      "GROWTH（成長/訓練）",
    "control":     "CONTROL（支配/構造図）",
    "strategy":    "STRATEGY（戦略立案）",
    "analysis":    "ANALYSIS（分析/解析）",
    "planning":    "PLANNING（計画/ロードマップ）",
    "risk":        "RISK（リスク評価）",
    "creative":    "CREATIVE（創造/アイデア）",
    "summary":     "SUMMARY（要約/整理）",
    "negotiation": "NEGOTIATION（交渉/説得）",
    "coaching":    "COACHING（コーチング）",
    "diagnosis":   "DIAGNOSIS（診断/課題発見）",
    "forecast":    "FORECAST（予測/シナリオ）",
    "legal":       "LEGAL（法務/規約）",
    "finance":     "FINANCE（財務/投資）",
    "marketing":   "MARKETING（マーケ/集客）",
    "hr":          "HR（人材/組織）",
    "ops":         "OPS（業務改善/効率化）",
    "tech":        "TECH（技術/エンジニア）",
}

def _load_rank_config(tenant_id: str) -> dict:
    try:
        db = get_db()
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"rank_config_{tid}").get()
            if doc.exists:
                return {**RANK_CFG_DEFAULT, **(doc.to_dict() or {})}
    except Exception:
        pass
    return dict(RANK_CFG_DEFAULT)

def _score_to_rank(score: int, cfg: dict) -> str:
    if score <= cfg["rank_1_threshold"]: return cfg["rank_1_name"]
    elif score <= cfg["rank_2_threshold"]: return cfg["rank_2_name"]
    elif score <= cfg["rank_3_threshold"]: return cfg["rank_3_name"]
    else: return cfg["rank_4_name"]

def _rank_next_pt(rank_name: str, score: int, cfg: dict) -> str:
    if rank_name == cfg["rank_4_name"]: return "達成済"
    thresholds = {
        cfg["rank_1_name"]: cfg["rank_1_threshold"] + 1,
        cfg["rank_2_name"]: cfg["rank_2_threshold"] + 1,
        cfg["rank_3_name"]: cfg["rank_3_threshold"] + 1,
    }
    return f"{thresholds.get(rank_name, cfg['rank_1_threshold'] + 1) - score} pt"

@router.get("/stats")
def get_user_stats(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() or {} if snap.exists else {}
    except Exception:
        d = {}
    level_score = int(d.get("level_score", 0))
    cfg = _load_rank_config(tenant_id)
    rank_name = _score_to_rank(level_score, cfg)
    next_pt = _rank_next_pt(rank_name, level_score, cfg)
    # DMスコアをチャット利用データからリアルタイム計算
    dm = None
    try:
        _DEFAULT_WEIGHTS = {"Q": 30, "R": 20, "S": 15, "V": 15, "P": 10, "E": 10}
        _w = {k: float(v) / 100.0 for k, v in _DEFAULT_WEIGHTS.items()}

        _logs_raw = list(db.collection("usage_logs").where("user_id","==",uid).limit(200).stream())
        _logs = [l for l in _logs_raw if not (l.to_dict() or {}).get("is_admin_test")][:60]
        turn_count = len(_logs)
        avg_prompt_len = 0.0
        session_count = 0
        unique_keywords = 0
        if _logs:
            _prompts = []
            _dates = set()
            for _l in _logs:
                _d = _l.to_dict() or {}
                _p = str(_d.get("prompt","")).strip()
                if _p: _prompts.append(_p)
                _ts = _d.get("timestamp")
                if _ts:
                    if hasattr(_ts, "strftime"):
                        _dates.add(_ts.strftime("%Y-%m-%d"))
                    elif isinstance(_ts, str) and len(_ts) >= 10:
                        _dates.add(_ts[:10])
            if _prompts:
                avg_prompt_len = sum(len(p) for p in _prompts) / len(_prompts)
                unique_keywords = len(set(" ".join(_prompts).split()))
            session_count = len(_dates)

        # Q: level_score基準
        if level_score >= 451:   dq = 90.0
        elif level_score >= 201: dq = 75.0 + (level_score-201)/250*15.0
        elif level_score >= 81:  dq = 60.0 + (level_score-81)/120*15.0
        else:                    dq = 50.0 + level_score/80*10.0
        dq = round(min(dq, 95.0), 1)
        # R: 語彙多様性
        rt = 85.0 if unique_keywords>=200 else 65.0 if unique_keywords>=80 else 55.0 if unique_keywords>=20 else 45.0
        # S: 平均プロンプト長
        si = 85.0 if avg_prompt_len>=150 else 65.0 if avg_prompt_len>=60 else 55.0 if avg_prompt_len>=20 else 45.0
        # V: ターン数
        dv = 85.0 if turn_count>=36 else 70.0 if turn_count>=18 else 60.0 if turn_count>=6 else 50.0
        # P: セッション日数
        pa = 85.0 if session_count>=10 else 70.0 if session_count>=4 else 60.0 if session_count>=2 else 50.0
        # E: 利用密度
        _e_raw = min(turn_count * avg_prompt_len / 1000.0, 100.0)
        ec = 85.0 if _e_raw>=15 else 70.0 if _e_raw>=5 else 55.0 if _e_raw>=1 else 45.0

        total = round(_w["Q"]*dq + _w["R"]*rt + _w["S"]*si + _w["V"]*dv + _w["P"]*pa + _w["E"]*ec, 1)
        rank_th = [("S",90),("A+",82),("A",78),("A-",74),("B+",70),("B",65),("B-",60),("C+",55),("C",50)]
        dr = "D"
        for _rk, _t in rank_th:
            if total >= _t: dr = _rk; break

        dm = {
            "decision_quality_score":  dq,
            "risk_tolerance":          rt,
            "structural_intelligence": si,
            "decision_velocity":       dv,
            "prediction_accuracy":     pa,
            "execution_consistency":   ec,
            "diagnosis_total_score":   total,
            "diagnosis_rank":          dr,
        }
    except Exception:
        pass
    use_count = int(d.get("use_count_since_report", 0))
    fc_report_threshold = 12
    diag_count = 0
    try:
        snaps = list(db.collection("user_diagnoses").where("uid", "==", uid).stream())
        diag_count = len(snaps)
    except Exception:
        pass
    total_chat_count = 0
    try:
        logs = list(
            db.collection("usage_logs")
            .where("user_id", "==", uid)
            .limit(500)
            .stream()
        )
        total_chat_count = len([l for l in logs if not (l.to_dict() or {}).get("is_admin_test")])
    except Exception:
        pass
    diag_window = 12
    diag_checkpoint = (total_chat_count // diag_window) * diag_window
    diag_hist_checkpoints = []
    try:
        hist = list(db.collection("user_diagnoses").where("uid", "==", uid).stream())
        diag_hist_checkpoints = [int((h.to_dict() or {}).get("n_chats", 0)) for h in hist]
    except Exception:
        pass
    last_diag_checkpoint = max(diag_hist_checkpoints) if diag_hist_checkpoints else 0
    diag_available = (diag_checkpoint >= diag_window) and (diag_checkpoint > last_diag_checkpoint)
    diag_next_unlock = diag_checkpoint + diag_window
    return {
        "uid": uid,
        "level_score": level_score,
        "rank_name": rank_name,
        "next_pt": next_pt,
        "rank_cfg": {
            "rank_1_name": cfg["rank_1_name"],
            "rank_2_name": cfg["rank_2_name"],
            "rank_3_name": cfg["rank_3_name"],
            "rank_4_name": cfg["rank_4_name"],
        },
        "decision_metrics": dm,
        "use_count_since_report": use_count,
        "fc_report_unlocked": use_count >= fc_report_threshold,
        "fc_report_threshold": fc_report_threshold,
        "diagnosis_count": diag_count,
        "total_chat_count": total_chat_count,
        "diag_available": diag_available,
        "diag_next_unlock": diag_next_unlock,
        "diag_checkpoint": diag_checkpoint,
        "fixed_concept_score": d.get("fixed_concept_score", None),
        "is_unlimited": True if payload.get("role") == "admin" else bool(d.get("is_unlimited", False)),
        "level_last_delta": int(d.get("level_last_delta", 0)),
        "expires_at": str(d.get("expires_at", "")),
        "tenant_id": tenant_id,
    }

@router.get("/usage_logs")
def get_usage_logs(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        docs = list(db.collection("usage_logs").where("user_id", "==", uid).limit(500).stream())
        logs = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_admin_test"):
                continue
            logs.append({
                "prompt": str(data.get("prompt", ""))[:100],
                "timestamp": str(data.get("timestamp", "")),
            })
        def _to_jst(t):
            from datetime import datetime, timedelta
            t = str(t or "").strip()
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f"]:
                try:
                    dt = datetime.strptime(t[:26], fmt[:len(t[:26])] if len(fmt) > len(t[:26]) else fmt)
                    if "+00:00" in t or t.endswith("Z") or "T" in t:
                        dt = dt + timedelta(hours=9)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            return t
        for lg in logs:
            lg["timestamp"] = _to_jst(lg["timestamp"])
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"logs": logs[:50]}
    except Exception:
        return {"logs": []}

@router.delete("/session/{chat_id}")
def delete_session(chat_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    doc_id = f"user__{tenant_id}__{uid}__{chat_id}"
    try:
        db.collection("chat_sessions").document(doc_id).update({"is_deleted": True})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/session/{chat_id}/rename")
def rename_session(chat_id: str, body: dict, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    doc_id = f"user__{tenant_id}__{uid}__{chat_id}"
    try:
        db.collection("chat_sessions").document(doc_id).update({"title": body.get("title", chat_id)})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/header_config")
def get_header_config(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    DEFAULT = {
        "title": "Ys Consulting Office",
        "subtitle": "自己変容を通じて、意思決定精度を高めるトレーニング領域。",
        "point_1_label": "現在地の可視化",
        "point_1_body": "発言の質・構造理解・行動履歴を計測し、あなたの思考階層を明示します。",
        "point_2_label": "3層分析",
        "point_2_body": "構造（力学）／戦術（実行）／マインド（姿勢）の3軸から最適解を提示。",
        "point_3_label": "設計者化",
        "point_3_body": "他者依存ではなく、自ら意思決定を設計できる状態へ導きます。",
    }
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"ascend_header_config_{tid}").get()
            if doc.exists:
                return {**DEFAULT, **(doc.to_dict() or {})}
    except Exception:
        pass
    return DEFAULT

@router.get("/user_guide")
def get_user_guide(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("tenant_settings").document(tid).get()
            if doc.exists:
                guide = (doc.to_dict() or {}).get("guide", "")
                if guide and "{mode_guide_section}" not in guide:
                    return {"guide": guide}
    except Exception:
        pass
    return {"guide": "## ASCEND 使い方ガイド\n\n- 具体的な数字・目標・課題を伝えると精度が上がります\n- 構造的・戦略的な言語で問いかけるほどスコアが上がります\n- 問題の原因・打ち手・優先順位を明確にする\n- 数値目標を設定して進捗を確認する"}

@router.get("/fc_report")
def get_fc_report(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    from api.core.features import is_feature_enabled
    if not is_feature_enabled(uid, "fixed_concept_report"):
        return {"report": None, "use_count_since_report": 0, "locked": True}
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() or {} if snap.exists else {}
        use_count = int(d.get("use_count_since_report", 0))
        existing = d.get("latest_fc_report")
        fc_threshold = 12
        if use_count >= fc_threshold and not existing:
            try:
                from api.core.llm_client import call_llm as _cllm
                from api.core.firestore_client import get_db as _gdb
                import re as _re, json as _json
                chat_docs = []
                try:
                    sessions = list(_gdb().collection("chat_sessions")
                        .where("uid","==",uid).where("scope","==","user").limit(5).stream())
                    for s in sessions:
                        msgs = list(s.reference.collection("messages").order_by("ts").limit(20).stream())
                        chat_docs.extend([m.to_dict().get("content","") for m in msgs if m.to_dict().get("role")=="user"])
                except Exception:
                    pass
                history_text = "\n".join(chat_docs[:30])[:3000]
                fc_prompt = f"""以下のユーザーのチャット履歴から固定概念（思考の癖・バイアス・盲点）を分析してレポートを生成せよ。JSONのみ出力。
【履歴】{history_text}
出力形式: {{"report_text":"Markdownレポート本文","fixed_patterns":["固定パターン1","固定パターン2"],"blind_spots":["盲点1"],"growth_suggestions":["成長提案1"],"fc_score":65}}"""
                res = _cllm(
                    system_prompt="深層心理と認知バイアスの専門家。JSONのみ出力。",
                    messages=[{"role":"user","content":fc_prompt}],
                    ai_tier="core", max_tokens=1024
                )
                m = _re.search(r'\{.*\}', res, _re.DOTALL)
                fc_data = _json.loads(m.group(0)) if m else {"report_text": res}
                db.collection("users").document(uid).set({
                    "latest_fc_report": fc_data,
                    "use_count_since_report": 0,
                    "fc_report_generated_at": __import__("datetime").datetime.now().isoformat(),
                }, merge=True)
                return {"report": fc_data, "use_count_since_report": 0}
            except Exception:
                pass
        return {
            "report": existing,
            "use_count_since_report": use_count,
        }
    except Exception:
        return {"report": None, "use_count_since_report": 0}

@router.get("/rankup_tips")
def get_rankup_tips(payload: dict = Depends(verify_token)):
    content = """# 🏆 ランクアップ完全攻略

## ランクとは何か

ASCENDのランクは「ゲームの勝敗」ではありません。
あなたの**意思決定の構造的精度**をリアルタイムで計測するシステムです。
スコアが上がるほどAIはより高度な分析を返します。つまりランクは、あなたへの投資収益率です。

---

## スコアの設計思想

ASCENDは「何を言ったか」ではなく「どう考えたか」を評価します。

| 評価軸 | 配点 | 意味 |
|---|---|---|
| 構造キーワード | +3pt | 問題を構造として把握している証拠 |
| 戦略キーワード | +2pt | 選択肢と優先順位を設計している証拠 |
| 実行キーワード | +1pt | 判断を行動に落とし込んでいる証拠 |
| 感情的語彙 | -3pt | 判断ではなく感情で動いているサイン |

代表キーワード例
- 構造：構造・資本・市場・制度・最適化・期待値・アーキテクチャ・因果
- 戦略：差別化・競合・ポジショニング・KPI・ROI・仮説・検証
- 実行：実行・手順・タスク・改善・フェーズ・マイルストーン

---

## 最速でランクを上げる5つの原則

## 原則1：問いを構造化してから送る

感情や直感のまま送ることは最もスコアを下げます。
送信前に「目的・現状・制約・出力形式」の4軸を揃えてください。

【目的】3ヶ月以内に月商を300万から500万にする
【現状】客単価8,000円・月間客数375名・新規:リピート=7:3
【制約】追加投資不可・スタッフ増員なし・値下げ禁止
【出力形式】優先度付き施策3つ＋各KPIと計測方法

この構造で送ると、AIは施策の現実性を正確に評価できます。

---

## 原則2：数字で観測・数字で判断する

「売上が伸びない」は観測ではありません。
「先月比-15%・3ヶ月連続・新規顧客の離脱率が上昇」が観測です。

ビジネスの問題はすべて数字に変換できます。変換できない問題は、まだ定義されていない問題です。
ASCENDは数字が入った相談ほど精度の高い分析を返します。

---

## 原則3：成功条件と失敗条件を両方書く

戦略的思考の核心は「何が起きれば成功か」と「何が起きれば撤退するか」を事前に定義することです。

- 成功条件：月商500万達成かつリピート率60%以上を3ヶ月維持
- 失敗条件：広告費がCPA3,000円を超えた時点で施策変更

この反証思考がスコアに最も大きく影響します。

---

## 原則4：目的モードを使い分ける

同じ問題でもモードを変えると異なる切り口の分析が得られます。

| 場面 | 推奨モード | 理由 |
|---|---|---|
| 数値が改善しない | NUMERIC | KPIの構造的問題を特定 |
| チームが動かない | CONTROL | 権限・フロー・構造の問題を特定 |
| 競合に負けている | STRATEGY | ポジショニングと差別化を再設計 |
| 同じ失敗を繰り返す | DIAGNOSIS | 根本原因と思考パターンを特定 |
| 重要な意思決定がある | RISK | リスクと想定外を事前に洗い出す |

---

## 原則5：継続がスコアの基盤をつくる

Decision Metricsの「P（予測精度）」はセッション継続日数に直結します。
単発の高品質な相談より、週3回の継続的な相談の方が総合スコアは高くなります。

12回の利用で現状課題診断が解放されます。これはあなたの思考パターンをAIが解析した診断レポートです。
継続することで、自分では気づけない判断の癖と盲点が可視化されます。

---

## Decision Metricsとランクの関係

ランクスコアとDecision Metricsは連動しています。
どの指標が低いかを確認して、強化すべき領域を特定してください。

| 指標が低い場合 | 改善アクション |
|---|---|
| Q（意思決定精度）が低い | 構造キーワードを意識して問いを設計し直す |
| R（リスク耐性）が低い | 失敗条件・撤退基準を毎回明示する |
| S（構造理解）が低い | 問題を「原因・構造・影響」の3層で分解する |
| V（判断速度）が低い | 選択肢を2〜3に絞り、判断軸を事前に定義する |
| P（予測精度）が低い | 利用頻度を週3回以上に増やす |
| E（実行一貫性）が低い | 前回の判断を振り返り、実行状況を報告してから次の相談をする |
"""
    return {"content": content}

@router.get("/manual")
def get_manual(payload: dict = Depends(verify_token)):
    content = """# 📖 ASCEND 完全マニュアル

## ASCENDが解決する本質的な問題

多くの経営者・事業者が直面する課題は同じです。
「誰に相談すればいいかわからない」「コンサルタントは高すぎる」「情報はあるが判断できない」。
ASCENDはその空白を埋めるために設計されました。

戦略・数値・構造・リスク——あらゆる経営判断に即応するAIコンサルティングエンジンです。
単なる情報提供ではなく、あなたが自ら意思決定できる状態へ引き上げることが目的です。

---

## 1. チャットモード — 目的に合わせて使い分ける

| モード | アイコン | 特徴 | 向いている場面 |
|---|---|---|---|
| 会話モード | 💬 | 柔らかく対話しながら考えを整理 | アイデア出し・情報収集・気軽な相談 |
| 相談モード | 🎯 | 構造的・戦略的な分析と提言 | 意思決定・施策設計・問題解決 |

入力欄左下のアイコンで切り替えられます。迷ったら相談モード（🎯）を使ってください。

---

## 2. 戦略相談（メインチャット）— ASCENDの中核

### 入力の黄金フォーマット

【目的】達成したい数値目標・状態（例：3ヶ月以内に月商500万達成）
【現状】観測できる数値・事実（例：現在300万、スタッフ3名、広告費5万/月）
【制約】予算・期限・変えられない条件（例：値下げ不可、追加採用なし）
【出力形式】欲しい形式（例：優先度付き施策リスト＋各KPIと計測方法）

制約を書くことで実行可能な施策に変わります。制約がないと理論上最適な施策を並べるだけになります。

### 目的モード一覧

| モード | 用途 | 代表的な活用シーン |
|---|---|---|
| AUTO | AIが文脈から自動判別 | 迷ったらこれ |
| NUMERIC | 数値・KPI・売上・コスト分析 | 客単価を上げる施策を数値で整理したい |
| GROWTH | スキル・習慣・成長設計 | 営業力を3ヶ月で体系的に鍛えたい |
| CONTROL | 組織・権限・業務フロー構造化 | 属人化を解消してチームを動かしたい |
| STRATEGY | 競合分析・差別化・ポジション設計 | 価格競争から脱却する戦略を作りたい |
| ANALYSIS | データ・事象の多角的解析 | なぜこの指標が下がっているか分析したい |
| PLANNING | ロードマップ・フェーズ設計 | 半年後の状態から逆算して計画を立てたい |
| RISK | リスク特定・評価・対策設計 | この意思決定に潜むリスクを洗い出したい |
| NEGOTIATION | 交渉・説得・合意形成戦略 | 価格交渉・条件変更を通すための準備をしたい |
| MARKETING | 集客・ブランディング・広告施策 | SNSと広告を組み合わせた新規獲得戦略が欲しい |
| DIAGNOSIS | 現状課題の発見・根本原因分析 | 売上が下がり続ける本当の原因を特定したい |
| FORECAST | 将来予測・シナリオ分析 | 3パターンのシナリオで6ヶ月後を予測したい |
| FINANCE | 財務・投資・資金計画分析 | この投資の回収期間とリターンを試算したい |
| HR | 採用・評価・組織設計・人材育成 | 評価制度を設計して離職率を下げたい |
| CREATIVE | アイデア発想・コンセプト設計 | 競合と全く違う切り口でサービスを再設計したい |
| SUMMARY | 要約・情報整理 | 長文や複数情報を簡潔にまとめたい |
| COACHING | コーチング・成長支援 | 思考の癖を発見して行動変容を促したい |
| LEGAL | 法務・規約・契約 | 契約書のリスクや法的注意点を確認したい |
| OPS | 業務改善・効率化 | 現場フローのボトルネックを特定して改善したい |
| TECH | 技術・エンジニアリング | システム設計や技術的意思決定の壁打ちをしたい |

---

## 3. AIエンジン：判断の重さに合わせて使い分ける

| エンジン | 処理特性 | 最適な用途 |
|---|---|---|
| SWIFT（迅速） | 高速レスポンス・日常的な戦略相談 | 毎日の意思決定・アイデア出し・施策整理 |
| ADVANCE（高度） | 深い文脈理解・複雑な因果関係の分析 | 重要な戦略判断・資金調達・組織改革 |
| SUPREME（至高） | 最高精度・多層的な問題構造の解体 | 事業再構築・M&A検討・大型案件の意思決定 |

ADVANCE・SUPREMEは管理者からの権限付与が必要です。重要な判断ほど上位エンジンを使うことで見落としリスクが大幅に低減します。

---

## 4. ファイル診断 — 数値データを戦略に変換する

ExcelやCSV・PDF等のファイルを添付するだけで、AIが自動解析します。

| 対応形式 | xlsx / xls / csv / txt / pdf / md |
|---|---|
| 解析内容 | 全シート横断・数値トレンド・異常値検出・課題特定 |
| 使用エンジン | Ultra（高精度モデル）を自動適用 |

### 活用フロー
1. 診断タブ →「ファイル診断」を選択
2. ファイルをアップロード
3. AIが専門用語・数値構造を確認（双方向チャットで精度向上）
4. 全シート横断の診断レポートを生成
5. 追加質問で深掘り可能

数式・条件分岐・異常値まで自動検出します。「なぜこの数字になっているか」の根本原因まで追います。

---

## 5. データ分析（テーブル操作）— コマンドで即分析

CSVやExcelを添付すると自動でテーブルモードに切り替わります。

| コマンド | 機能 | 活用例 |
|---|---|---|
| /rank 列名 | 降順ランキング化 | /rank 売上 → 店舗別売上ランキング |
| /filter 条件 | 条件フィルタ抽出 | /filter 売上 >= 100 → 高売上店舗のみ抽出 |
| /derive 式 | 派生指標の追加 | /derive 客単価=売上/客数 → 新指標を即生成 |
| /top N 列 | 上位N件抽出 | /top 5 売上 → 売上上位5店舗 |
| /consult | AIによる総合戦略分析 | テーブル全体をAIが分析しインサイトを提示 |

/consultが最も強力です。数字の羅列を経営判断に使えるインサイトに変換します。

---

## 6. ランクシステム — 思考の設計力を可視化する

| 評価軸 | 配点 | 代表キーワード |
|---|---|---|
| 構造キーワード | +3pt | 構造・資本・市場・制度・最適化・期待値・アーキテクチャ |
| 戦略キーワード | +2pt | 戦略・差別化・競合・ポジショニング・KPI・ROI |
| 実行キーワード | +1pt | 実行・手順・タスク・改善・運用・効率 |
| 感情的語彙 | -3pt | 不安・ムカつく・どうせ・無理・クソ |

---

## 7. 診断機能 — 自分では気づけない盲点を暴く

### 現状課題診断
- チャット12回ごとに生成可能
- 直近の会話履歴をAIが解析し、思考パターン・判断の癖・未解決の課題構造を診断

### 固定概念レポート
- 12回利用後に解放
- 思考パターンのバイアス・盲点・成長の阻害要因を可視化

### Decision Metrics（意思決定精度診断）

| 指標 | 意味 | 低い場合の影響 |
|---|---|---|
| Q 意思決定精度 | 判断の質・構造的思考力 | 感情・直感で重要決定をしてしまう |
| R リスク耐性 | リスクを定量化する能力 | 過度な慎重さまたは無謀な判断 |
| S 構造理解 | 問題の構造を把握する力 | 表面的な対処に終始する |
| V 判断速度 | 適切なスピードで判断する力 | 機会損失または拙速な決断 |
| P 予測精度 | 未来を見通す思考の精度 | 短期思考・場当たり的な対応 |
| E 実行一貫性 | 判断と実行の整合性 | 戦略と現場のズレが常態化 |

---

## 8. コンサル解析機能 — プロのフレームワークをAIで即実行

| 機能 | 何ができるか | 活用シーン |
|---|---|---|
| 構造診断 | 事業・組織・戦略の力学構造を解剖しボトルネックを特定 | なぜ施策を打っても改善しないのかの解明 |
| 課題仮説 | 状況から複数の課題仮説を優先度付きで生成 | 問題の全体像を整理して手を打つ順番を決める |
| 比較分析 | A案/B案/C案を複数の評価軸で客観的に比較 | 迷っている選択肢を構造的に整理する |
| 矛盾検知 | 戦略・方針・行動間の矛盾と整合性を検証 | 言っていることとやっていることが違うを発見 |
| 実行計画 | フェーズ別・期限付きのアクションプランを生成 | 戦略を実際に動かせる計画に落とし込む |
| 投資シグナル | 反発底打ち候補・大口売り込み監視・個別銘柄分析 | 資金配分・参入タイミングの判断材料 |
| ファイル診断 | Excel・CSV・PDFを全シート横断で自動解析 | 数値データの課題特定・改善提言 |

---

## 9. 個人相談（DM） — AIを超えた案件はコンサルタントへ直接

複雑な人間関係・経営危機・事業承継・機密性の高い戦略判断は、コンサルタントに直接相談できます。
マイページの「個人相談」タブから送信してください。

---

## 10. 最大限活用するための3原則

原則1：具体的な数字で話す
「売上が下がった」ではなく「先月比-15%、3ヶ月連続、特に新規顧客の離脱率が高い」。
数字が入ると、AIの分析精度が根本的に変わります。

原則2：制約を必ず書く
「予算なし」「人員増員不可」「値下げ禁止」——制約を書くことで施策が現実的になります。
制約のない戦略提案は実行不可能な理想論になります。

原則3：診断機能を定期的に使う
12回のチャットごとに現状課題診断を実行してください。
自分では気づけない思考パターンの偏りと成長の停滞要因を定期的に洗い出すことが、
長期的な意思決定精度の向上につながります。
"""
    return {"content": content}

RANK_CFG_DEFAULT = {
    "rank_1_name": "追従者", "rank_1_threshold": 80,
    "rank_2_name": "実行者", "rank_2_threshold": 200,
    "rank_3_name": "戦略家", "rank_3_threshold": 450,
    "rank_4_name": "設計者",
}

ALL_PURPOSE_MODES = {
    "auto":        "AUTO（自動判別）",
    "numeric":     "NUMERIC（数字/指標）",
    "growth":      "GROWTH（成長/訓練）",
    "control":     "CONTROL（支配/構造図）",
    "strategy":    "STRATEGY（戦略立案）",
    "analysis":    "ANALYSIS（分析/解析）",
    "planning":    "PLANNING（計画/ロードマップ）",
    "risk":        "RISK（リスク評価）",
    "creative":    "CREATIVE（創造/アイデア）",
    "summary":     "SUMMARY（要約/整理）",
    "negotiation": "NEGOTIATION（交渉/説得）",
    "coaching":    "COACHING（コーチング）",
    "diagnosis":   "DIAGNOSIS（診断/課題発見）",
    "forecast":    "FORECAST（予測/シナリオ）",
    "legal":       "LEGAL（法務/規約）",
    "finance":     "FINANCE（財務/投資）",
    "marketing":   "MARKETING（マーケ/集客）",
    "hr":          "HR（人材/組織）",
    "ops":         "OPS（業務改善/効率化）",
    "tech":        "TECH（技術/エンジニア）",
}

def _load_rank_config(tenant_id: str) -> dict:
    try:
        db = get_db()
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"rank_config_{tid}").get()
            if doc.exists:
                return {**RANK_CFG_DEFAULT, **(doc.to_dict() or {})}
    except Exception:
        pass
    return dict(RANK_CFG_DEFAULT)

def _score_to_rank(score: int, cfg: dict) -> str:
    if score <= cfg["rank_1_threshold"]: return cfg["rank_1_name"]
    elif score <= cfg["rank_2_threshold"]: return cfg["rank_2_name"]
    elif score <= cfg["rank_3_threshold"]: return cfg["rank_3_name"]
    else: return cfg["rank_4_name"]

def _rank_next_pt(rank_name: str, score: int, cfg: dict) -> str:
    if rank_name == cfg["rank_4_name"]: return "達成済"
    thresholds = {
        cfg["rank_1_name"]: cfg["rank_1_threshold"] + 1,
        cfg["rank_2_name"]: cfg["rank_2_threshold"] + 1,
        cfg["rank_3_name"]: cfg["rank_3_threshold"] + 1,
    }
    return f"{thresholds.get(rank_name, cfg['rank_1_threshold'] + 1) - score} pt"

@router.get("/stats")
def get_user_stats(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() or {} if snap.exists else {}
    except Exception:
        d = {}
    level_score = int(d.get("level_score", 0))
    cfg = _load_rank_config(tenant_id)
    rank_name = _score_to_rank(level_score, cfg)
    next_pt = _rank_next_pt(rank_name, level_score, cfg)
    # DMスコアをチャット利用データからリアルタイム計算
    dm = None
    try:
        _DEFAULT_WEIGHTS = {"Q": 30, "R": 20, "S": 15, "V": 15, "P": 10, "E": 10}
        _w = {k: float(v) / 100.0 for k, v in _DEFAULT_WEIGHTS.items()}

        _logs_raw = list(db.collection("usage_logs").where("user_id","==",uid).limit(200).stream())
        _logs = [l for l in _logs_raw if not (l.to_dict() or {}).get("is_admin_test")][:60]
        turn_count = len(_logs)
        avg_prompt_len = 0.0
        session_count = 0
        unique_keywords = 0
        if _logs:
            _prompts = []
            _dates = set()
            for _l in _logs:
                _d = _l.to_dict() or {}
                _p = str(_d.get("prompt","")).strip()
                if _p: _prompts.append(_p)
                _ts = _d.get("timestamp")
                if _ts and hasattr(_ts,"strftime"): _dates.add(_ts.strftime("%Y-%m-%d"))
            if _prompts:
                avg_prompt_len = sum(len(p) for p in _prompts) / len(_prompts)
                unique_keywords = len(set(" ".join(_prompts).split()))
            session_count = len(_dates)

        # Q: level_score基準
        if level_score >= 451:   dq = 90.0
        elif level_score >= 201: dq = 75.0 + (level_score-201)/250*15.0
        elif level_score >= 81:  dq = 60.0 + (level_score-81)/120*15.0
        else:                    dq = 50.0 + level_score/80*10.0
        dq = round(min(dq, 95.0), 1)
        # R: 語彙多様性
        rt = 85.0 if unique_keywords>=200 else 65.0 if unique_keywords>=80 else 55.0 if unique_keywords>=20 else 45.0
        # S: 平均プロンプト長
        si = 85.0 if avg_prompt_len>=150 else 65.0 if avg_prompt_len>=60 else 55.0 if avg_prompt_len>=20 else 45.0
        # V: ターン数
        dv = 85.0 if turn_count>=36 else 70.0 if turn_count>=18 else 60.0 if turn_count>=6 else 50.0
        # P: セッション日数
        pa = 85.0 if session_count>=10 else 70.0 if session_count>=4 else 60.0 if session_count>=2 else 50.0
        # E: 利用密度
        _e_raw = min(turn_count * avg_prompt_len / 1000.0, 100.0)
        ec = 85.0 if _e_raw>=15 else 70.0 if _e_raw>=5 else 55.0 if _e_raw>=1 else 45.0

        total = round(_w["Q"]*dq + _w["R"]*rt + _w["S"]*si + _w["V"]*dv + _w["P"]*pa + _w["E"]*ec, 1)
        rank_th = [("S",90),("A+",82),("A",78),("A-",74),("B+",70),("B",65),("B-",60),("C+",55),("C",50)]
        dr = "D"
        for _rk, _t in rank_th:
            if total >= _t: dr = _rk; break

        dm = {
            "decision_quality_score":  dq,
            "risk_tolerance":          rt,
            "structural_intelligence": si,
            "decision_velocity":       dv,
            "prediction_accuracy":     pa,
            "execution_consistency":   ec,
            "diagnosis_total_score":   total,
            "diagnosis_rank":          dr,
        }
    except Exception:
        pass
    use_count = int(d.get("use_count_since_report", 0))
    fc_report_threshold = 12
    diag_count = 0
    try:
        snaps = list(db.collection("user_diagnoses").where("uid", "==", uid).stream())
        diag_count = len(snaps)
    except Exception:
        pass
    total_chat_count = 0
    try:
        logs = list(
            db.collection("usage_logs")
            .where("user_id", "==", uid)
            .limit(500)
            .stream()
        )
        total_chat_count = len([l for l in logs if not (l.to_dict() or {}).get("is_admin_test")])
    except Exception:
        pass
    diag_window = 12
    diag_checkpoint = (total_chat_count // diag_window) * diag_window
    diag_hist_checkpoints = []
    try:
        hist = list(db.collection("user_diagnoses").where("uid", "==", uid).stream())
        diag_hist_checkpoints = [int((h.to_dict() or {}).get("n_chats", 0)) for h in hist]
    except Exception:
        pass
    last_diag_checkpoint = max(diag_hist_checkpoints) if diag_hist_checkpoints else 0
    diag_available = (diag_checkpoint >= diag_window) and (diag_checkpoint > last_diag_checkpoint)
    diag_next_unlock = diag_checkpoint + diag_window
    return {
        "uid": uid,
        "level_score": level_score,
        "rank_name": rank_name,
        "next_pt": next_pt,
        "rank_cfg": {
            "rank_1_name": cfg["rank_1_name"],
            "rank_2_name": cfg["rank_2_name"],
            "rank_3_name": cfg["rank_3_name"],
            "rank_4_name": cfg["rank_4_name"],
        },
        "decision_metrics": dm,
        "use_count_since_report": use_count,
        "fc_report_unlocked": use_count >= fc_report_threshold,
        "fc_report_threshold": fc_report_threshold,
        "diagnosis_count": diag_count,
        "total_chat_count": total_chat_count,
        "diag_available": diag_available,
        "diag_next_unlock": diag_next_unlock,
        "diag_checkpoint": diag_checkpoint,
        "fixed_concept_score": d.get("fixed_concept_score", None),
        "is_unlimited": True if payload.get("role") == "admin" else bool(d.get("is_unlimited", False)),
        "level_last_delta": int(d.get("level_last_delta", 0)),
        "expires_at": str(d.get("expires_at", "")),
        "tenant_id": tenant_id,
    }

@router.get("/usage_logs")
def get_usage_logs(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        docs = list(db.collection("usage_logs").where("user_id", "==", uid).limit(500).stream())
        logs = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_admin_test"):
                continue
            logs.append({
                "prompt": str(data.get("prompt", ""))[:100],
                "timestamp": str(data.get("timestamp", "")),
            })
        def _to_jst(t):
            from datetime import datetime, timedelta
            t = str(t or "").strip()
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f"]:
                try:
                    dt = datetime.strptime(t[:26], fmt[:len(t[:26])] if len(fmt) > len(t[:26]) else fmt)
                    if "+00:00" in t or t.endswith("Z") or "T" in t:
                        dt = dt + timedelta(hours=9)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            return t
        for lg in logs:
            lg["timestamp"] = _to_jst(lg["timestamp"])
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"logs": logs[:50]}
    except Exception:
        return {"logs": []}

@router.delete("/session/{chat_id}")
def delete_session(chat_id: str, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    doc_id = f"user__{tenant_id}__{uid}__{chat_id}"
    try:
        db.collection("chat_sessions").document(doc_id).update({"is_deleted": True})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/session/{chat_id}/rename")
def rename_session(chat_id: str, body: dict, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    doc_id = f"user__{tenant_id}__{uid}__{chat_id}"
    try:
        db.collection("chat_sessions").document(doc_id).update({"title": body.get("title", chat_id)})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/header_config")
def get_header_config(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    DEFAULT = {
        "title": "Ys Consulting Office",
        "subtitle": "自己変容を通じて、意思決定精度を高めるトレーニング領域。",
        "point_1_label": "現在地の可視化",
        "point_1_body": "発言の質・構造理解・行動履歴を計測し、あなたの思考階層を明示します。",
        "point_2_label": "3層分析",
        "point_2_body": "構造（力学）／戦術（実行）／マインド（姿勢）の3軸から最適解を提示。",
        "point_3_label": "設計者化",
        "point_3_body": "他者依存ではなく、自ら意思決定を設計できる状態へ導きます。",
    }
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"ascend_header_config_{tid}").get()
            if doc.exists:
                return {**DEFAULT, **(doc.to_dict() or {})}
    except Exception:
        pass
    return DEFAULT

@router.get("/user_guide")
def get_user_guide(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("tenant_settings").document(tid).get()
            if doc.exists:
                guide = (doc.to_dict() or {}).get("guide", "")
                if guide and "{mode_guide_section}" not in guide:
                    return {"guide": guide}
    except Exception:
        pass
    return {"guide": "## ASCEND 使い方ガイド\n\n- 具体的な数字・目標・課題を伝えると精度が上がります\n- 構造的・戦略的な言語で問いかけるほどスコアが上がります\n- 問題の原因・打ち手・優先順位を明確にする\n- 数値目標を設定して進捗を確認する"}

@router.get("/fc_report")
def get_fc_report(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    from api.core.features import is_feature_enabled
    if not is_feature_enabled(uid, "fixed_concept_report"):
        return {"report": None, "use_count_since_report": 0, "locked": True}
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() or {} if snap.exists else {}
        use_count = int(d.get("use_count_since_report", 0))
        existing = d.get("latest_fc_report")
        fc_threshold = 12
        if use_count >= fc_threshold and not existing:
            try:
                from api.core.llm_client import call_llm as _cllm
                from api.core.firestore_client import get_db as _gdb
                import re as _re, json as _json
                chat_docs = []
                try:
                    sessions = list(_gdb().collection("chat_sessions")
                        .where("uid","==",uid).where("scope","==","user").limit(5).stream())
                    for s in sessions:
                        msgs = list(s.reference.collection("messages").order_by("ts").limit(20).stream())
                        chat_docs.extend([m.to_dict().get("content","") for m in msgs if m.to_dict().get("role")=="user"])
                except Exception:
                    pass
                history_text = "\n".join(chat_docs[:30])[:3000]
                fc_prompt = f"""以下のユーザーのチャット履歴から固定概念（思考の癖・バイアス・盲点）を分析してレポートを生成せよ。JSONのみ出力。
【履歴】{history_text}
出力形式: {{"report_text":"Markdownレポート本文","fixed_patterns":["固定パターン1","固定パターン2"],"blind_spots":["盲点1"],"growth_suggestions":["成長提案1"],"fc_score":65}}"""
                res = _cllm(
                    system_prompt="深層心理と認知バイアスの専門家。JSONのみ出力。",
                    messages=[{"role":"user","content":fc_prompt}],
                    ai_tier="core", max_tokens=1024
                )
                m = _re.search(r'\{.*\}', res, _re.DOTALL)
                fc_data = _json.loads(m.group(0)) if m else {"report_text": res}
                db.collection("users").document(uid).set({
                    "latest_fc_report": fc_data,
                    "use_count_since_report": 0,
                    "fc_report_generated_at": __import__("datetime").datetime.now().isoformat(),
                }, merge=True)
                return {"report": fc_data, "use_count_since_report": 0}
            except Exception:
                pass
        return {
            "report": existing,
            "use_count_since_report": use_count,
        }
    except Exception:
        return {"report": None, "use_count_since_report": 0}

@router.get("/rankup_tips")
def get_rankup_tips(payload: dict = Depends(verify_token)):
    content = """# 🏆 ランクアップ完全攻略

## ランクとは何か

ASCENDのランクは「ゲームの勝敗」ではありません。
あなたの**意思決定の構造的精度**をリアルタイムで計測するシステムです。
スコアが上がるほどAIはより高度な分析を返します。つまりランクは、あなたへの投資収益率です。

---

## スコアの設計思想

ASCENDは「何を言ったか」ではなく「どう考えたか」を評価します。

| 評価軸 | 配点 | 意味 |
|---|---|---|
| 構造キーワード | +3pt | 問題を構造として把握している証拠 |
| 戦略キーワード | +2pt | 選択肢と優先順位を設計している証拠 |
| 実行キーワード | +1pt | 判断を行動に落とし込んでいる証拠 |
| 感情的語彙 | -3pt | 判断ではなく感情で動いているサイン |

代表キーワード例
- 構造：構造・資本・市場・制度・最適化・期待値・アーキテクチャ・因果
- 戦略：差別化・競合・ポジショニング・KPI・ROI・仮説・検証
- 実行：実行・手順・タスク・改善・フェーズ・マイルストーン

---

## 最速でランクを上げる5つの原則

## 原則1：問いを構造化してから送る

感情や直感のまま送ることは最もスコアを下げます。
送信前に「目的・現状・制約・出力形式」の4軸を揃えてください。

【目的】3ヶ月以内に月商を300万から500万にする
【現状】客単価8,000円・月間客数375名・新規:リピート=7:3
【制約】追加投資不可・スタッフ増員なし・値下げ禁止
【出力形式】優先度付き施策3つ＋各KPIと計測方法

この構造で送ると、AIは施策の現実性を正確に評価できます。

---

## 原則2：数字で観測・数字で判断する

「売上が伸びない」は観測ではありません。
「先月比-15%・3ヶ月連続・新規顧客の離脱率が上昇」が観測です。

ビジネスの問題はすべて数字に変換できます。変換できない問題は、まだ定義されていない問題です。
ASCENDは数字が入った相談ほど精度の高い分析を返します。

---

## 原則3：成功条件と失敗条件を両方書く

戦略的思考の核心は「何が起きれば成功か」と「何が起きれば撤退するか」を事前に定義することです。

- 成功条件：月商500万達成かつリピート率60%以上を3ヶ月維持
- 失敗条件：広告費がCPA3,000円を超えた時点で施策変更

この反証思考がスコアに最も大きく影響します。

---

## 原則4：目的モードを使い分ける

同じ問題でもモードを変えると異なる切り口の分析が得られます。

| 場面 | 推奨モード | 理由 |
|---|---|---|
| 数値が改善しない | NUMERIC | KPIの構造的問題を特定 |
| チームが動かない | CONTROL | 権限・フロー・構造の問題を特定 |
| 競合に負けている | STRATEGY | ポジショニングと差別化を再設計 |
| 同じ失敗を繰り返す | DIAGNOSIS | 根本原因と思考パターンを特定 |
| 重要な意思決定がある | RISK | リスクと想定外を事前に洗い出す |

---

## 原則5：継続がスコアの基盤をつくる

Decision Metricsの「P（予測精度）」はセッション継続日数に直結します。
単発の高品質な相談より、週3回の継続的な相談の方が総合スコアは高くなります。

12回の利用で現状課題診断が解放されます。これはあなたの思考パターンをAIが解析した診断レポートです。
継続することで、自分では気づけない判断の癖と盲点が可視化されます。

---

## Decision Metricsとランクの関係

ランクスコアとDecision Metricsは連動しています。
どの指標が低いかを確認して、強化すべき領域を特定してください。

| 指標が低い場合 | 改善アクション |
|---|---|
| Q（意思決定精度）が低い | 構造キーワードを意識して問いを設計し直す |
| R（リスク耐性）が低い | 失敗条件・撤退基準を毎回明示する |
| S（構造理解）が低い | 問題を「原因・構造・影響」の3層で分解する |
| V（判断速度）が低い | 選択肢を2〜3に絞り、判断軸を事前に定義する |
| P（予測精度）が低い | 利用頻度を週3回以上に増やす |
| E（実行一貫性）が低い | 前回の判断を振り返り、実行状況を報告してから次の相談をする |
"""
    return {"content": content}

@router.get("/manual")
def get_manual(payload: dict = Depends(verify_token)):
    content = """# 📖 ASCEND 完全マニュアル

## ASCENDが解決する本質的な問題

多くの経営者・事業者が直面する課題は同じです。
「誰に相談すればいいかわからない」「コンサルタントは高すぎる」「情報はあるが判断できない」。
ASCENDはその空白を埋めるために設計されました。

戦略・数値・構造・リスク——あらゆる経営判断に即応するAIコンサルティングエンジンです。
単なる情報提供ではなく、あなたが自ら意思決定できる状態へ引き上げることが目的です。

---

## 1. 戦略相談（メインチャット）— ASCENDの中核

### なぜ普通のAIと違うのか

一般的なAIは「質問に答える」だけです。
ASCENDは入力の構造を分析し、事業の文脈に沿った戦略的アウトプットを生成します。
目的モード・AIエンジン・スコアリングが連動することで、回答の質が根本的に変わります。

### 入力の黄金フォーマット

【目的】達成したい数値目標・状態（例：3ヶ月以内に月商500万達成）
【現状】観測できる数値・事実（例：現在300万、スタッフ3名、広告費5万/月）
【制約】予算・期限・変えられない条件（例：値下げ不可、追加採用なし）
【出力形式】欲しい形式（例：優先度付き施策リスト＋各KPIと計測方法）

制約を書くことで実行可能な施策に変わります。制約がないと理論上最適な施策を並べるだけになります。

### 目的モード一覧

| モード | 用途 | 代表的な活用シーン |
|---|---|---|
| AUTO | AIが文脈から自動判別 | 迷ったらこれ |
| NUMERIC | 数値・KPI・売上・コスト分析 | 客単価を上げる施策を数値で整理したい |
| GROWTH | スキル・習慣・成長設計 | 営業力を3ヶ月で体系的に鍛えたい |
| CONTROL | 組織・権限・業務フロー構造化 | 属人化を解消してチームを動かしたい |
| STRATEGY | 競合分析・差別化・ポジション設計 | 価格競争から脱却する戦略を作りたい |
| ANALYSIS | データ・事象の多角的解析 | なぜこの指標が下がっているか分析したい |
| PLANNING | ロードマップ・フェーズ設計 | 半年後の状態から逆算して計画を立てたい |
| RISK | リスク特定・評価・対策設計 | この意思決定に潜むリスクを洗い出したい |
| NEGOTIATION | 交渉・説得・合意形成戦略 | 価格交渉・条件変更を通すための準備をしたい |
| MARKETING | 集客・ブランディング・広告施策 | SNSと広告を組み合わせた新規獲得戦略が欲しい |
| DIAGNOSIS | 現状課題の発見・根本原因分析 | 売上が下がり続ける本当の原因を特定したい |
| FORECAST | 将来予測・シナリオ分析 | 3パターンのシナリオで6ヶ月後を予測したい |
| FINANCE | 財務・投資・資金計画分析 | この投資の回収期間とリターンを試算したい |
| HR | 採用・評価・組織設計・人材育成 | 評価制度を設計して離職率を下げたい |
| CREATIVE | アイデア発想・コンセプト設計 | 競合と全く違う切り口でサービスを再設計したい |
| SUMMARY | 要約・情報整理 | 長文や複数情報を簡潔にまとめたい |
| COACHING | コーチング・成長支援 | 思考の癖を発見して行動変容を促したい |
| LEGAL | 法務・規約・契約 | 契約書のリスクや法的注意点を確認したい |
| OPS | 業務改善・効率化 | 現場フローのボトルネックを特定して改善したい |
| TECH | 技術・エンジニアリング | システム設計や技術的意思決定の壁打ちをしたい |

---

## 2. AIエンジン：判断の重さに合わせて使い分ける

| エンジン | 処理特性 | 最適な用途 |
|---|---|---|
| SWIFT（迅速） | 高速レスポンス・日常的な戦略相談 | 毎日の意思決定・アイデア出し・施策整理 |
| ADVANCE（高度） | 深い文脈理解・複雑な因果関係の分析 | 重要な戦略判断・資金調達・組織改革 |
| SUPREME（至高） | 最高精度・多層的な問題構造の解体 | 事業再構築・M&A検討・大型案件の意思決定 |

ADVANCE・SUPREMEは管理者からの権限付与が必要です。重要な判断ほど上位エンジンを使うことで見落としリスクが大幅に低減します。

---

## 3. データ分析（テーブル操作）— 数字を戦略に変換する

CSVやExcelを添付すると自動でテーブルモードに切り替わります。

| コマンド | 機能 | 活用例 |
|---|---|---|
| /rank 列名 | 降順ランキング化 | /rank 売上 → 店舗別売上ランキング |
| /filter 条件 | 条件フィルタ抽出 | /filter 売上 >= 100 → 高売上店舗のみ抽出 |
| /derive 式 | 派生指標の追加 | /derive 客単価=売上/客数 → 新指標を即生成 |
| /top N 列 | 上位N件抽出 | /top 5 売上 → 売上上位5店舗 |
| /diff 列 | 前行との差分 | /diff 売上 → 月次増減を可視化 |
| /growth 列 | 増減率(%) | /growth 売上 → 成長率を自動算出 |
| /consult | AIによる総合戦略分析 | テーブル全体をAIが分析しインサイトを提示 |

/consultが最も強力です。数字の羅列を経営判断に使えるインサイトに変換します。

---

## 4. ランクシステム — 思考の設計力を可視化する

| 評価軸 | 配点 | 代表キーワード |
|---|---|---|
| 構造キーワード | +3pt | 構造・資本・市場・制度・最適化・期待値・アーキテクチャ |
| 戦略キーワード | +2pt | 戦略・差別化・競合・ポジショニング・KPI・ROI |
| 実行キーワード | +1pt | 実行・手順・タスク・改善・運用・効率 |
| 感情的語彙 | -3pt | 不安・ムカつく・どうせ・無理・クソ |

---

## 5. 診断機能 — 自分では気づけない盲点を暴く

### 現状課題診断
- チャット12回ごとに生成可能
- 直近の会話履歴をAIが解析し、思考パターン・判断の癖・未解決の課題構造を診断
- なぜ同じ問題が繰り返されるのかの根本原因を提示

### 固定概念レポート
- 12回利用後に解放
- 思考パターンのバイアス・盲点・成長の阻害要因を可視化

### Decision Metrics（意思決定精度診断）

| 指標 | 意味 | 低い場合の影響 |
|---|---|---|
| Q 意思決定精度 | 判断の質・構造的思考力 | 感情・直感で重要決定をしてしまう |
| R リスク耐性 | リスクを定量化する能力 | 過度な慎重さまたは無謀な判断 |
| S 構造理解 | 問題の構造を把握する力 | 表面的な対処に終始する |
| V 判断速度 | 適切なスピードで判断する力 | 機会損失または拙速な決断 |
| P 予測精度 | 未来を見通す思考の精度 | 短期思考・場当たり的な対応 |
| E 実行一貫性 | 判断と実行の整合性 | 戦略と現場のズレが常態化 |

---

## 6. コンサル解析機能 — プロのフレームワークをAIで即実行

| 機能 | 何ができるか | 活用シーン |
|---|---|---|
| 構造診断 | 事業・組織・戦略の力学構造を解剖しボトルネックを特定 | なぜ施策を打っても改善しないのかの解明 |
| 課題仮説 | 状況から複数の課題仮説を優先度付きで生成 | 問題の全体像を整理して手を打つ順番を決める |
| 比較分析 | A案/B案/C案を複数の評価軸で客観的に比較 | 迷っている選択肢を構造的に整理する |
| 矛盾検知 | 戦略・方針・行動間の矛盾と整合性を検証 | 言っていることとやっていることが違うを発見 |
| 実行計画 | フェーズ別・期限付きのアクションプランを生成 | 戦略を実際に動かせる計画に落とし込む |
| 投資シグナル | 反発底打ち候補・大口売り込み監視・個別銘柄分析 | 資金配分・参入タイミングの判断材料 |

---

## 7. 個人相談（DM） — AIを超えた案件はコンサルタントへ直接

複雑な人間関係・経営危機・事業承継・機密性の高い戦略判断は、コンサルタントに直接相談できます。
マイページの「個人相談」タブから送信してください。

---

## 8. 最大限活用するための3原則

原則1：具体的な数字で話す
「売上が下がった」ではなく「先月比-15%、3ヶ月連続、特に新規顧客の離脱率が高い」。
数字が入ると、AIの分析精度が根本的に変わります。

原則2：制約を必ず書く
「予算なし」「人員増員不可」「値下げ禁止」——制約を書くことで施策が現実的になります。
制約のない戦略提案は実行不可能な理想論になります。

原則3：診断機能を定期的に使う
12回のチャットごとに現状課題診断を実行してください。
自分では気づけない思考パターンの偏りと成長の停滞要因を定期的に洗い出すことが、
長期的な意思決定精度の向上につながります。
"""
    return {"content": content}

@router.get("/chat_examples")
def get_chat_examples(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("tenant_settings").document(tid).get()
            if doc.exists:
                examples = (doc.to_dict() or {}).get("chat_examples") or []
                if examples:
                    return {"examples": examples}
    except Exception:
        pass
    return {"examples": []}

@router.get("/purpose_modes")
def get_purpose_modes(payload: dict = Depends(verify_token)):
    uid = payload.get("uid", "")
    from api.core.features import load_user_plan
    plan = load_user_plan(uid)
    # プラン別モード制御
    STANDARD_MODES = ["auto", "numeric", "growth", "control", "analysis", "planning", "risk"]
    if plan == "starter":
        keys = ["auto"]
    elif plan == "standard":
        keys = STANDARD_MODES
    else:
        # pro/apex/ultra_admin/ultra_member → 全モード
        keys = list(ALL_PURPOSE_MODES.keys())
    return {"modes": [{"id": k, "label": ALL_PURPOSE_MODES[k]} for k in keys if k in ALL_PURPOSE_MODES]}

@router.get("/theme")
def get_theme(payload: dict = Depends(verify_token)):
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    DEFAULT = {
        "logo_url": "",
        "logo_size": 32,
        "favicon_url": "",
        "color_primary": "#6366f1",
        "color_secondary": "#8b5cf6",
        "color_bg": "#070710",
        "color_nav_bg": "rgba(10,10,20,0.95)",
        "color_sidebar_bg": "rgba(8,8,18,0.98)",
        "color_card_bg": "rgba(255,255,255,0.04)",
        "color_text_main": "#e8e8f0",
        "color_text_sub": "#9ca3af",
        "color_border": "rgba(99,102,241,0.15)",
        "color_user_bubble": "linear-gradient(135deg,#6366f1,#7c3aed)",
        "color_ai_bubble": "rgba(255,255,255,0.04)",
    }
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"theme_config_{tid}").get()
            if doc.exists:
                return {**DEFAULT, **(doc.to_dict() or {})}
    except Exception:
        pass
    return DEFAULT

from fastapi import Body

@router.post("/theme/save")
def save_theme(body: dict = Body(...), payload: dict = Depends(verify_token)):
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ設定可能")
    tenant_id = body.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    db.collection("system_settings").document(f"theme_config_{tenant_id}").set(body, merge=True)
    return {"ok": True}


from fastapi import Body
import base64 as _b64
import pickle as _pickle

@router.post("/lgbm/feedback")
def lgbm_feedback(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """LGBMフィードバック記録（チャット評価→教師データ）"""
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        label = 1 if body.get("label") == "good" else 0
        db.collection("tenants").document(tenant_id).collection("lgbm_training_logs").add({
            "uid": uid,
            "prompt": body.get("prompt",""),
            "response": body.get("response",""),
            "label": label,
            "purpose_mode": body.get("purpose_mode",""),
            "recorded_at": __import__("datetime").datetime.now().isoformat(),
            "tenant_id": tenant_id,
        })
    except Exception:
        pass
    return {"ok": True}

@router.get("/lgbm/predict")
def lgbm_predict(prompt: str = "", payload: dict = Depends(verify_token)):
    """LGBMモデルによるpurpose_mode予測"""
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            snap = db.collection("tenants").document(tid).collection("lgbm_models").document("purpose_mode").get()
            if snap.exists:
                d = snap.to_dict() or {}
                model_b64 = d.get("model_b64","")
                if model_b64:
                    model = _pickle.loads(_b64.b64decode(model_b64))
                    # 特徴量：テキスト長・キーワード数
                    t = prompt.lower()
                    feats = [[
                        len(t),
                        sum(1 for w in ["数字","売上","KPI","目標","戦略","施策"] if w in t),
                        sum(1 for w in ["成長","改善","訓練","スキル"] if w in t),
                        sum(1 for w in ["構造","支配","管理","制御"] if w in t),
                        sum(1 for w in ["創造","アイデア","発想","クリエイティブ"] if w in t),
                        sum(1 for w in ["集客","マーケ","SNS","広告"] if w in t),
                    ]]
                    pred = model.predict(feats)
                    modes = ["auto","numeric","growth","control","creative","marketing"]
                    predicted = modes[int(pred[0])] if int(pred[0]) < len(modes) else "auto"
                    return {"mode": predicted, "source": "lgbm"}
    except Exception:
        pass
    return {"mode": "auto", "source": "fallback"}

@router.post("/lgbm/train")
def lgbm_train(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """LGBMモデル学習（管理者のみ）"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    tenant_id = body.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        logs = [d.to_dict() for d in db.collection("tenants").document(tenant_id).collection("lgbm_training_logs").stream()]
        if len(logs) < 20:
            return {"trained": False, "message": f"学習データ不足: {len(logs)}件（最低20件必要）"}
        import pandas as _pd_l
        df = _pd_l.DataFrame(logs)
        modes = ["auto","numeric","growth","control","creative","marketing"]
        df["mode_idx"] = df["purpose_mode"].apply(lambda x: modes.index(x) if x in modes else 0)
        def _feats(row):
            t = str(row.get("prompt","")).lower()
            return [len(t),
                sum(1 for w in ["数字","売上","KPI","目標","戦略","施策"] if w in t),
                sum(1 for w in ["成長","改善","訓練","スキル"] if w in t),
                sum(1 for w in ["構造","支配","管理","制御"] if w in t),
                sum(1 for w in ["創造","アイデア","発想","クリエイティブ"] if w in t),
                sum(1 for w in ["集客","マーケ","SNS","広告"] if w in t),
            ]
        X = [_feats(r) for _, r in df.iterrows()]
        y = df["mode_idx"].tolist()
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(X, y)
        model_b64 = _b64.b64encode(_pickle.dumps(model)).decode()
        db.collection("tenants").document(tenant_id).collection("lgbm_models").document("purpose_mode").set({
            "model_b64": model_b64, "n_samples": len(logs),
            "trained_at": __import__("datetime").datetime.now().isoformat(),
        })
        return {"trained": True, "n_samples": len(logs), "message": "学習完了"}
    except Exception as e:
        return {"trained": False, "message": str(e)}


@router.delete("/account")
def deep_delete_account(payload: dict = Depends(verify_token)):
    """ユーザーの全データを完全削除"""
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    from google.cloud import firestore as _fs6
    deleted = {}
    try: db.collection("users").document(uid).delete(); deleted["users"] = True
    except: deleted["users"] = False
    try:
        docs = list(db.collection("usage_logs").where("user_id","==",uid).limit(500).stream())
        for d in docs: d.reference.delete()
        deleted["usage_logs"] = len(docs)
    except: deleted["usage_logs"] = 0
    try:
        docs = list(db.collection("chat_sessions").where("uid","==",uid).limit(200).stream())
        for d in docs:
            msgs = list(d.reference.collection("messages").limit(500).stream())
            for m in msgs: m.reference.delete()
            d.reference.delete()
        deleted["chat_sessions"] = len(docs)
    except: deleted["chat_sessions"] = 0
    try:
        docs = list(db.collection("user_diagnoses").where("uid","==",uid).limit(100).stream())
        for d in docs: d.reference.delete()
        deleted["user_diagnoses"] = len(docs)
    except: deleted["user_diagnoses"] = 0
    try:
        docs = list(db.collection("lgbm_training_logs").where("uid","==",uid).limit(1000).stream())
        for d in docs: d.reference.delete()
        deleted["lgbm_training_logs"] = len(docs)
    except: deleted["lgbm_training_logs"] = 0
    return {"ok": True, "deleted": deleted}


@router.get("/session_timeout")
def get_session_timeout(payload: dict = Depends(verify_token)):
    """セッションタイムアウト設定を取得"""
    db = get_db()
    DEFAULT_TIMEOUT = 15
    try:
        doc = db.collection("system_settings").document("global_config").get()
        if doc.exists:
            v = (doc.to_dict() or {}).get("session_timeout_minutes")
            if v is not None:
                try:
                    val = int(v)
                    if val > 0:
                        return {"session_timeout_minutes": val}
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return {"session_timeout_minutes": DEFAULT_TIMEOUT}


# ── 管理機能追加エンドポイント ──────────────────────────────

@router.get("/admin/export_users")
def export_users(payload: dict = Depends(verify_token)):
    """全ユーザーのスコア・ランク・利用状況をCSV出力"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    try:
        import io, csv as _csv
        users = list(db.collection("users").limit(500).stream())
        rows = []
        for u in users:
            d = u.to_dict() or {}
            uid = d.get("uid", u.id)
            score = int(d.get("level_score", 0))
            tenant_id = d.get("tenant_id", DEFAULT_TENANT)
            cfg = _load_rank_config(tenant_id)
            rank = _score_to_rank(score, cfg)
            rows.append({
                "uid": uid,
                "tenant_id": tenant_id,
                "level_score": score,
                "rank": rank,
                "is_active": d.get("is_active", True),
                "expires_at": str(d.get("expires_at", "")),
                "created_at": str(d.get("created_at", ""))[:19],
                "last_login": str(d.get("last_login", ""))[:19],
            })
        rows.sort(key=lambda x: x["level_score"], reverse=True)
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=["uid","tenant_id","level_score","rank","is_active","expires_at","created_at","last_login"])
        w.writeheader()
        w.writerows(rows)
        return {"csv": buf.getvalue(), "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/tenant_stats")
def get_tenant_stats(payload: dict = Depends(verify_token)):
    """テナント別利用統計"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    try:
        from collections import defaultdict
        logs = list(db.collection("usage_logs").limit(2000).stream())
        users = list(db.collection("users").limit(500).stream())
        tenant_logs = defaultdict(int)
        tenant_uids = defaultdict(set)
        for l in logs:
            d = l.to_dict() or {}
            if d.get("is_admin_test"): continue
            t = d.get("tenant_id", DEFAULT_TENANT)
            tenant_logs[t] += 1
            uid = d.get("user_id", "")
            if uid: tenant_uids[t].add(uid)
        tenant_active = defaultdict(int)
        for u in users:
            d = u.to_dict() or {}
            if d.get("is_active", True):
                tenant_active[d.get("tenant_id", DEFAULT_TENANT)] += 1
        all_tenants = set(list(tenant_logs.keys()) + list(tenant_active.keys()))
        result = []
        for t in all_tenants:
            result.append({
                "tenant_id": t,
                "total_chats": tenant_logs[t],
                "active_users": tenant_active[t],
                "unique_chatters": len(tenant_uids[t]),
            })
        result.sort(key=lambda x: x["total_chats"], reverse=True)
        return {"tenants": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/reset_score")
def reset_user_score(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """特定ユーザーのlevel_scoreをリセット"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    uid = body.get("uid", "").strip()
    score = int(body.get("score", 0))
    if not uid:
        raise HTTPException(status_code=400, detail="uid必須")
    db = get_db()
    try:
        from google.cloud import firestore as _fs8
        db.collection("users").document(uid).set(
            {"level_score": score, "updated_at": _fs8.SERVER_TIMESTAMP},
            merge=True
        )
        return {"ok": True, "uid": uid, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/set_expires")
def set_user_expires(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """ユーザー期限設定"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    uid = body.get("uid", "").strip()
    expires_at = body.get("expires_at", "")
    if not uid:
        raise HTTPException(status_code=400, detail="uid必須")
    db = get_db()
    try:
        from google.cloud import firestore as _fs9
        db.collection("users").document(uid).set(
            {"expires_at": expires_at, "updated_at": _fs9.SERVER_TIMESTAMP},
            merge=True
        )
        return {"ok": True, "uid": uid, "expires_at": expires_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/bulk_feature_flags")
def bulk_set_feature_flags(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """テナント単位で全ユーザーの機能フラグを一括変更"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    tenant_id = body.get("tenant_id", DEFAULT_TENANT)
    flags = body.get("flags", {})
    if not flags:
        raise HTTPException(status_code=400, detail="flags必須")
    db = get_db()
    try:
        from google.cloud import firestore as _fs10
        users = list(db.collection("users").where("tenant_id", "==", tenant_id).limit(500).stream())
        updated = 0
        for u in users:
            existing = (u.to_dict() or {}).get("feature_overrides", {})
            existing.update(flags)
            db.collection("users").document(u.id).set(
                {"feature_overrides": existing, "updated_at": _fs10.SERVER_TIMESTAMP},
                merge=True
            )
            updated += 1
        return {"ok": True, "tenant_id": tenant_id, "updated": updated, "flags": flags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/health")
def system_health(payload: dict = Depends(verify_token)):
    """システム稼働状況ダッシュボード"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    result = {}
    # Firestore確認
    try:
        db = get_db()
        db.collection("system_settings").document("global_config").get()
        result["firestore"] = "OK"
    except Exception as e:
        result["firestore"] = f"NG: {e}"
    # GCS確認
    try:
        import os as _os2
        from google.cloud import storage as _gcs2
        bucket = _os2.environ.get("CENTRAL_BLOB_BUCKET", "")
        if bucket:
            _gc2 = _gcs2.Client()
            list(_gc2.bucket(bucket).list_blobs(max_results=1))
            result["gcs"] = "OK"
        else:
            result["gcs"] = "BUCKET未設定"
    except Exception as e:
        result["gcs"] = f"NG: {e}"
    # Gemini API確認
    try:
        import os as _os3
        api_key = _os3.environ.get("GEMINI_API_KEY", "")
        result["gemini_api_key"] = "あり" if api_key else "なし"
        from google import genai as _genai2
        _client2 = _genai2.Client(api_key=api_key)
        models = list(_client2.models.list())
        result["gemini_models"] = len(models)
        result["gemini"] = "OK"
    except Exception as e:
        result["gemini"] = f"NG: {e}"
    # セッションタイムアウト設定
    try:
        db = get_db()
        doc = db.collection("system_settings").document("global_config").get()
        timeout = (doc.to_dict() or {}).get("session_timeout_minutes", 15)
        result["session_timeout_minutes"] = timeout
    except Exception:
        result["session_timeout_minutes"] = 15
    # ユーザー数
    try:
        db = get_db()
        users = list(db.collection("users").limit(500).stream())
        active = sum(1 for u in users if (u.to_dict() or {}).get("is_active", True))
        result["total_users"] = len(users)
        result["active_users"] = active
    except Exception as e:
        result["users"] = f"NG: {e}"
    return result

@router.get("/custom_prompt")
def get_custom_prompt(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        doc = db.collection("users").document(uid).get()
        d = doc.to_dict() or {} if doc.exists else {}
        return {
            "custom_sys_prompt": d.get("custom_sys_prompt", ""),
            "custom_prompt_mode": d.get("custom_prompt_mode", "append"),
            "has_custom": bool(d.get("custom_sys_prompt", "")),
        }
    except Exception as e:
        return {"custom_sys_prompt": "", "custom_prompt_mode": "append", "has_custom": False}

@router.post("/custom_prompt")
def save_custom_prompt(body: dict = Body(...), payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        db.collection("users").document(uid).set({
            "custom_sys_prompt": body.get("custom_sys_prompt", ""),
            "custom_prompt_mode": body.get("custom_prompt_mode", "append"),
        }, merge=True)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plan")
def get_user_plan(payload: dict = Depends(verify_token)):
    """ログイン中ユーザーのプラン情報を返す"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = (snap.to_dict() or {}) if snap.exists else {}
        plan = d.get("plan") or ""
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user_ai_settings")
def get_user_ai_settings(payload: dict = Depends(verify_token)):
    """ユーザーのAI設定（説明・会話のきっかけ）を返す"""
    from fastapi import Body
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    snap = db.collection("users").document(uid).get()
    d = (snap.to_dict() or {}) if snap.exists else {}
    return {
        "ai_description":        d.get("ai_description", ""),
        "conversation_starters": d.get("conversation_starters", []),
        "use_admin_settings":    d.get("use_admin_settings", False),
        "member_extra_prompt":   d.get("member_extra_prompt", ""),
    }

@router.post("/user_ai_settings")
def save_user_ai_settings(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """ユーザーのAI設定（説明・会話のきっかけ）を保存する"""
    from fastapi import Body
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    db.collection("users").document(uid).set({
        "ai_description":        (body.get("ai_description") or "").strip(),
        "conversation_starters": [s.strip() for s in (body.get("conversation_starters") or []) if s.strip()][:4],
        "use_admin_settings":    bool(body.get("use_admin_settings", False)),
        "member_extra_prompt":   (body.get("member_extra_prompt") or "").strip(),
        "updated_at":            fs.SERVER_TIMESTAMP,
    }, merge=True)
    return {"ok": True}

@router.get("/admin_ai_settings")
def get_admin_ai_settings(payload: dict = Depends(verify_token)):
    """同テナントのultra_adminのAI設定を返す（ultra_member専用）"""
    uid = payload.get("uid", "")
    tenant_id = payload.get("tenant_id", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    u_snap = db.collection("users").document(uid).get()
    u_data = (u_snap.to_dict() or {}) if u_snap.exists else {}
    plan = u_data.get("plan", "")
    if plan != "ultra_member":
        raise HTTPException(status_code=403, detail="ultra_memberのみ利用可能")
    tenant_id = u_data.get("tenant_id", tenant_id)
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id必須")
    try:
        admin_docs = list(
            db.collection("users")
            .where("tenant_id", "==", tenant_id)
            .limit(20)
            .stream()
        )
        for ad in admin_docs:
            ad_data = ad.to_dict() or {}
            if ad_data.get("plan") == "ultra_admin":
                return {
                    "ai_description":        ad_data.get("ai_description", ""),
                    "conversation_starters": ad_data.get("conversation_starters", []),
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ai_description": "", "conversation_starters": []}

@router.get("/user_knowledge_list")
def get_user_knowledge_list(payload: dict = Depends(verify_token)):
    """ユーザー専用知識ファイル一覧を返す"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    user_tenant = f"user__{uid}"
    try:
        links = list(db.collection("tenant_source_links").where("tenant_id", "==", user_tenant).limit(60).stream())
        result = []
        for lnk in links:
            ld = lnk.to_dict() or {}
            result.append({
                "source_id": ld.get("source_id", ""),
                "title":     ld.get("title", ""),
                "link_id":   lnk.id,
                "chunks":    ld.get("chunks", 0),
                "summaries": ld.get("summaries", 0),
            })
        return {"files": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/user_knowledge_upload")
async def upload_user_knowledge(
    file: UploadFile = File(...),
    payload: dict = Depends(verify_token),
):
    """ユーザー専用知識ファイルをアップロード・チャンク・ベクトル登録する"""
    from api.core.rag import embed_text as _embed_text
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    user_tenant = f"user__{uid}"
    _bytes = await file.read()
    fname = file.filename or "file"
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "txt"
    text = ""
    if ext in ("txt", "md", "csv"):
        text = _bytes.decode("utf-8", errors="replace")
    elif ext == "odt":
        try:
            import zipfile, re as _re2, io as _io2
            with zipfile.ZipFile(_io2.BytesIO(_bytes)) as _z:
                with _z.open("content.xml") as _cx:
                    _xml = _cx.read().decode("utf-8", errors="replace")
            text = _re2.sub(r"<[^>]+>", " ", _xml)
        except Exception:
            text = ""
    elif ext in ("xlsx", "xls"):
        try:
            import pandas as _pd2, io as _io3
            _dfs = _pd2.read_excel(_io3.BytesIO(_bytes), sheet_name=None)
            text = "\n".join(f"[{sn}]\n{df.to_string(index=False)}" for sn, df in _dfs.items())
        except Exception:
            text = ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="テキスト抽出失敗または空ファイル")

    # URL検出→fetch→本文追記
    import re as _re_url
    import urllib.request as _urllib_req
    _url_pattern = _re_url.compile(r'https?://[^\s　》「」】〕｠〉』、。，．《〈〖『【〔］［）（＞＜]+')
    _found_urls = list(dict.fromkeys(_url_pattern.findall(text)))[:10]
    for _url in _found_urls:
        try:
            _req = _urllib_req.Request(_url, headers={"User-Agent": "Mozilla/5.0"})
            with _urllib_req.urlopen(_req, timeout=8) as _resp:
                _raw = _resp.read()
                _charset = "utf-8"
                _ct = _resp.headers.get_content_charset()
                if _ct:
                    _charset = _ct
                _html = _raw.decode(_charset, errors="replace")
            # タグ除去・本文抽出
            _html = _re_url.sub(r'<script[^>]*>.*?</script>', '', _html, flags=_re_url.DOTALL|_re_url.IGNORECASE)
            _html = _re_url.sub(r'<style[^>]*>.*?</style>', '', _html, flags=_re_url.DOTALL|_re_url.IGNORECASE)
            _plain = _re_url.sub(r'<[^>]+>', ' ', _html)
            _plain = _re_url.sub('[ \t]+', ' ', _plain).strip()
            _plain = _re_url.sub('\n{3,}', '\n\n', _plain)
            if _plain.strip():
                text += "\n\n[URL: " + _url + "]\n" + _plain[:3000]
        except Exception:
            pass

    source_id = f"user__{uid}__{fname}"
    # 同一ファイル名の重複チェック
    existing_link = db.collection("tenant_source_links").document(f"{user_tenant}__{fname}").get()
    if existing_link.exists:
        raise HTTPException(status_code=409, detail=f"同じファイル名（{fname}）は既にアップロード済みです。削除してから再度アップロードしてください。")
    # 全角スペース・連続スペース・連続改行を正規化
    import re as _re_clean
    text = text.replace("　", " ")
    text = _re_clean.sub(r"\n{3,}", "\n\n", text).strip()
    text = _re_clean.sub(r" {2,}", " ", text)

    chunk_size = 800
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    db.collection("sources").document(source_id).set({
        "source_id":   source_id,
        "title":       fname,
        "content":     text[:12000],
        "category":    "ユーザー知識",
        "source_type": "file",
        "created_at":  fs.SERVER_TIMESTAMP,
        "updated_at":  fs.SERVER_TIMESTAMP,
    }, merge=True)
    link_id = f"{user_tenant}__{fname}"
    db.collection("tenant_source_links").document(link_id).set({
        "tenant_id":   user_tenant,
        "source_id":   source_id,
        "title":       fname,
        "category":    "ユーザー知識",
        "source_type": "file",
        "enabled":     True,
        "priority":    1,
        "updated_at":  fs.SERVER_TIMESTAMP,
    }, merge=True)
    wrote = 0
    for ci, chunk in enumerate(chunks):  # 全チャンク処理（上限撤廃）
        if not chunk.strip():
            continue
        try:
            emb = _embed_text(chunk)
            cid = f"{source_id}_c{ci}"
            db.collection("source_chunks").document(cid).set({
                "chunk_id":    cid,
                "doc_id":      source_id,
                "source_id":   source_id,
                "title":       fname,
                "text":        chunk,
                "embedding":   emb,
                "category":    "ユーザー知識",
                "source_type": "file",
                "chunk_index": ci,
            }, merge=True)
            wrote += 1
        except Exception as _chunk_e:
            print(f"[CHUNK ERROR] {_chunk_e}", flush=True)
    # サマリー生成・保存
    summaries_wrote = 0
    try:
        from api.core.llm_client import call_llm as _call_llm
        _section_size = 10000
        _sections = [text[i:i+_section_size] for i in range(0, len(text), _section_size)]
        for _si, _section in enumerate(_sections):
            if not _section.strip():
                continue
            try:
                _summary_prompt = (
                    f"以下のファイルのセクション{_si+1}/{len(_sections)}を300字以内で要約せよ。"
                    f"主要論点・固有名詞・数値・結論を漏らさず含めよ。抽象的表現禁止。\n\n"
                    f"ファイル名: {fname}\n\n{_section}"
                )
                _summary_text = _call_llm(
                    system_prompt="要約のみ出力。前置き・後置き禁止。",
                    messages=[{"role": "user", "content": _summary_prompt}],
                    ai_tier="core",
                    max_tokens=500,
                )
                if _summary_text:
                    _summary_id = f"{source_id}__summary__{_si}"
                    _semb = _embed_text(_summary_text)
                    db.collection("source_chunks").document(_summary_id).set({
                        "chunk_id":    _summary_id,
                        "doc_id":      source_id,
                        "source_id":   source_id,
                        "title":       fname,
                        "text":        _summary_text,
                        "embedding":   _semb,
                        "category":    "ユーザー知識",
                        "source_type": "summary",
                        "chunk_index": -1 - _si,
                    }, merge=True)
                    summaries_wrote += 1
            except Exception as _sum_e:
                print(f"[SUMMARY ERROR] section={_si} {_sum_e}", flush=True)
    except Exception as _sum_outer_e:
        print(f"[SUMMARY OUTER ERROR] {_sum_outer_e}", flush=True)
    db.collection("tenant_source_links").document(link_id).set({
        "chunks": wrote,
        "summaries": summaries_wrote,
    }, merge=True)
    return {"ok": True, "source_id": source_id, "chunks": wrote, "summaries": summaries_wrote}

@router.delete("/user_knowledge/{source_id:path}")
def delete_user_knowledge(source_id: str, payload: dict = Depends(verify_token)):
    """ユーザー専用知識ファイルを削除する"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    db = get_db()
    user_tenant = f"user__{uid}"
    link_id = source_id
    try:
        db.collection("tenant_source_links").document(link_id).delete()
        db.collection("sources").document(source_id).delete()
        # チャンクを全件削除（limit(20)ループ）
        while True:
            batch = list(db.collection("source_chunks").where("source_id", "==", source_id).limit(50).stream())
            if not batch:
                break
            for c in batch:
                c.reference.delete()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag_settings")
def get_rag_settings(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        snap = db.collection("users").document(uid).get()
        d = (snap.to_dict() or {}) if snap.exists else {}
        rag = d.get("rag_settings") or {}
        return {
            "threshold": float(rag.get("threshold", 0.42)),
            "top_k": int(rag.get("top_k", 5)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag_settings")
def save_rag_settings(body: dict = Body(...), payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        threshold = float(body.get("threshold", 0.42))
        top_k = int(body.get("top_k", 5))
        threshold = max(0.10, min(0.90, threshold))
        top_k = max(1, min(20, top_k))
        db.collection("users").document(uid).set({
            "rag_settings": {"threshold": threshold, "top_k": top_k}
        }, merge=True)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_slides")
def generate_slides(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """コンサルレベルのプレゼン資料を2段階パイプラインで生成する"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")

    import json as _json
    from api.core.llm_client import call_llm

    # 14項目構造入力
    target_role          = str(body.get("target_role", "")).strip()
    decision_goal        = str(body.get("decision_goal", "")).strip()
    decision_criteria    = str(body.get("decision_criteria", "")).strip()
    current_state        = str(body.get("current_state", "")).strip()
    problem              = str(body.get("problem", "")).strip()
    root_cause           = str(body.get("root_cause", "")).strip()
    options_comparison   = str(body.get("options_comparison", "")).strip()
    proposal             = str(body.get("proposal", "")).strip()
    evidence             = str(body.get("evidence", "")).strip()
    risk                 = str(body.get("risk", "")).strip()
    rejection_risk       = str(body.get("rejection_risk", "")).strip()
    execution            = str(body.get("execution", "")).strip()
    priority             = str(body.get("priority", "")).strip()
    success_kpi          = str(body.get("success_kpi", "")).strip()
    slide_count          = max(3, min(12, int(body.get("slide_count", 6))))

    if not decision_goal and not current_state and not proposal:
        raise HTTPException(status_code=400, detail="意思決定ゴール・現状・提案のいずれかは必須です")

    # ── Stage 1: 論理構造生成 ──
    stage1_prompt = f"""あなたはMcKinsey水準のストラテジストです。
以下の入力から、プレゼンテーションの論理骨格をPyramid Principleで設計してください。

【対象】{target_role or "未記入"}
【意思決定ゴール】{decision_goal or "未記入"}
【評価軸】{decision_criteria or "未記入"}
【現状（数値・事実）】{current_state or "未記入"}
【問題（ズレ）】{problem or "未記入"}
【原因（構造）】{root_cause or "未記入"}
【選択肢比較】{options_comparison or "未記入"}
【提案（施策）】{proposal or "未記入"}
【根拠（データ・ロジック）】{evidence or "未記入"}
【リスク・障壁】{risk or "未記入"}
【非採用リスク】{rejection_risk or "未記入"}
【実行条件】{execution or "未記入"}
【優先順位】{priority or "未記入"}
【成功定義（KPI）】{success_kpi or "未記入"}

以下のJSON形式のみで返答してください（コードブロック不要）:
{{
  "governing_thought": "この資料の1行結論（So What）",
  "situation": "状況（Situation）の整理",
  "complication": "複雑化（Complication）の整理",
  "resolution": "解決策（Resolution）の核心",
  "key_messages": ["スライドごとのキーメッセージ候補（{slide_count}個）"],
  "logic_flow": "全体の論理の流れ（2〜3文）",
  "objections": ["想定される反論1", "想定される反論2"],
  "success_metrics": ["成功指標1", "成功指標2"]
}}"""

    try:
        raw1 = call_llm(system_prompt=stage1_prompt, messages=[{"role":"user","content":"上記の指示に従いJSONを生成してください"}], ai_tier="core", max_tokens=2048)
        c1 = raw1.strip()
        if c1.startswith("```"): c1 = "\n".join(c1.split("\n")[1:])
        if c1.endswith("```"): c1 = "\n".join(c1.split("\n")[:-1])
        logic = _json.loads(c1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage1エラー: {str(e)}")

    # ── Stage 2: スライド生成 ──
    stage2_prompt = f"""あなたはMcKinsey/BCG水準のシニアコンサルタントです。
以下の論理骨格をもとに、プレゼンテーション資料を生成してください。

【論理骨格】
- 1行結論: {logic.get('governing_thought','')}
- Situation: {logic.get('situation','')}
- Complication: {logic.get('complication','')}
- Resolution: {logic.get('resolution','')}
- 論理の流れ: {logic.get('logic_flow','')}
- キーメッセージ候補: {logic.get('key_messages',[])}
- 想定反論: {logic.get('objections',[])}
- 成功指標: {logic.get('success_metrics',[])}

【元データ】
対象: {target_role or "経営陣"}
意思決定ゴール: {decision_goal or "戦略承認"}
評価軸: {decision_criteria or ""}
現状: {current_state or ""}
問題: {problem or ""}
原因: {root_cause or ""}
選択肢比較: {options_comparison or ""}
提案: {proposal or ""}
根拠: {evidence or ""}
リスク: {risk or ""}
非採用リスク: {rejection_risk or ""}
実行条件: {execution or ""}
優先順位: {priority or ""}
成功定義（KPI）: {success_kpi or ""}
スライド枚数: {slide_count}枚

以下のJSON形式のみで返答してください（コードブロック不要）:
{{
  "title": "資料タイトル",
  "subtitle": "サブタイトル（役職・日付など）",
  "executive_summary": "エグゼクティブサマリー（2〜3文・結論先行）",
  "slides": [
    {{
      "slide_number": 1,
      "type": "cover|agenda|situation|complication|resolution|data|recommendation|risk|execution|conclusion のいずれか",
      "title": "スライドタイトル",
      "headline": "このスライドのSo What（1文・動詞で終わる）",
      "bullets": ["根拠・事実・施策（3〜5点、各40文字以内）"],
      "data_label": "データソース・数値根拠（任意）",
      "note": "補足・反論対策（任意）",
      "chart": {{
        "type": "bar|line|compare|beforeafter|roadmap|none のいずれか",
        "title": "グラフタイトル（なければ空文字）",
        "labels": ["ラベル1", "ラベル2", "ラベル3"],
        "values": [100, 80, 60],
        "unit": "単位（億円・%・件など）",
        "before_label": "Before（beforeafterのみ）",
        "after_label": "After（beforeafterのみ）",
        "before_value": "改善前の状態説明",
        "after_value": "改善後の状態説明",
        "phases": ["Phase1：名称（期間）", "Phase2：名称（期間）"]
      }}
    }}
  ],
  "appendix_notes": "補足資料・データ出典・前提条件"
}}

【最重要ルール】入力に記載のない数値・固有名詞・事例・データは絶対に作らないこと。数値が不明な場合はchart type:"none"にすること。

視覚化判定ルール（必ず適用すること）:
- situation/complicationスライド → 数値データがあればbar or line、なければnone
- data/evidenceスライド → 数値があればbar必須、ROI系ならcompare
- resolution/recommendationスライド → beforeafter推奨
- executionスライド → roadmap必須
- riskスライド → compare（採用リスク vs 非採用リスク）
- cover/agenda/conclusionスライド → none
- labelsとvaluesは必ず対応させ、valuesは数値のみ（文字列不可）
- データが不明・推定不能な場合はtype:"none"にすること（数値のないグラフは生成禁止）

品質基準:
- 全スライドがSCRの論理連鎖で繋がること
- 各headlineは「〜である」「〜すべき」「〜により〜が実現する」形式
- データ・数値・比較を積極的に含める
- 想定反論に対する根拠をどこかのスライドに組み込む
- 実行可能性・リソース・優先順位を明示する"""

    try:
        raw2 = call_llm(system_prompt=stage2_prompt, messages=[{"role":"user","content":"上記の指示に従いJSONを生成してください"}], ai_tier="core", max_tokens=4096)
        c2 = raw2.strip()
        if c2.startswith("```"): c2 = "\n".join(c2.split("\n")[1:])
        if c2.endswith("```"): c2 = "\n".join(c2.split("\n")[:-1])
        data = _json.loads(c2)
        data["logic_skeleton"] = logic
        return {"ok": True, "data": data}
    except _json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Stage2 JSON解析エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_event_plan")
def generate_event_plan(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """イベント企画書をLLMで生成する"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")

    import json as _json
    from api.core.llm_client import call_llm

    event_name      = str(body.get("event_name", "")).strip()
    event_purpose   = str(body.get("event_purpose", "")).strip()
    concept         = str(body.get("concept", "")).strip()
    target          = str(body.get("target", "")).strip()
    current_state   = str(body.get("current_state", "")).strip()
    overview        = str(body.get("overview", "")).strip()
    experience      = str(body.get("experience", "")).strip()
    program         = str(body.get("program", "")).strip()
    promotion       = str(body.get("promotion", "")).strip()
    monetize        = str(body.get("monetize", "")).strip()
    budget          = str(body.get("budget", "")).strip()
    competitor      = str(body.get("competitor", "")).strip()
    risk            = str(body.get("risk", "")).strip()
    rejection_risk  = str(body.get("rejection_risk", "")).strip()
    team            = str(body.get("team", "")).strip()
    kpi             = str(body.get("kpi", "")).strip()

    if not event_name and not event_purpose:
        raise HTTPException(status_code=400, detail="イベント名または開催目的は必須です")

    prompt = f"""あなたはMcKinsey水準のイベントプロデューサー兼コンサルタントです。
以下の情報をもとに、売上と意思決定を動かすイベント企画書を生成してください。

【イベント名】{event_name or "未記入"}
【開催目的】{event_purpose or "未記入"}
【コンセプト（核）】{concept or "未記入"}
【ターゲット】{target or "未記入"}
【現状・課題・原因】{current_state or "未記入"}
【開催概要】{overview or "未記入"}
【体験設計（来場前→当日→来場後）】{experience or "未記入"}
【プログラム構成】{program or "未記入"}
【集客戦略（チャネル別）】{promotion or "未記入"}
【マネタイズ構造】{monetize or "未記入"}
【予算・収支計画】{budget or "未記入"}
【競合・代替比較】{competitor or "未記入"}
【リスク・対策】{risk or "未記入"}
【非実施リスク】{rejection_risk or "未記入"}
【実行体制】{team or "未記入"}
【成功指標（KPI）】{kpi or "未記入"}

以下のJSON形式のみで返答してください（コードブロック不要）:
{{
  "title": "企画書タイトル",
  "subtitle": "サブタイトル（主催・日付など）",
  "executive_summary": "企画概要（3文・目的・規模・期待効果）",
  "sections": [
    {{
      "section_number": 1,
      "type": "purpose|concept|target|situation|overview|experience|program|promotion|monetize|budget|competitor|risk|rejection_risk|team|kpi|appendix のいずれか",
      "title": "セクションタイトル",
      "headline": "このセクションのキーメッセージ（1文）",
      "content": "本文（詳細説明・200文字以内）",
      "bullets": ["箇条書き1", "箇条書き2", "箇条書き3"],
      "chart": {{
        "type": "roadmap|none のいずれか（roadmapはprogramセクションのみ・phases必須）",
        "title": "図表タイトル",
        "labels": [],
        "values": [],
        "unit": "",
        "phases": ["Phase1：内容（期間）"],
        "before_label": "",
        "after_label": "",
        "before_value": "",
        "after_value": ""
      }},

    }}
  ],
  "appendix_notes": "補足・参考資料・前提条件"
}}

品質基準:
- 目的・ターゲット・効果の論理連鎖を明確にすること
- 予算セクションは収支内訳をtableで表現すること
- プログラムセクションはroadmapまたはtimeline chartを使うこと
- KPIは数値目標を明示すること
- リスクは発生確率・影響度・対策をセットで記載すること
- 実現可能性・費用対効果・独自性を随所に示すこと"""

    try:
        raw = call_llm(system_prompt=prompt, messages=[{"role":"user","content":"上記の指示に従いJSONを生成してください"}], ai_tier="core", max_tokens=4096)
        cleaned = raw.strip()
        if cleaned.startswith("```"): cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"): cleaned = "\n".join(cleaned.split("\n")[:-1])
        data = _json.loads(cleaned)
        return {"ok": True, "data": data}
    except _json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON解析エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_slides_stage1")
def generate_slides_stage1(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """Stage1: 論理骨格生成のみ"""
    uid = payload.get("uid", "")
    if not uid:
        raise HTTPException(status_code=401, detail="uid必須")
    import json as _json
    from api.core.llm_client import call_llm
    decision_goal = str(body.get("decision_goal","")).strip()
    if not decision_goal and not str(body.get("current_state","")).strip() and not str(body.get("proposal","")).strip():
        raise HTTPException(status_code=400, detail="意思決定ゴール・現状・提案のいずれかは必須です")
    slide_count = max(3, min(12, int(body.get("slide_count", 6))))
    target_role=str(body.get("target_role","")).strip()
    decision_criteria=str(body.get("decision_criteria","")).strip()
    current_state=str(body.get("current_state","")).strip()
    problem=str(body.get("problem","")).strip()
    root_cause=str(body.get("root_cause","")).strip()
    options_comparison=str(body.get("options_comparison","")).strip()
    proposal=str(body.get("proposal","")).strip()
    evidence=str(body.get("evidence","")).strip()
    risk=str(body.get("risk","")).strip()
    rejection_risk=str(body.get("rejection_risk","")).strip()
    execution=str(body.get("execution","")).strip()
    priority=str(body.get("priority","")).strip()
    success_kpi=str(body.get("success_kpi","")).strip()
    stage1_prompt = f"""あなたはMcKinsey水準のストラテジストです。
以下の入力から、プレゼンテーションの論理骨格をPyramid Principleで設計してください。
【最重要ルール】入力に記載のない数値・固有名詞・事例は絶対に作らないこと。不明な情報は「要確認」と記載すること。

【対象】{target_role or "未記入"}
【意思決定ゴール】{decision_goal or "未記入"}
【評価軸】{decision_criteria or "未記入"}
【現状（数値・事実）】{current_state or "未記入"}
【問題（ズレ）】{problem or "未記入"}
【原因（構造）】{root_cause or "未記入"}
【選択肢比較】{options_comparison or "未記入"}
【提案（施策）】{proposal or "未記入"}
【根拠（データ・ロジック）】{evidence or "未記入"}
【リスク・障壁】{risk or "未記入"}
【非採用リスク】{rejection_risk or "未記入"}
【実行条件】{execution or "未記入"}
【優先順位】{priority or "未記入"}
【成功定義（KPI）】{success_kpi or "未記入"}

以下のJSON形式のみで返答してください（コードブロック不要）:
{{"governing_thought":"1行結論","situation":"状況の整理","complication":"複雑化の整理","resolution":"解決策の核心","key_messages":["メッセージ1"],"logic_flow":"論理の流れ","objections":["反論1"],"success_metrics":["指標1"]}}"""
    try:
        raw = call_llm(system_prompt=stage1_prompt, messages=[{"role":"user","content":"JSONを生成してください"}], ai_tier="core", max_tokens=2048)
        c = raw.strip()
        if c.startswith("```"): c = "\n".join(c.split("\n")[1:])
        if c.endswith("```"): c = "\n".join(c.split("\n")[:-1])
        return {"ok": True, "logic": _json.loads(c)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
