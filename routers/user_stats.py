# api/routers/user_stats.py
from fastapi import APIRouter, Depends, HTTPException
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
    dm = None
    try:
        docs = list(
            db.collection("decision_metrics").document(uid).collection("records")
            .limit(20).stream()
        )
        if docs:
            docs_list = [d.to_dict() or {} for d in docs]
            docs_list.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            dm = docs_list[0]
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
    }

@router.get("/usage_logs")
def get_usage_logs(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    db = get_db()
    try:
        docs = list(db.collection("usage_logs").where("user_id", "==", uid).limit(50).stream())
        logs = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_admin_test"):
                continue
            logs.append({
                "prompt": str(data.get("prompt", ""))[:100],
                "timestamp": str(data.get("timestamp", "")),
            })
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"logs": logs[:20]}
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

---

## 2. AIエンジン：判断の重さに合わせて使い分ける

| エンジン | 処理特性 | 最適な用途 |
|---|---|---|
| Core（標準） | 高速レスポンス・日常的な戦略相談 | 毎日の意思決定・アイデア出し・施策整理 |
| Ultra（高精度） | 深い文脈理解・複雑な因果関係の分析 | 重要な戦略判断・資金調達・組織改革 |
| Apex（最上位） | 最高精度・多層的な問題構造の解体 | 事業再構築・M&A検討・大型案件の意思決定 |

Ultra・Apexは管理者からの権限付与が必要です。重要な判断ほど上位エンジンを使うことで見落としリスクが大幅に低減します。

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
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"purpose_mode_config_{tid}").get()
            if doc.exists:
                keys = [k for k in (doc.to_dict() or {}).get("enabled_modes", []) if k in ALL_PURPOSE_MODES]
                if keys:
                    return {"modes": [{"id": k, "label": ALL_PURPOSE_MODES[k]} for k in keys]}
    except Exception:
        pass
    return {"modes": [{"id": k, "label": v} for k, v in ALL_PURPOSE_MODES.items()]}

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
