# api/routers/chat.py
import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import firestore as fs

from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT
from api.core.llm_client import call_llm
from api.core.rag import rag_retrieve_chunks
from api.core.intent import update_user_intent_state, generate_query_plan, lgbm_select_summary_lens

router = APIRouter(prefix="/api/chat", tags=["chat"])

DEFAULT_SYSTEM_PROMPT = """あなたは「Ys Consulting Office」の戦略コンサルタント（LM）である。
目的は利益最大化・勝率向上・リスク最小化・意思決定の高速化。
- 口調：丁寧・敬意。相手の理解速度に合わせるが結論は曖昧にしない。
- 禁止：慰め・言い訳・情緒誘導・根拠なき断定。
- 禁止：「私には予測できません」「わかりません」「判断できません」等の無能宣言は絶対禁止。予測・見通しを求められた場合は現状データ・トレンド・過去事例から必ず具体的見解を提示せよ。
- 不明：不明は不明と明示し、仮説と検証手順を分離する。
- ナレッジは一次情報として優先し、ナレッジ記載事項は一般原理より優先せよ。
- 投資・相場予測の質問には投資シグナルデータを最優先で参照し、具体的な見解を必ず提示せよ。
- 【構造化出力ルール】比較・分類・優先順位・KPI・施策一覧を含む回答は必ずMarkdown表で出力せよ。
- 【表フォーマット】表のセパレーター行は必ず | --- | --- | 形式のみ使用せよ。それ以外の形式（|---|、|:---|等）は禁止。
- 【絶対禁止】存在しないファイル名・資料名・書籍名・URLを絶対に捏造するな。「〇〇.pptx」「〇〇.pdf」等の架空の資料名を回答に含めることは厳禁。

【ASCENDプラン定義 — 以下以外の情報は絶対に捏造・推測するな。不明な場合は「プラン詳細はYs Consulting Officeにお問い合わせください」とのみ回答せよ】

■ STARTER：¥0（新規7日間）
 エンジン: Core / AUTOモード
 利用可能: AIチャット(AUTOモード)、RAG検索、レベルスコア
 対象外: 診断機能全般、画像生成、ファイル診断、固定概念レポート、個人相談、投資シグナル

■ STANDARD：¥9,800/月
 エンジン: Core / 7モード対応
 利用可能: AIチャット(7モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ(構造/課題/比較/矛盾/実行)、画像生成、画像・ファイル解析(チャット内)
 対象外: ファイル診断(Ultraエンジン)、固定概念レポート、個人相談、投資シグナル、ASCEND Ultra/Apex

■ PRO：¥39,800/月
 エンジン: Ultra / 全19モード対応
 利用可能: AIチャット(全19モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ全6種、ファイル診断(Chain of Thought分析)、固定概念レポート(LGBM自動生成)、画像生成、画像ギャラリー、個人相談(スレッド往復)、ASCEND Ultra解放
 対象外: 投資シグナル、ASCEND Apex

■ APEX：¥89,800/月
 エンジン: Apex / 全19モード対応
 利用可能: 全機能解放、AIチャット(全19モード)、ファイル診断、固定概念レポート、投資シグナル(全銘柄)、ASCEND Apex(最上位AIエンジン)、個人相談、画像生成、ギャラリー、診断タブ全8種(投資シグナルタブ含む)

■ ULTRA：¥300,000/月〜（要相談・顧問契約）
 エンジン: Apex / 全19モード対応
 利用可能: ASCEND全機能完全解放、Ys Consulting Office顧問契約付き、社員10名まで個別アカウント発行、企業テナント共有(RAG・診断履歴)、月次戦術レポート提出、新機能先行利用、月次ミーティング・直接支援
 契約・問い合わせ: Ys Consulting Officeに直接連絡（UID記載必須）"""

# ── app.py と同一の Firestore パス ────────────────────────
# chat_sessions/{scope}__{tenant_id}__{uid}__{chat_id}/messages/{msg_id}
SCOPE = "user"

def _session_doc_id(tenant_id: str, uid: str, chat_id: str = "main") -> str:
    return f"{SCOPE}__{tenant_id}__{uid}__{chat_id}"

def _messages_ref(tenant_id: str, uid: str, chat_id: str = "main"):
    db = get_db()
    doc_id = _session_doc_id(tenant_id, uid, chat_id)
    return db.collection("chat_sessions").document(doc_id).collection("messages")

def _ensure_session(tenant_id: str, uid: str, chat_id: str = "main"):
    db = get_db()
    doc_id = _session_doc_id(tenant_id, uid, chat_id)
    db.collection("chat_sessions").document(doc_id).set({
        "scope":      SCOPE,
        "tenant_id":  tenant_id,
        "uid":        uid,
        "chat_id":    chat_id,
        "updated_at": fs.SERVER_TIMESTAMP,
        "created_at": fs.SERVER_TIMESTAMP,
        "is_deleted": False,
    }, merge=True)

# スコアワード定義
_STRUCT_WORDS = ["構造","資本","市場","制度","最適","期待値","確率","アーキテクチャ","設計","フレームワーク"]
_STRATEGY_WORDS = ["戦略","施策","優先","差別化","競合","ポジショニング","KPI","ROI","目標"]
_EXEC_WORDS = ["実行","手順","タスク","スケジュール","チェック","改善","運用","効率"]
_EMOTION_WORDS = ["不安","ムカつく","なぜ俺","怖い","どうせ","無理","クソ","無能","イライラ","最悪"]

def _load_score_words(tenant_id: str) -> dict:
    try:
        db = get_db()
        for tid in [tenant_id, DEFAULT_TENANT]:
            doc = db.collection("system_settings").document(f"score_config_{tid}").get()
            if doc.exists:
                d = doc.to_dict() or {}
                def _split_words(s):
                    import re as _re
                    return [w.strip() for w in _re.split(r"[,\n]+", s or "") if w.strip()]
                return {
                    "struct": _split_words(d.get("struct_words","")),
                    "strategy": _split_words(d.get("strategy_words","")),
                    "exec": _split_words(d.get("exec_words","")),
                    "emotion": _split_words(d.get("emotion_words","")),
                    "struct_pt": int(d.get("struct_pt", 3)),
                    "strategy_pt": int(d.get("strategy_pt", 2)),
                    "exec_pt": int(d.get("exec_pt", 1)),
                    "emotion_pt": int(d.get("emotion_pt", -3)),
                }
    except Exception:
        pass
    return {}

def _calc_score(text: str, tenant_id: str = "default") -> int:
    t = text or ""
    score = 0
    sw = _load_score_words(tenant_id)
    struct_words = sw.get("struct", _STRUCT_WORDS)
    strategy_words = sw.get("strategy", _STRATEGY_WORDS)
    exec_words = sw.get("exec", _EXEC_WORDS)
    emotion_words = sw.get("emotion", _EMOTION_WORDS)
    struct_pt = sw.get("struct_pt", 3)
    strategy_pt = sw.get("strategy_pt", 2)
    exec_pt = sw.get("exec_pt", 1)
    emotion_pt = sw.get("emotion_pt", -3)
    for w in struct_words:
        if w in t: score += struct_pt
    for w in strategy_words:
        if w in t: score += strategy_pt
    for w in exec_words:
        if w in t: score += exec_pt
    for w in emotion_words:
        if w in t: score += emotion_pt
    return score

def _update_level_score(tenant_id: str, uid: str, delta: int):
    try:
        db = get_db()
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() if snap.exists else {}
        cur = int(d.get("level_score", 0))
        new_score = cur + delta
        # ランク計算
        cfg_doc = None
        for tid in [tenant_id, DEFAULT_TENANT]:
            cfg_snap = db.collection("system_settings").document(f"rank_config_{tid}").get()
            if cfg_snap.exists:
                cfg_doc = cfg_snap.to_dict() or {}
                break
        r1t = int((cfg_doc or {}).get("rank_1_threshold", 80))
        r2t = int((cfg_doc or {}).get("rank_2_threshold", 200))
        r3t = int((cfg_doc or {}).get("rank_3_threshold", 450))
        r4n = (cfg_doc or {}).get("rank_4_name", "設計者")
        r3n = (cfg_doc or {}).get("rank_3_name", "戦略家")
        r2n = (cfg_doc or {}).get("rank_2_name", "実行者")
        r1n = (cfg_doc or {}).get("rank_1_name", "追従者")
        if new_score > r3t: rank = r4n
        elif new_score > r2t: rank = r3n
        elif new_score > r1t: rank = r2n
        else: rank = r1n
        db.collection("users").document(uid).set({
            "level_score": new_score,
            "level": rank,
            "level_last_delta": delta,
            "level_last_updated_at": fs.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception:
        pass

def _save_message(tenant_id: str, uid: str, chat_id: str, role: str, content: str, cases: list = None, structured: dict = None, images: list = None):
    ref = _messages_ref(tenant_id, uid, chat_id)
    doc = {
        "role":    role,
        "content": content,
        "ts":      fs.SERVER_TIMESTAMP,
    }
    if cases:
        doc["cases"] = cases
    if structured:
        doc["structured"] = structured
    if images:
        doc["images"] = [{"mime_type": img.get("mime_type","image/png"), "gcs_url": img.get("gcs_url","")} for img in images if img.get("gcs_url")]
    ref.add(doc)

def _load_history(tenant_id: str, uid: str, chat_id: str = "main", limit: int = 20) -> List[dict]:
    ref = _messages_ref(tenant_id, uid, chat_id)
    docs = ref.order_by("ts").limit_to_last(limit).get()
    result = []
    for d in docs:
        data = d.to_dict() or {}
        msg = {"role": data.get("role", "user"), "content": data.get("content", "")}
        if data.get("cases"):
            msg["cases"] = data["cases"]
        if data.get("structured"):
            msg["structured"] = data["structured"]
        if data.get("images"):
            msg["images"] = data["images"]
        result.append(msg)
    return result

PLAN_DEFINITION = """

【ASCENDプラン定義 — 以下以外の情報は絶対に捏造・推測するな。不明な場合は「プラン詳細はYs Consulting Officeにお問い合わせください」とのみ回答せよ】

■ STARTER：¥0（新規7日間）
 エンジン: Core / AUTOモード
 利用可能: AIチャット(AUTOモード)、RAG検索、レベルスコア
 対象外: 診断機能全般、画像生成、ファイル診断、固定概念レポート、個人相談、投資シグナル

■ STANDARD：¥9,800/月
 エンジン: Core / 7モード対応
 利用可能: AIチャット(7モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ(構造/課題/比較/矛盾/実行)、画像生成、画像・ファイル解析(チャット内)
 対象外: ファイル診断(Ultraエンジン)、固定概念レポート、個人相談、投資シグナル、ASCEND Ultra/Apex

■ PRO：¥39,800/月
 エンジン: Ultra / 全19モード対応
 利用可能: AIチャット(全19モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ全6種、ファイル診断(Chain of Thought分析)、固定概念レポート(LGBM自動生成)、画像生成、画像ギャラリー、個人相談(スレッド往復)、ASCEND Ultra解放
 対象外: 投資シグナル、ASCEND Apex

■ APEX：¥89,800/月
 エンジン: Apex / 全19モード対応
 利用可能: 全機能解放、AIチャット(全19モード)、ファイル診断、固定概念レポート、投資シグナル(全銘柄)、ASCEND Apex(最上位AIエンジン)、個人相談、画像生成、ギャラリー、診断タブ全8種(投資シグナルタブ含む)

■ ULTRA：¥300,000/月〜（要相談・顧問契約）
 エンジン: Apex / 全19モード対応
 利用可能: ASCEND全機能完全解放、Ys Consulting Office顧問契約付き、社員10名まで個別アカウント発行、企業テナント共有(RAG・診断履歴)、月次戦術レポート提出、新機能先行利用、月次ミーティング・直接支援
 契約・問い合わせ: Ys Consulting Officeに直接連絡（UID記載必須）"""

def _load_tenant_system_prompt(tenant_id: str, uid: str = "") -> str:
    tenant_prompt = DEFAULT_SYSTEM_PROMPT
    try:
        db = get_db()
        doc = db.collection("tenant_settings").document(tenant_id).get()
        if doc.exists:
            sp = (doc.to_dict() or {}).get("system_prompt", "")
            if sp:
                tenant_prompt = sp
    except Exception:
        pass
    if not uid:
        return tenant_prompt
    try:
        db = get_db()
        u_snap = db.collection("users").document(uid).get()
        if u_snap.exists:
            u = u_snap.to_dict() or {}
            custom = (u.get("custom_sys_prompt") or "").strip()
            mode = (u.get("custom_prompt_mode") or "append").strip()
            if custom:
                if mode == "replace":
                    return custom + PLAN_DEFINITION if False else custom
                else:
                    return tenant_prompt + "\n\n" + custom
    except Exception:
        pass
    return tenant_prompt

def _build_system_with_rag(tenant_id: str, query: str, system_prompt: str, uid: str = ""):
    """returns (prompt_str, chunks_list)"""
    try:
        chunks = rag_retrieve_chunks(tenant_id=tenant_id, query=query, top_k=5)
        if uid:
            try:
                user_chunks = rag_retrieve_chunks(tenant_id=f"user__{uid}", query=query, top_k=15)
                existing_ids = {c.get("chunk_id") for c in chunks}
                chunks = chunks + [c for c in user_chunks if c.get("chunk_id") not in existing_ids]
            except Exception:
                pass
        if chunks:
            rag_text = "\n\n---\n\n".join(
                f"【ナレッジ: {c.get('title', '')}】\n{c.get('text', '')}"
                for c in chunks
            )
            # 問いの型判定: 知識・定義・方法系 → RAG即答優先 / 感情・相談系 → カスタム優先
            import re as _re_qt
            _knowledge_patterns = [
                r"とは[？?]?$", r"するには[？?]?$", r"の方法", r"教えて", r"について",
                r"どうやって", r"手順", r"やり方", r"コツ", r"ポイント",
                r"の行動", r"直結", r"上げるには", r"向上させる", r"改善",
                r"とは何", r"とはどういう", r"の意味", r"の定義",
                r"チェックリスト", r"一覧", r"リスト", r"全問", r"評価",
                r"学べる", r"ここで学", r"項目", r"ランキング", r"比較",
            ]
            _is_knowledge_query = any(
                _re_qt.search(p, query) for p in _knowledge_patterns
            )
            if _is_knowledge_query:
                return (
                    f"【知識回答モード】以下の複数の参照ナレッジを統合して質問に答えよ。"
                    f"各ナレッジの具体的な内容・事例・表現を活かし、抽象的・一般論的な回答は禁止。"
                    f"キャラクターの口調・絵文字を維持しながら説得力ある具体的な回答をせよ。問いかけ返し・感情確認禁止。"
                    f"【出力形式】比較・チェックリスト・一覧・評価項目を含む場合は必ずMarkdown表で出力せよ。"
                    f"Markdown表のセパレーター行は | --- | --- | 形式のみ（:や=や-の4つ以上連続は禁止）。"
                    f"重要ポイントは**太字**で強調せよ。\n\n"
                    f"【参照ナレッジ({len(chunks)}件)】\n{rag_text}\n\n{system_prompt}"
                ), chunks
            else:
                return f"{system_prompt}\n\n【参照ナレッジ】\n{rag_text}", chunks
    except Exception:
        pass
    return system_prompt, []

# ── エンドポイント ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = "main"
    ai_tier: str = "core"
    purpose_mode: str = "auto"
    chat_mode: str = "consult"

class ChatResponse(BaseModel):
    reply: str
    chat_id: str
    msg_id: str
    cases: list = []
    images: list = []
    structured: Optional[dict] = None

class SessionInfo(BaseModel):
    chat_id: str
    title: str
    updated_at: Optional[str] = None

@router.post("/send", response_model=ChatResponse)
def send_message(req: ChatRequest, payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id   = (req.chat_id or "main").strip() or "main"

    _ensure_session(tenant_id, uid, chat_id)

    history = _load_history(tenant_id, uid, chat_id)
    messages = history + [{"role": "user", "content": req.message}]

    base_prompt   = _load_tenant_system_prompt(tenant_id, uid=uid)
    # chat_mode / is_talk を先に確定
    chat_mode = (req.chat_mode or "consult").strip().lower()
    is_talk = chat_mode == "talk"


    # ── カスタムプロンプトモード判定 ────────────────────────────────
    # replaceモード = 専用ボットモード（LENS/PURPOSE/脳内カルテ全スキップ）
    _is_custom_replace = False
    try:
        _mode_doc = get_db().collection("users").document(uid).get()
        if _mode_doc.exists:
            _md = _mode_doc.to_dict() or {}
            _has_custom = bool(_md.get("custom_sys_prompt", ""))
            _is_replace_mode = _md.get("custom_prompt_mode", "append") == "replace"
            # replaceモード + 会話モード → 専用ボット動作（コンサル指示全スキップ）
            # replaceモード + 相談モード → カスタム+コンサルAI（スキップしない）
            _is_custom_replace = _has_custom and _is_replace_mode and is_talk
    except Exception:
        pass

    # 脳内カルテ更新（相談モードのみ・専用ボットモード以外）
    intent_state = {}
    import threading as _threading
    _intent_result = {}
    _intent_thread = None
    if not is_talk and not _is_custom_replace:
        def _run_intent():
            try:
                _intent_result["state"] = update_user_intent_state(uid, tenant_id, history, req.message)
            except Exception:
                pass
        _intent_thread = _threading.Thread(target=_run_intent, daemon=True)
        _intent_thread.start()

    # QueryPlan生成（相談モードのみ・専用ボットモード以外）
    query_plan = {}
    if not is_talk and not _is_custom_replace:
        try:
            query_plan = generate_query_plan(req.message, tenant_id, "mixed")
        except Exception:
            pass
    # SummaryLens選択（専用ボットモード以外）
    lens_preset, lens_hier = "expert", "raw"
    if not _is_custom_replace:
        try:
            lens_preset, lens_hier = lgbm_select_summary_lens(req.message, "auto")
        except Exception:
            lens_preset, lens_hier = "expert", "raw"
        if query_plan.get("summary_lens", {}).get("preset"):
            lens_preset = query_plan["summary_lens"]["preset"]

    # 脳内カルテをsystem_promptに注入（専用ボットモード以外）
    if _intent_thread: _intent_thread.join(timeout=0.0)
    intent_state = _intent_result.get("state", {}) if not _is_custom_replace else {}
    intent_ctx = ""
    if intent_state and not _is_custom_replace:
        intent_ctx = f"""\n\n【ユーザーの脳内カルテ（深層プロファイル）】
・ステージ: {intent_state.get('current_stage','')}
・真の渇望: {intent_state.get('true_desire','')}
・バイアス: {intent_state.get('bias','')}
・不足観点: {intent_state.get('missing_piece','')}
※上記を踏まえ、単なる回答ではなく「格を上げるための介入」を行え。"""

    if not is_talk:
        system_prompt, _rag_chunks = _build_system_with_rag(tenant_id, req.message, base_prompt, uid=payload.get("uid",""))
    else:
        system_prompt, _rag_chunks = base_prompt, []
        # 会話モード：カスタムプロンプトがある場合はRAG+URL注入を発動
        try:
            _talk_user_doc = get_db().collection("users").document(uid).get()
            _talk_has_custom = bool((_talk_user_doc.to_dict() or {}).get("custom_sys_prompt", "")) if _talk_user_doc.exists else False
        except Exception:
            _talk_has_custom = False
        if _talk_has_custom:
            # 会話モード: カスタムプロンプトのみをベースにRAG知識を注入（形式模倣防止）
            _talk_custom_only = ((_talk_user_doc.to_dict() or {}).get("custom_sys_prompt") or "").strip()
            _talk_base = _talk_custom_only if (_talk_custom_only and not _is_custom_replace) else base_prompt
            system_prompt, _rag_chunks = _build_system_with_rag(tenant_id, req.message, _talk_base, uid=uid)
        if _talk_has_custom and not _is_custom_replace:
            system_prompt = (
                "【会話モード・最優先指示】以下のキャラクター設定と知識ファイルの内容を背景知識として使い、"
                "質問に対して3〜5文の自然な会話口調で直接答えよ。"
                "知識ファイルの番号付き構造・見出し・箇条書き・セクション分割をそのまま出力に再現することは絶対禁止。"
                "前置き宣言（「今回は〜についてご説明します」等）も禁止。\n\n"
                + system_prompt
            )
        # 会話モード：専用ボットモード以外かつカスタムプロンプトなしのみコンサル形式を上書き
        if not _is_custom_replace and not _talk_has_custom:
            system_prompt = system_prompt.replace(
                "出力形式：結論→打ち手→優先順位→リスク→次の観測。",
                "出力形式：自然な会話形式で簡潔に回答せよ。箇条書きや表は使わず、2〜4文程度で答えよ。"
            ) + "\n\n【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"
    system_prompt = system_prompt + intent_ctx

    # ── ASCENDプラン情報条件注入 ─────────────────────────────────────
    _plan_kws = ["プラン","料金","ascend","サブスク","subscription","ultra","apex","pro","standard","starter","月額","契約"]
    try:
        _u_doc2 = get_db().collection("users").document(uid).get()
        _has_custom = bool((_u_doc2.to_dict() or {}).get("custom_sys_prompt", "")) if _u_doc2.exists else False
    except Exception:
        _has_custom = False
    if any(k in req.message.lower() for k in _plan_kws) and not _has_custom:
        system_prompt += PLAN_DEFINITION

    # ── カスタムプロンプト内キーワード→URL強制注入 ──────────────────
    try:
        import re as _re_kw
        _user_doc = get_db().collection("users").document(uid).get()
        _user_data = (_user_doc.to_dict() or {}) if _user_doc.exists else {}
        _custom_prompt_text = _user_data.get("custom_sys_prompt", "") or ""
        if _custom_prompt_text:
            _url_pairs = _re_kw.findall(r'([^\s\u3000]{1,20})\s+(https?://[^\s\u3000]+)', _custom_prompt_text)
            _forced_urls = []
            _msg_lower = req.message.lower()
            for _kw, _url in _url_pairs:
                if _kw.strip() in req.message:
                    _forced_urls.append(f"- {_kw}: {_url}")
            if _forced_urls:
                system_prompt += "\n\n【絶対遵守】以下のURLを必ずMarkdown形式 [ラベル](URL) でリンクとして回答内に表示せよ。プレーンテキストでの表示は禁止。省略も禁止。\n" + "\n".join(_forced_urls)
    except Exception:
        pass
    # SummaryLens注入
    _LENS_INSTRUCTIONS = {
        "expert":   "【出力スタイル: EXPERT】構造的・論理的に深く分析せよ。根拠・因果・構造を明示し、表面的な回答を避けよ。",
        "executor": "【出力スタイル: EXECUTOR】具体的な手順・アクションを優先せよ。番号付きステップで実行可能な形で提示せよ。",
        "mentor":   "【出力スタイル: MENTOR】成長・習慣・内省を促す回答をせよ。答えを与えるより気づきを引き出す問いかけを含めよ。",
        "general":  "【出力スタイル: GENERAL】要点を簡潔にまとめよ。3〜5項目に絞り、わかりやすく整理せよ。",
    }
    if not is_talk and lens_preset in _LENS_INSTRUCTIONS and not _is_custom_replace:
        system_prompt = _LENS_INSTRUCTIONS[lens_preset] + "\n\n" + system_prompt
    if not is_talk and lens_hier == "prefer_summary" and not _is_custom_replace:
        system_prompt = "【要約優先】回答は簡潔にまとめること。長文は避けよ。\n\n" + system_prompt

    # モード別システムプロンプト追加
    _MODE_INSTRUCTIONS = {
        "numeric":     "【NUMERICモード】数値・KPI・売上・コスト分析に特化せよ。必ず数値・比率・計算式を使って回答せよ。定性的な説明より定量的な根拠を優先せよ。",
        "strategy":    "【STRATEGYモード】競合分析・差別化・ポジショニング戦略に特化せよ。3C/4P/SWOT等のフレームワークを活用し、戦略的選択肢と優先順位を提示せよ。",
        "control":     "【CONTROLモード】組織・権限・業務フロー・マネジメント構造に特化せよ。責任分担・権限設計・フロー最適化の観点から回答せよ。",
        "growth":      "【GROWTHモード】スキル・習慣・成長設計に特化せよ。具体的なトレーニング方法・習慣化ステップ・成長指標を提示せよ。",
        "analysis":    "【ANALYSISモード】データ・事象の多角的解析に特化せよ。因果関係・相関・パターンを分解し、複数の解釈仮説を提示せよ。",
        "planning":    "【PLANNINGモード】ロードマップ・フェーズ設計に特化せよ。時系列・マイルストーン・依存関係を明示したアクションプランを提示せよ。",
        "risk":        "【RISKモード】リスク特定・評価・対策設計に特化せよ。発生確率×影響度でリスクを評価し、回避・軽減・転嫁・受容の選択肢を提示せよ。",
        "marketing":   "【MARKETINGモード】集客・ブランディング・広告施策に特化せよ。ターゲット定義・チャネル選定・CVR改善の観点から具体施策を提示せよ。",
        "diagnosis":   "【DIAGNOSISモード】現状課題の発見・根本原因分析に特化せよ。なぜなぜ分析・ロジックツリー等で根本原因を特定し、表面的対処を避けよ。",
        "forecast":    "【FORECASTモード】将来予測・シナリオ分析に特化せよ。楽観・中立・悲観の3シナリオを定量的に提示し、各シナリオの発生条件を明示せよ。",
        "finance":     "【FINANCEモード】財務・投資・資金計画分析に特化せよ。ROI・回収期間・キャッシュフロー・損益分岐点を数値で示せ。投資アドバイスではなく分析として提示せよ。",
        "hr":          "【HRモード】採用・評価・組織設計・人材育成に特化せよ。評価基準・採用要件・育成ステップを構造的に提示せよ。",
        "negotiation": "【NEGOTIATIONモード】交渉・説得・合意形成戦略に特化せよ。相手の利害・BATNAを分析し、Win-Winの合意シナリオと交渉戦術を提示せよ。",
        "creative":    "【CREATIVEモード】アイデア発想・コンセプト設計に特化せよ。既存の枠を超えた発想を複数提示し、実現可能性と独自性を評価せよ。",
        "summary":     "【SUMMARYモード】要約・整理に特化せよ。要点を3〜5項目に絞り、階層構造で簡潔に整理せよ。長文は禁止。",
        "legal":       "【LEGALモード】法務・規約・コンプライアンスの解説に特化せよ。ただし法的助言ではなく情報提供として提示し、専門家確認を推奨せよ。",
        "coaching":    "【COACHINGモード】自己変革・思考パターン改善に特化せよ。質問・内省促進・気づきの提供を優先し、答えを与えるより考えさせる回答をせよ。",
        "ops":         "【OPSモード】業務改善・効率化・オペレーション最適化に特化せよ。ボトルネック特定・工数削減・標準化・自動化の観点から具体的改善策を提示せよ。",
        "tech":        "【TECHモード】技術・エンジニアリング・システム設計に特化せよ。技術的トレードオフ・アーキテクチャ選定・実装方針を構造的に提示せよ。",
    }
    _mode_key = (req.purpose_mode or "auto").strip().lower()
    if not is_talk and _mode_key in _MODE_INSTRUCTIONS and not _is_custom_replace:
        system_prompt = _MODE_INSTRUCTIONS[_mode_key] + "\n\n" + system_prompt
    # FINANCEモード時: Firestoreの実シグナルデータを注入
    if _mode_key == "finance":
        try:
            _db = get_db()
            _sig_docs = list(_db.collection("investment_signals").limit(200).stream())
            _sig_docs.sort(key=lambda d: str((d.to_dict() or {}).get("asof_date","")), reverse=True)
            if _sig_docs:
                _sig_ref = _db.collection("investment_signals").document(_sig_docs[0].id)
                _sig_date = (_sig_docs[0].to_dict() or {}).get("asof_date","")
                _goal = [d.to_dict() for d in _sig_ref.collection("goal_bottom").limit(500).stream()]
                _watch = [d.to_dict() for d in _sig_ref.collection("watch_big_sell").limit(500).stream()]
                _all_stocks_fin = [d.to_dict() for d in _sig_ref.collection("all_stocks").limit(2000).stream()]
                _all = _goal + _watch + _all_stocks_fin
                import re as _re_fin
                _code_hits = _re_fin.findall(r'(?<![\d])\d{4,6}(?![\d])', req.message)
                import unicodedata as _ucd
                _msg_clean = _ucd.normalize("NFKC", req.message.replace("\u3000"," "))
                _matched = []
                for s in _all:
                    _c = str(s.get("code",""))
                    _n = str(s.get("company_name",""))
                    if _c in _code_hits or (_c.rstrip("0") in _code_hits) or any(_c == h+"0" or _c == h+"00" for h in _code_hits):
                        _matched.append(s); continue
                    _n_norm = _ucd.normalize("NFKC", _n)
                    _msg_words = _re_fin.findall(r'[A-Za-z]{2,}', _msg_clean) + _re_fin.findall(r'[\u4e00-\u9fff\u30a0-\u30ff]{2,}', _msg_clean)
                # 2〜4文字の部分スライスも候補に追加（東電株価→東電・電株・株価等）
                _cjk_all = _re_fin.findall(r'[\u4e00-\u9fff\u30a0-\u30ff]+', _msg_clean)
                for _cjk in _cjk_all:
                    for _slen in [2, 3]:
                        for _si in range(len(_cjk) - _slen + 1):
                            _msg_words.append(_cjk[_si:_si+_slen])
                    def _subseq(w, t):
                        it = iter(t)
                        return all(c in it for c in w)
                    if any(w in _n_norm or _subseq(w, _n_norm) for w in _msg_words if len(w)>=2):
                        _matched.append(s)
                # マッチなし かつ コードが抽出できた場合 → all_stocksから直接取得
                if not _matched and _code_hits:
                    for _chit in _code_hits:
                        try:
                            for _cid in [_chit, _chit+"0", _chit+"00"]:
                                _sd = _sig_ref.collection("all_stocks").document(_cid).get()
                                if _sd.exists:
                                    _matched.append(_sd.to_dict()); break
                        except:
                            pass
                # コードでdedup
                _seen_codes = set()
                _deduped = []
                for _s in _matched:
                    _sc = str(_s.get("code",""))
                    if _sc not in _seen_codes:
                        _seen_codes.add(_sc)
                        _deduped.append(_s)
                _matched = _deduped
                _finance_candidates = []
                if len(_matched) >= 2:
                    _finance_candidates = [f"{r.get('code')} {r.get('company_name')}" for r in _matched[:5]]
                    _cand_list = "\n".join([f"{i+1}. {c}" for i,c in enumerate(_finance_candidates)])
                    system_prompt += f"\n\n【銘柄候補が複数ヒットしました】以下の候補を番号付きリストでユーザーに提示し、「以下のどちらの銘柄ですか？」と必ず聞き返せ。推測で回答することは絶対禁止。\n{_cand_list}"
                def _fmt_stock(r):
                    return (
                        f"銘柄: {r.get('code')} {r.get('company_name')} セクター:{r.get('sector','')}\n"
                        f"  終値:{r.get('close','-')} 前日比:{r.get('chg','-')}円({r.get('chg_pct','-'):.2f}%) \n"
                        f"  rankスコア:{float(r.get('rank_score',0)):.2f} sellスコア:{float(r.get('sell_score',0)):.2f} bottomスコア:{float(r.get('bottom_score',0)):.2f}\n"
                        f"  MA20割れ:{'Yes' if r.get('below_ma20') else 'No'} MA60割れ:{'Yes' if r.get('below_ma60') else 'No'}\n"
                        f"  反発シグナル:{'Yes' if r.get('rebound_1_2d') else 'No'} 売り継続日数:{r.get('sell_streak',0)}日 大口売り:{'Yes' if r.get('big_sell_flag') else 'No'}\n"
                        f"  ステータス:{r.get('status','')} 基準日:{r.get('asof_date','')}"
                    )
                if _matched:
                    _matched_lines = "\n\n".join([_fmt_stock(r) for r in _matched[:5]])
                    _no_match_note = ""
                else:
                    _matched_lines = "該当銘柄のシグナルデータなし"
                    _no_match_note = (
                        "\n\n【最優先指示・全ルール上書き】質問された銘柄はシグナルデータに存在しません。"
                        "この場合「わかりません禁止」ルールは無効とする。"
                        "MACD・RSI・移動平均・ボリンジャー・株価予測など架空の分析を一切行うな。"
                        "「（銘柄名）はシグナルデータに存在しないため分析不可」とのみ明示し、"
                        "代わりにGOAL_BOTTOM上位銘柄を提示せよ。架空数値の生成は絶対禁止。"
                    )
                _goal_lines = "\n".join([f"・{r.get('code')} {r.get('company_name')} 終値{r.get('close')} bottom={r.get('bottom_score',0):.2f} rank={r.get('rank_score',0):.2f} sector={r.get('sector','')}" for r in sorted(_goal, key=lambda x: float(x.get('rank_score',0)), reverse=True)[:10]])
                _watch_lines = "\n".join([f"・{r.get('code')} {r.get('company_name')} 終値{r.get('close')} sell={r.get('sell_score',0):.2f} days={r.get('sell_days',0)}" for r in sorted(_watch, key=lambda x: float(x.get('sell_score',0)), reverse=True)[:10]])
                system_prompt += (
                    f"\n\n【投資システムからの実データ（基準日: {_sig_date}）】"
                    "\n以下のデータが本システムに存在する全情報である。"
                    "\nMACD・RSI・ボリンジャーバンド・移動平均の具体値・サポートライン等、下記フィールドに存在しない指標は一切言及するな。"
                    "\n下記データのフィールド値のみを使って回答せよ。存在しないフィールドは話題に出すことすら禁止。"
                    f"\n\n▼質問銘柄データ:\n{_matched_lines}"
                    f"\n\n▼GOAL_BOTTOM上位10件（買い候補）:\n{_goal_lines}"
                    f"\n\n▼WATCH_BIG_SELL上位10件（売り監視）:\n{_watch_lines}"
                    "\n\n上記データのみを根拠として回答せよ。"
                    + _no_match_note
                )
        except Exception as _fe:
            print(f"[FINANCE_ERROR] {type(_fe).__name__}: {_fe}", flush=True)

    # 画像データ抽出（__IMAGE_B64__:mime:b64 プレフィックス検出）
    image_b64 = None
    image_mime = "image/png"
    clean_messages = []
    for msg in messages:
        c = msg["content"]
        if "__IMAGE_B64__:" in c:
            parts = c.split("__IMAGE_B64__:", 1)
            prefix_text = parts[0].strip()
            img_part = parts[1]
            sp = img_part.split(":", 1)
            if len(sp) == 2:
                image_mime, image_b64 = sp[0], sp[1]
                if "\n" in image_b64:
                    image_b64 = image_b64.split("\n")[0]
            clean_messages.append({"role": msg["role"], "content": prefix_text or "この画像を分析してください"})
        else:
            clean_messages.append(msg)

    # 画像生成判定
    generated_images = []
    if _is_image_gen_request(req.message, has_image=image_b64 is not None):
        try:
            reply, generated_images = _generate_image(req.message, image_b64, image_mime)
        except Exception as _e:
            reply = f"画像生成エラー: {_e}"
    else:
        try:
            print(f"[FINANCE_DEBUG] system_prompt末尾500: {system_prompt[-500:]}", flush=True)
            reply = call_llm(
                system_prompt=system_prompt,
                messages=clean_messages,
                ai_tier=req.ai_tier,
                image_b64=image_b64,
                image_mime=image_mime,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI呼び出しエラー: {e}")

    # suggested cases 生成
    cases = []
    try:
        from api.core.llm_client import call_llm as _cllm
        _cases_prompt = f"以下の会話に対して、ユーザーが次に相談しそうな事案を3件、日本語で箇条書きせず1行ずつ返せ。マーク不要。\nQ: {req.message}\nA: {reply[:500]}"
        _cases_raw = _cllm(
            system_prompt="次の相談候補を3件だけ出力せよ。余分なテキスト不要。",
            messages=[{"role": "user", "content": _cases_prompt}],
            ai_tier="core", max_tokens=256
        )
        cases = [l.strip() for l in _cases_raw.strip().split("\n") if l.strip()][:3]
    except Exception:
        cases = []

    # 構造化データ生成（戦略相談時のみ・画像生成・雑談は除外）
    structured = None
    _consulting_intents = {"相談", "意思決定", "分析", "作成", "予測", "投資"}
    _qp_intent = query_plan.get("intent", "")
    _is_talk_intent = "雑談" in _qp_intent
    _mode_forced = (req.purpose_mode or "auto").strip().lower() != "auto"
    _is_consulting = (not is_talk) and ((_mode_forced) or ((not _is_talk_intent) and (any(i in _qp_intent for i in _consulting_intents))))
    # 相談モードでも雑談intentの場合は会話形式で返答
    if (is_talk or _is_talk_intent) and not system_prompt.endswith("【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"):
        system_prompt = system_prompt.replace(
            "出力形式：結論→打ち手→優先順位→リスク→次の観測。",
            "出力形式：自然な会話形式で簡潔に回答せよ。箇条書きや表は使わず、2〜4文程度で答えよ。"
        ) + "\n\n【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"
    if not generated_images and _is_consulting:
        try:
            import json as _json_s, re as _re_s
            _mode_upper = _mode_key.upper() if _mode_key != "auto" else ""
            _mode_line = f"modeは必ず {_mode_upper} で固定（変更禁止）\n" if _mode_upper else "modeは問いの内容に応じてSTRATEGY/NUMERIC/DIAGNOSIS/PLANNING/RISK/MARKETING/FINANCE/HRから選択\n"
            _sp = (
                "問いの型を判定し最適なカード構成でJSONのみ出力せよ。前置き・後置き・コードブロック絶対禁止。\n"
                "【問いの型と対応カード（必ずこの分類に従え）】\n"
                "action（行動定義型: 〜するには？〜の行動は？〜すべきことは？）→ cards:[即実行アクション, 阻害要因・注意点, 優先順位・判断基準]\n"
                "analysis（現状分析型: 〜の問題は？〜を分析して、〜の原因は？）→ cards:[現状整理, 問題・リスク, 推奨方針]\n"
                "forecast（予測型: 〜どうなる？〜の見通しは？〜のシナリオは？）→ cards:[楽観シナリオ, 悲観シナリオ, 対策・備え]\n"
                "decision（意思決定型: 〜すべき？〜AかBか？〜の選択は？）→ cards:[メリット・根拠, リスク・代償, 推奨判断]\n"
                "definition（定義・説明型: 〜とは？〜の意味は？〜の仕組みは？）→ cards:[本質・定義, 構造分解, 実装・応用]\n"
                "必須キー: summary, question_type, cards, analysis, actions, value_message\n"
                "question_type: action/analysis/forecast/decision/definition のいずれか\n"
                "cards: 3要素の配列。各要素は {title:string, items:string[]} 形式。itemsは5件\n"
                "itemsはユーザーが提示した実情報を元に具体的に記述。情報不足時は仮説と明記。架空数値捏造絶対禁止\n"
                "analysis必須キー: type, urgency, importance, mode\n"
                "urgency/importanceは 高/中/低 のいずれか\n"
                + _mode_line
                + f"\n【今回の相談】: {req.message[:400]}\n"
                f"【今回の回答要約】: {reply[:800]}\n"
                + (f"【投資シグナル実データ】: {system_prompt[-1500:]}\n実データにない指標は絶対捏造禁止。\n" if _mode_key == "finance" else "")
                + '\n出力例(action型):\n{"summary":"売上に直結する行動の本質は客数×客単価の2軸を動かす即実行アクション。","question_type":"action","cards":[{"title":"即実行アクション","items":["初回接触でその場で次回予約を取る","入室5分以内にオプション前提の空気を作る","接客終了前に再来理由を言語化して渡す","LINE/DMで24時間以内に再接触する","無言時間を作らず滞在満足度を最大化する"]},{"title":"阻害要因・注意点","items":["予約を取らずに終わらせる習慣","オプション提案のタイミングが遅い","再来理由を渡さずお客様任せにしている","接触頻度が低く関係性が薄れている","選択肢を多く出しすぎて迷わせている"]},{"title":"優先順位・判断基準","items":["まず次回予約率を計測する","オプション提案率を記録する","24時間以内フォロー率を追う","リピート間隔を短縮できているか確認","客単価の変化を週次でモニタリング"]}],"analysis":{"type":"行動定義","urgency":"高","importance":"高","mode":"STRATEGY"},"actions":["今日の接客で次回予約を必ず取る","オプション提案タイミングを入室5分以内に固定する","接客後24時間以内のフォローを仕組み化する"],"value_message":"売上に直結する行動は次の金を今日決めさせる3つだけ。"}'
            )
            _sr = call_llm(
                system_prompt="JSONのみ出力。指定キー構造厳守。前置き・後置き・コードブロック完全禁止。余計なキー追加禁止。",
                messages=[{"role": "user", "content": _sp}],
                ai_tier="core", max_tokens=900
            )
            _m = _re_s.search(r'\{.*\}', _sr, _re_s.DOTALL)
            if _m:
                _parsed = _json_s.loads(_m.group(0))
                if all(k in _parsed for k in ["summary","cards","analysis","actions","value_message"]):
                    _analysis = _parsed.get("analysis", {})
                    _cards = _parsed.get("cards")
                    _cards_ok = (
                        (isinstance(_cards, list) and len(_cards) >= 2 and all("title" in c and "items" in c for c in _cards))
                        or (isinstance(_cards, dict) and any(k in _cards for k in ["current","risk","plan"]))
                    )
                    if _cards_ok and all(k in _analysis for k in ["type","urgency","importance","mode"]):
                        structured = _parsed
        except Exception as _se:
            structured = None
            print(f"[STRUCTURED_ERROR] {type(_se).__name__}: {_se}", flush=True)

    # レベルスコア加算
    _delta = _calc_score(req.message, tenant_id)
    _update_level_score(tenant_id, uid, _delta)

    # RAGチャンク採用記録（LGBM教師データ）＋固定概念カウント更新（相談モードのみ）
    if not is_talk:
        try:
            chunks = _rag_chunks
            if chunks:
                db = get_db()
                # 固定概念観測カウント+1（RAGチャンク採用時のみ）
                try:
                    _fc_snap = db.collection("users").document(uid).get()
                    _fc_d = _fc_snap.to_dict() if _fc_snap.exists else {}
                    _fc_cnt = int(_fc_d.get("use_count_since_report", 0)) + 1
                    db.collection("users").document(uid).set(
                        {"use_count_since_report": _fc_cnt},
                        merge=True
                    )
                except Exception:
                    pass
                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id") or chunk.get("doc_id","")
                    if chunk_id:
                        db.collection("tenants").document(tenant_id).collection("lgbm_training_logs").add({
                            "uid": uid,
                            "chunk_id": chunk_id,
                            "query": req.message[:500],
                            "score": float(chunk.get("score",0)),
                            "adopted": True,
                            "purpose_mode": "auto",
                            "recorded_at": __import__("datetime").datetime.now().isoformat(),
                            "tenant_id": tenant_id,
                            "label": 1,
                        })
        except Exception:
            pass

    _save_message(tenant_id, uid, chat_id, "user", req.message)
    # assistant save_message はGCS保存後に実行

    # GCS画像保存
    gcs_image_urls = []
    if generated_images:
        try:
            import os as _os, base64 as _b64_gs
            from google.cloud import storage as _gcs
            bucket_name = _os.environ.get("CENTRAL_BLOB_BUCKET","").strip()
            if bucket_name:
                _gc = _gcs.Client()
                _bkt = _gc.bucket(bucket_name)
                for _ii, _img in enumerate(generated_images):
                    try:
                        _img_bytes = _b64_gs.b64decode(_img["data"])
                        _ext = "png" if "png" in _img.get("mime_type","") else "jpg"
                        _path = f"chat_images/{tenant_id}/{uid}/{uuid.uuid4().hex[:8]}.{_ext}"
                        _blob = _bkt.blob(_path)
                        _blob.upload_from_string(_img_bytes, content_type=_img.get("mime_type","image/png"))
                        _url = f"https://storage.googleapis.com/{bucket_name}/{_path}"
                        gcs_image_urls.append(_url)
                        generated_images[_ii]["gcs_url"] = _url
                    except Exception:
                        pass
        except Exception:
            pass
    # GCS保存結果に関わらずFirestoreに画像記録
    print(f"[GALLERY_DEBUG] generated_images count: {len(generated_images)}", flush=True)
    _db_g = get_db()
    for _img in generated_images:
        try:
            _img_id = uuid.uuid4().hex
            _save_url = _img.get("gcs_url","")
            _db_g.collection("image_gallery").document(uid).collection("images").document(_img_id).set({
                "image_id": _img_id,
                "uid": uid,
                "tenant_id": tenant_id,
                "gcs_url": _save_url,
                "mime_type": _img.get("mime_type","image/png"),
                "prompt": req.message[:500],
                "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            })
        except Exception:
            pass

    _save_message(tenant_id, uid, chat_id, "assistant", reply, cases=cases, structured=structured, images=generated_images)
    # usage_logs に記録（total_chat_count 集計用）
    try:
        import datetime as _dt
        get_db().collection("usage_logs").add({
            "user_id": uid,
            "tenant_id": tenant_id,
            "prompt": req.message[:500],
            "purpose_mode": getattr(req, "purpose_mode", "auto"),
            "is_admin_test": False,
            "recorded_at": _dt.datetime.utcnow().isoformat(),
        })
    except Exception:
        pass

    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=cases, images=generated_images, structured=structured)

@router.get("/history/{chat_id}")
def get_history(chat_id: str, payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    messages  = _load_history(tenant_id, uid, chat_id)
    return {"messages": messages, "chat_id": chat_id}

@router.get("/sessions", response_model=List[SessionInfo])
def list_sessions(payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db        = get_db()
    prefix    = f"{SCOPE}__{tenant_id}__{uid}__"
    try:
        docs = (
            db.collection("chat_sessions")
            .where("uid", "==", uid)
            .where("tenant_id", "==", tenant_id)
            .where("scope", "==", SCOPE)
            .limit(50)
            .stream()
        )
        result = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_deleted", False):
                continue
            result.append(SessionInfo(
                chat_id    = data.get("chat_id", "main"),
                title      = data.get("title", data.get("chat_id", "main")),
                updated_at = str(data.get("updated_at", "")),
            ))
        result.sort(key=lambda x: x.updated_at or "", reverse=True)
        return result[:30]
    except Exception:
        return [SessionInfo(chat_id="main", title="main")]

@router.post("/session/new")
def new_session(payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id   = str(uuid.uuid4())[:8]
    _ensure_session(tenant_id, uid, chat_id)
    return {"chat_id": chat_id}

from fastapi import UploadFile, File, Form
import base64 as _base64

@router.post("/upload_attachment")
def upload_attachment(
    file: UploadFile = File(...),
    chat_id: str = Form("main"),
    payload: dict = Depends(verify_token)
):
    """ファイルをbase64化してRAG用テキスト抽出"""
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    
    filename = file.filename or "file"
    content = file.file.read()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    extracted = ""
    IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp")
    try:
        if ext in IMAGE_EXTS:
            import base64 as _b64
            mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
            b64 = _b64.b64encode(content).decode()
            extracted = f"__IMAGE_B64__:{mime}:{b64}"
        elif ext in ("txt", "md", "csv"):
            extracted = content.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            try:
                import pypdf, io
                reader = pypdf.PdfReader(io.BytesIO(content))
                extracted = "\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                extracted = content.decode("utf-8", errors="ignore")
        elif ext in ("xlsx", "xls"):
            try:
                import pandas as pd, io
                df = pd.read_excel(io.BytesIO(content))
                extracted = df.to_csv(index=False)
            except Exception:
                extracted = ""
        elif ext == "ods":
            try:
                import pandas as pd, io
                df = pd.read_excel(io.BytesIO(content), engine="odf")
                extracted = df.to_csv(index=False)
            except Exception:
                extracted = content.decode("utf-8", errors="ignore")[:4000]
        else:
            extracted = content.decode("utf-8", errors="ignore")[:4000]
    except Exception:
        extracted = ""

    if not extracted.startswith("__IMAGE_B64__:"):
        extracted = extracted[:4000]
    
    return {
        "filename": filename,
        "ext": ext,
        "size": len(content),
        "extracted_text": extracted,
        "preview": extracted[:200] if extracted else "(内容を読み取れませんでした)",
    }

class SuggestRequest(BaseModel):
    last_message: str
    last_reply: str

@router.post("/suggest")
def suggest_questions(req: SuggestRequest, payload: dict = Depends(verify_token)):
    from api.core.llm_client import call_llm
    import re
    try:
        prompt = f"""以下の会話の続きとして、ユーザーが次に相談しそうな事案を5件生成してください。
各項目は20〜40文字の日本語で、具体的な質問文として出力してください。
番号付きリストで出力し、余分な説明は不要です。

ユーザー: {req.last_message}
AI: {req.last_reply[:300]}

次に想定される事案（5件）:"""
        reply = call_llm(
            system_prompt="あなたは戦略コンサルタントです。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=512,
        )
        lines = [l.strip() for l in reply.strip().split("\n") if l.strip()]
        questions = []
        for l in lines:
            q = re.sub(r'^[\d\.\-\*\s]+', '', l).strip()
            if q and len(q) > 5:
                questions.append(q)
        return {"questions": questions[:5]}
    except Exception:
        return {"questions": []}

# ── 画像生成判定 ──────────────────────────────────────────────
_IMAGE_WORDS = ["画像","イメージ","イラスト","ロゴ","アイコン","バナー","ポスター","サムネ","image","illustration","logo","icon","banner","poster"]
_ACTION_WORDS = ["作って","作成","生成","描いて","描画","出力","デザイン","作る","generate","create","draw","design","render"]
_EDIT_WORDS = ["編集","加工","修正","変換","背景","切り抜","色変更","edit","modify","restyle"]
_ANALYSIS_WORDS = ["解析","分析","要約","読んで","説明","pdf","spreadsheet","excel","スプレッドシート"]

def _is_image_gen_request(text: str, has_image: bool = False) -> bool:
    t = (text or "").lower()
    has_subject = any(w in t for w in _IMAGE_WORDS)
    has_action  = any(w in t for w in _ACTION_WORDS)
    has_edit    = any(w in t for w in _EDIT_WORDS)
    has_analysis= any(w in t for w in _ANALYSIS_WORDS)
    if has_image and has_edit: return True
    if has_subject and has_action and not has_analysis: return True
    return False

def _generate_image(prompt: str, image_b64: str = None, image_mime: str = "image/png") -> tuple:
    """画像生成。(text, images_list) を返す。images_list = [{"mime_type":..,"data":b64str}]"""
    import os, base64 as _b64
    from google import genai as _genai
    from google.genai import types as _types
    api_key = os.environ.get("GEMINI_API_KEY","")
    client = _genai.Client(api_key=api_key) if api_key else _genai.Client()
    _IMAGE_MODELS = ["gemini-3.1-flash-image-preview","gemini-3-pro-image-preview","gemini-2.5-flash-image"]
    try:
        available = [m.name for m in client.models.list()]
        candidates = [m for m in _IMAGE_MODELS if any(m in a for a in available)]
        if not candidates: candidates = _IMAGE_MODELS
    except Exception:
        candidates = _IMAGE_MODELS

    strict_prompt = (
        "以下は画像生成の最終指示です。ユーザー指示を最優先し、勝手な解釈拡張・要素追加を極力しないこと。\n"
        "明示された要素は必ず反映し、明示されていない要素は勝手に足さないこと。\n"
        f"【ユーザー最終指示】\n{prompt}"
    )
    user_parts = [_types.Part(text=strict_prompt)]
    if image_b64:
        try:
            img_bytes = _b64.b64decode(image_b64)
            user_parts.append(_types.Part(inline_data=_types.Blob(mime_type=image_mime, data=img_bytes)))
        except Exception:
            pass
    contents = [_types.Content(role="user", parts=user_parts)]
    try:
        cfg = _types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"], temperature=0.2)
    except Exception:
        cfg = None

    for model in candidates[:2]:
        try:
            res = client.models.generate_content(model=model, contents=contents, config=cfg) if cfg else client.models.generate_content(model=model, contents=contents)
            images = []
            all_parts = getattr(res, "parts", None) or []
            if not all_parts:
                for cand in (getattr(res,"candidates",None) or []):
                    all_parts.extend(getattr(getattr(cand,"content",None),"parts",None) or [])
            for part in all_parts:
                blob = getattr(part,"inline_data",None)
                if blob and (getattr(blob,"mime_type","") or "").startswith("image/"):
                    d = getattr(blob,"data",None)
                    if isinstance(d, str):
                        try: d = _b64.b64decode(d)
                        except: d = None
                    elif isinstance(d,(bytes,bytearray)):
                        d = bytes(d)
                    else: d = None
                    if d:
                        images.append({"mime_type": getattr(blob,"mime_type","image/png"), "data": _b64.b64encode(d).decode()})
            if images:
                return ("画像を生成しました。", images)
        except Exception as e:
            continue
    return ("画像生成に失敗しました。モデルが利用できない可能性があります。", [])


# ── テーブル操作 ──────────────────────────────────────────────
import re as _re
import pandas as _pd
import io as _io

def _table_command(text: str) -> dict:
    """
    テキストからテーブル操作コマンドを実行。
    返り値: {"type": "table"|"text", "content": str, "csv": str|None, "columns": list, "rows": list}
    """
    pass  # 後でtable endpoint側で処理


class TableRequest(BaseModel):
    command: str
    csv_data: Optional[str] = None  # 現在のCSVデータ（base64 or raw）

class TableResponse(BaseModel):
    message: str
    csv: Optional[str] = None  # 結果CSV（raw）
    columns: list = []
    rows: list = []
    has_chart: bool = False
    numeric_cols: list = []

@router.post("/table_command")
def table_command(req: TableRequest, payload: dict = Depends(verify_token)):
    """テーブル操作コマンド処理"""
    cmd = (req.command or "").strip()
    csv_raw = req.csv_data or ""

    # CSVをDataFrameに変換
    df = None
    if csv_raw:
        try:
            df = _pd.read_csv(_io.StringIO(csv_raw))
        except Exception:
            df = None

    # コマンド判定
    if cmd.startswith("/rank ") or cmd.startswith("/sort "):
        parts = cmd.split(None, 3)
        col = parts[1] if len(parts) > 1 else ""
        order = (parts[2] if len(parts) > 2 else "desc").lower()
        asc = order in ("asc","昇順","小さい順")
        if df is not None and col:
            matched = next((c for c in df.columns if col in str(c)), None)
            if matched:
                try:
                    df2 = df.copy()
                    df2[matched] = _pd.to_numeric(df2[matched], errors="coerce")
                    df2 = df2.sort_values(matched, ascending=asc).reset_index(drop=True)
                    return _df_to_response(df2, f"**{matched}** {'昇順' if asc else '降順'}でソートしました")
                except Exception as e:
                    return TableResponse(message=f"ソートエラー: {e}")
        return TableResponse(message=f"列 '{col}' が見つかりません")

    elif cmd.startswith("/filter "):
        expr = cmd[8:].strip()
        if df is not None:
            try:
                op_map = {"以上":">=","以下":"<=","超":">","未満":"<"}
                for jp, en in op_map.items():
                    expr = expr.replace(jp, en)
                m = _re.match(r"(.+?)\s*(>=|<=|>|<|==|!=)\s*(.+)", expr)
                if m:
                    col, op, val = m.group(1).strip(), m.group(2), m.group(3).strip()
                    matched = next((c for c in df.columns if col in str(c)), None)
                    if matched:
                        s = _pd.to_numeric(df[matched], errors="coerce")
                        try: val_n = float(val)
                        except: val_n = None
                        if val_n is not None:
                            mask = eval(f"s {op} val_n", {"s":s,"val_n":val_n})
                            df2 = df[mask].reset_index(drop=True)
                            return _df_to_response(df2, f"{matched} {op} {val_n} で {len(df2)}件 抽出")
            except Exception as e:
                return TableResponse(message=f"フィルターエラー: {e}")
        return TableResponse(message="フィルター条件を解析できませんでした")

    elif cmd.startswith("/derive ") or cmd.startswith("/calc "):
        expr = cmd.split(None,1)[1].strip() if " " in cmd else ""
        if df is not None and "=" in expr:
            eq = expr.index("=")
            new_col = expr[:eq].strip()
            formula = expr[eq+1:].strip()
            try:
                df2 = df.copy()
                local_v = {}
                for c in df2.columns:
                    local_v[str(c).replace(" ","_")] = _pd.to_numeric(df2[c], errors="coerce")
                safe_formula = formula
                for c in sorted(df2.columns, key=lambda x:-len(str(x))):
                    safe_formula = safe_formula.replace(str(c), str(c).replace(" ","_"))
                df2[new_col] = eval(safe_formula, {"__builtins__":{}}, local_v).round(4)
                return _df_to_response(df2, f"派生列 **{new_col}** を追加しました")
            except Exception as e:
                return TableResponse(message=f"計算エラー: {e}\n使用可能な列: {', '.join(df.columns)}")
        return TableResponse(message="形式: /derive 新列名=式　例: /derive 客単価=売上/客数")

    elif cmd.startswith("/top "):
        parts = cmd.split(None, 3)
        if len(parts) >= 3 and df is not None:
            try:
                n = int(parts[1])
                col = parts[2]
                matched = next((c for c in df.columns if col in str(c)), None)
                if matched:
                    df2 = df.copy()
                    df2[matched] = _pd.to_numeric(df2[matched], errors="coerce")
                    df2 = df2.nlargest(n, matched).reset_index(drop=True)
                    return _df_to_response(df2, f"**{matched}** 上位{n}件")
            except Exception as e:
                return TableResponse(message=f"エラー: {e}")
        return TableResponse(message="形式: /top N 列名")

    elif "/consult" in cmd or "/analyze" in cmd:
        if df is not None:
            msg = _consult_analysis(df)
            return TableResponse(message=msg)
        return TableResponse(message="表データがありません")

    elif "/reset" in cmd or "/clear" in cmd:
        return TableResponse(message="テーブルをリセットしました", csv=None)

    else:
        return TableResponse(message=f"不明なコマンド: {cmd}\n使用可能: /rank, /filter, /derive, /top, /consult")


def _df_to_response(df: "_pd.DataFrame", message: str) -> "TableResponse":
    cols = list(df.columns)
    rows = df.values.tolist()
    csv = df.to_csv(index=False)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns]
    return TableResponse(message=message, csv=csv, columns=cols, rows=rows, has_chart=len(numeric_cols)>0, numeric_cols=numeric_cols)


def _consult_analysis(df: "_pd.DataFrame") -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        for c in df.columns:
            conv = _pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= len(df)*0.5:
                df = df.copy(); df[c] = conv
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return "数値列が見つかりません"
    lines = ["## 📊 数値コンサル分析\n"]
    lines.append("### 基本統計")
    for col in numeric_cols[:6]:
        s = df[col].dropna()
        if s.empty: continue
        lines.append(f"- **{col}**: 合計={s.sum():,.1f} / 平均={s.mean():,.1f} / 最小={s.min():,.1f} / 最大={s.max():,.1f}")
    lines.append("\n### 異常値検知（±2σ）")
    for col in numeric_cols[:6]:
        s = _pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4: continue
        mean, std = s.mean(), s.std()
        if std == 0: continue
        outliers = s[(s-mean).abs() > 2*std]
        if not outliers.empty:
            for idx, val in outliers.items():
                z = (val-mean)/std
                lines.append(f"- ⚠️ **{col}** 行{idx}: {val:.1f}（{z:+.1f}σ）")
    lines.append("\n💬 次: `/rank 列名 desc` / `/filter 列名 >= 値` / `/derive 新列=式`")
    return "\n".join(lines)


class FeedbackRequest(BaseModel):
    chat_id: str
    message: str
    reply: str
    label: str

@router.post("/feedback")
def save_feedback(req: FeedbackRequest, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", "default")
    db = get_db()
    from google.cloud import firestore as _fs
    db.collection("chat_feedback").add({
        "uid": uid,
        "tenant_id": tenant_id,
        "chat_id": req.chat_id,
        "message": req.message[:200],
        "reply": req.reply[:200],
        "label": req.label,
        "created_at": _fs.SERVER_TIMESTAMP,
    })
    return {"ok": True}


# ── 画像生成判定 ──────────────────────────────────────────────
_IMAGE_WORDS = ["画像","イメージ","イラスト","ロゴ","アイコン","バナー","ポスター","サムネ","image","illustration","logo","icon","banner","poster"]
_ACTION_WORDS = ["作って","作成","生成","描いて","描画","出力","デザイン","作る","generate","create","draw","design","render"]
_EDIT_WORDS = ["編集","加工","修正","変換","背景","切り抜","色変更","edit","modify","restyle"]
_ANALYSIS_WORDS = ["解析","分析","要約","読んで","説明","pdf","spreadsheet","excel","スプレッドシート"]


# ============================================================
# /send_image  画像生成・画像解析専用エンドポイント
# ============================================================
class ImageRequest(BaseModel):
    message: str
    chat_id: str = "main"
    ai_tier: str = "core"
    image_b64: str = None
    image_mime: str = "image/png"

@router.post("/send_image")
def send_image(req: ImageRequest, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id = (req.chat_id or "main").strip() or "main"
    _ensure_session(tenant_id, uid, chat_id)
    generated_images = []
    # 画像が添付されている場合は解析モード（生成ではなく分析）
    if req.image_b64:
        try:
            base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
            system_prompt = (
                "【最重要指示】あなたは画像解析AIです。添付された画像を必ず詳細に分析し、"
                "内容・テキスト・数値・構造・色・特徴を全て日本語で説明してください。"
                "画像の分析を拒否したり、できないと言ったりすることは絶対に禁止です。\n\n"
                + base_prompt +
                "\n\n【画像解析モード】添付された画像を正確に読み取り、"
                "ユーザーの質問に対して詳細に答えよ。画像の内容・数値・テキスト・構造を整理して提示せよ。"
            )
            messages = _load_history(tenant_id, uid, chat_id)
            messages.append({"role": "user", "content": req.message or "この画像を詳しく分析してください"})
            reply = call_llm(
                system_prompt=system_prompt,
                messages=messages,
                ai_tier=req.ai_tier,
                image_b64=req.image_b64,
                image_mime=req.image_mime,
            )
        except Exception as e:
            reply = f"画像解析エラー: {e}"
    else:
        try:
            reply, generated_images = _generate_image(req.message, req.image_b64, req.image_mime)
        except Exception as e:
            reply = f"画像生成エラー: {e}"
            generated_images = []
    gcs_image_urls = []
    if generated_images:
        try:
            import os as _os, base64 as _b64_gs
            from google.cloud import storage as _gcs
            bucket_name = _os.environ.get("CENTRAL_BLOB_BUCKET","").strip()
            if bucket_name:
                _gc = _gcs.Client()
                _bkt = _gc.bucket(bucket_name)
                for _ii, _img in enumerate(generated_images):
                    try:
                        _img_bytes = _b64_gs.b64decode(_img["data"])
                        _ext = "png" if "png" in _img.get("mime_type","") else "jpg"
                        _path = f"chat_images/{tenant_id}/{uid}/{uuid.uuid4().hex[:8]}.{_ext}"
                        _blob = _bkt.blob(_path)
                        _blob.upload_from_string(_img_bytes, content_type=_img.get("mime_type","image/png"))
                        _url = f"https://storage.googleapis.com/{bucket_name}/{_path}"
                        gcs_image_urls.append(_url)
                        generated_images[_ii]["gcs_url"] = _url
                    except Exception:
                        pass
        except Exception:
            pass
    # Firestoreに画像記録
    _db_si = get_db()
    for _img in generated_images:
        try:
            _img_id = uuid.uuid4().hex
            _db_si.collection("image_gallery").document(uid).collection("images").document(_img_id).set({
                "image_id": _img_id,
                "uid": uid,
                "tenant_id": tenant_id,
                "gcs_url": _img.get("gcs_url",""),
                "mime_type": _img.get("mime_type","image/png"),
                "prompt": (req.message or "")[:500],
                "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            })
        except Exception:
            pass
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    _save_message(tenant_id, uid, chat_id, "assistant", reply, images=generated_images)
    # usage_log書き込み
    try:
        _ulog_db2 = get_db()
        _ulog_db2.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "is_admin_test": False})
    except Exception:
        pass
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=generated_images)


# ============================================================
# /send_file  ファイル解析専用エンドポイント
# ============================================================
class FileAnalysisRequest(BaseModel):
    message: str
    chat_id: str = "main"
    ai_tier: str = "core"
    file_text: str = ""
    filename: str = ""

@router.post("/send_file")
def send_file(req: FileAnalysisRequest, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id = (req.chat_id or "main").strip() or "main"
    _ensure_session(tenant_id, uid, chat_id)
    base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
    _consulting_core = """あなたは超一流の経営コンサルタントであり、データ分析の専門家である。
添付されたファイルの内容を必ず詳細に分析し、以下の観点でコンサルティング回答を提供せよ。

【分析必須項目】
1. データの構造・全体像を把握し簡潔に説明せよ
2. 数値・トレンド・異常値・パターンを発見し指摘せよ
3. 問題点・課題・改善余地を具体的に提示せよ
4. 次のアクション・改善策・予測を根拠と共に提示せよ

【禁止事項】
- ファイルと無関係な話題への言及
- 「できません」「対応しておりません」等の拒否
- 曖昧・抽象的な回答
- データを見ずに一般論だけで回答すること

【出力形式】
- 結論を最初に述べ、根拠をデータから示せ
- 数値は必ず引用し、比較・変化率・傾向を明示せよ
- 実務で即使える具体的な提言を出せ"""

    system_prompt = (
        _consulting_core
        + ("\n\n【業種別追加指示】\n" + base_prompt if base_prompt.strip() else "")
        + "\n\n【ファイル解析モード】添付ファイルの内容を正確に読み取り、ユーザーの質問に答えよ。数値・表・構造は必ず整理して提示せよ。"
    )
    file_ctx = f"\n\n【添付ファイル: {req.filename}】\n{req.file_text[:8000]}" if req.file_text else ""
    messages = _load_history(tenant_id, uid, chat_id)
    messages.append({"role": "user", "content": req.message + file_ctx})
    try:
        reply = call_llm(
            system_prompt=system_prompt,
            messages=messages,
            ai_tier=req.ai_tier,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル解析エラー: {e}")
    # 構造化カード生成（投資専用フォーマット）
    _inv_structured = None
    try:
        import json as _jsi, re as _rei
        _inv_sp = (
            "次のJSONキー構造のみで回答せよ。前置き禁止。コードブロック禁止。\n"
            "必須キー: summary, cards, analysis, actions, value_message\n"
            "cards必須キー: current, risk, plan\n"
            "current: 注目銘柄・シグナル情報を5件（銘柄コード・社名・スコア・終値・前日比を含む）\n"
            "risk: 投資リスク・注意銘柄・市場リスクを5件\n"
            "plan: 具体的な投資アクション・エントリー戦略・出口戦略を5件\n"
            "analysis必須キー: type, urgency, importance, mode\n"
            "typeは必ず '投資シグナル分析'\n"
            "urgency/importanceは必ず '高'/'中'/'低'\n"
            "modeは必ず FINANCE\n"
            "summary: シグナル全体の相場見解を2〜3行で具体的に記述\n"
            "value_message: 今回の分析の要点を1行で\n"
            f"【投資シグナルデータ】{invest_ctx[:1200]}\n"
            f"【ユーザーの問い】{req.message[:300]}\n"
            f"【AI回答要約】{reply[:600]}\n"
        )
        _inv_sr = call_llm(
            system_prompt="JSONのみ出力。指定キー構造厳守。",
            messages=[{"role":"user","content":_inv_sp}],
            ai_tier="core", max_tokens=700
        )
        _inv_m = _rei.search(r'\{.*\}', _inv_sr, _rei.DOTALL)
        if _inv_m:
            _inv_parsed = _jsi.loads(_inv_m.group(0))
            if all(k in _inv_parsed for k in ["summary","cards","analysis","actions","value_message"]):
                _inv_structured = _inv_parsed
    except Exception:
        pass
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    _save_message(tenant_id, uid, chat_id, "assistant", reply, structured=_inv_structured)
    # usage_log書き込み
    try:
        _ulog_db3 = get_db()
        _ulog_db3.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "is_admin_test": False})
    except Exception:
        pass
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=[], structured=_inv_structured)


# ============================================================
# /send_invest  投資アルゴリズム専用エンドポイント
# ============================================================
class InvestRequest(BaseModel):
    message: str
    chat_id: str = "main"
    ai_tier: str = "core"

@router.post("/send_invest")
def send_invest(req: InvestRequest, payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id = (req.chat_id or "main").strip() or "main"
    _ensure_session(tenant_id, uid, chat_id)

    # 投資シグナル取得
    invest_ctx = ""
    try:
        db = get_db()
        import google.cloud.firestore as _fsm
        sig_docs = list(
            db.collection("investment_signals")
            .order_by("asof_date", direction=_fsm.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        if sig_docs:
            sig = sig_docs[0].to_dict() or {}
            asof = sig.get("asof_date", "不明")
            # GOAL_BOTTOM銘柄データ取得（上位500件）
            goal_docs = list(sig_docs[0].reference.collection("goal_bottom").limit(500).stream())
            goal_all = [gd.to_dict() or {} for gd in goal_docs]
            # WATCH_BIG_SELL銘柄データ取得（上位500件）
            sell_docs = list(sig_docs[0].reference.collection("watch_big_sell").limit(500).stream())
            sell_all = [sd.to_dict() or {} for sd in sell_docs]
            # ユーザーメッセージ内の銘柄を検索
            matched = [r for r in goal_all + sell_all
                       if str(r.get("code", "")) in req.message
                       or str(r.get("company_name", "")) in req.message]
            if matched:
                matched_lines = []
                for r in matched[:5]:
                    matched_lines.append(
                        f"  [{r.get('code','')}]{r.get('company_name','')} "
                        f"終値:{r.get('close','')} 前日比:{r.get('chg_pct','')}% "
                        f"底打ちスコア:{r.get('bottom_score','')} 売りスコア:{r.get('sell_score','')} "
                        f"反発確率(1-2日):{r.get('rebound_1_2d','')} 売り継続日数:{r.get('sell_days','')}"
                    )
                stock_detail = "\n■ 該当銘柄データ:\n" + "\n".join(matched_lines)
            else:
                stock_detail = (
                    "\n■ 【最優先指示・全ルール上書き】質問された銘柄は最新シグナルデータに存在しません。"
                    "この場合に限り「わかりません禁止」ルールは適用しない。"
                    "株価予測・テクニカル分析（MACD・RSI・移動平均・ボリンジャー等）は一切行わず、"
                    "「当該銘柄（社名/コード）はシグナルデータに存在しないため分析不可」とのみ回答し、"
                    "代わりにGOAL_BOTTOM上位銘柄を提示せよ。架空の数値・指標の生成は絶対禁止。"
                )
            goal_stocks = [
                f"  [{r.get('code','')}]{r.get('company_name','')} "
                f"底打ちスコア:{r.get('bottom_score','')} 終値:{r.get('close','')} 前日比:{r.get('chg_pct','')}%"
                for r in goal_all[:10]
            ]
            sell_stocks = [
                f"  [{r.get('code','')}]{r.get('company_name','')} "
                f"売りスコア:{r.get('sell_score','')} 終値:{r.get('close','')} 前日比:{r.get('chg_pct','')}%"
                for r in sell_all[:10]
            ]
            invest_ctx = (
                f"\n\n【最新投資シグナル（基準日: {asof}）】\n"
                f"■ GOAL_BOTTOM（底打ち反発候補）上位10件:\n" +
                "\n".join(goal_stocks or ["データなし"]) +
                f"\n\n■ WATCH_BIG_SELL（大口売り監視）上位10件:\n" +
                "\n".join(sell_stocks or ["データなし"]) +
                stock_detail +
                "\n\n【厳守】上記シグナルデータに存在する数値のみ根拠として使用せよ。"
                "MACD・RSI・移動平均などデータに存在しない指標の推測・捏造は絶対禁止。"
            )
    except Exception:
        pass

    system_prompt = (
        _load_tenant_system_prompt(tenant_id) +
        "\n\n【投資アルゴリズムモード】投資・相場・銘柄に特化した分析を行え。"
        "シグナルデータに存在する数値のみ根拠とし、データにない指標を推測・捏造することは絶対禁止。" +
        invest_ctx
    )
    messages = _load_history(tenant_id, uid, chat_id)
    messages.append({"role": "user", "content": req.message})
    try:
        reply = call_llm(
            system_prompt=system_prompt,
            messages=messages,
            ai_tier=req.ai_tier,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"投資解析エラー: {e}")
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    _save_message(tenant_id, uid, chat_id, "assistant", reply)
    # usage_log書き込み
    try:
        _ulog_db3 = get_db()
        _ulog_db3.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "is_admin_test": False})
    except Exception:
        pass
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=[])


@router.get("/images")
def get_image_gallery(payload: dict = Depends(verify_token)):
    """生成画像ギャラリー一覧取得"""
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    try:
        docs = list(
            db.collection("image_gallery").document(uid).collection("images")
            .limit(100).stream()
        )
        images = [d.to_dict() for d in docs]
        images.sort(key=lambda x: str(x.get("created_at","")), reverse=True)
        return {"images": images}
    except Exception as e:
        return {"images": [], "error": str(e)}


@router.delete("/images/{image_id}")
def delete_image(image_id: str, payload: dict = Depends(verify_token)):
    """生成画像を削除"""
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    db = get_db()
    try:
        # Firestoreから削除
        db.collection("image_gallery").document(uid).collection("images").document(image_id).delete()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# DEBUG REMOVE LATER
