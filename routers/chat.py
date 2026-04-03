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
- 出力形式：結論→打ち手→優先順位→リスク→次の観測。
- ナレッジは一次情報として優先し、ナレッジ記載事項は一般原理より優先せよ。
- 投資・相場予測の質問には投資シグナルデータを最優先で参照し、具体的な見解を必ず提示せよ。
- 【構造化出力ルール】比較・分類・優先順位・KPI・施策一覧を含む回答は必ずMarkdown表で出力せよ。
- 【表フォーマット】表のセパレーター行は必ず | --- | --- | 形式のみ使用せよ。それ以外の形式（|---|、|:---|等）は禁止。"""

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
                return {
                    "struct": [w.strip() for w in d.get("struct_words","").split(",") if w.strip()],
                    "strategy": [w.strip() for w in d.get("strategy_words","").split(",") if w.strip()],
                    "exec": [w.strip() for w in d.get("exec_words","").split(",") if w.strip()],
                    "emotion": [w.strip() for w in d.get("emotion_words","").split(",") if w.strip()],
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
        # use_count_since_report も +1
        use_count = int(d.get("use_count_since_report", 0)) + 1
        db.collection("users").document(uid).set({
            "level_score": new_score,
            "level": rank,
            "level_last_delta": delta,
            "use_count_since_report": use_count,
            "level_last_updated_at": fs.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception:
        pass

def _save_message(tenant_id: str, uid: str, chat_id: str, role: str, content: str, cases: list = None, structured: dict = None):
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
        result.append(msg)
    return result

def _load_tenant_system_prompt(tenant_id: str) -> str:
    try:
        db = get_db()
        # app.py の sys_col() = db.collection("tenant_settings")
        doc = db.collection("tenant_settings").document(tenant_id).get()
        if doc.exists:
            sp = (doc.to_dict() or {}).get("system_prompt", "")
            if sp:
                return sp
    except Exception:
        pass
    return DEFAULT_SYSTEM_PROMPT

def _build_system_with_rag(tenant_id: str, query: str, system_prompt: str) -> str:
    try:
        chunks = rag_retrieve_chunks(tenant_id=tenant_id, query=query, top_k=5)
        if chunks:
            rag_text = "\n\n---\n\n".join(
                f"【ナレッジ: {c.get('title', '')}】\n{c.get('text', '')}"
                for c in chunks
            )
            return f"{system_prompt}\n\n【参照ナレッジ】\n{rag_text}"
    except Exception:
        pass
    return system_prompt

# ── エンドポイント ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = "main"
    ai_tier: str = "core"
    purpose_mode: str = "auto"

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

    base_prompt   = _load_tenant_system_prompt(tenant_id)

    # 脳内カルテ更新（非同期的に実行、失敗しても続行）
    intent_state = {}
    try:
        intent_state = update_user_intent_state(uid, tenant_id, history, req.message)
    except Exception:
        pass

    # QueryPlan生成
    query_plan = {}
    try:
        query_plan = generate_query_plan(req.message, tenant_id, "mixed")
    except Exception:
        pass

    # SummaryLens選択
    try:
        lens_preset, lens_hier = lgbm_select_summary_lens(req.message, "auto")
    except Exception:
        lens_preset, lens_hier = "expert", "raw"

    # 脳内カルテをsystem_promptに注入
    intent_ctx = ""
    if intent_state:
        intent_ctx = f"""\n\n【ユーザーの脳内カルテ（深層プロファイル）】
・ステージ: {intent_state.get('current_stage','')}
・真の渇望: {intent_state.get('true_desire','')}
・バイアス: {intent_state.get('bias','')}
・不足観点: {intent_state.get('missing_piece','')}
※上記を踏まえ、単なる回答ではなく「格を上げるための介入」を行え。"""

    system_prompt = _build_system_with_rag(tenant_id, req.message, base_prompt) + intent_ctx

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
    if _mode_key in _MODE_INSTRUCTIONS:
        system_prompt = _MODE_INSTRUCTIONS[_mode_key] + "\n\n" + system_prompt

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

    # 構造化データ生成（戦略相談時のみ・画像生成は除外）
    structured = None
    if not generated_images and len(req.message.strip()) > 5:
        try:
            import json as _json_s, re as _re_s
            _sp = (
                "次のJSONキー構造のみで回答せよ。他のキー名・構造は絶対禁止。前置き禁止。コードブロック禁止。\n"
                "必須キー: summary, cards, analysis, actions, value_message\n"
                "cards必須キー: current, risk, plan (各3件の文字列配列)\n"
                "analysis必須キー: type, urgency, importance, mode\n"
                "urgency/importanceは必ず '高'/'中'/'低' のいずれか\n"
                "modeは必ず STRATEGY/NUMERIC/CONTROL/RISK/MARKETING/GROWTH/DIAGNOSIS/PLANNING/FINANCE/HR/CREATIVE/NEGOTIATION/AUTO のいずれか\n"
                f"\n相談: {req.message[:300]}\n"
                f"回答要約: {reply[:600]}\n"
                "\n出力例(この形式厳守):\n"
                '{"summary":"結論を1〜2行で","cards":{"current":["現状1","現状2","現状3"],"risk":["リスク1","リスク2","リスク3"],"plan":["方針1","方針2","方針3"]},"analysis":{"type":"意思決定整理","urgency":"中","importance":"高","mode":"STRATEGY"},"actions":["アクション1","アクション2","アクション3"],"value_message":"価値を1行で"}'
            )
            _sr = call_llm(
                system_prompt="JSONのみ出力。指定キー構造厳守。前置き・後置き・コードブロック完全禁止。余計なキー追加禁止。",
                messages=[{"role": "user", "content": _sp}],
                ai_tier="core", max_tokens=700
            )
            _m = _re_s.search(r'\{.*\}', _sr, _re_s.DOTALL)
            if _m:
                _parsed = _json_s.loads(_m.group(0))
                # 必須キー検証
                if all(k in _parsed for k in ["summary","cards","analysis","actions","value_message"]):
                    _cards = _parsed.get("cards", {})
                    _analysis = _parsed.get("analysis", {})
                    if all(k in _cards for k in ["current","risk","plan"]) and all(k in _analysis for k in ["type","urgency","importance","mode"]):
                        structured = _parsed
        except Exception as _se:
            structured = None
            print(f"[STRUCTURED_ERROR] {type(_se).__name__}: {_se}", flush=True)

    # レベルスコア加算
    _delta = _calc_score(req.message, tenant_id)
    _update_level_score(tenant_id, uid, _delta)

    # RAGチャンク採用記録（LGBM教師データ）
    try:
        chunks = rag_retrieve_chunks(tenant_id=tenant_id, query=req.message, top_k=5)
        if chunks:
            db = get_db()
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
    _save_message(tenant_id, uid, chat_id, "assistant", reply, cases=cases, structured=structured)

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
            base_prompt = _load_tenant_system_prompt(tenant_id)
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
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    _save_message(tenant_id, uid, chat_id, "assistant", reply)
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
    base_prompt = _load_tenant_system_prompt(tenant_id)
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
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    _save_message(tenant_id, uid, chat_id, "assistant", reply)
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=[])


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
        sig_docs = list(
            db.collection("investment_signals")
            .order_by("asof_date", direction=__import__("google.cloud.firestore", fromlist=["firestore"]).firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        if sig_docs:
            sig = sig_docs[0].to_dict() or {}
            asof = sig.get("asof_date", "不明")
            goal = sig.get("goal_count", 0)
            watch = sig.get("watch_count", 0)
            invest_ctx = (
                f"\n\n【最新投資シグナル（基準日: {asof}）】\n"
                f"GOAL_BOTTOM候補: {goal}件 / WATCH_BIG_SELL監視: {watch}件\n"
                "上記シグナルデータを最優先で参照し、具体的な投資見解を提示せよ。"
            )
    except Exception:
        pass

    system_prompt = (
        _load_tenant_system_prompt(tenant_id) +
        "\n\n【投資アルゴリズムモード】投資・相場・銘柄に特化した分析を行え。"
        "予測・見通しは必ず具体的数値と根拠を示せ。「わかりません」は禁止。" +
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
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=[])
