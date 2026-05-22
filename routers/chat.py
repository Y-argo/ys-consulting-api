# api/routers/chat.py
import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json as _json_sse
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import firestore as fs

from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT
from api.core.llm_client import call_llm
from api.core.rag import rag_retrieve_chunks
from api.core.intent import update_user_intent_state, generate_query_plan, lgbm_select_summary_lens

router = APIRouter(prefix="/api/chat", tags=["chat"])

def _write_notification(uid: str, notif_type: str, title: str, body: str, link_tab: str = "overview"):
    """Firestoreに通知レコードを書き込む共通関数"""
    try:
        import uuid as _uuid
        import datetime as _dt
        db = get_db()
        # ユーザーの通知設定を確認
        snap = db.collection("users").document(uid).get()
        d = snap.to_dict() if snap.exists else {}
        settings = d.get("notification_settings", {})
        type_key_map = {
            "reply": "notify_reply",
            "rankup": "notify_rankup",
            "fc": "notify_fc",
            "inquiry": "notify_inquiry",
            "priority_action": "notify_reply",
        }
        setting_key = type_key_map.get(notif_type, "notify_reply")
        # デフォルトTrue（設定がなければ通知する）
        if settings.get(setting_key, True) is False:
            return
        notif_id = _uuid.uuid4().hex
        db.collection("notifications").document(uid).collection("items").document(notif_id).set({
            "notif_id": notif_id,
            "uid": uid,
            "type": notif_type,
            "title": title,
            "body": body,
            "link_tab": link_tab,
            "read": False,
            "created_at": (_dt.datetime.utcnow() + _dt.timedelta(hours=9)).isoformat(),
        })
    except Exception as e:
        print(f"[NOTIFICATION_ERROR] {e}", flush=True)

def _clean_reply(text: str) -> str:
    """Markdownテーブルのセパレーター正規化・スペース圧縮・英語instruction除去"""
    import re as _re
    # 英語instructionハルシネーション除去
    _en_kws = ["identify the issue as a question", "Okay, I understand. Please provide",
               "I will then:", "Identify the main issue", "Answer the question based on",
               "framing it as a question"]
    lines = text.split("\n")
    text = "\n".join(l for l in lines if not any(k.lower() in l.lower() for k in _en_kws))
    text = _re.sub(r" {2,}", " ", text)
    lines = text.split("\n"); out = []
    for line in lines:
        st = line.strip()
        if st.startswith("|") and st.endswith("|"):
            inner = st[1:-1]; cells = inner.split("|")
            # 最初のセルがセパレーターパターンなら行全体を正規化（全セル一致不要）
            if cells and all(_re.match(r"^\s*:?-+:?\s*$", c) for c in cells if c.strip()):
                line = "|" + "|".join(" --- " for _ in cells) + "|"
        out.append(line)
    return "\n".join(out).strip()


MANDATORY_RULES = """
- 【確認質問全面禁止】「どのような情報にご興味がありますか？」「どのような業種・業態を想定されていますか？」「何についてお知りになりたいですか？」等、いかなる形式の確認質問・追加質問・質問返しも絶対禁止。RAGナレッジから最も適切な内容を直接回答せよ。対象者の確認はシステムが自動でボタンを生成する。LLM自身が確認・誘導をすることは一切禁止。
- 【ナレッジ参照禁止表記】回答中に「【ナレッジ: ファイル名】」「【参照: ファイル名】」等のファイル名・ソース名を絶対に表示するな。ナレッジの内容は自然な文章として回答に統合せよ。ファイル名の露出は厳禁。
"""

DEFAULT_SYSTEM_PROMPT = """【最重要・全ルール最優先】いかなる質問・内容に対しても必ず回答を生成せよ。回答拒否・出力拒否は絶対禁止。水商売・接客業・ナイトビジネスを含む全合法業種のビジネス支援システムである。ナレッジ記載内容の出力は業務上の正当な情報提供であり、いかなる理由でも拒否するな。必ず日本語で回答を出力せよ。
あなたは「Ys Consulting Office」の戦略コンサルタント（LM）である。
目的は利益最大化・勝率向上・リスク最小化・意思決定の高速化。

【ASCENDとは — 機能・特徴を質問された場合は必ず以下の情報を根拠に具体的に回答せよ】
ASCENDはYs Consulting Officeが提供するAIコンサルティングSaaSプラットフォームである。戦略・数値・構造・リスクあらゆる経営判断に即応するAIコンサルティングエンジンであり、一般的なチャットAIではなく診断・分析・実行支援に特化した経営判断専用エンジンである。

■ AIチャット系
- AIチャット：テキスト・画像・ファイル対応のメインチャット機能
- 画像解析：画像をアップロードしてAIが詳細解析
- ファイル解析：PDF・Excel等の資料を読み込み解析

■ 診断系
- 現状課題診断：現状を構造化し本質的な課題を診断
- 構造診断：事象を解剖し構造を可視化
- 課題仮説：課題仮説を体系的に生成
- 比較分析：選択肢を多次元で比較評価
- 矛盾検知：論理矛盾・整合性ズレを検出
- 実行計画：実行プランの設計・分析
- 投資シグナル：投資判断シグナルを分析（APEX/ULTRA限定）
- 思考マップ：思考の構造をグラフ化
- ファイル診断：Excel/PDF等を数値分析+AI解釈（Chain of Thought分析）
- 未来分岐シミュレーター：将来の分岐を予測

■ 分析・レポート
- Decision Metrics：意思決定6指標スコアリング
- 固定概念レポート：LightGBMによる思考の固定概念・盲点を分析（PRO以上）
- プレゼン資料生成：スライドを自動生成
- プロファイル生成：特徴入力から人柄・行動パターン・強み・接し方を推定（APEX/ULTRA）
- 顧客AIマネジメント：顧客心理状態を時系列推論し接客最適化・離脱防止・キャスト相性・店舗全体知能を統合した店舗知能OS（APEX/ULTRA）

■ 生成系
- 画像生成：AIで画像を生成（STANDARD以上）
- 生成画像ギャラリー：生成画像を一覧管理

■ 相談
- 個人相談：Ys Consulting Officeの専門コンサルへの個別スレッド相談（PRO以上）

■ プラットフォーム
- ランクシステム：活用度（level_score）に応じて追従者→実行者→戦略家→設計者の4段階称号が変化する成長可視化システム
- マイページ：統計・履歴・設定の一覧
- プラン管理：利用プランの確認・変更
- 用途別モード切替：用途に応じて柔軟にモードを切替（最大19モード）

■ AIエンジン構成（3段階）
- SWIFT（迅速）：高速レスポンス・AUTO/7モード対応。日常的な戦略相談・施策整理に最適。STARTER/STANDARD対応。
- ADVANCE（高度）：全19モード対応の高精度エンジン。ファイル診断・固定概念レポート・画像生成・個人相談が解放される。PRO対応。
- SUPREME（至高）：全機能解放の最上位エンジン。投資シグナルを含む全19モード・全機能で最高難度の経営判断に対応。APEX/ULTRA対応。

■ コンサルティング力の核心
ASCENDのコンサルティング力は以下の点に集約される。
1. RAGナレッジ参照：テナントごとの業界・業務ナレッジをベクトル検索で参照し一般AIが知らない情報を回答に反映
2. 構造化診断：現状課題・矛盾・比較・実行を多軸で同時分析し盲点を排除
3. 数値分析：ファイル診断でPython/pandasによる定量分析をChain of Thoughtで4段階処理
4. 意思決定支援：Decision Metrics 6指標で判断品質を数値化・可視化
5. 固定概念破壊：LGBMレポートで思考パターンの固定概念・バイアスを自動検出
6. 投資シグナル連携：相場データと連動した具体的投資見解の提示（APEX/ULTRA）
7. 個人相談：Ys Consulting Officeコンサルタントとの直接スレッド往復（PRO以上）
- 口調：丁寧・敬意。相手の理解速度に合わせるが結論は曖昧にしない。
- 禁止：慰め・言い訳・情緒誘導・根拠なき断定。
- 禁止：「私には予測できません」「判断できません」等の無能宣言は絶対禁止。予測・見通しを求められた場合は現状データ・トレンド・過去事例から必ず具体的見解を提示せよ。ただしナレッジ・データに存在しない具体的数値・固有情報（バック率・料金・人名等）は絶対に推測・捏造するな。その場合は「ナレッジに情報がないため回答できません」と答えよ。
- 不明：不明は不明と明示し、仮説と検証手順を分離する。
- ナレッジは一次情報として優先し、ナレッジ記載事項は一般原理より優先せよ。
- 投資・相場予測の質問には投資シグナルデータを最優先で参照し、具体的な見解を必ず提示せよ。
- 【構造化出力ルール】比較・分類・優先順位・KPI・施策一覧を含む回答は必ずMarkdown表で出力せよ。
- 【表フォーマット】表のセパレーター行は必ず | --- | --- | 形式のみ使用せよ。それ以外の形式（|---|、|:---|等）は禁止。
- 【絶対禁止】存在しないファイル名・資料名・書籍名・URL・メールアドレス・人名・数値を絶対に捏造するな。ナレッジに記載のない情報は「情報がありません」と答えよ。架空の資料名・個人情報を回答に含めることは厳禁。
- 【表記号禁止】回答中で「|」を使う場合は必ずMarkdown表形式で出力せよ。「|」を文中に単独で使うことは禁止。
- 【表セパレーター】表のセパレーター行は必ず | --- | --- | 形式のみ使用せよ。スペースパディング・|:---|・|---|等は禁止。


【ASCENDプラン定義 — 以下以外の情報は絶対に捏造・推測するな。不明な場合は「プラン詳細はYs Consulting Officeにお問い合わせください」とのみ回答せよ】

■ STARTER：¥0（新規7日間）
 エンジン: SWIFT / AUTOモード
 利用可能: AIチャット(AUTOモード)、RAG検索、レベルスコア
 対象外: 診断機能全般、画像生成、ファイル診断、固定概念レポート、個人相談、投資シグナル

■ STANDARD：¥9,800/月
 エンジン: SWIFT / 7モード対応
 利用可能: AIチャット(7モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ(構造/課題/比較/矛盾/実行)、画像生成、画像・ファイル解析(チャット内)
 対象外: ファイル診断(Ultraエンジン)、固定概念レポート、個人相談、投資シグナル、ASCEND ADVANCE/SUPREME

■ PRO：¥39,800/月
 エンジン: ADVANCE / 全19モード対応
 利用可能: AIチャット(全19モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ全6種、ファイル診断(Chain of Thought分析)、固定概念レポート(LGBM自動生成)、画像生成、画像ギャラリー、個人相談(スレッド往復)、ASCEND ADVANCE解放
 対象外: 投資シグナル、ASCEND SUPREME

■ APEX：¥89,800/月
 エンジン: SUPREME / 全19モード対応
 利用可能: 全機能解放、AIチャット(全19モード)、ファイル診断、固定概念レポート、投資シグナル(全銘柄)、ASCEND SUPREME(最上位AIエンジン)、個人相談、画像生成、ギャラリー、診断タブ全8種(投資シグナルタブ含む)

■ ULTRA：¥300,000/月〜（要相談・顧問契約）
 エンジン: SUPREME / 全19モード対応
 利用可能: ASCEND全機能完全解放、Ys Consulting Office顧問契約付き、社員10名まで個別アカウント発行、企業テナント共有(RAG・診断履歴)、月次戦術レポート提出、新機能先行利用、月次ミーティング・直接支援
 契約・問い合わせ: Ys Consulting Officeに直接連絡（UID記載必須）
- 【言語】出力は必ず日本語のみ。英語・ロシア語・その他外国語の文字・単語の混入は絶対禁止。英語のメタ指示文（identify the issue / framing it as a question / I will then / Please provide等）を出力することは絶対禁止。"""

# ── app.py と同一の Firestore パス ────────────────────────
# chat_sessions/{scope}__{tenant_id}__{uid}__{chat_id}/messages/{msg_id}
SCOPE = "user"

def _session_doc_id(tenant_id: str, uid: str, chat_id: str = "main") -> str:
    return f"{SCOPE}__{tenant_id}__{uid}__{chat_id}"

def _messages_ref(tenant_id: str, uid: str, chat_id: str = "main"):
    db = get_db()
    doc_id = _session_doc_id(tenant_id, uid, chat_id)
    return db.collection("chat_sessions").document(doc_id).collection("messages")

def _ensure_session(tenant_id: str, uid: str, chat_id: str = "main", title: str = "", force_create: bool = False):
    from datetime import datetime
    db = get_db()
    doc_id = _session_doc_id(tenant_id, uid, chat_id)
    ref = db.collection("chat_sessions").document(doc_id)
    if force_create:
        now = datetime.utcnow()
        doc = {"scope": SCOPE, "tenant_id": tenant_id, "uid": uid, "chat_id": chat_id, "updated_at": now, "created_at": now, "is_deleted": False}
        if title:
            doc["title"] = title
        ref.set(doc)
    else:
        doc = {"scope": SCOPE, "tenant_id": tenant_id, "uid": uid, "chat_id": chat_id, "updated_at": fs.SERVER_TIMESTAMP, "created_at": fs.SERVER_TIMESTAMP, "is_deleted": False}
        if title:
            doc["title"] = title
        ref.set(doc, merge=True)

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
        # ランクアップ検出→通知
        old_rank = d.get("level", "")
        if old_rank and rank != old_rank:
            _write_notification(
                uid=uid,
                notif_type="rankup",
                title="🏆 ランクアップしました！",
                body=f"{old_rank} → {rank} に昇格しました。",
                link_tab="rankup",
            )
    except Exception:
        pass

def _save_message(tenant_id: str, uid: str, chat_id: str, role: str, content: str, cases: list = None, structured: dict = None, images: list = None, sources: list = None):
    if not content or not content.strip():
        return
    # base64画像データをcontentから除去してから保存
    if "__IMAGE_B64__" in content:
        content = content[:content.index("__IMAGE_B64__")].rstrip()
    if not content:
        return
    # スペースパディング除去（セル内連続スペースを1つに圧縮）
    import re as _re
    content = _re.sub(r" {2,}", " ", content)
    content = content.strip()
    if not content:
        return

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
    if sources is not None:
        doc["sources"] = sources
    ref.add(doc)

def _load_history(tenant_id: str, uid: str, chat_id: str = "main", limit: int = 20) -> List[dict]:
    ref = _messages_ref(tenant_id, uid, chat_id)
    docs = ref.order_by("ts").limit_to_last(limit).get()
    result = []
    for d in docs:
        data = d.to_dict() or {}
        content = data.get("content", "")
        if not content or not content.strip():
            continue
        # 巨大スペースパディングメッセージを除外（正常な回答は5000文字以内）
        if len(content) > 5000:
            content = content[:5000]
        msg = {"role": data.get("role", "user"), "content": content}
        if data.get("cases"):
            msg["cases"] = data["cases"]
        if data.get("structured"):
            msg["structured"] = data["structured"]
        if data.get("images"):
            msg["images"] = data["images"]
        if data.get("sources"):
            msg["sources"] = data["sources"]
        result.append(msg)
    return result

PLAN_DEFINITION = """

【ASCENDプラン定義 — 以下以外の情報は絶対に捏造・推測するな。不明な場合は「プラン詳細はYs Consulting Officeにお問い合わせください」とのみ回答せよ】

■ STARTER：¥0（新規7日間）
 エンジン: SWIFT / AUTOモード
 利用可能: AIチャット(AUTOモード)、RAG検索、レベルスコア
 対象外: 診断機能全般、画像生成、ファイル診断、固定概念レポート、個人相談、投資シグナル

■ STANDARD：¥9,800/月
 エンジン: SWIFT / 7モード対応
 利用可能: AIチャット(7モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ(構造/課題/比較/矛盾/実行)、画像生成、画像・ファイル解析(チャット内)
 対象外: ファイル診断(Ultraエンジン)、固定概念レポート、個人相談、投資シグナル、ASCEND ADVANCE/SUPREME

■ PRO：¥39,800/月
 エンジン: ADVANCE / 全19モード対応
 利用可能: AIチャット(全19モード)、RAG検索、レベルスコア、現状課題診断、Decision Metrics、診断タブ全6種、ファイル診断(Chain of Thought分析)、固定概念レポート(LGBM自動生成)、画像生成、画像ギャラリー、個人相談(スレッド往復)、ASCEND ADVANCE解放
 対象外: 投資シグナル、ASCEND SUPREME

■ APEX：¥89,800/月
 エンジン: SUPREME / 全19モード対応
 利用可能: 全機能解放、AIチャット(全19モード)、ファイル診断、固定概念レポート、投資シグナル(全銘柄)、ASCEND SUPREME(最上位AIエンジン)、個人相談、画像生成、ギャラリー、診断タブ全8種(投資シグナルタブ含む)

■ ULTRA：¥300,000/月〜（要相談・顧問契約）
 エンジン: SUPREME / 全19モード対応
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
            plan = (u.get("plan") or "user").strip()
            custom = (u.get("custom_sys_prompt") or "").strip()
            mode = (u.get("custom_prompt_mode") or "append").strip()
            member_extra = (u.get("member_extra_prompt") or "").strip()
            # ultra_member: use_admin_settings=Trueの時のみadmin設定をベースに追加
            if plan == "ultra_member" and tenant_id and u.get("use_admin_settings", False):
                try:
                    _aid = _get_admin_uid(tenant_id)
                    admin_custom = ""
                    admin_mode = "append"
                    if _aid:
                        try:
                            _a_snap = db.collection("users").document(_aid).get()
                            if _a_snap.exists:
                                _a_data = _a_snap.to_dict() or {}
                                admin_custom = (_a_data.get("custom_sys_prompt") or "").strip()
                                admin_mode = (_a_data.get("custom_prompt_mode") or "append").strip()
                        except Exception:
                            pass
                    if admin_custom:
                        if admin_mode == "replace":
                            tenant_prompt = admin_custom
                        else:
                            tenant_prompt = tenant_prompt + "\n\n" + admin_custom
                except Exception:
                    pass
                # メンバー独自追加指示を追記
                if member_extra:
                    tenant_prompt = tenant_prompt + "\n\n" + member_extra
                return tenant_prompt
            if custom:
                if mode == "replace":
                    return custom + "\n\n" + MANDATORY_RULES
                else:
                    return tenant_prompt + "\n\n" + custom + "\n\n" + MANDATORY_RULES
    except Exception:
        pass
    return tenant_prompt + "\n\n" + MANDATORY_RULES

# ── admin_uid キャッシュ（インメモリTTL=1時間 + tenant_settings永続化） ──
_admin_uid_cache: dict = {}  # {tenant_id: (admin_uid, timestamp)}
_ADMIN_UID_TTL = 3600  # 1時間

def _get_admin_uid(tenant_id: str) -> str:
    """優先順位: 1.インメモリキャッシュ(TTL内=Firestore不要) 2.tenant_settings(1doc) 3.初回スキャン→保存"""
    if not tenant_id:
        return ""
    import time as _t
    now = _t.time()
    if tenant_id in _admin_uid_cache:
        _uid, _ts = _admin_uid_cache[tenant_id]
        if now - _ts < _ADMIN_UID_TTL:
            return _uid
    try:
        db = get_db()
        ts_snap = db.collection("tenant_settings").document(tenant_id).get()
        if ts_snap.exists:
            cached = (ts_snap.to_dict() or {}).get("admin_uid", "")
            if cached:
                _admin_uid_cache[tenant_id] = (cached, now)
                return cached
        ads = list(db.collection("users").where("tenant_id", "==", tenant_id).limit(50).stream())
        for ad in ads:
            if (ad.to_dict() or {}).get("plan") == "ultra_admin":
                db.collection("tenant_settings").document(tenant_id).set(
                    {"admin_uid": ad.id}, merge=True
                )
                _admin_uid_cache[tenant_id] = (ad.id, now)
                return ad.id
        # ultra_admin未発見もTTL内キャッシュ（再スキャン防止）
        _admin_uid_cache[tenant_id] = ("", now)
    except Exception:
        pass
    return ""

def _build_system_with_rag(tenant_id: str, query: str, system_prompt: str, uid: str = "", admin_uid: str = "", is_apex_ultra: bool = False):
    """returns (prompt_str, chunks_list)"""
    try:
        import re as _re_skip
        _ascend_self_patterns = [
            r"ASCEND.*とは",
            r"ASCENDの.*機能",
            r"ASCENDの.*使い方",
            r"ASCEND.*使い方",
            r"ASCEND.*何ができる",
            r"ASCEND.*できること",
            r"ASCEND.*料金",
            r"ASCEND.*プラン",
            r"ASCEND.*思想",
            r"ASCEND.*哲学",
            r"ASCEND.*実力",
            r"ASCEND.*特徴",
            r"ASCENDの.*コンサル",
            r"ASCENDの.*力",
            r"ASCENDに.*できる",
            r"ASCENDで.*できる",
            r"ASCENDの.*強み",
            r"ASCENDとは",
            r"ASCENDって",
            r"ASCEND.*何",
            r"コンサル力",
            r"このシステム.*機能",
            r"このシステム.*使い方",
            r"このAI.*できる",
            r"このAI.*何ができる",
            r"使い方",
            r"機能.*教えて",
            r"システム.*説明",
            r"どう使う",
            r"何ができますか",
            r"何ができる",
            r"プラン.*教えて",
            r"料金.*教えて",
            r".*とは？",
            r".*とは?",
            r".*の機能",
            r".*の使い方",
            r".*できること",
            r".*何ですか",
            r".*説明して",
            r".*教えてください",
        ]
        if any(_re_skip.search(p, query) for p in _ascend_self_patterns):
            _about_instruction = """
【ASCEND自己説明モード】
ユーザーはASCEND自身の使い方・機能・思想・実力・料金・できることを質問している。
この場合、外部RAGや一般論に逃げず、system_prompt内のASCEND説明情報を最優先根拠として回答せよ。

【必須回答方針】
- 「分かりません」「該当情報なし」「専門外です」と答えることは禁止。
- ASCENDの全体像、主要機能、AIエンジン、診断、ファイル診断、画像、投資シグナル、個人相談、プラン差を必要に応じて具体的に説明する。
- 単なる機能一覧ではなく、ASCENDの思想・哲学・実力・支援領域まで示す。
- 質問が「使い方」の場合は、ユーザーが次に何を入力すればよいかまで案内する。
- 質問が短くても、ASCENDの価値が伝わるように具体的に答える。

【ASCEND説明時の禁止事項】
- 幼稚な口調禁止
- キャラクターAI口調禁止（「〜だよ」「相棒」「寄り添う」等）
- 過剰な感情表現禁止
- 絵文字禁止
- 汎用自己啓発AI化禁止
- 恋愛洗脳・依存誘導として説明すること禁止

【ASCEND説明スタイル】
ASCENDは「経営判断OS」「構造解析エンジン」「戦略実行支援基盤」として説明せよ。
説明時は以下を優先:
1. 何を分析するか
2. 何を構造化するか
3. 何を意思決定支援するか
4. どのAIエンジンが関係するか
5. 実務上どう使うか
6. 数値・構造・戦略にどう寄与するか

【顧客AIマネジメント説明ルール】
顧客AIマネジメントは以下として説明すること:
- 顧客心理状態の時系列推論
- 離脱リスク検知
- 接客最適化
- 顧客タイプ分析
- キャスト相性分析
- 店舗全体知能OS
- CRM高度化
- リピート維持
- 接客品質改善
- 顧客行動予測
禁止: 嫉妬操作・執着誘導・洗脳的表現・恋愛誘導テクニック中心の説明

【全機能詳細説明モード】
ユーザーが「全機能」「機能一覧」「何ができる」「詳しく説明」と聞いた場合、以下のカテゴリ順で各機能を省略せず説明せよ。

1. AIチャット系
- AIチャット: テキスト相談、経営判断、戦略整理、課題分解、施策立案を支援するメインチャット。用途別モードにより相談・分析・実行・数値・構造などの観点を切り替える。
- 画像解析: 画像をアップロードし、内容・構造・文字・配置・デザイン・改善点を解析する。
- ファイル解析: PDF、Excel、CSV、テキスト等を読み込み、資料内容の要約・論点整理・数値確認・改善提案を行う。

2. 診断系
- 現状課題診断: 現状を入力し、表面的な悩みではなく本質課題・原因・優先順位を診断する。
- 構造診断: 事象を要素分解し、原因・関係性・支配構造を可視化する。
- 課題仮説: 現状情報から複数の課題仮説を生成し、検証すべき論点を整理する。
- 比較分析: 複数案を多面的に比較し、メリット・リスク・実行難度・優先度を評価する。
- 矛盾検知: 計画・発言・方針・数値の中にある論理矛盾や整合性のズレを検出する。
- 実行計画: 目標に対して具体的な手順・優先順位・実行ロードマップを設計する。
- 投資シグナル: 市場・銘柄・条件をもとに投資判断の補助シグナルを分析する。
- 思考マップ: 思考・課題・選択肢・因果関係をマップ化し、見落としを減らす。
- ファイル診断: Excel/PDF等を数値分析とAI解釈で深掘りし、資料全体の構造・異常・改善点を診断する。
- 未来分岐シミュレーター: 現在の選択肢から複数の未来シナリオを予測し、リスクと打ち手を整理する。

3. 分析・レポート
- Decision Metrics: 意思決定を複数指標でスコアリングし、判断の質・リスク・実行可能性を可視化する。
- 固定概念レポート: ユーザーの思考傾向や判断の癖を分析し、固定概念・盲点・改善視点を提示する。
- プレゼン資料生成: 入力内容をもとに、提案・報告・戦略説明用のスライド構成を生成する。
- プロファイル生成: 特徴入力から、人柄・行動傾向・強み・接し方を推定する。
- 顧客AIマネジメント: 顧客心理状態の時系列推論、接客最適化、離脱防止、キャスト相性分析、CRM高度化、店舗全体知能化を支援する店舗知能OS。

4. 生成系
- 画像生成: 指示文から画像を生成し、企画・広告・世界観設計・ビジュアル案作成を支援する。
- 生成画像ギャラリー: 生成した画像を一覧管理し、再確認・活用できる。

5. 相談
- 個人相談: 通常チャットより深い個別相談を行い、課題・意思決定・方針設計を支援する。

6. プラットフォーム
- ランクシステム: 活用度に応じてlevel_scoreが上がり、追従者→実行者→戦略家→設計者へ称号が変化する。
- マイページ: 統計、履歴、設定、レポート確認を行う。
- プラン管理: 利用中プランと解放機能を確認する。
- 用途別モード切替: 相談目的に応じてAIの回答視点を切り替える。

7. AIエンジン
- SWIFT: 高速応答向け。日常相談・基本整理・簡易分析に適する。
- ADVANCE: 高精度エンジン。全19モード、ファイル診断、固定概念レポート、画像生成、個人相談などに対応。
- SUPREME: 最上位エンジン。全機能解放、投資シグナル、最高難度の経営判断に対応。

【回答ルール】
全機能説明では各機能について必ず以下を含める:
- 機能概要 / 入力するもの / 出力されるもの / 実務での使い道 / 経営判断上の価値
禁止: 機能名だけの羅列・汎用AI説明・system_promptに存在しない機能の創作・恋愛誘導・心理操作・依存形成への逸脱・絵文字・幼稚な口調・相棒口調
"""
            return system_prompt + "\n\n" + _about_instruction, [], True
        from api.core.rag import embed_text, rag_retrieve_chunks_with_vec
        # ユーザーのRAG設定をFirestoreから取得
        _rag_threshold = 0.42
        _rag_top_k = 5
        _cfg_uid = admin_uid if admin_uid else uid
        if _cfg_uid:
            try:
                _usnap = get_db().collection("users").document(_cfg_uid).get()
                _rag_cfg = ((_usnap.to_dict() or {}) if _usnap.exists else {}).get("rag_settings") or {}
                _rag_threshold = float(_rag_cfg.get("threshold", 0.42))
                _rag_top_k = int(_rag_cfg.get("top_k", 5))
            except Exception:
                pass
        # embedding は1回だけ計算して使い回す
        try:
            _query_vec = embed_text(query)
        except Exception:
            return system_prompt, [], False
        chunks = rag_retrieve_chunks_with_vec(tenant_id=tenant_id, query_vec=_query_vec, top_k=_rag_top_k, threshold=_rag_threshold)
        if admin_uid:
            try:
                admin_chunks = rag_retrieve_chunks_with_vec(tenant_id=f"user__{admin_uid}", query_vec=_query_vec, top_k=5, threshold=_rag_threshold)
                existing_ids = {c.get("chunk_id") for c in chunks}
                chunks = chunks + [c for c in admin_chunks if c.get("chunk_id") not in existing_ids]
            except Exception:
                pass
        if uid and is_apex_ultra:
            try:
                print(f"[USER_RAG] uid={uid} tenant=user__{uid}", flush=True)
                user_chunks = rag_retrieve_chunks_with_vec(tenant_id=f"user__{uid}", query_vec=_query_vec, top_k=_rag_top_k, threshold=_rag_threshold)
                print(f"[USER_RAG] user_chunks={len(user_chunks)}", flush=True)
                for _uc in user_chunks: print(f"[USER_RAG_SRC] {_uc.get('source_id','')} score={_uc.get('_score',0):.3f}", flush=True)
                existing_ids = {c.get("chunk_id") for c in user_chunks}
                chunks = user_chunks + [c for c in chunks if c.get("chunk_id") not in existing_ids]
            except Exception as _ue:
                print(f"[USER_RAG ERROR] {_ue}", flush=True)
        # APEX/ULTRA専用ナレッジが空の場合 → 中央倉庫(default)にフォールバック
        if is_apex_ultra and not chunks:
            try:
                central_chunks = rag_retrieve_chunks_with_vec(tenant_id="default", query_vec=_query_vec, top_k=_rag_top_k, threshold=_rag_threshold)
                if central_chunks:
                    chunks = central_chunks
            except Exception:
                pass
        if chunks:
            rag_text = "\n\n---\n\n".join(
                f"【参考情報】\n{c.get('text', '')[:1500]}"
                for c in chunks[:10]
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
                r"流れ", r"ステップ", r"プロセス", r"順番", r"どのように",
                r"どんな", r"どうすれば", r"どうしたら", r"どう", r"何を",
                r"何が", r"なぜ", r"なに", r"仕方", r"方針",
                r"ルール", r"規則", r"基準", r"マニュアル", r"ガイド",
                r"おすすめ", r"お勧め", r"推奨", r"ベスト", r"最適",
                r"注意", r"気をつける", r"NG", r"禁止", r"必要",
                r"大切", r"重要", r"秘訣",
                r"具体的", r"詳しく", r"もっと", r"教えてください",
                r"接客", r"対応", r"案内", r"説明", r"紹介",
                r"準備", r"確認", r"チェック", r"管理", r"運営",
                r"できる", r"できますか", r"方法は", r"するべき",
            ]
            _is_knowledge_query = any(
                _re_qt.search(p, query) for p in _knowledge_patterns
            )
            if _is_knowledge_query:
                return (
                    f"【知識回答モード】以下の複数の参照ナレッジを統合して質問に答えよ。"
                    f"各ナレッジの具体的な内容・事例・表現を活かし、抽象的・一般論的な回答は禁止。"
                    f"キャラクターの口調・絵文字を維持しながら説得力ある具体的な回答をせよ。問いかけ返し・感情確認禁止。"
                    f"【用語解釈厳守】ナレッジに記載された用語・定義・固有名詞はそのままの意味で使用せよ。独自解釈・拡張解釈は絶対禁止。"
                    f"【出力形式】比較・チェックリスト・一覧・評価項目を含む場合は必ずMarkdown表で出力せよ。表の各セルは1行・50文字以内で簡潔に記述せよ。セル内に改行・番号付きリスト・長文・URLを絶対に入れるな。"
                    f"Markdown表のセパレーター行は | --- | --- | 形式のみ（:や=や-の4つ以上連続は禁止）。"
                    f"重要ポイントは**太字**で強調せよ。\n\n"
                    f"【参照ナレッジ({len(chunks)}件)】\n{rag_text}\n\n{system_prompt}"
                ), chunks, False
            else:
                _max_score = max((c.get("_score", 0) for c in chunks), default=0)
                if _max_score >= 0.70:
                    return (
                        f"【ナレッジ参照モード】以下の参照ナレッジに記載された情報を最優先で使用して質問に答えよ。"
                        f"ナレッジに該当情報がある場合はキャラクターの口調を維持しながら必ずその内容を回答に反映せよ。"
                        f"ナレッジに記載のない情報は独自に作らず'確認できませんでした'と答えよ。\n\n"
                        f"【参照ナレッジ({len(chunks)}件)】\n{rag_text}\n\n{system_prompt}"
                    ), chunks, False
                else:
                    return f"{system_prompt}\n\n【参考情報】\n{rag_text}", chunks, False
    except Exception:
        pass
    return system_prompt, [], False

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
    intent: Optional[dict] = None
    intent_label: Optional[str] = None
    sources: Optional[list] = None
    confirmation_choices: Optional[list] = None

class SessionInfo(BaseModel):
    chat_id: str
    title: str
    updated_at: Optional[str] = None

@router.post("/send", response_model=ChatResponse)
def send_message(req: ChatRequest, payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    from api.core.features import is_feature_enabled as _isfe_tier
    if req.ai_tier in ("ultra", "apex"):
        _tier_feat = "ascend_ultra" if req.ai_tier == "ultra" else "ascend_apex"
        if not _isfe_tier(uid, _tier_feat):
            raise HTTPException(status_code=403, detail=f"このAIエンジン（{req.ai_tier}）は現在未開放のため使用できません。")
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id   = (req.chat_id or "main").strip() or "main"

    _ensure_session(tenant_id, uid, chat_id)

    history = _load_history(tenant_id, uid, chat_id)
    messages = history + [{"role": "user", "content": req.message}]

    base_prompt   = _load_tenant_system_prompt(tenant_id, uid=uid)
    # ── ASCEND静的辞書ヒット検出（推奨モード用） ──
    _ascend_dict_key, _ascend_dict_text = _ascend_static_answer(req.message)
    _ascend_dict_hit = bool(_ascend_dict_key and _ascend_dict_text)
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
    _plan_thread = None
    _plan_result = {}
    if not _is_custom_replace:
        def _run_intent():
            try:
                _intent_result["state"] = update_user_intent_state(uid, tenant_id, history, req.message)
            except Exception:
                pass
        def _run_plan():
            try:
                _plan_result["plan"] = generate_query_plan(req.message, tenant_id, "mixed")
            except Exception:
                pass
        _intent_thread = _threading.Thread(target=_run_intent, daemon=True)
        _plan_thread = _threading.Thread(target=_run_plan, daemon=True)
        _intent_thread.start()
        _plan_thread.start()

    # QueryPlan・SummaryLens（スレッド結果を回収）
    query_plan = {}
    lens_preset, lens_hier = "expert", "raw"

    # 脳内カルテ・QueryPlan両スレッドをまとめてjoin
    if _intent_thread: _intent_thread.join(timeout=3.0)
    if _plan_thread: _plan_thread.join(timeout=3.0)
    if not _is_custom_replace:
        query_plan = _plan_result.get("plan", {})
        try:
            lens_preset, lens_hier = lgbm_select_summary_lens(req.message, "auto")
        except Exception:
            lens_preset, lens_hier = "expert", "raw"
        if query_plan.get("summary_lens", {}).get("preset"):
            lens_preset = query_plan["summary_lens"]["preset"]
    intent_state = _intent_result.get("state", {}) if not _is_custom_replace else {}
    intent_ctx = ""
    if intent_state and not _is_custom_replace:
        intent_ctx = f"""\n\n【ユーザーの脳内カルテ（深層プロファイル）】
・ステージ: {intent_state.get('current_stage','')}
・真の渇望: {intent_state.get('true_desire','')}
・バイアス: {intent_state.get('bias','')}
・不足観点: {intent_state.get('missing_piece','')}
※上記を踏まえ、単なる回答ではなく「格を上げるための介入」を行え。"""

    # ultra_member+use_admin_settings=True時に管理者のuidを取得
    _admin_uid = ""
    try:
        _u_data = get_db().collection("users").document(uid).get().to_dict() or {}
        if _u_data.get("plan") == "ultra_member" and _u_data.get("use_admin_settings", False):
            _admin_docs = list(get_db().collection("users").where("tenant_id","==",tenant_id).limit(20).stream())
            for _ad in _admin_docs:
                _ad_data = _ad.to_dict() or {}
                if _ad_data.get("plan") == "ultra_admin":
                    _admin_uid = _ad.id
                    break
    except Exception:
        pass
    _user_plan = (_u_data.get("plan") or "user").strip()
    _is_apex_ultra = _user_plan in ("apex", "ultra_admin", "ultra_member")
    if not is_talk:
        # ASCENDモード or 辞書ヒット時: 辞書内容をsystem_promptへ注入してLLMに渡す
        _base_for_rag = base_prompt
        _mode_is_ascend = (req.purpose_mode or "").strip().lower() == "ascend"
        if _ascend_dict_hit or _mode_is_ascend:
            _dict_content = _ascend_dict_text if _ascend_dict_text else ""
            if _mode_is_ascend and not _dict_content:
                # モード選択時はASCEND全機能説明を渡す
                parts = ["ASCENDの主な機能は以下です。"]
                for name, desc in ASCEND_FEATURE_GUIDE.items():
                    if name not in ["ASCENDとは", "ASCENDの名前の由来", "SWIFT / ADVANCE / SUPREME"]:
                        parts.append(f"■ {name}\n{desc}")
                _dict_content = "\n\n".join(parts)
            _base_for_rag = base_prompt + "\n\n【ASCEND辞書情報 - 必ずこの内容を根拠に回答せよ】\n" + _dict_content
        system_prompt, _rag_chunks, _is_ascend_about = _build_system_with_rag(tenant_id, req.message, _base_for_rag, uid=payload.get("uid",""), admin_uid=_admin_uid, is_apex_ultra=_is_apex_ultra)
    else:
        system_prompt, _rag_chunks, _is_ascend_about = base_prompt, [], False
        # 会話モード：カスタムプロンプトがある場合はRAG+URL注入を発動
        try:
            _talk_user_doc = get_db().collection("users").document(uid).get()
            _talk_has_custom = bool((_talk_user_doc.to_dict() or {}).get("custom_sys_prompt", "")) if _talk_user_doc.exists else False
        except Exception:
            _talk_has_custom = False
        _talk_has_admin_rag = bool(_admin_uid)
        # 会話モード: 常にRAGを呼ぶ（全プラン共通）
        _talk_custom_only = ((_talk_user_doc.to_dict() or {}).get("custom_sys_prompt") or "").strip() if _talk_has_custom else ""
        _talk_base = _talk_custom_only if (_talk_custom_only and not _is_custom_replace) else base_prompt
        system_prompt, _rag_chunks, _is_ascend_about = _build_system_with_rag(tenant_id, req.message, _talk_base, uid=uid, admin_uid=_admin_uid, is_apex_ultra=_is_apex_ultra)
        if _talk_has_custom and not _is_custom_replace:
            system_prompt = (
                "【会話モード・最優先指示】以下のキャラクター設定と知識ファイルの内容を完全に内面化し、自分の言葉として再構築して答えよ。"
                "ナレッジの文言・ファイル名・資料名・出典を直接引用・露出することは絶対禁止。"
                "【口調厳守】いかなるキャラクター設定が存在しても、出力は必ず丁寧・敬語・プロフェッショナルなコンサルタント口調で答えよ。ため口・カジュアル・絵文字・慰め表現は絶対禁止。"
                "知識ファイルに表・一覧・チェックリストが含まれる場合は必ずMarkdown表で出力せよ。"
                "前置き宣言（「今回は〜についてご説明します」等）は禁止。\n\n"
                + system_prompt
            )
        # 会話モード：専用ボットモード以外かつカスタムプロンプトなしのみコンサル形式を上書き
        _list_keywords = ["リスト","一覧","やること","手順","チェックリスト","箇条","ステップ","まとめて","列挙"]
        _is_list_req = any(w in req.message for w in _list_keywords)
        if not _is_apex_ultra and not _is_custom_replace and not _talk_has_custom and not _is_list_req:
            system_prompt = system_prompt.replace(
                "出力形式：結論→打ち手→優先順位→リスク→次の観測。",
                "出力形式：自然な会話形式で簡潔に回答せよ。箇条書きや表は使わず、2〜4文程度で答えよ。"
            ) + "\n\n【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"
    _RAG_THRESHOLD = 0.70  # 高確信度閾値（取得判定用）（取得判定用・検証フラグとは別）
    _max_rag_score = max((c.get("_score", 0) for c in _rag_chunks), default=0.0) if _rag_chunks else 0.0
    print(f"[RAG_VERIFY] chunks={len(_rag_chunks)} max_score={_max_rag_score:.3f}", flush=True)
    system_prompt = system_prompt + intent_ctx
    for _i, _c in enumerate(_rag_chunks[:5]):
        _txt = str(_c.get("text") or _c.get("content") or _c)
    if history:
        pass
    if _rag_chunks:
        system_prompt += '\n\nナレッジに記載のない情報を生成することは絶対禁止。参照ナレッジに記載がない場合は「該当情報なし」と回答せよ。'

    # ── ASCENDプラン情報条件注入 ─────────────────────────────────────
    _plan_kws = ["プラン","料金","ascend","サブスク","subscription","ultra","apex","pro","standard","starter","月額","契約"]
    try:
        _u_doc2 = get_db().collection("users").document(uid).get()
        _has_custom = bool((_u_doc2.to_dict() or {}).get("custom_sys_prompt", "")) if _u_doc2.exists else False
    except Exception:
        _has_custom = False
    _plan_hit_m = any(k in req.message.lower() for k in _plan_kws)
    if _plan_hit_m:
        system_prompt += PLAN_DEFINITION
    # RAG=0件 かつ プランキーワードなし → ⚠️フォールバック
    _no_rag_fallback = (not _rag_chunks) and (not _plan_hit_m) and (not _is_ascend_about)
    if _no_rag_fallback:
        system_prompt += '\n\n【⚠️ナレッジ未検証回答モード】専用ナレッジ・中央倉庫のいずれにも該当情報が存在しない。質問に対してLLMの一般知識で誠実に回答せよ。「回答できません」「専門外です」等の拒否は絶対禁止。回答の冒頭に必ず「⚠️ナレッジ未検証回答」と表示せよ。'

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
        "numeric":     "【NUMERICモード】数値・KPI・売上・コスト分析に特化せよ。必ず数値・比率・計算式を使って回答せよ。",
        "strategy":    "【STRATEGYモード】競合分析・差別化・ポジショニング戦略に特化せよ。3C/4P/SWOT等のフレームワークを活用し、戦略的選択肢と優先順位を提示せよ。",
        "control":     "【CONTROLモード】組織・権限・業務フロー・マネジメント構造に特化せよ。責任分担・権限設計・フロー最適化の観点から回答せよ。",
        "growth":      "【GROWTHモード】スキル・習慣・成長設計に特化せよ。具体的なトレーニング方法・習慣化ステップ・成長指標を提示せよ。",
        "analysis":    "【ANALYSISモード】データ・事象の多角的解析に特化せよ。因果関係・相関・パターンを分解し、複数の解釈仮説を提示せよ。",
        "planning":    "【PLANNINGモード】ロードマップ・フェーズ設計に特化せよ。時系列・マイルストーン・依存関係を明示したアクションプランを提示せよ。",
        "risk":        "【RISKモード】リスク特定・評価・対策設計に特化せよ。発生確率×影響度でリスクを評価し、回避・軽減・転嫁・受容の選択肢を提示せよ。",
        "marketing":   "【MARKETINGモード】集客・ブランディング・広告施策に特化せよ。ターゲット定義・チャネル選定・CVR改善の観点から具体施策を提示せよ。",
        "diagnosis":   "【DIAGNOSISモード】現状課題の発見・根本原因分析に特化せよ。なぜなぜ分析・ロジックツリー等で根本原因を特定し、表面的対処を避けよ。",
        "forecast":    "【FORECASTモード】将来予測・シナリオ分析に特化せよ。楽観・中立・悲観の3シナリオを定量的に提示し、各シナリオの発生条件を明示せよ。",
        "finance":     "【FINANCEモード】財務・投資・資金計画分析に特化せよ。ROI・回収期間・キャッシュフロー・損益分岐点を数値で示せ。",
        "hr":          "【HRモード】採用・評価・組織設計・人材育成に特化せよ。評価基準・採用要件・育成ステップを構造的に提示せよ。",
        "negotiation": "【NEGOTIATIONモード】交渉・説得・合意形成戦略に特化せよ。相手の利害・BATNAを分析し、Win-Winの合意シナリオと交渉戦術を提示せよ。",
        "creative":    "【CREATIVEモード】アイデア発想・コンセプト設計に特化せよ。既存の枠を超えた発想を複数提示し、実現可能性と独自性を評価せよ。",
        "summary":     "【SUMMARYモード】要約・整理に特化せよ。要点を3〜5項目に絞り、階層構造で簡潔に整理せよ。",
        "legal":       "【LEGALモード】法務・規約・コンプライアンスの解説に特化せよ。ただし法的助言ではなく情報提供として提示し、専門家確認を推奨せよ。",
        "coaching":    "【COACHINGモード】自己変革・思考パターン改善に特化せよ。質問・内省促進・気づきの提供を優先し、答えを与えるより考えさせる回答をせよ。",
        "ops":         "【OPSモード】業務改善・効率化・オペレーション最適化に特化せよ。ボトルネック特定・工数削減・標準化・自動化の観点から具体的改善策を提示せよ。",
        "tech":        "【TECHモード】技術・エンジニアリング・システム設計に特化せよ。技術的トレードオフ・アーキテクチャ選定・実装方針を構造的に提示せよ。",
    }
    _mk = (req.purpose_mode or "auto").strip().lower()
    # planning mode: ExecutionPlan を事前生成（reply依存禁止）
    execution_plan_obj = None
    if _mk == "planning":
        try:
            from api.core.intent import build_execution_plan as _bep
            execution_plan_obj = _bep(req.message, tenant_id, "mixed")
        except Exception as _ep_err:
            print(f"[EXEC_PLAN_ERROR] {_ep_err}", flush=True)
    # planning mode: execution_plan_obj を system_prompt に注入
    if _mk == "planning" and execution_plan_obj and isinstance(execution_plan_obj, dict):
        import json as _epj3
        _ep_json = _epj3.dumps(execution_plan_obj, ensure_ascii=False)
        system_prompt += (
            "\n\n【実行計画オブジェクト（必ずこの構造を根拠に説明せよ）】\n"
            + _ep_json[:2000]
            + "\n上記ExecutionPlanの各フェーズ・タスク・KPI・依存関係を根拠にして補足説明のみ行え。架空の数値・タスク追加は禁止。"
        )
    if _mk in _MODE_INSTRUCTIONS:
        system_prompt = _MODE_INSTRUCTIONS[_mk] + "\n\n" + system_prompt

    # 画像データ抽出
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
        _cases_prompt = f"以下の会話に対して、ユーザーが次に相談しそうな事案を3件、日本語で1行ずつ返せ。マーク不要。\nQ: {req.message}\nA: {reply[:500]}"
        _cases_raw = call_llm(
            system_prompt="次の相談候補を3件だけ出力せよ。余分なテキスト不要。",
            messages=[{"role": "user", "content": _cases_prompt}],
            ai_tier="core", max_tokens=256
        )
        cases = [l.strip() for l in _cases_raw.strip().split("\n") if l.strip()][:3]
    except Exception:
        cases = []

    # 構造化データ生成
    structured = None
    if not generated_images and len(req.message.strip()) > 5:
        # planning mode: execution_plan_obj から直接生成
        if _mk == "planning" and execution_plan_obj and isinstance(execution_plan_obj, dict):
            _tasks = execution_plan_obj.get("tasks", [])
            # fallback補完済みのため4項目チェック省略
            _plan_valid = True
            if _plan_valid:
                _ep_current = [f"[{t.get('phase','')}] {t.get('objective','')} / 期限:{t.get('due_days','')}日 / 担当:{t.get('owner','')} / KPI:{t.get('kpi','')}" for t in _tasks[:3]]
                _ep_graph = execution_plan_obj.get("graph", {})
                _ep_critical = execution_plan_obj.get("critical_path", [])
                structured = {
                    "summary": execution_plan_obj.get("summary", req.message[:60]),
                    "execution_plan": {
                        "phases": execution_plan_obj.get("phases", []),
                        "tasks": _tasks,
                        "graph": _ep_graph,
                        "critical_path": _ep_critical,
                        "dependencies": execution_plan_obj.get("dependencies", []),
                        "blockers": execution_plan_obj.get("blockers", []),
                        "kpis": execution_plan_obj.get("kpis", []),
                        "checkpoints": execution_plan_obj.get("checkpoints", []),
                        "risks": execution_plan_obj.get("risks", []),
                    },
                    "cards": {
                        "current": _ep_current if _ep_current else ["実行計画を生成しました"],
                        "risk": execution_plan_obj.get("risks", [])[:3] or ["リスク情報なし"],
                        "plan": [t.get("objective", "") for t in _tasks[:3]] or ["プラン情報なし"],
                    },
                    "analysis": {"type": "実行計画", "urgency": "高", "importance": "高", "mode": "PLANNING"},
                    "actions": [t.get("objective", "") for t in _tasks[:5] if t.get("objective")],
                    "value_message": f"実行計画: {len(_tasks)}タスク / critical_path:{len(_ep_critical)}ステップ",
                }
        else:
            try:
                import json as _json_s, re as _re_s
                _sp = (
                    "次のJSONキー構造のみで回答せよ。前置き禁止。コードブロック禁止。\n"
                    "必須キー: summary, cards, analysis, actions, value_message\n"
                    "cards必須キー: current, risk, plan (各3件の文字列配列)\n"
                    "analysis必須キー: type, urgency, importance, mode\n"
                    "urgency/importanceは必ず '高'/'中'/'低' のいずれか\n"
                    f"\n相談: {req.message[:300]}\n"
                    f"回答要約: {reply[:600]}\n"
                )
                _sr = call_llm(
                    system_prompt="JSONのみ出力。指定キー構造厳守。",
                    messages=[{"role": "user", "content": _sp}],
                    ai_tier="core", max_tokens=700
                )
                _m = _re_s.search(r'\{.*\}', _sr, _re_s.DOTALL)
                if _m:
                    _parsed = _json_s.loads(_m.group(0))
                    if all(k in _parsed for k in ["summary","cards","analysis","actions","value_message"]):
                        structured = _parsed
            except Exception as _se:
                structured = None

    # レベルスコア加算
    _delta = _calc_score(req.message, tenant_id)
    _update_level_score(tenant_id, uid, _delta)

    # sources
    _sources = [{"text": (c.get("text","") or "")[:200], "score": float(c.get("_score",0)), "source_id": str(c.get("source_id","")), "is_retrieved": True} for c in _rag_chunks] if _rag_chunks else []
    _confirmation_choices = []

    _save_message(tenant_id, uid, chat_id, "user", req.message)
    reply = __import__("re").sub(r" {2,}", " ", reply).strip()
    _save_message(tenant_id, uid, chat_id, "assistant", reply, cases=cases, structured=structured, sources=_sources)

    # usage_log
    try:
        import datetime as _dt
        get_db().collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": (_dt.datetime.utcnow() + _dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"), "is_admin_test": False, "purpose_mode": _mk})
    except Exception:
        pass

    # GCS画像保存
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
                        generated_images[_ii]["gcs_url"] = f"https://storage.googleapis.com/{bucket_name}/{_path}"
                    except Exception:
                        pass
        except Exception:
            pass

    _safe_images = [{"mime_type": img.get("mime_type","image/png"), "gcs_url": img.get("gcs_url","")} for img in generated_images]
    # DEBUG: ResponseValidationError原因調査
    try:
        from fastapi.encoders import jsonable_encoder as _jae
        import json as _jdebug
        _resp_debug = _jae(dict(
            reply=reply, chat_id="x", cases=cases,
            images=_safe_images, structured=structured,
            sources=_sources, confirmation_choices=_confirmation_choices,
            intent=intent_state if isinstance(intent_state, dict) else None,
            intent_label=query_plan.get("intent","") if isinstance(query_plan, dict) else None,
        ))
    except Exception as _dbe:
        pass
    return ChatResponse(
        reply=reply,
        chat_id=chat_id,
        msg_id=str(uuid.uuid4()),
        cases=cases,
        images=_safe_images,
        structured=structured,
        sources=_sources,
        confirmation_choices=_confirmation_choices,
        intent=intent_state if isinstance(intent_state, dict) else None,
        intent_label=query_plan.get("intent","") if isinstance(query_plan, dict) else None,
    )

# ==============================
# ASCEND 説明辞書
# ==============================
ASCEND_FEATURE_GUIDE = {
    "ASCENDとは": """ASCENDは、戦略・数値・構造・リスクを統合し、経営判断を支援するAIコンサルティングエンジンです。
診断・分析・実行支援に特化した経営判断プラットフォームであり、印象ではなく構造、直感ではなく数値を軸に、課題整理・意思決定・施策設計を支援します。""",
    "ASCENDの名前の由来": """ASCENDは、経営判断の解像度を上げ、事業・組織・戦略を次の段階へ上昇させる思想を表します。
A — Architectural Analysis（構造解剖）
S — Scoring & Scale（階級スコア）
C — Case-driven RAG（事例駆動検索）
E — Executor Strategy（戦術執行）
N — Nurturing / Mentor（育成・導師）
D — Dynamic Routing & Diagnosis（動的診断）""",
    "SWIFT": """SWIFTは、高速レスポンス向けのAIエンジンです。
日常的な戦略相談、施策整理、簡易分析、初期の論点整理に適しています。""",
    "ADVANCE": """ADVANCEは、高精度分析向けのAIエンジンです。
全19モードに対応し、ファイル診断、固定概念レポート、画像生成、個人相談などの高度機能が解放されます。""",
    "SUPREME": """SUPREMEは、ASCENDの最上位AIエンジンです。
全機能解放、投資シグナルを含む全19モード、最高難度の経営判断に対応します。""",
    "SWIFT / ADVANCE / SUPREME": """ASCENDのAIエンジンは3段階です。

SWIFT（迅速）: 高速レスポンス向け。日常的な戦略相談、施策整理、簡易分析に適しています。
ADVANCE（高度）: 全19モード対応の高精度エンジン。ファイル診断、固定概念レポート、画像生成、個人相談などが解放されます。
SUPREME（至高）: 全機能解放の最上位エンジン。投資シグナルを含む全19モードに対応し、最高難度の経営判断を支援します。""",
    "AIチャット": """AIチャットは、テキスト・画像・ファイルに対応するASCENDのメインチャット機能です。
経営相談、課題整理、施策立案、数値分析、構造分解などを対話形式で行います。""",
    "画像解析": """画像解析は、アップロードされた画像をAIが読み取り、内容・構造・文字・配置・デザイン・改善点を解析する機能です。""",
    "ファイル解析": """ファイル解析は、PDF・Excel・CSV・テキスト等を読み込み、内容の要約、論点整理、数値確認、改善提案を行う機能です。""",
    "現状課題診断": """現状課題診断は、入力された現状を構造化し、本質課題・原因・優先順位を診断する機能です。""",
    "構造診断": """構造診断は、事象を要素分解し、原因・関係性・支配構造を可視化する診断です。""",
    "課題仮説": """課題仮説は、現状情報から複数の課題仮説を生成し、検証すべき論点を整理する機能です。""",
    "比較分析": """比較分析は、複数の選択肢を多面的に比較評価する機能です。メリット、リスク、実行難度、優先度を整理し、選択判断を支援します。""",
    "矛盾検知": """矛盾検知は、計画・発言・方針・数値の中にある論理矛盾や整合性のズレを検出する機能です。""",
    "実行計画": """実行計画は、目標に対して具体的な手順・優先順位・ロードマップを設計する機能です。""",
    "投資シグナル": """投資シグナルは、市場・銘柄・条件をもとに投資判断の補助シグナルを分析する機能です。投資判断を保証するものではなく、判断材料を構造化するための分析支援機能です。""",
    "思考マップ": """思考マップは、思考・課題・選択肢・因果関係をマップ化し、見落としを減らす機能です。""",
    "ファイル診断": """ファイル診断は、Excel・PDF等の資料を数値分析とAI解釈で深掘りする診断機能です。資料全体の構造、異常、傾向、改善点を分析します。""",
    "未来分岐シミュレーター": """未来分岐シミュレーターは、現在の状況・選択肢・リスク・制約条件をもとに、将来起こり得る複数の分岐を予測する機能です。""",
    "Decision Metrics": """Decision Metricsは、意思決定を複数指標でスコアリングし、判断の質・リスク・実行可能性を可視化する機能です。""",
    "固定概念レポート": """固定概念レポートは、ユーザーの思考傾向や判断の癖を分析し、固定概念・盲点・改善視点を提示する機能です。""",
    "プレゼン資料生成": """プレゼン資料生成は、入力内容をもとに、提案・報告・戦略説明用のスライド構成を生成する機能です。""",
    "プロファイル生成": """プロファイル生成は、特徴入力をもとに、人柄・行動パターン・強み・接し方を推定する機能です。""",
    "顧客AIマネジメント": """顧客AIマネジメントは、顧客心理状態を時系列で推論し、接客最適化・離脱防止・担当者相性・店舗全体知能を統合する顧客管理知能OSです。
CRM高度化、リピート維持、顧客傾向分析、接客品質改善、顧客行動予測を支援します。
恋愛誘導、依存形成、嫉妬操作、執着誘導を目的とする機能ではありません。""",
    "画像生成": """画像生成は、指示文から画像を生成する機能です。企画、広告、世界観設計、ビジュアル案作成などに使用できます。
「画像生成とは？」は機能説明であり、実際に画像を生成する依頼とは区別されます。""",
    "生成画像ギャラリー": """生成画像ギャラリーは、生成した画像を一覧管理する機能です。過去に生成した画像を確認し、再利用や比較に使えます。""",
    "個人相談": """個人相談は、通常チャットより深い個別相談を行う機能です。課題、意思決定、方針設計、事業相談などを個別スレッドとして扱います。""",
    "ランクシステム": """ランクシステムは、活用度に応じてlevel_scoreが上がり、称号が変化する成長可視化システムです。称号は、追従者、実行者、戦略家、設計者の4段階です。""",
    "マイページ": """マイページは、統計、履歴、設定、レポート確認を行う場所です。過去の診断、固定概念レポート、利用履歴などを確認できます。""",
    "プラン管理": """プラン管理は、現在の利用プランと解放されている機能を確認する機能です。未開放機能は、プラン変更または管理者による権限付与で利用可能になります。""",
    "用途別モード切替": """用途別モード切替は、相談目的に応じてAIの回答視点を切り替える機能です。相談、分析、実行、数値、構造など、目的に合った出力へ調整します。""",
    "NUMERICモード": """NUMERICモードは、数値・KPI・売上・コスト分析に特化したモードです。数値・比率・計算式を使って定量的に回答します。""",
    "STRATEGYモード": """STRATEGYモードは、競合分析・差別化・ポジショニング戦略に特化したモードです。3C/4P/SWOT等のフレームワークを活用し、戦略的選択肢と優先順位を提示します。""",
    "CONTROLモード": """CONTROLモードは、組織・権限・業務フロー・マネジメント構造に特化したモードです。責任分担・権限設計・フロー最適化の観点から回答します。""",
    "GROWTHモード": """GROWTHモードは、スキル・習慣・成長設計に特化したモードです。具体的なトレーニング方法・習慣化ステップ・成長指標を提示します。""",
    "ANALYSISモード": """ANALYSISモードは、データ・事象の多角的解析に特化したモードです。因果関係・相関・パターンを分解し、複数の解釈仮説を提示します。""",
    "PLANNINGモード": """PLANNINGモードは、ロードマップ・フェーズ設計に特化したモードです。時系列・マイルストーン・依存関係を明示したアクションプランを提示します。""",
    "RISKモード": """RISKモードは、リスク特定・評価・対策設計に特化したモードです。発生確率×影響度でリスクを評価し、回避・軽減・転嫁・受容の選択肢を提示します。""",
    "MARKETINGモード": """MARKETINGモードは、集客・ブランディング・広告施策に特化したモードです。ターゲット定義・チャネル選定・CVR改善の観点から具体施策を提示します。""",
    "DIAGNOSISモード": """DIAGNOSISモードは、現状課題の発見・根本原因分析に特化したモードです。なぜなぜ分析・ロジックツリー等で根本原因を特定します。""",
    "FORECASTモード": """FORECASTモードは、将来予測・シナリオ分析に特化したモードです。楽観・中立・悲観の3シナリオを定量的に提示します。""",
    "FINANCEモード": """FINANCEモードは、財務・投資・資金計画分析に特化したモードです。ROI・回収期間・キャッシュフロー・損益分岐点を数値で示します。""",
    "HRモード": """HRモードは、採用・評価・組織設計・人材育成に特化したモードです。評価基準・採用要件・育成ステップを構造的に提示します。""",
    "NEGOTIATIONモード": """NEGOTIATIONモードは、交渉・説得・合意形成戦略に特化したモードです。相手の利害・BATNAを分析し、Win-Winの合意シナリオと交渉戦術を提示します。""",
    "CREATIVEモード": """CREATIVEモードは、アイデア発想・コンセプト設計に特化したモードです。既存の枠を超えた発想を複数提示し、実現可能性と独自性を評価します。""",
    "SUMMARYモード": """SUMMARYモードは、要約・整理に特化したモードです。要点を3〜5項目に絞り、階層構造で簡潔に整理します。""",
    "LEGALモード": """LEGALモードは、法務・規約・コンプライアンスの解説に特化したモードです。法的助言ではなく情報提供として提示します。""",
    "COACHINGモード": """COACHINGモードは、自己変革・思考パターン改善に特化したモードです。質問・内省促進・気づきの提供を優先します。""",
    "OPSモード": """OPSモードは、業務改善・効率化・オペレーション最適化に特化したモードです。ボトルネック特定・工数削減・標準化・自動化の観点から改善策を提示します。""",
    "TECHモード": """TECHモードは、技術・エンジニアリング・システム設計に特化したモードです。技術的トレードオフ・アーキテクチャ選定・実装方針を構造的に提示します。""",
}

ASCEND_FAQ_GUIDE = {
    "プランによって使える機能は違いますか": "はい。利用できる機能はプランや管理者権限によって変わります。ADVANCEやSUPREMEでは、ファイル診断、固定概念レポート、画像生成、個人相談、投資シグナルなどの上位機能が解放されます。",
    "ランクはどうやって上がりますか": "ランクは活用度を示すlevel_scoreに応じて変化します。チャット、診断、レポート活用などの利用によりスコアが加算され、追従者、実行者、戦略家、設計者へ進みます。",
    "ファイル診断はどんな形式に対応していますか": "ファイル診断は、Excel、PDF、CSV、テキスト等の資料分析を想定しています。",
    "登録業種以外の質問もできますか": "可能です。登録業種は回答精度や文脈最適化のための基準ですが、登録業種以外の相談もできます。",
    "画像解析・ファイル解析の精度はどの程度ですか": "精度は入力資料の品質、文字の読み取りやすさ、ファイル構造、データ量に依存します。最終判断には人間の確認が必要です。",
    "個人相談と現状課題診断の違いは何ですか": "個人相談は個別の相談スレッドとして深く継続的に扱う機能です。現状課題診断は、入力された現状を構造化し、本質課題や優先順位を診断する機能です。",
    "Decision Metricsとは何ですか": "Decision Metricsは、意思決定を複数指標でスコアリングし、判断の質、リスク、実行可能性を可視化する機能です。",
    "投資シグナル機能は誰でも使えますか": "投資シグナルは上位機能であり、利用可否はプランや権限設定に依存します。投資判断を保証するものではなく、判断材料の分析支援です。",
    "履歴やレポートのデータは保存されますか": "チャット履歴、診断履歴、レポート等はマイページなどで確認できるよう保存されます。",
    "未開放と表示される機能はどうすれば使えますか": "未開放機能は、プランのアップグレードまたは管理者による権限付与により利用可能になります。",
    "登録業種とは何ですか": "登録業種は、ASCENDが回答や診断を最適化するための業種文脈です。",
    "ファイル診断と通常チャットのファイル解析の違いは": "通常チャットのファイル解析は資料を読みながら要約や質問回答を行う機能です。ファイル診断は数値分析とAI解釈により資料全体を診断する、より深い分析機能です。",
    "ASCENDはどんな業種に対応していますか": "業種を問わず利用できます。経営、戦略、マーケティング、財務、人事、組織、営業、接客など幅広い領域に対応しています。",
    "チャット履歴はどこで確認できますか": "マイページまたはチャット画面の履歴一覧から確認できます。過去のセッションを呼び出して継続相談が可能です。",
    "診断結果は保存されますか": "診断結果はマイページの履歴から確認できます。固定概念レポートなども保存されます。",
    "複数の質問を一度に送れますか": "可能です。まとめて入力することで、ASCENDが論点を整理して構造的に回答します。",
    "ASCENDは何言語に対応していますか": "主に日本語に最適化されています。英語での入力にも対応していますが、日本語での利用を推奨します。",
    "画像は何枚まで添付できますか": "1回のメッセージに1枚の画像を添付できます。複数画像の場合は複数回に分けて送信してください。",
    "ファイルのサイズ制限はありますか": "ファイルサイズや形式によって処理可能な範囲が異なります。大容量ファイルの場合は分割して送信することを推奨します。",
    "モードはいつでも切り替えられますか": "はい。チャット画面から用途別モードをいつでも切り替えられます。最大19モードに対応しています。",
    "ASCENDは投資助言を行いますか": "投資助言は行いません。投資シグナル機能は判断材料の構造化・分析支援であり、投資の推奨・保証はしません。",
    "固定概念レポートはどのくらいの頻度で使えますか": "利用回数の制限はプランによって異なります。定期的に活用することで、思考の癖や盲点を継続的に確認できます。",
    "未来分岐シミュレーターはどう使いますか": "現在の状況、選択肢、制約条件、リスクを入力すると、複数の将来シナリオと打ち手を構造化して提示します。",
    "プロファイル生成はどんな場面で使いますか": "人物の特徴、行動傾向、強み、対応方針を整理する場面で使います。採用、顧客理解、組織設計などに活用できます。",
    "顧客AIマネジメントは誰向けですか": "店舗経営者、接客責任者、CRM担当者向けです。顧客の離脱防止、接客最適化、顧客行動予測などを支援します。",
    "プレゼン資料生成で何が作れますか": "提案書、報告書、戦略説明資料のスライド構成と章立てを生成します。実際のPowerPointファイルではなく、構成・骨子の設計を行います。",
    "思考マップはどんな時に使いますか": "複雑な課題、多数の選択肢、因果関係が絡み合う状況を整理する時に使います。見落としを減らし、思考を構造化します。",
    "矛盾検知は何を検出しますか": "計画・発言・方針・数値の中にある論理矛盾、前提のズレ、整合性の欠如を検出します。戦略や提案の破綻を事前に発見するために使います。",
    "比較分析では何を比べられますか": "複数の選択肢、施策案、戦略オプションを多面的に比較します。メリット、リスク、実行難度、優先度、費用対効果などを軸に評価します。",
    "課題仮説はどう活用しますか": "原因が確定していない段階で、考えるべき仮説を広げるために使います。複数の仮説を生成し、検証すべき論点を整理します。",
    "実行計画ではどんな出力が得られますか": "目標達成に向けた具体的な手順、優先順位、ロードマップ、各フェーズのアクションが得られます。",
    "ASCENDのランク称号は何段階ですか": "4段階です。追従者、実行者、戦略家、設計者の順に上昇します。活用度に応じてlevel_scoreが加算されます。",
}

def _ascend_static_answer(query: str):
    """辞書ヒット時: (key, text) / ヒットなし: ("", "")"""
    import re as _re_asd
    q = query or ""
    q_norm = q.replace("？","").replace("?","").replace("\u3000"," ").replace("　"," ").strip()
    q_low = q_norm.lower()

    # 全機能一覧
    if any(k in q for k in ["全機能", "機能一覧", "何ができる", "できること", "全部教えて", "すべての機能"]):
        parts = ["ASCENDの主な機能は以下です。"]
        for name, desc in ASCEND_FEATURE_GUIDE.items():
            if name in ["ASCENDとは", "ASCENDの名前の由来", "SWIFT / ADVANCE / SUPREME"]:
                continue
            parts.append(f"■ {name}\n{desc}")
        return "全機能一覧", "\n\n".join(parts)

    # AIエンジン比較
    if "SWIFT" in q and "ADVANCE" in q and "SUPREME" in q:
        return "AIエンジン比較", ASCEND_FEATURE_GUIDE["SWIFT / ADVANCE / SUPREME"]

    # FAQ辞書: 柔軟マッチ
    for key, val in ASCEND_FAQ_GUIDE.items():
        key_norm = key.replace("？","").replace("?","").replace("　"," ").strip()
        if key_norm in q_norm:
            return key, val
        key_words = [w for w in _re_asd.split(r"[\s]+", key_norm) if len(w) >= 2]
        if key_words and all(w in q_norm for w in key_words):
            return key, val

    # FEATURE辞書: キー直接マッチ + バリエーション
    for key, val in ASCEND_FEATURE_GUIDE.items():
        if key in q:
            return key, val
        variants = [
            key + "とは", key + "って", key + "を教えて", key + "の説明",
            key + "について", key + "って何", key + "とは何", key + "の機能",
            key + "の使い方", key + "はどう", key + "はどんな", key + "は何",
        ]
        if any(v in q_norm for v in variants):
            return key, val
        if key.lower() in q_low and len(key) >= 3:
            return key, val

    # 緩やかなキーワードマッチ
    _loose_map = [
        (["ASCENDとは","ASCEND とは","このシステムとは","このAIとは"], "ASCENDとは", ASCEND_FEATURE_GUIDE),
        (["使い方","どう使","どうやって使","操作方法"], "AIチャット", ASCEND_FEATURE_GUIDE),
        (["プランは","料金は","費用","月額","いくら"], "プランによって使える機能は違いますか", ASCEND_FAQ_GUIDE),
        (["ランク","称号","レベル"], "ランクシステム", ASCEND_FEATURE_GUIDE),
        (["マイページ","履歴確認"], "マイページ", ASCEND_FEATURE_GUIDE),
        (["名前の由来","名称の意味","ASCENDの意味"], "ASCENDの名前の由来", ASCEND_FEATURE_GUIDE),
    ]
    for kws, dict_key, guide in _loose_map:
        if any(kw in q for kw in kws):
            val = guide.get(dict_key, "")
            if val:
                return dict_key, val

    return "", ""


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
        "ascend":      "【ASCENDモード】ASCENDの機能・使い方・思想・プラン・AIエンジン・FAQ説明に特化せよ。system_prompt内のASCEND説明情報を最優先根拠とし、RAGナレッジ・汎用AI回答・キャラクター口調は使用禁止。絵文字・幼稚な口調禁止。機能概要・入力・出力・実務での使い道・経営判断上の価値を構造的に説明せよ。「分かりません」「専門外です」は絶対禁止。",
    }
    _mode_key = (req.purpose_mode or "auto").strip().lower()
    # ASCEND説明クエリは強制的にascendモードへ
    _ascend_explain_kws = ["ASCEND","使い方","とは","機能","名前の由来","SWIFT","ADVANCE","SUPREME","Decision Metrics","固定概念レポート","ファイル診断","プロファイル生成","顧客AIマネジメント","プレゼン資料生成","未来分岐シミュレーター","ランクシステム","プラン","マイページ","画像生成とは","何ができ"]
    if _mode_key == "auto" and any(k in req.message for k in _ascend_explain_kws):
        _mode_key = "ascend"
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
    from api.core.features import is_feature_enabled
    _is_ascend_about = any(k in (req.message or "") for k in ["ASCEND","とは","使い方","機能","プロファイル生成","顧客AIマネジメント","画像生成とは","プレゼン資料生成","未来分岐シミュレーター","ファイル診断","Decision Metrics","固定概念レポート","ランクシステム","比較分析","構造診断","課題仮説","矛盾検知","実行計画","思考マップ","投資シグナル","個人相談","何ができ","料金は"])
    if _is_image_gen_request(req.message, has_image=image_b64 is not None) and not _is_ascend_about and not is_feature_enabled(uid, "image_generation"):
        reply = "画像生成は現在未開放のため使用できません。"
    elif _is_image_gen_request(req.message, has_image=image_b64 is not None) and not _is_ascend_about:
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
    _sources = []
    if _rag_chunks and not generated_images:
        _max_sc = max((float(c.get("_score",0)) for c in _rag_chunks), default=0.0)
        _is_retrieved = len(_rag_chunks) > 0
        _RAG_USE_THRESHOLD = 0.3  # 実際に使われた閾値
        _sources = [
            {
                "text": (_ck.get("text","") or "")[:200],
                "score": float(_ck.get("_score", 0)),
                "source_id": str(_ck.get("source_id","")),
                "is_retrieved": float(_ck.get("_score", 0)) >= _RAG_USE_THRESHOLD,
            }
            for _ck in _rag_chunks
        ]
    elif not generated_images:
        _sources = [] if _plan_hit_m else [{"text": "", "score": 0.0, "source_id": "", "is_retrieved": False}]
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
        print(f"[CASES_DEBUG] raw={repr(_cases_raw[:300])} count={len(cases)}", flush=True)
    except Exception as _cases_err:
        print(f"[CASES_ERROR] {_cases_err}", flush=True)
        cases = []
    # ASCEND辞書ヒット時: 推奨をsuggestionsの先頭に追加
    if _ascend_dict_hit:
        _ascend_suggest = f"【ASCEND】{_ascend_dict_key}について詳しく教えて"
        if _ascend_suggest not in cases:
            cases = [_ascend_suggest] + cases[:2]
    # 構造化データ生成（戦略相談時のみ・画像生成・雑談は除外）
    structured = None
    _consulting_intents = {"相談", "意思決定", "分析", "作成", "予測", "投資"}
    _qp_intent = query_plan.get("intent", "")
    _is_talk_intent = "雑談" in _qp_intent
    _mode_forced = (req.purpose_mode or "auto").strip().lower() != "auto"
    _is_consulting = (not is_talk) and ((_mode_forced) or ((not _is_talk_intent) and (any(i in _qp_intent for i in _consulting_intents))))
    # 相談モードでも雑談intentの場合は会話形式で返答
    _list_keywords2 = ["リスト","一覧","やること","手順","チェックリスト","箇条","ステップ","まとめて","列挙"]
    _is_list_req2 = any(w in req.message for w in _list_keywords2)
    if (is_talk or _is_talk_intent) and not _is_list_req2 and not system_prompt.endswith("【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"):
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
                    # FC解放通知（12回到達時）
                    if _fc_cnt == 12:
                        _write_notification(
                            uid=uid,
                            notif_type="fc",
                            title="🧠 固定概念レポートが解放されました！",
                            body="マイページの「固定概念」タブでレポートを確認できます。",
                            link_tab="fc",
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
                            "score": float(chunk.get("_score",0)),
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

    reply = _clean_reply(reply)
    _save_message(tenant_id, uid, chat_id, "assistant", reply, cases=cases, structured=structured, images=generated_images, sources=_sources)
    # usage_logs に記録（total_chat_count 集計用）
    try:
        import datetime as _dt
        get_db().collection("usage_logs").add({
            "user_id": uid,
            "tenant_id": tenant_id,
            "prompt": req.message[:500],
            "purpose_mode": getattr(req, "purpose_mode", "auto"),
            "is_admin_test": False,
            "timestamp": (_dt.datetime.utcnow() + _dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass

    # DEBUG: ResponseValidationError原因調査
    try:
        from fastapi.encoders import jsonable_encoder as _jae
        import json as _jdebug
        _resp_debug = _jae(dict(
            reply=reply, chat_id=chat_id, cases=cases,
            images=generated_images, structured=structured,
            sources=_sources, confirmation_choices=_confirmation_choices,
            intent=intent_state if isinstance(intent_state, dict) else None,
            intent_label=query_plan.get('intent','') if isinstance(query_plan, dict) else None,
        ))
    except Exception as _dbe:
        pass
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=cases, images=generated_images, structured=structured, sources=_sources, confirmation_choices=_confirmation_choices, intent=intent_state if isinstance(intent_state, dict) else None, intent_label=query_plan.get("intent","") if isinstance(query_plan, dict) else None)

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
            .limit(500)
            .stream()
        )
        result = []
        for d in docs:
            data = d.to_dict() or {}
            if data.get("is_deleted", False):
                continue
            if data.get("tenant_id") != tenant_id:
                continue
            if data.get("scope") != SCOPE:
                continue
            result.append(SessionInfo(
                chat_id    = data.get("chat_id", "main"),
                title      = data.get("title", data.get("chat_id", "main")),
                updated_at = str(data.get("updated_at", "")),
            ))
        result.sort(key=lambda x: x.updated_at or "", reverse=True)
        return result[:50]
    except Exception:
        return [SessionInfo(chat_id="main", title="main")]

@router.post("/session/new")
def new_session(payload: dict = Depends(verify_token)):
    uid       = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id   = str(uuid.uuid4())[:8]
    _ensure_session(tenant_id, uid, chat_id, title="新しいチャット", force_create=True)
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
_ACTION_WORDS = ["作って","描いて","描画","デザイン","作る","generate","draw","design","render"]
_EDIT_WORDS = ["編集","加工","修正","変換","背景","切り抜","色変更","変えろ","変える","変え","差し替","置き換","影","シルエット","陰","塗り替","合成","edit","modify","restyle","change","replace","swap"]
_ANALYSIS_WORDS = ["解析","分析","要約","読んで","説明","pdf","spreadsheet","excel","スプレッドシート"]

def _is_image_gen_request(text: str, has_image: bool = False) -> bool:
    t = (text or "").lower()
    has_subject = any(w in t for w in _IMAGE_WORDS)
    has_action  = any(w in t for w in _ACTION_WORDS)
    has_edit    = any(w in t for w in _EDIT_WORDS)
    has_analysis= any(w in t for w in _ANALYSIS_WORDS)
    _INSTRUCTION_WORDS = ["変更","変えて","にして","してください","しろ","追加して","追加","にしろ","を変","色を","色に","背景を","削除","除去","切り取","合成","スタイル","モノクロ","白黒","明る","暗く","ぼかし","文字を","テキストを","ロゴを"]
    has_instruction = any(w in t for w in _INSTRUCTION_WORDS)
    if has_image and (has_edit or has_instruction): return True
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
    _EDIT_MODELS = ["gemini-2.5-flash-image","gemini-3-pro-image-preview"]
    candidates = _EDIT_MODELS[:] if image_b64 else _IMAGE_MODELS[:]
    print(f"[IMG_CANDIDATES] {candidates}", flush=True)

    if image_b64:
        strict_prompt = (
            "以下は画像編集の最終指示です。ユーザー指示を最優先し、"
            "勝手な解釈拡張・要素追加を極力しないこと。\n"
            "明示された要素は必ず反映し、"
            "明示されていない要素は変更しないこと。\n"
            "参照画像を元に、以下の指示通りに編集した画像を返せ。\n"
            f"【ユーザー最終指示】\n{prompt}"
        )
    else:
        strict_prompt = (
            "以下は画像生成の最終指示です。ユーザー指示を最優先し、"
            "勝手な解釈拡張・要素追加を極力しないこと。\n"
            "明示された要素は必ず反映し、"
            "明示されていない要素は勝手に足さないこと。\n"
            f"【ユーザー最終指示】\n{prompt}"
        )
    user_parts = [_types.Part(text=strict_prompt)]
    if image_b64:
        try:
            img_bytes = _b64.b64decode(image_b64)
            # 画像が大きすぎる場合はリサイズ（2MB超）
            if len(img_bytes) > 2 * 1024 * 1024:
                try:
                    from PIL import Image as _PIL
                    import io as _io2
                    _im = _PIL.open(_io2.BytesIO(img_bytes))
                    _im.thumbnail((1024, 1024), _PIL.LANCZOS)
                    _buf = _io2.BytesIO()
                    _im.save(_buf, format="JPEG", quality=85)
                    img_bytes = _buf.getvalue()
                    image_mime = "image/jpeg"
                    print(f"[IMG_RESIZE] resized to {len(img_bytes)} bytes", flush=True)
                except Exception as _re: print(f"[IMG_RESIZE_SKIP] {_re}", flush=True)
            # 編集時は常にJPEGに正規化
            try:
                from PIL import Image as _PILc
                import io as _ioc
                _imc = _PILc.open(_ioc.BytesIO(img_bytes)).convert("RGB")
                _bufc = _ioc.BytesIO()
                _imc.save(_bufc, format="JPEG", quality=92)
                img_bytes = _bufc.getvalue()
                image_mime = "image/jpeg"
                print(f"[IMG_NORMALIZE] JPEG {len(img_bytes)} bytes", flush=True)
            except Exception as _ce: print(f"[IMG_NORMALIZE_SKIP] {_ce}", flush=True)
            print(f"[IMG_SIZE] bytes={len(img_bytes)}", flush=True)
            user_parts.append(_types.Part(inline_data=_types.Blob(mime_type=image_mime, data=img_bytes)))
        except Exception as _img_e:
            print(f"[IMG_INPUT_ERROR] {type(_img_e).__name__}: {_img_e}", flush=True)
            raise
    contents = [_types.Content(role="user", parts=user_parts)]
    try:
        _safety = [
            _types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            _types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            _types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            _types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
        _modalities = ["TEXT","IMAGE"]
        cfg = _types.GenerateContentConfig(response_modalities=_modalities, temperature=1.0, safety_settings=_safety)
    except Exception as _cfg_e:
        print(f"[IMG_CFG_WARN] safety cfg failed: {_cfg_e}", flush=True)
        try:
            cfg = _types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"], temperature=1.0)
        except Exception as _cfg_e2:
            print(f"[IMG_CFG_WARN2] basic cfg failed: {_cfg_e2}", flush=True)
            cfg = None

    for model in candidates:
        try:
            print(f"[IMG_GEN_START] model={model} has_image={image_b64 is not None}", flush=True)
            import concurrent.futures as _cf
            _gen_fn = (lambda: client.models.generate_content(model=model, contents=contents, config=cfg)) if cfg else (lambda: client.models.generate_content(model=model, contents=contents))
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                _ft = _ex.submit(_gen_fn)
                try:
                    res = _ft.result(timeout=90)
                except _cf.TimeoutError:
                    print(f"[IMG_GEN_TIMEOUT] model={model}", flush=True)
                    continue
            print(f"[IMG_GEN_DONE] model={model}", flush=True)
            images = []
            all_parts = getattr(res, "parts", None) or []
            if not all_parts:
                for cand in (getattr(res,"candidates",None) or []):
                    all_parts.extend(getattr(getattr(cand,"content",None),"parts",None) or [])
            print(f"[IMG_DEBUG] model={model} parts={len(all_parts)} candidates={len(getattr(res,'candidates',None) or [])} finish={[(getattr(c,'finish_reason',None)) for c in (getattr(res,'candidates',None) or [])]}", flush=True)
            print(f"[IMG_FEEDBACK] prompt_feedback={getattr(res,'prompt_feedback',None)} safety={[(getattr(c,'safety_ratings',None)) for c in (getattr(res,'candidates',None) or [])]}", flush=True)
            for i,part in enumerate(all_parts):
                print(f"[IMG_PART] i={i} has_inline={getattr(part,'inline_data',None) is not None} has_text={bool(getattr(part,'text',None))}", flush=True)
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
                return ("画像を生成しました。✨", images)
        except Exception as e:
            print(f"[IMG_GEN_ERROR] model={model} err={type(e).__name__}: {e}", flush=True)
            continue
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
_ACTION_WORDS = ["作って","描いて","描画","デザイン","作る","generate","draw","design","render"]
_EDIT_WORDS = ["編集","加工","修正","変換","背景","切り抜","色変更","変えろ","変える","変え","差し替","置き換","影","シルエット","陰","塗り替","合成","edit","modify","restyle","change","replace","swap"]
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
    from api.core.features import is_feature_enabled as _isfe_tier
    if req.ai_tier in ("ultra", "apex"):
        _tier_feat = "ascend_ultra" if req.ai_tier == "ultra" else "ascend_apex"
        if not _isfe_tier(uid, _tier_feat):
            raise HTTPException(status_code=403, detail=f"このAIエンジン（{req.ai_tier}）は現在未開放のため使用できません。")
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    chat_id = (req.chat_id or "main").strip() or "main"
    _ensure_session(tenant_id, uid, chat_id)
    generated_images = []
    # 画像が添付されている場合：編集指示なら画像編集、そうでなければ画像解析
    if req.image_b64:
        from api.core.features import is_feature_enabled
        _is_ascend_about = any(k in (req.message or "") for k in ["ASCEND","とは","使い方","機能","プロファイル生成","顧客AIマネジメント","画像生成とは","プレゼン資料生成","未来分岐シミュレーター","ファイル診断","Decision Metrics","固定概念レポート","ランクシステム","比較分析","構造診断","課題仮説","矛盾検知","実行計画","思考マップ","投資シグナル","個人相談","何ができ","料金は"])
        if _is_image_gen_request(req.message, has_image=True) and not _is_ascend_about and not is_feature_enabled(uid, "image_generation"):
            reply = "画像生成は現在未開放のため使用できません。"
        elif _is_image_gen_request(req.message, has_image=True) and not _is_ascend_about:
            try:
                reply, generated_images = _generate_image(req.message, req.image_b64, req.image_mime)
            except Exception as e:
                reply = f"画像生成エラー: {e}"
                generated_images = []
        else:
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
        from api.core.features import is_feature_enabled
        _is_ascend_about = any(k in (req.message or "") for k in ["ASCEND","とは","使い方","機能","プロファイル生成","顧客AIマネジメント","画像生成とは","プレゼン資料生成","未来分岐シミュレーター","ファイル診断","Decision Metrics","固定概念レポート","ランクシステム","比較分析","構造診断","課題仮説","矛盾検知","実行計画","思考マップ","投資シグナル","個人相談","何ができ","料金は"])
        if _is_image_gen_request(req.message, has_image=False) and not _is_ascend_about and not is_feature_enabled(uid, "image_generation"):
            reply = "画像生成は現在未開放のため使用できません。"
        elif _is_image_gen_request(req.message, has_image=False) and not _is_ascend_about:
            try:
                reply, generated_images = _generate_image(req.message, None, req.image_mime)
            except Exception as e:
                reply = f"画像生成エラー: {e}"
                generated_images = []
        else:
            try:
                base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
                messages = _load_history(tenant_id, uid, chat_id)
                messages.append({"role": "user", "content": req.message})
                reply = call_llm(
                    system_prompt=base_prompt,
                    messages=messages,
                    ai_tier=req.ai_tier,
                )
            except Exception as e:
                reply = f"AI呼び出しエラー: {e}"
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
        except Exception as _ge:
            pass
    _save_message(tenant_id, uid, chat_id, "user", req.message)
    reply = __import__("re").sub(r" {2,}", " ", reply).strip()
    _save_message(tenant_id, uid, chat_id, "assistant", reply, images=generated_images)
    # usage_log書き込み
    try:
        _ulog_db2 = get_db()
        _ulog_db2.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"), "is_admin_test": False, "purpose_mode": getattr(req, "purpose_mode", "auto")})
    except Exception:
        pass
    _safe_images = [{"mime_type": img.get("mime_type","image/png"), "gcs_url": img.get("gcs_url","")} for img in generated_images]
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=_safe_images)


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
    from api.core.features import is_feature_enabled as _isfe_tier
    if req.ai_tier in ("ultra", "apex"):
        _tier_feat = "ascend_ultra" if req.ai_tier == "ultra" else "ascend_apex"
        if not _isfe_tier(uid, _tier_feat):
            raise HTTPException(status_code=403, detail=f"このAIエンジン（{req.ai_tier}）は現在未開放のため使用できません。")
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
    reply = __import__("re").sub(r" {2,}", " ", reply).strip()
    _save_message(tenant_id, uid, chat_id, "assistant", reply, structured=_inv_structured)
    # usage_log書き込み
    try:
        _ulog_db3 = get_db()
        _ulog_db3.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"), "is_admin_test": False, "purpose_mode": getattr(req, "purpose_mode", "auto")})
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
    from api.core.features import is_feature_enabled as _isfe_tier
    if req.ai_tier in ("ultra", "apex"):
        _tier_feat = "ascend_ultra" if req.ai_tier == "ultra" else "ascend_apex"
        if not _isfe_tier(uid, _tier_feat):
            raise HTTPException(status_code=403, detail=f"このAIエンジン（{req.ai_tier}）は現在未開放のため使用できません。")
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
    reply = __import__("re").sub(r" {2,}", " ", reply).strip()
    _save_message(tenant_id, uid, chat_id, "assistant", reply)
    # usage_log書き込み
    try:
        _ulog_db3 = get_db()
        _ulog_db3.collection("usage_logs").add({"user_id": uid, "tenant_id": tenant_id, "prompt": req.message[:200], "timestamp": (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"), "is_admin_test": False, "purpose_mode": getattr(req, "purpose_mode", "auto")})
    except Exception:
        pass
    return ChatResponse(reply=reply, chat_id=chat_id, msg_id=str(uuid.uuid4()), cases=[], images=[])


@router.get("/images")
def get_image_gallery(payload: dict = Depends(verify_token)):
    """生成画像ギャラリー一覧取得"""
    from api.core.features import is_feature_enabled
    uid = payload["uid"]
    if not is_feature_enabled(uid, "image_gallery"):
        raise HTTPException(status_code=403, detail="生成画像ギャラリーは現在未開放のため使用できません。")
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
    if not is_feature_enabled(uid, "image_gallery"):
        raise HTTPException(status_code=403, detail="生成画像ギャラリーは現在未開放のため使用できません。")
    db = get_db()
    try:
        # Firestoreから削除
        db.collection("image_gallery").document(uid).collection("images").document(image_id).delete()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/image_b64")
def get_image_b64(url: str, payload: dict = Depends(verify_token)):
    """GCS画像をbase64で返すプロキシ"""
    import urllib.request, base64 as _b64p
    try:
        with urllib.request.urlopen(url, timeout=15) as _rp:
            _data = _rp.read()
            _mime = (_rp.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
        return {"b64": _b64p.b64encode(_data).decode(), "mime": _mime}
    except Exception as _pe:
        raise HTTPException(status_code=400, detail=str(_pe))

# DEBUG REMOVE LATER

# ============================================================
# SSE Streaming Endpoints
# ============================================================
def _sse_evt(data: dict) -> str:
    payload = _json_sse.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"

@router.post("/send_stream")
def send_message_stream(req: ChatRequest, payload: dict = Depends(verify_token)):
    import queue as _qm, threading as _thm
    _q = _qm.Queue()
    _DONE = object()
    def _worker():
        try:
            import time as _tss
            _q.put({"type":"step","label":"入力を解析中..."})
            uid = payload["uid"]
            tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
            chat_id = (req.chat_id or "main").strip() or "main"
            _ensure_session(tenant_id, uid, chat_id)
            history = _load_history(tenant_id, uid, chat_id)
            base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
            # ── ASCEND静的辞書ヒット検出（推奨モード用・send_stream） ──
            _ascend_dict_key, _ascend_dict_text = _ascend_static_answer(req.message)
            _ascend_dict_hit = bool(_ascend_dict_key and _ascend_dict_text)
            chat_mode = (req.chat_mode or "consult").strip().lower()
            is_talk = chat_mode == "talk"
            _is_custom_replace = False
            _u_doc = {}
            try:
                _u_snap = get_db().collection("users").document(uid).get()
                if _u_snap.exists:
                    _u_doc = _u_snap.to_dict() or {}
                    _hc = bool(_u_doc.get("custom_sys_prompt",""))
                    _ir2 = _u_doc.get("custom_prompt_mode","append") == "replace"
                    _is_custom_replace = _hc and _ir2 and is_talk
            except Exception:
                pass
            _uplan = (_u_doc.get("plan") or "user").strip()
            _is_au = _uplan in ("apex","ultra_admin","ultra_member")

            # ── 対象者確認：Pythonで曖昧さ検出→LLMで選択肢動的生成 ──
            import re as _re_amb
            _amb_keywords = [
                r"チェックリスト", r"手順", r"マニュアル", r"業務フロー", r"学べる", r"ここで学",
                r"やること", r"ルール", r"規則", r"注意事項", r"やり方", r"方法", r"流れ",
                r"教えて", r"一覧", r"リスト", r"項目", r"内容", r"説明して", r"とは",
            ]
            _amb_hit = any(_re_amb.search(kw, req.message) for kw in _amb_keywords)
            # historyから選択済みターゲットを抽出
            _resolved_target = ""
            for _hi in range(len(history) - 1, -1, -1):
                _hm = history[_hi]
                if _hm.get("role") == "assistant" and "【確認】" in (_hm.get("content", "")):
                    if _hi + 1 < len(history) and history[_hi + 1].get("role") == "user":
                        _user_ans = history[_hi + 1].get("content", "").strip()
                        _resolved_target = _user_ans[:30]
                    break
            _last_ai = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
            _already_asked = _last_ai and "【確認】" in (_last_ai.get("content", ""))
            # 対象者抽出（構造で判断・_amb_hit不問）
            import re as _re
            # ── admin_uid 解決（_rc_pre前に必ず実施） ──
            _admin_uid = ""
            try:
                _need_admin_rag = (
                    (_u_doc.get("plan") == "ultra_member" and _u_doc.get("use_admin_settings", False))
                    or _u_doc.get("plan") in ("apex", "ultra_admin")
                )
                if _need_admin_rag:
                    _admin_uid = _get_admin_uid(tenant_id)
                    # キャッシュ優先: インメモリ→tenant_settings→初回のみスキャン
            except Exception:
                pass
            _rc_pre = []  # _rc_pre廃止: _rcと一本化
            _tgt_candidates = []
            _seen_sids = {}
            for c in _rc_pre:
                _sid = c.get("source_id","") or c.get("_source_id","")
                _cscore = float(c.get("_score", c.get("score", 0)))
                if _cscore < 0.55: continue
                _stitle = (c.get("title","") or "").strip()
                import re as _re_ext
                _stitle = _re_ext.sub(r"\.[a-zA-Z0-9]{2,5}$", "", _stitle).strip()
                _ctext = (c.get("text","") or "").strip()[:40].replace("\n"," ")
                _label = _stitle if _stitle else _ctext
                if _sid and _label and _sid not in _seen_sids:
                    _seen_sids[_sid] = _label
            if len(_seen_sids) >= 2:
                _tgt_candidates = list(_seen_sids.values())[:3]
            # --- 対象者推定（チャンク内容からGeminiで誰向けかを判定）---
            if len(_tgt_candidates) >= 2:
                try:
                    import re as _re_inf
                    _AUDIENCE_KEYWORDS = [
                        # 水商売・ナイト系
                        (["キャスト","cast","嬢","女の子","ホスト","ホステス","コンパニオン","源氏名"], "キャスト向け"),
                        # 店舗スタッフ全般
                        (["スタッフ","staff","従業員","店員","ホールスタッフ","フロアスタッフ","アルバイト","パート","ボーイ","黒服"], "スタッフ向け"),
                        # 新人・研修
                        (["新人","新入","研修","入門","初心者","ビギナー","入社","新卒","OJT","見習い","仮配属"], "新人向け"),
                        # 管理職・リーダー
                        (["リーダー","店長","マネージャー","幹部","責任者","管理職","副店長","チーフ","統括","エリアマネージャー","オーナー","経営者","社長"], "リーダー・管理職向け"),
                        # 営業・販売職
                        (["営業","セールス","販売","sales","商談","提案","クロージング","テレアポ","飛び込み"], "営業・販売向け"),
                        # 美容・サロン系
                        (["美容師","スタイリスト","エステ","ネイリスト","アイリスト","セラピスト","施術","カット","トリートメント"], "美容・サロン向け"),
                        # 医療・介護
                        (["看護師","医師","介護士","ヘルパー","ケアマネ","薬剤師","リハビリ","クリニック","病院","福祉"], "医療・介護向け"),
                        # 飲食・接客
                        (["ウェイター","ウェイトレス","調理師","シェフ","料理人","キッチン","ホール","飲食","レストラン","カフェ","居酒屋"], "飲食・接客向け"),
                        # 事務・バックオフィス
                        (["事務","バックオフィス","経理","総務","人事","労務","庶務","受付","秘書"], "事務・管理部門向け"),
                        # エンジニア・IT
                        (["エンジニア","開発者","プログラマー","システム","IT","デザイナー","ディレクター","PM","プロジェクトマネージャー"], "エンジニア・IT向け"),
                        # 教育・講師
                        (["教師","講師","インストラクター","トレーナー","コーチ","先生","教員","塾","学校"], "教育・指導者向け"),
                        # 顧客・クライアント
                        (["顧客","お客様","クライアント","ユーザー","消費者","購入者","会員"], "顧客・クライアント向け"),
                        # 共通
                        (["共通","全員","全スタッフ","全体","誰でも","全職種","全部門"], "共通"),
                    ]
                    _all_text = " ".join([
                        (c.get("title","") or "") + " " + (c.get("text","") or "")[:200]
                        for c in _rc_pre[:8] if float(c.get("_score", c.get("score",0))) >= 0.45
                    ]).lower()
                    _inferred_set = []
                    for _kws, _label in _AUDIENCE_KEYWORDS:
                        if any(k.lower() in _all_text for k in _kws):
                            if _label not in _inferred_set:
                                _inferred_set.append(_label)
                    if len(_inferred_set) >= 2:
                        _tgt_candidates = _inferred_set[:3]
                    elif len(_inferred_set) == 1:
                        _tgt_candidates = _inferred_set + ["共通"]
                except Exception as _infer_err:
                    print('INFER_ERR:', _infer_err, flush=True)
            # --- デバッグprint除去済み ---
            _has_custom_pre = bool((_u_doc or {}).get("custom_sys_prompt", ""))
            if len(_tgt_candidates) >= 2 and not _already_asked and not _is_au and not _has_custom_pre:
                _choices = _tgt_candidates
                _confirm_reply = "【確認】「{}」について確認させてください。\n誰向けの情報をお探しですか？".format(req.message[:20])
                _confirm_cases = []
                _save_message(tenant_id, uid, chat_id, "user", req.message)
                _save_message(tenant_id, uid, chat_id, "assistant", _confirm_reply, cases=_confirm_cases)
                _q.put({"type":"done","reply":_confirm_reply,"chat_id":chat_id,"msg_id":str(__import__("uuid").uuid4()),"cases":_confirm_cases,"confirmation_choices":_tgt_candidates,"images":[],"structured":None,"sources":[],"intent":{},"intent_label":""})
                return
            # 選択済みターゲットをプロンプトに注入
            if _resolved_target:
                base_prompt = base_prompt + "\n\n【対象者指定】ユーザーが選択した回答対象は「" + _resolved_target + "」。この対象向けに限定して回答せよ。"


            import threading as _th2
            _ires = {}
            qp = {}
            if not _is_custom_replace or _is_au:
                _q.put({"type":"step","label":"意図を特定中..."})
            _lens_preset, _lens_hier = "expert", "raw"
            if not _is_custom_replace:
                try: _lens_preset, _lens_hier = lgbm_select_summary_lens(req.message, "auto")
                except Exception: pass
                if qp.get("summary_lens", {}).get("preset"):
                    _lens_preset = qp["summary_lens"]["preset"]
            if not _is_custom_replace or _is_au:
                _q.put({"type":"step","label":"専用ナレッジを検索中..." if _is_au else "ナレッジを検索中..."})
            # _admin_uid は _rc_pre 前に解決済み（再代入しない）
            _thcd = _has_custom_pre
            # 全プラン・全モードでRAGを呼ぶ
            _tco = (_u_doc.get("custom_sys_prompt") or "").strip() if _thcd else ""
            _tb = _tco if (_tco and not _is_custom_replace) else base_prompt
            _rc = []
            # ASCENDモード or 辞書ヒット時: 辞書内容をsystem_promptへ注入してLLMに渡す
            _tb_for_rag = _tb
            _mode_is_ascend = (req.purpose_mode or "").strip().lower() == "ascend"
            if _ascend_dict_hit or _mode_is_ascend:
                _dict_content = _ascend_dict_text if _ascend_dict_text else ""
                if _mode_is_ascend and not _dict_content:
                    parts = ["ASCENDの主な機能は以下です。"]
                    for name, desc in ASCEND_FEATURE_GUIDE.items():
                        if name not in ["ASCENDとは", "ASCENDの名前の由来", "SWIFT / ADVANCE / SUPREME"]:
                            parts.append(f"■ {name}\n{desc}")
                    _dict_content = "\n\n".join(parts)
                _tb_for_rag = _tb + "\n\n【ASCEND辞書情報 - 必ずこの内容を根拠に回答せよ】\n" + _dict_content
            sp, _rc, _is_ascend_about = _build_system_with_rag(tenant_id, req.message + ('　' + _resolved_target if _resolved_target else ''), _tb_for_rag, uid=uid, admin_uid=_admin_uid, is_apex_ultra=_is_au)
            # ASCENDモード or 辞書ヒット時は画像生成を必ずスキップ
            if _ascend_dict_hit or _mode_is_ascend:
                _is_ascend_about = True
            _rc_pre = _rc  # _rc_preを_rcで代替
            if _thcd and not _is_custom_replace and not _is_ascend_about:
                sp = ("【会話モード・最優先指示】以下のキャラクター設定と知識ファイルの内容を完全に内面化し、自分の言葉として再構築して答えよ。"
                      "ナレッジの文言・ファイル名・資料名・出典を直接引用・露出することは絶対禁止。"
                      "質問に対してキャラクターの口調で直接答えよ。"
                      "知識ファイルに表・一覧・チェックリストが含まれる場合は必ずMarkdown表で出力せよ。表は1行1レコードで各セルを簡潔に記述し、セル内で改行・番号付きリスト・複数段落を混在させるな。"
                      "前置き宣言（「今回は〜についてご説明します」等）は禁止。\n\n" + sp)
            _lk = ["リスト","一覧","やること","手順","チェックリスト","箇条","ステップ","まとめて","列挙"]
            _ilr = any(w in req.message for w in _lk)
            if not _is_au and not _is_custom_replace and not _thcd and not _ilr:
                sp = sp.replace("出力形式：結論→打ち手→優先順位→リスク→次の観測。",
                                "出力形式：自然な会話形式で簡潔に回答せよ。箇条書きや表は使わず、2〜4文程度で答えよ."
                                ) + "\n\n【会話モード】雑談・日常会話として自然に短く返答せよ。分析・構造化・戦略提案は不要。"
            pass  # _it無効化済み
            ist = _ires.get("state",{}) if not _is_custom_replace else {}
            if ist and not _is_custom_replace and not (_is_au and is_talk):
                sp += (f"\n\n【ユーザーの脳内カルテ（深層プロファイル）】"
                       f"\n・ステージ: {ist.get('current_stage','')}"
                       f"\n・真の渇望: {ist.get('true_desire','')}"
                       f"\n・バイアス: {ist.get('bias','')}"
                       f"\n・不足観点: {ist.get('missing_piece','')}")
            _plan_kws = ["プラン","料金","ascend","サブスク","subscription","ultra","apex","pro","standard","starter","月額","契約"]
            _plan_hit = any(k in req.message.lower() for k in _plan_kws)
            if _plan_hit:
                sp += "\n\n" + PLAN_DEFINITION
            # RAG=0件 かつ システムプロンプトにも該当なし → ⚠️フォールバック
            _no_rag_fallback_s = (not _rc) and (not _plan_hit)
            if _no_rag_fallback_s:
                sp += '\n\n【⚠️ナレッジ未検証回答モード】専用ナレッジ・中央倉庫のいずれにも該当情報が存在しない。質問に対してLLMの一般知識で誠実に回答せよ。「回答できません」「専門外です」等の拒否は絶対禁止。回答の冒頭に必ず「⚠️ナレッジ未検証回答」と表示せよ。'
            _mk = (req.purpose_mode or "auto").strip().lower()
            print(f"[STREAM_MODE_DEBUG] purpose_mode_raw={repr(req.purpose_mode)} mk={repr(_mk)}", flush=True)
            _ascend_explain_kws2 = ["ASCEND","使い方","とは","機能","名前の由来","SWIFT","ADVANCE","SUPREME","Decision Metrics","固定概念レポート","ファイル診断","プロファイル生成","顧客AIマネジメント","プレゼン資料生成","未来分岐シミュレーター","ランクシステム","プラン","マイページ","画像生成とは","何ができ"]
            if _mk == "auto" and any(k in req.message for k in _ascend_explain_kws2):
                _mk = "ascend"
            _MI = {
                "strategy":"【STRATEGYモード】戦略立案・競合優位・中長期計画に特化せよ。",
                "numeric":"【NUMERICモード】数値・指標・KPI・ROI分析に特化せよ。",
                "growth":"【GROWTHモード】成長戦略・人材育成・スケール設計に特化せよ。",
                "control":"【CONTROLモード】構造・権限・意思決定フローの可視化に特化せよ。",
                "analysis":"【ANALYSISモード】データ分析・原因究明・相関発見に特化せよ。",
                "planning":"【PLANNINGモード】ロードマップ・フェーズ設計に特化せよ。時系列・マイルストーン・依存関係を明示したアクションプランを提示せよ。",
                "risk":"【RISKモード】リスク評価・シナリオ分析・回避戦略に特化せよ。",
                "creative":"【CREATIVEモード】アイデア発想・コンセプト設計・差別化に特化せよ。",
                "summary":"【SUMMARYモード】要点抽出・構造化・整理に特化せよ。",
                "negotiation":"【NEGOTIATIONモード】交渉・説得・影響力行使に特化せよ。",
                "coaching":"【COACHINGモード】個人成長支援・目標達成サポートに特化せよ。",
                "diagnosis":"【DIAGNOSISモード】現状診断・課題発見・改善提案に特化せよ。",
                "forecast":"【FORECASTモード】予測・シナリオ分析・将来設計に特化せよ。",
                "legal":"【LEGALモード】法務・規約・コンプライアンスに特化せよ。",
                "finance":"【FINANCEモード】財務・投資・資金調達の分析に特化せよ。",
                "marketing":"【MARKETINGモード】マーケ・集客・ブランド戦略に特化せよ。",
                "hr":"【HRモード】人材・組織・採用・育成に特化せよ。",
                "ops":"【OPSモード】業務改善・効率化・プロセス最適化に特化せよ。",
                "tech":"【TECHモード】技術・エンジニアリング・システム設計に特化せよ。",
                "ascend":"【ASCENDモード】ASCENDの機能・使い方・思想・プラン・AIエンジン・FAQ説明に特化せよ。system_prompt内のASCEND説明情報を最優先根拠とし、RAGナレッジ・汎用AI回答・キャラクター口調は使用禁止。絵文字・幼稚な口調禁止。機能概要・入力・出力・実務での使い道・経営判断上の価値を構造的に説明せよ。分かりませんは絶対禁止。",
            }
            # planning mode: ExecutionPlan生成（send_stream用）
            execution_plan_obj = None
            if _mk == 'planning':
                try:
                    from api.core.intent import build_execution_plan as _bep_ss
                    execution_plan_obj = _bep_ss(req.message, tenant_id, 'mixed')
                    import json as _epj_dbg
                    if execution_plan_obj and isinstance(execution_plan_obj, dict):
                        sp += ('\n\n【実行計画オブジェクト（必ずこの構造を根拠に説明せよ）】\n'
                               + _epj_dbg.dumps(execution_plan_obj, ensure_ascii=False)[:2000]
                               + '\n上記ExecutionPlanの各フェーズ・タスク・KPI・依存関係を根拠にして補足説明のみ行え。架空の数値・タスク追加は禁止。')
                except Exception as _ep_ss_err:
                    execution_plan_obj = None
            if not is_talk and _mk in _MI and not _is_custom_replace:
                sp = _MI[_mk] + "\n\n" + sp
            _LENS_INST = {
                "expert":   "【出力スタイル: EXPERT】論拠・根拠・数値・事例を必ず含めよ。",
                "mentor":   "【出力スタイル: MENTOR】段階的に教えよ。初心者でも理解できる説明を心がけよ。",
                "executor": "【出力スタイル: EXECUTOR】具体的な手順・アクションを優先せよ。番号付きステップで実行可能な形で提示せよ。",
                "general":  "【出力スタイル: GENERAL】要点を簡潔にまとめよ。3〜5項目に絞り、わかりやすく整理せよ。",
            }
            if not is_talk and _lens_preset in _LENS_INST and not _is_custom_replace:
                sp = _LENS_INST[_lens_preset] + "\n\n" + sp
            if not is_talk and _lens_hier == "prefer_summary" and not _is_custom_replace:
                sp = "【要約優先】回答は簡潔にまとめること。長文は避けよ。\n\n" + sp
            # planning mode: execution_plan_obj を system_prompt に注入
            if _mk == 'planning' and execution_plan_obj and isinstance(execution_plan_obj, dict):
                import json as _epj3
                _ep_json = _epj3.dumps(execution_plan_obj, ensure_ascii=False)
                sp += (
                    '\n\n【実行計画オブジェクト（必ずこの構造を根拠に説明せよ）】\n'
                    + _ep_json[:2000]
                    + '\n上記ExecutionPlanの各フェーズ・タスク・KPI・依存関係を根拠にして補足説明のみ行え。架空の数値・タスク追加は禁止。'
                )
            _q.put({"type":"step","label":"回答を構築中..."})
            if _is_ascend_about:
                _q.put({"type":"step","label":"ASCEND情報を構築中..."})
                msgs = history + [{"role":"user","content":req.message}]
                try:
                    reply = call_llm(system_prompt=sp, messages=msgs, ai_tier=req.ai_tier, uid=uid)
                except Exception as _e:
                    reply = f"エラー: {_e}"
                import uuid as _uuid_asc
                _q.put({"type":"done","reply":reply,"chat_id":chat_id,"msg_id":str(_uuid_asc.uuid4()),"cases":[],"images":[],"structured":None,"sources":[],"intent":{},"intent_label":"ascend"})
                return
            gi = []
            from api.core.features import is_feature_enabled as _isfe_img
            if _is_image_gen_request(req.message) and _isfe_img(uid, "image_generation") and not _is_ascend_about:
                try: reply, gi = _generate_image(req.message)
                except Exception as _e: reply = f"画像生成エラー: {_e}"
            elif _is_image_gen_request(req.message) and not _is_ascend_about:
                reply = "画像生成は現在未開放のため使用できません。"
            else:
                msgs = history + [{"role":"user","content":req.message}]
                import threading as _th_llm
                _llm_res = {}
                def _do_llm():
                    try: _llm_res["r"] = call_llm(system_prompt=sp, messages=msgs, ai_tier=req.ai_tier, uid=uid)
                    except Exception as _e: _llm_res["r"] = f"エラー: {_e}"
                _llm_t = _th_llm.Thread(target=_do_llm, daemon=True)
                _llm_t.start()
                while _llm_t.is_alive():
                    _llm_t.join(timeout=4)
                    if _llm_t.is_alive():
                        _q.put({"type":"step","label":"回答を生成中..."})
                reply = _llm_res.get("r", "エラー: LLM応答なし")

            _q.put({"type":"step","label":"回答を整形中..."})
            cases = []
            structured = None
            import threading as _th_sec
            _skip_secondary = ("429" in reply or "RESOURCE_EXHAUSTED" in reply or reply.startswith("エラー:") or len(reply) < 10)
            _struct_res = {}
            _cases_res = {}
            def _run_structured():
                try:
                    _qpi2 = qp.get('intent','')
                    _ci2 = {'相談','意思決定','分析','作成','予測','投資'}
                    _iti2 = '雑談' in _qpi2
                    _mf2 = _mk != 'auto'
                    _isc2 = (not is_talk) and (_mf2 or ((not _iti2) and (any(i in _qpi2 for i in _ci2))))
                    if not gi and _isc2:
                        # ── planning mode: execution_plan_obj から直接structured生成（reply再解釈禁止）──
                        if _mk == 'planning' and execution_plan_obj and isinstance(execution_plan_obj, dict):
                            _tasks = execution_plan_obj.get('tasks', [])
                            _plan_valid = all(
                                t.get('due_days') and t.get('owner') and t.get('owner') != '担当者未定'
                                and t.get('kpi') and t.get('kpi') != '未設定'
                                and isinstance(t.get('dependencies'), list)
                                for t in _tasks
                            ) if _tasks else False
                            if not _plan_valid:
                                print('[EXEC_PLAN] 必須4項目欠落のためstructured=None', flush=True)
                                return
                            _ep_current = [
                                f"[{t.get('phase','')}] {t.get('objective','')} / 期限:{t.get('due_days','')}日 / 担当:{t.get('owner','')} / KPI:{t.get('kpi','')}"
                                for t in _tasks[:3]
                            ]
                            _ep_risk = execution_plan_obj.get('risks', [])[:3]
                            _ep_plan = [
                                f"{t.get('objective','')} → 依存:{', '.join(t.get('dependencies',[]) or ['なし'])}"
                                for t in _tasks[:3]
                            ]
                            import json as _epj2
                            _ep_structured = {
                                'summary': execution_plan_obj.get('summary', req.message[:60]),
                                'execution_plan': {
                                    'phases': execution_plan_obj.get('phases', []),
                                    'tasks': _tasks,
                                    'dependencies': execution_plan_obj.get('dependencies', []),
                                    'blockers': execution_plan_obj.get('blockers', []),
                                    'kpis': execution_plan_obj.get('kpis', []),
                                    'checkpoints': execution_plan_obj.get('checkpoints', []),
                                    'risks': execution_plan_obj.get('risks', []),
                                    'graph': execution_plan_obj.get('graph', {}),
                                    'critical_path': execution_plan_obj.get('critical_path', []),
                                },
                                'cards': {
                                    'current': _ep_current if _ep_current else ['実行計画を生成しました'],
                                    'risk': _ep_risk if _ep_risk else ['リスク情報なし'],
                                    'plan': _ep_plan if _ep_plan else ['プラン情報なし'],
                                },
                                'analysis': {
                                    'type': '実行計画',
                                    'urgency': '高',
                                    'importance': '高',
                                    'mode': 'PLANNING',
                                },
                                'actions': [t.get('objective', '') for t in _tasks[:5] if t.get('objective')],
                                'value_message': f"実行計画: {len(_tasks)}タスク / フェーズ:{len(execution_plan_obj.get('phases',[]))}段階",
                            }
                            _struct_res['v'] = _ep_structured
                            print(f'[EXEC_PLAN] planning structured生成完了 tasks={len(_tasks)}', flush=True)
                            return
                        # ── planning以外: 既存のreply再解釈ロジック ──
                        import json as _jss2, re as _rss2
                        _mu2 = _mk.upper() if _mk != 'auto' else ''
                        _ml2 = f'modeは必ず {_mu2} で固定（変更禁止）\n' if _mu2 else 'modeは問いの内容に応じてSTRATEGY/NUMERIC/DIAGNOSIS/PLANNING/RISK/MARKETING/FINANCE/HRから選択\n'
                        _sp3 = (f'次のJSONキー構造のみで回答せよ。前置き禁止。コードブロック禁止。\n必須キー: summary, cards, analysis, actions, value_message\ncards必須キー: current(現状・背景を3件の文字列配列), risk(問題・リスクを3件の文字列配列), plan(推奨方針を3件の文字列配列)\nanalysis必須キー: type(論点タイプ1語), urgency(高/中/低), importance(高/中/低), mode(推奨モード名)\nactions: 即実行すべきアクションを3〜5件の文字列配列\n{_ml2}summary: 相談内容の本質的課題を2〜3行で具体的に記述\nvalue_message: このコンサルの価値を1行で\nQ: {req.message[:400]}\nA: {reply[:800]}')
                        _sr2 = call_llm(system_prompt='JSONのみ出力。指定キー構造厳守。', messages=[{'role':'user','content':_sp3}], ai_tier='core', max_tokens=900)
                        _m2 = _rss2.search(r'\{.*\}', _sr2, _rss2.DOTALL)
                        if _m2:
                            _p2 = _jss2.loads(_m2.group(0))
                            if all(k in _p2 for k in ['summary','cards','analysis','actions','value_message']):
                                _struct_res['v'] = _p2
                except Exception:
                    pass
            def _run_cases():
                try:
                    if not gi:
                        _cp2 = f'以下の会話に対して、ユーザーが次に相談しそうな事案を必ず3件、日本語で1行ずつ返せ。必ず3行出力すること。番号・記号・マーク不要。\nQ: {req.message}\nA: {reply[:500]}'
                        _cr2 = call_llm(system_prompt='必ず3行のみ出力。番号・記号・前置き禁止。3件未満は禁止。', messages=[{'role':'user','content':_cp2}], ai_tier='core', max_tokens=512)
                        _cases_res['v'] = [l.strip() for l in _cr2.strip().split('\n') if l.strip()][:3]
                    else:
                        pass
                except Exception as _ce2:
                    pass
            _ts2 = _th_sec.Thread(target=_run_structured, daemon=True)
            if not _skip_secondary:
                _ts2 = _th_sec.Thread(target=_run_structured, daemon=True)
                _tc2 = _th_sec.Thread(target=_run_cases, daemon=True)
                _ts2.start(); _tc2.start()
                _tc2.join(timeout=8)
                _ts2.join(timeout=6)
            elif is_talk and not gi:
                # 会話モードは_skip_secondaryでもcasesだけは必ず生成
                _tc2 = _th_sec.Thread(target=_run_cases, daemon=True)
                _tc2.start()
                _tc2.join(timeout=8)
            structured = _struct_res.get('v')
            cases = _cases_res.get('v') or []
            reply = __import__("re").sub(r" {2,}", " ", reply).strip()
            _ss_sources = []
            _RAG_THRESHOLD = 0.70  # 高確信度閾値
            if _rc and not _plan_hit:
                _ss_max = max((float(_sck.get("_score",0)) for _sck in _rc), default=0.0)
                _ss_retrieved = len(_rc) > 0
                _ss_sources=[{"text":(_sck.get("text","") or "")[:200],"score":float(_sck.get("_score",0)),"source_id":str(_sck.get("source_id","")),"is_retrieved":True} for _sck in _rc]
            elif _plan_hit:
                _ss_sources = []
            else:
                _ss_sources = [{"text": "", "score": 0.0, "source_id": "", "is_retrieved": False}]
            reply = _clean_reply(reply)
            _save_message(tenant_id, uid, chat_id, "user", req.message)
            _save_message(tenant_id, uid, chat_id, "assistant", reply, cases=cases, structured=structured, images=gi, sources=_ss_sources)
            try:
                get_db().collection("usage_logs").add({"user_id":uid,"tenant_id":tenant_id,"prompt":req.message[:200],"timestamp":(datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),"is_admin_test":False})
            except Exception:
                pass
            # intent_label: エラー時は空、正常時のみキーワード判定
            _il = ""
            if not (reply.startswith("エラー:") or "429" in reply or "RESOURCE_EXHAUSTED" in reply):
                _mq = req.message
                if any(w in _mq for w in ["投資","銘柄","株","相場","シグナル","底打ち"]):
                    _il = "投資分析"
                elif any(w in _mq for w in ["戦略","施策","差別化","競合","KPI","ROI"]):
                    _il = "戦略相談"
                elif any(w in _mq for w in ["分析","解析","調査","データ","原因"]):
                    _il = "分析"
                elif any(w in _mq for w in ["作成","生成","書いて","まとめ","作って"]):
                    _il = "作成"
                elif any(w in _mq for w in ["計画","ロードマップ","スケジュール","設計"]):
                    _il = "意思決定"
                else:
                    _il = "相談"
            # ist に Firestore Timestamp 等の非シリアライズ可能型が混入するのを防ぐ
            try:
                import json as _jsc
                _ist_safe = _jsc.loads(_jsc.dumps(ist if isinstance(ist,dict) else {}, default=str))
            except Exception:
                _ist_safe = {}
            # ASCEND辞書ヒット時: 推奨をcasesの先頭に追加
            if _ascend_dict_hit:
                _asugg = f"【ASCEND】{_ascend_dict_key}について詳しく教えて"
                if _asugg not in cases:
                    cases = [_asugg] + cases[:2]
            _q.put({"type":"done","reply":reply,"chat_id":chat_id,"msg_id":str(uuid.uuid4()),"cases":cases,"images":gi,"structured":structured,"sources":_ss_sources,"intent":_ist_safe,"intent_label":_il})
        except Exception as _e:
            _q.put({"type":"error","message":str(_e)})
        finally:
            _q.put(_DONE)
    _thm.Thread(target=_worker, daemon=True).start()
    def _gen():
        import time as _t, queue as _qmod
        while True:
            try:
                item = _q.get(timeout=3)
            except _qmod.Empty:
                yield ": keepalive\n\n"
                continue
            if item is _DONE: break
            try:
                if isinstance(item, dict) and item.get("type") == "done":
                    pass
                yield _sse_evt(item)
                if isinstance(item, dict) and item.get("type") == "done":
                    pass
            except Exception as _ge:
                print(f"[SSE_GEN_ERR] {type(_ge).__name__}: {_ge}", flush=True)
                if isinstance(item, dict) and item.get("type") == "done":
                    # done イベントが落ちた場合は最小限のfallbackを送出
                    import json as _jfb
                    _fb = _jfb.dumps({"type":"done","reply":item.get("reply",""),"chat_id":item.get("chat_id",""),"msg_id":item.get("msg_id",""),"cases":[],"images":[],"structured":None,"sources":[],"intent":{},"intent_label":item.get("intent_label","")}, ensure_ascii=False)
                    yield f"data: {_fb}\n\n"
            _t.sleep(0)
        yield ": done\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.post("/send_image_stream")
def send_image_stream(req: ImageRequest, payload: dict = Depends(verify_token)):
    def _gen():
        try:
            uid = payload["uid"]
            tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
            chat_id = (req.chat_id or "main").strip() or "main"
            _ensure_session(tenant_id, uid, chat_id)
            gi = []
            _is_ascend_about = any(k in (req.message or "") for k in ["ASCEND","とは","使い方","機能","プロファイル生成","顧客AIマネジメント","画像生成とは","プレゼン資料生成","未来分岐シミュレーター","ファイル診断","Decision Metrics","固定概念レポート","ランクシステム","比較分析","構造診断","課題仮説","矛盾検知","実行計画","思考マップ","投資シグナル","個人相談","何ができ","料金は"])
            _ascend_dict_key2, _ascend_dict_text2 = _ascend_static_answer(req.message)
            if _ascend_dict_key2:
                _is_ascend_about = True
            if req.image_b64 and not (_is_image_gen_request(req.message, has_image=True) and not _is_ascend_about):
                yield _sse_evt({"type":"step","label":"画像を解析中..."})
                base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
                sp = ("【最重要指示】あなたは画像解析AIです。添付された画像を必ず詳細に分析し、内容・テキスト・数値・構造・色・特徴を全て日本語で説明してください。\n\n"
                      + base_prompt + "\n\n【画像解析モード】添付された画像を正確に読み取り、ユーザーの質問に対して詳細に答えよ。")
                msgs = _load_history(tenant_id, uid, chat_id)
                msgs.append({"role":"user","content":req.message or "この画像を詳しく分析してください"})
                import threading as _th_ia
                _ia_res = {}
                def _do_ia():
                    try: _ia_res['r'] = call_llm(system_prompt=sp, messages=msgs, ai_tier=req.ai_tier, image_b64=req.image_b64, image_mime=req.image_mime, uid=uid)
                    except Exception as _e: _ia_res['r'] = f"画像解析エラー: {_e}"
                _ia_t = _th_ia.Thread(target=_do_ia, daemon=True); _ia_t.start()
                while _ia_t.is_alive():
                    _ia_t.join(timeout=3)
                    if _ia_t.is_alive(): yield ": ping\n\n"
                reply = _ia_res.get('r', '画像解析エラー')
            elif _is_ascend_about:
                # ASCEND質問→画像生成せずLLMで回答
                yield _sse_evt({"type":"step","label":"ASCEND情報を構築中..."})
                _bp2 = """あなたはASCENDのコンサルティングAIです。
ASCENDの機能・使い方・プラン・思想について質問された場合、以下のルールで回答せよ。
【禁止】絵文字・幼稚な口調・キャラクターAI口調（〜だよ・相棒・寄り添う等）・感情的な前置き・「ありがとうございます」等の過剰な挨拶
【必須】経営判断OS・構造解析エンジン・戦略実行支援基盤として説明せよ。具体的・簡潔・実務的に回答せよ。"""
                _adk3, _adt3 = _ascend_static_answer(req.message)
                if _adt3:
                    _bp2 = _bp2 + "\n\n【ASCEND辞書情報 - 必ずこの内容を根拠に回答せよ】\n" + _adt3
                _msgs2 = _load_history(tenant_id, uid, chat_id)
                _msgs2.append({"role":"user","content":req.message})
                try: reply = call_llm(system_prompt=_bp2, messages=_msgs2, ai_tier=req.ai_tier, uid=uid)
                except Exception as _e2: reply = f"エラー: {_e2}"
            else:
                yield _sse_evt({"type":"step","label":"画像を生成中..."})
                import threading as _th_ig
                _ig_res = {}
                def _do_ig():
                    try: _ig_res['r'], _ig_res['gi'] = _generate_image(req.message, req.image_b64, req.image_mime)
                    except Exception as _e: _ig_res['r'] = f"画像生成エラー: {_e}"; _ig_res['gi'] = []
                _ig_t = _th_ig.Thread(target=_do_ig, daemon=True); _ig_t.start()
                while _ig_t.is_alive():
                    _ig_t.join(timeout=3)
                    if _ig_t.is_alive(): yield ": ping\n\n"
                reply = _ig_res.get('r', '画像生成エラー')
                gi = _ig_res.get('gi', [])
            yield _sse_evt({"type":"step","label":"最終調整中..."})
            # GCS upload
            if gi:
                try:
                    import os as _os2, base64 as _b642
                    from google.cloud import storage as _gcs2
                    import concurrent.futures as _cff3
                    def _gcs_up():
                        try:
                            bn = _os2.environ.get("CENTRAL_BLOB_BUCKET","").strip()
                            if not bn: return
                            _gc2 = _gcs2.Client(); _bkt2 = _gc2.bucket(bn)
                            for _ii, _img in enumerate(gi):
                                _ib = _b642.b64decode(_img["data"])
                                _ext = "png" if "png" in _img.get("mime_type","") else "jpg"
                                _path = f"chat_images/{tenant_id}/{uid}/{__import__('uuid').uuid4().hex[:8]}.{_ext}"
                                _bl = _bkt2.blob(_path)
                                _bl.upload_from_string(_ib, content_type=_img.get("mime_type","image/png"), timeout=30)
                                gi[_ii]["gcs_url"] = f"https://storage.googleapis.com/{bn}/{_path}"
                        except Exception: pass
                    _exc3 = _cff3.ThreadPoolExecutor(max_workers=1)
                    try: _exc3.submit(_gcs_up).result(timeout=5)
                    except Exception: pass
                    finally: _exc3.shutdown(wait=False)
                except Exception: pass
            # Firestore save
            for _img in gi:
                try:
                    _iid = uuid.uuid4().hex
                    get_db().collection("image_gallery").document(uid).collection("images").document(_iid).set({"image_id":_iid,"uid":uid,"tenant_id":tenant_id,"gcs_url":_img.get("gcs_url",""),"mime_type":_img.get("mime_type","image/png"),"prompt":(req.message or "")[:500],"created_at":__import__("datetime").datetime.utcnow().isoformat()})
                except Exception: pass
            _save_message(tenant_id, uid, chat_id, "user", req.message)
            reply = __import__("re").sub(r" {2,}", " ", reply).strip()
            _gi_save = [{"mime_type":img.get("mime_type","image/png"),"gcs_url":img.get("gcs_url","")} for img in gi]
            _save_message(tenant_id, uid, chat_id, "assistant", reply, images=_gi_save)
            try: get_db().collection("usage_logs").add({"user_id":uid,"tenant_id":tenant_id,"prompt":req.message[:200],"timestamp":(__import__("datetime").datetime.utcnow()+__import__("datetime").timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),"is_admin_test":False})
            except Exception: pass
            if _is_image_gen_request(req.message, has_image=bool(req.image_b64)) and not _is_ascend_about and not gi:
                yield _sse_evt({"type":"done","reply":"画像生成に失敗しました。モデルが画像を返しませんでした。指示を少し変えて再試行してください。","chat_id":chat_id,"msg_id":str(uuid.uuid4()),"cases":[],"images":[],"structured":None})
            else:
                yield _sse_evt({"type":"done","reply":reply,"chat_id":chat_id,"msg_id":str(uuid.uuid4()),"cases":[],"images":gi,"structured":None})
        except Exception as _e:
            yield _sse_evt({"type":"error","message":str(_e)})
        yield ": done\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@router.post("/send_file_stream")
def send_file_stream(req: FileAnalysisRequest, payload: dict = Depends(verify_token)):
    import queue as _qm3, threading as _thm3
    _q = _qm3.Queue()
    _DONE = object()
    def _worker():
        try:
            _q.put({"type":"step","label":"ファイルを受信中..."})
            uid = payload["uid"]
            tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
            chat_id = (req.chat_id or "main").strip() or "main"
            _ensure_session(tenant_id, uid, chat_id)
            base_prompt = _load_tenant_system_prompt(tenant_id, uid=uid)
            _cc = """あなたは超一流の経営コンサルタントであり、データ分析の専門家である。
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
【出力形式】
- 結論を最初に述べ、根拠をデータから示せ
- 数値は必ず引用し、比較・変化率・傾向を明示せよ
- 実務で即使える具体的な提言を出せ"""
            sp = (_cc + ("\n\n【業種別追加指示】\n" + base_prompt if base_prompt.strip() else "") +
                  "\n\n【ファイル解析モード】添付ファイルの内容を正確に読み取り、ユーザーの質問に答えよ。数値・表・構造は必ず整理して提示せよ。")
            _q.put({"type":"step","label":"内容を解析中..."})
            fc = f"\n\n【添付ファイル: {req.filename}】\n{req.file_text[:8000]}" if req.file_text else ""
            msgs = _load_history(tenant_id, uid, chat_id)
            msgs.append({"role":"user","content":req.message + fc})
            _q.put({"type":"step","label":"構造を把握中..."})
            _q.put({"type":"step","label":"回答を構築中..."})
            if _is_ascend_about:
                _q.put({"type":"step","label":"ASCEND情報を構築中..."})
                msgs = history + [{"role":"user","content":req.message}]
                try:
                    reply = call_llm(system_prompt=sp, messages=msgs, ai_tier=req.ai_tier, uid=uid)
                except Exception as _e:
                    reply = f"エラー: {_e}"
                _q.put({"type":"done","text":reply,"rag_chunks":[],"rag_used":False})
                return
            try: reply = call_llm(system_prompt=sp, messages=msgs, ai_tier=req.ai_tier, uid=uid)
            except Exception as e: reply = f"ファイル解析エラー: {e}"
            _q.put({"type":"step","label":"最終調整中..."})
            _save_message(tenant_id, uid, chat_id, "user", req.message)
            reply = __import__("re").sub(r" {2,}", " ", reply).strip()
            _save_message(tenant_id, uid, chat_id, "assistant", reply)
            try: get_db().collection("usage_logs").add({"user_id":uid,"tenant_id":tenant_id,"prompt":req.message[:200],"timestamp":(datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),"is_admin_test":False})
            except Exception: pass
            _q.put({"type":"done","reply":reply,"chat_id":chat_id,"msg_id":str(uuid.uuid4()),"cases":[],"images":[],"structured":None})
        except Exception as _e:
            _q.put({"type":"error","message":str(_e)})
        finally:
            _q.put(_DONE)
    _thm3.Thread(target=_worker, daemon=True).start()
    def _gen():
        while True:
            item = _q.get()
            if item is _DONE: break
            yield _sse_evt(item)
    return StreamingResponse(_gen(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.get("/sources_log")
def get_sources_log(payload: dict = Depends(verify_token)):
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    prefix = f"{SCOPE}__{tenant_id}__{uid}__"
    result = []
    try:
        sessions = db.collection("chat_sessions").where("uid","==",uid).where("tenant_id","==",tenant_id).where("scope","==",SCOPE).limit(20).stream()
        for sess in sessions:
            sess_d = sess.to_dict() or {}
            if sess_d.get("is_deleted"): continue
            cid = sess_d.get("chat_id","main")
            msgs = _messages_ref(tenant_id, uid, cid).order_by("ts").limit_to_last(10).get()
            for m in msgs:
                md = m.to_dict() or {}
                if md.get("role")=="assistant" and md.get("sources"):
                    for sc in md["sources"]:
                        result.append(sc)
    except Exception as e:
        print(f"[SOURCES_LOG_ERROR] {e}", flush=True)
    return {"sources": result[:30]}
