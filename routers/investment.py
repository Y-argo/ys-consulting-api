# api/routers/investment.py - 反発判断.py連携
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Body
from api.routers.auth import verify_token
from api.core.firestore_client import get_db, DEFAULT_TENANT
import datetime

router = APIRouter(prefix="/api/investment", tags=["investment"])

@router.get("/signals")
def get_investment_signals(payload: dict = Depends(verify_token)):
    """最新の投資シグナルを取得"""
    db = get_db()
    try:
        docs = list(db.collection("investment_signals").limit(20).stream())
        docs.sort(key=lambda d: str((d.to_dict() or {}).get("asof_date","")), reverse=True)
        docs = docs[:1]
        if not docs:
            return {"signals": None, "asof_date": None}
        doc = docs[0]
        data = doc.to_dict() or {}
        doc_ref = db.collection("investment_signals").document(doc.id)
        data["goal_bottom"]    = [d.to_dict() for d in doc_ref.collection("goal_bottom").limit(500).stream()]
        data["watch_big_sell"] = [d.to_dict() for d in doc_ref.collection("watch_big_sell").limit(500).stream()]
        data["all_stocks"]     = [d.to_dict() for d in doc_ref.collection("all_stocks").limit(2000).stream()]
        return {"signals": data, "asof_date": data.get("asof_date")}
    except Exception as e:
        return {"signals": None, "error": str(e)}

@router.post("/feedback")
def record_signal_feedback(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """銘柄評価をLGBM教師データとして記録"""
    uid = payload["uid"]
    tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
    db = get_db()
    code = body.get("code","")
    asof_date = body.get("asof_date","")
    signal_type = body.get("signal_type","goal_bottom")
    label = 1 if body.get("label") in (1, "good", True) else 0
    try:
        log_id = f"{asof_date}:{signal_type}:{code}:{uid}"
        db.collection("tenants").document(tenant_id).collection("lgbm_training_logs").document(log_id).set({
            "log_id": log_id, "code": code, "asof_date": asof_date,
            "signal_type": signal_type, "label": label,
            "uid": uid, "tenant_id": tenant_id,
            "recorded_at": datetime.datetime.now().isoformat(),
        }, merge=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}

@router.post("/push")
def push_signals(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """管理者のみ：投資シグナルをFirestoreに保存"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    db = get_db()
    asof_date = body.get("asof_date", datetime.date.today().isoformat())
    goal_bottom = body.get("goal_bottom", [])
    watch_big_sell = body.get("watch_big_sell", [])
    try:
        doc_ref = db.collection("investment_signals").document(asof_date)
        doc_ref.set({"asof_date": asof_date, "updated_at": datetime.datetime.now().isoformat()}, merge=True)
        for item in goal_bottom:
            doc_ref.collection("goal_bottom").document(str(item.get("code",""))).set(item, merge=True)
        for item in watch_big_sell:
            doc_ref.collection("watch_big_sell").document(str(item.get("code",""))).set(item, merge=True)
        return {"ok": True, "asof_date": asof_date, "goal_count": len(goal_bottom), "watch_count": len(watch_big_sell)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/push_to_rag")
def push_signals_to_rag(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """投資シグナルをRAGナレッジとして取り込む（旧push_signals_to_rag同仕様）"""
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="管理者のみ")
    from api.core.rag import embed_text
    from google.cloud import firestore as _fs8
    import datetime as _dt8
    db = get_db()
    signal_list = body.get("signal_list", [])
    asof_date   = body.get("asof_date", _dt8.date.today().isoformat())
    signal_type = body.get("signal_type", "goal_bottom")
    tenant_id   = body.get("tenant_id", "default")
    if not signal_list:
        return {"ok": False, "error": "signal_listが空"}
    try:
        lines = []
        for r in signal_list[:50]:
            code = r.get("code","")
            name = r.get("company_name","")
            close = r.get("close","")
            bottom = r.get("bottom_score",0)
            sell = r.get("sell_score",0)
            rank = r.get("rank_score",0)
            if signal_type == "goal_bottom":
                lines.append(f"銘柄{code} {name} 終値{close} bottom_score={bottom:.2f} rank_score={rank:.2f}")
            else:
                lines.append(f"銘柄{code} {name} 終値{close} sell_score={sell:.2f}")
        _hdr = "[signal:" + signal_type + " date:" + asof_date + "]"
        text = _hdr + "\n" + "\n".join(lines)
        sid = f"signal_{signal_type}_{asof_date}"
        # sources登録
        db.collection("sources").document(sid).set({
            "source_id": sid, "title": "signal:" + signal_type + ":" + asof_date,
            "content": text, "category": "investment",
            "source_type": "investment_signal",
            "created_at": _fs8.SERVER_TIMESTAMP, "updated_at": _fs8.SERVER_TIMESTAMP,
        }, merge=True)
        # tenant_source_links登録
        db.collection("tenant_source_links").document(f"{tenant_id}__{sid}").set({
            "tenant_id": tenant_id, "source_id": sid,
            "category": "investment", "enabled": True, "priority": 2,
            "updated_at": _fs8.SERVER_TIMESTAMP,
        }, merge=True)
        # チャンク化してベクトル登録
        _chunk_size = 800
        _chunks = [text[i:i+_chunk_size] for i in range(0, len(text), _chunk_size)]
        for _ci, _chunk in enumerate(_chunks[:5]):
            if not _chunk.strip(): continue
            _emb = embed_text(_chunk)
            _cid = f"{sid}_c{_ci}"
            db.collection("source_chunks").document(_cid).set({
                "chunk_id": _cid, "doc_id": sid, "source_id": sid,
                "tenant_id": tenant_id, "title": "signal:" + signal_type + ":" + asof_date,
                "text": _chunk, "embedding": _emb,
                "category": "investment", "source_type": "investment_signal",
                "chunk_index": _ci,
            }, merge=True)
        return {"ok": True, "doc_id": sid, "chunks": len(_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis")
def get_investment_analysis(payload: dict = Depends(verify_token)):
    """投資シグナル全データ+コンサル分析"""
    from api.core.llm_client import call_llm
    db = get_db()
    try:
        docs = list(db.collection("investment_signals").limit(30).stream())
        docs.sort(key=lambda d: str((d.to_dict() or {}).get("asof_date","")), reverse=True)
        if not docs:
            return {"ok": False, "error": "データなし"}
        
        # 最新3件のデータを取得
        result = []
        for doc in docs[:3]:
            data = doc.to_dict() or {}
            doc_ref = db.collection("investment_signals").document(doc.id)
            goal = [d.to_dict() for d in doc_ref.collection("goal_bottom").limit(100).stream()]
            watch = [d.to_dict() for d in doc_ref.collection("watch_big_sell").limit(100).stream()]
            data["goal_bottom"] = goal
            data["watch_big_sell"] = watch
            result.append(data)
        
        latest = result[0]
        goal_list = latest.get("goal_bottom", [])
        watch_list = latest.get("watch_big_sell", [])

        # 3日分トレンドサマリー生成
        def _trend_summary(result_list):
            lines = []
            for r in result_list:
                date = r.get("asof_date", "")
                g = sorted(r.get("goal_bottom", []), key=lambda x: float(x.get("rank_score", 0)), reverse=True)[:5]
                w = sorted(r.get("watch_big_sell", []), key=lambda x: float(x.get("sell_score", 0)), reverse=True)[:5]
                g_codes = ", ".join([f"{x.get('code')}({float(x.get('rank_score',0)):.2f})" for x in g])
                w_codes = ", ".join([f"{x.get('code')}({float(x.get('sell_score',0)):.2f})" for x in w])
                lines.append(f"[{date}] GOAL上位: {g_codes} | WATCH上位: {w_codes}")
            return "\n".join(lines)

        trend_summary = _trend_summary(result)

        # コンサル分析プロンプト
        goal_summary = "\n".join([
            f"・{r.get('code')} {r.get('company_name')} 終値{r.get('close')} "
            f"bottom={float(r.get('bottom_score',0)):.2f} rank={float(r.get('rank_score',0)):.2f} "
            f"sector={r.get('sector','')}"
            for r in sorted(goal_list, key=lambda x: float(x.get('rank_score',0)), reverse=True)[:10]
        ])
        watch_summary = "\n".join([
            f"・{r.get('code')} {r.get('company_name')} 終値{r.get('close')} "
            f"sell={float(r.get('sell_score',0)):.2f} days={r.get('sell_days',0)}"
            for r in sorted(watch_list, key=lambda x: float(x.get('sell_score',0)), reverse=True)[:10]
        ])

        prompt = (
            "あなたはファンダメンタルズとテクニカル両面に精通した戦略投資コンサルタントです。\n"
            "以下の投資シグナルデータ（3日分トレンド含む）を分析し、JSONのみ出力せよ。\n\n"
            "【共通ルール】\n"
            "- 感想ではなく根拠ある分析を返すこと\n"
            "- 数値を必ず引用すること\n"
            "- JSON以外の余計な文を絶対に返さないこと\n"
            "- 出力は必ず有効なJSONオブジェクトのみとすること\n\n"
            f"【3日間トレンド（ランク推移）】\n{trend_summary}\n\n"
            f"【最新基準日】: {latest.get('asof_date')}\n\n"
            f"【GOAL_BOTTOM（反発底打ち候補）最新上位10件】\n{goal_summary}\n\n"
            f"【WATCH_BIG_SELL（大口売り込み監視）最新上位10件】\n{watch_summary}\n\n"
            "以下のJSONスキーマで返してください:\n"
            "{\n"
            '  "market_summary": "市場全体の状況と特徴（2-3行・数値引用必須）",\n'
            '  "trend_insight": "3日間トレンドから読み取れる方向性・セクターローテーション",\n'
            '  "sector_analysis": [{"sector": "セクター名", "signal": "買い/売り/中立", "trend": "上昇/下降/横ばい", "reason": "理由"}],\n'
            '  "top_picks": [{"code": "銘柄コード", "name": "社名", "action": "買い検討/様子見/回避", "rank_trend": "上昇/下降/横ばい", "reason": "根拠", "risk": "リスク"}],\n'
            '  "risk_alerts": [{"title": "リスク名", "detail": "詳細", "severity": "high/mid/low"}],\n'
            '  "strategy": "総合戦略提言（3-4行・数値根拠必須）",\n'
            '  "next_actions": ["アクション1", "アクション2", "アクション3"]\n'
            "}"
        )

        import re as _re, json as _json, time as _time
        _cache_buster = str(int(_time.time()))
        res = call_llm(
            system_prompt=f"あなたはファンダメンタルズとテクニカル両面に精通した戦略投資コンサルタントです。必ず純粋なJSONオブジェクトのみを出力すること。```json や ``` などのマークダウン記法は絶対に使用禁止。[{_cache_buster}]",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=2048
        )
        _res_clean = _re.sub(r"```json\s*", "", res.strip())
        _res_clean = _re.sub(r"```", "", _res_clean).strip()
        m = _re.search(r"\{.*\}", _res_clean, _re.DOTALL)
        try:
            analysis = _json.loads(m.group(0)) if m else {"market_summary": res}
            # market_summaryが入れ子JSONになっている場合は再パース
            _ms = analysis.get("market_summary", "")
            if isinstance(_ms, str):
                _ms_strip = _ms.strip()
                if _ms_strip.startswith("{"):
                    try:
                        _inner = _json.loads(_ms_strip)
                        if isinstance(_inner, dict) and "market_summary" in _inner:
                            analysis = _inner
                    except Exception:
                        _m2 = _re.search(r"\{.*\}", _ms_strip, _re.DOTALL)
                        if _m2:
                            try:
                                analysis = _json.loads(_m2.group(0))
                            except Exception:
                                pass
        except Exception:
            analysis = {"market_summary": res}
        
        return {
            "ok": True,
            "latest": latest,
            "history": result,
            "analysis": analysis,
            "asof_date": latest.get("asof_date"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/stock_analysis")
def stock_analysis(body: dict = Body(...), payload: dict = Depends(verify_token)):
    """個別銘柄のコンサル分析"""
    from api.core.llm_client import call_llm
    import re as _re, json as _json
    query = body.get("query", "").strip()
    if not query:
        return {"ok": False, "error": "銘柄コードまたは社名を入力してください"}
    
    db = get_db()
    # Firestoreから該当銘柄を検索
    matched = []
    try:
        docs = list(db.collection("investment_signals").limit(30).stream())
        docs.sort(key=lambda d: str((d.to_dict() or {}).get("asof_date","")), reverse=True)
        if docs:
            doc_ref = db.collection("investment_signals").document(docs[0].id)
            asof_date = (docs[0].to_dict() or {}).get("asof_date","")
            
            # goal_bottomとwatch_big_sellを全件取得して検索
            for col in ["goal_bottom", "watch_big_sell", "all_stocks"]:
                stocks = [d.to_dict() for d in doc_ref.collection(col).limit(2000).stream()]
                for s in stocks:
                    code = str(s.get("code",""))
                    name = str(s.get("company_name",""))
                    if (query.isdigit() and query == code) or (not query.isdigit() and query.lower() in name.lower()):
                        s["_source"] = col
                        s["_asof_date"] = asof_date
                        matched.append(s)
                if matched:
                    break
    except Exception as e:
        pass

    if matched:
        r = matched[0]
        stock_info = f"""
銘柄コード: {r.get('code')}
社名: {r.get('company_name')}
セクター: {r.get('sector','')}
基準日: {r.get('asof_date','')}
終値: {r.get('close','')}
前日比: {r.get('chg','')} ({r.get('chg_pct','')}%)
本日ランク: {r.get('rank_today','')} / 前日ランク: {r.get('rank_prev','')} / 変動: {r.get('rank_diff','')}
売りスコア: {r.get('sell_score','')}
底打ちスコア: {r.get('bottom_score','')}
ランクスコア: {r.get('rank_score','')}
売り継続日数: {r.get('sell_days','')}
反発確率(1-2日): {r.get('rebound_1_2d','')}
大口売りフラグ: {r.get('big_sell_flag','')}
底打ちフラグ: {r.get('goal_flag','')}
シグナル種別: {r.get('_source','')}
"""
    else:
        stock_info = f"銘柄「{query}」のシグナルデータは見つかりませんでした。一般的な市場情報で分析します。"

    prompt = f"""以下の銘柄データを戦略投資コンサルタントとして徹底分析せよ。JSONのみ出力。confidenceはrebound_1_2dを0-100換算した値を使用し、データがない場合はbottom_scoreとsell_scoreから推定せよ。

{stock_info}

出力形式:
{{
  "code": "銘柄コード",
  "name": "社名",
  "action": "買い検討/様子見/回避",
  "confidence": rebound_1_2dの値を0-100に換算した数値（データがない場合はsell_score・bottom_scoreから推定）,
  "summary": "投資判断サマリー（2-3行）",
  "signal_analysis": {{
    "rank_trend": "ランクトレンドの評価",
    "sell_pressure": "売り圧力の分析",
    "rebound_potential": "反発可能性の評価"
  }},
  "strengths": ["強み1", "強み2"],
  "risks": ["リスク1", "リスク2"],
  "strategy": {{
    "short_term": "短期戦略（1-2週間）",
    "mid_term": "中期戦略（1-3ヶ月）",
    "entry_condition": "エントリー条件",
    "exit_condition": "エグジット条件"
  }},
  "next_actions": ["アクション1", "アクション2", "アクション3"]
}}"""

    try:
        res = call_llm(
            system_prompt="あなたはファンダメンタルズとテクニカル両面に精通した戦略投資コンサルタントです。必ず純粋なJSONオブジェクトのみを出力すること。```json や ``` などのマークダウン記法は絶対に使用禁止。",
            messages=[{"role":"user","content":prompt}],
            ai_tier="core", max_tokens=2048
        )
        _res_clean2 = _re.sub(r"```json\s*", "", res.strip())
        _res_clean2 = _re.sub(r"```", "", _res_clean2).strip()
        m = _re.search(r"\{.*\}", _res_clean2, _re.DOTALL)
        try:
            result = _json.loads(m.group(0)) if m else {"summary": res}
        except Exception:
            result = {"summary": res}
        return {"ok": True, "result": result, "raw_data": matched[0] if matched else None}
    except Exception as e:
        return {"ok": False, "error": str(e)}
