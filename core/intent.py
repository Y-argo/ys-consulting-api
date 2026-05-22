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

        q = (current_p or "").lower()
        if any(w in q for w in ["構造","設計","アーキテクチャ","最適化","支配","力学"]):
            stage = r4
        elif any(w in q for w in ["戦略","差別化","競合","ポジション","kpi","roi"]):
            stage = r3
        elif any(w in q for w in ["実行","手順","タスク","改善","運用","効率"]):
            stage = r2
        else:
            stage = r1
        if any(w in q for w in ["売上","収益","利益","稼ぎ","収入"]):
            desire = "収益を最大化したい"
        elif any(w in q for w in ["競合","勝ち","差別化","優位"]):
            desire = "競争優位を確立したい"
        elif any(w in q for w in ["効率","時間","コスト","削減"]):
            desire = "業務を効率化したい"
        else:
            desire = "課題を解決したい"
        if any(w in q for w in ["絶対","必ず","確実","間違いない"]):
            bias = "確証バイアス"
        elif any(w in q for w in ["不安","怖い","リスク","失敗"]):
            bias = "損失回避バイアス"
        else:
            bias = "現状維持バイアス"
        state = {
            "current_stage": stage,
            "true_desire": desire,
            "bias": bias,
            "missing_piece": f"{r4}視点での構造把握",
            "confidence": 0.7,
        }
        if state:
            db.collection("users").document(uid).set({"intent_state": state}, merge=True)
        return state
    except Exception:
        return {}

def generate_query_plan(user_prompt: str, tenant_id: str, level: str) -> dict:
    """入力から最適な検索・要約計画をキーワードベースで生成（LLM不使用）"""
    q = (user_prompt or "").lower()
    # intent判定
    intent = "相談"
    if any(w in q for w in ["分析","解析","なぜ","原因","要因","調べ"]):
        intent = "分析"
    elif any(w in q for w in ["どうすべき","どちら","選択","判断","決め","べきか"]):
        intent = "意思決定"
    elif any(w in q for w in ["まとめ","要約","整理","概要"]):
        intent = "要約"
    elif any(w in q for w in ["作成","書いて","作って","生成","作り"]):
        intent = "作成"
    elif any(w in q for w in ["こんにちは","ありがとう","おはよう","こんばん","やあ","どうも"]):
        intent = "雑談"
    # summary_lens判定
    preset = "expert"
    if any(w in q for w in ["手順","実装","チェック","運用","具体","todo"]):
        preset = "executor"
    elif any(w in q for w in ["習慣","訓練","メンタル","マインド","継続","成長"]):
        preset = "mentor"
    elif any(w in q for w in ["概要","まとめ","全体","要点","要約"]):
        preset = "general"
    # execution_type 判定
    if any(w in q for w in ["ロードマップ","計画","フェーズ","マイルストーン","スケジュール","実行計画"]):
        execution_type = "roadmap"
    elif any(w in q for w in ["戦略","施策","差別化","競合"]):
        execution_type = "strategy"
    elif any(w in q for w in ["改善","効率","運用","コスト削減"]):
        execution_type = "improvement"
    else:
        execution_type = "general"

    # risk_level 判定
    if any(w in q for w in ["リスク","失敗","損失","不安","危険"]):
        risk_level = "high"
    elif any(w in q for w in ["慎重","検討","確認","判断"]):
        risk_level = "medium"
    else:
        risk_level = "low"

    # planning_horizon 判定
    if any(w in q for w in ["長期","3年","5年","10年","将来","未来"]):
        planning_horizon = "long"
    elif any(w in q for w in ["中期","半年","1年","年間"]):
        planning_horizon = "mid"
    else:
        planning_horizon = "short"

    # required_kpis 判定
    required_kpis = []
    if any(w in q for w in ["売上","収益","利益"]):
        required_kpis.append("売上KPI")
    if any(w in q for w in ["コスト","費用","削減"]):
        required_kpis.append("コスト削減率")
    if any(w in q for w in ["顧客","獲得","CVR"]):
        required_kpis.append("顧客獲得数")
    if not required_kpis:
        required_kpis = ["目標未定"]

    # likely_blockers 判定
    likely_blockers = []
    if any(w in q for w in ["人材","採用","育成"]):
        likely_blockers.append("人材不足")
    if any(w in q for w in ["予算","資金","コスト"]):
        likely_blockers.append("予算制約")
    if any(w in q for w in ["承認","決裁","組織"]):
        likely_blockers.append("意思決定遅延")
    if not likely_blockers:
        likely_blockers = ["未特定"]

    # suggested_mode 判定
    if execution_type == "roadmap":
        suggested_mode = "planning"
    elif execution_type == "strategy":
        suggested_mode = "strategy"
    elif execution_type == "improvement":
        suggested_mode = "ops"
    else:
        suggested_mode = "auto"

    return {
        "intent": intent,
        "why": "",
        "retrieval": {"top_k_total": 20, "recency_bias": "med"},
        "summary_lens": {"preset": preset, "chars": 900},
        "output_style": {"format": "結論→根拠→打ち手→KPI", "tone": "断定"},
        "execution_type": execution_type,
        "risk_level": risk_level,
        "required_kpis": required_kpis,
        "likely_blockers": likely_blockers,
        "planning_horizon": planning_horizon,
        "dependency_required": execution_type in ("roadmap", "strategy"),
        "suggested_mode": suggested_mode,
    }

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

# ============================================================
# ExecutionPlan 構造定義 (Phase 2: Execution Graph Engine)
# ============================================================
from dataclasses import dataclass, asdict, field as dc_field
from typing import List as _List
import datetime as _dt_ep

@dataclass
class ExecutionTask:
    task_id: str = ""
    objective: str = ""
    phase: str = ""
    priority: str = "中"
    impact: str = "中"
    difficulty: str = "中"
    owner: str = "未割当"
    due_days: int = 30
    dependencies: _List[str] = dc_field(default_factory=list)  # task_id参照
    blockers: _List[str] = dc_field(default_factory=list)
    kpi: str = "KPI未設定"
    success_condition: str = ""
    priority_score: float = 0.0
    impact_score: float = 0.0
    risk_score: float = 0.0
    execution_state: str = "todo"  # todo/in_progress/blocked/done
    progress_percent: int = 0
    created_at: str = ""
    updated_at: str = ""

@dataclass
class ExecutionPlan:
    summary: str = ""
    phases: _List[str] = dc_field(default_factory=list)
    tasks: _List[ExecutionTask] = dc_field(default_factory=list)
    graph: dict = dc_field(default_factory=dict)
    critical_path: _List[str] = dc_field(default_factory=list)
    dependencies: _List[str] = dc_field(default_factory=list)
    blockers: _List[str] = dc_field(default_factory=list)
    kpis: _List[str] = dc_field(default_factory=list)
    checkpoints: _List[str] = dc_field(default_factory=list)
    risks: _List[str] = dc_field(default_factory=list)


def _calc_priority_score(task: dict) -> float:
    """priority_score = impact*0.5 + urgency*0.3 + dependency_weight*0.2"""
    _impact_map = {"高": 3.0, "中": 2.0, "低": 1.0}
    _priority_map = {"高": 3.0, "中": 2.0, "低": 1.0}
    impact = _impact_map.get(str(task.get("impact", "中")), 2.0)
    urgency = _priority_map.get(str(task.get("priority", "中")), 2.0)
    dep_weight = min(len(task.get("dependencies", [])) * 0.5, 3.0)
    return round(impact * 0.5 + urgency * 0.3 + dep_weight * 0.2, 2)


def _validate_and_fix_graph(tasks: list) -> tuple:
    """
    task graph validation:
    - 循環依存検出・自己依存除去・存在しないtask_id依存を削除
    - 戻り値: (fixed_tasks, invalid_deps)
    """
    valid_ids = {t.get("task_id", "") for t in tasks if t.get("task_id")}
    invalid_deps = []

    for task in tasks:
        tid = task.get("task_id", "")
        deps = task.get("dependencies", [])
        if not isinstance(deps, list):
            task["dependencies"] = []
            continue
        fixed_deps = []
        for d in deps:
            if d == tid:
                invalid_deps.append(f"self_dep:{tid}")
                continue
            if d not in valid_ids:
                invalid_deps.append(f"missing:{d}(in {tid})")
                continue
            fixed_deps.append(d)
        task["dependencies"] = fixed_deps

    # 循環依存検出（DFS）
    def _has_cycle(tid, visited, rec_stack, adj):
        visited.add(tid)
        rec_stack.add(tid)
        for neighbor in adj.get(tid, []):
            if neighbor not in visited:
                if _has_cycle(neighbor, visited, rec_stack, adj):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(tid)
        return False

    adj = {t.get("task_id", ""): t.get("dependencies", []) for t in tasks}
    visited, rec_stack = set(), set()
    for tid in list(valid_ids):
        if tid not in visited:
            if _has_cycle(tid, visited, rec_stack, adj):
                # 循環を含むtaskのdependenciesをクリア
                for task in tasks:
                    if task.get("task_id") == tid:
                        invalid_deps.append(f"cycle:{tid}")
                        task["dependencies"] = []

    return tasks, invalid_deps


def _build_critical_path(tasks: list) -> list:
    """最長依存チェーン（critical path）をtask_idリストで返す"""
    if not tasks:
        return []
    adj = {t.get("task_id", ""): t.get("dependencies", []) for t in tasks}
    due_map = {t.get("task_id", ""): int(t.get("due_days", 30)) for t in tasks}

    memo = {}
    def _longest(tid):
        if tid in memo:
            return memo[tid]
        deps = adj.get(tid, [])
        if not deps:
            memo[tid] = (due_map.get(tid, 30), [tid])
            return memo[tid]
        best = (0, [])
        for d in deps:
            sub = _longest(d)
            if sub[0] > best[0]:
                best = sub
        total = due_map.get(tid, 30) + best[0]
        path = best[1] + [tid]
        memo[tid] = (total, path)
        return memo[tid]

    best_path = []
    best_len = 0
    for t in tasks:
        tid = t.get("task_id", "")
        if tid:
            result = _longest(tid)
            if result[0] > best_len:
                best_len = result[0]
                best_path = result[1]
    return best_path


def build_execution_plan(user_prompt: str, tenant_id: str = "", level: str = "mixed") -> dict:
    """
    ユーザープロンプトからExecution Graph Engineを生成する。
    - task_idはtask_001形式で固定
    - dependenciesはtask_id参照
    - priority_scoreを自動計算
    - graph validation（循環依存・自己依存・存在しないtask_id）
    - critical_path計算
    - JSON parse失敗時はfallback planを返す
    """
    import json as _json, re as _re

    now_str = _dt_ep.datetime.utcnow().isoformat()

    prompt = f"""以下のユーザー入力に対して、実行計画をJSONで出力せよ。
コードブロック禁止。JSONのみ出力。前置き禁止。
必須トップキー: summary, phases, tasks, blockers, kpis, checkpoints, risks
tasksは配列（3件以内）。各taskの必須キー:
- task_id: "task_001"形式（task_001/task_002/task_003）
- objective: タスクの目的
- phase: フェーズ名
- priority: 高/中/低
- impact: 高/中/低
- difficulty: 高/中/低
- owner: 担当者名
- due_days: 整数（日数）
- dependencies: task_idの配列（例: ["task_001"]）※循環依存禁止
- blockers: 障害の文字列配列
- kpi: KPI文字列
- success_condition: 成功条件
blockers/kpis/checkpoints/risks は文字列配列
【ユーザー入力】
{user_prompt[:400]}"""

    raw = ""
    try:
        raw = call_llm(
            system_prompt="JSONのみ出力。指定キー構造厳守。前置き・コードブロック禁止。",
            messages=[{"role": "user", "content": prompt}],
            ai_tier="core",
            max_tokens=8000,
        )
    except Exception as _ce:
        print(f'[BEP_CALL_ERR] {_ce}', flush=True)

    parsed = None
    if raw:
        try:
            _raw_clean = _re.sub(r'```[a-z]*', '', raw).replace('```', '').strip()
            m = _re.search(r'\{.*\}', _raw_clean, _re.DOTALL)
            if m:
                parsed = _json.loads(m.group(0))
        except Exception as _pe:
            pass

    def _make_task(obj: dict, idx: int) -> dict:
        """taskを補完・正規化する（task_idはサーバー側で強制付番）"""
        if not isinstance(obj, dict):
            obj = {}
        tid = f"task_{idx+1:03d}"  # LLMのtask_idは無視してサーバー側で付番
        return {
            "task_id": tid,
            "objective": obj.get("objective") or user_prompt[:40],
            "phase": obj.get("phase") or f"フェーズ{idx+1}",
            "priority": obj.get("priority") or "中",
            "impact": obj.get("impact") or "中",
            "difficulty": obj.get("difficulty") or "中",
            "owner": obj.get("owner") or "未割当",
            "due_days": int(obj.get("due_days") or 30),
            "dependencies": [],  # 後でobjective→task_id変換で上書き
            "blockers": obj.get("blockers") if isinstance(obj.get("blockers"), list) else [],
            "kpi": obj.get("kpi") or "KPI未設定",
            "success_condition": obj.get("success_condition") or "",
            "execution_state": "todo",
            "progress_percent": 0,
            "created_at": now_str,
            "updated_at": now_str,
            "_raw_deps": obj.get("dependencies") if isinstance(obj.get("dependencies"), list) else [],
        }

    if parsed and isinstance(parsed, dict):
        raw_tasks = parsed.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        raw_tasks = raw_tasks[:3]  # 最大3件

        tasks = [_make_task(t, i) for i, t in enumerate(raw_tasks)]

        # objective文字列 → task_id 変換
        # LLMがdependenciesにobjective文字列を入れるため、部分一致でtask_idに変換
        _obj_to_id = {t["objective"]: t["task_id"] for t in tasks}
        for task in tasks:
            raw_deps = task.pop("_raw_deps", [])
            resolved = []
            for dep in raw_deps:
                matched = None
                for obj_str, tid in _obj_to_id.items():
                    if dep == tid:
                        matched = tid
                        break
                    if dep in obj_str or obj_str[:20] in dep:
                        matched = tid
                        break
                if matched and matched != task["task_id"]:
                    resolved.append(matched)
            task["dependencies"] = resolved

        # graph validation
        tasks, invalid_deps = _validate_and_fix_graph(tasks)

        # priority_score/impact_score/risk_score 自動計算
        _impact_map = {"高": 3.0, "中": 2.0, "低": 1.0}
        _diff_map = {"高": 3.0, "中": 2.0, "低": 1.0}
        for t in tasks:
            t["priority_score"] = _calc_priority_score(t)
            t["impact_score"] = _impact_map.get(t.get("impact", "中"), 2.0)
            t["risk_score"] = round(_diff_map.get(t.get("difficulty", "中"), 2.0) * 0.5, 2)

        # critical_path計算
        critical_path = _build_critical_path(tasks)

        # graph構造生成
        graph = {
            "nodes": [{"task_id": t["task_id"], "objective": t["objective"][:40]} for t in tasks],
            "edges": [
                {"from": dep, "to": t["task_id"]}
                for t in tasks for dep in t.get("dependencies", [])
            ],
        }


        return {
            "summary": parsed.get("summary", user_prompt[:60]),
            "phases": parsed.get("phases", []) if isinstance(parsed.get("phases"), list) else [],
            "tasks": tasks,
            "graph": graph,
            "critical_path": critical_path,
            "dependencies": parsed.get("dependencies", []) if isinstance(parsed.get("dependencies"), list) else [],
            "blockers": parsed.get("blockers", []) if isinstance(parsed.get("blockers"), list) else [],
            "kpis": parsed.get("kpis", []) if isinstance(parsed.get("kpis"), list) else [],
            "checkpoints": parsed.get("checkpoints", []) if isinstance(parsed.get("checkpoints"), list) else [],
            "risks": parsed.get("risks", []) if isinstance(parsed.get("risks"), list) else [],
            "_plan_valid": True,
            "_raw_prompt": user_prompt[:100],
        }

    # fallback graph
    fallback_task = _make_task({}, 0)
    fallback_task["priority_score"] = 2.0
    fallback_task["impact_score"] = 2.0
    fallback_task["risk_score"] = 1.0
    return {
        "summary": user_prompt[:60],
        "phases": ["フェーズ1: 現状分析", "フェーズ2: 施策立案", "フェーズ3: 実行"],
        "tasks": [fallback_task],
        "graph": {"nodes": [{"task_id": "task_001", "objective": user_prompt[:40]}], "edges": []},
        "critical_path": ["task_001"],
        "dependencies": [],
        "blockers": [],
        "kpis": ["KPI未設定"],
        "checkpoints": [],
        "risks": [],
        "_plan_valid": False,
        "_raw_prompt": user_prompt[:100],
    }


