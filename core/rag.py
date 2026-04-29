# api/core/rag.py
import os
import math
import struct
from typing import List, Dict
from google import genai
from google.genai import types
from api.core.firestore_client import get_db

EMBED_MODEL = "gemini-embedding-001"

def embed_text(text: str) -> List[float]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _bytes_to_vec(b: bytes) -> List[float]:
    """embedding_bytes (float32 LE) をfloatリストに変換"""
    try:
        n = len(b) // 4
        return list(struct.unpack(f"{n}f", b))
    except Exception:
        return []

def _get_source_ids(db, tenant_id: str) -> set:
    """tenant_source_links から有効な source_id を取得"""
    links = [d.to_dict() for d in db.collection("tenant_source_links").where("tenant_id", "==", tenant_id).stream()]
    links = [lnk for lnk in links if lnk.get("enabled", True)]
    return {lnk["source_id"] for lnk in links if "source_id" in lnk}

def _search_by_source_ids(db, source_ids: set, query_vec: List[float], top_k: int, threshold: float) -> List[Dict]:
    """指定source_idsのチャンクをembedding比較で検索（並列化）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    chunks_ref = db.collection("source_chunks")

    def _search_one(sid):
        results = []
        for chunk_doc in chunks_ref.where("source_id", "==", sid).stream():
            chunk = chunk_doc.to_dict()
            raw = chunk.get("embedding_bytes")
            vec = _bytes_to_vec(raw) if raw else []
            if not vec:
                continue
            score = cosine_similarity(query_vec, vec)
            if score >= threshold:
                results.append({**chunk, "_score": score})
        return results

    scored = []
    with ThreadPoolExecutor(max_workers=min(8, len(source_ids))) as ex:
        futures = {ex.submit(_search_one, sid): sid for sid in source_ids}
        for future in as_completed(futures):
            try:
                scored.extend(future.result())
            except Exception:
                pass
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]

def rag_retrieve_chunks_with_vec(
    tenant_id: str,
    query_vec: List[float],
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Dict]:
    """query_vec（計算済み）を使ってRAG検索"""
    db = get_db()
    HIGH_CONFIDENCE = 0.70
    source_ids = _get_source_ids(db, tenant_id)
    scored = _search_by_source_ids(db, source_ids, query_vec, top_k, threshold) if source_ids else []
    if scored and max(c["_score"] for c in scored) >= HIGH_CONFIDENCE:
        return scored
    if tenant_id != "default":
        default_ids = _get_source_ids(db, "default") - source_ids
        if default_ids:
            default_scored = _search_by_source_ids(db, default_ids, query_vec, top_k, threshold)
            existing_chunk_ids = {c.get("chunk_id") for c in scored}
            scored = scored + [c for c in default_scored if c.get("chunk_id") not in existing_chunk_ids]
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]

def rag_retrieve_chunks(
    tenant_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Dict]:
    """
    ① tenant_id の tenant_source_links → source_chunks を検索
    ② 結果なし → default テナントの tenant_source_links → source_chunks にフォールバック
    """
    db = get_db()

    try:
        query_vec = embed_text(query)
    except Exception:
        return []

    HIGH_CONFIDENCE = 0.70  # 専用ナレッジの確信度上限閾値

    # ① テナント専用ナレッジを検索
    source_ids = _get_source_ids(db, tenant_id)
    scored = _search_by_source_ids(db, source_ids, query_vec, top_k, threshold) if source_ids else []

    # 専用ナレッジのmax_scoreが HIGH_CONFIDENCE 以上 → 専用ナレッジで確信あり→そのまま返す
    if scored and max(c["_score"] for c in scored) >= HIGH_CONFIDENCE:
        return scored

    # 専用ナレッジが0件 or 低スコア → default（中央倉庫）も検索してスコア順に返す
    if tenant_id != "default":
        default_ids = _get_source_ids(db, "default") - source_ids
        if default_ids:
            default_scored = _search_by_source_ids(db, default_ids, query_vec, top_k, threshold)
            existing_chunk_ids = {c.get("chunk_id") for c in scored}
            scored = scored + [c for c in default_scored if c.get("chunk_id") not in existing_chunk_ids]

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]
