# api/core/rag.py
# app.py の rag_retrieve_chunks / embed_text を FastAPI 用に移植
import os
import math
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

def rag_retrieve_chunks(
    tenant_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Dict]:
    """
    Firestoreの tenant_source_links → source_chunks を検索し、
    上位 top_k チャンクを返す。
    """
    db = get_db()

    # ① tenant_source_links から有効な source_id を取得
    links_ref = db.collection("tenant_source_links")
    links = [
        d.to_dict()
        for d in links_ref
        .where("tenant_id", "==", tenant_id)
        .where("enabled", "==", True)
        .stream()
    ]
    source_ids = {lnk["source_id"] for lnk in links if "source_id" in lnk}

    if not source_ids:
        return []

    # ② クエリを embed
    try:
        query_vec = embed_text(query)
    except Exception:
        return []

    # ③ source_chunks から候補を取得してスコアリング
    chunks_ref = db.collection("source_chunks")
    scored = []
    for sid in source_ids:
        for chunk_doc in chunks_ref.where("source_id", "==", sid).stream():
            chunk = chunk_doc.to_dict()
            vec = chunk.get("embedding")
            if not vec:
                continue
            score = cosine_similarity(query_vec, vec)
            if score >= threshold:
                scored.append({**chunk, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]
