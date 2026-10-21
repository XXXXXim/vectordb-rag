"""
검색 모듈: 03_hybrid_rerank.ipynb 의 검색 함수들을 모듈로 분리한 것

    from src.rag_search import search, get_chunk
    ids = search("HBM이란 무엇인가?", k=5, mode="rerank")   # chunk_id 목록 (1등부터)
    get_chunk(ids[0])["text"]

mode: "dense" | "bm25" | "hybrid" | "rerank"  (03과 같은 파라미터: 후보 50개, RRF k=60, Rerank 후보 20개)

- 청크는 outputs/chunks.jsonl, 청크 벡터는 outputs/embeddings.npy 를 그대로 재사용한다 (다시 임베딩하지 않음)
- Qdrant 컬렉션이 이미 있고 포인트 수가 청크 수와 같으면 재사용, 아니면 02 방식대로 새로 만든다
- 모델(BGE-M3, Reranker)은 처음 필요할 때 한 번만 불러온다 (import만 할 때는 가볍게)
"""
import logging

# ⚠️ import 순서 (macOS): torch(FlagEmbedding)를 bm25s·faiss보다 먼저 불러와야 OpenMP 충돌로 죽지 않는다
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker

import json
import re
from functools import lru_cache
from pathlib import Path

import bm25s
import numpy as np
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/ 의 한 칸 위
OUT_DIR = PROJECT_ROOT / "outputs"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
EMB_PATH = OUT_DIR / "embeddings.npy"

COLLECTION = "policy_reports"
PAYLOAD_KEYS = ["source", "publisher", "category", "year", "page"]

CANDIDATES = 50   # Hybrid에서 Dense·BM25 각각 뽑는 후보 수
RRF_K = 60        # RRF 완충 상수
RERANK_TOP = 20   # Reranker가 다시 채점할 후보 수

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# 데이터 · 모델 · DB (처음 호출될 때 한 번만 만들고 이후엔 재사용 = lru_cache)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    assert all(c["chunk_id"] == i for i, c in enumerate(chunks)), "chunk_id 순서가 어긋남! 01 노트북을 다시 실행하세요."
    return chunks


def get_chunk(chunk_id: int) -> dict:
    return load_chunks()[chunk_id]


@lru_cache(maxsize=None)
def load_embed_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", devices=DEVICE, use_fp16=(DEVICE != "cpu"), normalize_embeddings=True)


@lru_cache(maxsize=None)
def load_reranker() -> FlagReranker:
    return FlagReranker("BAAI/bge-reranker-v2-m3", devices=DEVICE, use_fp16=(DEVICE != "cpu"), max_length=512)


@lru_cache(maxsize=None)
def get_qdrant() -> QdrantClient:
    """Qdrant에 연결하고, 컬렉션이 없거나 개수가 다르면 embeddings.npy로 02 방식대로 다시 만든다."""
    client = QdrantClient(host="localhost", port=6333, timeout=60, check_compatibility=False)
    chunks = load_chunks()

    if client.collection_exists(COLLECTION) and client.count(COLLECTION, exact=True).count == len(chunks):
        return client   # 재사용

    embeddings = np.load(EMB_PATH)
    assert embeddings.shape[0] == len(chunks), "embeddings.npy 개수 ≠ 청크 수 → 02 노트북을 다시 실행하세요."

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=c["chunk_id"],
            vector=embeddings[c["chunk_id"]].tolist(),
            payload={**{k: c[k] for k in PAYLOAD_KEYS}, "chunk_id": c["chunk_id"], "text": c["text"]},
        )
        for c in chunks
    ]
    client.upload_points(collection_name=COLLECTION, points=points, batch_size=256, wait=True)
    for field, schema in [
        ("category", models.PayloadSchemaType.KEYWORD),
        ("publisher", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema, wait=True)
    return client


# ─────────────────────────────────────────────────────────────
# BM25 (03의 한국어 혼합 토크나이저: 한글/영문 분리 + 한글 bigram)
# ─────────────────────────────────────────────────────────────
def tokenize_korean(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[가-힣]+|[a-z0-9]+", text.lower()):
        if "가" <= word[0] <= "힣" and len(word) >= 2:
            tokens += [word[i:i + 2] for i in range(len(word) - 1)]
        else:
            tokens.append(word)
    return tokens


@lru_cache(maxsize=None)
def load_bm25() -> bm25s.BM25:
    retriever = bm25s.BM25()
    retriever.index([tokenize_korean(c["text"]) for c in load_chunks()], show_progress=False)
    return retriever


# ─────────────────────────────────────────────────────────────
# 검색 함수들 (모두 chunk_id 목록을 1등부터 돌려준다)
# ─────────────────────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    return load_embed_model().encode([text], max_length=512)["dense_vecs"][0].tolist()


def dense_search(query: str, k: int = 50, query_filter: models.Filter | None = None) -> list[int]:
    response = get_qdrant().query_points(
        collection_name=COLLECTION, query=embed_query(query), query_filter=query_filter, limit=k, with_payload=False,
    )
    return [int(p.id) for p in response.points]


def bm25_search(query: str, k: int = 50, mask: np.ndarray | None = None) -> list[int]:
    ids, scores = load_bm25().retrieve([tokenize_korean(query)], k=k, show_progress=False, weight_mask=mask)
    return [int(i) for i, s in zip(ids[0], scores[0]) if s > 0]


def rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    scores = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(query: str, k: int = 50, query_filter=None, mask=None) -> list[int]:
    dense_ids = dense_search(query, k=CANDIDATES, query_filter=query_filter)
    bm25_ids = bm25_search(query, k=CANDIDATES, mask=mask)
    return rrf_fuse([dense_ids, bm25_ids])[:k]


def rerank(query: str, candidate_ids: list[int]) -> tuple[list[int], list[float]]:
    chunks = load_chunks()
    scores = load_reranker().compute_score([(query, chunks[cid]["text"]) for cid in candidate_ids], normalize=True)
    order = np.argsort(scores)[::-1]
    return [candidate_ids[i] for i in order], [float(scores[i]) for i in order]


def hybrid_rerank_search(query: str, k: int = 10, query_filter=None, mask=None, top_n: int = RERANK_TOP) -> list[int]:
    candidates = hybrid_search(query, k=top_n, query_filter=query_filter, mask=mask)
    ids, _ = rerank(query, candidates)
    return ids[:k]


def search(query: str, k: int = 5, mode: str = "rerank", category: str | None = None, year_gte: int | None = None) -> list[int]:
    """최종 검색 함수 (03 Step 12-2와 같음). 필터는 Dense(Qdrant)·BM25(mask) 양쪽에 똑같이 적용."""
    conditions = []
    if category:
        conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if year_gte:
        conditions.append(models.FieldCondition(key="year", range=models.Range(gte=year_gte)))
    query_filter = models.Filter(must=conditions) if conditions else None

    mask = None
    if conditions:
        mask = np.array([
            (category is None or c["category"] == category) and (year_gte is None or c["year"] >= year_gte)
            for c in load_chunks()
        ], dtype=np.float32)

    if mode == "dense":
        return dense_search(query, k=k, query_filter=query_filter)
    if mode == "bm25":
        return bm25_search(query, k=k, mask=mask)
    if mode == "hybrid":
        return hybrid_search(query, k=k, query_filter=query_filter, mask=mask)
    if mode == "rerank":
        return hybrid_rerank_search(query, k=k, query_filter=query_filter, mask=mask)
    raise ValueError(f"알 수 없는 mode: {mode}")
