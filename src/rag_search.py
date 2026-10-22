"""
검색 모듈: 03_hybrid_rerank.ipynb 의 검색 함수들을 모듈로 분리한 것

    from src.rag_search import search, get_chunk
    ids = search("HBM이란 무엇인가?", k=5, mode="rerank")   # chunk_id 목록 (1등부터)
    get_chunk(ids[0])["text"]

mode: "dense" | "bm25" | "hybrid" | "rerank"  (03과 같은 파라미터: 후보 50개, RRF k=60, Rerank 후보 20개)

- 청크는 outputs/chunks.jsonl, 청크 벡터는 outputs/embeddings.npy 를 그대로 재사용한다 (다시 임베딩하지 않음)
- Qdrant 컬렉션이 이미 있고 포인트 수가 청크 수와 같으면 재사용, 아니면 02 방식대로 새로 만든다
- 모델(BGE-M3, Reranker)은 처음 필요할 때 한 번만 불러온다 (import만 할 때는 가볍게)

환경 변수 (모두 선택. 없으면 기본값으로 동작, 프로젝트 루트 .env.example 참고)
    QDRANT_HOST    Qdrant 서버 주소          (기본 localhost)
    QDRANT_PORT    Qdrant 서버 포트          (기본 6333)
    RAG_DEVICE     cuda / mps / cpu 강제 지정 (기본: 자동 감지)
    RAG_USE_FP16   1 / 0 으로 fp16 강제 지정  (기본: CUDA에서만 1)

캐시 관리
    clear_caches()  청크·BM25·Qdrant 클라이언트·모델 캐시를 비우고 메모리를 돌려준다.
                    chunks.jsonl / embeddings.npy 를 다시 만든 뒤, 또는 오래 띄워 둔 서버에서 메모리를 비울 때 호출.
"""
import logging

# ⚠️ import 순서 (macOS): torch(FlagEmbedding)를 bm25s·faiss보다 먼저 불러와야 OpenMP 충돌로 죽지 않는다
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker

import gc
import json
import os
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

# ─────────────────────────────────────────────────────────────
# 실행 환경 설정 (교수님 피드백 2·3 반영)
# ─────────────────────────────────────────────────────────────
# [피드백 2] Qdrant 접속 정보를 코드에 박아두지 않고 환경 변수로 받는다.
#   로컬 Docker → 기본값(localhost:6333) 그대로
#   다른 서버/컨테이너로 배포 → 코드 수정 없이 QDRANT_HOST, QDRANT_PORT만 바꾸면 된다.
try:  # 프로젝트 루트의 .env 도 읽는다 (이미 설정된 환경 변수가 있으면 그쪽이 우선)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def _detect_device() -> str:
    """RAG_DEVICE가 있으면 그대로 쓰고, 없으면 cuda → mps → cpu 순서로 자동 선택."""
    forced = os.getenv("RAG_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _env_flag(name: str) -> bool | None:
    """환경 변수 "1"/"true" → True, "0"/"false" → False, 없으면 None."""
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes")


DEVICE = _detect_device()

# [피드백 3] fp16(절반 정밀도)은 CUDA에서만 기본으로 켠다.
#   MPS(Apple Silicon)·CPU에서는 PyTorch 커널이 fp16을 일부 지원하지 않아 예외가 날 수 있어서 fp32가 기본.
#   (이 프로젝트의 M5 MPS에서는 fp16도 동작했으므로, 속도가 필요하면 RAG_USE_FP16=1 로 직접 켤 수 있게 열어 둠)
_fp16_override = _env_flag("RAG_USE_FP16")
USE_FP16 = (DEVICE == "cuda") if _fp16_override is None else _fp16_override


def describe_config() -> dict:
    """지금 어떤 설정으로 돌고 있는지 확인용 (노트북에서 print 해보기)."""
    return {"device": DEVICE, "use_fp16": USE_FP16, "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}", "collection": COLLECTION}


# ─────────────────────────────────────────────────────────────
# 데이터 · 모델 · DB (처음 호출될 때 한 번만 만들고 이후엔 재사용 = lru_cache)
#   인자가 없는 함수라 저장할 결과는 항상 1개 → maxsize=1
#   비울 때는 아래 clear_caches() 사용
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    assert all(c["chunk_id"] == i for i, c in enumerate(chunks)), "chunk_id 순서가 어긋남! 01 노트북을 다시 실행하세요."
    return chunks


def get_chunk(chunk_id: int) -> dict:
    return load_chunks()[chunk_id]


@lru_cache(maxsize=1)
def load_embed_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", devices=DEVICE, use_fp16=USE_FP16, normalize_embeddings=True)


@lru_cache(maxsize=1)
def load_reranker() -> FlagReranker:
    return FlagReranker("BAAI/bge-reranker-v2-m3", devices=DEVICE, use_fp16=USE_FP16, max_length=512)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Qdrant에 연결하고, 컬렉션이 없거나 개수가 다르면 embeddings.npy로 02 방식대로 다시 만든다."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60, check_compatibility=False)
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


@lru_cache(maxsize=1)
def load_bm25() -> bm25s.BM25:
    retriever = bm25s.BM25()
    retriever.index([tokenize_korean(c["text"]) for c in load_chunks()], show_progress=False)
    return retriever


# ─────────────────────────────────────────────────────────────
# 캐시 관리 (교수님 피드백 4 반영)
# ─────────────────────────────────────────────────────────────
_DATA_CACHES = (load_chunks, load_bm25, get_qdrant)       # 데이터 파일이 바뀌면 다시 만들어야 하는 것
_MODEL_CACHES = (load_embed_model, load_reranker)        # 메모리를 가장 많이 차지하는 것 (각 2GB 안팎)


def clear_caches(data: bool = True, models: bool = True) -> None:
    """lru_cache로 잡아 둔 객체를 비우고 메모리를 돌려준다.

    - data=True  : chunks.jsonl / embeddings.npy를 다시 만든 뒤 호출 → 다음 검색 때 새 데이터로 다시 로딩
                   (청크 수가 바뀌었으면 get_qdrant()가 컬렉션도 다시 만든다)
    - models=True: 모델 2개(BGE-M3, Reranker)를 메모리에서 내린다 → 다음 검색 때 다시 로딩(수 초)
    """
    if data and get_qdrant.cache_info().currsize:
        get_qdrant().close()                 # 열려 있는 Qdrant 연결부터 정리
    targets = (_DATA_CACHES if data else ()) + (_MODEL_CACHES if models else ())
    for fn in targets:
        fn.cache_clear()
    gc.collect()                             # 파이썬이 들고 있던 객체 정리
    if models:                               # GPU/MPS 메모리도 돌려준다
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def cache_status() -> dict:
    """어떤 캐시가 지금 메모리에 올라와 있는지 확인용 (1 = 로딩됨, 0 = 비어 있음)."""
    return {fn.__name__: fn.cache_info().currsize for fn in _DATA_CACHES + _MODEL_CACHES}


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
    """mask: 청크 수와 길이가 같은 bool 배열 (True = 검색 대상). 조건 밖 청크는 점수 0이 되어 아래에서 빠진다."""
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

    # [피드백 1] BM25 필터 mask는 "넣을지/뺄지"를 나타내는 값이라 bool 배열로 만든다.
    #   (float32 0/1도 곱셈으로는 같게 동작하지만, 의미가 분명하고 타입 불일치로 필터가 무시될 여지를 없앤다)
    mask = None
    if conditions:
        mask = np.array([
            (category is None or c["category"] == category) and (year_gte is None or c["year"] >= year_gte)
            for c in load_chunks()
        ], dtype=bool)

    if mode == "dense":
        return dense_search(query, k=k, query_filter=query_filter)
    if mode == "bm25":
        return bm25_search(query, k=k, mask=mask)
    if mode == "hybrid":
        return hybrid_search(query, k=k, query_filter=query_filter, mask=mask)
    if mode == "rerank":
        return hybrid_rerank_search(query, k=k, query_filter=query_filter, mask=mask)
    raise ValueError(f"알 수 없는 mode: {mode}")
