"""
src/rag_search.py 동작 점검 스크립트 (교수님 피드백 반영 후 회귀 확인용)

실행 (프로젝트 루트에서, Qdrant Docker가 켜져 있어야 함):
    uv run python tests/smoke_test.py

확인하는 것
    1. 설정: 환경 변수(QDRANT_HOST/PORT, RAG_DEVICE, RAG_USE_FP16)가 반영되는가
    2. 필터: bool mask로 BM25 필터가 정확히 걸리는가 (조건 밖 청크 0개)
             예전 float32 mask와 결과가 같은가
    3. 회귀: 04 노트북에 캐시된 검색 결과(outputs/rag_answers.json)와 top-5가 얼마나 같은가
    4. 캐시: clear_caches() 후 비워지고, 다시 검색하면 정상 동작하는가
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import rag_search as rs  # noqa: E402


def check(cond: bool, msg: str) -> None:
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        raise SystemExit(1)


def main(min_overlap: float = 0.8) -> None:
    print("① 설정")
    cfg = rs.describe_config()
    print("   ", cfg)
    if "RAG_USE_FP16" not in os.environ:
        check(cfg["use_fp16"] == (cfg["device"] == "cuda"), "fp16은 CUDA에서만 기본으로 켜짐")

    print("\n② BM25 필터 (bool mask)")
    chunks = rs.load_chunks()
    for category, year in [("반도체", None), (None, 2026), ("AI동향", 2026)]:
        ids = rs.search("HBM 반도체 시장", k=20, mode="bm25", category=category, year_gte=year)
        bad = [i for i in ids if (category and chunks[i]["category"] != category) or (year and chunks[i]["year"] < year)]
        check(len(ids) > 0 and not bad, f"category={category}, year>={year}: {len(ids)}개, 조건 밖 {len(bad)}개")

    cond = np.array([c["category"] == "반도체" for c in chunks])
    by_bool = rs.bm25_search("퓨리오사AI NPU", k=20, mask=cond.astype(bool))
    by_float = rs.bm25_search("퓨리오사AI NPU", k=20, mask=cond.astype(np.float32))
    check(by_bool == by_float, "bool mask와 예전 float32 mask의 결과가 같음")

    print("\n③ 회귀 (04 캐시 결과와 top-5 비교)")
    cached = json.loads((ROOT / "outputs" / "rag_answers.json").read_text(encoding="utf-8"))
    overlaps = {}
    t0 = time.perf_counter()
    for key, row in cached.items():
        new = rs.search(row["question"], k=5, mode=row["mode"])
        overlaps[key] = len(set(new) & set(row["chunk_ids"])) / len(row["chunk_ids"])
    elapsed = time.perf_counter() - t0
    by_mode = {}
    for key, ov in overlaps.items():
        by_mode.setdefault(key.split("|")[0], []).append(ov)
    for mode, vals in by_mode.items():
        print(f"    {mode:7s} top-5 평균 일치율 {np.mean(vals):.2f}  (최저 {min(vals):.2f})")
    print(f"    {len(overlaps)}회 검색 {elapsed:.1f}초")
    check(np.mean(list(overlaps.values())) >= min_overlap, f"전체 평균 일치율 ≥ {min_overlap}")

    print("\n④ 캐시 관리")
    before = rs.cache_status()
    rs.clear_caches()
    after = rs.cache_status()
    print("    비우기 전", before)
    print("    비운 후  ", after)
    check(all(v == 0 for v in after.values()), "clear_caches() 후 모든 캐시가 비워짐")
    ids = rs.search("HBM이란 무엇인가?", k=5, mode="rerank")
    check(len(ids) == 5, f"비운 뒤 다시 검색 정상 (rerank top-5: {ids})")

    print("\n🎉 모든 점검 통과")


if __name__ == "__main__":
    main()
