from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHAT_MODEL  # noqa: E402
from services.clients_genai import generate_content  # noqa: E402
from services.question_service_genai import build_ask_context  # noqa: E402

DEFAULT_TOP_K_PRINT = 10


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "\n".join(parts).strip()


def call_gemini(prompt: str, *, system_instruction: str | None = None) -> str:
    response = generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        system_instruction=system_instruction,
    )
    return response_text(response)


def parse_numbered_lines(text: str, limit: int) -> list[str]:
    rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[\).\:]?)\s*", "", line).strip()
        line = line.strip("\"'` ")
        if line and line not in rows:
            rows.append(line)
        if len(rows) >= limit:
            break
    return rows


def retrieve_once(query: str, *, mode: str = "strict", request_id: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    context = build_ask_context(
        question=query,
        mode=mode,
        session_state={},
        request_id=request_id,
    )
    context["experimentElapsedMs"] = round((perf_counter() - started) * 1000.0, 2)
    return context


def merge_contexts(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    merged_refs: dict[str, dict[str, Any]] = {}
    merged_chunks: dict[str, dict[str, Any]] = {}
    
    rrf_scores: dict[str, float] = {}
    k = 60

    for context in contexts:
        for rank, ref in enumerate(context.get("references") or [], start=1):
            row_id = str(ref.get("id"))
            if row_id not in merged_refs:
                merged_refs[row_id] = dict(ref)
                rrf_scores[row_id] = 0.0
            rrf_scores[row_id] += 1.0 / (k + rank)
            
        for chunk in context.get("chunksStructured") or []:
            doc_id = str(chunk.get("doc_id"))
            if doc_id not in merged_chunks:
                merged_chunks[doc_id] = dict(chunk)

    refs = list(merged_refs.values())
    for ref in refs:
        ref["rrf_score"] = round(rrf_scores[str(ref.get("id"))], 5)

    refs.sort(key=lambda row: row.get("rrf_score", 0.0), reverse=True)

    return {
        "references": refs,
        "chunksStructured": list(merged_chunks.values()),
        "sourceContextCount": len(contexts),
    }


def summarize_refs(refs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in refs[:limit]:
        metadata = ref.get("metadata") or {}
        rows.append(
            {
                "id": ref.get("id"),
                "rrfScore": ref.get("rrf_score"),
                "finalScore": ref.get("finalScore"),
                "semantic": ref.get("semanticScore", ref.get("similarity")),
                "lexical": ref.get("lexicalScoreUsed", ref.get("keywordScore")),
                "lexicalSource": ref.get("lexicalSource"),
                "bm25": ref.get("bm25Score"),
                "auxiliarySource": ref.get("auxiliarySource"),
                "siblingReason": ref.get("siblingReason"),
                "source": metadata.get("source"),
                "page": metadata.get("page_number") or metadata.get("page_start"),
                "path": metadata.get("path"),
                "contentPreview": str(ref.get("content") or "")[:180],
            }
        )
    return rows


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))
