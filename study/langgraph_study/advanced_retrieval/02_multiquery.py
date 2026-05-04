from __future__ import annotations

import argparse
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from common import (
    DEFAULT_TOP_K_PRINT,
    call_gemini,
    merge_contexts,
    parse_numbered_lines,
    print_json,
    retrieve_once,
    summarize_refs,
)


class MultiQueryState(TypedDict, total=False):
    query: str
    mode: str
    print_top: int
    request_id: str
    multiquery_count: int
    queries: list[str]
    contexts: list[dict[str, Any]]
    elapsed_started: float
    result: dict[str, Any]


def generate_multiqueries(query: str, count: int) -> list[str]:
    prompt = f"""
다음 RAG 검색 질문을 서로 다른 검색 표현 {count}개로 바꿔주세요.

조건:
- 원래 질문의 의미를 바꾸지 마세요.
- 한국어 표현, 핵심 명사 표현, 필요하면 영문/약어 표현을 섞어도 됩니다.
- 답변하지 말고 검색 질의만 한 줄에 하나씩 출력하세요.

원 질문:
{query}
""".strip()
    text = call_gemini(
        prompt,
        system_instruction="You generate concise search queries for retrieval experiments.",
    )
    return parse_numbered_lines(text, count) or [query]


def expand_query_node(state: MultiQueryState) -> MultiQueryState:
    count = int(state.get("multiquery_count", 3))
    generated = generate_multiqueries(state["query"], count)
    queries = [state["query"], *[item for item in generated if item != state["query"]]]
    return {"queries": queries, "elapsed_started": perf_counter()}


def retrieve_node(state: MultiQueryState) -> MultiQueryState:
    request_id = state.get("request_id", "advanced-multiquery-exp")
    contexts = [
        retrieve_once(item, mode=state.get("mode", "strict"), request_id=f"{request_id}-{index}")
        for index, item in enumerate(state.get("queries") or [state["query"]], start=1)
    ]
    return {"contexts": contexts}


def summarize_node(state: MultiQueryState) -> MultiQueryState:
    merged = merge_contexts(state.get("contexts") or [])
    result = {
        "strategy": "multiquery",
        "queries": state.get("queries") or [state["query"]],
        "elapsedMs": round((perf_counter() - float(state.get("elapsed_started", perf_counter()))) * 1000.0, 2),
        "sourceContextCount": merged["sourceContextCount"],
        "refs": summarize_refs(merged["references"], state.get("print_top", DEFAULT_TOP_K_PRINT)),
    }
    return {"result": result}


def build_graph():
    graph = StateGraph(MultiQueryState)
    graph.add_node("expand_query", expand_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "expand_query")
    graph.add_edge("expand_query", "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiQuery retrieval LangGraph experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "creative"])
    parser.add_argument("--multiquery-count", type=int, default=3)
    parser.add_argument("--print-top", type=int, default=DEFAULT_TOP_K_PRINT)
    parser.add_argument("--request-id", default="advanced-multiquery-exp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = build_graph()
    final_state = app.invoke(
        {
            "query": args.query,
            "mode": args.mode,
            "multiquery_count": args.multiquery_count,
            "print_top": args.print_top,
            "request_id": args.request_id,
        }
    )
    print_json(final_state["result"])


if __name__ == "__main__":
    main()
