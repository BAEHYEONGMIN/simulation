from __future__ import annotations

import argparse
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from common import DEFAULT_TOP_K_PRINT, call_gemini, print_json, retrieve_once, summarize_refs


class SelfCheckState(TypedDict, total=False):
    query: str
    mode: str
    print_top: int
    request_id: str
    context: dict[str, Any]
    self_check: str
    result: dict[str, Any]


def retrieve_node(state: SelfCheckState) -> SelfCheckState:
    context = retrieve_once(
        state["query"],
        mode=state.get("mode", "strict"),
        request_id=state.get("request_id", "advanced-selfcheck-exp"),
    )
    return {"context": context}


def self_check_node(state: SelfCheckState) -> SelfCheckState:
    context = state.get("context", {})
    chunks = context.get("chunksStructured") or []
    evidence = "\n\n".join(
        f"[Doc ID: {chunk.get('doc_id')}]\n"
        f"Source: {chunk.get('source')} p.{chunk.get('page')}\n"
        f"Path: {chunk.get('path')}\n"
        f"{str(chunk.get('content') or '')[:900]}"
        for chunk in chunks[: state.get("print_top", DEFAULT_TOP_K_PRINT)]
    )
    prompt = f"""
다음 검색 결과가 질문에 답하기 충분한지 평가하세요.

출력 형식:
- sufficient: yes/no
- missing: 부족한 정보가 있으면 한 줄 요약
- suggested_queries: 추가 검색어 1~3개

질문:
{state["query"]}

검색 결과:
{evidence or "No retrieved chunks."}
""".strip()
    check_text = call_gemini(
        prompt,
        system_instruction="You evaluate retrieval evidence quality. Do not answer the original question.",
    )
    return {"self_check": check_text}


def summarize_node(state: SelfCheckState) -> SelfCheckState:
    context = state.get("context", {})
    result = {
        "strategy": "selfcheck",
        "queries": [state["query"]],
        "elapsedMs": context.get("experimentElapsedMs"),
        "selfCheck": state.get("self_check", ""),
        "refs": summarize_refs(context.get("references") or [], state.get("print_top", DEFAULT_TOP_K_PRINT)),
    }
    return {"result": result}


def build_graph():
    graph = StateGraph(SelfCheckState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("self_check", self_check_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "self_check")
    graph.add_edge("self_check", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval self-check LangGraph experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "creative"])
    parser.add_argument("--print-top", type=int, default=DEFAULT_TOP_K_PRINT)
    parser.add_argument("--request-id", default="advanced-selfcheck-exp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = build_graph()
    final_state = app.invoke(
        {
            "query": args.query,
            "mode": args.mode,
            "print_top": args.print_top,
            "request_id": args.request_id,
        }
    )
    print_json(final_state["result"])


if __name__ == "__main__":
    main()
