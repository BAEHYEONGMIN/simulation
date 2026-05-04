from __future__ import annotations

import argparse
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from common import DEFAULT_TOP_K_PRINT, call_gemini, merge_contexts, print_json, retrieve_once, summarize_refs


class HyDEState(TypedDict, total=False):
    query: str
    mode: str
    print_top: int
    request_id: str
    hyde_document: str
    contexts: list[dict[str, Any]]
    elapsed_started: float
    result: dict[str, Any]


def generate_hyde_document(query: str) -> str:
    prompt = f"""
다음 질문에 답할 수 있을 법한 짧은 가상 문서 단락을 작성하세요.

주의:
- 실제 답변으로 사용하지 않고 검색 query 확장용으로만 사용합니다.
- 모르는 구체 수치나 고유명사는 꾸며내지 마세요.
- 질문에 포함된 핵심 용어를 자연스럽게 포함하세요.
- 5문장 이내로 작성하세요.

질문:
{query}
""".strip()
    return call_gemini(
        prompt,
        system_instruction="You write hypothetical retrieval documents, not final user answers.",
    )


def generate_hyde_node(state: HyDEState) -> HyDEState:
    return {"hyde_document": generate_hyde_document(state["query"]), "elapsed_started": perf_counter()}


def retrieve_node(state: HyDEState) -> HyDEState:
    request_id = state.get("request_id", "advanced-hyde-exp")
    baseline = retrieve_once(state["query"], mode=state.get("mode", "strict"), request_id=f"{request_id}-original")
    hyde_context = retrieve_once(
        state.get("hyde_document", ""),
        mode=state.get("mode", "strict"),
        request_id=f"{request_id}-doc",
    )
    return {"contexts": [baseline, hyde_context]}


def summarize_node(state: HyDEState) -> HyDEState:
    merged = merge_contexts(state.get("contexts") or [])
    result = {
        "strategy": "hyde",
        "queries": [state["query"]],
        "hydeDocument": state.get("hyde_document", ""),
        "elapsedMs": round((perf_counter() - float(state.get("elapsed_started", perf_counter()))) * 1000.0, 2),
        "sourceContextCount": merged["sourceContextCount"],
        "refs": summarize_refs(merged["references"], state.get("print_top", DEFAULT_TOP_K_PRINT)),
    }
    return {"result": result}


def build_graph():
    graph = StateGraph(HyDEState)
    graph.add_node("generate_hyde", generate_hyde_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "generate_hyde")
    graph.add_edge("generate_hyde", "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyDE retrieval LangGraph experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "creative"])
    parser.add_argument("--print-top", type=int, default=DEFAULT_TOP_K_PRINT)
    parser.add_argument("--request-id", default="advanced-hyde-exp")
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
