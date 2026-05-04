from __future__ import annotations

import argparse
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from common import DEFAULT_TOP_K_PRINT, print_json, retrieve_once, summarize_refs


class BaselineState(TypedDict, total=False):
    query: str
    mode: str
    print_top: int
    request_id: str
    context: dict[str, Any]
    result: dict[str, Any]


def retrieve_node(state: BaselineState) -> BaselineState:
    context = retrieve_once(
        state["query"],
        mode=state.get("mode", "strict"),
        request_id=state.get("request_id", "advanced-baseline-exp"),
    )
    return {"context": context}


def summarize_node(state: BaselineState) -> BaselineState:
    context = state.get("context", {})
    result = {
        "strategy": "baseline",
        "queries": [state["query"]],
        "elapsedMs": context.get("experimentElapsedMs"),
        "diagnostic": context.get("retrievalDiagnostic"),
        "refs": summarize_refs(context.get("references") or [], state.get("print_top", DEFAULT_TOP_K_PRINT)),
    }
    return {"result": result}


def build_graph():
    graph = StateGraph(BaselineState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline retrieval LangGraph experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "creative"])
    parser.add_argument("--print-top", type=int, default=DEFAULT_TOP_K_PRINT)
    parser.add_argument("--request-id", default="advanced-baseline-exp")
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
