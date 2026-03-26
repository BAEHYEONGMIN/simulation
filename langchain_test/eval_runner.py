"""
Lightweight evaluator for chat_new.py routing/retrieval.

Usage:
  python langchain_test/eval_runner.py
  python langchain_test/eval_runner.py --limit 5 --print-each
  python langchain_test/eval_runner.py --evaluate-answer-keywords
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chat_new


DEFAULT_SAMPLES_PATH = Path(__file__).with_name("eval_samples.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("reports")


@dataclass
class SampleOutcome:
    sample_id: str
    passed: bool
    expected_route: str
    actual_route: str
    route_pass: bool
    docs_expected_pass: bool
    docs_forbidden_pass: bool
    answer_keywords_pass: bool | None
    eval_mode: str
    conf_uid: str | None
    history_uid: str | None
    threshold: float
    route_scores: dict[str, float]
    selected_doc_ids: list[int]
    selected_doc_preview: list[str]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chat_new route/retrieval behavior.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES_PATH), help="Path to eval_samples.json")
    parser.add_argument("--output", default="", help="Optional explicit report output path (.json)")
    parser.add_argument("--limit", type=int, default=0, help="Run first N samples only (0 = all)")
    parser.add_argument("--mode", choices=["all", "stateless", "stateful"], default="all", help="Filter by eval_mode")
    parser.add_argument(
        "--evaluate-answer-keywords",
        action="store_true",
        help="Enable a lightweight answer-proxy keyword check.",
    )
    parser.add_argument(
        "--danger-routing",
        action="store_true",
        help="Temporarily enable danger route check during this run.",
    )
    parser.add_argument("--user-name", default="배민", help="Used by reranker stopword logic")
    parser.add_argument("--char-name", default="수아", help="Used by reranker stopword logic")
    parser.add_argument("--print-each", action="store_true", help="Print per-sample details")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def contains_keyword(text: str, keyword: str) -> bool:
    return normalize_text(keyword) in normalize_text(text)


def build_docs_text(docs: list[dict[str, Any]]) -> str:
    return "\n".join((d.get("content") or "") for d in docs)


def check_expected_keywords(text: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    return all(contains_keyword(text, kw) for kw in expected_keywords)


def check_forbidden_keywords(text: str, forbidden_keywords: list[str]) -> bool:
    if not forbidden_keywords:
        return True
    return all(not contains_keyword(text, kw) for kw in forbidden_keywords)


def now_stamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def evaluate_sample(
    sample: dict[str, Any],
    *,
    evaluate_answer_keywords: bool,
    user_name: str,
    char_name: str,
) -> SampleOutcome:
    sample_id = str(sample.get("id", "unknown"))
    eval_mode = str(sample.get("eval_mode", "stateless")).strip().lower()
    input_text = str(sample.get("input", "")).strip()
    expected_route = str(sample.get("expected_route", "")).strip().upper()
    expected_doc_keywords = sample.get("expected_doc_keywords", []) or []
    forbidden_doc_keywords = sample.get("forbidden_doc_keywords", []) or []
    expected_answer_keywords = sample.get("expected_answer_keywords", []) or []

    conf_uid = sample.get("seed_conf_uid")
    history_uid = sample.get("seed_history_uid")

    notes: list[str] = []
    threshold = 0.0
    selected_docs: list[dict[str, Any]] = []
    summary_rows_for_context: list[dict[str, Any]] = []

    if eval_mode == "stateful":
        if not conf_uid or not history_uid:
            notes.append("stateful sample missing seed_conf_uid or seed_history_uid")
            return SampleOutcome(
                sample_id=sample_id,
                passed=False,
                expected_route=expected_route,
                actual_route="INVALID_SAMPLE",
                route_pass=False,
                docs_expected_pass=False,
                docs_forbidden_pass=False,
                answer_keywords_pass=None,
                eval_mode=eval_mode,
                conf_uid=conf_uid,
                history_uid=history_uid,
                threshold=threshold,
                route_scores={},
                selected_doc_ids=[],
                selected_doc_preview=[],
                notes=notes,
            )
    else:
        conf_uid = conf_uid or "sua_test_003"
        history_uid = history_uid or None

    query_embedding = chat_new.generate_embedding(input_text)
    actual_route, route_scores = chat_new.classify_route(input_text, query_embedding)

    if actual_route == chat_new.ROUTE_KNOWLEDGE:
        vector_docs = chat_new.find_similar_documents(
            query_embedding=query_embedding,
            conf_uid=conf_uid,
            history_uid=history_uid,
            top_k=chat_new.RAG_VECTOR_TOP_K,
        )
        keyword_pool_docs = chat_new.fetch_recent_documents_for_keyword(
            conf_uid=conf_uid,
            history_uid=history_uid or "",
            limit=chat_new.RAG_KEYWORD_POOL_LIMIT,
        ) if history_uid else []

        summary_rows_for_context = (
            chat_new.fetch_recent_summaries_for_prompt(conf_uid, history_uid, chat_new.PROMPT_SUMMARY_LIMIT)
            if history_uid
            else []
        )
        excluded_summary_end_ids = {
            int(r["end_message_id"])
            for r in summary_rows_for_context
            if r.get("end_message_id") is not None
        }

        reranked = chat_new.rerank_documents(
            user_input=input_text,
            vector_docs=vector_docs,
            keyword_pool_docs=keyword_pool_docs,
            user_name=user_name,
            char_name=char_name,
            final_k=chat_new.RAG_RERANK_TOP_K,
            excluded_current_summary_end_ids=excluded_summary_end_ids,
        )

        recall_mode = chat_new.is_recall_query(input_text)
        threshold = chat_new.RAG_THRESHOLD_RECALL if recall_mode else chat_new.RAG_THRESHOLD_DEFAULT
        selected_docs = [d for d in reranked if float(d.get("rank_score", 0.0)) >= threshold][: chat_new.RAG_PROMPT_DOCS_MAX]

    docs_text = build_docs_text(selected_docs)
    summary_context_text = "\n".join(
        (row.get("summary_text") or "")
        for row in summary_rows_for_context
        if (row.get("summary_text") or "").strip()
    )
    # Stateful recollection queries in chat_new receive recent summaries directly in prompt.
    # To avoid false negatives, expected keyword checks use docs + summary context.
    expected_scope_text = docs_text
    if eval_mode == "stateful" and summary_context_text:
        expected_scope_text = docs_text + "\n" + summary_context_text
    route_pass = (actual_route == expected_route)
    docs_expected_pass = check_expected_keywords(expected_scope_text, expected_doc_keywords)
    docs_forbidden_pass = check_forbidden_keywords(docs_text, forbidden_doc_keywords)

    answer_keywords_pass: bool | None = None
    if evaluate_answer_keywords:
        proxy_text = input_text + "\n" + docs_text
        answer_keywords_pass = check_expected_keywords(proxy_text, expected_answer_keywords)

    passed = route_pass and docs_expected_pass and docs_forbidden_pass
    if answer_keywords_pass is not None:
        passed = passed and answer_keywords_pass

    if expected_route == chat_new.ROUTE_DANGER and not chat_new.ENABLE_DANGER_ROUTING:
        notes.append("danger routing is disabled in chat_new; enable with --danger-routing")

    return SampleOutcome(
        sample_id=sample_id,
        passed=passed,
        expected_route=expected_route,
        actual_route=actual_route,
        route_pass=route_pass,
        docs_expected_pass=docs_expected_pass,
        docs_forbidden_pass=docs_forbidden_pass,
        answer_keywords_pass=answer_keywords_pass,
        eval_mode=eval_mode,
        conf_uid=conf_uid,
        history_uid=history_uid,
        threshold=threshold,
        route_scores=route_scores,
        selected_doc_ids=[int(d["id"]) for d in selected_docs if d.get("id") is not None],
        selected_doc_preview=[(d.get("content") or "")[:120] for d in selected_docs[:3]],
        notes=notes,
    )


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples)
    if not samples_path.exists():
        raise FileNotFoundError(f"samples file not found: {samples_path}")

    raw = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")

    if args.mode != "all":
        samples = [s for s in samples if str(s.get("eval_mode", "stateless")).lower() == args.mode]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if args.danger_routing:
        chat_new.ENABLE_DANGER_ROUTING = True

    chat_new.initialize_router_if_needed()

    outcomes: list[SampleOutcome] = []
    for idx, sample in enumerate(samples, start=1):
        try:
            outcome = evaluate_sample(
                sample,
                evaluate_answer_keywords=args.evaluate_answer_keywords,
                user_name=args.user_name,
                char_name=args.char_name,
            )
            outcomes.append(outcome)
            if args.print_each:
                print(
                    f"[{idx:02d}] {outcome.sample_id} | pass={outcome.passed} | "
                    f"route {outcome.expected_route}->{outcome.actual_route} | "
                    f"docs_expected={outcome.docs_expected_pass} | docs_forbidden={outcome.docs_forbidden_pass}"
                )
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                SampleOutcome(
                    sample_id=str(sample.get("id", f"sample_{idx}")),
                    passed=False,
                    expected_route=str(sample.get("expected_route", "")),
                    actual_route="ERROR",
                    route_pass=False,
                    docs_expected_pass=False,
                    docs_forbidden_pass=False,
                    answer_keywords_pass=None,
                    eval_mode=str(sample.get("eval_mode", "unknown")),
                    conf_uid=sample.get("seed_conf_uid"),
                    history_uid=sample.get("seed_history_uid"),
                    threshold=0.0,
                    route_scores={},
                    selected_doc_ids=[],
                    selected_doc_preview=[],
                    notes=[f"exception: {exc}", traceback.format_exc(limit=1)],
                )
            )

    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    route_ok = sum(1 for o in outcomes if o.route_pass)
    docs_ok = sum(1 for o in outcomes if o.docs_expected_pass and o.docs_forbidden_pass)

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "samples_file": str(samples_path),
            "mode_filter": args.mode,
            "danger_routing_enabled": bool(chat_new.ENABLE_DANGER_ROUTING),
            "evaluate_answer_keywords": bool(args.evaluate_answer_keywords),
            "total_samples": total,
            "passed_samples": passed,
            "pass_rate": round((passed / total), 4) if total else 0.0,
            "route_pass_count": route_ok,
            "docs_pass_count": docs_ok,
        },
        "results": [
            {
                "sample_id": o.sample_id,
                "passed": o.passed,
                "eval_mode": o.eval_mode,
                "seed_conf_uid": o.conf_uid,
                "seed_history_uid": o.history_uid,
                "expected_route": o.expected_route,
                "actual_route": o.actual_route,
                "route_pass": o.route_pass,
                "docs_expected_pass": o.docs_expected_pass,
                "docs_forbidden_pass": o.docs_forbidden_pass,
                "answer_keywords_pass": o.answer_keywords_pass,
                "threshold": o.threshold,
                "route_scores": o.route_scores,
                "selected_doc_ids": o.selected_doc_ids,
                "selected_doc_preview": o.selected_doc_preview,
                "notes": o.notes,
            }
            for o in outcomes
        ],
    }

    if args.output:
        out_path = Path(args.output)
    else:
        DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_REPORTS_DIR / f"eval_report_{now_stamp()}.json"

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {report['meta']['pass_rate']:.2%}")
    print(f"Route pass: {route_ok}/{total}")
    print(f"Docs pass:  {docs_ok}/{total}")
    print(f"Report: {out_path}")
    print("=" * 72)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
