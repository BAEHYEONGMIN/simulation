import json
import math
import threading
from collections.abc import Callable


_router_lock = threading.Lock()
_router_initialized = False
_router_samples: dict[str, list[str]] = {}
_router_prototypes: dict[str, list[float]] = {}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_router_samples(
    *,
    router_samples_path: str,
    route_chitchat: str,
    route_knowledge: str,
    route_danger: str,
) -> dict[str, list[str]]:
    with open(router_samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        route_chitchat: data.get("chitchat", []),
        route_knowledge: data.get("knowledge", []),
        route_danger: data.get("danger", []),
    }


def initialize_router_if_needed(
    *,
    router_samples_path: str,
    route_chitchat: str,
    route_knowledge: str,
    route_danger: str,
    generate_embedding_fn: Callable[[str], list[float]],
) -> None:
    global _router_initialized, _router_samples, _router_prototypes
    with _router_lock:
        if _router_initialized:
            return
        _router_samples = _load_router_samples(
            router_samples_path=router_samples_path,
            route_chitchat=route_chitchat,
            route_knowledge=route_knowledge,
            route_danger=route_danger,
        )
        for route_name, samples in _router_samples.items():
            if route_name == route_danger:
                # DANGER는 명시 패턴 기반 처리. 프로토타입 임베딩은 사용하지 않음.
                continue
            joined = "\n".join(samples) if samples else route_name
            _router_prototypes[route_name] = generate_embedding_fn(joined)
        _router_initialized = True


def classify_route(
    user_input: str,
    query_embedding: list[float],
    *,
    ensure_initialized_fn: Callable[[], None],
    enable_danger_routing: bool,
    danger_patterns: list[str],
    route_chitchat: str,
    route_knowledge: str,
    route_danger: str,
    router_margin: float,
    chitchat_hints: list[str],
    knowledge_hints: list[str],
) -> tuple[str, dict[str, float]]:
    if not _router_initialized:
        ensure_initialized_fn()

    text = (user_input or "").lower()

    if enable_danger_routing and any(p in text for p in danger_patterns):
        return route_danger, {
            route_chitchat: 0.0,
            route_knowledge: 0.0,
            route_danger: 1.0,
        }

    semantic_scores = {
        route: _cosine_similarity(query_embedding, proto)
        for route, proto in _router_prototypes.items()
    }

    knowledge_hit_count = sum(1 for k in knowledge_hints if k in text)
    knowledge_lex = float(knowledge_hit_count * 0.08)

    if knowledge_hit_count > 0 and ("?" in text or "!" in text):
        knowledge_lex += 0.04

    chitchat_hit_count = sum(1 for k in chitchat_hints if k in text)
    chitchat_lex = float(chitchat_hit_count * 0.08)

    if chitchat_hit_count > 0 and ("?" in text or "!" in text):
        chitchat_lex += 0.02

    scores = {
        route_chitchat: (semantic_scores.get(route_chitchat, 0.0) * 0.82) + chitchat_lex,
        route_knowledge: (semantic_scores.get(route_knowledge, 0.0) * 0.82) + knowledge_lex,
        route_danger: 0.0,
    }

    if scores[route_chitchat] >= (scores[route_knowledge] + router_margin):
        return route_chitchat, scores
    return route_knowledge, scores

