"""
chat.py — RAG 챗봇 (완성형)
input_version.py (저장/임베딩/검색) + test.py (프롬프트/LLM 호출) 통합

흐름:
  사용자 입력
    → 임베딩 생성 (gemini-embedding-001)
    → 유사 문서 검색 (documents_gemini 벡터 검색)
    → 최근 대화 이력 조회 (chat_messages)
    → 프롬프트 조립 (ChatPromptTemplate)
    → LLM 답변 생성 (ChatGoogleGenerativeAI)
    → chat_messages에 사용자/AI 메시지 저장
    → documents_gemini에 사용자 임베딩 저장

실행:
  py -3.12 chat.py
"""

import os
import sys
import re
import asyncio
import threading
from uuid import uuid4
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supabase import create_client
from google import genai as google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from config import (
    SUPABASE_URL, SUPABASE_KEY,
    GEMINI_API_KEY_FREE,
    CHAT_MODEL, EMBEDDING_MODEL,SUMMARY_MODEL,TEST_MODEL
)
try:
    from chat_modules.prompting import (
        RESPONSE_POLICY,
        CHARACTER_PERSONA,
        SYSTEM_PROMPT_TEMPLATE,
        SUMMARY_PROMPT_TEMPLATE,
        get_mode_instruction,
        build_memory_extraction_prompt,
    )
    from chat_modules.formatting import (
        format_recent_summaries as fmt_recent_summaries,
        format_long_memories as fmt_long_memories,
        to_langchain_history_messages as fmt_to_lc_history,
        format_documents as fmt_documents,
        format_history_messages_debug as fmt_history_debug,
        format_messages_for_summary as fmt_messages_for_summary,
        normalize_summary_text as fmt_normalize_summary,
    )
    from chat_modules.routing import (
        initialize_router_if_needed as routing_initialize_router_if_needed,
        classify_route as routing_classify_route,
    )
    from chat_modules.memory_extractor import (
        queue_memory_extraction_job as memory_queue_memory_extraction_job,
        get_memory_progress as memory_get_memory_progress,
        trigger_memory_extraction_if_needed as memory_trigger_memory_extraction_if_needed,
    )
    from db.chat_history import (
        fetch_recent_messages,
        insert_message,
        fetch_messages_since,
        fetch_user_messages_since,
        fetch_bridge_messages,
    )
    from db.document_store import (
        find_similar_documents,
        insert_document,
    )
    from db.memory_store import (
        fetch_active_memories,
        upsert_user_memory,
        get_last_processed_user_id_from_memories,
    )
    from db.summary_store import (
        fetch_latest_summary,
        fetch_recent_summaries_for_prompt,
        insert_summary,
    )
    from chat_constants import (
        DEBUG, LOG_DIR, ROUTER_SAMPLES_PATH,
        ENABLE_PROVIDER_CONTEXT_CACHE, CONTEXT_CACHE_MIN_TOKENS,
        SCHEMA, TABLE_CHAT, TABLE_SUMMARIES, TABLE_DOCUMENTS, TABLE_MEMORIES, RPC_MATCH,
        SUMMARY_TRIGGER_COUNT, SUMMARY_BRIDGE_COUNT, BACKGROUND_MAX_BLOCKS_PER_RUN,
        RAG_VECTOR_TOP_K, RAG_RERANK_TOP_K, RAG_PROMPT_DOCS_MAX, RAG_KEYWORD_POOL_LIMIT,
        RAG_THRESHOLD_RECALL, RAG_THRESHOLD_DEFAULT,
        PROMPT_HISTORY_LIMIT, PROMPT_SUMMARY_LIMIT, PROMPT_MEMORY_LIMIT, FETCH_HISTORY_MULTIPLIER,
        MEMORY_TRIGGER_USER_TURNS, MEMORY_CONTEXT_MESSAGES,
        ROUTE_CHITCHAT, ROUTE_KNOWLEDGE, ROUTE_DANGER, ENABLE_DANGER_ROUTING, ROUTER_MARGIN,
        CHITCHAT_HINTS, KNOWLEDGE_HINTS, DANGER_PATTERNS, MIN_STORE_LENGTH_FOR_CHITCHAT,
        STORE_FACT_HINTS, INPUT_MODE_NORMAL, INPUT_MODE_OOC, INPUT_MODE_IC, INPUT_MODE_MIXED,
        OOC_IC_PATTERN_STR, OOC_IC_SEGMENT_PATTERN_STR, DANGER_FALLBACK_RESPONSE,
        ALLOWED_MEMORY_TYPES, ALLOWED_MEMORY_KEYS, DEFAULT_IMPORTANCE_BY_TYPE, MEMORY_NORMALIZATION_MAP,
        RECALL_HINT_PATTERNS, KEYWORD_STOPWORDS, KEYWORD_SYNONYMS, TECH_KEYWORDS,
        TECH_KEYWORD_BONUS, TECH_QUERY_NOISE_PENALTY, LIFESTYLE_KEYWORDS,
        ONE_CHAR_KEYWORDS, JOSA_SUFFIXES, JOSA_SUFFIXES_SORTED,
        NOISE_EXACT_PATTERNS, NOISE_PREFIX_PATTERNS, MEMORY_SIGNAL_PATTERNS,
    )
except ImportError:
    from langchain_test.chat_modules.prompting import (
        RESPONSE_POLICY,
        CHARACTER_PERSONA,
        SYSTEM_PROMPT_TEMPLATE,
        SUMMARY_PROMPT_TEMPLATE,
        get_mode_instruction,
        build_memory_extraction_prompt,
    )
    from langchain_test.chat_modules.formatting import (
        format_recent_summaries as fmt_recent_summaries,
        format_long_memories as fmt_long_memories,
        to_langchain_history_messages as fmt_to_lc_history,
        format_documents as fmt_documents,
        format_history_messages_debug as fmt_history_debug,
        format_messages_for_summary as fmt_messages_for_summary,
        normalize_summary_text as fmt_normalize_summary,
    )
    from langchain_test.chat_modules.routing import (
        initialize_router_if_needed as routing_initialize_router_if_needed,
        classify_route as routing_classify_route,
    )
    from langchain_test.chat_modules.memory_extractor import (
        queue_memory_extraction_job as memory_queue_memory_extraction_job,
        get_memory_progress as memory_get_memory_progress,
        trigger_memory_extraction_if_needed as memory_trigger_memory_extraction_if_needed,
    )
    from langchain_test.db.chat_history import (
        fetch_recent_messages,
        insert_message,
        fetch_messages_since,
        fetch_user_messages_since,
        fetch_bridge_messages,
    )
    from langchain_test.db.document_store import (
        find_similar_documents,
        insert_document,
    )
    from langchain_test.db.memory_store import (
        fetch_active_memories,
        upsert_user_memory,
        get_last_processed_user_id_from_memories,
    )
    from langchain_test.db.summary_store import (
        fetch_latest_summary,
        fetch_recent_summaries_for_prompt,
        insert_summary,
    )
    from langchain_test.chat_constants import (
    DEBUG, LOG_DIR, ROUTER_SAMPLES_PATH,
    ENABLE_PROVIDER_CONTEXT_CACHE, CONTEXT_CACHE_MIN_TOKENS,
    SCHEMA, TABLE_CHAT, TABLE_SUMMARIES, TABLE_DOCUMENTS, TABLE_MEMORIES, RPC_MATCH,
    SUMMARY_TRIGGER_COUNT, SUMMARY_BRIDGE_COUNT, BACKGROUND_MAX_BLOCKS_PER_RUN,
    RAG_VECTOR_TOP_K, RAG_RERANK_TOP_K, RAG_PROMPT_DOCS_MAX, RAG_KEYWORD_POOL_LIMIT,
    RAG_THRESHOLD_RECALL, RAG_THRESHOLD_DEFAULT,
    PROMPT_HISTORY_LIMIT, PROMPT_SUMMARY_LIMIT, PROMPT_MEMORY_LIMIT, FETCH_HISTORY_MULTIPLIER,
    MEMORY_TRIGGER_USER_TURNS, MEMORY_CONTEXT_MESSAGES,
    ROUTE_CHITCHAT, ROUTE_KNOWLEDGE, ROUTE_DANGER, ENABLE_DANGER_ROUTING, ROUTER_MARGIN,
    CHITCHAT_HINTS, KNOWLEDGE_HINTS, DANGER_PATTERNS, MIN_STORE_LENGTH_FOR_CHITCHAT,
    STORE_FACT_HINTS, INPUT_MODE_NORMAL, INPUT_MODE_OOC, INPUT_MODE_IC, INPUT_MODE_MIXED,
    OOC_IC_PATTERN_STR, OOC_IC_SEGMENT_PATTERN_STR, DANGER_FALLBACK_RESPONSE,
    ALLOWED_MEMORY_TYPES, ALLOWED_MEMORY_KEYS, DEFAULT_IMPORTANCE_BY_TYPE, MEMORY_NORMALIZATION_MAP,
    RECALL_HINT_PATTERNS, KEYWORD_STOPWORDS, KEYWORD_SYNONYMS, TECH_KEYWORDS,
    TECH_KEYWORD_BONUS, TECH_QUERY_NOISE_PENALTY, LIFESTYLE_KEYWORDS,
    ONE_CHAR_KEYWORDS, JOSA_SUFFIXES, JOSA_SUFFIXES_SORTED,
    NOISE_EXACT_PATTERNS, NOISE_PREFIX_PATTERNS, MEMORY_SIGNAL_PATTERNS,
)

_summary_state_lock = threading.Lock()
_summary_inflight_sessions: set[tuple[str, str]] = set()
OOC_IC_PATTERN = re.compile(OOC_IC_PATTERN_STR, re.IGNORECASE | re.DOTALL)
OOC_IC_SEGMENT_PATTERN = re.compile(OOC_IC_SEGMENT_PATTERN_STR, re.IGNORECASE | re.DOTALL)

# ─── 클라이언트 초기화 ────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 임베딩: google-genai SDK 직접 호출 (v1beta, output 768차원)
genai_client = google_genai.Client(
    api_key=GEMINI_API_KEY_FREE,
    http_options={"api_version": "v1beta"},
)

# LLM: langchain-google-genai (채팅 답변 생성용)
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=GEMINI_API_KEY_FREE,
)

summary_llm = ChatGoogleGenerativeAI(
    model=SUMMARY_MODEL,
    google_api_key=GEMINI_API_KEY_FREE,
)

# ─── 유틸 ────────────────────────────────────────────────────
class TeeStream:
    """stdout/stderr를 콘솔 + 파일로 동시에 기록하는 간단한 tee 스트림."""
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data: str):
        self.stream.write(data)
        self.file_obj.write(data)

    def flush(self):
        self.stream.flush()
        self.file_obj.flush()


def open_session_log_file() -> tuple[str, object]:
    os.makedirs(LOG_DIR, exist_ok=True)
    local_now = datetime.now(timezone.utc).astimezone()
    log_name = local_now.strftime("chat_session_%Y%m%d.log")
    log_path = os.path.join(LOG_DIR, log_name)
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    return log_path, log_fp


def maybe_prepare_context_cache(
    *,
    conf_uid: str,
    history_uid: str,
    system_block_text: str,
) -> str | None:
    """Provider context cache 준비용 스텁.

    현재는 의도적으로 OFF이며(None 반환), 추후 공급자 SDK 캐시 API를 연결한다.
    """
    if not ENABLE_PROVIDER_CONTEXT_CACHE:
        return None

    # TODO: 실제 토큰 계산기로 대체
    rough_tokens = max(1, len(system_block_text) // 4)
    if rough_tokens < CONTEXT_CACHE_MIN_TOKENS:
        return None

    # TODO: Provider cache create/get 호출 후 cache_id 반환
    return None


def initialize_router_if_needed() -> None:
    routing_initialize_router_if_needed(
        router_samples_path=ROUTER_SAMPLES_PATH,
        route_chitchat=ROUTE_CHITCHAT,
        route_knowledge=ROUTE_KNOWLEDGE,
        route_danger=ROUTE_DANGER,
        generate_embedding_fn=generate_embedding,
    )


def classify_route(user_input: str, query_embedding: list[float]) -> tuple[str, dict[str, float]]:
    return routing_classify_route(
        user_input,
        query_embedding,
        ensure_initialized_fn=initialize_router_if_needed,
        enable_danger_routing=ENABLE_DANGER_ROUTING,
        danger_patterns=DANGER_PATTERNS,
        route_chitchat=ROUTE_CHITCHAT,
        route_knowledge=ROUTE_KNOWLEDGE,
        route_danger=ROUTE_DANGER,
        router_margin=ROUTER_MARGIN,
        chitchat_hints=CHITCHAT_HINTS,
        knowledge_hints=KNOWLEDGE_HINTS,
    )


# ─── 임베딩 ──────────────────────────────────────────────────
def generate_embedding(text: str) -> list[float]:
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": 768},
    )
    return list(result.embeddings[0].values)


def fetch_recent_documents_for_keyword(
    conf_uid: str,
    history_uid: str,
    limit: int = 150,
) -> list[dict]:
    """키워드 보정용 후보 문서 풀 조회."""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_DOCUMENTS)
        .select("id, content, metadata, created_at")
        .eq("metadata->>conf_uid", conf_uid)
        .eq("metadata->>history_uid", history_uid)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def is_recall_query(user_input: str) -> bool:
    text = (user_input or "").lower()
    return any(p in text for p in RECALL_HINT_PATTERNS)


def is_worth_storing(user_input: str, route_label: str | None = None) -> bool:
    """documents_gemini 저장 가치 판단 (룰 베이스)."""
    text = (user_input or "").strip().lower()
    if not text:
        return False

    # 너무 짧은 입력 제외
    if len(text) <= 3:
        return False

    # 완전 노이즈 패턴 제외
    if text in NOISE_EXACT_PATTERNS:
        return False

    # 반복 웃음/감탄 단문 제외
    if text.startswith(NOISE_PREFIX_PATTERNS) and len(text) <= 6:
        return False

    # DANGER는 저장 제외
    if route_label == ROUTE_DANGER:
        return False

    # CHITCHAT도 길이가 충분하고 자기정보/계획 신호가 있으면 저장 허용
    if route_label == ROUTE_CHITCHAT:
        if len(text) >= MIN_STORE_LENGTH_FOR_CHITCHAT and any(h in text for h in STORE_FACT_HINTS):
            return True
        return False

    return True


def normalize_keyword_token(token: str) -> str:
    t = token.strip().lower()
    # 긴 조사부터 제거해야 ("으로" vs "로") 단절 오탐을 줄일 수 있음
    for suf in JOSA_SUFFIXES_SORTED:
        if len(t) > len(suf) + 1 and t.endswith(suf):
            t = t[:-len(suf)]
            break
    return t


def extract_keyword_terms(user_input: str) -> list[str]:
    base_tokens = re.findall(r"[0-9a-zA-Z가-힣]+", (user_input or "").lower())
    terms = []

    for token in base_tokens:
        token = normalize_keyword_token(token)
        if not token:
            continue
        if token in KEYWORD_STOPWORDS:
            continue
        if len(token) < 2 and token not in ONE_CHAR_KEYWORDS and token not in KEYWORD_SYNONYMS:
            continue
        terms.append(token)
        for syn in KEYWORD_SYNONYMS.get(token, []):
            terms.append(syn)

    # 중복 제거 + 순서 유지
    seen = set()
    deduped = []
    for t in terms:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def build_dynamic_stopwords(user_name: str | None, char_name: str | None) -> set[str]:
    """대화 참여자 이름/호칭을 동적으로 불용어에 추가."""
    words = set(KEYWORD_STOPWORDS)
    name_suffixes = [
        "", "아", "야", "는", "은", "이", "가", "를", "을", "와", "과",
        "에게", "한테", "님", "씨", "아님", "야님",
    ]
    for name in [user_name, char_name]:
        n = (name or "").strip().lower()
        if not n:
            continue
        for suf in name_suffixes:
            words.add(f"{n}{suf}")
    # 호출/대명사류는 의미 신호가 약하므로 고정 제외(이름 하드코딩은 사용하지 않음)
    words.update({"너", "네", "니", "니가", "너가", "나", "내", "내가", "님"})
    return words


def filter_terms_with_dynamic_stopwords(terms: list[str], dynamic_stopwords: set[str]) -> list[str]:
    out = []
    for t in terms:
        if t in dynamic_stopwords:
            continue
        out.append(t)
    return out


def keyword_match_score(content: str, terms: list[str]) -> float:
    if not terms:
        return 0.0

    text = (content or "").lower()
    if not text:
        return 0.0

    hit_terms = [t for t in terms if t in text]
    coverage = len(hit_terms) / len(terms)

    # 긴 토큰(3글자 이상)이 매칭되면 더 강하게 점수 반영
    long_hit_bonus = 0.12 if any(len(t) >= 3 for t in hit_terms) else 0.0
    score = min(1.0, (coverage * 0.88) + long_hit_bonus)
    return round(score, 4)


def extract_source_type(doc: dict) -> str:
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        return (metadata.get("source_type") or "").strip()
    return ""


def extract_related_message_id(doc: dict) -> int | None:
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("related_message_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def rerank_documents(
    user_input: str,
    vector_docs: list[dict],
    keyword_pool_docs: list[dict],
    user_name: str | None = None,
    char_name: str | None = None,
    final_k: int = 6,
    excluded_current_summary_end_ids: set[int] | None = None,
) -> list[dict]:
    recall_mode = is_recall_query(user_input)
    dynamic_stopwords = build_dynamic_stopwords(user_name, char_name)
    terms = filter_terms_with_dynamic_stopwords(extract_keyword_terms(user_input), dynamic_stopwords)
    excluded_current_summary_end_ids = excluded_current_summary_end_ids or set()

    by_id: dict[int, dict] = {}
    vector_score_by_id: dict[int, float] = {}

    for d in vector_docs:
        doc_id = d.get("id")
        if doc_id is None:
            continue
        by_id[doc_id] = dict(d)
        vector_score_by_id[doc_id] = float(d.get("similarity", 0.0) or 0.0)

    keyword_candidates = []
    for d in keyword_pool_docs:
        doc_id = d.get("id")
        if doc_id is None:
            continue
        kscore = keyword_match_score(d.get("content", ""), terms)
        if kscore <= 0:
            continue
        candidate = dict(d)
        candidate["_keyword_score"] = kscore
        keyword_candidates.append(candidate)
        if doc_id not in by_id:
            by_id[doc_id] = dict(candidate)
        else:
            # metadata 등 누락 필드 보강
            if not by_id[doc_id].get("metadata") and candidate.get("metadata"):
                by_id[doc_id]["metadata"] = candidate.get("metadata")

    # 키워드 강한 후보 상위만 병합
    keyword_candidates.sort(key=lambda x: x.get("_keyword_score", 0.0), reverse=True)
    keyword_top_ids = {d["id"] for d in keyword_candidates[:20]}

    vector_w = 0.45 if recall_mode else 0.72
    keyword_w = 0.55 if recall_mode else 0.28

    query_lower = (user_input or "").lower()
    query_has_tech = any(k in query_lower for k in TECH_KEYWORDS)

    scored = []
    for doc_id, d in by_id.items():
        if doc_id not in vector_score_by_id and doc_id not in keyword_top_ids:
            continue

        source_type = extract_source_type(d)
        related_message_id = extract_related_message_id(d)

        # 현재 세션에서 이미 [이전 누적 요약]에 직접 주입한 최신 요약 2개는
        # RAG 후보에서 제외해 중복 주입(요약 섹션 + 참고 문서)을 방지한다.
        if (
            source_type == "summary"
            and related_message_id is not None
            and related_message_id in excluded_current_summary_end_ids
        ):
            continue

        vscore = vector_score_by_id.get(doc_id, 0.0)
        kscore = d.get("_keyword_score")
        if kscore is None:
            kscore = keyword_match_score(d.get("content", ""), terms)

        # 기술 질의에서 기술 키워드가 직접 매칭되는 문서는 추가 가산
        content_lower = (d.get("content") or "").lower()
        tech_hit = any(k in content_lower for k in TECH_KEYWORDS) and any(k in terms for k in TECH_KEYWORDS)
        tech_bonus = TECH_KEYWORD_BONUS if tech_hit else 0.0
        lifestyle_hit = any(k in content_lower for k in LIFESTYLE_KEYWORDS)
        noise_penalty = 0.0
        if query_has_tech and (not tech_hit) and lifestyle_hit:
            noise_penalty = TECH_QUERY_NOISE_PENALTY

        source_boost = 0.0
        if source_type == "chat_message":
            source_boost = 0.14 if recall_mode else 0.04
        elif source_type == "summary":
            source_boost = 0.03 if recall_mode else -0.02

        rank_score = (vector_w * vscore) + (keyword_w * float(kscore)) + source_boost + tech_bonus - noise_penalty
        d["rank_score"] = round(rank_score, 4)
        d["keyword_score"] = round(float(kscore), 4)
        d["similarity"] = round(float(vscore), 4)
        d["source_type"] = source_type or "unknown"
        d["tech_bonus"] = round(tech_bonus, 4)
        d["noise_penalty"] = round(noise_penalty, 4)
        scored.append(d)

    scored.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return scored[:final_k]


def _extract_input_mode_from_metadata(metadata: dict | None) -> str:
    if not isinstance(metadata, dict):
        return INPUT_MODE_NORMAL
    return str(metadata.get("input_mode") or INPUT_MODE_NORMAL).upper()


def _is_segment_row(metadata: dict | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("is_segment"))


def should_exclude_from_context(row: dict) -> bool:
    """프롬프트/요약/메모리 컨텍스트에서 제외할 메시지 판정."""
    metadata = row.get("metadata") if isinstance(row, dict) else None
    if _is_segment_row(metadata):
        return str(metadata.get("segment_mode") or INPUT_MODE_NORMAL).upper() == INPUT_MODE_OOC
    if isinstance(metadata, dict) and bool(metadata.get("has_tagged_segment")):
        return True
    if _extract_input_mode_from_metadata(metadata) == INPUT_MODE_OOC:
        return True
    return False


def split_input_segments_for_storage(raw_input: str) -> tuple[list[dict], bool]:
    """OOC/IC 태그를 저장용 세그먼트로 분리.

    반환:
      - segments: [{"mode": "OOC|IC|NORMAL", "text": "...", "is_tagged": bool}, ...]
      - duplicate_ignored: 동일 태그(OOC/IC)가 2회 이상 등장해 무시했는지 여부
    """
    text = (raw_input or "").strip()
    if not text:
        return [{"mode": INPUT_MODE_NORMAL, "text": "", "is_tagged": False}], False

    matches = list(OOC_IC_SEGMENT_PATTERN.finditer(text))
    if not matches:
        return [{"mode": INPUT_MODE_NORMAL, "text": text, "is_tagged": False}], False

    ooc_count = sum(1 for m in matches if (m.group(1) or "").upper() == INPUT_MODE_OOC)
    ic_count = sum(1 for m in matches if (m.group(1) or "").upper() == INPUT_MODE_IC)

    # 동일 태그 중복 출현은 무시: 특수 파싱 자체를 하지 않고 일반 발화로 저장
    if ooc_count > 1 or ic_count > 1:
        return [{"mode": INPUT_MODE_NORMAL, "text": text, "is_tagged": False}], True

    segments: list[dict] = []
    cursor = 0
    for m in matches:
        start, end = m.span()
        if start > cursor:
            chunk = text[cursor:start].strip()
            if chunk:
                segments.append({"mode": INPUT_MODE_NORMAL, "text": chunk, "is_tagged": False})
        mode = (m.group(1) or "").upper()
        body = (m.group(2) or "").strip()
        if body:
            segments.append({"mode": mode, "text": body, "is_tagged": True})
        cursor = end

    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            segments.append({"mode": INPUT_MODE_NORMAL, "text": tail, "is_tagged": False})

    if not segments:
        segments = [{"mode": INPUT_MODE_NORMAL, "text": text, "is_tagged": False}]
    return segments, False


def derive_primary_input_mode(segments: list[dict]) -> str:
    modes = {str(s.get("mode", INPUT_MODE_NORMAL)).upper() for s in segments if (s.get("text") or "").strip()}
    if not modes:
        return INPUT_MODE_NORMAL
    if modes == {INPUT_MODE_OOC}:
        return INPUT_MODE_OOC
    if modes == {INPUT_MODE_IC}:
        return INPUT_MODE_IC
    if INPUT_MODE_OOC in modes and (INPUT_MODE_NORMAL in modes or INPUT_MODE_IC in modes):
        return INPUT_MODE_MIXED
    if INPUT_MODE_IC in modes and INPUT_MODE_NORMAL in modes:
        return INPUT_MODE_MIXED
    return INPUT_MODE_NORMAL


def parse_input_mode(raw_input: str) -> tuple[str, str]:
    """OOC/IC 태그 파싱. 반환값: (mode, cleaned_text)."""
    text = (raw_input or "").strip()
    m = OOC_IC_PATTERN.match(text)
    if not m:
        return INPUT_MODE_NORMAL, text
    mode = (m.group(1) or "").upper()
    content = (m.group(2) or "").strip()
    if mode == INPUT_MODE_OOC:
        return INPUT_MODE_OOC, content
    if mode == INPUT_MODE_IC:
        return INPUT_MODE_IC, content
    return INPUT_MODE_NORMAL, text


# ─── 프롬프트 & 체인 ──────────────────────────────────────────
# ChatPromptTemplate: 시스템 지시 + 컨텍스트(문서+이력) + 사용자 입력

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="history_messages"),
    ("human", "{user_input}"),
])
chain = prompt | llm | StrOutputParser()


# ─── 메인 루프 ────────────────────────────────────────────────
async def stream_answer(chain_inputs: dict, char_name: str) -> str:
    """astream으로 응답을 실시간 출력하고 최종 문자열을 반환."""
    chunks: list[str] = []
    print(f"\n{char_name}: ", end="", flush=True)
    async for chunk in chain.astream(chain_inputs):
        # 방어 코드: 일부 모델/파서 조합에서 빈 청크가 올 수 있어 무시
        if not chunk:
            continue
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n")
    print("=" * 80)
    return "".join(chunks).strip()


async def insert_message_in_thread(**kwargs) -> dict:
    """DB message INSERT를 스레드로 오프로딩."""
    return await asyncio.to_thread(insert_message, **kwargs)


async def insert_document_in_thread(**kwargs) -> dict:
    """DB document INSERT를 스레드로 오프로딩."""
    return await asyncio.to_thread(insert_document, **kwargs)


def schedule_background_db_task(
    pending_tasks: set[asyncio.Task],
    coro,
    label: str,
) -> None:
    """백그라운드 DB 작업 등록 + 예외 로깅 + 참조 정리."""
    task = asyncio.create_task(coro)
    pending_tasks.add(task)

    def _on_done(done_task: asyncio.Task) -> None:
        pending_tasks.discard(done_task)
        try:
            done_task.result()
        except Exception as e:
            print(f"[DB-BG-ERROR] {label}: {e}")

    task.add_done_callback(_on_done)


async def run_chat_loop() -> None:
    conf_uid    = "sua_test_003" # 대화 요약 테스트용
    history_uid = "session_001"
    user_id     = "user_baemin"
    user_name   = "배민"
    char_id     = "char_sua"
    char_name   = "수아"

    log_path, log_fp = open_session_log_file()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_fp)
    sys.stderr = TeeStream(original_stderr, log_fp)

    print(f"[로그 파일] {log_path}")
    print(f"[시작 시각] {datetime.now(timezone.utc).astimezone().isoformat()}")
    print(f"[모델] LLM={CHAT_MODEL} / Embedding={EMBEDDING_MODEL}")
    print("대화를 시작합니다. 종료: exit / quit / q")
    print("=" * 80)
    initialize_router_if_needed()
    pending_bg_tasks: set[asyncio.Task] = set()

    try:
        while True:
            raw_user_input = (await asyncio.to_thread(input, "나: ")).strip()
            print(f"[사용자 입력] {raw_user_input}")

            if raw_user_input.lower() in {"exit", "quit", "q"}:
                print("종료합니다.")
                break
            if not raw_user_input:
                continue

            try:
                segments, duplicate_tag_ignored = split_input_segments_for_storage(raw_user_input)
                input_mode = derive_primary_input_mode(segments)
                user_input = raw_user_input  # LLM 호출은 항상 원문 사용
                mode_instruction = get_mode_instruction(input_mode)
                if DEBUG:
                    print(f"[입력 모드] {input_mode} | 원문 입력 사용")
                    if duplicate_tag_ignored:
                        print("[입력 파서] 동일 OOC/IC 태그 중복 감지: 특수 파싱 무시하고 NORMAL 저장 처리")

                # 현재 시간 주입 (LLM이 '어제', '지난주' 같은 표현을 계산할 수 있도록)
                current_time = datetime.now(timezone.utc).astimezone().strftime("%Y년 %m월 %d일 %A %H:%M")

                # 1) 임베딩 생성 (라우팅 + RAG 검색에 모두 재활용)
                query_embedding = generate_embedding(user_input)
                route_label, route_scores = classify_route(user_input, query_embedding)

                # 2) 라우팅 기반 데이터 수집
                if route_label == ROUTE_CHITCHAT or (ENABLE_DANGER_ROUTING and route_label == ROUTE_DANGER):
                    with ThreadPoolExecutor() as executor:
                        future_history = executor.submit(fetch_recent_messages, conf_uid, history_uid, PROMPT_HISTORY_LIMIT)
                        future_summaries = executor.submit(fetch_recent_summaries_for_prompt, conf_uid, history_uid, PROMPT_SUMMARY_LIMIT)
                        future_memories = executor.submit(fetch_active_memories, conf_uid, char_id, user_id, PROMPT_MEMORY_LIMIT)
                        history = future_history.result()
                        summary_rows = future_summaries.result()
                        memory_rows = future_memories.result()
                    vector_docs_raw = []
                    keyword_pool_docs = []
                else:
                    with ThreadPoolExecutor() as executor:
                        future_docs = executor.submit(find_similar_documents, query_embedding, conf_uid, history_uid, RAG_VECTOR_TOP_K)
                        future_keyword_docs = executor.submit(fetch_recent_documents_for_keyword, conf_uid, history_uid, RAG_KEYWORD_POOL_LIMIT)
                        future_history = executor.submit(fetch_recent_messages, conf_uid, history_uid, PROMPT_HISTORY_LIMIT)
                        future_summaries = executor.submit(fetch_recent_summaries_for_prompt, conf_uid, history_uid, PROMPT_SUMMARY_LIMIT)
                        future_memories = executor.submit(fetch_active_memories, conf_uid, char_id, user_id, PROMPT_MEMORY_LIMIT)
                        vector_docs_raw = future_docs.result()
                        keyword_pool_docs = future_keyword_docs.result()
                        history = future_history.result()
                        summary_rows = future_summaries.result()
                        memory_rows = future_memories.result()

                previous_summary = fmt_recent_summaries(summary_rows)
                long_memories = fmt_long_memories(memory_rows)
                excluded_summary_end_ids = {
                    int(r["end_message_id"])
                    for r in summary_rows
                    if r.get("end_message_id") is not None
                }

                # 회상형 질문일수록 키워드 가중치를 높여 재랭크
                recall_mode = is_recall_query(user_input)
                dynamic_stopwords = build_dynamic_stopwords(user_name, char_name)
                terms_for_debug = filter_terms_with_dynamic_stopwords(
                    extract_keyword_terms(user_input),
                    dynamic_stopwords,
                )
                history_messages = fmt_to_lc_history(history)
                if route_label == ROUTE_KNOWLEDGE:
                    similar_docs_raw = rerank_documents(
                        user_input=user_input,
                        vector_docs=vector_docs_raw,
                        keyword_pool_docs=keyword_pool_docs,
                        user_name=user_name,
                        char_name=char_name,
                        final_k=RAG_RERANK_TOP_K,
                        excluded_current_summary_end_ids=excluded_summary_end_ids,
                    )
                    THRESHOLD = RAG_THRESHOLD_RECALL if recall_mode else RAG_THRESHOLD_DEFAULT
                    similar_docs = [d for d in similar_docs_raw if d.get('rank_score', 0) >= THRESHOLD][:RAG_PROMPT_DOCS_MAX]
                else:
                    similar_docs_raw = []
                    THRESHOLD = RAG_THRESHOLD_RECALL if recall_mode else RAG_THRESHOLD_DEFAULT
                    similar_docs = []

                if DEBUG:
                    print("\n" + "=" * 80)
                    print(f"[라우팅] {route_label} | scores={{{ROUTE_CHITCHAT}:{route_scores.get(ROUTE_CHITCHAT,0):.3f}, {ROUTE_KNOWLEDGE}:{route_scores.get(ROUTE_KNOWLEDGE,0):.3f}, {ROUTE_DANGER}:{route_scores.get(ROUTE_DANGER,0):.3f}}}")
                    print(f"[키워드 추출] {terms_for_debug if terms_for_debug else '(없음)'}")
                    print(f"[🔍 검색된 유사 문서 (Threshold >= {THRESHOLD})]")
                    for d in similar_docs_raw:
                        rank_score = d.get("rank_score", 0)
                        sim = d.get("similarity", 0)
                        kscore = d.get("keyword_score", 0)
                        tbonus = d.get("tech_bonus", 0)
                        source_type = d.get("source_type", "unknown")
                        if rank_score >= THRESHOLD:
                            print(f" ✅ [점수 {rank_score:.3f} | vec {sim:.3f} | key {kscore:.3f} | tech {tbonus:.3f} | {source_type}] {d['content']}")
                        else:
                            print(f" ❌ [점수 {rank_score:.3f} | vec {sim:.3f} | key {kscore:.3f} | tech {tbonus:.3f} | {source_type}] (버려짐) {d['content']}")
                    if not similar_docs_raw:
                        print(" - 검색된 문서 없음")
                    print("=" * 80)

                if DEBUG:
                    formatted_prompt = prompt.format_messages(
                        policy=RESPONSE_POLICY,
                        persona=CHARACTER_PERSONA,
                        current_time=current_time,
                        mode_instruction=mode_instruction,
                        previous_summary=previous_summary,
                        long_memories=long_memories,
                        documents=fmt_documents(similar_docs),
                        history_messages=history_messages,
                        user_input=user_input,
                    )
                    print(f"\n[🧠 LLM에 주입된 전체 프롬프트 전문]")
                    for msg in formatted_prompt:
                        print(f"[{msg.type.upper()}]")
                        print(f"{msg.content}\n")
                    print("=" * 80)

                # 3) LLM 비동기 스트리밍 호출 (LCEL 체인)
                chain_inputs = {
                    "policy": RESPONSE_POLICY,
                    "persona": CHARACTER_PERSONA,
                    "current_time": current_time,
                    "mode_instruction": mode_instruction,
                    "previous_summary": previous_summary,
                    "long_memories": long_memories,
                    "documents": fmt_documents(similar_docs),
                    "history_messages": history_messages,
                    "user_input": user_input,
                }
                _cache_id = maybe_prepare_context_cache(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    system_block_text=(
                        f"{RESPONSE_POLICY}\n{CHARACTER_PERSONA}\n{previous_summary}\n{long_memories}\n"
                        f"{fmt_documents(similar_docs)}\n{fmt_history_debug(history_messages)}"
                    ),
                )
                if ENABLE_DANGER_ROUTING and route_label == ROUTE_DANGER:
                    answer = DANGER_FALLBACK_RESPONSE
                    print(f"\n{char_name}: {answer}\n")
                    print("=" * 80)
                else:
                    answer = await stream_answer(chain_inputs, char_name)

                # 5) 사용자 원문 저장 + 세그먼트 저장 (추적용)
                origin_group_id = f"{history_uid}:{uuid4().hex}"
                saved_user_msg = await insert_message_in_thread(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    content=user_input,
                    role="human",
                    speaker_type="user",
                    speaker_id=user_id,
                    display_name=user_name,
                    metadata={
                        "input_mode": input_mode,
                        "raw_input": raw_user_input,
                        "origin_group_id": origin_group_id,
                        "is_segment": False,
                        "segment_count": len(segments),
                        "has_tagged_segment": any(bool(seg.get("is_tagged")) for seg in segments),
                        "duplicate_tag_ignored": duplicate_tag_ignored,
                    },
                )

                # 분리 세그먼트 저장: 태그(OOC/IC)가 실제 포함된 경우에만 저장.
                # 태그가 전혀 없는 일반 입력은 원문 1건만 저장해 중복을 방지한다.
                has_tagged_segment = any(bool(seg.get("is_tagged")) for seg in segments)
                if has_tagged_segment:
                    for idx, seg in enumerate(segments, start=1):
                        seg_text = (seg.get("text") or "").strip()
                        if not seg_text:
                            continue
                        schedule_background_db_task(
                            pending_bg_tasks,
                            insert_message_in_thread(
                                conf_uid=conf_uid,
                                history_uid=history_uid,
                                content=seg_text,
                                role="human",
                                speaker_type="user",
                                speaker_id=user_id,
                                display_name=user_name,
                                reply_to_message_id=saved_user_msg["id"],
                                metadata={
                                    "input_mode": str(seg.get("mode") or INPUT_MODE_NORMAL).upper(),
                                    "is_segment": True,
                                    "segment_index": idx,
                                    "segment_mode": str(seg.get("mode") or INPUT_MODE_NORMAL).upper(),
                                    "origin_message_id": saved_user_msg["id"],
                                    "origin_group_id": origin_group_id,
                                    "duplicate_tag_ignored": duplicate_tag_ignored,
                                },
                            ),
                            label="insert_user_segment",
                        )

                non_ooc_segment_texts = [
                    (seg.get("text") or "").strip()
                    for seg in segments
                    if str(seg.get("mode") or INPUT_MODE_NORMAL).upper() != INPUT_MODE_OOC
                    and (seg.get("text") or "").strip()
                ]
                store_candidate_text = " ".join(non_ooc_segment_texts).strip() or user_input
                has_ooc_only = all(str(seg.get("mode") or INPUT_MODE_NORMAL).upper() == INPUT_MODE_OOC for seg in segments if (seg.get("text") or "").strip())

                # 6) 사용자 메시지 임베딩 → documents 저장
                if (not has_ooc_only) and is_worth_storing(store_candidate_text, route_label=route_label):
                    if store_candidate_text == user_input:
                        schedule_background_db_task(
                            pending_bg_tasks,
                            insert_document_in_thread(
                                conf_uid=conf_uid,
                                history_uid=history_uid,
                                speaker_id=user_id,
                                content=store_candidate_text,
                                embedding=query_embedding,
                                related_message_id=saved_user_msg["id"],
                                extra_metadata={
                                    "input_mode": input_mode,
                                    "origin_group_id": origin_group_id,
                                },
                            ),
                            label="insert_user_document",
                        )
                    else:
                        async def _insert_user_document_with_new_embedding() -> None:
                            store_embedding = await asyncio.to_thread(generate_embedding, store_candidate_text)
                            await insert_document_in_thread(
                                conf_uid=conf_uid,
                                history_uid=history_uid,
                                speaker_id=user_id,
                                content=store_candidate_text,
                                embedding=store_embedding,
                                related_message_id=saved_user_msg["id"],
                                extra_metadata={
                                    "input_mode": input_mode,
                                    "origin_group_id": origin_group_id,
                                },
                            )

                        schedule_background_db_task(
                            pending_bg_tasks,
                            _insert_user_document_with_new_embedding(),
                            label="insert_user_document_with_embedding",
                        )
                elif DEBUG:
                    print("[GATING] documents 저장 스킵: 저가치 입력 또는 OOC-only 입력")

                # 7) AI 응답 메시지 저장 (RAG 출처 트래킹 포함)
                retrieved_info = [
                    {"id": doc["id"], "similarity": doc.get("similarity", 0)}
                    for doc in similar_docs
                ]
                ai_meta = {"retrieved_docs": retrieved_info} if retrieved_info else {}
                ai_meta["input_mode"] = input_mode
                ai_meta["origin_group_id"] = origin_group_id
                await insert_message_in_thread(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    role="ai",
                    speaker_type="character",
                    speaker_id=char_id,
                    display_name=char_name,
                    content=answer,
                    reply_to_message_id=saved_user_msg["id"],
                    metadata=ai_meta,
                )

                # 8) 장기 기억 추출(비동기): OOC-only 입력은 제외, 그 외는 5턴 배치 처리
                if not has_ooc_only:
                    queue_memory_extraction_job(
                        conf_uid=conf_uid,
                        history_uid=history_uid,
                        owner_speaker_id=char_id,
                        target_speaker_id=user_id,
                    )
                if DEBUG:
                    mem_last_id, mem_pending = get_memory_progress(
                        conf_uid=conf_uid,
                        history_uid=history_uid,
                    )
                    print(f"[🧠 메모리 진행] 마지막 처리 user_id={mem_last_id}, 미처리 사용자발화={mem_pending}개")

                # 9) 대화 요약 진행상태 출력 + 백그라운드 트리거
                # DANGER 응답은 후속 요약 맥락 오염을 막기 위해 요약 큐에서 제외
                if (not ENABLE_DANGER_ROUTING) or route_label != ROUTE_DANGER:
                    if DEBUG:
                        last_id, pending_count = get_summary_progress(conf_uid, history_uid)
                        print(f"[📊 요약 트리거] 마지막 요약 id={last_id}, 미처리 메시지={pending_count}개")
                    if not has_ooc_only:
                        queue_summarization_job(conf_uid, history_uid)

            except Exception as e:
                print(f"\n[오류] {e}\n")
    finally:
        if pending_bg_tasks:
            await asyncio.gather(*pending_bg_tasks, return_exceptions=True)
        print(f"[종료 시각] {datetime.now(timezone.utc).astimezone().isoformat()}")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()


# ─── 장기 기억 추출 서비스 ───────────────────────────────────────
def queue_memory_extraction_job(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
) -> None:
    memory_queue_memory_extraction_job(
        conf_uid=conf_uid,
        history_uid=history_uid,
        owner_speaker_id=owner_speaker_id,
        target_speaker_id=target_speaker_id,
        debug_enabled=DEBUG,
        memory_trigger_user_turns=MEMORY_TRIGGER_USER_TURNS,
        background_max_blocks_per_run=BACKGROUND_MAX_BLOCKS_PER_RUN,
        memory_signal_patterns=MEMORY_SIGNAL_PATTERNS,
        summary_llm=summary_llm,
        memory_context_messages=MEMORY_CONTEXT_MESSAGES,
        fetch_recent_messages_fn=fetch_recent_messages,
        fetch_user_messages_since_fn=lambda c, h, after_id: fetch_user_messages_since(
            c,
            h,
            after_id,
            exclude_fn=should_exclude_from_context,
        ),
        get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_memories,
        should_exclude_from_context_fn=should_exclude_from_context,
        is_worth_storing_fn=lambda text: is_worth_storing(text, route_label=None),
        fetch_active_memories_fn=fetch_active_memories,
        format_long_memories_fn=fmt_long_memories,
        build_memory_extraction_prompt_fn=build_memory_extraction_prompt,
        upsert_user_memory_fn=upsert_user_memory,
        logger_fn=print,
    )


def trigger_memory_extraction_if_needed(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    debug_log: bool = True,
) -> None:
    memory_trigger_memory_extraction_if_needed(
        conf_uid=conf_uid,
        history_uid=history_uid,
        owner_speaker_id=owner_speaker_id,
        target_speaker_id=target_speaker_id,
        debug_log=debug_log,
        debug_enabled=DEBUG,
        memory_trigger_user_turns=MEMORY_TRIGGER_USER_TURNS,
        background_max_blocks_per_run=BACKGROUND_MAX_BLOCKS_PER_RUN,
        memory_signal_patterns=MEMORY_SIGNAL_PATTERNS,
        summary_llm=summary_llm,
        memory_context_messages=MEMORY_CONTEXT_MESSAGES,
        fetch_recent_messages_fn=fetch_recent_messages,
        fetch_user_messages_since_fn=lambda c, h, after_id: fetch_user_messages_since(
            c,
            h,
            after_id,
            exclude_fn=should_exclude_from_context,
        ),
        get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_memories,
        should_exclude_from_context_fn=should_exclude_from_context,
        is_worth_storing_fn=lambda text: is_worth_storing(text, route_label=None),
        fetch_active_memories_fn=fetch_active_memories,
        format_long_memories_fn=fmt_long_memories,
        build_memory_extraction_prompt_fn=build_memory_extraction_prompt,
        upsert_user_memory_fn=upsert_user_memory,
        logger_fn=print,
    )


def get_memory_progress(conf_uid: str, history_uid: str) -> tuple[int, int]:
    """메모리 진행상태 조회: (마지막 처리 user_message_id, 미처리 사용자 발화 수)."""
    return memory_get_memory_progress(
        conf_uid,
        history_uid,
        fetch_user_messages_since_fn=lambda c, h, after_id: fetch_user_messages_since(
            c,
            h,
            after_id,
            exclude_fn=should_exclude_from_context,
        ),
        get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_memories,
    )


# ─── 대화 요약 서비스 ─────────────────────────────────────────
def trigger_summarization_if_needed(conf_uid: str, history_uid: str, debug_log: bool = True) -> None:
    """AI 답변 저장 직후마다 호출됨. 10턴 블록을 최대 N개까지 캐치업 처리."""
    processed_blocks = 0
    while processed_blocks < BACKGROUND_MAX_BLOCKS_PER_RUN:
        latest_summary = fetch_latest_summary(conf_uid, history_uid)
        last_id = latest_summary["end_message_id"] if latest_summary else 0
        unprocessed = fetch_messages_since(
            conf_uid,
            history_uid,
            after_id=last_id,
            exclude_fn=should_exclude_from_context,
        )

        if len(unprocessed) < SUMMARY_TRIGGER_COUNT:
            break  # 아직 10턴 안 쌓임

        summarize_and_save(conf_uid, history_uid, latest_summary, unprocessed)
        processed_blocks += 1

    if debug_log and DEBUG and processed_blocks >= BACKGROUND_MAX_BLOCKS_PER_RUN:
        print(f"[요약 캐치업 제한] 1회 최대 {BACKGROUND_MAX_BLOCKS_PER_RUN}블록 처리 후 종료")


def _run_summarization_job(conf_uid: str, history_uid: str) -> None:
    """요약 백그라운드 작업 본체."""
    try:
        trigger_summarization_if_needed(conf_uid, history_uid, debug_log=False)
    except Exception as e:
        print(f"[요약 실패 — 다음 턴 재시도 예정] {e}")
    finally:
        key = (conf_uid, history_uid)
        try:
            with _summary_state_lock:
                _summary_inflight_sessions.discard(key)
        except Exception as e:
            print(f"[요약 상태 해제 실패] {e}")


def get_summary_progress(conf_uid: str, history_uid: str) -> tuple[int, int]:
    """요약 진행상태 조회: (마지막 요약 end_message_id, 미처리 메시지 수)."""
    latest_summary = fetch_latest_summary(conf_uid, history_uid)
    last_id = latest_summary["end_message_id"] if latest_summary else 0
    unprocessed = fetch_messages_since(
        conf_uid,
        history_uid,
        after_id=last_id,
        exclude_fn=should_exclude_from_context,
    )
    return last_id, len(unprocessed)


def queue_summarization_job(conf_uid: str, history_uid: str) -> None:
    """같은 세션에서 중복 요약 작업이 겹치지 않도록 백그라운드 큐잉."""
    key = (conf_uid, history_uid)
    with _summary_state_lock:
        if key in _summary_inflight_sessions:
            if DEBUG:
                print(f"[요약 스킵] 이미 백그라운드 요약 실행 중: {conf_uid}/{history_uid}")
            return
        _summary_inflight_sessions.add(key)

    worker = threading.Thread(
        target=_run_summarization_job,
        args=(conf_uid, history_uid),
        daemon=True,
    )
    worker.start()


def summarize_and_save(
    conf_uid: str,
    history_uid: str,
    latest_summary: dict | None,
    unprocessed: list[dict],
) -> None:
    """누적 요약을 생성하고 conversation_summaries + documents_gemini에 저장합니다."""

    # 재료 1: 이전 누적 요약본
    prev_summary_text = (
        latest_summary["summary_text"] if latest_summary
        else "(첫 번째 요약 — 이전 내용 없음)"
    )
    last_id = latest_summary["end_message_id"] if latest_summary else 0

    # 재료 2: 이음새 대화 (이전 블록 마지막 2개 — 요약 경계의 낮 단절 방지)
    bridge = []
    if latest_summary:
        bridge = fetch_bridge_messages(
            conf_uid, history_uid,
            up_to_id=last_id,
            n=SUMMARY_BRIDGE_COUNT,
            exclude_fn=should_exclude_from_context,
        )

    # 재료 3: 새로 쌓인 10개 (레이스 컨디션 방지: 정확히 TRIGGER_COUNT만 처리)
    new_block = unprocessed[:SUMMARY_TRIGGER_COUNT]

    # 참여자 명단 추출 (중복 제거)
    participant_names = list(set(r.get("display_name") for r in new_block if r.get("display_name")))

    bridge_text    = fmt_messages_for_summary(bridge)    if bridge    else "(없음)"
    new_block_text = fmt_messages_for_summary(new_block)

    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        prev_summary_text=prev_summary_text,
        bridge_text=bridge_text,
        new_block_text=new_block_text,
    )

    raw_summary_text = summary_llm.invoke(summary_prompt).content
    new_summary_text = fmt_normalize_summary(raw_summary_text)

    if DEBUG:
        print(f"\n[📝 누적 요약 완료]\n{new_summary_text}\n")

    # 저장 1: conversation_summaries 테이블 (SQL 원본)
    insert_summary(
        conf_uid=conf_uid,
        history_uid=history_uid,
        summary=new_summary_text,
        start_message_id=new_block[0]["id"],
        end_message_id=new_block[-1]["id"],
        covered_message_count=len(new_block),
        summary_seq=(latest_summary["summary_seq"] + 1) if latest_summary else 1,
        participants=participant_names,
    )

    # 저장 2: documents_gemini 테이블 (오래된 세션 RAG 검색용 벡터화)
    # "3달 전에 무슨 이야기했지?" 같은 질문에 과거 세션 요약본을 검색할 수 있도록 임베딩
    summary_embedding = generate_embedding(new_summary_text)
    insert_document(
        conf_uid=conf_uid,
        history_uid=history_uid,
        speaker_id="system_summary",  # source_type 구분용
        content=new_summary_text,
        embedding=summary_embedding,
        related_message_id=new_block[-1]["id"],
        speaker_type="system", #chat_message와 구분하기 위해
        source_type="summary", #chat_message와 구분하기 위해
    )


if __name__ == "__main__":
    asyncio.run(run_chat_loop())
