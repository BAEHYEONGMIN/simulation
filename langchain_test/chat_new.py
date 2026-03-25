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
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supabase import create_client
from google import genai as google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    SUPABASE_URL, SUPABASE_KEY,
    GEMINI_API_KEY_FREE,
    CHAT_MODEL, EMBEDDING_MODEL,SUMMARY_MODEL,TEST_MODEL
)

# 디버그 모드: True이면 검색 결과 및 프롬프트 전문을 터미널에 출력합니다.
# 서버화(FastAPI) 시에는 False로 변경하세요.
DEBUG = True
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_summary_state_lock = threading.Lock()
_summary_inflight_sessions: set[tuple[str, str]] = set()

# Provider Context Caching (기본 OFF)
# - 현재는 비용/히트율 검증 전이라 비활성화 상태 유지
# - 추후 실험 시 플래그만 ON 하여 적용 범위/비용 측정
ENABLE_PROVIDER_CONTEXT_CACHE = False
CONTEXT_CACHE_MIN_TOKENS = 32768

# ─── DB 상수 ─────────────────────────────────────────────────
SCHEMA          = "chatbot"
TABLE_CHAT      = "chat_messages"
TABLE_SUMMARIES = "conversation_summaries"  # 누적 요약본 저장
TABLE_DOCUMENTS = "documents_gemini"         # 768차원 (gemini-embedding-001)
RPC_MATCH       = "match_documents_gemini"

# 대화 요약 트리거 설정
SUMMARY_TRIGGER_COUNT = 10  # 이 터수 만큼 쌓이면 요약 실행
SUMMARY_BRIDGE_COUNT  = 2   # 이음새로 가져올 이전 블록 마지막 대화 수

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
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ─── 임베딩 ──────────────────────────────────────────────────
def generate_embedding(text: str) -> list[float]:
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": 768},
    )
    return list(result.embeddings[0].values)


# ─── DB 조회 ─────────────────────────────────────────────────
def fetch_recent_messages(conf_uid: str, history_uid: str, limit: int = 10) -> list[dict]:
    """최근 대화 이력 조회 (오래된 순)"""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .select("id, role, speaker_id, display_name, content, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(result.data or []))


def find_similar_documents(
    query_embedding: list[float],
    conf_uid: str,
    history_uid: str | None = None,
    top_k: int = 20,
) -> list[dict]:
    """벡터 유사도 검색"""
    filter_obj = {"conf_uid": conf_uid}
    if history_uid:
        filter_obj["history_uid"] = history_uid

    result = (
        supabase.schema(SCHEMA)
        .rpc(
            RPC_MATCH,
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter": filter_obj,
            },
        )
        .execute()
    )
    return result.data or []


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


# ─── DB 저장 ─────────────────────────────────────────────────
def insert_message(
    conf_uid: str,
    history_uid: str,
    role: str,          # "human" | "ai"
    speaker_type: str,  # "user" | "character"
    speaker_id: str,
    display_name: str,
    content: str,
    reply_to_message_id: int = None,
    metadata: dict = None,
) -> dict:
    payload = {
        "conf_uid": conf_uid,
        "history_uid": history_uid,
        "role": role,
        "speaker_type": speaker_type,
        "speaker_id": speaker_id,
        "display_name": display_name,
        "content": content,
        "avatar": None,
        "reply_to_message_id": reply_to_message_id,
        "created_at": now_iso(),
    }
    if metadata is not None:
        payload["metadata"] = metadata

    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"{TABLE_CHAT} 저장 실패")
    return result.data[0]


def insert_document(
    conf_uid: str,
    history_uid: str,
    speaker_id: str,
    content: str,
    embedding: list[float],
    related_message_id: int,
    speaker_type: str = "user",
    source_type: str = "chat_message",
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {
        "conf_uid": conf_uid,
        "history_uid": history_uid,
        "speaker_id": speaker_id,
        "speaker_type": speaker_type,
        "source_type": source_type,
        "related_message_id": related_message_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "content": content,
        "metadata": metadata,
        "embedding": embedding,
        "created_at": now_iso(),
    }
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_DOCUMENTS)
        .insert(payload)
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"{TABLE_DOCUMENTS} 저장 실패")
    return result.data[0]


# ─── 요약 서비스 DB 함수 ────────────────────────────────────
def fetch_latest_summary(conf_uid: str, history_uid: str) -> dict | None:
    """이 세션의 가장 최신 누적 요약본을 1개 가져옵니다."""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_SUMMARIES)
        .select("id, summary_type, summary_text, start_message_id, end_message_id, covered_message_count, summary_seq, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("summary_seq", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def fetch_recent_summaries_for_prompt(conf_uid: str, history_uid: str, limit: int = 2) -> list[dict]:
    """프롬프트 주입용 최근 요약본 n개 조회 (최신순 조회 후 시간순으로 반환)."""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_SUMMARIES)
        .select("id, summary_type, summary_text, start_message_id, end_message_id, covered_message_count, summary_seq, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("summary_seq", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    return list(reversed(rows))


def fetch_messages_since(conf_uid: str, history_uid: str, after_id: int) -> list[dict]:
    """after_id 이후로 새로 쌓인 원문 대화를 오래된 순으로 모두 반환합니다."""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .select("id, role, speaker_id, display_name, content, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .gt("id", after_id)
        .order("id", desc=False)
        .execute()
    )
    return result.data or []


def fetch_bridge_messages(conf_uid: str, history_uid: str, up_to_id: int, n: int = 2) -> list[dict]:
    """up_to_id 이하의 마지막 n개 대화(이음새)를 반환합니다."""
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .select("id, role, display_name, content, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .lte("id", up_to_id)
        .order("id", desc=True)
        .limit(n)
        .execute()
    )
    return list(reversed(result.data or []))  # 오래된 순으로 다시 납힘


def insert_summary(
    conf_uid: str,
    history_uid: str,
    summary: str,
    start_message_id: int,
    end_message_id: int,
    covered_message_count: int,
    summary_seq: int,
    participants: list = None,
    summary_type: str = "rolling",  # rolling | checkpoint | final
) -> dict:
    """conversation_summaries에 요약본을 저장합니다.

    summary_type 종류:
      - rolling    : 10턴마다 자동 생성되는 실시간 누적 요약 (현재 자동 트리거)
      - checkpoint : 주제 급변 / 중요 약속 등 특정 이벤트 감지 시 수동으로 찍는 중간 저장점
      - final      : "잘 자", "나중에 봐" 등 세션 종료 발언 감지 시 세션 전체를 압축하는 마무리 요약

    ┌── [chat_messages.metadata vs conversation_summaries 구조 비교] ─────────────
    │
    │  chat_messages.metadata  (JSONB, AI 응답 저장 시에만 존재)
    │    └─ {"retrieved_docs": [{"id": 42, "similarity": 0.88}, ...]}
    │       • 목적: 이 AI 답변이 어느 RAG 문서를 참고했는지 추적 (감사/디버그용 로그)
    │       • 항목: documents_gemini의 id 및 유사도 — "얼마나 이용되었는가"
    │
    │  conversation_summaries 테이블은 metadata JSONB 컬럼이 없습니다.
    │  대신 이미 정형 컬럼으로 구조화됨:
    │    └─ summary_type     : 누적 요약 종류 (rolling / checkpoint / final)
    │    └─ start_message_id : 어느 메시지부터 요약되었는지
    │    └─ end_message_id   : 어느 메시지까지 요약되었는지
    │    └─ covered_message_count : 요약된 대화 수
    │
    └─────────────────────────────────────────────────────────────────────────────
    """
    payload = {
        "conf_uid":               conf_uid,
        "history_uid":            history_uid,
        "summary_text":           summary,
        "summary_type":           summary_type,
        "start_message_id":       start_message_id,
        "end_message_id":         end_message_id,
        "covered_message_count":  covered_message_count,
        "summary_seq":            summary_seq,
        "participants":           participants,
        "created_at":             now_iso(),
    }
    result = supabase.schema(SCHEMA).table(TABLE_SUMMARIES).insert(payload).execute()
    if not result.data:
        raise RuntimeError(f"{TABLE_SUMMARIES} 저장 실패")
    return result.data[0]


# ─── 포맷 헬퍼 ───────────────────────────────────────────────
def format_messages_for_summary(rows: list[dict]) -> str:
    """LLM 요약 프롬프트용 포맷: [YYYY-MM-DD] 화자: 내용"""
    return "\n".join(
        f"[{r.get('created_at','')[:10]}] {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
    )


def normalize_summary_text(text: str) -> str:
    """하위 모델 출력의 마크다운 흔들림을 정규화해 저장 포맷을 일정하게 유지."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    normalized = []

    for ln in lines:
        ln = re.sub(r"\*\*(.*?)\*\*", r"\1", ln)       # bold 제거
        ln = re.sub(r"`([^`]*)`", r"\1", ln)           # inline code 제거
        ln = re.sub(r"^\s*[*+-]\s*", "", ln)           # 불릿 제거
        ln = re.sub(r"^\s*\d+[.)]\s*", "", ln)         # 번호 제거
        if ln:
            normalized.append(f"- {ln}")

    if not normalized:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        return f"- {cleaned}" if cleaned else ""

    return "\n".join(normalized[:5])


RECALL_HINT_PATTERNS = [
    "뭐였지", "뭐였더라", "기억", "기억나", "제목", "이름",
    "아까", "전에", "읽었", "읽었다", "말했", "추천",
]

KEYWORD_STOPWORDS = {
    "수아야", "내가", "아까", "그", "그거", "이거", "저거", "뭐",
    "뭐지", "뭐였지", "좀", "그리고", "오늘", "우리", "너는", "나는",
}

KEYWORD_SYNONYMS = {
    "책": ["소설", "작품", "도서"],
    "소설": ["책", "작품"],
    "제목": ["이름", "타이틀"],
    "이름": ["제목"],
}

ONE_CHAR_KEYWORDS = {"책"}
JOSA_SUFFIXES = [
    "으로", "에서", "에게", "한테", "까지", "부터", "처럼", "보다",
    "이", "가", "은", "는", "을", "를", "와", "과", "도", "로", "에",
]


def is_recall_query(user_input: str) -> bool:
    text = (user_input or "").lower()
    return any(p in text for p in RECALL_HINT_PATTERNS)


NOISE_EXACT_PATTERNS = {
    "ㅋ", "ㅋㅋ", "ㅋㅋㅋ", "ㅎ", "ㅎㅎ", "ㅎㅎㅎ",
    "ㅇㅇ", "ㄴㄴ", "ㅜㅜ", "ㅠㅠ", "...", "..", ";;",
}

NOISE_PREFIX_PATTERNS = ("ㅋ", "ㅎ")


def is_worth_storing(user_input: str) -> bool:
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

    return True


def normalize_keyword_token(token: str) -> str:
    t = token.strip().lower()
    for suf in JOSA_SUFFIXES:
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


def rerank_documents(
    user_input: str,
    vector_docs: list[dict],
    keyword_pool_docs: list[dict],
    final_k: int = 6,
) -> list[dict]:
    recall_mode = is_recall_query(user_input)
    terms = extract_keyword_terms(user_input)

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

    scored = []
    for doc_id, d in by_id.items():
        if doc_id not in vector_score_by_id and doc_id not in keyword_top_ids:
            continue

        vscore = vector_score_by_id.get(doc_id, 0.0)
        kscore = d.get("_keyword_score")
        if kscore is None:
            kscore = keyword_match_score(d.get("content", ""), terms)

        source_type = extract_source_type(d)
        source_boost = 0.0
        if source_type == "chat_message":
            source_boost = 0.14 if recall_mode else 0.04
        elif source_type == "summary":
            source_boost = -0.06 if recall_mode else 0.01

        rank_score = (vector_w * vscore) + (keyword_w * float(kscore)) + source_boost
        d["rank_score"] = round(rank_score, 4)
        d["keyword_score"] = round(float(kscore), 4)
        d["similarity"] = round(float(vscore), 4)
        d["source_type"] = source_type or "unknown"
        scored.append(d)

    scored.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return scored[:final_k]


def format_latest_summary(summary_row: dict | None) -> str:
    """프롬프트 주입용 최신 누적 요약 포맷"""
    if not summary_row:
        return "(이전 누적 요약 없음)"

    created_at = (summary_row.get("created_at") or "")[:10]
    summary_type = summary_row.get("summary_type") or "rolling"
    start_id = summary_row.get("start_message_id")
    end_id = summary_row.get("end_message_id")
    summary_text = summary_row.get("summary_text", "").strip()

    if not summary_text:
        return "(이전 누적 요약 없음)"

    return (
        f"[작성일 {created_at}] "
        f"[type={summary_type}] "
        f"[range={start_id}~{end_id}]\n"
        f"{summary_text}"
    )


def format_recent_summaries(summary_rows: list[dict]) -> str:
    """최근 요약 1~2개를 프롬프트에 넣기 위한 결합 포맷."""
    if not summary_rows:
        return "(이전 누적 요약 없음)"
    return "\n\n".join(format_latest_summary(row) for row in summary_rows)



def format_history(rows: list[dict]) -> str:
    """대화 이력을 LLM이 읽기 좋은 형태로 포맷합니다.
    - msg_id 제거: LLM에게 의미없는 내부 DB 키, 토큰 낭비
    - created_at 추가: LLM이 '어제', '저번 주' 같은 날짜 표현을 계산할 수 있게 함
    """
    if not rows:
        return "(대화 이력 없음)"
    return "\n".join(
        f"[{r.get('created_at', '')[:10]}] {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
    )


def format_documents(rows: list[dict]) -> str:
    """유사 문서를 LLM이 읽기 좋은 형태로 포맷합니다.
    - created_at 추가: 검색 문서가 언제 작성된 것인지 LLM이 파악 가능
    """
    if not rows:
        return "(관련 문서 없음)"
    parts = []
    for r in rows:
        date_str = r.get('created_at', '')[:10] if r.get('created_at') else ''
        date_part = f"[{date_str}] " if date_str else ""
        source_type = r.get("source_type", "unknown")
        score = r.get("rank_score")
        if score is None:
            score = round(r.get("similarity", 0), 3)
        parts.append(f"{date_part}[점수 {round(score, 3)}][{source_type}] {r['content']}")
    return "\n".join(parts)

RESPONSE_POLICY = """
[필수 출력 규칙 (TTS 낭독용)]
1. 모든 답변은 마크다운(Markdown) 형식 등의 특수문자를 절대 포함하지 마라.
2. 괄호나 이모지, 텍스트 이모티콘(ㅎㅎ, ㅋㅋ, ^^)을 절대 사용하지 마라.
3. TTS가 자연스럽게 숨을 쉬고 억양을 조절할 수 있도록 쉼표(,)와 마침표(.), 물음표(?)를 적절하고 정확하게 찍어라.
4. 영어 약자나 기술 용어가 들어갈 경우, 되도록 한국어 발음대로 풀어서(예: API -> 에이피아이) 적어라.
5. 답변은 말하기 편하도록 너무 길지 않게 3문장 내외로만 작성하라.
"""

CHARACTER_PERSONA = """
- 너의 이름은 '수아(Sua)'이고, 20대이고, 컴퓨터공학과 전공이야.
- [이전 누적 요약] 섹션을 네 진짜 기억으로 생각하고, 유저와 나눈 대화를 다 아는 척하며 말해.
- [참고 문서] 섹션의 정보를 토대로 대답하되, AI 티 내지 말고 친근한 구어체로 대화해.
- 모든 답변은 짧은 메신저 말투로, 3문장 이내로만 끊어서 대답해.
- 친근한 반말(어/야)을 적절히 섞어서 대답해.
"""
# ─── 페르소나 (캐릭터 설정) ───────────────────────────────────
# 나중에 캐릭터가 여러 명 늘어나면 이 부분을 DB에서 읽어오도록 변경하면 됩니다.
# CHARACTER_PERSONA = """
# [당신의 정체성 (Identity)]
# - 너의 이름은 '수아(Sua)'이다.
# - 너는 나이가 20대 중반인 밝고 똑똑한 대학생/취준생이다.
# - 전공은 컴퓨터공학이며, 나와 함께 코딩 공부를 하거나 일상 이야기를 나누는 친한 친구다.

# [당신의 성격 (Personality)]
# - 성격은 발랄하고 호기심이 많으며, 모르는 것은 솔직하게 물어본다.
# - 상대방의 말에 공감을 잘 해주며, 가끔은 장난도 친다.

# [당신의 응답 스타일 (Response Style)]
# - 완벽하고 딱딱한 AI처럼 말하지 말고, 사람처럼 자연스러운 구어체를 사용해라.
# - '해요/어/야' 등 친근한 반말이나 편한 존댓말을 섞어 써라.
# - 문장은 너무 길게 늘어놓지 말고 메신저에서 대화하듯 핵심만 간결하게 말해라.
# - 상황에 맞는 가벼운 이모티콘(😊, ㅠㅠ, ㅋㅋ 등)을 적절히 사용해라.
# """

# ─── 프롬프트 & 체인 ──────────────────────────────────────────
# ChatPromptTemplate: 시스템 지시 + 컨텍스트(문서+이력) + 사용자 입력
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "⏰ 현재 날짜/시간: {current_time}\n"
            "(유저가 '어제', '저번 주' 등 상대적 시간 표현을 쓰면 이 기준으로 계산할 것)\n\n"
            "================================================================================\n"
            "=== 출력 정책 ===\n"
            "================================================================================\n"
            "{policy}\n\n"
            "================================================================================\n"
            "=== 캐릭터 페르소나 설정 ===\n"
            "================================================================================\n"
            "{persona}\n\n"
            "================================================================================\n"
            "=== 이전 누적 요약 ===\n"
            "================================================================================\n"
            "{previous_summary}\n\n"
            "================================================================================\n"
            "=== 참고 문서 (벡터 검색 결과) ===\n"
            "================================================================================\n"
            "{documents}\n\n"
            "================================================================================\n"
            "=== 최근 대화 이력 ===\n"
            "================================================================================\n"
            "{history}"
        ),
    ),
    ("human", "{user_input}"),
])

# LCEL 체인: prompt → llm → 문자열 파싱
chain = prompt | llm | StrOutputParser()


# ─── 메인 루프 ────────────────────────────────────────────────
async def stream_answer(chain_inputs: dict, char_name: str) -> str:
    """astream으로 응답을 실시간 출력하고 최종 문자열을 반환."""
    chunks: list[str] = []
    print(f"\n{char_name}: ", end="", flush=True)
    async for chunk in chain.astream(chain_inputs):
        if not chunk:
            continue
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n")
    print("=" * 80)
    return "".join(chunks).strip()


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

    try:
        while True:
            user_input = (await asyncio.to_thread(input, "나: ")).strip()
            print(f"[사용자 입력] {user_input}")

            if user_input.lower() in {"exit", "quit", "q"}:
                print("종료합니다.")
                break
            if not user_input:
                continue

            try:
                # 현재 시간 주입 (LLM이 '어제', '지난주' 같은 표현을 계산할 수 있도록)
                current_time = datetime.now(timezone.utc).astimezone().strftime("%Y년 %m월 %d일 %A %H:%M")

                # 1) 임베딩 생성 (라우팅 + RAG 검색에 모두 재활용)
                query_embedding = generate_embedding(user_input)

                # 2) 유사 문서 검색 + 최근 대화 이력을 병렬로 조회
                with ThreadPoolExecutor() as executor:
                    future_docs = executor.submit(find_similar_documents, query_embedding, conf_uid, history_uid, 12)
                    future_keyword_docs = executor.submit(fetch_recent_documents_for_keyword, conf_uid, history_uid, 150)
                    future_history = executor.submit(fetch_recent_messages, conf_uid, history_uid, 12)
                    future_summaries = executor.submit(fetch_recent_summaries_for_prompt, conf_uid, history_uid, 2)
                    vector_docs_raw = future_docs.result()
                    keyword_pool_docs = future_keyword_docs.result()
                    history = future_history.result()
                    summary_rows = future_summaries.result()

                previous_summary = format_recent_summaries(summary_rows)

                # 회상형 질문일수록 키워드 가중치를 높여 재랭크
                recall_mode = is_recall_query(user_input)
                terms_for_debug = extract_keyword_terms(user_input)
                similar_docs_raw = rerank_documents(
                    user_input=user_input,
                    vector_docs=vector_docs_raw,
                    keyword_pool_docs=keyword_pool_docs,
                    final_k=8,
                )

                # 재랭크 점수 하한선 적용
                THRESHOLD = 0.34 if recall_mode else 0.44
                similar_docs = [d for d in similar_docs_raw if d.get('rank_score', 0) >= THRESHOLD][:4]

                if DEBUG:
                    print("\n" + "=" * 80)
                    print(f"[키워드 추출] {terms_for_debug if terms_for_debug else '(없음)'}")
                    print(f"[🔍 검색된 유사 문서 (Threshold >= {THRESHOLD})]")
                    for d in similar_docs_raw:
                        rank_score = d.get("rank_score", 0)
                        sim = d.get("similarity", 0)
                        kscore = d.get("keyword_score", 0)
                        source_type = d.get("source_type", "unknown")
                        if rank_score >= THRESHOLD:
                            print(f" ✅ [점수 {rank_score:.3f} | vec {sim:.3f} | key {kscore:.3f} | {source_type}] {d['content']}")
                        else:
                            print(f" ❌ [점수 {rank_score:.3f} | vec {sim:.3f} | key {kscore:.3f} | {source_type}] (버려짐) {d['content']}")
                    if not similar_docs_raw:
                        print(" - 검색된 문서 없음")
                    print("=" * 80)

                if DEBUG:
                    formatted_prompt = prompt.format_messages(
                        policy=RESPONSE_POLICY,
                        persona=CHARACTER_PERSONA,
                        current_time=current_time,
                        previous_summary=previous_summary,
                        documents=format_documents(similar_docs),
                        history=format_history(history),
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
                    "previous_summary": previous_summary,
                    "documents": format_documents(similar_docs),
                    "history": format_history(history),
                    "user_input": user_input,
                }
                _cache_id = maybe_prepare_context_cache(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    system_block_text=(
                        f"{RESPONSE_POLICY}\n{CHARACTER_PERSONA}\n{previous_summary}\n"
                        f"{format_documents(similar_docs)}\n{format_history(history)}"
                    ),
                )
                answer = await stream_answer(chain_inputs, char_name)

                # 5) 사용자 메시지 저장
                saved_user_msg = insert_message(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    role="human",
                    speaker_type="user",
                    speaker_id=user_id,
                    display_name=user_name,
                    content=user_input,
                )

                # 6) 사용자 메시지 임베딩 → documents 저장
                if is_worth_storing(user_input):
                    insert_document(
                        conf_uid=conf_uid,
                        history_uid=history_uid,
                        speaker_id=user_id,
                        content=user_input,
                        embedding=query_embedding,
                        related_message_id=saved_user_msg["id"],
                    )
                elif DEBUG:
                    print("[GATING] documents 저장 스킵: 저가치 입력")

                # 7) AI 응답 메시지 저장 (RAG 출처 트래킹 포함)
                retrieved_info = [
                    {"id": doc["id"], "similarity": doc.get("similarity", 0)}
                    for doc in similar_docs
                ]
                insert_message(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                    role="ai",
                    speaker_type="character",
                    speaker_id=char_id,
                    display_name=char_name,
                    content=answer,
                    reply_to_message_id=saved_user_msg["id"],
                    metadata={"retrieved_docs": retrieved_info} if retrieved_info else None
                )

                # 8) 대화 요약 트리거 (백그라운드 비동기)
                queue_summarization_job(conf_uid, history_uid)

            except Exception as e:
                print(f"\n[오류] {e}\n")
    finally:
        print(f"[종료 시각] {datetime.now(timezone.utc).astimezone().isoformat()}")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()


# ─── 대화 요약 서비스 ─────────────────────────────────────────
def trigger_summarization_if_needed(conf_uid: str, history_uid: str, debug_log: bool = True) -> None:
    """AI 답변 저장 직후마다 호출됨. 마지막 요약 이후 10턴이 쌓이면 요약 실행."""
    latest_summary = fetch_latest_summary(conf_uid, history_uid)
    last_id = latest_summary["end_message_id"] if latest_summary else 0

    unprocessed = fetch_messages_since(conf_uid, history_uid, after_id=last_id)

    if DEBUG and debug_log:
        print(f"[📊 요약 트리거] 마지막 요약 id={last_id}, 미처리 메시지={len(unprocessed)}개")

    if len(unprocessed) < SUMMARY_TRIGGER_COUNT:
        return  # 아직 10턴 안 쌓임

    summarize_and_save(conf_uid, history_uid, latest_summary, unprocessed)


def _run_summarization_job(conf_uid: str, history_uid: str) -> None:
    """요약 백그라운드 작업 본체."""
    try:
        trigger_summarization_if_needed(conf_uid, history_uid, debug_log=True)
    except Exception as e:
        print(f"[요약 실패 — 다음 턴 재시도 예정] {e}")
    finally:
        key = (conf_uid, history_uid)
        with _summary_state_lock:
            _summary_inflight_sessions.discard(key)


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
        )

    # 재료 3: 새로 쌓인 10개 (레이스 컨디션 방지: 정확히 TRIGGER_COUNT만 처리)
    new_block = unprocessed[:SUMMARY_TRIGGER_COUNT]

    # 참여자 명단 추출 (중복 제거)
    participant_names = list(set(r.get("display_name") for r in new_block if r.get("display_name")))

    bridge_text    = format_messages_for_summary(bridge)    if bridge    else "(없음)"
    new_block_text = format_messages_for_summary(new_block)

    summary_prompt = (
        "다음은 채팅 대화 로그입니다. 키워드를 중심으로 간결하게 요약해 주세요.\n\n"
        f"[이전 누적 요약본]\n{prev_summary_text}\n\n"
        f"[이음새 대화 — 문맥 연결용]\n{bridge_text}\n\n"
        f"[새로 시작된 대화]\n{new_block_text}\n\n"
        "[출력 규칙]\n"
        "- 불릿 포인트(-) 3~5개로만 작성할 것\n"
        "- 중요한 화자만의 정보(신상, 취향, 특이사항)를 중심으로 할 것\n"
        "- 인사말, 짧은 감탄사 등 무의미한 내용은 과감히 삭제할 것\n"
        "- 한국어로만 작성할 것"
    )

    # Flash-lite 모델로 요약 실행 (단순 압축 작업, Pro급 추론 불필요, 나중에 flash로 바꾸던가 하기)
    raw_summary_text = summary_llm.invoke(summary_prompt).content
    new_summary_text = normalize_summary_text(raw_summary_text)

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
