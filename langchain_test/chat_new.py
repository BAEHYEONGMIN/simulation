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
    CHAT_MODEL, EMBEDDING_MODEL,
)

# 디버그 모드: True이면 검색 결과 및 프롬프트 전문을 터미널에 출력합니다.
# 서버화(FastAPI) 시에는 False로 변경하세요.
DEBUG = True

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


# ─── 유틸 ────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    top_k: int = 3,
) -> list[dict]:
    """벡터 유사도 검색"""
    result = (
        supabase.schema(SCHEMA)
        .rpc(
            RPC_MATCH,
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter": {"conf_uid": conf_uid},
            },
        )
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
) -> dict:
    payload = {
        "content": content,
        "metadata": {
            "conf_uid": conf_uid,
            "history_uid": history_uid,
            "speaker_id": speaker_id,
            "speaker_type": "user",
            "source_type": "chat_message",
            "related_message_id": related_message_id,
        },
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
        .select("id, summary, start_message_id, end_message_id, message_count, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


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
    message_count: int,
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
    │    └─ message_count    : 요약된 대화 수
    │
    └─────────────────────────────────────────────────────────────────────────────
    """
    payload = {
        "conf_uid":         conf_uid,
        "history_uid":      history_uid,
        "summary":          summary,
        "summary_type":     summary_type,
        "start_message_id": start_message_id,
        "end_message_id":   end_message_id,
        "message_count":    message_count,
        "created_at":       now_iso(),
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
        parts.append(f"{date_part}[유사도 {round(r.get('similarity', 0), 3)}] {r['content']}")
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
- 너는 친근한 AI 챗봇이다.
- 아래 참고 문서와 최근 대화 이력을 바탕으로 답변하라.
- 모르는 내용은 지어내지 말고 솔직하게 말하라.
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
if __name__ == "__main__":
    conf_uid    = "sua_test_002"
    history_uid = "session_001"
    user_id     = "user_baemin"
    user_name   = "배민"
    char_id     = "char_sua"
    char_name   = "수아"

    print(f"[모델] LLM={CHAT_MODEL} / Embedding={EMBEDDING_MODEL}")
    print("대화를 시작합니다. 종료: exit / quit / q")
    print("=" * 80)

    while True:
        user_input = input("나: ").strip()

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
            #    두 작업은 서로 독립적이므로 ThreadPoolExecutor로 동시 실행하여 응답 속도 개선
            with ThreadPoolExecutor() as executor:
                future_docs    = executor.submit(find_similar_documents, query_embedding, conf_uid, 3)
                future_history = executor.submit(fetch_recent_messages, conf_uid, history_uid, 8)
                similar_docs_raw = future_docs.result()
                history          = future_history.result()

            # 유사도 하한선(Threshold) 적용: 0.75 미만의 쓰레기 데이터 필터링
            THRESHOLD = 0.75
            similar_docs = [d for d in similar_docs_raw if d.get('similarity', 0) >= THRESHOLD]

            if DEBUG:
                print("\n" + "=" * 80)
                print(f"[🔍 검색된 유사 문서 (Threshold >= {THRESHOLD})]")
                for d in similar_docs_raw:
                    sim = d.get('similarity', 0)
                    if sim >= THRESHOLD:
                        print(f" ✅ [유사도 {sim:.3f}] {d['content']}")
                    else:
                        print(f" ❌ [유사도 {sim:.3f}] (버려짐) {d['content']}")
                if not similar_docs_raw:
                    print(" - 검색된 문서 없음")
                print("=" * 80)

            if DEBUG:
                # 프롬프트 전문 출력 (개발/디버그 용도)
                formatted_prompt = prompt.format_messages(
                    policy=RESPONSE_POLICY,
                    persona=CHARACTER_PERSONA,
                    current_time=current_time,
                    documents=format_documents(similar_docs),
                    history=format_history(history),
                    user_input=user_input,
                )
                print(f"\n[🧠 LLM에 주입된 전체 프롬프트 전문]")
                for msg in formatted_prompt:
                    print(f"[{msg.type.upper()}]")
                    print(f"{msg.content}\n")
                print("=" * 80)

            # 3) LLM 호출 (LCEL 체인)
            answer = chain.invoke({
                "policy":       RESPONSE_POLICY,
                "persona":      CHARACTER_PERSONA,
                "current_time": current_time,
                "documents":    format_documents(similar_docs),
                "history":      format_history(history),
                "user_input":   user_input,
            })

            print(f"\n{char_name}: {answer}\n")
            print("=" * 80)

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
            insert_document(
                conf_uid=conf_uid,
                history_uid=history_uid,
                speaker_id=user_id,
                content=user_input,
                embedding=query_embedding,
                related_message_id=saved_user_msg["id"],
            )

            # 7) AI 응답 메시지 저장 (임베딩 없이 로그만 남기되, RAG 출처 트래킹)
            # 검색했던 문서들의 id와 점수만 간추려서 메타데이터로 저장
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

            # 8) 대화 요약 트리거 (마지막 요약 이후 10턴 쌓이면 자동 실행)
            try:
                trigger_summarization_if_needed(conf_uid, history_uid)
            except Exception as e:
                # 요약 실패해도 메인 응답 흐름은 절대 막히지 않음
                # 다음 턴 진입 시 unprocessed >= 10 조건이 다시 충족되어 자동 재시도됨
                print(f"[요약 실패 — 다음 턴 재시도 예정] {e}")

        except Exception as e:
            print(f"\n[오류] {e}\n")


# ─── 대화 요약 서비스 ─────────────────────────────────────────
def trigger_summarization_if_needed(conf_uid: str, history_uid: str) -> None:
    """AI 답변 저장 직후마다 호출됨. 마지막 요약 이후 10턴이 쌓이면 요약 실행."""
    latest_summary = fetch_latest_summary(conf_uid, history_uid)
    last_id = latest_summary["end_message_id"] if latest_summary else 0

    unprocessed = fetch_messages_since(conf_uid, history_uid, after_id=last_id)

    if DEBUG:
        print(f"[📊 요약 트리거] 마지막 요약 id={last_id}, 미처리 메시지={len(unprocessed)}개")

    if len(unprocessed) < SUMMARY_TRIGGER_COUNT:
        return  # 아직 10턴 안 쌓임

    summarize_and_save(conf_uid, history_uid, latest_summary, unprocessed)


def summarize_and_save(
    conf_uid: str,
    history_uid: str,
    latest_summary: dict | None,
    unprocessed: list[dict],
) -> None:
    """누적 요약을 생성하고 conversation_summaries + documents_gemini에 저장합니다."""

    # 재료 1: 이전 누적 요약본
    prev_summary_text = (
        latest_summary["summary"] if latest_summary
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

    bridge_text    = format_messages_for_summary(bridge)    if bridge    else "(없음)"
    new_block_text = format_messages_for_summary(new_block)

    summary_prompt = (
        "다음은 카카오톡 대화 로그입니다. 키워드를 중심으로 간결하게 요약해 주세요.\n\n"
        f"[이전 누적 요약본]\n{prev_summary_text}\n\n"
        f"[이음새 대화 — 문맥 연결용]\n{bridge_text}\n\n"
        f"[새로 시작된 대화]\n{new_block_text}\n\n"
        "[출력 규칙]\n"
        "- 불릿 포인트(-) 3~5개로만 작성할 것\n"
        "- 중요한 화자만의 정보(신상, 취향, 특이사항)를 중심으로 할 것\n"
        "- 인사말, 짧은 감탄사 등 무의미한 내용은 과감히 삭제할 것\n"
        "- 한국어로만 작성할 것"
    )

    # Flash 모델로 요약 실행 (단순 압축 작업, Pro급 추론 불필요)
    new_summary_text = llm.invoke(summary_prompt).content

    if DEBUG:
        print(f"\n[📝 누적 요약 완료]\n{new_summary_text}\n")

    # 저장 1: conversation_summaries 테이블 (SQL 원본)
    insert_summary(
        conf_uid=conf_uid,
        history_uid=history_uid,
        summary=new_summary_text,
        start_message_id=(last_id + 1) if last_id else 0,
        end_message_id=new_block[-1]["id"],
        message_count=len(new_block),
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
    )
