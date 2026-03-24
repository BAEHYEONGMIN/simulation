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

# ─── DB 상수 ─────────────────────────────────────────────────
SCHEMA          = "chatbot"
TABLE_CHAT      = "chat_messages"
TABLE_DOCUMENTS = "documents_gemini"   # 768차원 (gemini-embedding-001)
RPC_MATCH       = "match_documents_gemini"

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


# ─── 포맷 헬퍼 ───────────────────────────────────────────────
def format_history(rows: list[dict]) -> str:
    if not rows:
        return "(대화 이력 없음)"
    return "\n".join(
        f"[msg_id:{r.get('id', '?')}] {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
    )


def format_documents(rows: list[dict]) -> str:
    if not rows:
        return "(관련 문서 없음)"
    return "\n".join(
        f"[유사도 {round(r.get('similarity', 0), 3)}] {r['content']}"
        for r in rows
    )

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
            # 1) 임베딩 생성
            query_embedding = generate_embedding(user_input)

            # 2) 유사 문서 검색 (RAG 핵심)
            similar_docs_raw = find_similar_documents(
                query_embedding=query_embedding,
                conf_uid=conf_uid,
                top_k=3,
            )

            # 유사도 하한선(Threshold) 적용: 0.75 미만의 쓰레기 데이터 필터링
            THRESHOLD = 0.75
            similar_docs = [d for d in similar_docs_raw if d.get('similarity', 0) >= THRESHOLD]

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

            # 3) 최근 대화 이력 조회
            history = fetch_recent_messages(conf_uid, history_uid, limit=8)
            
            # 4) 프롬프트 조립 로직 (출력용으로 먼저 포맷팅)
            formatted_prompt = prompt.format_messages(
                policy=RESPONSE_POLICY,
                persona=CHARACTER_PERSONA,
                documents=format_documents(similar_docs),
                history=format_history(history),
                user_input=user_input,
            )
            print(f"\n[🧠 LLM에 주입된 전체 프롬프트 전문]")
            for msg in formatted_prompt:
                print(f"[{msg.type.upper()}]")
                print(f"{msg.content}\n")
            print("=" * 80)

            # 5) LLM 호출 (LCEL 체인)
            answer = chain.invoke({
                "policy":     RESPONSE_POLICY,
                "persona":    CHARACTER_PERSONA,
                "documents":  format_documents(similar_docs),
                "history":    format_history(history),
                "user_input": user_input,
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

        except Exception as e:
            print(f"\n[오류] {e}\n")
