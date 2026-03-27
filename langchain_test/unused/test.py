import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_recent_messages(conf_uid: str, history_uid: str, limit: int = 10):
    result = (
        supabase.schema("chatbot")
        .table("chat_messages")
        .select("id, role, speaker_id, display_name, content, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def fetch_documents(conf_uid: str, limit: int = 2):
    result = (
        supabase.schema("chatbot")
        .table("documents")
        .select("id, content, metadata, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    return [r for r in rows if (r.get("metadata") or {}).get("conf_uid") == conf_uid][:limit]


def format_messages(rows):
    return "\n".join(
        f"{r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
    )


def format_documents(rows):
    if not rows:
        return "(없음)"
    return "\n".join(
        f"[doc:{r['id']}] {(r.get('metadata') or {}).get('source_type', 'unknown')} | {r['content']}"
        for r in rows
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 캐릭터 챗봇이다. 주어진 문서와 최근 대화만 참고해 답변하라."),
        (
            "human",
            "참고 문서:\n{documents}\n\n최근 대화:\n{history}\n\n현재 입력:\n{user_input}"
        ),
    ]
)

conf_uid = "sua_test_002"
history_uid = "session_001"
user_input = "내가 전에 좋아한다고 말한 장르가 뭐였지?"

messages = fetch_recent_messages(conf_uid, history_uid)
documents = fetch_documents(conf_uid)

history_text = format_messages(messages)
documents_text = format_documents(documents)

# 조회된 원본 확인
print("=== FETCHED MESSAGES ===")
for row in messages:
    print(row)

print("\n=== FETCHED DOCUMENTS ===")
for row in documents:
    print(row)

# 최종 프롬프트 조립 확인
formatted = prompt.invoke(
    {
        "documents": documents_text,
        "history": history_text,
        "user_input": user_input,
    }
)

print("\n=== FINAL PROMPT MESSAGES ===")
for msg in formatted.messages:
    print(f"[{msg.type}]")
    print(msg.content)
    print("---")