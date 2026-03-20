import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 로컬 임베딩 모델 (384차원)
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_embedding(text: str) -> list[float]:
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def insert_chat_message(
    conf_uid: str,
    history_uid: str,
    speaker_id: str,
    display_name: str,
    content: str,
) -> dict:
    payload = {
        "conf_uid": conf_uid,
        "history_uid": history_uid,
        "role": "human",
        "speaker_type": "user",
        "speaker_id": speaker_id,
        "display_name": display_name,
        "content": content,
        "avatar": None,
        "reply_to_message_id": None,
        "created_at": now_iso(),
    }

    result = (
        supabase.schema("chatbot")
        .table("chat_messages")
        .insert(payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError("chat_messages 저장 실패")

    return result.data[0]


def insert_document_from_message(
    conf_uid: str,
    history_uid: str,
    speaker_id: str,
    content: str,
    related_message_id: int,
    embedding: list[float],
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
        supabase.schema("chatbot")
        .table("documents")
        .insert(payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError("documents 저장 실패")

    return result.data[0]


def find_similar_documents(
    query_embedding: list[float],
    conf_uid: str,
    exclude_document_id: int | None = None,
    top_k: int = 3,
) -> list[dict]:
    # 자기 자신 제외하려고 넉넉히 4~5개 가져온다
    fetch_count = top_k + 2

    result = (
        supabase.schema("chatbot")
        .rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": fetch_count,
                "filter": {"conf_uid": conf_uid},
            },
        )
        .execute()
    )

    rows = result.data or []

    filtered = []
    for row in rows:
        if exclude_document_id is not None and row["id"] == exclude_document_id:
            continue
        filtered.append(row)

    return filtered[:top_k]


def fetch_recent_messages(conf_uid: str, history_uid: str, limit: int = 10):
    result = (
        supabase.schema("chatbot")
        .table("chat_messages")
        .select("id, role, speaker_id, display_name, content, created_at")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    return list(reversed(rows))


if __name__ == "__main__":
    conf_uid = "sua_test_002"
    history_uid = "session_001"
    speaker_id = "user_baemin"
    display_name = "배민"

    print("입력 루프 시작. 종료하려면 exit / quit / q 입력")
    print("-" * 50)

    while True:
        user_input = input("사용자 입력: ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("종료합니다.")
            break

        if not user_input:
            print("빈 입력은 건너뜁니다.")
            print("-" * 50)
            continue

        try:
            # 1) 원문 메시지 저장
            inserted_message = insert_chat_message(
                conf_uid=conf_uid,
                history_uid=history_uid,
                speaker_id=speaker_id,
                display_name=display_name,
                content=user_input,
            )

            # 2) 로컬 임베딩 생성
            query_embedding = generate_embedding(user_input)

            # 3) documents에 임베딩 저장
            inserted_doc = insert_document_from_message(
                conf_uid=conf_uid,
                history_uid=history_uid,
                speaker_id=speaker_id,
                content=user_input,
                related_message_id=inserted_message["id"],
                embedding=query_embedding,
            )

            # 4) 유사 문서 top 3 찾기
            similar_docs = find_similar_documents(
                query_embedding=query_embedding,
                conf_uid=conf_uid,
                exclude_document_id=inserted_doc["id"],  # 자기 자신 제외
                top_k=3,
            )

            # 출력
            print("\n=== INSERTED CHAT MESSAGE ===")
            print(inserted_message)

            print("\n=== INSERTED DOCUMENT ===")
            print(inserted_doc)

            print("\n=== SIMILAR DOCUMENTS TOP 3 ===")
            if not similar_docs:
                print("유사 문서 없음")
            else:
                for i, doc in enumerate(similar_docs, start=1):
                    similarity = doc.get("similarity")

                    print(f"\n[{i}]")
                    print("id:", doc["id"])
                    if similarity is None:
                        print("similarity: None")
                    else:
                        print("similarity:", round(similarity, 4))
                    print("content:", doc["content"])
                    print("metadata:", doc["metadata"])

            print("\n=== RECENT CHAT MESSAGES ===")
            recent = fetch_recent_messages(conf_uid, history_uid, limit=10)
            for row in recent:
                print(row)

            print("\n" + "-" * 50)

        except Exception as e:
            print(f"\n오류 발생: {e}")
            print("-" * 50)