try:
    from chat_constants import SCHEMA, TABLE_DOCUMENTS, RPC_MATCH
    from db.common import supabase, now_iso
except ImportError:
    from langchain_test.chat_constants import SCHEMA, TABLE_DOCUMENTS, RPC_MATCH
    from langchain_test.db.common import supabase, now_iso


def find_similar_documents(
    query_embedding: list[float],
    conf_uid: str,
    history_uid: str | None = None,
    top_k: int = 20,
) -> list[dict]:
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
