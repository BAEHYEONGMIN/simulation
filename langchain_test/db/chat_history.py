from collections.abc import Callable

try:
    from chat_constants import SCHEMA, TABLE_CHAT, FETCH_HISTORY_MULTIPLIER
    from db.common import supabase, now_iso
except ImportError:
    from langchain_test.chat_constants import SCHEMA, TABLE_CHAT, FETCH_HISTORY_MULTIPLIER
    from langchain_test.db.common import supabase, now_iso


def fetch_recent_messages(
    conf_uid: str,
    history_uid: str,
    limit: int = 10,
    exclude_fn: Callable[[dict], bool] | None = None,
) -> list[dict]:
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .select("id, role, speaker_id, display_name, content, created_at, metadata")
        .eq("conf_uid", conf_uid)
        .eq("history_uid", history_uid)
        .order("created_at", desc=True)
        .limit(limit * FETCH_HISTORY_MULTIPLIER)
        .execute()
    )
    rows = list(reversed(result.data or []))
    if exclude_fn is not None:
        rows = [r for r in rows if not exclude_fn(r)]
    return rows[-limit:] if len(rows) > limit else rows


def insert_message(
    conf_uid: str,
    history_uid: str,
    role: str,
    speaker_type: str,
    speaker_id: str,
    display_name: str,
    content: str,
    reply_to_message_id: int | None = None,
    metadata: dict | None = None,
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
