try:
    from chat_constants import SCHEMA, TABLE_SUMMARIES
    from db.common import supabase, now_iso
except ImportError:
    from langchain_test.chat_constants import SCHEMA, TABLE_SUMMARIES
    from langchain_test.db.common import supabase, now_iso


def fetch_latest_summary(conf_uid: str, history_uid: str) -> dict | None:
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


def insert_summary(
    conf_uid: str,
    history_uid: str,
    summary: str,
    start_message_id: int,
    end_message_id: int,
    covered_message_count: int,
    summary_seq: int,
    participants: list | None = None,
    summary_type: str = "rolling",
) -> dict:
    payload = {
        "conf_uid": conf_uid,
        "history_uid": history_uid,
        "summary_text": summary,
        "summary_type": summary_type,
        "start_message_id": start_message_id,
        "end_message_id": end_message_id,
        "covered_message_count": covered_message_count,
        "summary_seq": summary_seq,
        "participants": participants,
        "created_at": now_iso(),
    }
    result = supabase.schema(SCHEMA).table(TABLE_SUMMARIES).insert(payload).execute()
    if not result.data:
        raise RuntimeError(f"{TABLE_SUMMARIES} 저장 실패")
    return result.data[0]
