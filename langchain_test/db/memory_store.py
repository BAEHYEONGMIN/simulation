try:
    from chat_constants import (
        SCHEMA,
        TABLE_MEMORIES,
        ALLOWED_MEMORY_TYPES,
        ALLOWED_MEMORY_KEYS,
        DEFAULT_IMPORTANCE_BY_TYPE,
        MEMORY_NORMALIZATION_MAP,
    )
    from db.common import supabase, now_iso
except ImportError:
    from langchain_test.chat_constants import (
        SCHEMA,
        TABLE_MEMORIES,
        ALLOWED_MEMORY_TYPES,
        ALLOWED_MEMORY_KEYS,
        DEFAULT_IMPORTANCE_BY_TYPE,
        MEMORY_NORMALIZATION_MAP,
    )
    from langchain_test.db.common import supabase, now_iso


def fetch_active_memories(
    conf_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    limit: int = 8,
) -> list[dict]:
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_MEMORIES)
        .select("id, owner_speaker_id, target_speaker_id, memory_type, memory_key, memory_value, importance, confidence, last_seen_at")
        .eq("conf_uid", conf_uid)
        .eq("owner_speaker_id", owner_speaker_id)
        .eq("target_speaker_id", target_speaker_id)
        .eq("status", "active")
        .order("importance", desc=True)
        .order("last_seen_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_last_processed_user_id_from_memories(conf_uid: str, history_uid: str) -> int:
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_MEMORIES)
        .select("source_message_id")
        .eq("conf_uid", conf_uid)
        .eq("source_history_uid", history_uid)
        .gt("source_message_id", 0)
        .order("source_message_id", desc=True)
        .limit(1)
        .execute()
    )
    return int(result.data[0]["source_message_id"]) if result.data else 0


def normalize_memory_value(memory_key: str, memory_value: str) -> str:
    value = (memory_value or "").strip()
    if not value:
        return value
    table = MEMORY_NORMALIZATION_MAP.get(memory_key)
    if not table:
        return value
    return table.get(value, value)


def upsert_user_memory(
    *,
    conf_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    memory_type: str,
    memory_key: str,
    memory_value: str,
    confidence: float = 0.75,
    source_history_uid: str | None = None,
    source_message_id: int | None = None,
) -> None:
    memory_type = (memory_type or "").strip()
    memory_key = (memory_key or "").strip()
    memory_value = normalize_memory_value(memory_key, memory_value)

    if memory_type not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"invalid memory_type: {memory_type}")

    allowed_keys = ALLOWED_MEMORY_KEYS.get(memory_type, set())
    if memory_key not in allowed_keys:
        raise ValueError(f"invalid memory_key for {memory_type}: {memory_key}")

    now = now_iso()
    result = (
        supabase.schema(SCHEMA)
        .table(TABLE_MEMORIES)
        .select("id, evidence_count, memory_value")
        .eq("conf_uid", conf_uid)
        .eq("owner_speaker_id", owner_speaker_id)
        .eq("target_speaker_id", target_speaker_id)
        .eq("memory_type", memory_type)
        .eq("memory_key", memory_key)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    rows = result.data or []

    if not rows:
        payload = {
            "conf_uid": conf_uid,
            "owner_speaker_id": owner_speaker_id,
            "target_speaker_id": target_speaker_id,
            "memory_type": memory_type,
            "memory_key": memory_key,
            "memory_value": memory_value,
            "importance": DEFAULT_IMPORTANCE_BY_TYPE.get(memory_type, 5),
            "confidence": confidence,
            "evidence_count": 1,
            "source_history_uid": source_history_uid,
            "source_message_id": source_message_id or 0,
            "first_seen_at": now,
            "last_seen_at": now,
            "status": "active",
            "updated_at": now,
        }
        (
            supabase.schema(SCHEMA)
            .table(TABLE_MEMORIES)
            .insert(payload)
            .execute()
        )
        return

    row = rows[0]
    row_id = row["id"]
    old_evidence = int(row.get("evidence_count") or 1)
    old_value = row.get("memory_value") or ""

    if old_value != memory_value:
        patch = {
            "memory_value": memory_value,
            "evidence_count": old_evidence + 1,
            "last_seen_at": now,
            "confidence": max(confidence, 0.7),
            "source_history_uid": source_history_uid,
            "source_message_id": source_message_id,
            "updated_at": now,
        }
    else:
        patch = {
            "evidence_count": old_evidence + 1,
            "last_seen_at": now,
            "source_history_uid": source_history_uid,
            "source_message_id": source_message_id,
            "updated_at": now,
        }

    (
        supabase.schema(SCHEMA)
        .table(TABLE_MEMORIES)
        .update(patch)
        .eq("id", row_id)
        .execute()
    )
