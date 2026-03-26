import threading
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    memory_type: Literal["profile", "style", "preference", "taboo", "relationship", "fact"]
    memory_key: str = Field(min_length=1, max_length=100)
    memory_value: str = Field(min_length=1, max_length=300)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    is_memory_worthy: bool = False
    memories: list[MemoryItem] = Field(default_factory=list)


_memory_state_lock = threading.Lock()
_memory_inflight_sessions: set[tuple[str, str]] = set()
_memory_last_processed_user_msg: dict[tuple[str, str], int] = {}


def format_rows_for_memory_context(rows: list[dict]) -> str:
    if not rows:
        return "(없음)"
    return "\n".join(
        f"- {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
        if (r.get("content") or "").strip()
    )


def _format_user_block_for_memory(rows: list[dict]) -> str:
    if not rows:
        return "(없음)"
    return "\n".join(
        f"- {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
        if (r.get("content") or "").strip()
    )


def get_memory_last_processed_user_id(
    conf_uid: str,
    history_uid: str,
    *,
    get_last_processed_user_id_from_db_fn: Callable[[str, str], int],
) -> int:
    key = (conf_uid, history_uid)
    cached = _memory_last_processed_user_msg.get(key)
    if cached is not None:
        return cached
    last_id = int(get_last_processed_user_id_from_db_fn(conf_uid, history_uid))
    _memory_last_processed_user_msg[key] = last_id
    return last_id


def set_memory_last_processed_user_id(conf_uid: str, history_uid: str, last_id: int) -> None:
    _memory_last_processed_user_msg[(conf_uid, history_uid)] = int(last_id)


def is_memory_block_worthy(
    user_rows: list[dict],
    *,
    is_worth_storing_fn: Callable[[str], bool],
    memory_signal_patterns: list[str],
) -> tuple[bool, int]:
    """5턴 사용자 발화 블록에서 장기기억 추출 가치 여부를 점수로 판단."""
    score = 0
    signal_hits = 0
    for r in user_rows:
        text = (r.get("content") or "").strip()
        if not text:
            continue
        # 메모리 배치 게이팅은 per-turn 라우팅 문맥이 없으므로 텍스트 규칙만 사용.
        if not is_worth_storing_fn(text):
            score -= 1
            continue
        has_signal = any(p in text for p in memory_signal_patterns)
        if has_signal:
            score += 2
            signal_hits += 1
        else:
            if text.endswith("?") or text.endswith("？"):
                score -= 1
            elif len(text) >= 8:
                score += 1

    if signal_hits >= 1 and score >= 1:
        return True, score
    if score >= 3:
        return True, score
    return False, score


def extract_and_upsert_memories_for_block(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    user_block_rows: list[dict],
    block_end_user_message_id: int,
    summary_llm,
    memory_context_messages: int,
    fetch_recent_messages_fn: Callable[..., list[dict]],
    should_exclude_from_context_fn: Callable[[dict], bool],
    fetch_active_memories_fn: Callable[..., list[dict]],
    format_long_memories_fn: Callable[[list[dict]], str],
    build_memory_extraction_prompt_fn: Callable[..., str],
    upsert_user_memory_fn: Callable[..., None],
) -> None:
    history_rows = fetch_recent_messages_fn(
        conf_uid,
        history_uid,
        memory_context_messages,
        exclude_fn=should_exclude_from_context_fn,
    )
    existing_memories = fetch_active_memories_fn(
        conf_uid=conf_uid,
        owner_speaker_id=owner_speaker_id,
        target_speaker_id=target_speaker_id,
        limit=6,
    )
    extractor_prompt = build_memory_extraction_prompt_fn(
        recent_dialogue=format_rows_for_memory_context(history_rows),
        prior_memories=format_long_memories_fn(existing_memories),
        block_user_inputs=_format_user_block_for_memory(user_block_rows),
    )

    extractor = summary_llm.with_structured_output(MemoryExtractionResult)
    extracted = extractor.invoke(extractor_prompt)

    if not extracted or not extracted.is_memory_worthy:
        return

    for item in extracted.memories:
        upsert_user_memory_fn(
            conf_uid=conf_uid,
            owner_speaker_id=owner_speaker_id,
            target_speaker_id=target_speaker_id,
            memory_type=item.memory_type,
            memory_key=item.memory_key,
            memory_value=item.memory_value,
            confidence=item.confidence,
            source_history_uid=history_uid,
            source_message_id=block_end_user_message_id,
        )


def trigger_memory_extraction_if_needed(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    debug_log: bool,
    debug_enabled: bool,
    memory_trigger_user_turns: int,
    background_max_blocks_per_run: int,
    memory_signal_patterns: list[str],
    summary_llm,
    memory_context_messages: int,
    fetch_recent_messages_fn: Callable[..., list[dict]],
    fetch_user_messages_since_fn: Callable[..., list[dict]],
    get_last_processed_user_id_from_db_fn: Callable[[str, str], int],
    should_exclude_from_context_fn: Callable[[dict], bool],
    is_worth_storing_fn: Callable[[str], bool],
    fetch_active_memories_fn: Callable[..., list[dict]],
    format_long_memories_fn: Callable[[list[dict]], str],
    build_memory_extraction_prompt_fn: Callable[..., str],
    upsert_user_memory_fn: Callable[..., None],
    logger_fn: Callable[[str], None] = print,
) -> None:
    last_processed_user_id = get_memory_last_processed_user_id(
        conf_uid,
        history_uid,
        get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_db_fn,
    )
    pending_user_rows = fetch_user_messages_since_fn(conf_uid, history_uid, after_id=last_processed_user_id)
    processed_blocks = 0
    while len(pending_user_rows) >= memory_trigger_user_turns and processed_blocks < background_max_blocks_per_run:
        block_rows = pending_user_rows[:memory_trigger_user_turns]
        block_end_id = int(block_rows[-1]["id"])
        worthy, gate_score = is_memory_block_worthy(
            block_rows,
            is_worth_storing_fn=is_worth_storing_fn,
            memory_signal_patterns=memory_signal_patterns,
        )

        if debug_log and debug_enabled:
            logger_fn(
                f"[🧠 메모리 트리거] 마지막 처리 user_id={last_processed_user_id}, "
                f"남은 사용자발화={len(pending_user_rows)}개, 게이트점수={gate_score}"
            )

        if worthy:
            extract_and_upsert_memories_for_block(
                conf_uid=conf_uid,
                history_uid=history_uid,
                owner_speaker_id=owner_speaker_id,
                target_speaker_id=target_speaker_id,
                user_block_rows=block_rows,
                block_end_user_message_id=block_end_id,
                summary_llm=summary_llm,
                memory_context_messages=memory_context_messages,
                fetch_recent_messages_fn=fetch_recent_messages_fn,
                should_exclude_from_context_fn=should_exclude_from_context_fn,
                fetch_active_memories_fn=fetch_active_memories_fn,
                format_long_memories_fn=format_long_memories_fn,
                build_memory_extraction_prompt_fn=build_memory_extraction_prompt_fn,
                upsert_user_memory_fn=upsert_user_memory_fn,
            )
        elif debug_log and debug_enabled:
            logger_fn("[🧠 메모리 스킵] 5턴 블록 게이팅 미통과")

        set_memory_last_processed_user_id(conf_uid, history_uid, block_end_id)
        last_processed_user_id = block_end_id
        pending_user_rows = pending_user_rows[memory_trigger_user_turns:]
        processed_blocks += 1

    if debug_log and debug_enabled and processed_blocks >= background_max_blocks_per_run:
        logger_fn(f"[메모리 캐치업 제한] 1회 최대 {background_max_blocks_per_run}블록 처리 후 종료")


def _run_memory_job(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    debug_enabled: bool,
    memory_trigger_user_turns: int,
    background_max_blocks_per_run: int,
    memory_signal_patterns: list[str],
    summary_llm,
    memory_context_messages: int,
    fetch_recent_messages_fn: Callable[..., list[dict]],
    fetch_user_messages_since_fn: Callable[..., list[dict]],
    get_last_processed_user_id_from_db_fn: Callable[[str, str], int],
    should_exclude_from_context_fn: Callable[[dict], bool],
    is_worth_storing_fn: Callable[[str], bool],
    fetch_active_memories_fn: Callable[..., list[dict]],
    format_long_memories_fn: Callable[[list[dict]], str],
    build_memory_extraction_prompt_fn: Callable[..., str],
    upsert_user_memory_fn: Callable[..., None],
    logger_fn: Callable[[str], None] = print,
) -> None:
    try:
        trigger_memory_extraction_if_needed(
            conf_uid=conf_uid,
            history_uid=history_uid,
            owner_speaker_id=owner_speaker_id,
            target_speaker_id=target_speaker_id,
            debug_log=False,
            debug_enabled=debug_enabled,
            memory_trigger_user_turns=memory_trigger_user_turns,
            background_max_blocks_per_run=background_max_blocks_per_run,
            memory_signal_patterns=memory_signal_patterns,
            summary_llm=summary_llm,
            memory_context_messages=memory_context_messages,
            fetch_recent_messages_fn=fetch_recent_messages_fn,
            fetch_user_messages_since_fn=fetch_user_messages_since_fn,
            get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_db_fn,
            should_exclude_from_context_fn=should_exclude_from_context_fn,
            is_worth_storing_fn=is_worth_storing_fn,
            fetch_active_memories_fn=fetch_active_memories_fn,
            format_long_memories_fn=format_long_memories_fn,
            build_memory_extraction_prompt_fn=build_memory_extraction_prompt_fn,
            upsert_user_memory_fn=upsert_user_memory_fn,
            logger_fn=logger_fn,
        )
    except Exception as e:
        logger_fn(f"[장기기억 추출 실패 — 다음 턴 재시도 가능] {e}")
    finally:
        key = (conf_uid, history_uid)
        try:
            with _memory_state_lock:
                _memory_inflight_sessions.discard(key)
        except Exception as e:
            logger_fn(f"[장기기억 상태 해제 실패] {e}")


def queue_memory_extraction_job(
    *,
    conf_uid: str,
    history_uid: str,
    owner_speaker_id: str,
    target_speaker_id: str,
    debug_enabled: bool,
    memory_trigger_user_turns: int,
    background_max_blocks_per_run: int,
    memory_signal_patterns: list[str],
    summary_llm,
    memory_context_messages: int,
    fetch_recent_messages_fn: Callable[..., list[dict]],
    fetch_user_messages_since_fn: Callable[..., list[dict]],
    get_last_processed_user_id_from_db_fn: Callable[[str, str], int],
    should_exclude_from_context_fn: Callable[[dict], bool],
    is_worth_storing_fn: Callable[[str], bool],
    fetch_active_memories_fn: Callable[..., list[dict]],
    format_long_memories_fn: Callable[[list[dict]], str],
    build_memory_extraction_prompt_fn: Callable[..., str],
    upsert_user_memory_fn: Callable[..., None],
    logger_fn: Callable[[str], None] = print,
) -> None:
    key = (conf_uid, history_uid)
    with _memory_state_lock:
        if key in _memory_inflight_sessions:
            return
        _memory_inflight_sessions.add(key)

    worker = threading.Thread(
        target=_run_memory_job,
        kwargs={
            "conf_uid": conf_uid,
            "history_uid": history_uid,
            "owner_speaker_id": owner_speaker_id,
            "target_speaker_id": target_speaker_id,
            "debug_enabled": debug_enabled,
            "memory_trigger_user_turns": memory_trigger_user_turns,
            "background_max_blocks_per_run": background_max_blocks_per_run,
            "memory_signal_patterns": memory_signal_patterns,
            "summary_llm": summary_llm,
            "memory_context_messages": memory_context_messages,
            "fetch_recent_messages_fn": fetch_recent_messages_fn,
            "fetch_user_messages_since_fn": fetch_user_messages_since_fn,
            "get_last_processed_user_id_from_db_fn": get_last_processed_user_id_from_db_fn,
            "should_exclude_from_context_fn": should_exclude_from_context_fn,
            "is_worth_storing_fn": is_worth_storing_fn,
            "fetch_active_memories_fn": fetch_active_memories_fn,
            "format_long_memories_fn": format_long_memories_fn,
            "build_memory_extraction_prompt_fn": build_memory_extraction_prompt_fn,
            "upsert_user_memory_fn": upsert_user_memory_fn,
            "logger_fn": logger_fn,
        },
        daemon=True,
    )
    worker.start()


def get_memory_progress(
    conf_uid: str,
    history_uid: str,
    *,
    fetch_user_messages_since_fn: Callable[..., list[dict]],
    get_last_processed_user_id_from_db_fn: Callable[[str, str], int],
) -> tuple[int, int]:
    """메모리 진행상태 조회: (마지막 처리 user_message_id, 미처리 사용자 발화 수)."""
    last_user_id = get_memory_last_processed_user_id(
        conf_uid,
        history_uid,
        get_last_processed_user_id_from_db_fn=get_last_processed_user_id_from_db_fn,
    )
    pending_rows = fetch_user_messages_since_fn(conf_uid, history_uid, after_id=last_user_id)
    return last_user_id, len(pending_rows)

