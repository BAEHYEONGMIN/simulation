import re
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


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
        ln = re.sub(r"\*\*(.*?)\*\*", r"\1", ln)
        ln = re.sub(r"`([^`]*)`", r"\1", ln)
        ln = re.sub(r"^\s*[*+-]\s*", "", ln)
        ln = re.sub(r"^\s*\d+[.)]\s*", "", ln)
        if ln:
            normalized.append(f"- {ln}")

    if not normalized:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        return f"- {cleaned}" if cleaned else ""

    return "\n".join(normalized[:5])


def format_latest_summary(summary_row: dict | None) -> str:
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
    if not summary_rows:
        return "(이전 누적 요약 없음)"
    return "\n\n".join(format_latest_summary(row) for row in summary_rows)


def format_long_memories(rows: list[dict]) -> str:
    if not rows:
        return "(저장된 장기 기억 없음)"
    lines = []
    for r in rows:
        mem_type = r.get("memory_type") or "unknown"
        mem_key = r.get("memory_key") or "unknown"
        mem_value = r.get("memory_value") or ""
        lines.append(f"- [{mem_type}] {mem_key}: {mem_value}")
    return "\n".join(lines)


def format_history(rows: list[dict]) -> str:
    if not rows:
        return "(대화 이력 없음)"
    return "\n".join(
        f"[{r.get('created_at', '')[:10]}] {r.get('display_name') or r.get('speaker_id')}: {r.get('content')}"
        for r in rows
    )


def to_langchain_history_messages(rows: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for r in rows:
        role = (r.get("role") or "").lower()
        content = (r.get("content") or "").strip()
        if not content:
            continue
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def format_history_messages_debug(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(대화 이력 없음)"
    lines = []
    for m in messages:
        role = "HUMAN" if isinstance(m, HumanMessage) else "AI"
        lines.append(f"[{role}] {m.content}")
    return "\n".join(lines)


def format_documents(rows: list[dict]) -> str:
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

