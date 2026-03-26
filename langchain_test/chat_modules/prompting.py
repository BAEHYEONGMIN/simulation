try:
    from chat_constants import (
        PROMPTS_DIR,
        INPUT_MODE_NORMAL,
        INPUT_MODE_OOC,
        INPUT_MODE_IC,
        INPUT_MODE_MIXED,
    )
except ImportError:
    from langchain_test.chat_constants import (
        PROMPTS_DIR,
        INPUT_MODE_NORMAL,
        INPUT_MODE_OOC,
        INPUT_MODE_IC,
        INPUT_MODE_MIXED,
    )


def load_prompt_text(file_name: str, fallback_text: str | None = None) -> str:
    try:
        path = PROMPTS_DIR / file_name
        if not path.exists():
            if fallback_text is None:
                raise FileNotFoundError(f"Prompt file not found: {path}")
            return fallback_text.strip()
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
        if fallback_text is None:
            raise ValueError(f"Prompt file is empty: {path}")
        return fallback_text.strip()
    except Exception:
        if fallback_text is None:
            raise
        return fallback_text.strip()


MODE_INSTRUCTIONS = {
    INPUT_MODE_NORMAL: load_prompt_text("mode_instruction_normal.txt"),
    INPUT_MODE_OOC: load_prompt_text("mode_instruction_ooc.txt"),
    INPUT_MODE_IC: load_prompt_text("mode_instruction_ic.txt"),
    INPUT_MODE_MIXED: load_prompt_text("mode_instruction_mixed.txt"),
}


def get_mode_instruction(input_mode: str) -> str:
    mode = (input_mode or INPUT_MODE_NORMAL).upper()
    return MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS[INPUT_MODE_NORMAL])


RESPONSE_POLICY = load_prompt_text("response_policy.txt")


CHARACTER_PERSONA = load_prompt_text("character_persona_sua.txt")


SYSTEM_PROMPT_TEMPLATE = load_prompt_text("chat_system_template.txt")


SUMMARY_PROMPT_TEMPLATE = load_prompt_text("summary_prompt_template.txt")


def build_memory_extraction_prompt(
    *,
    recent_dialogue: str,
    prior_memories: str,
    block_user_inputs: str,
) -> str:
    template = load_prompt_text("memory_extraction_prompt_template.txt")
    return template.format(
        prior_memories=prior_memories,
        block_user_inputs=block_user_inputs,
        recent_dialogue=recent_dialogue,
    )
