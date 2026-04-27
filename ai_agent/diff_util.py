import re
from difflib import SequenceMatcher
from diff_match_patch import diff_match_patch


UNIT_PATTERN = re.compile(
    r"(?m)^(?P<num>(?:\d+의\d+|\d+|[가-힣])\.)\s*(?P<body>.*)"
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_numbered_units(text: str) -> dict[str, str]:
    """
    조문 텍스트 안의 호/목 단위를 번호 기준으로 분리합니다.

    예:
    1. ...
    1의2. ...
    2. ...
    가. ...
    나. ...

    반환:
    {
        "1.": "1. ...",
        "1의2.": "1의2. ...",
        "2.": "2. ..."
    }
    """
    text = normalize_text(text)
    matches = list(UNIT_PATTERN.finditer(text))

    if not matches:
        return {"본문": text} if text else {}

    units = {}

    # 번호 목록 앞의 본문, 예: "제62조(벌칙) ..."
    prefix = text[:matches[0].start()].strip()
    if prefix:
        units["본문"] = prefix

    for i, match in enumerate(matches):
        key = match.group("num")
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        units[key] = text[start:end].strip()

    return units


def sort_unit_key(key: str):
    """
    법령 번호 정렬용.
    1. < 1의2. < 2. < 2의2. < 가. < 나.
    """
    if key == "본문":
        return (0, 0, 0, "")

    clean = key.rstrip(".")

    m = re.match(r"^(\d+)(?:의(\d+))?$", clean)
    if m:
        main = int(m.group(1))
        sub = int(m.group(2) or 0)
        return (1, main, sub, "")

    # 한글 목: 가, 나, 다...
    hangul_order = {
        "가": 1, "나": 2, "다": 3, "라": 4, "마": 5,
        "바": 6, "사": 7, "아": 8, "자": 9, "차": 10,
        "카": 11, "타": 12, "파": 13, "하": 14,
    }

    return (2, hangul_order.get(clean, 999), 0, clean)


def inline_diff(old_text: str, new_text: str, markdown: bool = False) -> str:
    """
    같은 번호 단위 안에서만 세부 diff를 수행합니다.
    """
    old_text = normalize_text(old_text)
    new_text = normalize_text(new_text)

    if old_text == new_text:
        return old_text

    dmp = diff_match_patch()
    diffs = dmp.diff_main(old_text, new_text)
    dmp.diff_cleanupSemantic(diffs)

    parts = []
    for op, text in diffs:
        if not text:
            continue

        if op == 0:
            parts.append(text)
        elif op == -1:
            if markdown:
                parts.append(f"[삭제: {text}]")
            else:
                parts.append(f"[삭제: {text}]")
        elif op == 1:
            if markdown:
                parts.append(f"[추가: {text}]")
            else:
                parts.append(f"[추가: {text}]")

    return "".join(parts)


def compare_units(old_text: str, new_text: str, markdown: bool = False) -> str:
    old_units = parse_numbered_units(old_text)
    new_units = parse_numbered_units(new_text)

    all_keys = sorted(
        set(old_units.keys()) | set(new_units.keys()),
        key=sort_unit_key,
    )

    result = []

    for key in all_keys:
        old_value = old_units.get(key)
        new_value = new_units.get(key)

        if old_value is None:
            result.append(f"[신설] {key}\n{new_value}")
            continue

        if new_value is None:
            result.append(f"[삭제] {key}\n{old_value}")
            continue

        if normalize_text(old_value) == normalize_text(new_value):
            continue

        result.append(
            f"[변경] {key}\n"
            f"{inline_diff(old_value, new_value, markdown=markdown)}"
        )

    if not result:
        return "변경 없음"

    return "\n\n".join(result)


def get_text_diff(old_text, new_text):
    """
    사람이 읽기 쉬운 태그형 diff.
    법령 번호 단위로 먼저 비교한 뒤, 같은 번호 안에서만 세부 diff를 수행합니다.
    """
    return compare_units(old_text, new_text, markdown=False)


def get_markdown_diff(old_text, new_text):
    """
    LLM에게 전달하기 좋은 diff.
    Markdown의 ~~삭제~~, **추가**는 법령 번호와 충돌할 수 있으므로 쓰지 않습니다.
    """
    return compare_units(old_text, new_text, markdown=True)