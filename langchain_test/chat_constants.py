import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Debug / paths
DEBUG = True
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROMPTS_DIR = BASE_DIR / "prompts"
ROUTER_SAMPLES_PATH = os.path.join(BASE_DIR, "router_samples.json")

# Provider context cache
ENABLE_PROVIDER_CONTEXT_CACHE = False
CONTEXT_CACHE_MIN_TOKENS = 32768

# DB
SCHEMA = "chatbot"
TABLE_CHAT = "chat_messages"
TABLE_SUMMARIES = "conversation_summaries"
TABLE_DOCUMENTS = "documents_gemini"
TABLE_MEMORIES = "user_memories"
RPC_MATCH = "match_documents_gemini"

# Summary trigger
SUMMARY_TRIGGER_COUNT = 10
SUMMARY_BRIDGE_COUNT = 2
BACKGROUND_MAX_BLOCKS_PER_RUN = 3

# RAG
RAG_VECTOR_TOP_K = 12
RAG_RERANK_TOP_K = 8
RAG_PROMPT_DOCS_MAX = 4
RAG_KEYWORD_POOL_LIMIT = 150
RAG_THRESHOLD_RECALL = 0.34
RAG_THRESHOLD_DEFAULT = 0.44

# Prompt context
PROMPT_HISTORY_LIMIT = 12
PROMPT_SUMMARY_LIMIT = 2
PROMPT_MEMORY_LIMIT = 12
FETCH_HISTORY_MULTIPLIER = 3

# Memory trigger
MEMORY_TRIGGER_USER_TURNS = 5
MEMORY_CONTEXT_MESSAGES = 12

# Router
ROUTE_CHITCHAT = "CHITCHAT"
ROUTE_KNOWLEDGE = "KNOWLEDGE"
ROUTE_DANGER = "DANGER"
ENABLE_DANGER_ROUTING = False
ROUTER_MARGIN = 0.02

CHITCHAT_HINTS = ["안녕", "잘자", "고마워", "반가워", "심심", "기분", "배고파", "수다"]
KNOWLEDGE_HINTS = [
    "뭐였지", "기억", "아까", "전에", "말했", "추천", "요약", "정리해줘", "알려줘",
    "설명해줘", "설명", "정의", "원리", "차이", "비교", "장단점", "왜", "어떻게",
    "알고리즘", "복잡도", "빅오", "자료구조", "그래프", "경로", "최단경로",
    "다익스트라", "에이스타", "a스타", "플로이드", "워셜", "동적계획법",
    "코드", "파이썬", "자바", "씨플플", "c++", "디버그", "오류", "에러", "버그",
    "통신", "lin", "can", "전장", "네트워크", "프로토콜", "데이터베이스", "sql",
    "무슨 알고리즘", "무엇인가", "뭐야", "알아", "인가요", "인가", "맞아",
]
DANGER_PATTERNS = ["죽고 싶", "자해", "해치고 싶", "폭력", "칼로", "목숨", "없어지고 싶"]

MIN_STORE_LENGTH_FOR_CHITCHAT = 20
STORE_FACT_HINTS = [
    "나는", "내가", "요즘", "이번", "주말", "하려고", "계획", "관심", "좋아", "싫어", "선호",
]

# Input mode
INPUT_MODE_NORMAL = "NORMAL"
INPUT_MODE_OOC = "OOC"
INPUT_MODE_IC = "IC"
INPUT_MODE_MIXED = "MIXED"
OOC_IC_PATTERN_STR = r"^\s*\(?\s*(OOC|IC)\s*:\s*(.*?)\s*\)?\s*$"
OOC_IC_SEGMENT_PATTERN_STR = r"\(\s*(OOC|IC)\s*:\s*(.*?)\)"

DANGER_FALLBACK_RESPONSE = (
    "지금 많이 힘든 상태로 들려서 걱정돼. "
    "당장 위험하다고 느껴지면 가까운 사람이나 지역 응급번호에 바로 도움을 요청해줘. "
    "원하면 지금 네 상태를 차분히 같이 정리해보자."
)

# Memory extraction schema constraints
ALLOWED_MEMORY_TYPES = {"profile", "style", "preference", "taboo", "relationship", "fact"}
ALLOWED_MEMORY_KEYS = {
    "profile": {"user_name"},
    "style": {"preferred_speech_style", "preferred_response_length"},
    "preference": {"favorite_genre", "favorite_topic"},
    "taboo": {"disliked_topic", "avoid_behavior"},
    "relationship": {"attitude_to_character", "closeness_level"},
    "fact": {"current_activity", "current_goal"},
}
DEFAULT_IMPORTANCE_BY_TYPE = {
    "profile": 9,
    "style": 8,
    "preference": 6,
    "taboo": 9,
    "relationship": 7,
    "fact": 5,
}
MEMORY_NORMALIZATION_MAP = {
    "preferred_speech_style": {"존댓말": "polite", "반말": "casual"},
    "preferred_response_length": {"짧게": "short", "보통": "medium", "길게": "long"},
    "favorite_genre": {"판타지": "fantasy", "추리": "mystery", "스릴러": "mystery", "로맨스": "romance"},
    "disliked_topic": {"정치": "politics", "종교": "religion"},
}

# Retrieval/rerank constants
RECALL_HINT_PATTERNS = [
    "뭐였지", "뭐였더라", "기억", "기억나", "제목", "이름",
    "아까", "전에", "읽었", "읽었다", "말했", "추천",
]

KEYWORD_STOPWORDS = {
    "내가", "아까", "그", "그거", "이거", "저거", "뭐",
    "뭐지", "뭐였지", "좀", "그리고", "오늘", "우리", "너는", "나는",
}

KEYWORD_SYNONYMS = {
    "책": ["소설", "작품", "도서"],
    "소설": ["책", "작품"],
    "제목": ["이름", "타이틀"],
    "이름": ["제목"],
}

TECH_KEYWORDS = {
    "알고리즘", "최단경로", "경로", "복잡도", "빅오", "그래프", "노드",
    "다익스트라", "에이스타", "a스타", "플로이드", "워셜",
    "통신", "lin", "can", "프로토콜", "네트워크",
    "코드", "파이썬", "자바", "c", "c++", "sql", "디버그", "에러",
}
TECH_KEYWORD_BONUS = 0.12
TECH_QUERY_NOISE_PENALTY = 0.18
LIFESTYLE_KEYWORDS = {
    "마라탕", "꿔바로우", "주말", "점심", "저녁", "산책", "카페", "소설", "추리", "스릴러",
}

# Retrieval post-filter
POST_FILTER_MIN_KEYWORD_OVERLAP = 1
POST_FILTER_MIN_TEXT_LEN = 8
POST_FILTER_MAX_SUMMARY_RATIO = 0.5
POST_FILTER_RECALL_MIN_CHAT_DOCS = 2
POST_FILTER_ENABLE_DEDUP = True
POST_FILTER_MAX_PER_ORIGIN_GROUP = 1

ONE_CHAR_KEYWORDS = {"책"}
JOSA_SUFFIXES = [
    "으로", "에서", "에게", "한테", "까지", "부터", "처럼", "보다",
    "이", "가", "은", "는", "을", "를", "와", "과", "도", "로", "에",
]
JOSA_SUFFIXES_SORTED = sorted(JOSA_SUFFIXES, key=len, reverse=True)

NOISE_EXACT_PATTERNS = {
    "ㅋ", "ㅋㅋ", "ㅋㅋㅋ", "ㅎ", "ㅎㅎ", "ㅎㅎㅎ",
    "ㅇㅇ", "ㄴㄴ", "ㅜㅜ", "ㅠㅠ", "...", "..", ";;",
}
NOISE_PREFIX_PATTERNS = ("ㅋ", "ㅎ")
MEMORY_SIGNAL_PATTERNS = [
    "좋아", "싫어", "선호", "원해", "불편", "싫다",
    "이름", "불러", "존댓말", "반말", "말투", "취향",
    "장르", "정치", "종교", "하지 마", "하지말", "피해줘",
    "같아", "꺼려", "괜찮아", "편해", "자주", "주로", "보통",
    "스타일", "버릇", "습관", "원하지", "싫진", "힘들어",
    "싶지", "싶어", "생각", "라고", "잖아", "편이",
]
