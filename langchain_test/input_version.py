import os
import sys
import argparse
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from google import genai as google_genai
from config import SUPABASE_KEY, SUPABASE_URL, GEMINI_API_KEY,GEMINI_API_KEY_FREE, CHAT_MODEL, SUMMARY_MODEL, MEMORY_MODEL, EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL

# ─── 실행 옵션 파싱 ───────────────────────────────────────────
# 사용법:
#   python input_version.py           → 제미나이 임베딩 모드 (기본)
#   python input_version.py --local   → 로컬 SentenceTransformer 모드
parser = argparse.ArgumentParser(description="챗봇 입력 테스트")
parser.add_argument(
    "--local",
    action="store_true",
    help="로컬 임베딩 모델(SentenceTransformer)을 사용합니다."
)
args = parser.parse_args()
is_local = args.local

print(f"[모드] {'로컬(SentenceTransformer)' if is_local else f'제미나이 API (v1 / {EMBEDDING_MODEL})'}")

# ─── DB 상수 (모드에 따라 한 곳에서 결정) ────────────────────
#   이 블록만 수정하면 전체 코드에 반영됩니다.
SCHEMA             = "chatbot"
TABLE_CHAT         = "chat_messages"
if is_local:
    TABLE_DOCUMENTS = "documents"           # 384차원 (로컬 모델용)
    RPC_MATCH       = "match_documents"
else:
    TABLE_DOCUMENTS = "documents_gemini"    # 768차원 (제미나이용)
    RPC_MATCH       = "match_documents_gemini"

# ─── Supabase 클라이언트 ──────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── 임베딩 모델 초기화 (모드에 따라 분기) ────────────────────
if is_local:
    # 로컬 모델 (384차원)
    embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    genai_client = None
else:
    # 제미나이 SDK 직접 호출 (v1beta + gemini-embedding-001)
    embedding_model = None
    genai_client = google_genai.Client(
        api_key=GEMINI_API_KEY_FREE,
        http_options={"api_version": "v1beta"}   # gemini-embedding-001은 v1beta에 있음
    )



def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_embedding_local(text: str) -> list[float]:
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return vector.tolist()

def generate_embedding_gemini(text: str) -> list[float]:
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,           # models/gemini-embedding-001
        contents=text,
        config={"output_dimensionality": 768}  # 768차원으로 출력 (documents_gemini 테이블 호환)
    )
    return list(result.embeddings[0].values)

def generate_embedding(text: str) -> list[float]:
    """is_local 모드에 따라 자동으로 올바른 임베딩 함수를 호출합니다."""
    if is_local:
        return generate_embedding_local(text)
    return generate_embedding_gemini(text)



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
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
        .insert(payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(f"{TABLE_CHAT} 저장 실패")

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
        supabase.schema(SCHEMA)
        .table(TABLE_DOCUMENTS)
        .insert(payload)
        .execute()
    )

    if not result.data:
        raise RuntimeError(f"{TABLE_DOCUMENTS} 저장 실패")

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
        supabase.schema(SCHEMA)
        .rpc(
            RPC_MATCH,
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
        supabase.schema(SCHEMA)
        .table(TABLE_CHAT)
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

    # ─── sample_inputs.txt에서 문장 불러오기 ────────────────────
    # 사용법:
    #   --section sample_inputs  → [SECTION: sample_inputs] 구간 사용 (기본)
    #   --section test_queries   → [SECTION: test_queries] 구간 사용
    import argparse as _ap
    _section_parser = _ap.ArgumentParser(add_help=False)
    _section_parser.add_argument("--section", default="sample_inputs")
    _section_args, _ = _section_parser.parse_known_args()
    target_section = _section_args.section

    def load_inputs_from_file(path: str, section: str) -> list[str]:
        lines = []
        in_section = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"[SECTION: {section}]"):
                    in_section = True
                    continue
                if line.startswith("[SECTION:") and in_section:
                    break  # 다음 섹션 시작 → 종료
                if not in_section or not line or line.startswith("#"):
                    continue
                lines.append(line)
        return lines

    _inputs_file = os.path.join(os.path.dirname(__file__), "sample_inputs.txt")
    sample_inputs = load_inputs_from_file(_inputs_file, target_section)
    print(f"[섹션: {target_section}] {len(sample_inputs)}개 문장 로드됨")
    print("-" * 50)


    # while(sample_inputs):
    while True:
        user_input = input("사용자 입력: ").strip()
        # user_input = sample_inputs.pop()
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

            # 2) 임베딩 생성 (모드에 따라 자동 분기)
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