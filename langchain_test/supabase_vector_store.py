import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config import (
    SUPABASE_URL, SUPABASE_KEY,
    GEMINI_API_KEY_FREE,
    EMBEDDING_MODEL
)
from supabase import create_client, ClientOptions
# 1. Supabase 클라이언트 세팅
supabase = create_client(SUPABASE_URL, SUPABASE_KEY,options=ClientOptions(schema="chatbot"))

# 2. 랭체인 임베딩 모델 준비
# 주의: 이전에 우리가 쌩 코드로 돌렸던 것을 랭체인 전용 객체로 감싸는 겁니다.
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY_FREE,
    output_dimensionality=768,
    # 랭체인 내부에서 v1beta를 태우거나 구형 에러를 우회해야 할 수도 있습니다 (현재 테스트용)
)

# 3. 테스트용 가짜 문서 배열 (Document 객체 묶음)
# 회원님이 준비하실 것은 이 부분을 나중에 PDF 파일이나 Text 로더로 바꿔치기 하는 것뿐입니다!
docs = [
    Document(
        page_content="바나나는 노랗고 달콤한 과일이다. 원숭이가 좋아한다.",
        metadata={"category": "과일", "source": "fake_doc_1", "conf_uid": "test_folder"}
    ),
    Document(
        page_content="파이썬은 배우기 쉽고 간결한 프로그래밍 언어다. 데이터 분석에 많이 쓰인다.",
        metadata={"category": "IT", "source": "fake_doc_2", "conf_uid": "test_folder"}
    ),
    Document(
        page_content="배민은 매운 음식을 잘 먹지 못하며, SF 영화보다는 판타지 장르를 좋아한다.",
        metadata={"category": "인물", "source": "fake_doc_3", "conf_uid": "test_folder"}
    ),
]

def run_vector_store_demo():
    print("🚀 Supabase에 문서 3개를 왕창 밀어넣습니다...")
    # 4. VectorStore에 문서들 한 방에 쏟아붓기 (add_documents)
    ### 현재 이 부분에서 에러 발생. ids를 하드코딩 해놔도 내부적으로 UUID를 생성해서 id타입으로 저장하려다가 에러가 발생함.
    # vector_store = SupabaseVectorStore.from_documents(
    vector_store = SupabaseVectorStore(
        # docs,   
        embedding = embeddings,
        client=supabase,
        table_name="documents_gemini",
        query_name="match_documents_gemini",
        # ids=[1001, 1002, 1003],  
    )
    # print("✅ 데이터 인서트 완료!\n")
    
    # 5. 검색 로직 (as_retriever)
    query = "배민이 좋아하는 영화 장르가 뭐야?"
    print(f"🔍 질문: '{query}'")
    
    # similarity_search 로 아주 쉽게 DB를 긁어옵니다.
    matched_docs = vector_store.similarity_search(
        query,
        k=2,
        # conf_uid 필터링을 걸 수도 있습니다.
        filter={"conf_uid": "test_folder"}
    )
    
    print("\n[검색된 결과]")
    for i, d in enumerate(matched_docs):
        print(f"[{i+1}] {d.page_content} (메타데이터: {d.metadata})")

if __name__ == "__main__":
    run_vector_store_demo()
