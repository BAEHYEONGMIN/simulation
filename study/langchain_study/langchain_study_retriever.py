import os
import sys

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY, EMBEDDING_MODEL

# LangChain 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# ==============================================================================
# [스터디 테마 5] Parent Document Retriever (검색은 작게, 정보는 크게)
# 검색은 "조각(Child)" 단위로 정교하게 수행하지만,
# LLM에게는 그 조각이 포함된 "원본 전체(Parent)" 맥락을 가져오는 기술입니다.
# 챗봇의 대화 한 토막을 찾아서 앞뒤 맥락 전체를 가져올 때 1등인 기술입니다.
# ==============================================================================

def test_parent_document_retriever():
    print("\n--- [테마 5] Parent Document Retriever (문맥 확장형 검색) ---")
    
    # 1. 원본 문서 (Parent) - 아주 긴 대화 뭉치라고 가정
    full_docs = [
        Document(
            page_content="안녕! 나는 배민이야. 나는 판타지 소설을 정말 좋아해. 그중에서도 해리포터를 가장 좋아하지. 어제는 새벽 3시까지 정주행하느라 잠을 못 잤어.",
            metadata={"conf_uid": "sua_001"}
        ),
        Document(
            page_content="내 성격은 좀 까칠한 편이야. 하지만 친해지면 말을 아주 많이 하는 편이지. 나는 매운 음식을 못 먹어서 돈까스에 매운 소스 있으면 절대 안 먹어.",
            metadata={"conf_uid": "sua_001"}
        )
    ]

    # 2. 쪼개기 도구 세팅
    # 원본(Parent)은 크게 두고, 검색용(Child)은 아주 작게 쪼갭니다.
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
    
    # 3. 저장소 세팅
    vectorstore = InMemoryVectorStore(GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY))
    store = InMemoryStore() # Parent 원본 문서를 따로 보관할 저장소
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 4. 문서 추가 (내부적으로 조교들이 원본/조각을 나눠서 따로 저장함)
    retriever.add_documents(full_docs)

    # 5. 아주 짧은 단어로 검색 시도
    query = "돈까스 소스"
    print(f"🔍 쿼리: '{query}'")
    
    # [일반 검색이었다면]: "돈까스 소스 있으면" 이라는 단독 조각만 튀어나와서 맥락이 끊김
    # [Parent 검색]: 해당 조각이 포함된 '내 성격은 좀~' 전체 문단을 가져옴
    results = retriever.invoke(query)
    
    print(f"👉 검색된 문서 개수: {len(results)}")
    if results:
        print(f"👉 최종 반환된 전체 문맥:\n{results[0].page_content}")


# ==============================================================================
# [스터디 테마 6] Self-Querying Retriever (AI가 주는 DB 필터링 조건)
# 사용자의 질문 속에 숨겨진 "필터 조건(예: 어제 쓴 거, 배민이가 한 말)"을 
# LLM이 직접 분석해서 DB의 WHERE 절(Metadata Filter)로 변환해주는 기술입니다.
# ==============================================================================

def test_self_querying():
    print("\n--- [테마 6] Self-Querying Retriever (메타데이터 스스로 필터링) ---")
    
    # 1. 메타데이터가 포함된 샘플 데이터
    docs = [
        Document(page_content="해리포터 존잼", metadata={"type": "preference", "subject": "novel", "rating": 5}),
        Document(page_content="정치 이야기 금지", metadata={"type": "taboo", "subject": "politics", "rating": 10}),
        Document(page_content="공포 영화 극혐", metadata={"type": "preference", "subject": "movie", "rating": 2}),
    ]
    
    vectorstore = InMemoryVectorStore.from_documents(
        docs, 
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    )

    # 2. **핵심** : AI에게 우리 메타데이터 컬럼이 어떻게 생겼는지 설명서를 줍니다. (AttributeInfo)
    metadata_field_info = [
        AttributeInfo(name="type", description="정보의 성격 (preference, taboo 중 하나)", type="string"),
        AttributeInfo(name="subject", description="주제 카테고리 (novel, movie, politics 처럼 대상을 의미)", type="string"),
        AttributeInfo(name="rating", description="중요도 또는 선호도 (1~10 사이의 숫자)", type="integer"),
    ]
    
    document_content_description = "사용자의 취향과 기피 대상에 대한 메모"
    
    # 3. Self-Querying 검색기 생성
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    retriever = SelfQueryRetriever.from_llm(
        llm, 
        vectorstore, 
        document_content_description, 
        metadata_field_info, 
        verbose=True # AI가 쿼리를 어떻게 조작하는지 과정을 보여줌
    )

    # 4. 필터가 필요한 질문을 던져봅니다.
    query = "사용자가 싫어하는 것들(taboo) 중에서 주제(subject)가 정치인 거 찾아줘"
    print(f"🔍 쿼리: '{query}'")
    
    results = retriever.invoke(query)
    
    print(f"👉 결과 개수: {len(results)}")
    for doc in results:
        print(f"   [문서]: {doc.page_content} | [메타데이터]: {doc.metadata}")


if __name__ == "__main__":
    test_parent_document_retriever()
    # 주의: Self-Querying은 내부적으로 'lark' 라이브러리를 사용하므로, 
    # 설치가 안 되어 있다면 pip install lark 가 필요할 수 있습니다.
    test_self_querying()
