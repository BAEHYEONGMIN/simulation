import os
import sys

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY, EMBEDDING_MODEL

# LangChain 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# [스터디 테마 7] Multi-Query Retriever (검색 명중률 극대화)
# 유저가 짧고 애매하게 질문해도, AI가 스스로 질문을 3개로 다시 써서(Rewrite)
# 훨씬 넓은 범위에서 정답을 찾아내는 기술입니다.
# ==============================================================================

def test_multi_query():
    print("\n--- [테마 7] Multi-Query (질문 재작성형 검색) ---")
    
    # 1. 샘플 데이터
    docs = [
        Document(page_content="어제 배민이는 헬스장에서 등 운동을 열심히 했습니다."),
        Document(page_content="배민이는 최근 보디빌딩 대회 준비에 매진하고 있습니다."),
        Document(page_content="배민이는 평소 건강 관리를 위해 식단을 철저히 지킵니다.")
    ]
    vectorstore = InMemoryVectorStore.from_documents(
        docs, 
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    )

    # 2. 멀티 쿼리 리트리버 생성
    # 내부적으로 유저의 질문 하나를 가지고 "3개 정도의 변형된 질문"을 생성한 뒤 각각 검색합니다.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), 
        llm=llm
    )

    # 3. 애매한 질문 던지기
    query = "배민이의 최근 활동은?" # "활동"이라는 단어는 본문에 없지만, AI가 재작성해서 찾아냅니다.
    print(f"🔍 원본 쿼리: '{query}'")
    
    unique_docs = retriever_from_llm.invoke(query)
    
    print(f"👉 검색된 결과 개수: {len(unique_docs)}")
    for i, doc in enumerate(unique_docs):
        print(f"   [{i+1}] {doc.page_content}")


# ==============================================================================
# [스터디 테마 8] Routing (지능형 길 찾기)
# 사용자의 의도를 파악하여 'RAG를 쓸 것인지', '일상 대화를 할 것인지'
# 또는 '특정 도구를 쓸 것인지' 길을 갈라주는 기술입니다.
# 여기서는 LLM 호출 없이 "임베딩 거리(Similarity)"로 판단하는 초고속 방식을 소개합니다.
# ==============================================================================

def test_semantic_routing():
    print("\n--- [테마 8] Semantic Routing (초고속 의도 분석) ---")
    
    # 1. 카테고리별 대표 질문들을 벡터로 변환해둡니다. (이게 길잡이 이정표가 됩니다.)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    
    # [인사/안부] 벡터 창고
    chitchat_samples = ["안녕", "오늘 기분 어때?", "배고파", "심심해"]
    chitchat_store = InMemoryVectorStore.from_texts(chitchat_samples, embeddings)
    
    # [기억/정보] 벡터 창고
    knowledge_samples = ["내 이름이 뭐야?", "내가 전에 말한 거 기억나?", "내 취미가 뭐였지?", "기억 알려줘"]
    knowledge_store = InMemoryVectorStore.from_texts(knowledge_samples, embeddings)

    # 2. 라우터 함수 정의 (질문이 들어오면 어디 창고랑 더 가까운지 계산)
    def route_query(query):
        # 각 창고에서 가장 비슷한 놈 하나씩을 찾아 '거리(Score)'를 봅니다.
        chitchat_score = chitchat_store.similarity_search_with_score(query, k=1)[0][1]
        knowledge_score = knowledge_store.similarity_search_with_score(query, k=1)[0][1]
        
        print(f"📊 [판단] 일상점수: {chitchat_score:.4f} / 기억점수: {knowledge_score:.4f}")
        
        # 점수가 더 높은(거리가 더 짧은) 쪽으로 보냅니다.
        if chitchat_score > knowledge_score: # (코사인 유사도는 점수가 높을수록 가깝습니다)
            return "CHITCHAT"
        else:
            return "KNOWLEDGE"

    # 3. 이제 두 갈래 길의 로직(Chain)을 만듭니다.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # 경로 1: 그냥 수다 떨기
    chitchat_chain = ChatPromptTemplate.from_template("너는 다정한 친구다. 다음 인사에 짧게 대답해: {input}") | llm | StrOutputParser()
    
    # 경로 2: RAG(기록 찾기) 로직 시뮬레이션
    knowledge_chain = ChatPromptTemplate.from_template("너는 기억 비서다. 과거 기록을 뒤져서 답해주겠다: {input}") | llm | StrOutputParser()

    # 4. 최종 라우팅 체인 조립
    total_chain = RunnableBranch(
        (lambda x: route_query(x["input"]) == "CHITCHAT", chitchat_chain),
        knowledge_chain # 위 조건에 안 맞으면 무조건 지식 체인으로
    )

    # 5. 테스트
    q1 = "안녕! 오늘 날씨 참 좋다."
    print(f"\n💬 질문1: '{q1}'")
    print(f"👉 답변: {total_chain.invoke({'input': q1})}")

    q2 = "내가 전에 말했던 거 기억나?"
    print(f"\n💬 질문2: '{q2}'")
    print(f"👉 답변: {total_chain.invoke({'input': q2})}")


if __name__ == "__main__":
    # 테마 7: 멀티 쿼리 (검색 범위 확장)
    test_multi_query()
    
    # 테마 8: 라우팅 (비용 효율적인 길 찾기)
    test_semantic_routing()
