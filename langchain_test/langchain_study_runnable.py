import os
import sys
import time

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# [히스토리] 관련 임포트
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ==============================================================================
# [스터디 테마 11] RunnableParallel (양손잡이 파이프라인)
# 여러 개의 독립적인 작업(예: 검색 + 이력조회)을 동시에 출발시키는 부품입니다.
# ==============================================================================

def test_runnable_parallel():
    print("\n--- [테마 11] RunnableParallel (속도 2배 뻥튀기) ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # 작업 1: 장점 찾기 엔진
    chain_pros = ChatPromptTemplate.from_template("{topic}의 장점 1가지만 짧게 말해.") | llm | StrOutputParser()
    
    # 작업 2: 단점 찾기 엔진
    chain_cons = ChatPromptTemplate.from_template("{topic}의 단점 1가지만 짧게 말해.") | llm | StrOutputParser()
    
    # 💡 [핵심] 병렬 엔진 조립! (이 안의 작업들은 서로 기다리지 않고 '동시에' 발사됩니다.)
    parallel_chain = RunnableParallel(
        pros=chain_pros,
        cons=chain_cons
    )
    
    # 최종 요약 체인: 앞선 두 개의 결과를 한꺼번에 받아서 요약합니다.
    final_prompt = ChatPromptTemplate.from_template(
        "주제: {topic}\n장점: {pros}\n단점: {cons}\n\n위 장단점을 보고 팩폭 한줄 평을 해줘."
    )
    
    total_chain = parallel_chain | final_prompt | llm | StrOutputParser()
    
    start = time.time()
    # "전기차"를 던지면 parallel_chain이 장점과 단점을 '동시에' 생각합니다.
    result = total_chain.invoke({"topic": "전기차"})
    
    print(f"👉 최종 결과:\n{result}")
    print(f"⏱️ 걸린 시간: {time.time() - start:.2f}초 (장점, 단점을 동시에 파싱하므로 체감 속도가 매우 빠릅니다!)")


# ==============================================================================
# [스터디 테마 12] RunnableWithMessageHistory (수동 이력관리 안녕👋)
# ==============================================================================

# DB 역할을 대신할 임시 램(RAM) 저장소 딕셔너리
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """랭체인이 세션 아이디를 던져주면, 그 방에 맞는 대화 기록 뭉치를 반환하는 함수"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def test_runnable_with_history():
    print("\n--- [테마 12] RunnableWithMessageHistory (자동 기억장치) ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # 1. 💡 [핵심] 프롬프트에 '과거 기억이 들어갈 빈 방'을 뚫어줍니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 다정한 친구야. 짧게 대답해."),
        MessagesPlaceholder(variable_name="chat_history"), # 마법의 공간! (어제 대화가 여기 통째로 꽂힘)
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # 2. 💡 [핵심] 자동 기억 관리자로 체인을 예쁘게 포장(Wrapping)합니다.
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        get_session_history, # 이력 조회용 함수 매핑 (실무에선 여기에 Supabase SELECT 코드가 들어감)
        input_messages_key="question", # 유저 입력 변수명 지정
        history_messages_key="chat_history",# 아까 뚫어둔 빈 방 이름 지정
    )
    
    # 실행할 때 '누구의 대화인지' 이름표만 주면 됩니다.
    config = {"configurable": {"session_id": "sua_user_001"}}
    
    print("🗨️ 1번째 턴 (저장 중...)")
    # 예전 수동 방식: 텍스트를 꺼내서 붙여야 했음. 지금은 알아서 함!
    res1 = wrapped_chain.invoke({"question": "안녕! 나는 배민이라고 해. 바나나를 제일 좋아해."}, config=config)
    print(f"🤖 AI: {res1}\n")
    
    print("🗨️ 2번째 턴 (과거 내용 알아서 들고 옴...)")
    res2 = wrapped_chain.invoke({"question": "내 이름이 뭐고, 뭘 좋아한다고 했지?"}, config=config)
    print(f"🤖 AI: {res2}")
    
    print("\n💡 실제 서비스(chat.py)에서는 `get_session_history` 함수 안에 Supabase 연동 코드만 깔끔하게 넣으면 코드가 1/3로 줄어듭니다!")


if __name__ == "__main__":
    test_runnable_parallel()
    test_runnable_with_history()
