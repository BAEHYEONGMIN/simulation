import os
import sys

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY_FREE, GEMINI_API_KEY

# LangChain 모듈
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==============================================================================
# [스터디 테마 3] LCEL (LangChain Expression Language) 체인 조립
# 파이썬 코드를 리눅스 명령어처럼 파이프(|)로 길게 연결해서 데이터가 폭포수처럼
# 흐르게 만드는 랭체인 최고의 핵심 마법(Syntax)입니다.
# ==============================================================================

def test_lcel_chain():
    print("\n--- [테마 3] LCEL (파이프 체인) 기술 ---")
    
    # [1] 부품들을 준비합니다.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # LLM이 뱉는 다양한 메타데이터 객체들(AIMessage 등)에서 순수하게 문자열(String)만 쏙 뽑아주는 파서
    str_parser = StrOutputParser()
    
    # 프롬프트 템플릿 (구멍이 2개(topic, language) 뚫려있습니다)
    prompt = ChatPromptTemplate.from_template("'{topic}' 에 대해 1줄로 짧게 설명해줘. 단, 답변은 {language} 언어로 해.")

    # [2] 부품들을 파이프(|) 기호로 엮어서 하나의 완성된 조립 라인(Chain)을 만듭니다!!!
    # 데이터 흐름: 입력(Dictionary) -> 프롬프트 구멍 뚫기 -> LLM 대답 생성 -> 파서로 글자만 뽑기
    chain = prompt | llm | str_parser

    print("조립 라인(Chain) 가동 중...")
    
    # [3] 한 번에 실행 (투입구에 변수 딕셔너리만 던지면 알아서 끝까지 흘러갑니다)
    result = chain.invoke({"topic": "블랙홀", "language": "한국어"})
    
    print(f"👉 결과(한국어): {result}")

    result_eng = chain.invoke({"topic": "블랙홀", "language": "영어"})
    print(f"👉 결과(영  어): {result_eng}")


# ==============================================================================
# [스터디 테마 4] 실시간 출력 (Streaming)
# 대기 시간(Latency) 이슈를 가려주는 챗봇 앱의 필수 기술!
# ==============================================================================

def test_streaming():
    print("\n--- [테마 4] 스트리밍(Streaming) 실시간 타이핑 효과 ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY_FREE)
    prompt = ChatPromptTemplate.from_template("'{topic}' 에 대해 3줄짜리 시를 지어줘.")
    
    # 체인 조립
    chain = prompt | llm | StrOutputParser()

    print("👉 답변 작성 시작(스트리밍): ", end="")
    
    # .invoke() 대신 .stream()을 사용하면, LLM이 단어 하나를 뱉을 때마다 for문을 돕니다!
    for chunk in chain.stream({"topic": "파이썬 코딩의 고통"}):
        print(chunk, end="", flush=True)  # 끊기지 않고 한 글자씩 실시간으로 터미널에 타닥타닥 찍힙니다.
        
    print("\n(완료)")


if __name__ == "__main__":
    test_lcel_chain()
    test_streaming()
