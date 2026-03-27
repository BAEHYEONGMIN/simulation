import os
import sys
import time
import asyncio

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# [스터디 테마 9] Asynchronous (비동기 처리 기초)
# 챗봇 서버에 10명 이상의 유저가 접속했을 때, 한 유저의 답변을 기다리느라
# 전체 서버가 멈추거나 뻗지 않도록(Non-blocking) 해주는 '실시간 병렬 처리'입니다.
# 랭체인의 3대 비동기 무기: ainvoke, astream, abatch
# ==============================================================================

async def test_async_ainvoke_astream():
    print("\n--- [비동기 1] ainvoke와 astream (응답 대기시간 훔치기) ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    chain = ChatPromptTemplate.from_template("{topic}에 대해 짧은 시 한 편 지어줘.") | llm | StrOutputParser()
    
    print("⏳ [동기 방식] invoke: AI가 대답을 쓸 때까지 프로그램이 아예 멈춥니다.")
    start_time = time.time()
    sync_result = chain.invoke({"topic": "커피"})
    print(f"✅ 완료 ({time.time() - start_time:.2f}초 소요)\n")

    print("🚀 [비동기 방식] ainvoke: 'await'를 붙여놓으면 질문을 던지고, 대답이 오는 동안 파이썬은 다른 일을 할 수 있습니다!")
    start_time = time.time()
    async_result = await chain.ainvoke({"topic": "커피"})
    print(f"✅ 완료 ({time.time() - start_time:.2f}초 소요) -> (단일 속도는 비슷하지만, '안 막힌다'는 게 핵심)\n")
    
    print("🌊 [비동기 스트리밍] astream: 글자를 실시간으로 뿌리면서 동시에 비동기를 유지합니다.")
    print("💡 실제 FastAPI나 웹소켓 환경에서 가장 사랑받는 핵심 함수입니다.")
    # for 앞에 async를 붙이는 것이 포인트!
    async for chunk in chain.astream({"topic": "바람"}):
        print(chunk, end="", flush=True)
    print("\n✅ 스트리밍 종료")


# ==============================================================================
# [스터디 테마 10] abatch (병렬 폭격)
# 여러 개의 질문이나 여러 명의 사용자가 동시에 질문했을 때, 
# 차례대로 기다리지 않고 한 방에 쏴서 시간을 확 줄여버리는 마법 같은 함수입니다.
# ==============================================================================

async def test_async_abatch():
    print("\n--- [비동기 2] abatch (다중 요청 동시 처리) ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    chain = ChatPromptTemplate.from_template("{name}의 직업은 뭐야? 두 세단어로만 말해.") | llm | StrOutputParser()
    
    # 3명의 요청이 서버에 쌓였다고 가정
    requests = [{"name": "스티브 잡스"}, {"name": "세종대왕"}, {"name": "마이클 조던"}]
    
    print("⏳ [동기 순차 처리] 한 명씩 기다리면서 발송 (for 루프)")
    start_time = time.time()
    for req in requests:
        res = chain.invoke(req)
        print(f"  - {req['name']} 질문 완료")
    print(f"⏱️ 걸린 시간: {time.time() - start_time:.2f}초\n")
    
    print("🚀 [비동기 병렬 처리] abatch: 구글 서버로 질문 3개를 동시에 발사폭격!")
    start_time = time.time()
    # 여러 개의 딕셔너리를 리스트로 던지면, 랭체인이 알아서 "동시에" 처리해버립니다.
    results = await chain.abatch(requests)
    for req, res in zip(requests, results):
       print(f"  - {req['name']}: {res.strip()}")
    print(f"⏱️ 걸린 시간: {time.time() - start_time:.2f}초")
    print("💡 깨달음: 서버에 부하가 걸리거나 여러 데이터를 한 번에 요약할 때는 `abatch`가 압도적으로 빠르다!")


if __name__ == "__main__":
    # 주의: 비동기(async) 함수는 기존 파이썬처럼 냅다 부르면 에러가 납니다.
    # 반드시 asyncio.run() 이라는 엔진에 태워서 실행시켜야 합니다.
    async def main():
        await test_async_ainvoke_astream()
        await test_async_abatch()
        
    asyncio.run(main())
