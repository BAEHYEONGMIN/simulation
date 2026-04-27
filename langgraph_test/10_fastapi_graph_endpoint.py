"""
[실습 10단계: FastAPI Graph 연동 엔드포인트]
실습 목적: 작성된 Graph 엔진을 래핑하여 실제 FastAPI의 엔드포인트로 노출하는 구조를 알아봅니다.
- 기존의 /api/ask 경로 대신 /api/ask-graph 로 이식하는 서버 백엔드 계층 껍데기
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
# 실제 구현 시 9단계에서 컴파일한 app(그래프 객체)를 가져옵니다.
# from graphs.ask_graph import app as graph_app

app = FastAPI()

class AskGraphRequest(BaseModel):
    query: str

@app.post("/api/ask-graph")
async def ask_graph_endpoint(req: AskGraphRequest):
    print(f"--- [API] /api/ask-graph 수신: {req.query} ---")
    
    # 1. 초기 State 구성
    initial_state = {"raw_question": req.query}
    
    # 2. 그래프 비동기 실행 (실제: await graph_app.ainvoke(initial_state))
    # 여기서는 결과가 나왔다고 가정(Mock)
    result = {
        "question": req.query,
        "answer": "LangGraph를 거쳐 생성된 답변입니다.",
        "documents": [],
        "routing": {"gateRoute": "doc_qa"},
        "mode": "strict"
    }
    
    # 3. 프론트엔드 대시보드가 원래 받던 JSON 규격(Contract)에 맞춰 포장해서 반환
    # (이를 통해 프론트엔드는 백엔드가 LangGraph로 바뀐 사실을 모르고 똑같이 동작합니다)
    return {
        "answer": result.get("answer"),
        "chunks": result.get("documents", []),
        "routing": result.get("routing"),
        "modeUsed": result.get("mode", "strict")
    }

if __name__ == "__main__":
    print("=== 10단계: FastAPI 서버 연동 테스트 ===")
    print("터미널 명령어: uvicorn 10_fastapi_graph_endpoint:app --reload")
