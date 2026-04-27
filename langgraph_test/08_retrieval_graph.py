"""
[실습 8단계: Retrieval Graph (검색 인프라 연동)]
실습 목적: 7단계의 라우팅 결과를 바탕으로, HA1 프로젝트의 실제 검색(BM25 등) 모듈을 트리거하여 문서를 가져옵니다.
- services/bm25_service.py 의 실제 검색 쿼리 함수를 노드 내부에 삽입
- 라우팅(routePolicy) 값에 따라 어떤 범위(글로벌 vs 좁은범위)의 문서를 뒤질지 결정하는 분기 역할 실습
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from services.question_service import classify_question_route
from services.bm25_service import search_documents_bm25

class AskState(TypedDict, total=False):
    question: str
    routing: dict
    documents: list

def route_question(state: AskState):
    print("--- [Node] route_question ---")
    question = state.get("question", "")
    try:
        routing = classify_question_route(question)
    except Exception:
        routing = {"routePolicy": "global_manual_search"}
    return {"routing": routing}

def retrieve_documents(state: AskState):
    print("--- [Node] retrieve_documents ---")
    question = state.get("question", "")
    routing = state.get("routing", {})
    route_policy = routing.get("routePolicy", "global_manual_search")
    
    docs = []
    print(f" => 검색 정책: {route_policy}")
    
    # 7단계에서 결정된 정책에 따라 실제 BM25 검색 함수 호출
    try:
        # DB 연결이 안되어 있으면 에러를 뱉습니다.
        # 실제 BM25 함수 시그니처에 맞게 파라미터를 입력합니다 (limit -> top_k)
        docs = search_documents_bm25(
            question, 
            top_k=3, 
            manual_ids=routing.get("manualIds")
        )
        print(f" => 검색된 문서 수: {len(docs)}")
        
        # 검색된 문서 제목(또는 ID) 살짝 출력
        for d in docs:
            print(f"    - [문서ID: {d.get('id')}] 점수: {d.get('bm25Score')}")
    except Exception as e:
        print(f" => [주의] DB 연결 실패 등의 이유로 검색을 진행할 수 없습니다: {e}")
        
    return {"documents": docs}

workflow = StateGraph(AskState)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve_documents", retrieve_documents)

workflow.add_edge(START, "route_question")
workflow.add_edge("route_question", "retrieve_documents")
workflow.add_edge("retrieve_documents", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 8단계: 실제 HA1 검색(Retrieve) 인프라 연동 ===\n")
    # HA1 DB 환경(Postgres 등)이 실행 중이 아니라면 에러가 뜰 수 있습니다.
    result = app.invoke({"question": "도로점용허가에 대해 알려줘"})
