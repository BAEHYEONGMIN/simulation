"""
[실습 9단계: Generation Graph (실제 답변 생성 모듈 탑재)]
실습 목적: 8단계까지 이어진 라우팅 -> 검색 결과(Chunks)를 바탕으로 실제 LLM을 호출해 답변을 생성합니다.
- 실제 HA1의 generate_answer 함수를 연결하여 프롬프팅 및 LLM 호출 처리
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# HA1 코어 모듈들
from services.question_service import extract_mode_and_question, classify_question_route
from services.bm25_service import search_documents_bm25
from services.answer_service import generate_answer

class AskState(TypedDict, total=False):
    raw_question: str
    question: str
    mode: str
    routing: dict
    documents: list
    answer: str

def parse_input(state: AskState):
    print("--- [Node] parse_input ---")
    mode, question = extract_mode_and_question(state.get("raw_question", ""))
    return {"mode": mode, "question": question}

def route_question(state: AskState):
    print("--- [Node] route_question ---")
    try:
        routing = classify_question_route(state["question"])
    except Exception:
        routing = {"routePolicy": "global_manual_search"}
    return {"routing": routing}

def retrieve_documents(state: AskState):
    print("--- [Node] retrieve_documents ---")
    try:
        # BM25 결과를 LLM 컨텍스트 주입 형태(doc_id)로 매핑해줍니다
        raw_docs = search_documents_bm25(state["question"], top_k=3, manual_ids=state.get("routing", {}).get("manualIds"))
        mapped_docs = [{"doc_id": d.get("id"), "content": d.get("content")} for d in raw_docs]
    except Exception:
        mapped_docs = []
    print(f" => 검색된 문서 수: {len(mapped_docs)}")
    return {"documents": mapped_docs}

def generate_answer_node(state: AskState):
    print("--- [Node] generate_answer_node ---")
    question = state.get("question", "")
    mode = state.get("mode", "strict")
    routing = state.get("routing", {})
    docs = state.get("documents", [])
    
    try:
        # 실제 HA1 프롬프트 & LLM 추론 로직(OpenAI 등) 트리거
        ans = generate_answer(question=question, chunks=docs, mode=mode, routing=routing)
        print(" => LLM 답변 생성 완료")
    except Exception as e:
        # OPENAI_API_KEY 등이 환경변수에 없을 경우 예외 발생
        ans = f"[LLM 에러] API 키나 통신 환경을 확인하세요: {e}"
        
    return {"answer": ans}

# 1. 그래프 구성
workflow = StateGraph(AskState)
workflow.add_node("parse_input", parse_input)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer_node", generate_answer_node)

# 2. 선형 파이프라인
workflow.add_edge(START, "parse_input")
workflow.add_edge("parse_input", "route_question")
workflow.add_edge("route_question", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_answer_node")
workflow.add_edge("generate_answer_node", END)

# 3. 컴파일
app = workflow.compile()

if __name__ == "__main__":
    print("=== 9단계: 실제 HA1 답변 생성 로직 연동 ===\n")
    # API 키 세팅이 되어있다면 실제로 OpenAI API를 호출하여 답변을 만들어 냅니다.
    result = app.invoke({"raw_question": "@strict: 아반떼 제원 알려줘"})
    
    print("\n[최종 LLM 응답]")
    print(result.get("answer"))
