"""
[실습 6단계: Mock RAG 전체 파이프라인 검증]
실습 목적: 검색(Retrieve) -> 답변(Generate) -> 검증(Validate) 이라는 실제 RAG 흐름을 가상(Mock) 데이터로 연결합니다.
- LangGraph를 활용한 역할 정립(Node 분리) 및 후처리(Citation 유효성 체크)의 명시화
- 이 다음 단계부터는 실제 HA1 함수가 호출되므로 기존 로직 탑재 전의 최종 구조화 연습
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class RAGState(TypedDict, total=False):
    question: str
    refs: list[dict]
    answer: str
    citation_valid: bool

# 2. Node 정의
def retrieve_documents(state: RAGState):
    print("--- [Node] retrieve_documents 실행 (Mock) ---")
    # 2개의 가짜 참조 문서
    mock_docs = [
        {"id": 101, "text": "LangGraph는 그래프 기반 orchestration 라이브러리다."}, 
        {"id": 102, "text": "상태 제어가 용이하다."}
    ]
    return {"refs": mock_docs}

def generate_answer(state: RAGState):
    print("--- [Node] generate_answer 실행 (Mock) ---")
    # 답변 내에 Citation Number [1] 을 삽입했다고 모의함
    answer = "LangGraph는 상태 제어가 매우 용이한 그래프 기반 라이브러리입니다. [101] [102]"
    return {"answer": answer}

def validate_citations(state: RAGState):
    print("--- [Node] validate_citations 실행 ---")
    answer = state.get("answer", "")
    refs = state.get("refs", [])
    
    # 꼼꼼한 Post-processing 검사 로직이 들어가는 자리 (citation_filter.py 모의)
    # 여기서는 단순하게 본문에 해당 ref id가 박혀있는지만 확인
    is_valid = True
    for ref in refs:
        ref_tag = f"[{ref['id']}]"
        if ref_tag not in answer:
            is_valid = False
            break
            
    print(f" => Citation 검증 결과: {'정상(Pass)' if is_valid else '비정상(Fail)'}")
    return {"citation_valid": is_valid}

# 3. 그래프 구성
workflow = StateGraph(RAGState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("validate_citations", validate_citations)

# 선형 파이프라인
workflow.add_edge(START, "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", "validate_citations")
workflow.add_edge("validate_citations", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 6단계: Mock RAG 전체 파이프라인 그래프 검증 ===\n")
    result = app.invoke({"question": "LangGraph 장점이 뭐야?"})
    
    print("\n최종 상태 값:")
    for key, value in result.items():
        print(f" - {key}: {value}")
