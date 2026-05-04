"""
[실습 4단계: Fallback 무결성 보장]
실습 목적: 검색이나 처리가 실패했을 경우 대안 경로(Fallback)로 진입하는 우회 처리 구조를 잡습니다.
- 1차 검색 실패 시 2차 검색(조건 완화 등)으로 자동 복구 시도
- HA1의 'Manual(정확) 검색 실패 -> Broad(범위/메타) 검색으로 우회' 기능 이식의 핵심 토대
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class AskState(TypedDict, total=False):
    question: str
    documents: list[str]
    answer: str

# 2. Node 정의
def retrieve_primary(state: AskState):
    print("--- [Node] 1차 검색 시도 (Manual / Exact) ---")
    question = state.get("question", "")
    
    # 실패 모의: 질문에 '실패'라는 단어가 있으면 문서를 0건 찾았다고 가정
    if "실패" in question:
        docs = []
    else:
        docs = ["메인 문서 1", "메인 문서 2"]
    
    print(f" => 검색된 문서 수: {len(docs)}")
    return {"documents": docs}

def retrieve_fallback(state: AskState):
    print("--- [Node] 2차 검색 시도 (Fallback / Broad) ---")
    # 조건 완화된 넓은 범위의 검색을 시도했다고 가정
    fallback_docs = ["Fallback 완화 검색 문서 A", "Fallback 완화 검색 문서 B"]
    print(f" => 검색된 문서 수: {len(fallback_docs)}")
    return {"documents": fallback_docs}

def generate_answer(state: AskState):
    print("--- [Node] 답변 생성 ---")
    docs = state.get("documents", [])
    
    if not docs:
        answer = "관련 문서를 전혀 찾지 못해 답변을 드릴 수 없습니다."
    else:
        answer = f"총 {len(docs)}건의 문서를 바탕으로 답변을 작성했습니다."
    
    return {"answer": answer}

# 3. 문서 유무를 확인하고 분기(Fallback)하는 판별 함수
def check_retrieval(state: AskState) -> Literal["generate_answer", "retrieve_fallback"]:
    docs = state.get("documents", [])
    if len(docs) > 0:
        # 1차 검색에 성공했으면 바로 생성
        return "generate_answer"
    else:
        # 실패했으면 Fallback 로직으로
        return "retrieve_fallback"

# 4. 그래프 구성
workflow = StateGraph(AskState)
workflow.add_node("retrieve_primary", retrieve_primary)
workflow.add_node("retrieve_fallback", retrieve_fallback)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "retrieve_primary")

# 조건부 엣지: 문서 유무에 따른 Fallback
workflow.add_conditional_edges(
    "retrieve_primary",
    check_retrieval,
    {
        "generate_answer": "generate_answer",
        "retrieve_fallback": "retrieve_fallback"
    }
)

# 2차 검색(Fallback)이 끝나면 무조건 생성 노드로 넘어감
workflow.add_edge("retrieve_fallback", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 4단계: 검색 실패 시 Fallback 그래프 테스트 ===")
    
    print("\n[Target 1: 성공적인 검색 케이스]")
    result_success = app.invoke({"question": "LangGraph 소개"})
    print(f"⭐ 최종 답변: {result_success.get('answer')}")

    print("\n[Target 2: 의도적인 실패 -> Fallback 작동 케이스]")
    result_fail = app.invoke({"question": "이건 검색 실패하는 질문이야"})
    print(f"⭐ 최종 답변: {result_fail.get('answer')}")
