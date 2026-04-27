"""
[실습 2단계: 조건 분기(Branching)]
실습 목적: 입력에 따라 실행 경로를 나누는 '조건부 엣지(Conditional Edge)'를 다룹니다.
- 이전 노드의 실행 결과(route 등)에 따라 호출될 다음 노드를 동적으로 결정
- HA1 프로젝트의 'chitchat' vs 'retrieve' (답변 모드) 분기 로직 이식을 위한 사전 학습
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. State 정의: 질문, 판단된 라우팅 결과, 그리고 최종 답변을 담습니다.
class AskState(TypedDict, total=False):
    question: str       # 사용자 입력 질문
    route: str          # 라우팅 결과 (예: chitchat, doc_qa)
    answer: str         # 생성된 최종 답변

# 2. Node 함수 정의
def parse_and_route(state: AskState):
    print("--- [Node] parse_and_route 실행 ---")
    question = state.get("question", "")
    
    # 간단한 라우팅 로직 모의 (실제로는 extract_mode_and_question, route_question 활용)
    if "안녕" in question or "반가워" in question:
        route_decision = "chitchat"
    else:
        route_decision = "doc_qa"
        
    print(f" => 판단 결과: {route_decision}")
    return {"route": route_decision}

def generate_chitchat(state: AskState):
    print("--- [Node] generate_chitchat 실행 ---")
    return {"answer": "안녕하세요! 저는 HA1 AI 어시스턴트입니다. 무엇을 도와드릴까요?"}

def generate_doc_qa(state: AskState):
    print("--- [Node] generate_doc_qa 실행 ---")
    question = state.get("question", "")
    # 실제로는 여기서 Retrieval -> Generation이 일어납니다.
    return {"answer": f"문서 검색 결과, '{question}'에 대한 답변입니다."}

# 3. 분기를 결정하는 판단 함수
# - 이전 노드에서 반환한/업데이트된 상태 값을 기반으로 어떤 경로로 갈지 결정
def route_condition(state: AskState) -> Literal["chitchat", "doc_qa"]:
    # state 안에 저장된 'route' 값을 그대로 반환
    return state.get("route", "doc_qa")

# 4. Graph 구성
workflow = StateGraph(AskState)

# 노드 추가
workflow.add_node("parse_and_route", parse_and_route)
workflow.add_node("generate_chitchat", generate_chitchat)
workflow.add_node("generate_doc_qa", generate_doc_qa)

# 흐름(엣지) 연결
workflow.add_edge(START, "parse_and_route")

# ★ 조건부 엣지(Conditional Edge) 연결 부분 ★
# parse_and_route 이후에 route_condition 함수를 실행하여,
# 반환된 값("chitchat" 이나 "doc_qa")에 매칭되는 노드로 이동시킵니다.
workflow.add_conditional_edges(
    "parse_and_route",
    route_condition,
    {
        "chitchat": "generate_chitchat",
        "doc_qa": "generate_doc_qa"
    }
)

# 최종 노드들은 다시 END로 향하도록 연결
workflow.add_edge("generate_chitchat", END)
workflow.add_edge("generate_doc_qa", END)

# 5. 컴파일
app = workflow.compile()

if __name__ == "__main__":
    print("=== 2단계: 조건 분기(Branching) 그래프 테스트 ===")
    
    # [테스트 케이스 1] 인삿말을 건넸을 때
    print("\n[Target 1: 일상 대화]")
    state_1 = {"question": "안녕! 만나서 반가워."}
    result_1 = app.invoke(state_1)
    print(f"\n최종 결과: \n{result_1}")
    print(f"⭐️ 최종 답변: {result_1.get('answer')}")
    
    # [테스트 케이스 2] 문서 질문을 건넸을 때
    print("\n-------------------------------------------")
    print("\n[Target 2: RAG 문서 검색]")
    state_2 = {"question": "차체 프레임 구조가 궁금해"}
    result_2 = app.invoke(state_2)
    print(f"\n최종 결과: \n{result_2}")
    print(f"⭐️ 최종 답변: {result_2.get('answer')}")
