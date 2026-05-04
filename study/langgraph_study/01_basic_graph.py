"""
[실습 1단계: 기본 그래프]
실습 목적: LangGraph의 가장 기초적인 동작 구조와 문법을 확인합니다.
- 단순한 상태(State) 객체 정의 및 노드(Node) 연결
- 시작(START)부터 끝(END)까지 선형적인 데이터 흐름(Edge) 구성
- compile()과 invoke()를 통한 그래프 실행 방법 학습
"""

from typing import TypedDict
import operator
from typing import Annotated

from langgraph.graph import StateGraph, START, END

# 1. State 정의: 가장 간단한 상태를 정의합니다.
# Annotated[list[str], operator.add] 는 새로운 메시지가 올 때마다 기존 리스트에 추가(add)하도록 합니다.
class BasicState(TypedDict):
    messages: Annotated[list[str], operator.add]

# 2. Node 함수 정의: 각 노드는 상태(state)를 받아 새로운 상태 업데이트 값을 반환합니다.
def node_a(state: BasicState):
    print("--- [Node A] 실행 ---")
    # 현재 상태 출력
    print(f"현재 상태: {state}")
    return {"messages": ["Node A를 거쳤습니다."]}

def node_b(state: BasicState):
    print("--- [Node B] 실행 ---")
    print(f"현재 상태: {state}")
    return {"messages": ["Node B를 거쳤습니다."]}

# 3. Graph 구성
workflow = StateGraph(BasicState)

# 노드 등록
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

# 순서(Edge) 연결: START -> node_a -> node_b -> END
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 4. 그래프 컴파일
app = workflow.compile()

if __name__ == "__main__":
    print("=== 1단계: LangGraph 기본 동작 테스트 시작 ===\n")
    
    # 초기 상태 정의
    initial_state = {"messages": ["시작점"]}
    
    # 그래프 실행 (invoke)
    result = app.invoke(initial_state)
    
    print("\n=== 최종 결과 ===")
    print(result)
