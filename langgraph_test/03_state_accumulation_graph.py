"""
[실습 3단계: 상태 누적(State Accumulation)]
실습 목적: 대화 기록처럼 데이터를 덮어쓰지 않고 기존 상태에 계속 '누적'시키는 방법을 배웁니다.
- Annotated 문법과 operator.add를 통한 리스트 합치기(Merge) 처리
- 파이프라인을 지날 때마다 history 배열에 값이 하나씩 쌓이는 과정 확인
"""

import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# 1. State 정의
# 기존 필드들은 덮어씌워지고, history 리스트에는 값이 '추가(operator.add)'되도록 Annotated 적용
class AskState(TypedDict):
    question: str
    route: str
    answer: str
    history: Annotated[list[dict], operator.add]

# 2. Node 정의
def parse_and_route(state: AskState):
    print("--- [Node 1] parse_and_route ---")
    question = state["question"]
    
    # 상태 누적 증명을 위해 히스토리에 현재 사용자 질문을 남김 (새로운 리스트를 반환하면 합쳐짐)
    new_history = [{"role": "user", "content": question}]
    
    return {"route": "doc_qa", "history": new_history}

def generate_answer(state: AskState):
    print("--- [Node 2] generate_answer ---")
    answer = f"'{state['question']}'에 대한 문서 기반 답변입니다."
    
    # 상태 누적 증명을 위해 시스템 응답을 히스토리에 추가
    new_history = [{"role": "assistant", "content": answer}]
    
    return {"answer": answer, "history": new_history}

# 3. 그래프 구성
workflow = StateGraph(AskState)
workflow.add_node("parse_and_route", parse_and_route)
workflow.add_node("generate_answer", generate_answer)

# 흐름: START -> parse_and_route -> generate_answer -> END
workflow.add_edge(START, "parse_and_route")
workflow.add_edge("parse_and_route", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 3단계: State 누적(Accumulation) 그래프 테스트 ===\n")
    
    # 초기 상태에 빈 배열로 시작
    initial_state = {"question": "LangGraph 상태 누적 방식이 뭔가요?", "history": []}
    result = app.invoke(initial_state)
    
    print("\n최종 상태(결과):")
    for key, value in result.items():
        print(f" - {key}: {value}")
