"""
[실습 5단계: Tool(도구) 연동]
실습 목적: 시간 조회, 외부 통신 같은 외부 도구를 필요로 하는 질문을 별도로 우회하여 처리합니다.
- '현재 시간', '오늘 날짜' 등 외부 의존적인 문제를 처리하는 실시간(Realtime) 노드 생성
- HA1의 'realtime' 라우팅 모드를 LangGraph 내부에 이식하기 위한 학습
"""

from typing import TypedDict, Literal
import datetime
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class AskState(TypedDict, total=False):
    question: str
    route: str
    tool_result: str
    answer: str

# 2. Node 정의
def parse_and_route(state: AskState):
    print("--- [Node] 질문 라우팅 ---")
    question = state.get("question", "")
    
    # 시간이나 오늘을 물어보면 날짜조회(Tool) 모드로 빠짐
    if "시간" in question or "오늘" in question or "날짜" in question:
        return {"route": "realtime_tool"}
    return {"route": "general_qa"}

def execute_realtime_tools(state: AskState):
    print("--- [Node] execute_realtime_tools 실행 (Datetime Tool) ---")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"tool_result": f"현재 시스템 기준: {now}"}

def generate_general_answer(state: AskState):
    print("--- [Node] generate_general_answer 실행 ---")
    # 도구가 실행된 이력이 있으면 그걸 우선 사용
    if "tool_result" in state:
        answer = f"시스템 도구를 참고하여 답변합니다. {state['tool_result']}"
    else:
        answer = "저장된 데이터를 바탕으로 답변하는 일반 QA입니다."
    return {"answer": answer}

# 3. 분기 판단 함수
def route_condition(state: AskState) -> Literal["realtime_tool", "general_qa"]:
    if state.get("route") == "realtime_tool":
        return "realtime_tool"
    return "general_qa"

# 4. 그래프 구성
workflow = StateGraph(AskState)
workflow.add_node("parse_and_route", parse_and_route)
workflow.add_node("execute_realtime_tools", execute_realtime_tools)
workflow.add_node("generate_general_answer", generate_general_answer)

workflow.add_edge(START, "parse_and_route")
workflow.add_conditional_edges("parse_and_route", route_condition, {
    "realtime_tool": "execute_realtime_tools",
    "general_qa": "generate_general_answer"
})

workflow.add_edge("execute_realtime_tools", "generate_general_answer")
workflow.add_edge("generate_general_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 5단계: Tool(실시간 도구) 연동 테스트 ===\n")
    
    print("[Target 1: 실시간 데이터 요구]")
    r1 = app.invoke({"question": "오늘 날짜랑 시간을 알려줄래?"})
    print(f"=> {r1.get('answer')}\n")
    
    print("[Target 2: 일반 지식 질문]")
    r2 = app.invoke({"question": "파이썬이 뭐야?"})
    print(f"=> {r2.get('answer')}")
