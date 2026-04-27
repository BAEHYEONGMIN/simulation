from typing import TypedDict
from langgraph.graph import StateGraph, START, END
class WorkflowState(TypedDict):
    data: str
    steps_completed: list
    status: str

def step1(state: WorkflowState) -> dict:
    """첫 번째 처리 단계"""
    return {
        "data": state["data"] + " -> Step1",
        "steps_completed": state.get("steps_completed", []) + ["step1"],
        "status": "step1_complete"
    }

def step2(state: WorkflowState) -> dict:
    """두 번째 처리 단계"""
    return {
        "data": state["data"] + " -> Step2",
        "steps_completed": state["steps_completed"] + ["step2"],
        "status": "step2_complete"
    }

def step3(state: WorkflowState) -> dict:
    """세 번째 처리 단계"""
    return {
        "data": state["data"] + " -> Step3",
        "steps_completed": state["steps_completed"] + ["step3"],
        "status": "workflow_complete"
    }

# 순차적 실행 흐름 정의
workflow = StateGraph(WorkflowState)
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
workflow.add_node("step3", step3)

# 엣지로 실행 순서 정의
workflow.add_edge(START, "step1")    # 1. 시작 -> step1
workflow.add_edge("step1", "step2")  # 2. step1 -> step2
workflow.add_edge("step2", "step3")  # 3. step2 -> step3
workflow.add_edge("step3", END)      # 4. step3 -> 종료

compiled_workflow = workflow.compile()

# 실행 예시
result = compiled_workflow.invoke({
    "data": "Start",
    "steps_completed": [],
    "status": "initialized"
})

print(f"최종 데이터: {result['data']}")
print(f"완료된 단계: {result['steps_completed']}")
print(f"상태: {result['status']}")
