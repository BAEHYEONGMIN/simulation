"""
[실습 7단계: Mini HA1 Graph (실제 라우팅 모듈 탑재)]
실습 목적: 앞선 모의(Mock) 실습을 넘어, 실제 HA1 백엔드 코드의 함수를 불러와 LangGraph 노드에서 실행합니다.
- services 하위의 실제 비즈니스 로직(extract_mode_and_question, classify_question_route) 임포트
- 파싱 -> 라우팅으로 이어지는 가장 뼈대가 되는 초기 워크플로우 구성
"""

import sys
import os
# 상위 디렉터리(ha1-project-python-rag)의 모듈을 임포트하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 실제 HA1 모듈 임포트
from services.question_service import extract_mode_and_question, classify_question_route

class AskState(TypedDict, total=False):
    raw_question: str
    question: str
    mode: str
    routing: dict

def parse_input(state: AskState):
    print("--- [Node] parse_input ---")
    raw = state.get("raw_question", "")
    
    # 실제 HA1의 모드 파서 적용
    mode, question = extract_mode_and_question(raw)
    print(f" => 파싱 완료: 모드='{mode}', 질문='{question}'")
    return {"mode": mode, "question": question}

def route_question(state: AskState):
    print("--- [Node] route_question ---")
    question = state.get("question", "")
    raw_question = state.get("raw_question", "")
    
    # 실제 HA1의 라우팅 및 의도 분석기 실행
    # (주의: 로컬 DB나 환경 변수가 세팅되어 있지 않으면 에러가 날 수 있어 try-except 처리)
    try:
        routing = classify_question_route(question, raw_question=raw_question)
        print(f" => 라우팅 완료: 정책={routing.get('routePolicy', 'N/A')}")
    except Exception as e:
        print(f" => [주의] DB 등 의존성 에러가 발생하여 모의 결과로 대체합니다: {e}")
        routing = {"routePolicy": "global_manual_search", "error": str(e)}
        
    return {"routing": routing}

# 그래프 구성
workflow = StateGraph(AskState)
workflow.add_node("parse_input", parse_input)
workflow.add_node("route_question", route_question)

workflow.add_edge(START, "parse_input")
workflow.add_edge("parse_input", "route_question")
workflow.add_edge("route_question", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 7단계: 실제 HA1 라우팅 모듈 탑재 검증 ===\n")
    # 사용자가 @strict 플래그와 함께 질문을 던진 상황
    initial_state = {"raw_question": "@STRICT: 아반떼 최고속도를 알려줘"}
    result = app.invoke(initial_state)
    
    print("\n최종 상태:")
    for k, v in result.items():
        print(f" - {k}: {v}")
