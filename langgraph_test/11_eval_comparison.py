"""
[실습 11단계: 결과 검증 및 평가(Evaluation)]
실습 목적: 마이그레이션 작업이 완료된 후, 기존 함수 방식의 파이프라인과 신규 Graph 방식의 파이프라인이 
완전히 동일하게 동작하는지 비교 테스트(Regression Test) 하는 로직입니다.
"""

def mock_legacy_pipeline(question: str):
    # 기존 main.py -> _process_ask() 로직이라고 가정
    return {"answer": "제원은 다음과 같습니다...", "route": "doc_qa"}

def mock_graph_pipeline(question: str):
    # 신규 09_generation_graph.py 로직이라고 가정
    return {"answer": "제원은 다음과 같습니다...", "route": "doc_qa"}

def evaluate_migration():
    test_cases = [
        "투싼 제원 알려줘",       # RAG 동작해야 함
        "오늘 날씨 어때",         # Realtime 동작해야 함
        "안녕 반가워"             # Chitchat 동작해야 함
    ]
    
    print("=== 11단계: 레거시 vs LangGraph 마이그레이션 평가 결과 대조 ===\n")
    for q in test_cases:
        print(f"[질문] {q}")
        
        legacy_res = mock_legacy_pipeline(q)
        graph_res = mock_graph_pipeline(q)
        
        # 1. 라우팅 일치 여부 (필수 일치 항목)
        is_route_matched = legacy_res["route"] == graph_res["route"]
        print(f"  - 라우팅(엔진 판단) 통과 여부: {'✅ Pass' if is_route_matched else '❌ Fail'}")
        
        # 2. 답변 생성 결과 (LLM 특성상 100% 토큰 일치는 안 될 수 있으므로 의미 유사도나 육안 검사 진행)
        # RAGAs, LangSmith 등의 Eval 도구를 붙이는 자리입니다.
        print(f"  - (레거시) {legacy_res['answer']}")
        print(f"  - (그래프) {graph_res['answer']}")
        print("-" * 40)
        
if __name__ == "__main__":
    evaluate_migration()
