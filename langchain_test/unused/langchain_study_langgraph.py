import os
import sys

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ==============================================================================
# [스터디 테마 13] LangGraph (에이전트 아키텍처의 끝판왕)
# 단방향 파이프라인의 한계를 넘어, 노드(Node)와 엣지(Edge)로 '반복 루프'를 만듭니다.
# AI가 스스로 생각하고(Thought) -> 도구를 쓰고(Action) -> 결과를 분석하여(Observation) -> 대답을 완성할 때까지 계속 빙글빙글 도는 'ReAct'의 심장 구조입니다.
# ==============================================================================

# LangGraph 필수 모듈
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# 1️⃣ [State 정의] : 챗봇의 "화이트보드"
# 모든 노드(함수)가 돌아가면서 이 화이트보드에 글(메시지)을 추가하거나 읽어갑니다.
class AgentState(TypedDict):
    # 'operator.add'의 의미: 덮어쓰지 말고, 리스트형태로 계속 꼬리물기(누적) 하라는 뜻
    messages: Annotated[list, operator.add]


def test_langgraph_basics():
    print("\n--- [테마 13] LangGraph: 끊임없이 순환하며 도구를 쓰는 에이전트 ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # ----------------------------------------------------
    # 사전 준비: LLM에게 줄 '가짜 날씨 도구' 만들기
    # ----------------------------------------------------
    from langchain_core.tools import tool
    
    @tool
    def weather_tool(location: str) -> str:
        """특정 지역의 날씨 정보를 반드시 검색해야 할 때 쓰는 도구입니다."""
        print(f"     (📞 외부 API 접속 중... '{location}' 날씨 검색)")
        return f"{location}의 오늘 날씨는 아주 맑고 최고 기온은 25도입니다."

    # LLM의 두뇌에 도구를 묶어줍니다 (바인딩)
    llm_with_tools = llm.bind_tools([weather_tool])


    # 2️⃣ [노드(Node) 정의] : 업무를 수행하는 "작업자"들
    
    # 작업자 1: LLM (생각 담당)
    def llm_node(state: AgentState):
        print("🧠 [대기 중] LLM이 지금까지의 화이트보드를 읽고 대답을 생성합니다...")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]} # 화이트보드에 새 생각을 추가!

    # 작업자 2: 도구 (행동 담당)
    def tool_node(state: AgentState):
        print("🛠️ [대기 중] LLM이 도구를 쓰라고 명령했습니다. 도구를 가동합니다!")
        last_message = state["messages"][-1]
        
        # LLM이 내린 도구 명령서(tool_calls)를 분석해 진짜 도구를 켭니다.
        if last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call["name"] == "weather_tool":
                result = weather_tool.invoke(tool_call["args"])
                # 도구가 찾아낸 정보(결괏값)를 화이트보드에 추가!
                return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}
                
        return {"messages": []}


    # 3️⃣ [라우터(Conditional Edge) 정의] : "방향 지시등"
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        # 방금 막 LLM이 화이트보드에 적은 게 "나 도구 쓸래!" 라는 요청이라면?
        if last_message.tool_calls:
            print("🚦 [라우터 판단] LLM이 데이터를 모른다고 하네요 -> 도구 노드로 보냅니다!")
            return "tools"
        # 더 쓸 도구가 없거나, 이미 답을 다 찾았다면?
        print("🚦 [라우터 판단] 완벽한 대답이 완성되었네요 -> 종료(END)로 보냅니다!")
        return END


    # 4️⃣ [그래프(Graph) 조립] : 점을 찍고 선을 긋습니다.
    workflow = StateGraph(AgentState)
    
    # 점(로직) 찍기
    workflow.add_node("agent", llm_node)
    workflow.add_node("tools", tool_node)
    
    # 선 긋기
    workflow.add_edge(START, "agent")  # 무조건 시작하면 agent 노드로 가라.
    
    # 조건부 선 긋기: agent가 일 끝나면 아까 만든 '라우터'한테 방향을 물어보고,
    # "tools"로 떨어지면 tools 노드로, END로 떨어지면 프로그램 종료해라.
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    
    # [하이라이트!] 도구 사용이 끝나면, 무조건 다시 agent(LLM) 쪽으로 돌아가서 결과를 보고해라 (순환 루프!)
    workflow.add_edge("tools", "agent")
    
    # 컴파일하면 하나의 거대한 스마트앱 완성!
    app = workflow.compile()
    
    
    # 5️⃣ [테스트 실행]
    print("\n🗨️ 사용자 메세지: '안녕! 혹시 오늘 서울 날씨 어때요?'")
    initial_state = {"messages": [HumanMessage(content="안녕! 혹시 오늘 서울 날씨 어때요?")]}
    
    print("\n================== LangGraph 흐름 추적 ==================")
    # stream을 돌리면 어떤 놈이 언제 일을 마쳤는지 실시간으로 모니터링 가능합니다.
    for output in app.stream(initial_state):
        for key, value in output.items():
             print(f"✅ 노드 작업 완료 보고: [{key}] 부서")
             print("-" * 55)
    
    print("\n================== 최종 대답(결과) ==================")
    final_messages = app.invoke(initial_state)["messages"]
    print(f"🤖 챗봇: {final_messages[-1].content}")
    print("\n💡 배민님! 이처럼 LangGraph는 끊임없이 사이클을 돌며 자기 스스로 답을 쟁취해냅니다!")


if __name__ == "__main__":
    test_langgraph_basics()
