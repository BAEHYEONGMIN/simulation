import os
import sys
from pydantic import BaseModel, Field
from typing import List, Literal

# 상위 폴더의 config 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MEMORY_MODEL,CHAT_MODEL, SUMMARY_MODEL, GEMINI_API_KEY_FREE

# LangChain 모듈
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ==============================================================================
# [스터디 테마 1] Pydantic & 구조화된 출력 (Structured Output)
# memory_schema.txt 설계도를 완벽하게 준수하여 장기 기억 데이터를 추출합니다.
# ==============================================================================

# 1. memory_schema.txt에 정의된 허용된 값(Category)을 Pydantic으로 명확하게 고정합니다.
# Literal을 쓰면 LLM이 이 목록 외의 허튼소리(예: 'hobby' 같은 미승인 단어)를 뱉는 걸 차단합니다.
MemoryType = Literal["profile", "style", "preference", "taboo", "relationship", "fact"]

class MemoryCandidate(BaseModel):
    memory_type: MemoryType = Field(description="메모리의 카테고리 (허용된 6개 중 하나)")
    memory_key: str = Field(description="메모리의 키 (예: user_name, preferred_speech_style, favorite_genre, disliked_topic 등)")
    memory_value: str = Field(description="메모리의 정규화된 값 (예: 배민, polite, fantasy, politics 등)")
    confidence: float = Field(description="이 기억이 확실한지 평가한 퍼센티지 (0.0 ~ 1.0 사이)", ge=0.0, le=1.0)

# 2. 하나의 대화에서 여러 개의 기억을 뽑아낼 수 있으므로, 배열(List) 형태의 클래스를 하나 더 쌉니다.
class MemoryCandidateList(BaseModel):
    items: List[MemoryCandidate] = Field(description="추출된 장기 기억 후보들의 배열")

def test_structured_output():
    print("\n--- [테마 1] 구조화된 메모리 강제 추출기 (memory_schema 호환) ---")
    
    # 모델 호출 (with_structured_output을 달아주면 우리가 짠 MemoryCandidateList 배열만 내뱉습니다!)
    llm = ChatGoogleGenerativeAI(model=MEMORY_MODEL, google_api_key=GEMINI_API_KEY_FREE)
    structured_llm = llm.with_structured_output(MemoryCandidateList)
    
    # 모델이 규칙을 지키도록 memory_schema.txt의 가이드라인을 시스템 프롬프트로 강하게 박아줍니다.
    system_prompt = SystemMessage(content="""
너의 작업은 대화에서 장기 기억 후보를 추출하는 것이다.
다음 규칙을 무조건 지켜.

[허용된 memory_key 와 정규화된 memory_value 규칙]
- profile.user_name (값: 원래 이름)
- style.preferred_speech_style (값: polite 또는 casual)
- preference.favorite_genre (값: fantasy, sci_fi, romance, mystery 중 하나)
- taboo.disliked_topic (값: politics, religion, personal_life 중 하나)

모르는 정보는 억지로 추출하지 마라.
""")

    raw_chat_log = "안녕! 나는 배민이라고 해. 앞으로는 나한테 무조건 존댓말 써줘. 오늘 점심으로 돈까스 먹었는데 너무 맵더라. 밥 먹고 나서는 누워서 판타지 소설이나 보려고. 그리고 나 정치 이야기는 극혐하니까 절대 꺼내지마."
    human_msg = HumanMessage(content=raw_chat_log)
    
    print(f"입력 대화: {raw_chat_log}")
    print("LLM이 DB 스키마 규칙에 맞춰 파싱하는 중...\n")
    
    # 프롬프트와 대화 원문을 모델에 주입
    memory_result = structured_llm.invoke([system_prompt, human_msg])
    
    # 결과 출력
    for idx, memory in enumerate(memory_result.items, 1):
        print(f"[{idx}] 타입: {memory.memory_type:<10} | 키: {memory.memory_key:<22} | 값: {memory.memory_value:<9} | 신뢰도: {memory.confidence:.2f}")


# ==============================================================================
# [스터디 테마 2] Tool Calling (에이전트 함수 제어)
# LLM은 수학 계산이나 실시간 정보를 모릅니다. 하지만 도구(Tool)를 쥐여주면
# 지가 알아서 필요할 때 함수를 호출해서 스스로 계산해옵니다. (Agentic AI의 핵심)
# ==============================================================================

# 1. 챗봇이 쓸 수 있는 마법의 도구(함수)를 만듭니다. (반드시 docstring으로 설명을 적어줘야 LLM이 언제 쓸지 압니다)
@tool
def calculate_salary(hourly_wage: int, hours_per_week: int) -> str:
    """
    사용자의 시급과 주당 근무 시간을 받아서 세전 월급(4주 기준)을 계산해주는 도구입니다.
    돈이나 급여 계산과 관련된 질문이 들어오면 무조건 이 도구를 사용하세요.
    """
    monthly = hourly_wage * hours_per_week * 4
    return f"계산된 월급은 {monthly:,} 원입니다."

def test_tool_calling_agent():
    print("\n--- [테마 2] 스스로 도구를 쓰는 에이전트(Agent) ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY_FREE)
    tools = [calculate_salary]
    
    # 모델에게 도구가 든 가방을 묶어줍니다 (bind_tools)
    llm_with_tools = llm.bind_tools(tools)
    
    # 질문 1: 그냥 평범한 안부 인사 (도구 안 씀)
    system_prompt = SystemMessage(content="""
    <persona>
    You are a concise AI assistant. Respond briefly and naturally to casual conversation.
    </persona>
    <tool_rules>
    - NEVER advertise or mention your tools (e.g., calculator) unless directly asked by the user.
    - ONLY invoke a tool when the user explicitly provides the required parameters.
    - Respond with standard text if the input is general chitchat or lacks specific tool parameters.
    </tool_rules>
    """)
    query_1 = "안녕! 오늘 날씨 좋네."
    human_msg = HumanMessage(content=query_1)
    
    res_1 = llm_with_tools.invoke([system_prompt, human_msg])
    print(f"질문: {query_1}")
    print(f"도구 호출 여부: {bool(res_1.tool_calls)} / 대답: {res_1.content}\n")
    
    # 질문 2: 돈 계산을 요구함 (모델이 똑똑하게 '내가 수학은 직접 못 하니 도구를 쓰자!'고 판단함)
    query_2 = "나 알바 시작했어. 시급 12,000원 받고 일주일에 20시간 일하는데 월급 얼마 받을까?"
    res_2 = llm_with_tools.invoke(query_2)
    print(f"질문: {query_2}")
    if res_2.tool_calls:
        print(f"🔧 LLM이 계산기 도구를 꺼내 들었습니다!")
        print(f"   호출한 함수 이름: {res_2.tool_calls[0]['name']}")
        print(f"   LLM이 함수에 집어넣은 파라미터 값: {res_2.tool_calls[0]['args']}")
        tool_name_str = res_2.tool_calls[0]['name']   # "calculate_salary"
        tool_args_dict = res_2.tool_calls[0]['args']  # {'hourly_wage': 12000, ...}
        tools_map = {t.name: t for t in tools} 
        chosen_tool = tools_map[tool_name_str]
        answer = chosen_tool.invoke(tool_args_dict)
        print(f"   계산 결과: {answer}")


if __name__ == "__main__":
    #test_structured_output()
    test_tool_calling_agent()
