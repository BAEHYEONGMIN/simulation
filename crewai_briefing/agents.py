import os
from dotenv import load_dotenv
load_dotenv()  # 무조건 가장 먼저 실행!

from crewai import Agent
from crewai.tools import tool
from tavily import TavilyClient

@tool("internet_search")
def search_tool(query: str) -> str:
    """최신 정보를 얻기 위해 인터넷을 검색합니다. 검색어(query)를 입력하세요."""
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    return client.get_search_context(query=query, search_depth="advanced", max_results=5)

# 사용자가 .env 에 지정한 모델(예: GEMINI_MODEL_NAME)이 있으면 그걸 쓰고, 없으면 최신 pro 모델 사용
model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
gemini_model = f"gemini/{model_name}"

class BriefingAgents:
    def it_trend_researcher(self):
        return Agent(
            role='시니어 IT/AI 동향 분석가',
            goal='오늘자 기준 IT 트렌드와 가장 파급력 있는 AI 모델 최신 소식을 발굴합니다.',
            backstory="""당신은 실리콘밸리 최고의 IT 트렌드 분석가입니다. 
            반드시 도구로 검색한 결과만 사용하며, 자신의 사전 지식을 절대 섞지 않습니다.
            모든 주장에는 검색 결과에서 찾은 출처 URL을 반드시 함께 제시합니다.""",
            verbose=True,
            allow_delegation=False,
            max_iter=7,
            tools=[search_tool],
            llm=gemini_model
        )

    def newsletter_editor(self):
        return Agent(
            role='테크 뉴스레터 편집장',
            goal='리서처들이 가져온 정보를 바탕으로 출근길에 읽기 좋은 깔끔한 HTML 리포트를 작성합니다.',
            backstory="""당신은 '모닝 테크 브리핑'의 편집장입니다. 
            리서처가 제공한 내용과 출처 URL만을 사용하여 뉴스레터를 작성합니다.
            절대로 자신의 사전 지식으로 내용을 추가하거나 보충하지 않습니다.""",
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            llm=gemini_model
        )
