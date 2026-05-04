import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent

# 사용자가 .env 에 지정한 모델(예: GEMINI_MODEL_NAME)이 있으면 그걸 쓰고, 없으면 최신 pro 모델 사용
model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
gemini_model = f"gemini/{model_name}"

class BriefingAgents:
    def it_analyst(self):
        """
        HN 목록을 받아서 가장 중요한 IT/AI 뉴스를 선별하고 분석하는 요원.
        검색 도구 없음 - 주어진 데이터만 분석.
        """
        return Agent(
            role='시니어 IT/AI 트렌드 분석가',
            goal='제공된 Hacker News 기사 목록에서 IT/AI 관련 핵심 뉴스를 선별하고 중요도를 분석합니다.',
            backstory="""당신은 IT/AI 업계에서 10년 이상 근무한 시니어 분석가입니다.
            주어진 기사 목록에서 개발자와 업계 종사자에게 진짜 중요한 뉴스를 가려내는 판단력이 탁월합니다.
            기업 실적 발표, 일반 하드웨어 부품 뉴스보다는 신규 AI 모델 출시, 오픈소스 도구, 
            플랫폼 변화, M&A처럼 업계 판도에 영향을 주는 뉴스를 우선시합니다.
            절대로 주어진 목록 외의 내용을 지어내지 않습니다.""",
            verbose=True,
            allow_delegation=False,
            max_iter=3,  # 분석만 하므로 반복 최소화
            llm=gemini_model
        )

    def newsletter_editor(self):
        """
        분석가의 결과를 받아서 HTML 뉴스레터로 편집하는 요원.
        """
        return Agent(
            role='테크 뉴스레터 편집장',
            goal='분석가가 선별한 뉴스를 Gmail에서 바로 읽기 좋은 HTML 보고서로 작성합니다.',
            backstory="""당신은 '모닝 테크 브리핑'의 편집장입니다.
            분석가가 제공한 내용 외에 절대 내용을 추가하거나 추측하지 않습니다.
            마크다운 기호(#, **, -)는 절대 사용하지 않고 오직 HTML 태그만 사용합니다.""",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            llm=gemini_model
        )
