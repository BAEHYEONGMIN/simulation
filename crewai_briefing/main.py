import os
import datetime
from dotenv import load_dotenv

load_dotenv()  # 로컬 모듈 임포트보다 무조건 위로!

from crewai import Crew, Process
from agents import BriefingAgents
from tasks import BriefingTasks
from notify import send_email

def main():
    load_dotenv()

    # 0. API 키 확인
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        print("🚨 오류: .env 파일에 GEMINI_API_KEY 와 TAVILY_API_KEY 가 셋팅되어야 합니다.")
        return

    print("="*60)
    print("🚀 모닝 테크 브리핑 에이전트(CrewAI) 가동 시작...")
    print("="*60)

    # 1. 요원(Agent) 소집
    agents = BriefingAgents()
    researcher = agents.it_trend_researcher()
    editor = agents.newsletter_editor()

    # 2. 임무(Task) 할당
    tasks = BriefingTasks()
    research_task = tasks.research_it_trends(researcher)
    write_task = tasks.write_morning_briefing(editor)

    # 3. 크루(Crew) 결성
    crew = Crew(
        agents=[researcher, editor],
        tasks=[research_task, write_task],
        process=Process.sequential,  # 순차 진행: 리서치 -> 편집
        verbose=True,
        max_execution_time=300  # 5분 초과 시 강제 종료
    )

    # 4. 업무 지시 (Kickoff)
    print("\n🤖 요원들이 웹 검색 및 분석을 시작했습니다. (1~2분 정도 소요될 수 있습니다)")
    result = crew.kickoff()

    # 5. 결과물 이메일 발송 (HTML 그대로 전송)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    subject = f"🌅 [{today_str}] 모닝 테크 브리핑 by AI"

    email_body = str(result)

    print("\n✅ 보고서 작성이 완료되었습니다! 이메일을 발송합니다.")
    send_email(subject, email_body)
    print("📧 이메일 발송 완료!")

if __name__ == "__main__":
    main()
