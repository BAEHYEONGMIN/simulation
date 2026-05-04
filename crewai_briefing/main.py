import os
import datetime
from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process
from agents import BriefingAgents
from tasks import BriefingTasks
from notify import send_email
from hacker_news import fetch_top_stories, format_for_llm as hn_format
from rss_feeds import fetch_rss_stories, format_for_llm as rss_format

def main():
    # 0. API 키 확인
    if not os.environ.get("GEMINI_API_KEY"):
        print("🚨 오류: .env 파일에 GEMINI_API_KEY 가 셋팅되어야 합니다.")
        return

    print("=" * 60)
    print("🚀 모닝 테크 브리핑 에이전트 가동 시작...")
    print("=" * 60)

    # 1. 데이터 수집 (LLM 없이 Python으로 직접)
    hn_stories = fetch_top_stories(n_fetch=30, n_return=30)
    rss_stories = fetch_rss_stories(hours_back=24, n_per_feed=10)

    hn_data = hn_format(hn_stories) if hn_stories else "(HN 데이터 수집 실패)"
    rss_data = rss_format(rss_stories) if rss_stories else "(RSS 데이터 수집 실패)"

    print(f"\n✅ 데이터 수집 완료. (HN: {len(hn_stories)}개 / RSS: {len(rss_stories)}개)")
    print("🤖 LLM 분석 시작...\n")

    # 2. 요원(Agent) 소집
    agents = BriefingAgents()
    analyst = agents.it_analyst()
    editor = agents.newsletter_editor()

    # 3. 임무(Task) 할당
    tasks = BriefingTasks()
    analyze_task = tasks.analyze_stories(analyst, hn_data, rss_data)
    write_task = tasks.write_morning_briefing(editor)

    # 4. 크루(Crew) 결성
    crew = Crew(
        agents=[analyst, editor],
        tasks=[analyze_task, write_task],
        process=Process.sequential,
        verbose=True,
        max_execution_time=180
    )

    # 5. 업무 지시 (Kickoff)
    result = crew.kickoff()

    # 6. 결과물 이메일 발송
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    subject = f"🌅 [{today_str}] 모닝 테크 브리핑"

    print("\n✅ 보고서 작성 완료! 이메일을 발송합니다.")
    send_email(subject, str(result))
    print("📧 이메일 발송 완료!")

if __name__ == "__main__":
    main()
