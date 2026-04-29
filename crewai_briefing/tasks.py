from crewai import Task
import datetime

class BriefingTasks:
    def research_it_trends(self, agent):
        today = datetime.datetime.now().strftime("%Y년 %m월 %d일")
        return Task(
            description=f"""오늘({today}) 기준 가장 뜨거운 IT/AI 뉴스를 심층 조사하세요.

[조사 방법 - 반드시 아래 순서대로 검색 도구를 여러 번 호출하세요]

1차 검색: "AI LLM model news {today}"
2차 검색: "tech industry major announcements {today}"
3차 검색: "IT 기술 뉴스 {today}"
4차 검색: "most discussed AI developer news this week"

[각 뉴스 항목에 대해 반드시 수집해야 할 정보]
- 무슨 일이 일어났는가? (구체적인 사건/발표 내용)
- 언제, 누가, 어떤 수치/스펙과 함께 발표했는가?
- 기존 대비 무엇이 달라졌는가? (이전 버전, 경쟁사 대비)
- 이 뉴스가 업계에 미치는 파급력은?
- 출처 URL

[필수 규칙]
- 반드시 검색 도구를 최소 3번 이상 호출하세요.
- 검색 결과에 없는 내용은 절대 추가하지 마세요.
- 각 항목당 최소 5문장 이상의 상세 내용을 수집하세요.
- 출처 URL은 반드시 포함하세요.""",
            expected_output="""3~5개의 주요 IT/AI 뉴스 항목. 
            각 항목마다 구체적 수치/날짜/발표자, 배경 맥락, 업계 파급력, 출처 URL 포함. (한국어)""",
            agent=agent
        )

    def write_morning_briefing(self, agent):
        today = datetime.datetime.now().strftime("%Y년 %m월 %d일")
        return Task(
            description=f"""리서처가 수집한 원문 데이터를 바탕으로 전문 테크 보고서를 HTML로 작성하세요.

[작성 기준]
- 리서처가 제공한 내용과 URL 외에 절대 내용을 추가하거나 추측하지 마세요.
- 각 항목은 제목만 나열하는 수준이 아니라, 읽는 사람이 해당 기사를 직접 읽지 않아도 
  전체 내용을 파악할 수 있을 만큼 충분히 상세하게 작성하세요.
- 마크다운(#, **, -, ---) 절대 사용 금지. 오직 HTML 태그만 사용하세요.

[출력 HTML 포맷 - 반드시 이 구조를 따르세요]

<h2>🌅 {today} 모닝 테크 브리핑</h2>
<hr>

<h3>📰 주요 IT & AI 소식</h3>

<!-- 각 뉴스 항목마다 아래 구조 반복 -->
<h4>[순번]. [뉴스 제목]</h4>

<p><b>📌 무슨 일이?</b><br>
[구체적으로 무슨 발표/사건이 있었는지. 날짜, 발표 주체, 핵심 수치 포함. 3~5문장]</p>

<p><b>🔍 배경 & 맥락</b><br>
[이 뉴스가 나오게 된 배경, 이전 버전 또는 경쟁사 대비 무엇이 달라졌는지. 2~3문장]</p>

<p><b>💡 왜 중요한가?</b><br>
[이 소식이 개발자/업계에 미치는 실질적인 영향과 시사점. 2~3문장]</p>

<p>🔗 출처: <a href="[URL]">[URL]</a></p>
<hr>""",
            expected_output="""Gmail에서 바로 렌더링 가능한 HTML 형식의 상세 테크 보고서.
            마크다운 기호 없음. 항목당 무슨 일, 배경 맥락, 왜 중요한지 섹션 포함. 출처 링크 포함.""",
            agent=agent
        )
