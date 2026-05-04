from crewai import Task
import datetime

class BriefingTasks:
    def analyze_stories(self, agent, hn_data: str, rss_data: str):
        today = datetime.datetime.now().strftime("%Y년 %m월 %d일")
        return Task(
            description=f"""아래는 오늘({today}) 두 가지 소스에서 수집한 IT/AI 관련 최신 기사 목록입니다.

[소스 A] Hacker News - 개발자 커뮤니티에서 오늘 가장 많이 반응한 기사
{hn_data}

[소스 B] 테크 미디어 RSS (VentureBeat AI, TechCrunch AI) - 오늘 발행된 업계 주요 뉴스
{rss_data}

두 소스를 종합하여 오늘 가장 중요하고 의미있는 기사 4~5개를 선별하고 분석하세요.

[선별 기준 - 우선순위 높은 것]
- 신규 AI 모델 출시 또는 주요 성능 발표
- 주요 기업의 인수합병(M&A) 또는 전략적 파트너십
- 개발자에게 실질적 영향을 주는 플랫폼/도구 변화
- 업계 판도를 바꿀 수 있는 정책/투자 뉴스
- 오픈소스 주요 프로젝트 출시

[각 선별 기사에 대해 작성할 내용]
- 제목, 출처(HN 또는 RSS 미디어명), URL
- 커뮤니티 반응 (HN 기사의 경우 점수/댓글 수)
- 이 기사가 왜 중요한지 (2~3문장)""",
            expected_output="선별된 4~5개 기사의 제목, 출처, URL, 중요성 설명이 포함된 분석 리포트 (한국어)",
            agent=agent
        )

    def write_morning_briefing(self, agent):
        today = datetime.datetime.now().strftime("%Y년 %m월 %d일")
        return Task(
            description=f"""분석가가 선별한 기사 목록을 바탕으로 아침 브리핑 보고서를 HTML로 작성하세요.

[필수 규칙]
- 분석가가 제공한 내용 외에 절대 추가하거나 추측하지 마세요.
- 마크다운(#, **, -, ---) 절대 사용 금지. 오직 HTML 태그만 사용하세요.
- 모든 URL은 <a href="URL" target="_blank">링크</a> 형태로 삽입하세요.

[출력 HTML 포맷]
<h2>🌅 {today} 모닝 테크 브리핑</h2>
<p style="color:#888; font-size:13px;">출처: Hacker News + VentureBeat AI + TechCrunch AI</p>
<hr>

<h3>📰 오늘의 주요 IT & AI 소식</h3>

<!-- 각 기사마다 반복 -->
<h4>[번호]. [기사 제목]</h4>
<p style="color:#888; font-size:12px;">출처: [미디어명] | <a href="[URL]" target="_blank">원문 보기</a></p>

<p><b>📌 핵심 내용</b><br>
[기사의 핵심 내용. 3~4문장]</p>

<p><b>💡 왜 주목해야 하나?</b><br>
[개발자/업계에 미치는 영향. 2~3문장]</p>

<p style="color:#666; font-size:12px;">💬 HN 커뮤니티 반응: [점수]pts / [댓글]개 (HN 기사인 경우만 표시)</p>
<hr>""",
            expected_output="Gmail에서 바로 렌더링 가능한 HTML 형식의 아침 보고서. 마크다운 없음. 출처와 링크 포함.",
            agent=agent
        )
