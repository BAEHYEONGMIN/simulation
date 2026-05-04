import feedparser
import datetime

RSS_FEEDS = [
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/ai/feed/"
    },
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/"
    },
]

def fetch_rss_stories(hours_back: int = 24, n_per_feed: int = 10) -> list[dict]:
    """RSS 피드에서 최근 N시간 이내 기사를 가져옵니다."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)
    all_stories = []

    for feed_info in RSS_FEEDS:
        print(f"[RSS] '{feed_info['name']}' 피드 수집 중...")
        try:
            feed = feedparser.parse(feed_info["url"])
        except Exception as e:
            print(f"[RSS] '{feed_info['name']}' 수집 실패: {e}")
            continue

        count = 0
        for entry in feed.entries:
            if count >= n_per_feed:
                break

            # 발행일 파싱
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)

            # 발행일 필터 (없는 경우 일단 포함)
            if published and published < cutoff:
                continue

            all_stories.append({
                "source": feed_info["name"],
                "title": entry.get("title", "").strip(),
                "url": entry.get("link", ""),
                "published": published.strftime("%Y-%m-%d %H:%M UTC") if published else "날짜 불명",
            })
            count += 1

        print(f"[RSS] '{feed_info['name']}' {count}개 수집 완료.")

    return all_stories


def format_for_llm(stories: list[dict]) -> str:
    """LLM이 읽기 좋은 포맷으로 변환"""
    if not stories:
        return "(수집된 RSS 기사 없음)"

    lines = []
    for i, s in enumerate(stories, 1):
        lines.append(
            f"{i}. [{s['source']}] {s['title']}\n"
            f"   - 발행: {s['published']}\n"
            f"   - URL: {s['url']}"
        )
    return "\n\n".join(lines)
