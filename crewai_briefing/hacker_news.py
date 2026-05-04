import requests

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

def fetch_top_stories(n_fetch: int = 30, n_return: int = 30) -> list[dict]:
    """HN 상위 n개 스토리를 가져옵니다. URL 없는 항목(Ask HN, Jobs 등)만 제외."""
    print(f"[HN] 상위 {n_fetch}개 스토리 조회 중...")
    try:
        ids = requests.get(HN_TOP_STORIES_URL, timeout=10).json()[:n_fetch]
    except Exception as e:
        print(f"[HN] 스토리 목록 조회 실패: {e}")
        return []

    stories = []
    for story_id in ids:
        try:
            item = requests.get(HN_ITEM_URL.format(story_id), timeout=5).json()
        except Exception:
            continue

        # story 타입이 아니거나 URL이 없는 항목(Ask HN, Show HN 등) 제외
        if not item or item.get("type") != "story" or not item.get("url"):
            continue

        stories.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "score": item.get("score", 0),
            "comments": item.get("descendants", 0),
            "hn_link": f"https://news.ycombinator.com/item?id={story_id}"
        })

    # 커뮤니티 반응(점수 + 댓글) 기준 내림차순 정렬
    stories.sort(key=lambda x: x["score"] + x["comments"] * 2, reverse=True)

    print(f"[HN] 유효 기사 {len(stories)}개 수집 완료. 상위 {n_return}개 LLM에 전달.")
    return stories[:n_return]


def format_for_llm(stories: list[dict]) -> str:
    """LLM이 읽기 좋은 포맷으로 변환"""
    lines = []
    for i, s in enumerate(stories, 1):
        lines.append(
            f"{i}. {s['title']}\n"
            f"   - 커뮤니티 반응: 점수 {s['score']}pts / 댓글 {s['comments']}개\n"
            f"   - 원문 URL: {s['url']}\n"
            f"   - HN 토론: {s['hn_link']}"
        )
    return "\n\n".join(lines)
