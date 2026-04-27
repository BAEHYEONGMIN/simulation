from get_law import check_law_changes, QUERY
from get_law_detail import fetch_diff_articles
from diff_util import get_markdown_diff
from notify import send_email

def run_agent_workflow():
    print("=" * 70)
    print("[시스템] 법령 모니터링 자동화 워크플로우 시작 (Diff 모드)")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # STEP 1: 감지기 (get_law.py)
    # ---------------------------------------------------------
    print(f"\n[STEP 1] '{QUERY}' 변경 여부 감지 중...")
    is_changed, changes = check_law_changes(days_back=30)
    
    if not is_changed:
        print("\n[알림] 설정한 기간 내 변경된 이력이 없습니다. 오늘 작업을 종료합니다.")
        return
        
    print(f"\n[경고] 총 {len(changes)}건의 개정 이력이 발견되었습니다!")
    
    all_results = []

    for change in changes:
        mst_new = change["MST_NEW"]
        mst_old = change["MST_OLD"]

        print(f"\n-> 분석 타겟: MST {mst_new} / 이전 {mst_old}")
        print(f"   - 감지 사유: {change['reason']}")

        articles = fetch_diff_articles(mst_new, mst_old)
        
        for article in articles:
            all_results.append(article)
            
        for idx, article in enumerate(articles, 1):
            diff_text = get_markdown_diff(
                article["old_content"],
                article["new_content"]
            )

            print(f"\n- [{idx}] {article['title']}")
            print("[변경 내역]")
            print(diff_text)

    # ---------------------------------------------------------
    # STEP 2: 최종 결과 출력 (LLM 전송용)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("[최종 결과] LLM 요약용 Diff 데이터")
    print("=" * 70)
    seen = set()
    unique_results = []

    for article in all_results:
        key = (article["title"], article["new_content"])

        if key not in seen:
            seen.add(key)
            unique_results.append(article)
    for idx, article in enumerate(unique_results, 1):
        diff_text = get_markdown_diff(
            article["old_content"],
            article["new_content"]
        )

        print(f"\n- [{idx}] {article['title']}")
        print("[변경 내역]")
        print(diff_text)
        print("-" * 50)
        send_email(f"화학물질관리법령 개정 알림: {article['title']}", diff_text)

if __name__ == "__main__":
    run_agent_workflow()
