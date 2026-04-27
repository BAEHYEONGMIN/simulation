import os
import sys
import xml.etree.ElementTree as ET
import urllib3
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

OC = os.environ.get("LAW_API_OC")
if not OC:
    raise KeyError("환경 변수 'LAW_API_OC'를 찾을 수 없습니다. .env 파일을 확인해주세요.")

SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 세션 및 재시도 설정 (법제처 서버 불안정 10054 에러 대비)
session = requests.Session()
retry = requests.packages.urllib3.util.retry.Retry(
    total=3, read=3, connect=3, backoff_factor=1.0, status_forcelist=(500, 502, 504)
)
adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def fetch_law_xml(mst: str):
    for target in ["eflaw", "law", "eflawjosub"]:
        params = {
            "OC": OC,
            "target": target,
            "MST": mst,
            "type": "XML"
        }

        res = session.get(
            SERVICE_URL,
            params=params,
            headers=HEADERS,
            timeout=10,
            verify=False
        )
        res.raise_for_status()

        xml_text = res.content.decode("utf-8", errors="replace")
        if "조문단위" not in xml_text:
            xml_text = res.content.decode("cp949", errors="replace")

        xml_text = xml_text.replace('encoding="UTF-8"', "")
        xml_text = xml_text.replace("encoding='UTF-8'", "")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            continue

        count = len(root.findall(".//조문단위"))
        print(f"   -> target={target}, 조문단위={count}")

        if count > 0:
            return root

    print(f"[에러] MST={mst}에서 조문단위를 찾지 못했습니다.")
    return None

def extract_article_content(jomun_element):
    """조문 엘리먼트에서 전체 텍스트(조문, 항, 호)를 추출합니다."""
    content_lines = []
    main_content = jomun_element.findtext('조문내용')
    if main_content:
        content_lines.append(main_content.strip())
        
    for hang in jomun_element.findall('.//항내용'):
        if hang.text: content_lines.append(hang.text.strip())
    for ho in jomun_element.findall('.//호내용'):
        if ho.text: content_lines.append(ho.text.strip())

    return "\n".join(content_lines)

def make_article_key(jomun):
    num = (jomun.findtext("조문번호") or "").strip()
    return num


def make_article_title(jomun):
    jomun_num = (jomun.findtext("조문번호") or "").strip()
    jomun_title = jomun.findtext("조문제목") or ""
    return f"제{jomun_num}조{jomun_title}"


def build_article_map(root):
    article_map = {}

    for jomun in root.findall(".//조문단위"):
        key = make_article_key(jomun)
        if not key:
            continue

        article_map[key] = {
            "title": make_article_title(jomun),
            "content": extract_article_content(jomun),
        }

    return article_map


def fetch_diff_articles(mst_new: str, mst_old: str = None):
    print(f"[{mst_new}] 최신 본문을 조회합니다...")
    root_new = fetch_law_xml(mst_new)
    if root_new is None:
        return []

    if not mst_old or mst_old == mst_new:
        return []

    print(f"[{mst_old}] 과거 본문을 조회합니다...")
    root_old = fetch_law_xml(mst_old)
    if root_old is None:
        return []

    new_map = build_article_map(root_new)
    old_map = build_article_map(root_old)

    all_keys = sorted(set(new_map.keys()) | set(old_map.keys()))

    results = []

    for key in all_keys:
        old_article = old_map.get(key)
        new_article = new_map.get(key)

        if old_article is None:
            results.append({
                "title": new_article["title"],
                "old_content": "(과거 조문 없음 - 신설)",
                "new_content": new_article["content"],
            })
            continue

        if new_article is None:
            results.append({
                "title": old_article["title"],
                "old_content": old_article["content"],
                "new_content": "(최신 조문 없음 - 삭제)",
            })
            continue

        if old_article["content"] != new_article["content"]:
            results.append({
                "title": new_article["title"],
                "old_content": old_article["content"],
                "new_content": new_article["content"],
            })

    print(f"변경 조문 {len(results)}개 발견")
    return results

def main():
    mst_new = sys.argv[1] if len(sys.argv) > 1 else "285367"
    mst_old = sys.argv[2] if len(sys.argv) > 2 else "276815" # 테스트용 과거 번호
    
    articles = fetch_diff_articles(mst_new, mst_old)
    
    if not articles:
        print("변경된 조문을 찾을 수 없습니다.")
        return
        
    print(f"\n총 {len(articles)}개의 조문에 대한 신구대조 텍스트가 추출되었습니다.\n")
    print("="*60)
    for idx, article in enumerate(articles, 1):
        print(f"[{idx}] {article['title']}")
        print(f"🔴 [과거 내용]\n{article['old_content']}\n")
        print(f"🟢 [최신 내용]\n{article['new_content']}")
        print("-" * 60)
        
    print("\n* 에이전트: 위 텍스트를 LLM에 전달하여 '두 내용을 비교해 마크다운 diff 형식으로 요약해줘'라고 요청하세요.")

if __name__ == "__main__":
    main()
