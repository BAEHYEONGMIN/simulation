import os
import xml.etree.ElementTree as ET
import urllib3
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

OC = os.environ.get("LAW_API_OC")
if not OC:
    raise KeyError("환경 변수 'LAW_API_OC'를 찾을 수 없습니다. .env 파일을 확인해주세요.")

# 대상 설정 (전체 법령 모니터링)
LAW_ID = "000162"  # 화학물질관리법
QUERY = "화학물질관리법"

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 공통 헤더
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

# 세션 및 재시도 설정 (법제처 서버 불안정 10054 에러 대비)
session = requests.Session()
retry = requests.packages.urllib3.util.retry.Retry(
    total=3, read=3, connect=3, backoff_factor=1.0, status_forcelist=(500, 502, 504)
)
adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def check_law_changes(days_back: int = 7):
    """
    연혁법령(eflaw) API를 호출하여 화학물질관리법 전체를 대상으로
    최근 N일 내에 공포(발표)되거나 시행된 개정 이력이 있는지 확인합니다.
    """
    today = datetime.now()
    
    # 조회 시작일과 종료일 계산 (00:00:00 기준 비교를 위해 날짜 문자열 사용)
    start_date = (today - timedelta(days=days_back)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    
    params = {
        "OC": OC,
        "target": "eflaw",  # 연혁법령 (한글 태그 깨짐 버그가 없는 가장 안정적인 API)
        "query": QUERY,
        "type": "XML",
        "display": 100 # 충분히 많은 연혁 가져오기
    }

    print(f"[전체 모니터링] '{QUERY}' 개정 이력 조회 중... ({start_date} ~ {end_date} 사이 공포/시행 기준)")
    
    res = session.get(SEARCH_URL, params=params, headers=HEADERS, timeout=10, verify=False)
    res.raise_for_status()

    # XML 파싱
    xml_text = res.content.decode('utf-8', errors='replace')
    if 'law' not in xml_text:
        xml_text = res.content.decode('cp949', errors='replace')
        
    xml_text = xml_text.replace('encoding="UTF-8"', '').replace("encoding='UTF-8'", "")
    
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"XML 파싱 에러: {e}")
        return False, []
    
    revisions = []
    seen_msts = set()
    
    for law in root.findall('.//law'):
        # [핵심 수정] 하드코딩된 LAW_ID 대신, 이름과 법령 구분을 엄격히 필터링
        law_name = law.findtext('법령명한글') or ""
        law_type = law.findtext('법령구분명') or ""
        
        if law_name != "화학물질관리법":
            continue
        if law_type != "법률":  # 시행령, 시행규칙 등 제외
            continue
            
        law_mst = law.findtext('법령일련번호')
        
        # 중복 MST 제거 (동일 개정안 방지)
        if law_mst in seen_msts:
            continue
        seen_msts.add(law_mst)
        
        amend_type = law.findtext('제개정구분명') or "알수없음"
        promulgate_date = law.findtext('공포일자') or ""
        enforce_date = law.findtext('시행일자') or ""
        
        revisions.append({
            "MST": law_mst,
            "type": amend_type,
            "promulgate_date": promulgate_date,
            "enforce_date": enforce_date
        })
            
    # [핵심 수정] 시행일자(enforce_date) 및 공포일자 기준으로 내림차순 정렬
    # (최신 시행본이 가장 먼저 오도록 하여 MST_OLD 계산의 정확도를 높임)
    revisions.sort(key=lambda x: (x.get('enforce_date', ''), x.get('promulgate_date', '')), reverse=True)
    
    changes = []
    for i, rev in enumerate(revisions):
        p_date = rev['promulgate_date']
        e_date = rev['enforce_date']
        
        is_changed = False
        date_reason = ""
        
        if p_date and (start_date <= p_date <= end_date):
            is_changed = True
            date_reason = f"공포됨 ({p_date})"
        elif e_date and (start_date <= e_date <= end_date):
            is_changed = True
            date_reason = f"시행됨 ({e_date})"
            
        if is_changed:
            # 정렬된 리스트에서 바로 다음 인덱스가 '직전 시행본(MST_OLD)'이 됨
            mst_old = revisions[i+1]['MST'] if (i + 1) < len(revisions) else rev['MST']
            
            changes.append({
                "MST_NEW": rev['MST'],
                "MST_OLD": mst_old,
                "type": rev['type'],
                "promulgate_date": p_date,
                "enforce_date": e_date,
                "reason": date_reason
            })
            
    if not changes:
        return False, []
        
    return True, changes

def main():
    # 에이전트가 실행 시 7일(혹은 1일) 간격으로 모니터링
    # 테스트를 위해 범위를 1000일로 넓혀보겠습니다.
    is_changed, changes = check_law_changes(days_back=3)
    
    print("\n" + "="*50)
    if is_changed:
        print(f"[알림] '{QUERY}' 전체에서 새로운 개정 이력이 발견되었습니다!")
        for idx, change in enumerate(changes, 1):
            print(f"  {idx}. 일련번호: {change['MST_NEW']} (이전: {change['MST_OLD']}) | 유형: {change['type']}")
            print(f"     -> 알림 사유: 설정 기간 내 {change['reason']}")
            print(f"     -> (공포일자: {change['promulgate_date']}, 시행일자: {change['enforce_date']})")
        print("\n* 에이전트 다음 행동: 본문(lawService.do?MST=...)을 호출하여 상세 내용을 요약하세요.")
    else:
        print(f"[완료] 설정한 기간 동안 '{QUERY}'의 개정 이력이 없습니다.")
        print("* 에이전트 다음 행동: 변경 없음으로 리포트하고 프로세스를 종료합니다.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()