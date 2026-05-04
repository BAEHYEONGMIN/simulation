import pandas as pd
import yfinance as yf

ticker = "005930.KS" # 삼성전자 코스피
print(f"[{ticker}] 삼성전자 최근 10년치 주가 데이터를 다운로드 중입니다...")

# yf.download 대신 Ticker.history를 사용하면 CSV 헤더가 깔끔하게(단일 레벨로) 나옵니다.
stock = yf.Ticker(ticker)
df = stock.history(period="10y")

# 시간대(Timezone) 정보 제거하여 날짜만 깔끔하게 저장 (JS에서 파싱하기 쉽게)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

csv_filename = "005930_KS_OHLCV.csv"
df.to_csv(csv_filename)

print(f"완료! {csv_filename} 파일이 생성되었습니다.")
