# 🇰🇷 KR Stock Scanner

한국 주식 차트 기반 종목 검색기

## 파일 구조
```
stock-scanner/
├── main.py          ← FastAPI 서버
├── scanner.py       ← 분석 로직 (pykrx + 기술적 지표)
├── requirements.txt ← 라이브러리
├── Dockerfile       ← 서버 설정
└── static/
    └── index.html   ← 웹 화면
```

## 기능
- KOSPI / KOSDAQ 전 종목 스캔
- RSI 과매도 반등
- MACD 골든크로스
- 거래량 급증
- 삼각수렴 돌파 ⭐
- 눌림목 매수
- 52주 신고가 돌파

## 배포 방법 (Railway)

1. GitHub에 이 폴더를 업로드
2. railway.app 접속 → New Project → Deploy from GitHub
3. 자동 배포 완료 → URL 공유
