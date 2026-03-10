from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from scanner import get_stock_list, analyze_stock

app = FastAPI(title="KR Stock Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/scan")
async def scan(
    market: str = "all",
    rsi_thresh: float = 30,
    vol_mult: float = 2.0,
    min_signals: int = 1,
):
    """전 종목 스캔 - 조건에 맞는 종목 반환"""
    stocks = get_stock_list(market)
    results = []

    async def analyze(stock):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, analyze_stock, stock, rsi_thresh, vol_mult
        )
        return result

    tasks = [analyze(s) for s in stocks]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in all_results:
        if isinstance(r, Exception) or r is None:
            continue
        if len(r.get("signals", [])) >= min_signals:
            results.append(r)

    # 신호 수 많은 순으로 정렬
    results.sort(key=lambda x: len(x["signals"]), reverse=True)
    return {"results": results, "total": len(results)}


@app.get("/api/chart/{code}")
async def get_chart(code: str):
    """특정 종목 상세 차트 데이터"""
    from scanner import get_chart_data
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, get_chart_data, code)
    return data
