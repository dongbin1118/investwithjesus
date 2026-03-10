import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
import traceback

# ──────────────────────────────────────────
# 종목 목록
# ──────────────────────────────────────────
def get_stock_list(market: str = "all") -> list[dict]:
    today = datetime.now().strftime("%Y%m%d")
    result = []
    try:
        if market in ("all", "kospi"):
            tickers = stock.get_market_ticker_list(today, market="KOSPI")
            for t in tickers:
                name = stock.get_market_ticker_name(t)
                result.append({"code": t, "name": name, "market": "KOSPI"})
        if market in ("all", "kosdaq"):
            tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
            for t in tickers:
                name = stock.get_market_ticker_name(t)
                result.append({"code": t, "name": name, "market": "KOSDAQ"})
    except Exception as e:
        print(f"종목 목록 오류: {e}")
    return result


# ──────────────────────────────────────────
# OHLCV 데이터 로드
# ──────────────────────────────────────────
def load_ohlcv(code: str, days: int = 300) -> pd.DataFrame | None:
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = stock.get_market_ohlcv(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            code
        )
        if df is None or len(df) < 60:
            return None
        df.columns = ["open", "high", "low", "close", "volume", "change"]
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except Exception as e:
        print(f"OHLCV 오류 {code}: {e}")
        return None


# ──────────────────────────────────────────
# 기술적 지표 계산
# ──────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ──────────────────────────────────────────
# 삼각수렴 돌파 감지
# ──────────────────────────────────────────
def detect_triangle_breakout(df: pd.DataFrame, window: int = 60) -> bool:
    """
    최근 window일 내:
    - 고점들이 하락 추세 (내림 저항선)
    - 저점들이 상승 추세 (오름 지지선)
    - 최근 종가가 저항선 위로 돌파
    """
    if len(df) < window + 5:
        return False

    recent = df.iloc[-window:]
    highs = recent["high"].values
    lows = recent["low"].values
    x = np.arange(window)

    # 선형 회귀로 추세선 기울기
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    # 고점은 내려오고 저점은 올라가야 수렴
    converging = high_slope < 0 and low_slope > 0

    # 최근 5일 종가가 고점 추세선 위
    last_close = df["close"].iloc[-1]
    projected_high = np.poly1d(np.polyfit(x, highs, 1))(window - 1)
    breakout = last_close > projected_high

    return converging and breakout


# ──────────────────────────────────────────
# 신호 감지
# ──────────────────────────────────────────
def detect_signals(df: pd.DataFrame, rsi_thresh: float, vol_mult: float) -> list[str]:
    signals = []
    close = df["close"]
    volume = df["volume"]

    rsi = calc_rsi(close)
    macd, macd_signal, _ = calc_macd(close)
    sma20 = calc_sma(close, 20)
    sma60 = calc_sma(close, 60)

    cur_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    avg_vol = volume.iloc[-20:].mean()
    cur_vol = volume.iloc[-1]

    # ① RSI 과매도 반등
    if pd.notna(prev_rsi) and pd.notna(cur_rsi):
        if prev_rsi < rsi_thresh and cur_rsi > prev_rsi:
            signals.append("rsi")

    # ② MACD 골든크로스
    if (pd.notna(macd.iloc[-1]) and pd.notna(macd_signal.iloc[-1]) and
            pd.notna(macd.iloc[-2]) and pd.notna(macd_signal.iloc[-2])):
        if macd.iloc[-2] < macd_signal.iloc[-2] and macd.iloc[-1] >= macd_signal.iloc[-1]:
            signals.append("macd")

    # ③ 거래량 급증
    if avg_vol > 0 and cur_vol > avg_vol * vol_mult:
        signals.append("vol")

    # ④ 삼각수렴 돌파
    if detect_triangle_breakout(df):
        signals.append("triangle")

    # ⑤ 눌림목 (SMA60 위 + SMA20 근접)
    if pd.notna(sma20.iloc[-1]) and pd.notna(sma60.iloc[-1]):
        above60 = close.iloc[-1] > sma60.iloc[-1]
        near20 = abs(close.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1] < 0.02
        recent_high = close.iloc[-20:].max() > close.iloc[-1] * 1.05
        if above60 and near20 and recent_high:
            signals.append("pullback")

    # ⑥ 52주 신고가 돌파
    high_52w = df["high"].iloc[-252:].max() if len(df) >= 252 else df["high"].max()
    if close.iloc[-1] >= high_52w * 0.995:
        signals.append("break")

    return signals


# ──────────────────────────────────────────
# 종목 분석 (스캔용)
# ──────────────────────────────────────────
def analyze_stock(stock_info: dict, rsi_thresh: float, vol_mult: float) -> dict | None:
    try:
        code = stock_info["code"]
        df = load_ohlcv(code)
        if df is None:
            return None

        signals = detect_signals(df, rsi_thresh, vol_mult)
        if not signals:
            return None

        close = df["close"]
        rsi_val = calc_rsi(close).iloc[-1]
        avg_vol = df["volume"].iloc[-20:].mean()
        vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 0
        chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100

        return {
            "code": code,
            "name": stock_info["name"],
            "market": stock_info["market"],
            "close": int(close.iloc[-1]),
            "change": round(chg, 2),
            "rsi": round(float(rsi_val), 1) if pd.notna(rsi_val) else None,
            "vol_ratio": round(vol_ratio, 2),
            "signals": signals,
        }
    except Exception:
        traceback.print_exc()
        return None


# ──────────────────────────────────────────
# 차트 데이터 (상세)
# ──────────────────────────────────────────
def get_chart_data(code: str) -> dict:
    df = load_ohlcv(code, days=365)
    if df is None:
        return {"error": "데이터 없음"}

    close = df["close"]
    macd, macd_signal, macd_hist = calc_macd(close)
    rsi = calc_rsi(close)
    sma20 = calc_sma(close, 20)
    sma60 = calc_sma(close, 60)

    def to_list(s):
        return [round(v, 2) if pd.notna(v) else None for v in s]

    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    return {
        "code": code,
        "dates": dates,
        "open":   [int(v) for v in df["open"]],
        "high":   [int(v) for v in df["high"]],
        "low":    [int(v) for v in df["low"]],
        "close":  [int(v) for v in df["close"]],
        "volume": [int(v) for v in df["volume"]],
        "sma20":  to_list(sma20),
        "sma60":  to_list(sma60),
        "macd":   to_list(macd),
        "macd_signal": to_list(macd_signal),
        "macd_hist":   to_list(macd_hist),
        "rsi":    to_list(rsi),
    }
