import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import traceback

KOSPI_STOCKS = [
    ("005930","삼성전자"),("000660","SK하이닉스"),("035420","NAVER"),
    ("005380","현대차"),("051910","LG화학"),("006400","삼성SDI"),
    ("035720","카카오"),("012330","현대모비스"),("028260","삼성물산"),
    ("096770","SK이노베이션"),("034730","SK"),("055550","신한지주"),
    ("105560","KB금융"),("000270","기아"),("068270","셀트리온"),
    ("003670","POSCO홀딩스"),("032830","삼성생명"),("030200","KT"),
    ("017670","SK텔레콤"),("066570","LG전자"),("010950","S-Oil"),
    ("009150","삼성전기"),("018260","삼성에스디에스"),("011170","롯데케미칼"),
    ("000810","삼성화재"),("086790","하나금융지주"),("316140","우리금융지주"),
    ("024110","기업은행"),("139480","이마트"),("004020","현대제철"),
]

KOSDAQ_STOCKS = [
    ("247540","에코프로비엠"),("086520","에코프로"),("196170","알테오젠"),
    ("293490","카카오게임즈"),("357780","솔브레인"),("039030","이오테크닉스"),
    ("064760","티씨케이"),("145020","휴젤"),("214150","클래시스"),
    ("091990","셀트리온헬스케어"),("041510","에스엠"),("035900","JYP Ent"),
    ("122870","와이지엔터테인먼트"),("112040","위메이드"),("263750","펄어비스"),
    ("253450","스튜디오드래곤"),("067310","하나마이크론"),("078340","컴투스"),
    ("036570","엔씨소프트"),("095660","네오위즈"),
]

def get_stock_list(market: str = "all") -> list[dict]:
    result = []
    if market in ("all", "kospi"):
        for code, name in KOSPI_STOCKS:
            result.append({"code": code, "name": name, "market": "KOSPI"})
    if market in ("all", "kosdaq"):
        for code, name in KOSDAQ_STOCKS:
            result.append({"code": code, "name": name, "market": "KOSDAQ"})
    print(f"총 종목 수: {len(result)}")
    return result

def load_ohlcv(code: str, days: int = 300):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = fdr.DataReader(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return None
        if "volume" not in df.columns:
            df["volume"] = 0
        df.index = pd.to_datetime(df.index)
        return df.dropna(subset=["close"])
    except Exception as e:
        print(f"OHLCV 오류 {code}: {e}")
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def calc_sma(series, period):
    return series.rolling(period).mean()

def detect_triangle_breakout(df, window=60):
    if len(df) < window + 5:
        return False
    recent = df.iloc[-window:]
    highs = recent["high"].values if "high" in df.columns else recent["close"].values
    lows = recent["low"].values if "low" in df.columns else recent["close"].values
    x = np.arange(window)
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    converging = high_slope < 0 and low_slope > 0
    projected_high = np.poly1d(np.polyfit(x, highs, 1))(window - 1)
    return converging and df["close"].iloc[-1] > projected_high

def detect_signals(df, rsi_thresh, vol_mult):
    signals = []
    close = df["close"]
    volume = df["volume"]
    rsi = calc_rsi(close)
    macd, macd_sig, _ = calc_macd(close)
    sma20 = calc_sma(close, 20)
    sma60 = calc_sma(close, 60)

    if pd.notna(rsi.iloc[-2]) and pd.notna(rsi.iloc[-1]):
        if rsi.iloc[-2] < rsi_thresh and rsi.iloc[-1] > rsi.iloc[-2]:
            signals.append("rsi")

    if all(pd.notna(v) for v in [macd.iloc[-1], macd_sig.iloc[-1], macd.iloc[-2], macd_sig.iloc[-2]]):
        if macd.iloc[-2] < macd_sig.iloc[-2] and macd.iloc[-1] >= macd_sig.iloc[-1]:
            signals.append("macd")

    avg_vol = volume.iloc[-20:].mean()
    if avg_vol > 0 and volume.iloc[-1] > avg_vol * vol_mult:
        signals.append("vol")

    if detect_triangle_breakout(df):
        signals.append("triangle")

    if pd.notna(sma20.iloc[-1]) and pd.notna(sma60.iloc[-1]):
        above60 = close.iloc[-1] > sma60.iloc[-1]
        near20 = abs(close.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1] < 0.02
        recent_high = close.iloc[-20:].max() > close.iloc[-1] * 1.05
        if above60 and near20 and recent_high:
            signals.append("pullback")

    high_col = df["high"] if "high" in df.columns else close
    high_52w = high_col.iloc[-252:].max() if len(df) >= 60 else high_col.max()
    if close.iloc[-1] >= high_52w * 0.995:
        signals.append("break")

    return signals

def analyze_stock(stock_info: dict, rsi_thresh: float, vol_mult: float):
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

def get_chart_data(code: str) -> dict:
    df = load_ohlcv(code, days=365)
    if df is None:
        return {"error": "데이터 없음"}
    close = df["close"]
    macd, macd_sig, macd_hist = calc_macd(close)
    rsi = calc_rsi(close)
    sma20 = calc_sma(close, 20)
    sma60 = calc_sma(close, 60)
    def to_list(s):
        return [round(v, 2) if pd.notna(v) else None for v in s]
    return {
        "code": code,
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "open":  [int(v) for v in df.get("open", close)],
        "high":  [int(v) for v in df.get("high", close)],
        "low":   [int(v) for v in df.get("low", close)],
        "close": [int(v) for v in close],
        "volume":[int(v) for v in df["volume"]],
        "sma20": to_list(sma20),
        "sma60": to_list(sma60),
        "macd":  to_list(macd),
        "macd_signal": to_list(macd_sig),
        "macd_hist":   to_list(macd_hist),
        "rsi":   to_list(rsi),
    }
