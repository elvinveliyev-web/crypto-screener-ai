import os
import re
import json
import math
import time
import base64
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

# =============================
# OPTIONAL PDF SUPPORT (ReportLab)
# =============================
REPORTLAB_OK = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Bitcoin FA→TA + AI", layout="wide", page_icon="₿")

# =============================
# BASE DIR
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)


# =============================
# EXCHANGE
# =============================
exchange = ccxt.kucoin({"enableRateLimit": True})
DEFAULT_SYMBOL = "BTC/USDT"

TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1wk": 604_800_000,
}

PERIOD_TO_DAYS = {
    "45d": 45,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}

CATEGORY_MAP = {
    "BTC": "Majors",
    "ETH": "Majors",
    "BNB": "Majors",
    "SOL": "Layer-1",
    "ADA": "Layer-1",
    "AVAX": "Layer-1",
    "DOT": "Layer-1",
    "ATOM": "Layer-1",
    "LINK": "Oracle",
    "UNI": "DeFi",
    "AAVE": "DeFi",
    "MKR": "DeFi",
    "SUSHI": "DeFi",
    "XRP": "Payments",
    "XLM": "Payments",
    "LTC": "Payments",
    "DOGE": "Meme",
    "SHIB": "Meme",
    "PEPE": "Meme",
    "ARB": "Layer-2",
    "OP": "Layer-2",
    "MATIC": "Layer-2",
    "IMX": "Gaming",
    "GALA": "Gaming",
    "SAND": "Metaverse",
    "MANA": "Metaverse",
}


# =============================
# HELPERS
# =============================

def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"


def fmt_num(x: float, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "N/A"


def pct_dist(level: Optional[float], base: Optional[float]) -> Optional[float]:
    try:
        if level is None or base is None or not np.isfinite(level) or not np.isfinite(base) or base == 0:
            return None
        return ((level / base) - 1.0) * 100.0
    except Exception:
        return None


def fmt_rr(x: Optional[float]) -> str:
    try:
        if x is None or not np.isfinite(x):
            return "N/A"
        return f"{x:.2f}"
    except Exception:
        return "N/A"


def ms_from_timeframe(timeframe: str) -> int:
    return TIMEFRAME_TO_MS.get(timeframe, 86_400_000)


def clip_df_to_end_date(df: pd.DataFrame, end_date: Optional[datetime.date]) -> pd.DataFrame:
    if end_date is None or df.empty:
        return df
    cutoff = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return df[df.index < cutoff].copy()


def _plotly_fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", width=1400, height=850, scale=1)
    except Exception:
        return None


def recommendation_from_latest(latest: pd.Series) -> str:
    score = safe_float(latest.get("SCORE"))
    entry = int(latest.get("ENTRY", 0))
    exit_ = int(latest.get("EXIT", 0))
    rsi_v = safe_float(latest.get("RSI"))

    if entry == 1 and score >= 75:
        return "GÜÇLÜ AL"
    if entry == 1 or score >= 65:
        return "AL"
    if exit_ == 1 and score <= 45:
        return "SAT"
    if score <= 30 or (np.isfinite(rsi_v) and rsi_v < 35 and exit_ == 1):
        return "GÜÇLÜ SAT"
    return "BEKLE / NÖTR"


# =============================
# INDICATORS
# =============================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.replace([np.inf, -np.inf], np.nan).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return mid, upper, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def force_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    return volume * (close - close.shift(1))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 5, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    den = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * ((close - lowest_low) / den)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)


def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13):
    e = ema(close, period)
    bull_power = high - e
    bear_power = low - e
    return e, bull_power, bear_power


def check_bullish_divergence(close: pd.Series, indicator: pd.Series, lookback: int = 30) -> Tuple[bool, int]:
    if len(close) < lookback:
        return False, 0
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        min_idx = c.values.argmin()
        bars_ago = (lookback - 1) - min_idx
        prev_c = c.iloc[: min_idx - 2]
        if len(prev_c) < 3:
            return False, 0
        prev_min_idx = prev_c.values.argmin()
        p1, p2 = prev_c.iloc[prev_min_idx], c.iloc[min_idx]
        i1, i2 = ind.iloc[prev_min_idx], ind.iloc[min_idx]
        if p2 < p1 and i2 > i1:
            return True, bars_ago
    except Exception:
        pass
    return False, 0


def check_bearish_divergence(close: pd.Series, indicator: pd.Series, lookback: int = 30) -> Tuple[bool, int]:
    if len(close) < lookback:
        return False, 0
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        max_idx = c.values.argmax()
        bars_ago = (lookback - 1) - max_idx
        prev_c = c.iloc[: max_idx - 2]
        if len(prev_c) < 3:
            return False, 0
        prev_max_idx = prev_c.values.argmax()
        p1, p2 = prev_c.iloc[prev_max_idx], c.iloc[max_idx]
        i1, i2 = ind.iloc[prev_max_idx], ind.iloc[max_idx]
        if p2 > p1 and i2 < i1:
            return True, bars_ago
    except Exception:
        pass
    return False, 0


def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = true_range(high, low, close)
    tr_smooth = pd.Series(tr, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
    pdm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    mdm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    pdi = 100 * (pdm_smooth / tr_smooth.replace(0, np.nan))
    mdi = 100 * (mdm_smooth / tr_smooth.replace(0, np.nan))
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)


# =============================
# PATTERNS / OVERBOUGHT
# =============================

def add_kangaroo_tails(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["KANGAROO_BULL"] = 0
    df["KANGAROO_BEAR"] = 0

    body = (df["Close"] - df["Open"]).abs()
    trange = df["High"] - df["Low"]
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)

    rolling_min = df["Low"].rolling(window=lookback).min()
    rolling_max = df["High"].rolling(window=lookback).max()
    atr_approx = trange.rolling(10).mean()
    valid_trange = trange > 0

    bull_cond = valid_trange & (df["Low"] == rolling_min) & ((body / trange) <= 0.3) & ((lower_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)
    bear_cond = valid_trange & (df["High"] == rolling_max) & ((body / trange) <= 0.3) & ((upper_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)

    df.loc[bull_cond, "KANGAROO_BULL"] = 1
    df.loc[bear_cond, "KANGAROO_BEAR"] = 1
    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    O = df["Open"]
    H = df["High"]
    L = df["Low"]
    C = df["Close"]

    body = (C - O).abs()
    range_ = H - L
    upper_wick = H - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - L
    avg_range = range_.rolling(10).mean()

    is_bull = C > O
    is_bear = C < O

    df["PATTERN_DOJI"] = (body <= 0.1 * range_) & (range_ > 0)
    df["PATTERN_LL_DOJI"] = df["PATTERN_DOJI"] & (upper_wick >= 0.35 * range_) & (lower_wick >= 0.35 * range_) & (range_ > avg_range * 0.8)

    shape_hammer = (lower_wick >= 2 * body) & (upper_wick <= 0.2 * range_) & (body > 0.02 * range_)
    df["PATTERN_HAMMER"] = shape_hammer & (C < df.get("EMA50", C))
    df["PATTERN_HANGING_MAN"] = shape_hammer & (C > df.get("EMA50", C))

    shape_star = (upper_wick >= 2 * body) & (lower_wick <= 0.2 * range_) & (body > 0.02 * range_)
    df["PATTERN_SHOOTING_STAR"] = shape_star & (C > df.get("EMA50", C))
    df["PATTERN_INV_HAMMER"] = shape_star & (C < df.get("EMA50", C))

    df["PATTERN_MARUBOZU_BULL"] = is_bull & (body >= 0.85 * range_) & (range_ > avg_range * 0.5)
    df["PATTERN_MARUBOZU_BEAR"] = is_bear & (body >= 0.85 * range_) & (range_ > avg_range * 0.5)

    prev_is_bear = is_bear.shift(1)
    prev_is_bull = is_bull.shift(1)
    prev_O = O.shift(1)
    prev_C = C.shift(1)

    df["PATTERN_ENGULFING_BULL"] = is_bull & prev_is_bear & (O <= prev_C) & (C >= prev_O) & (body > (prev_O - prev_C))
    df["PATTERN_ENGULFING_BEAR"] = is_bear & prev_is_bull & (O >= prev_C) & (C <= prev_O) & (body > (prev_C - prev_O))

    df["PATTERN_HARAMI_BULL"] = is_bull & prev_is_bear & (O > prev_C) & (C < prev_O) & ((prev_O - prev_C) > avg_range * 0.5)
    df["PATTERN_HARAMI_BEAR"] = is_bear & prev_is_bull & (O < prev_C) & (C > prev_O) & ((prev_C - prev_O) > avg_range * 0.5)

    prev_H = H.shift(1)
    prev_L = L.shift(1)
    df["PATTERN_TWEEZER_TOP"] = (abs(H - prev_H) <= 0.002 * C) & is_bear & prev_is_bull & (H > df.get("EMA50", C))
    df["PATTERN_TWEEZER_BOTTOM"] = (abs(L - prev_L) <= 0.002 * C) & is_bull & prev_is_bear & (L < df.get("EMA50", C))

    df["PATTERN_PIERCING"] = is_bull & prev_is_bear & (O < L.shift(1)) & (C > (prev_O + prev_C) / 2) & (C < prev_O)
    df["PATTERN_DARK_CLOUD"] = is_bear & prev_is_bull & (O > H.shift(1)) & (C < (prev_O + prev_C) / 2) & (C > prev_O)

    prev2_is_bear = is_bear.shift(2)
    prev2_is_bull = is_bull.shift(2)
    prev2_O = O.shift(2)
    prev2_C = C.shift(2)
    df["PATTERN_MORNING_STAR"] = is_bull & prev2_is_bear & (prev_C < prev2_C) & (O > prev_C) & (C > (prev2_O + prev2_C) / 2)
    df["PATTERN_EVENING_STAR"] = is_bear & prev2_is_bull & (prev_C > prev2_C) & (O < prev_C) & (C < (prev2_O + prev2_C) / 2)
    return df


def add_overbought_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI_OVERBOUGHT"] = (df["RSI"] > 70).astype(int)
    df["RSI_OVERSOLD"] = (df["RSI"] < 30).astype(int)

    bb_den = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_PERCENT_B"] = ((df["Close"] - df["BB_lower"]) / bb_den).replace([np.inf, -np.inf], np.nan)
    df["BB_OVERBOUGHT"] = (df["Close"] > df["BB_upper"]).astype(int)
    df["BB_OVERSOLD"] = (df["Close"] < df["BB_lower"]).astype(int)

    df["VOLUME_SMA20"] = df["Volume"].rolling(20).mean()
    df["VOLUME_SPIKE"] = (df["Volume"] > df["VOLUME_SMA20"] * 1.5).astype(int)

    df["PRICE_TO_EMA50"] = (df["Close"] / df["EMA50"] - 1) * 100
    df["PRICE_TO_EMA200"] = (df["Close"] / df["EMA200"] - 1) * 100
    df["PRICE_EXTREME"] = ((df["PRICE_TO_EMA50"] > 20) | (df["PRICE_TO_EMA200"] > 30)).astype(int)

    def stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
        rsi_vals = series
        min_rsi = rsi_vals.rolling(period).min()
        max_rsi = rsi_vals.rolling(period).max()
        den = (max_rsi - min_rsi).replace(0, np.nan)
        stoch = 100 * (rsi_vals - min_rsi) / den
        stoch = stoch.replace([np.inf, -np.inf], np.nan).fillna(50)
        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k, d

    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = stoch_rsi(df["RSI"])
    df["STOCH_OVERBOUGHT"] = (df["STOCH_RSI_K"] > 80).astype(int)

    df["VOLUME_DIR"] = np.sign(df["Volume"].diff()).fillna(0)
    df["PRICE_DIR"] = np.sign(df["Close"].diff()).fillna(0)
    df["WEAK_UPTREND"] = ((df["PRICE_DIR"] > 0) & (df["VOLUME_DIR"] < 0)).astype(int)
    return df


def detect_speculation(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    result = {
        "overbought_score": 0,
        "oversold_score": 0,
        "speculation_score": 0,
        "details": {},
        "short_percent_float": np.nan,
        "short_ratio": np.nan,
    }

    if last["RSI"] > 70:
        result["overbought_score"] += 40
        result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        result["oversold_score"] += 50
        result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"

    if bool(last["BB_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger üst bandında"
    elif bool(last["BB_OVERSOLD"]):
        result["oversold_score"] += 50
        result["details"]["bb"] = "Fiyat Bollinger alt bandında"

    if bool(last["STOCH_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["stoch"] = "Stokastik RSI aşırı alımda"

    if bool(last["VOLUME_SPIKE"]):
        result["speculation_score"] += 60
        result["details"]["volume"] = "Ani hacim artışı (spekülasyon)"

    if bool(last["PRICE_EXTREME"]):
        result["overbought_score"] += 20
        result["details"]["price_extreme"] = f"Fiyat EMA'dan çok uzak (EMA50: %{last['PRICE_TO_EMA50']:.1f})"

    if bool(last["WEAK_UPTREND"]):
        result["speculation_score"] += 40
        result["details"]["weak_trend"] = "Fiyat yükselirken hacim düşüyor (zayıflama)"

    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])

    if result["overbought_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERLİ (SAT bölgesi)"
    elif result["oversold_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERSİZ (AL bölgesi)"
    elif result["speculation_score"] >= 60:
        result["verdict"] = "SPEKÜLATİF HAREKET (dikkatli olunmalı)"
    else:
        result["verdict"] = "NÖTR (normal değer aralığı)"

    return result


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["EMA50"] = ema(df["Close"], int(cfg["ema_fast"]))
    df["EMA200"] = ema(df["Close"], int(cfg["ema_slow"]))
    df["RSI"] = rsi(df["Close"], int(cfg["rsi_period"]))
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"], 12, 26, 9)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(df["Close"], int(cfg["bb_period"]), float(cfg["bb_std"]))
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], int(cfg["atr_period"]))
    df["OBV"] = obv(df["Close"], df["Volume"])
    df["OBV_EMA"] = ema(df["OBV"], 21)
    df["VOL_SMA"] = df["Volume"].rolling(int(cfg["vol_sma"])).mean()
    df["ATR_PCT"] = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)

    bb_mid_safe = df["BB_mid"].replace(0, np.nan)
    df["BB_WIDTH"] = ((df["BB_upper"] - df["BB_lower"]) / bb_mid_safe).replace([np.inf, -np.inf], np.nan)
    vol_sma_safe = df["VOL_SMA"].replace(0, np.nan)
    df["VOL_RATIO"] = (df["Volume"] / vol_sma_safe).replace([np.inf, -np.inf], np.nan)

    df = add_overbought_indicators(df)
    df = add_kangaroo_tails(df)
    df = add_candlestick_patterns(df)
    return df


# =============================
# DATA
# =============================
@st.cache_data(ttl=3600, show_spinner=False)
def get_top_symbols(limit: int = 120) -> List[str]:
    try:
        tickers = exchange.fetch_tickers()
        rows = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT") or ":" in sym:
                continue
            qv = safe_float(t.get("quoteVolume") or t.get("info", {}).get("volValue"))
            lv = safe_float(t.get("last"))
            rows.append((sym, qv if np.isfinite(qv) else -1, lv))
        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        syms = [r[0] for r in rows[:limit]]
        if DEFAULT_SYMBOL not in syms:
            syms.insert(0, DEFAULT_SYMBOL)
        seen = []
        for s in syms:
            if s not in seen:
                seen.append(s)
        return seen
    except Exception:
        return [DEFAULT_SYMBOL, "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def _estimate_candles(period: str, timeframe: str) -> int:
    days = PERIOD_TO_DAYS.get(period, 365)
    if timeframe == "1wk":
        return max(60, math.ceil(days / 7) + 10)
    if timeframe == "1d":
        return days + 10
    if timeframe == "4h":
        return days * 6 + 20
    if timeframe == "1h":
        return days * 24 + 50
    return days + 20


def _fetch_ohlcv_iter(symbol: str, timeframe: str, since_ms: int, end_ms: int, limit_hint: int) -> List[list]:
    all_bars: List[list] = []
    step_ms = ms_from_timeframe(timeframe)
    cursor = since_ms
    max_batch = 1000
    loops = 0

    while cursor < end_ms and loops < 30 and len(all_bars) < limit_hint + 300:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=max_batch)
        if not batch:
            break
        all_bars.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        loops += 1
        time.sleep(exchange.rateLimit / 1000.0 if hasattr(exchange, "rateLimit") else 0.05)

    uniq = {}
    for b in all_bars:
        uniq[b[0]] = b
    ordered = [uniq[k] for k in sorted(uniq.keys()) if k < end_ms]
    return ordered


@st.cache_data(ttl=180, show_spinner=False)
def load_data_cached(symbol: str, period: str, interval: str, end_date=None) -> pd.DataFrame:
    tf = interval
    if tf == "1wk":
        tf = "1w"

    days = PERIOD_TO_DAYS.get(period, 365)
    limit_est = _estimate_candles(period, interval)

    if end_date is not None:
        end_dt = datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time.min)
    else:
        end_dt = datetime.datetime.utcnow() + datetime.timedelta(days=1)

    start_dt = end_dt - datetime.timedelta(days=days + 10)
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    try:
        bars = _fetch_ohlcv_iter(symbol, tf, since_ms, end_ms, limit_est)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = clip_df_to_end_date(df, end_date)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(symbol: str) -> dict:
    out = {"last_price": np.nan, "change_pct": np.nan, "quote_volume": np.nan, "asof": ""}
    try:
        t = exchange.fetch_ticker(symbol)
        out["last_price"] = safe_float(t.get("last"))
        pct = safe_float(t.get("percentage"))
        out["change_pct"] = pct / 100.0 if np.isfinite(pct) else np.nan
        out["quote_volume"] = safe_float(t.get("quoteVolume"))
        ts = t.get("timestamp")
        if ts:
            out["asof"] = str(pd.to_datetime(ts, unit="ms"))
    except Exception:
        pass
    return out


@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_regime_series(symbol: str = DEFAULT_SYMBOL) -> pd.Series:
    df = load_data_cached(symbol, "2y", "1d")
    if df.empty or len(df) < 200:
        return pd.Series(dtype=bool)
    df = df.copy()
    df["EMA200"] = ema(df["Close"], 200)
    return (df["Close"] > df["EMA200"]).astype(bool)


@st.cache_data(ttl=600, show_spinner=False)
def get_higher_tf_trend_series(symbol: str, higher_tf_interval: str = "1wk", ema_period: int = 200) -> pd.Series:
    df = load_data_cached(symbol, "2y", higher_tf_interval)
    if df.empty or len(df) < min(ema_period, 60):
        return pd.Series(dtype=bool)
    df = df.copy()
    df["EMA"] = ema(df["Close"], ema_period)
    return (df["Close"] > df["EMA"]).astype(bool)


# =============================
# STRATEGY / BACKTEST
# =============================

def signal_with_checkpoints(
    df: pd.DataFrame,
    cfg: dict,
    market_filter_series: pd.Series = None,
    higher_tf_filter_series: pd.Series = None,
):
    df = df.copy()
    liq_ok = (df["Volume"] > df["VOL_SMA"]).fillna(False)
    trend_ok = (df["Close"] > df["EMA200"]) & (df["EMA50"] > df["EMA200"])

    if market_filter_series is not None and not market_filter_series.empty:
        aligned_market = market_filter_series.reindex(df.index).ffill().fillna(True)
    else:
        aligned_market = pd.Series(True, index=df.index)

    if higher_tf_filter_series is not None and not higher_tf_filter_series.empty:
        aligned_htf = higher_tf_filter_series.reindex(df.index).ffill().fillna(True)
    else:
        aligned_htf = pd.Series(True, index=df.index)

    rsi_ok = df["RSI"] > cfg["rsi_entry_level"]
    rsi_cross = (df["RSI"] > cfg["rsi_entry_level"]) & (df["RSI"].shift(1) <= cfg["rsi_entry_level"])
    macd_ok = df["MACD_hist"] > 0
    macd_turn = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)
    atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    vol_ok = atr_pct < cfg["atr_pct_max"]
    bb_ok = df["Close"] > df["BB_mid"]
    bb_break = (df["Close"] > df["BB_upper"]) & trend_ok
    obv_ok = df["OBV"] > df["OBV_EMA"]

    w = {"liq": 10, "trend": 25, "rsi": 15, "macd": 15, "vol": 10, "bb": 15, "obv": 10}
    score = (
        w["liq"] * liq_ok.astype(int)
        + w["trend"] * trend_ok.astype(int)
        + w["rsi"] * rsi_ok.astype(int)
        + w["macd"] * macd_ok.astype(int)
        + w["vol"] * vol_ok.astype(int)
        + w["bb"] * (bb_ok | bb_break).astype(int)
        + w["obv"] * obv_ok.astype(int)
    ).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 1
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & aligned_market & aligned_htf
    exit_ = (
        (df["Close"] < df["EMA50"])
        | (df["MACD_hist"] < 0)
        | (df["RSI"] < cfg["rsi_exit_level"])
        | (df["Close"] < df["BB_mid"])
    )

    df["SCORE"] = score
    df["ENTRY"] = entry.astype(int)
    df["EXIT"] = exit_.astype(int)

    last = df.iloc[-1]
    cp = {
        "Market Filter OK": bool(aligned_market.iloc[-1]),
        "Higher TF Filter OK": bool(aligned_htf.iloc[-1]),
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"])) if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"]) if pd.notna(last["ATR"]) and pd.notna(last["Close"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"])) if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }
    return df, cp


def backtest_long_only(
    df: pd.DataFrame,
    cfg: dict,
    risk_free_annual: float,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 365,
):
    df = df.copy()
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int)
    exit_sig = df["EXIT"].shift(1).fillna(0).astype(int)

    cash = float(cfg["initial_capital"])
    shares = 0.0
    stop = np.nan
    entry_price = np.nan
    target_price = np.nan
    bars_held = 0
    half_sold = False

    trades = []
    equity_curve = []

    commission = cfg["commission_bps"] / 10000.0
    slippage = cfg["slippage_bps"] / 10000.0
    time_stop_bars = cfg.get("time_stop_bars", 10)
    tp_mult = cfg.get("take_profit_mult", 2.0)
    risk_pct = float(cfg.get("risk_per_trade", 0.01))

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = float(row["Close"])

        if shares > 0 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        if shares == 0 and entry_sig.iloc[i] == 1:
            atrv = float(row.get("ATR", np.nan))
            if pd.notna(atrv) and atrv > 0:
                risk_amount = cash * risk_pct
                is_kangaroo = int(row.get("KANGAROO_BULL", 0)) == 1
                if is_kangaroo:
                    stop_price = float(row["Low"]) - (0.5 * atrv)
                    stop_dist = price - stop_price
                else:
                    stop_dist = cfg["atr_stop_mult"] * atrv
                    stop_price = price - stop_dist

                if stop_dist > 0:
                    potential_shares = risk_amount / stop_dist
                    max_shares = cash / (price * (1 + slippage + commission))
                    shares_to_buy = min(potential_shares, max_shares)
                    if shares_to_buy > 1e-9:
                        shares = shares_to_buy
                        entry_price = price * (1 + slippage)
                        fee = (shares * entry_price) * commission
                        cash -= ((shares * entry_price) + fee)
                        stop = stop_price
                        target_price = entry_price + (tp_mult * stop_dist)
                        trades.append({
                            "entry_date": date,
                            "entry_price": entry_price,
                            "equity_before": cash + (shares * price),
                        })

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value

        if shares > 0:
            bars_held += 1
            stop_hit = pd.notna(stop) and (price <= stop)
            target_hit = (not half_sold) and pd.notna(target_price) and (price >= target_price)
            time_stop_hit = (bars_held >= time_stop_bars) and (price < entry_price)

            if target_hit:
                sell_shares = shares * 0.5
                sell_price = price * (1 - slippage)
                gross = sell_shares * sell_price
                fee = gross * commission
                cash += (gross - fee)
                shares -= sell_shares
                half_sold = True
                stop = max(stop, entry_price)
                if len(trades) > 0:
                    trades[-1]["pnl"] = cash + (shares * price * (1 - slippage)) - trades[-1]["equity_before"]

            if exit_sig.iloc[i] == 1 or stop_hit or time_stop_hit:
                sell_price = price * (1 - slippage)
                gross = shares * sell_price
                fee = gross * commission
                cash += (gross - fee)
                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                if stop_hit:
                    reason = "STOP"
                elif time_stop_hit:
                    reason = "TIME_STOP"
                else:
                    reason = "RULE_EXIT"
                trades[-1]["exit_reason"] = reason
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]
                shares = 0.0
                stop = np.nan
                entry_price = np.nan
                target_price = np.nan
                bars_held = 0
                half_sold = False

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()
    ret = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (periods_per_year / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(periods_per_year)) if len(ret) > 1 else 0.0

    rf_daily = (1 + float(risk_free_annual)) ** (1 / periods_per_year) - 1
    excess = ret - rf_daily
    sharpe = float((excess.mean() * periods_per_year) / (excess.std() * np.sqrt(periods_per_year))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * periods_per_year) / downside_dev) if downside_dev > 0 else 0.0

    mdd = max_drawdown(eq)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0

    if benchmark_returns is not None:
        common_dates = ret.index.intersection(benchmark_returns.index)
        if len(common_dates) > 5:
            r_aligned = ret.loc[common_dates]
            b_aligned = benchmark_returns.loc[common_dates]
            cov = np.cov(r_aligned, b_aligned)[0, 1]
            var_b = np.var(b_aligned)
            beta = cov / var_b if var_b != 0 else 1.0
            mean_r = r_aligned.mean() * periods_per_year
            mean_b = b_aligned.mean() * periods_per_year
            alpha = (mean_r - risk_free_annual) - beta * (mean_b - risk_free_annual)
            diff = r_aligned - b_aligned
            info_ratio = (diff.mean() * periods_per_year) / (diff.std() * np.sqrt(periods_per_year)) if diff.std() > 0 else 0.0
        else:
            beta = 1.0
            alpha = 0.0
            info_ratio = 0.0
    else:
        beta = 1.0
        alpha = 0.0
        info_ratio = 0.0

    peak = eq.cummax()
    drawdown_pct = (eq - peak) / peak
    ulcer_index = np.sqrt((drawdown_pct**2).mean()) if len(drawdown_pct) > 0 else 0.0

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        if "pnl" not in tdf.columns:
            tdf["pnl"] = np.nan
        if "exit_date" not in tdf.columns:
            tdf["exit_date"] = pd.NaT
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["return_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["holding_days"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    profit_factor = 0.0
    if not tdf.empty and "pnl" in tdf.columns:
        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0:
            profit_factor = float("inf")

    if not tdf.empty and len(tdf) > 5 and "pnl" in tdf.columns:
        win_rate = (tdf["pnl"] > 0).mean()
        avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if win_rate > 0 else 0
        avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if win_rate < 1 else 0
        if avg_loss > 0 and win_rate > 0 and win_rate < 1:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly = (p * b - q) / b
            kelly = max(0, min(kelly, 0.10))
        else:
            kelly = 0.0
    else:
        kelly = 0.0

    metrics = {
        "Total Return": float(total_return),
        "Annualized Return": float(ann_return),
        "Annualized Volatility": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Max Drawdown": float(mdd),
        "Trades": int(len(tdf)) if not tdf.empty else 0,
        "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty and "pnl" in tdf.columns else 0.0,
        "Profit Factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Information Ratio": float(info_ratio),
        "Ulcer Index": float(ulcer_index),
        "Kelly % (öneri)": float(kelly * 100),
    }
    return eq, tdf, metrics


# =============================
# TARGET BAND / S-R
# =============================

def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs, ls = [], []
    n = len(high)
    for i in range(left, n - right):
        hwin = high.iloc[i - left : i + right + 1]
        lwin = low.iloc[i - left : i + right + 1]
        if high.iloc[i] == hwin.max():
            hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == lwin.min():
            ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls


def analyze_sr_levels(df: pd.DataFrame, lookback: int = 200, tol=0.02) -> List[dict]:
    h = df["High"].tail(lookback).dropna()
    l = df["Low"].tail(lookback).dropna()
    c = df["Close"].tail(lookback).dropna()
    if len(c) < 10:
        return []
    v = df["Volume"].tail(lookback) if "Volume" in df.columns else pd.Series(dtype=float)
    hs, ls = _swing_points(h, l, left=3, right=3)
    raw_levels = [val for _, val in hs] + [val for _, val in ls]
    raw_levels += [float(c.tail(20).max()), float(c.tail(20).min())]
    raw_levels = sorted(list(set([round(float(x), 2) for x in raw_levels if np.isfinite(x)])))
    if not raw_levels:
        return []

    clusters = []
    for rl in raw_levels:
        placed = False
        for cl in clusters:
            if abs(rl - cl["center"]) / cl["center"] <= tol:
                cl["points"].append(rl)
                placed = True
                break
        if not placed:
            clusters.append({"center": rl, "points": [rl]})

    avg_vol_normal = float(v.mean()) if not v.empty else 1.0
    if avg_vol_normal <= 0:
        avg_vol_normal = 1.0

    details = []
    df_lookback = df.tail(lookback)
    for cl in clusters:
        level_px = cl["center"]
        lower_bound = level_px * (1 - tol / 2)
        upper_bound = level_px * (1 + tol / 2)
        touches = df_lookback[(df_lookback["High"] >= lower_bound) & (df_lookback["Low"] <= upper_bound)]
        num_touches = len(touches)
        if num_touches == 0:
            continue
        first_touch_idx = touches.index[0]
        first_idx_num = df_lookback.index.get_loc(first_touch_idx)
        duration_bars = len(df_lookback) - first_idx_num
        vol_at_level = float(touches["Volume"].mean()) if "Volume" in df_lookback.columns and not touches.empty else avg_vol_normal
        vol_diff_pct = (vol_at_level / avg_vol_normal - 1.0) * 100.0
        score_touches = min(num_touches * 10, 40)
        score_vol = min(max(vol_diff_pct / 2.0, 0), 35)
        score_dur = min(duration_bars / 2.0, 25)
        strength_pct = min(score_touches + score_vol + score_dur, 99.0)
        details.append({
            "price": round(level_px, 2),
            "duration_bars": int(duration_bars),
            "vol_at_level": float(vol_at_level),
            "vol_diff_pct": float(vol_diff_pct),
            "strength_pct": float(strength_pct),
            "touches": int(num_touches),
        })
    return sorted(details, key=lambda x: x["price"])


def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px_close = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan
    lv_details = analyze_sr_levels(df)
    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px_close, "bull": None, "bear": None, "levels": lv_details, "r1_dict": None, "s1_dict": None}

    bull1 = px_close + 1.5 * atrv
    bull2 = px_close + 3.0 * atrv
    bear1 = px_close - 1.5 * atrv
    bear2 = px_close - 3.0 * atrv

    above = [x for x in lv_details if x["price"] >= px_close * 1.005]
    below = [x for x in lv_details if x["price"] <= px_close * 0.995]
    valid_above = [x for x in above if x["duration_bars"] >= 10 and x["touches"] >= 2]
    valid_below = [x for x in below if x["duration_bars"] >= 10 and x["touches"] >= 2]
    strong_above = [x for x in valid_above if x["strength_pct"] >= 25]
    strong_below = [x for x in valid_below if x["strength_pct"] >= 35]

    r1_dict = min(strong_above, key=lambda x: x["price"]) if strong_above else (min(valid_above, key=lambda x: x["price"]) if valid_above else None)
    s1_dict = max(strong_below, key=lambda x: x["price"]) if strong_below else (max(valid_below, key=lambda x: x["price"]) if valid_below else None)
    r1 = r1_dict["price"] if r1_dict else None
    s1 = s1_dict["price"] if s1_dict else None

    if r1 is None:
        pivot = (float(last["High"]) + float(last["Low"]) + px_close) / 3.0
        synth_r1 = (2 * pivot) - float(last["Low"])
        if synth_r1 > px_close:
            r1 = synth_r1
            r1_dict = {"price": synth_r1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    if s1 is None:
        pivot = (float(last["High"]) + float(last["Low"]) + px_close) / 3.0
        synth_s1 = (2 * pivot) - float(last["High"])
        if synth_s1 < px_close and synth_s1 > 0:
            s1 = synth_s1
            s1_dict = {"price": synth_s1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    return {
        "base": px_close,
        "bull": (bull1, bull2, r1),
        "bear": (bear1, bear2, s1),
        "levels": lv_details,
        "r1_dict": r1_dict,
        "s1_dict": s1_dict,
    }


def compute_rr_info(df: pd.DataFrame, tp: dict, latest: pd.Series) -> dict:
    out = {"rr": np.nan, "entry": np.nan, "stop": np.nan, "target": np.nan, "target_type": ""}
    try:
        base = float(tp["base"])
        atrv = float(latest["ATR"])
        entry = base
        stop = base - 1.5 * atrv
        target = None
        target_type = ""

        if tp.get("bull"):
            bull1, bull2, r1 = tp["bull"]
            if r1 is not None and np.isfinite(r1) and r1 > entry:
                target = min(float(bull1), float(r1)) if float(r1) < float(bull2) else float(r1)
                target_type = "Yakın Direnç"
            else:
                target = float(bull1)
                target_type = "ATR Hedef"

        if target is not None and entry > stop:
            rr = (target - entry) / (entry - stop)
            out = {"rr": rr, "entry": entry, "stop": stop, "target": target, "target_type": target_type}
    except Exception:
        pass
    return out


# =============================
# NEWS / GEMINI
# =============================

def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        if v is None:
            return default
        return str(v).strip()
    except Exception:
        return default


def _http_post_json(url: str, payload: dict, headers: dict = None, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"error": {"message": f"Non-JSON response (status={r.status_code})", "text": r.text[:500]}}
    if r.status_code >= 400 and "error" not in data:
        data["error"] = {"message": f"HTTP {r.status_code}", "text": str(data)[:500]}
    return data


def _extract_gemini_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return str(resp)
    if resp.get("error"):
        return f"Gemini API error: {resp['error'].get('message', '')}"
    cands = resp.get("candidates") or []
    if not cands:
        return "Gemini: boş cevap döndü (candidates yok)."
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        return "Gemini: boş cevap döndü (parts yok)."
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    return "\n".join(texts).strip() if texts else "Gemini: metin üretmedi."


def gemini_generate_text(
    *,
    prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    image_bytes: Optional[bytes] = None,
) -> str:
    api_key = _get_secret("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY bulunamadı. Streamlit secrets içine GEMINI_API_KEY ekleyin."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}
    parts = [{"text": prompt}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64_img}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)},
    }
    resp = _http_post_json(url, payload, headers=headers, timeout=90)
    return _extract_gemini_text(resp)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_news_sentiment(
    symbol: str,
    gemini_model: str = "gemini-1.5-flash",
    gemini_temp: float = 0.2,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    base_coin = symbol.split("/")[0].upper()
    query = f"{base_coin} OR Bitcoin OR BTC crypto"
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {"error": f"Haberler çekilemedi (HTTP {resp.status_code})", "sentiment": None, "summary": "", "news_items": []}
        root = ET.fromstring(resp.content)
        news_items = []
        for item in root.findall(".//item")[:10]:
            title_node = item.find("title")
            link_node = item.find("link")
            if title_node is not None and title_node.text:
                t = title_node.text
                l = link_node.text if (link_node is not None and link_node.text) else ""
                news_items.append({"title": t, "link": l})
        if not news_items:
            return {"error": "Haber bulunamadı", "sentiment": None, "summary": "", "news_items": []}

        prompt_titles = [item["title"] for item in news_items]
        prompt = f"""Aşağıdaki haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
Pozitif: [sayı]
Negatif: [sayı]
Nötr: [sayı]
- Bileşik skor: (pozitif - negatif) / toplam
- Kısa özet: 2-3 cümle

Haber Başlıkları:
{chr(10).join([f'- {t}' for t in prompt_titles])}
"""
        response = gemini_generate_text(
            prompt=prompt,
            model=gemini_model,
            temperature=gemini_temp,
            max_output_tokens=max_tokens,
            image_bytes=None,
        )
        pos_match = re.search(r"Pozitif:?\s*(\d+)", response, re.IGNORECASE)
        neg_match = re.search(r"Negatif:?\s*(\d+)", response, re.IGNORECASE)
        neu_match = re.search(r"Nötr:?\s*(\d+)", response, re.IGNORECASE)
        pos = int(pos_match.group(1)) if pos_match else 0
        neg = int(neg_match.group(1)) if neg_match else 0
        neu = int(neu_match.group(1)) if neu_match else 0
        total = pos + neg + neu
        compound = (pos - neg) / total if total > 0 else 0
        return {
            "error": None,
            "sentiment": compound,
            "summary": response,
            "pos": pos / total if total > 0 else 0,
            "neg": neg / total if total > 0 else 0,
            "neu": neu / total if total > 0 else 0,
            "news_items": news_items[:5],
        }
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": "", "news_items": []}


# =============================
# PRICE ACTION PACK
# =============================

def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if use.empty or len(use) < 10:
        return {"note": "insufficient_bars", "last_n": int(len(use))}

    o = use["Open"].astype(float)
    h = use["High"].astype(float)
    l = use["Low"].astype(float)
    c = use["Close"].astype(float)
    swing_highs, swing_lows = _swing_points(h, l, left=2, right=2)
    q20 = float(np.quantile(c.values, 0.20))
    q50 = float(np.quantile(c.values, 0.50))
    q80 = float(np.quantile(c.values, 0.80))
    recent_highs = [v for _, v in swing_highs[-5:]] if swing_highs else []
    recent_lows = [v for _, v in swing_lows[-5:]] if swing_lows else []
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())
    last_close = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
    last_high = float(h.iloc[-1])
    last_low = float(l.iloc[-1])
    bull_break = (last_close > res) and (prev_close <= res)
    bear_break = (last_close < sup) and (prev_close >= sup)

    vol_ok = None
    if "Volume" in use.columns:
        vol = use["Volume"].astype(float)
        vol_sma = float(vol.rolling(10).mean().iloc[-1]) if len(vol) >= 10 else float(vol.mean())
        vol_ok = float(vol.iloc[-1]) > vol_sma if np.isfinite(vol_sma) else None

    impulse_up = (c.diff().tail(3) > 0).all() and (last_close >= q80)
    impulse_dn = (c.diff().tail(3) < 0).all() and (last_close <= q20)
    ob = None
    if impulse_up:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] < o.iloc[i]:
                ob = {"type": "bullish_order_block_proxy", "index": str(use.index[i]), "open": float(o.iloc[i]), "high": float(h.iloc[i]), "low": float(l.iloc[i]), "close": float(c.iloc[i])}
                break
    elif impulse_dn:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] > o.iloc[i]:
                ob = {"type": "bearish_order_block_proxy", "index": str(use.index[i]), "open": float(o.iloc[i]), "high": float(h.iloc[i]), "low": float(l.iloc[i]), "close": float(c.iloc[i])}
                break

    return {
        "last_n": int(len(use)),
        "q20": q20,
        "q50": q50,
        "q80": q80,
        "support": sup,
        "resistance": res,
        "bull_breakout": bool(bull_break),
        "bear_breakout": bool(bear_break),
        "vol_confirm": (None if vol_ok is None else bool(vol_ok)),
        "last_bar": {"t": str(use.index[-1]), "open": float(o.iloc[-1]), "high": last_high, "low": last_low, "close": last_close},
        "swing_highs": [{"t": str(t), "p": float(p)} for t, p in swing_highs[-6:]],
        "swing_lows": [{"t": str(t), "p": float(p)} for t, p in swing_lows[-6:]],
        "order_block_proxy": ob,
    }


def df_snapshot_for_llm(df: pd.DataFrame, n: int = 25) -> dict:
    use_cols = [
        "Open", "High", "Low", "Close", "Volume", "EMA50", "EMA200", "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_mid", "BB_upper", "BB_lower", "ATR", "ATR_PCT", "VOL_SMA", "VOL_RATIO", "BB_WIDTH", "SCORE", "ENTRY", "EXIT",
        "RSI_OVERBOUGHT", "BB_OVERBOUGHT", "BB_OVERSOLD", "VOLUME_SPIKE", "PRICE_EXTREME", "STOCH_OVERBOUGHT", "WEAK_UPTREND",
        "KANGAROO_BULL", "KANGAROO_BEAR"
    ]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    summary = {}
    if not df.empty:
        summary["rsi_last"] = float(df["RSI"].iloc[-1]) if "RSI" in df else None
        summary["rsi_5d_avg"] = float(df["RSI"].tail(5).mean()) if "RSI" in df else None
        if "EMA50" in df and "EMA200" in df:
            summary["trend"] = "up" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "down"
    return {"cols": cols, "n": int(len(tail)), "last_index": str(tail.index[-1]) if len(tail) else None, "rows": tail.to_dict(orient="records"), "summary": summary}


# =============================
# PRESETS / REPORT
# =============================
PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 2.0, "time_stop_bars": 15, "take_profit_mult": 2.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 1.5, "time_stop_bars": 10, "take_profit_mult": 2.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 1.2, "time_stop_bars": 7, "take_profit_mult": 1.5},
}


def build_html_report(
    title: str,
    meta: dict,
    checkpoints: dict,
    metrics: dict,
    tp: dict,
    rr_info: dict,
    figs: Dict[str, go.Figure],
    gemini_insight: Optional[str] = None,
    pa_pack: Optional[dict] = None,
    sentiment_summary: Optional[str] = None,
    sentiment_items: Optional[List[dict]] = None,
    overbought_result: Optional[dict] = None,
) -> bytes:
    img_blocks = []
    for name, fig in figs.items():
        try:
            img = fig.to_image(format="png", width=1400, height=850, scale=1)
            b64 = base64.b64encode(img).decode("utf-8")
            img_blocks.append(f"<h3>{name}</h3><img src='data:image/png;base64,{b64}' style='width:100%;max-width:1200px;border:1px solid #ddd;border-radius:8px;' />")
        except Exception:
            pass

    cp_html = "".join([f"<tr><td>{k}</td><td>{'✅' if v else '❌'}</td></tr>" for k, v in checkpoints.items()])
    mt_html = "".join([f"<tr><td>{k}</td><td>{fmt_num(v, 4) if isinstance(v, (int, float)) else v}</td></tr>" for k, v in metrics.items()])
    sentiment_list = "".join([f"<li><a href='{item.get('link','')}'>{item.get('title','')}</a></li>" for item in (sentiment_items or [])])

    html = f"""
    <html>
    <head>
      <meta charset='utf-8' />
      <title>{title}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 28px; color: #111; }}
        h1, h2, h3 {{ color: #111; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
        td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
        .small {{ color: #666; font-size: 12px; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f7f7f7; padding: 12px; border-radius: 8px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <p class='small'>Bu rapor eğitim amaçlıdır. Otomatik emir üretmez.</p>
      <div class='card'><h2>Meta</h2><pre>{json.dumps(meta, ensure_ascii=False, indent=2, default=str)}</pre></div>
      <div class='card'><h2>Kontrol Noktaları</h2><table>{cp_html}</table></div>
      <div class='card'><h2>Backtest Metrikleri</h2><table>{mt_html}</table></div>
      <div class='card'><h2>Hedef Bandı</h2><pre>{json.dumps({'target_band': tp, 'rr_info': rr_info}, ensure_ascii=False, indent=2, default=str)}</pre></div>
      <div class='card'><h2>Aşırı Alım / Spekülasyon</h2><pre>{json.dumps(overbought_result or {}, ensure_ascii=False, indent=2, default=str)}</pre></div>
      <div class='card'><h2>Price Action Pack</h2><pre>{json.dumps(pa_pack or {}, ensure_ascii=False, indent=2, default=str)}</pre></div>
      <div class='card'><h2>Haber Duygu Analizi</h2><pre>{sentiment_summary or ''}</pre><ul>{sentiment_list}</ul></div>
      <div class='card'><h2>Gemini Insight</h2><pre>{gemini_insight or ''}</pre></div>
      {''.join(img_blocks)}
    </body>
    </html>
    """
    return html.encode("utf-8")


def generate_pdf_report(title: str, subtitle: str, meta: dict, checkpoints: dict, ta_summary: dict, target_band: dict, rr_info: dict, backtest_metrics: dict) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None
    try:
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        left = 1.5 * cm
        y = height - 2 * cm
        c.setFont("Helvetica-Bold", 16)
        c.drawString(left, y, title)
        y -= 0.8 * cm
        c.setFont("Helvetica", 10)
        c.drawString(left, y, subtitle)
        y -= 0.8 * cm
        for block_title, block in [
            ("Meta", meta),
            ("TA Summary", ta_summary),
            ("Checkpoints", checkpoints),
            ("Target Band", target_band),
            ("RR Info", rr_info),
            ("Backtest", backtest_metrics),
        ]:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(left, y, block_title)
            y -= 0.4 * cm
            c.setFont("Helvetica", 8)
            for line in json.dumps(block, ensure_ascii=False, indent=2, default=str).splitlines():
                c.drawString(left, y, line[:120])
                y -= 0.35 * cm
                if y < 2 * cm:
                    c.showPage()
                    y = height - 2 * cm
            y -= 0.3 * cm
        c.save()
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# =============================
# CHARTS
# =============================

def build_main_charts(df: pd.DataFrame) -> Dict[str, go.Figure]:
    latest = df.iloc[-1]
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Fiyat"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="orange", width=1.5)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(color="red", width=1.5)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Üst", line=dict(color="gray", dash="dash")))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Alt", line=dict(color="gray", dash="dash"), fill=None))

    entries = df[df["ENTRY"] == 1]
    exits = df[df["EXIT"] == 1]
    if not entries.empty:
        fig_price.add_trace(go.Scatter(x=entries.index, y=entries["Low"] * 0.995, mode="markers", name="AL", marker=dict(symbol="triangle-up", size=11, color="green")))
    if not exits.empty:
        fig_price.add_trace(go.Scatter(x=exits.index, y=exits["High"] * 1.005, mode="markers", name="SAT", marker=dict(symbol="triangle-down", size=11, color="red")))

    bull_tail = df[df["KANGAROO_BULL"] == 1]
    bear_tail = df[df["KANGAROO_BEAR"] == 1]
    if not bull_tail.empty:
        fig_price.add_trace(go.Scatter(x=bull_tail.index, y=bull_tail["Low"] * 0.99, mode="markers", name="Kanguru Boğa", marker=dict(symbol="star", size=9, color="limegreen")))
    if not bear_tail.empty:
        fig_price.add_trace(go.Scatter(x=bear_tail.index, y=bear_tail["High"] * 1.01, mode="markers", name="Kanguru Ayı", marker=dict(symbol="star", size=9, color="crimson")))

    fig_price.update_layout(title="Fiyat + EMA + Bollinger + Sinyaller", height=640, xaxis_rangeslider_visible=False)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="blue")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="RSI", height=260)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
    fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
    fig_macd.update_layout(title="MACD", height=260)

    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
    fig_atr.update_layout(title="ATR %", height=260)
    return {"price": fig_price, "rsi": fig_rsi, "macd": fig_macd, "atr": fig_atr}


# =============================
# SESSION STATE
# =============================
if "ta_ran" not in st.session_state:
    st.session_state.ta_ran = False
if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = ""
if "pa_pack" not in st.session_state:
    st.session_state.pa_pack = {}
if "sentiment_summary" not in st.session_state:
    st.session_state.sentiment_summary = ""
if "sentiment_items" not in st.session_state:
    st.session_state.sentiment_items = []
if "run_triple_screen" not in st.session_state:
    st.session_state.run_triple_screen = False


# =============================
# UI TITLE
# =============================
st.title("₿ Bitcoin Trading Uygulaması + 🤖 AI Analiz")
st.caption("Hisse uygulamasındaki teknik analiz mantığını kriptoya uyarlayan sürüm. Otomatik emir göndermez.")


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("Piyasa")
    st.selectbox("Market", ["CRYPTO"], index=0, disabled=True)

    symbol_list = get_top_symbols(150)
    default_idx = symbol_list.index(DEFAULT_SYMBOL) if DEFAULT_SYMBOL in symbol_list else 0
    symbol = st.selectbox("Kripto Sembol", symbol_list, index=default_idx)

    st.header("1) Teknik Ayarlar")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1, help="Defansif: düşük risk, Agresif: yüksek risk.")
    interval = st.selectbox("Interval", ["1d", "1wk", "4h", "1h"], index=0)
    period = st.selectbox("Periyot", ["45d", "3mo", "6mo", "1y", "2y"], index=3)

    use_custom_end_date = st.checkbox("Geçmiş Bir Tarihe Göre Analiz Yap", value=False)
    if use_custom_end_date:
        default_end = datetime.date.today() - datetime.timedelta(days=2)
        custom_end_date = st.date_input("Bitiş Tarihi", value=default_end)
    else:
        custom_end_date = None

    st.divider()
    st.subheader("Teknik Parametreler")
    ema_fast = st.number_input("EMA Fast", min_value=5, max_value=100, value=50, step=1)
    ema_slow = st.number_input("EMA Slow", min_value=50, max_value=400, value=200, step=1)
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1)
    bb_period = st.number_input("Bollinger Period", min_value=10, max_value=50, value=20, step=1)
    bb_std = st.number_input("Bollinger Std", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    atr_period = st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1)
    vol_sma = st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1)

    st.divider()
    st.subheader("Backtest")
    initial_capital = st.number_input("Başlangıç Sermayesi", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=100.0)
    risk_per_trade = st.slider("İşlem Başına Risk", 0.001, 0.050, 0.010, 0.001)
    commission_bps = st.number_input("Komisyon (bps)", min_value=0.0, max_value=50.0, value=8.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    risk_free_annual = st.number_input("Risksiz Faiz (yıllık)", min_value=0.0, max_value=0.30, value=0.03, step=0.01)

    st.divider()
    st.header("2) AI Ayarları (Gemini)")
    use_gemini = st.checkbox("Gemini AI aktif", value=False)
    gemini_model = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0, disabled=not use_gemini)
    gemini_temp = st.slider("Gemini Temperature", 0.0, 1.0, 0.2, 0.05, disabled=not use_gemini)
    gemini_max_tokens = st.slider("Gemini Max Tokens", 256, 4096, 1536, 128, disabled=not use_gemini)

    st.divider()
    st.header("3) Haber Duygu Analizi")
    use_sentiment = st.checkbox("Haber duygu analizini aktifleştir", value=True)

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn:
        st.session_state.ta_ran = True

cfg = {
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "rsi_period": rsi_period,
    "bb_period": bb_period,
    "bb_std": bb_std,
    "atr_period": atr_period,
    "vol_sma": vol_sma,
    "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade,
    "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])

if not st.session_state.ta_ran:
    st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirin.")
    st.stop()


# =============================
# MAIN COMPUTE
# =============================
with st.spinner("Kripto verileri çekiliyor ve analiz ediliyor..."):
    df_raw = load_data_cached(symbol, period, interval, end_date=custom_end_date)

if df_raw.empty or len(df_raw) < 60:
    st.error("Yeterli veri çekilemedi. Farklı periyot veya zaman dilimi deneyin.")
    st.stop()

# Benchmark BTC itself for alt pairs, or BTC regime for BTC
benchmark_symbol = DEFAULT_SYMBOL
benchmark_df = load_data_cached(benchmark_symbol, period, interval, end_date=custom_end_date)
benchmark_returns = None
if not benchmark_df.empty:
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

df = build_features(df_raw, cfg)
market_filter_series = get_crypto_regime_series(DEFAULT_SYMBOL)
higher_tf_filter_series = get_higher_tf_trend_series(symbol, "1wk", ema_period=200)
df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_series, higher_tf_filter_series)

year_map = {"1h": 365 * 24, "4h": 365 * 6, "1d": 365, "1wk": 52}
periods_per_year = year_map.get(interval, 365)

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual, benchmark_returns=benchmark_returns, periods_per_year=periods_per_year)
latest = df.iloc[-1]
live_info = get_live_price(symbol)
live_price = safe_float(live_info.get("last_price"))
rec = recommendation_from_latest(latest)
overbought_result = detect_speculation(df)
tp = target_price_band(df)
rr_info = compute_rr_info(df, tp, latest)
pa_pack = price_action_pack(df)
st.session_state.pa_pack = pa_pack
figs = build_main_charts(df)
fig_price = figs["price"]
fig_rsi = figs["rsi"]
fig_macd = figs["macd"]
fig_atr = figs["atr"]
figs_for_report = {"Price Chart": fig_price, "RSI": fig_rsi, "MACD": fig_macd, "ATR": fig_atr}

sentiment_info = {"error": None, "sentiment": None, "summary": "", "news_items": []}
if use_sentiment and use_gemini:
    with st.spinner("Google News'ten haberler çekiliyor ve Gemini ile analiz ediliyor..."):
        sentiment_info = get_news_sentiment(symbol, gemini_model=gemini_model, gemini_temp=gemini_temp, max_tokens=gemini_max_tokens)
        st.session_state.sentiment_summary = sentiment_info.get("summary", "")
        st.session_state.sentiment_items = sentiment_info.get("news_items", [])
elif use_sentiment and not use_gemini:
    sentiment_info = {"error": "Haber duygu analizi için Gemini'nin açık olması gerekir.", "sentiment": None, "summary": "", "news_items": []}


# =============================
# TABS
# =============================
tab_dash, tab_export, tab_heatmap, tab_triple = st.tabs(["📊 Dashboard", "📄 Rapor (PDF/HTML)", "🔥 Piyasa Heatmap", "📺 3 Ekranlı Sistem"])

with tab_dash:
    st.subheader("📊 Aşırı Alım / Spekülasyon Göstergeleri")
    col_ob1, col_ob2, col_ob3, col_ob4, col_ob5, col_ob6 = st.columns(6)
    col_ob1.metric("Aşırı Alım Skoru", f"{overbought_result['overbought_score']}/100")
    col_ob2.metric("Aşırı Satım Skoru", f"{overbought_result['oversold_score']}/100")
    col_ob3.metric("Spekülasyon Skoru", f"{overbought_result['speculation_score']}/100")
    col_ob4.metric("Genel Karar", overbought_result["verdict"])
    col_ob5.metric("24s Quote Vol", fmt_num(live_info.get("quote_volume"), 0))
    col_ob6.metric("Canlı Değişim", fmt_pct(live_info.get("change_pct")))

    with st.expander("Detaylı Aşırı Alım/Spekülasyon Analizi"):
        for _, v in overbought_result["details"].items():
            st.write(f"• {v}")

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Market", "CRYPTO")
    c2.metric("Sembol", symbol)
    c3.metric("Close", fmt_num(float(latest["Close"]), 2))
    c4.metric("Live/Last", fmt_num(float(live_price), 2) if np.isfinite(live_price) else "N/A")
    c5.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c6.metric("Sinyal", rec)
    c7.metric("Piyasa Filtresi", "BULL ✅" if checkpoints.get("Market Filter OK", True) else "BEAR ❌")
    c8.metric("Haftalık Trend", "BULL ✅" if checkpoints.get("Higher TF Filter OK", True) else "BEAR ❌")

    st.subheader("🕯️ Fiyat Aksiyonu (Price Action) Mum Formasyonları - Son Bar")
    is_bull_tail = latest.get("KANGAROO_BULL", 0) == 1
    is_bear_tail = latest.get("KANGAROO_BEAR", 0) == 1
    tail_val = "BOĞA 🦘" if is_bull_tail else ("AYI 🦘" if is_bear_tail else "YOK")
    tail_delta = "AL Yönlü" if is_bull_tail else ("SAT Yönlü" if is_bear_tail else None)

    pa_c1, pa_c2, pa_c3, pa_c4, pa_c5, pa_c6 = st.columns(6)
    pa_c1.metric("1. Kanguru", tail_val, delta=tail_delta)
    pa_c2.metric("2. Engulfing", "Boğa 🟢" if latest.get("PATTERN_ENGULFING_BULL") else ("Ayı 🔴" if latest.get("PATTERN_ENGULFING_BEAR") else "Yok"))
    pa_c3.metric("3. Hammer / Star", "Çekiç 🟢" if latest.get("PATTERN_HAMMER") else ("Kayan Yıldız 🔴" if latest.get("PATTERN_SHOOTING_STAR") else "Yok"))
    pa_c4.metric("4. Doji", "Uzun Bacak ⚪" if latest.get("PATTERN_LL_DOJI") else ("Doji ⚪" if latest.get("PATTERN_DOJI") else "Yok"))
    pa_c5.metric("5. Marubozu", "Boğa 🟢" if latest.get("PATTERN_MARUBOZU_BULL") else ("Ayı 🔴" if latest.get("PATTERN_MARUBOZU_BEAR") else "Yok"))
    pa_c6.metric("6. Harami", "Boğa 🟢" if latest.get("PATTERN_HARAMI_BULL") else ("Ayı 🔴" if latest.get("PATTERN_HARAMI_BEAR") else "Yok"))

    pa2_c1, pa2_c2, pa2_c3, pa2_c4, pa2_c5, pa2_c6 = st.columns(6)
    pa2_c1.metric("7. Tweezer", "Dip 🟢" if latest.get("PATTERN_TWEEZER_BOTTOM") else ("Tepe 🔴" if latest.get("PATTERN_TWEEZER_TOP") else "Yok"))
    pa2_c2.metric("8. M./E. Star", "Sabah 🟢" if latest.get("PATTERN_MORNING_STAR") else ("Akşam 🔴" if latest.get("PATTERN_EVENING_STAR") else "Yok"))
    pa2_c3.metric("9. Piercing / Dark", "Delen 🟢" if latest.get("PATTERN_PIERCING") else ("Kara Bulut 🔴" if latest.get("PATTERN_DARK_CLOUD") else "Yok"))
    pa2_c4.metric("10. Inv. H / Hang", "Ters Çekiç 🟢" if latest.get("PATTERN_INV_HAMMER") else ("Asılı Adam 🔴" if latest.get("PATTERN_HANGING_MAN") else "Yok"))
    pa2_c5.metric("11. Filtre Durumu", "Aktif ✅")
    pa2_c6.write("")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    cp_items = list(checkpoints.items())
    for i, (k, v) in enumerate(cp_items):
        with cp_cols[i % 3]:
            st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp["base"])
    rr_str = fmt_rr(rr_info.get("rr"))
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", fmt_num(base_px, 2))

    s1 = None
    r1 = None
    if tp.get("bull"):
        bull1, bull2, r1 = tp["bull"]
        bcol2.metric("Bull Band", f"{fmt_num(bull1, 2)} → {fmt_num(bull2, 2)}")
        if r1 is not None and np.isfinite(r1):
            r1_info = tp.get("r1_dict") or {}
            if r1_info.get("is_synthetic", False):
                bcol2.caption(f"Yakın direnç: {fmt_num(r1,2)} ({pct_dist(r1, base_px):+.2f}%)\n\nSentetik pivot direnci hesaplandı.")
            else:
                dur = r1_info.get("duration_bars", 0)
                vol_pct = r1_info.get("vol_diff_pct", 0)
                str_pct = r1_info.get("strength_pct", 0)
                bcol2.caption(f"Yakın direnç: {fmt_num(r1,2)} ({pct_dist(r1, base_px):+.2f}%)\n\nGüç: %{str_pct:.0f} | Uzunluk: {dur} Bar | Hacim: %{vol_pct:+.1f}")
    else:
        bcol2.metric("Bull Band", "N/A")

    if tp.get("bear"):
        bear1, bear2, s1 = tp["bear"]
        target_info = f" | Hedef: {rr_info.get('target_type','')}" if rr_info.get("target_type") else ""
        bcol3.metric("Bear Band", f"{fmt_num(bear1,2)} → {fmt_num(bear2,2)} | RR {rr_str}{target_info}")
        if s1 is not None and np.isfinite(s1):
            s1_info = tp.get("s1_dict") or {}
            if s1_info.get("is_synthetic", False):
                bcol3.caption(f"Yakın destek: {fmt_num(s1,2)} ({pct_dist(s1, base_px):+.2f}%)\n\nSentetik pivot desteği hesaplandı.")
            else:
                dur = s1_info.get("duration_bars", 0)
                vol_pct = s1_info.get("vol_diff_pct", 0)
                str_pct = s1_info.get("strength_pct", 0)
                bcol3.caption(f"Yakın destek: {fmt_num(s1,2)} ({pct_dist(s1, base_px):+.2f}%)\n\nGüç: %{str_pct:.0f} | Uzunluk: {dur} Bar | Hacim: %{vol_pct:+.1f}")
    else:
        bcol3.metric("Bear Band", f"N/A | RR {rr_str}")

    def render_levels_marked(levels: List[dict], base: float, s1, r1):
        lines = []
        for lv_dict in (levels or []):
            lv = float(lv_dict["price"])
            dur = lv_dict["duration_bars"]
            vol_pct = lv_dict["vol_diff_pct"]
            str_pct = lv_dict["strength_pct"]
            tag = ""
            if s1 is not None and np.isfinite(s1) and abs(lv - float(s1)) < 1e-9:
                tag = " 🟩 Yakın Destek"
            if r1 is not None and np.isfinite(r1) and abs(lv - float(r1)) < 1e-9:
                tag = " 🟥 Yakın Direnç"
            dist = pct_dist(lv, base)
            dist_txt = f"{dist:+.2f}%" if dist is not None else ""
            lines.append(f"- **{fmt_num(lv,2)}** ({dist_txt}) | Güç: %{str_pct:.0f} | Uzunluk: {dur} Bar | Hacim: %{vol_pct:+.1f}{tag}")
        return "\n".join(lines) if lines else "_Seviye yok_"

    with st.expander("Seviye listesi (yaklaşık) — işaretli + fiyata uzaklık %"):
        st.markdown(render_levels_marked(tp.get("levels", []), base_px, s1, r1))

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    with colA:
        st.plotly_chart(fig_rsi, use_container_width=True)
    with colB:
        st.plotly_chart(fig_macd, use_container_width=True)
    with colC:
        st.plotly_chart(fig_atr, use_container_width=True)

    st.subheader("📈 Backtest Özeti")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return", fmt_pct(metrics["Total Return"]))
    m2.metric("Sharpe", fmt_num(metrics["Sharpe"], 2))
    m3.metric("Max DD", fmt_pct(metrics["Max Drawdown"]))
    m4.metric("Trades", str(metrics["Trades"]))
    m5.metric("Win Rate", fmt_pct(metrics["Win Rate"]))
    m6.metric("Kelly %", fmt_num(metrics["Kelly % (öneri)"], 2))

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity Curve"))
    fig_eq.update_layout(title="Backtest Equity Curve", height=320)
    st.plotly_chart(fig_eq, use_container_width=True)

    with st.expander("Trade listesi"):
        st.dataframe(tdf.tail(50), use_container_width=True)

    st.subheader("📰 Haber Duygu Analizi (Google News + Gemini)")
    if sentiment_info.get("error"):
        st.warning(sentiment_info["error"])
    else:
        s1c, s2c, s3c, s4c = st.columns(4)
        s1c.metric("Bileşik Skor", fmt_num(sentiment_info.get("sentiment"), 2))
        s2c.metric("Pozitif", fmt_pct(sentiment_info.get("pos")))
        s3c.metric("Negatif", fmt_pct(sentiment_info.get("neg")))
        s4c.metric("Nötr", fmt_pct(sentiment_info.get("neu")))
        st.markdown(sentiment_info.get("summary", ""))
        for item in sentiment_info.get("news_items", []):
            st.markdown(f"- [{item['title']}]({item['link']})")

    st.subheader("🤖 Gemini Multimodal AI — Grafik + Price Action + Spekülasyon Analizi")
    if not use_gemini:
        st.info("Gemini kapalı (sol menüden açabilirsiniz).")
    else:
        user_msg = st.text_area(
            "Gemini'ye sor/talimat ver:",
            value="BTC için teknik görünümü, price action formasyonlarını, aşırı alım-satım riskini ve olası giriş/çıkış bölgelerini yorumla.",
            height=120,
        )
        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            if st.button("🖼️ Gemini'ye Sor (Görsel + Tüm Veriler)", use_container_width=True):
                snapshot = df_snapshot_for_llm(df)
                prompt = f"""Sen deneyimli bir kripto teknik analistisin. Aşağıda {symbol} için hesaplanan veriler var.

Teknik Özet:
{json.dumps({
    'symbol': symbol,
    'interval': interval,
    'period': period,
    'latest': {
        'close': safe_float(latest['Close']),
        'rsi': safe_float(latest['RSI']),
        'score': safe_float(latest['SCORE']),
        'entry': int(latest['ENTRY']),
        'exit': int(latest['EXIT']),
        'atr_pct': safe_float(latest['ATR_PCT']),
        'ema50': safe_float(latest['EMA50']),
        'ema200': safe_float(latest['EMA200'])
    },
    'checkpoints': checkpoints,
    'target_band': tp,
    'rr_info': rr_info,
    'price_action_pack': pa_pack,
    'sentiment': sentiment_info,
    'overbought_analysis': overbought_result,
    'snapshot': snapshot,
}, ensure_ascii=False, default=str)}

Kullanıcının Sorusu: {user_msg}

Analizin sonunda aşağıdaki gibi tablo ver:
| Hedef | Fiyat |
|-------|-------|
| Alış Fiyatı (önerilen giriş) | ... |
| Hedef Satış Fiyatı (ilk hedef) | ... |
| Stop Loss (ATR bazlı) | ... |
"""
                image_bytes = _plotly_fig_to_png_bytes(fig_price)
                text = gemini_generate_text(
                    prompt=prompt,
                    model=gemini_model,
                    temperature=gemini_temp,
                    max_output_tokens=gemini_max_tokens,
                    image_bytes=image_bytes,
                )
                st.session_state.gemini_text = text
        with col_g2:
            if st.button("Temizle", use_container_width=True):
                st.session_state.gemini_text = ""
        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)

    st.subheader("🌐 Kripto Tarayıcı (Top USDT Pairs)")
    scan_col1, scan_col2 = st.columns(2)
    scan_count = scan_col1.slider("Taranacak Coin Sayısı", 10, 80, 30)
    scan_tf = scan_col2.selectbox("Tarama Periyodu", ["1h", "4h", "1d"], index=2)
    if st.button("🚀 Taramayı Başlat", key="scan_market"):
        symbols_to_scan = get_top_symbols(scan_count)
        rows = []
        prog = st.progress(0)
        status = st.empty()

        def scan_one(sym: str):
            dfr = load_data_cached(sym, "3mo", scan_tf)
            if dfr.empty or len(dfr) < 60:
                return None
            dfr = build_features(dfr, cfg)
            mfs = get_crypto_regime_series(DEFAULT_SYMBOL)
            htf = get_higher_tf_trend_series(sym, "1wk", 200)
            dfr, _cp = signal_with_checkpoints(dfr, cfg, mfs, htf)
            last_row = dfr.iloc[-1]
            return {
                "Sembol": sym,
                "Fiyat": last_row["Close"],
                "RSI": round(safe_float(last_row["RSI"]), 2),
                "Skor": round(safe_float(last_row["SCORE"]), 0),
                "Sinyal": recommendation_from_latest(last_row),
                "Trend": "YUKARI 🟢" if last_row["Close"] > last_row["EMA50"] else "AŞAĞI 🔴",
                "Kanguru": "AL ✅" if last_row["KANGAROO_BULL"] else ("SAT ❌" if last_row["KANGAROO_BEAR"] else "-"),
            }

        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(scan_one, sym): sym for sym in symbols_to_scan}
            for i, future in enumerate(as_completed(futures)):
                sym = futures[future]
                try:
                    res = future.result()
                    if res:
                        rows.append(res)
                except Exception:
                    pass
                prog.progress((i + 1) / len(futures))
                status.text(f"İşleniyor: {sym}")
        status.empty()
        if rows:
            scan_df = pd.DataFrame(rows).sort_values(["Skor", "RSI"], ascending=[False, False])
            st.dataframe(scan_df, use_container_width=True)
        else:
            st.warning("Tarama sonucu alınamadı.")

with tab_heatmap:
    st.header("🔥 Kripto Heatmap")
    st.write("Top USDT paritelerinde günlük, haftalık ve aylık performans görselleştirmesi.")
    if st.button("Heatmap Verilerini Getir ve Oluştur (1D, 1W, 1M)", type="primary"):
        with st.spinner("Toplu veri çekiliyor ve hesaplanıyor..."):
            hm_tickers = get_top_symbols(60)
            hm_data = []
            for sym in hm_tickers:
                try:
                    df_hm = load_data_cached(sym, "2y", "1d")
                    if len(df_hm) < 25:
                        continue
                    c_last = float(df_hm["Close"].iloc[-1])
                    c_prev_1d = float(df_hm["Close"].iloc[-2])
                    c_prev_1wk = float(df_hm["Close"].iloc[-6]) if len(df_hm) >= 6 else float(df_hm["Close"].iloc[0])
                    c_prev_1mo = float(df_hm["Close"].iloc[-21]) if len(df_hm) >= 21 else float(df_hm["Close"].iloc[0])
                    ret_1d = (c_last / c_prev_1d - 1) * 100
                    ret_1wk = (c_last / c_prev_1wk - 1) * 100
                    ret_1mo = (c_last / c_prev_1mo - 1) * 100
                    base = sym.split("/")[0]
                    sector = CATEGORY_MAP.get(base, "Altcoin")
                    hm_data.append({"Ticker": sym, "Sector": sector, "1 Günlük %": ret_1d, "1 Haftalık %": ret_1wk, "1 Aylık %": ret_1mo})
                except Exception:
                    pass
            df_hm_all = pd.DataFrame(hm_data)

        if not df_hm_all.empty:
            df_hm_all["Abs_1D"] = df_hm_all["1 Günlük %"].abs()
            st.subheader("GÜNLÜK Performans")
            fig_hm_1d = px.treemap(df_hm_all, path=[px.Constant("Kripto Pazar"), "Sector", "Ticker"], values="Abs_1D", color="1 Günlük %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, custom_data=["1 Günlük %", "1 Haftalık %", "1 Aylık %"])
            fig_hm_1d.update_traces(hovertemplate="<b>%{label}</b><br>1 Günlük: %{customdata[0]:.2f}%<br>1 Haftalık: %{customdata[1]:.2f}%<br>1 Aylık: %{customdata[2]:.2f}%")
            st.plotly_chart(fig_hm_1d, use_container_width=True)

            st.subheader("HAFTALIK Performans")
            df_hm_all["Abs_1W"] = df_hm_all["1 Haftalık %"].abs()
            fig_hm_1w = px.treemap(df_hm_all, path=[px.Constant("Kripto Pazar"), "Sector", "Ticker"], values="Abs_1W", color="1 Haftalık %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
            st.plotly_chart(fig_hm_1w, use_container_width=True)

            st.subheader("AYLIK Performans")
            df_hm_all["Abs_1M"] = df_hm_all["1 Aylık %"].abs()
            fig_hm_1m = px.treemap(df_hm_all, path=[px.Constant("Kripto Pazar"), "Sector", "Ticker"], values="Abs_1M", color="1 Aylık %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
            st.plotly_chart(fig_hm_1m, use_container_width=True)
        else:
            st.error("Heatmap için yeterli veri çekilemedi.")

with tab_export:
    st.subheader("📄 Rapor İndir (En sorunsuz: HTML → tarayıcıdan PDF)")
    include_charts = st.checkbox("Rapor grafikleri dahil et", value=True)
    include_gemini = st.checkbox("Gemini çıktısını rapora ekle", value=True)
    include_pa = st.checkbox("Price Action Pack'i rapora ekle", value=True)
    include_sentiment = st.checkbox("Haber duygu analizini rapora ekle", value=True)
    include_overbought = st.checkbox("Aşırı alım/spekülasyon analizini rapora ekle", value=True)

    meta = {
        "market": "CRYPTO",
        "ticker": symbol,
        "interval": interval,
        "period": period,
        "preset": preset_name,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": rsi_period,
        "bb_period": bb_period,
        "bb_std": bb_std,
        "atr_period": atr_period,
        "vol_sma": vol_sma,
    }

    ta_summary = {
        "rec": rec,
        "close": fmt_num(float(latest["Close"]), 2),
        "live": fmt_num(float(live_price), 2) if np.isfinite(live_price) else "N/A",
        "score": fmt_num(float(latest.get("SCORE", np.nan)), 0),
        "rsi": fmt_num(float(latest.get("RSI", np.nan)), 2),
        "ema50": fmt_num(float(latest.get("EMA50", np.nan)), 2),
        "ema200": fmt_num(float(latest.get("EMA200", np.nan)), 2),
        "atr_pct": fmt_pct(float(latest.get("ATR_PCT", np.nan))) if pd.notna(latest.get("ATR_PCT", np.nan)) else "N/A",
    }

    html_bytes = build_html_report(
        title=f"Bitcoin FA→TA Report - {symbol}",
        meta=meta,
        checkpoints=checkpoints,
        metrics=metrics,
        tp=tp,
        rr_info=rr_info,
        figs=(figs_for_report if include_charts else {}),
        gemini_insight=(st.session_state.gemini_text if include_gemini else None),
        pa_pack=(st.session_state.pa_pack if include_pa else None),
        sentiment_summary=(st.session_state.sentiment_summary if include_sentiment else None),
        sentiment_items=(st.session_state.sentiment_items if include_sentiment else None),
        overbought_result=(overbought_result if include_overbought else None),
    )
    st.download_button("⬇️ HTML İndir (Önerilen)", data=html_bytes, file_name=f"{symbol.replace('/','_')}_report.html", mime="text/html", use_container_width=True)

    st.divider()
    if not REPORTLAB_OK:
        st.warning("Doğrudan PDF için reportlab gerekli. requirements.txt içine reportlab eklenmiştir; yine de ortamında yoksa kurmalısın.")
    else:
        if st.button("🧾 PDF Oluştur (reportlab)", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                pdf_bytes = generate_pdf_report(
                    title=f"Bitcoin FA→TA Report - {symbol}",
                    subtitle="Educational analysis.",
                    meta=meta,
                    checkpoints=checkpoints,
                    ta_summary=ta_summary,
                    target_band=tp,
                    rr_info=rr_info,
                    backtest_metrics=metrics,
                )
            if pdf_bytes:
                st.success("PDF hazır ✅")
                st.download_button("⬇️ PDF İndir", data=pdf_bytes, file_name=f"{symbol.replace('/','_')}_report.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.error("PDF üretilemedi.")

with tab_triple:
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    st.caption("Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan, trend, osilatör ve giriş seviyesi analizi.")

    if st.button("Üçlü Ekran Verilerini Getir ve Analiz Et", key="run_triple"):
        st.session_state.run_triple_screen = True

    if st.session_state.get("run_triple_screen", False):
        with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 1H)..."):
            df_1w = load_data_cached(symbol, "2y", "1wk")
            df_1d = load_data_cached(symbol, "1y", "1d")
            df_1h = load_data_cached(symbol, "45d", "1h")

            if df_1w.empty or df_1d.empty or df_1h.empty:
                st.error("Bazı zaman dilimleri için veri çekilemedi.")
            else:
                t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])

                with t_screen1:
                    st.subheader("1. Ekran: Haftalık (Ana Trend)")
                    df_1w = build_features(df_1w, cfg)
                    m_line, m_sig, m_hist = macd(df_1w["Close"])
                    adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"], 14)
                    bull_trend = bool(df_1w["EMA50"].iloc[-1] > df_1w["EMA200"].iloc[-1])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ana Trend", "YUKARI 🟢" if bull_trend else "AŞAĞI 🔴")
                    c2.metric("MACD Hist", fmt_num(m_hist.iloc[-1], 2))
                    c3.metric("ADX", fmt_num(adx_1w.iloc[-1], 2))

                    fig1_price = go.Figure()
                    fig1_price.add_trace(go.Candlestick(x=df_1w.index, open=df_1w["Open"], high=df_1w["High"], low=df_1w["Low"], close=df_1w["Close"], name="Price"))
                    fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema(df_1w["Close"], 13), name="EMA13"))
                    fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema(df_1w["Close"], 26), name="EMA26"))
                    fig1_price.update_layout(title="Haftalık Fiyat ve EMA (13 & 26)", height=350, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig1_price, use_container_width=True)

                    fig1 = go.Figure()
                    colors = ["green" if x > 0 else "red" for x in m_hist.diff().fillna(0)]
                    fig1.add_trace(go.Bar(x=df_1w.index, y=m_hist, name="MACD Hist", marker_color=colors))
                    fig1.update_layout(title="Haftalık MACD Histogramı", height=250)
                    st.plotly_chart(fig1, use_container_width=True)

                    fig1_adx = go.Figure()
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=adx_1w, name="ADX", line=dict(color="black", width=2.5)))
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=pdi_1w, name="+DI", line=dict(color="green")))
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=mdi_1w, name="-DI", line=dict(color="red")))
                    fig1_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                    fig1_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                    fig1_adx.update_layout(title="Haftalık ADX ve Yön Göstergeleri", height=250)
                    st.plotly_chart(fig1_adx, use_container_width=True)

                with t_screen2:
                    st.subheader("2. Ekran: Günlük (Osilatör / Geri Çekilme)")
                    df_1d = build_features(df_1d, cfg)
                    fi = force_index(df_1d["Close"], df_1d["Volume"])
                    k, d = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], 5, 3)
                    bull_div, bull_bars = check_bullish_divergence(df_1d["Close"], df_1d["RSI"], 30)
                    bear_div, bear_bars = check_bearish_divergence(df_1d["Close"], df_1d["RSI"], 30)

                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("RSI", fmt_num(df_1d["RSI"].iloc[-1], 2), delta="Aşırı Satım" if df_1d["RSI"].iloc[-1] < 30 else ("Aşırı Alım" if df_1d["RSI"].iloc[-1] > 70 else "Normal"))
                    d2.metric("Stoch %K", fmt_num(k.iloc[-1], 2))
                    d3.metric("Bull Divergence", f"{'VAR ✅' if bull_div else 'YOK'}", delta=(f"{bull_bars} bar önce" if bull_div else None))
                    d4.metric("Bear Divergence", f"{'VAR ❌' if bear_div else 'YOK'}", delta=(f"{bear_bars} bar önce" if bear_div else None))

                    fig2_rsi = go.Figure()
                    fig2_rsi.add_trace(go.Scatter(x=df_1d.index, y=df_1d["RSI"], name="RSI"))
                    fig2_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig2_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig2_rsi.update_layout(title="Günlük RSI", height=250)
                    st.plotly_chart(fig2_rsi, use_container_width=True)

                    fig2_stoch = go.Figure()
                    fig2_stoch.add_trace(go.Scatter(x=df_1d.index, y=k, name="%K"))
                    fig2_stoch.add_trace(go.Scatter(x=df_1d.index, y=d, name="%D"))
                    fig2_stoch.add_hline(y=80, line_dash="dash", line_color="red")
                    fig2_stoch.add_hline(y=20, line_dash="dash", line_color="green")
                    fig2_stoch.update_layout(title="Günlük Stochastic", height=250)
                    st.plotly_chart(fig2_stoch, use_container_width=True)

                    fig2_fi = go.Figure()
                    fig2_fi.add_trace(go.Bar(x=df_1d.index, y=fi, name="Force Index"))
                    fig2_fi.update_layout(title="Günlük Force Index", height=250)
                    st.plotly_chart(fig2_fi, use_container_width=True)

                with t_screen3:
                    st.subheader("3. Ekran: 1 Saatlik (Tetikleyici / Giriş Zamanlaması)")
                    df_1h = build_features(df_1h, cfg)
                    ema13 = ema(df_1h["Close"], 13)
                    ema13_high = ema(df_1h["High"], 13)
                    ema13_low = ema(df_1h["Low"], 13)
                    trigger_buy = bool((df_1h["Close"].iloc[-1] > ema13.iloc[-1]) and (df_1h["Close"].iloc[-2] <= ema13.iloc[-2]))
                    trigger_sell = bool((df_1h["Close"].iloc[-1] < ema13.iloc[-1]) and (df_1h["Close"].iloc[-2] >= ema13.iloc[-2]))
                    _, bull_power, bear_power = elder_ray(df_1h["High"], df_1h["Low"], df_1h["Close"], 13)

                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Saatlik Tetik", "AL ✅" if trigger_buy else ("SAT ❌" if trigger_sell else "BEKLE"))
                    h2.metric("Elder Bull Power", fmt_num(bull_power.iloc[-1], 2))
                    h3.metric("Elder Bear Power", fmt_num(bear_power.iloc[-1], 2))
                    h4.metric("Saatlik RSI", fmt_num(df_1h["RSI"].iloc[-1], 2))

                    fig3 = go.Figure()
                    fig3.add_trace(go.Candlestick(x=df_1h.index, open=df_1h["Open"], high=df_1h["High"], low=df_1h["Low"], close=df_1h["Close"], name="Price"))
                    fig3.add_trace(go.Scatter(x=df_1h.index, y=ema13, name="EMA13 Close", line=dict(color="orange")))
                    fig3.add_trace(go.Scatter(x=df_1h.index, y=ema13_high, name="EMA13 High", line=dict(color="gray", dash="dot")))
                    fig3.add_trace(go.Scatter(x=df_1h.index, y=ema13_low, name="EMA13 Low", line=dict(color="gray", dash="dot")))
                    fig3.update_layout(title="Saatlik Fiyat + 13 EMA Kanalı", height=380, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig3, use_container_width=True)

                    fig3_er = go.Figure()
                    fig3_er.add_trace(go.Bar(x=df_1h.index, y=bull_power, name="Bull Power", marker_color="green"))
                    fig3_er.add_trace(go.Bar(x=df_1h.index, y=bear_power, name="Bear Power", marker_color="red"))
                    fig3_er.update_layout(title="Saatlik Elder Ray", height=260, barmode="relative")
                    st.plotly_chart(fig3_er, use_container_width=True)
