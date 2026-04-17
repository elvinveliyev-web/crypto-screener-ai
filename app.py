import os
import re
import json
import time
import base64
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ccxt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# =============================
# OPTIONAL PDF SUPPORT (ReportLab)
# =============================
REPORTLAB_OK = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Crypto Master 5 AI Pro | FA→TA", layout="wide")

# =============================
# BASE DIR & BORSA MOTORU (ccxt)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# KuCoin (Kısıtlamaları ve limitleri kripto için en ideal olanlardan biri)
exchange = ccxt.kucoin({'enableRateLimit': True})

def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)

# =============================
# Universe Loader
# =============================
@st.cache_data(ttl=3600, show_spinner=False)
def get_crypto_universe() -> List[str]:
    try:
        tickers = exchange.fetch_tickers()
        symbols = [s for s, t in tickers.items() if s.endswith('/USDT') and ':' not in s]
        prio = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", "MATIC/USDT"]
        return prio + sorted([p for p in symbols if p not in prio])
    except Exception:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

# =============================
# Helpers
# =============================
def safe_float(x):
    try:
        if x is None: return np.nan
        if isinstance(x, (int, float, np.number)): return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)): return "N/A"
        return f"{x*100:.2f}%"
    except Exception: return "N/A"

def fmt_num(x: float, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)): return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception: return "N/A"

# =============================
# Indicators
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
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0: return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())

def force_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    return volume * (close - close.shift(1))

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 5, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13):
    e = ema(close, period)
    bull_power = high - e
    bear_power = low - e
    return e, bull_power, bear_power

def check_bullish_divergence(close: pd.Series, indicator: pd.Series, lookback: int = 30) -> Tuple[bool, int]:
    if len(close) < lookback: return False, 0
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        min_idx = c.values.argmin()
        bars_ago = (lookback - 1) - min_idx
        prev_c = c.iloc[:min_idx-2]
        if len(prev_c) < 3: return False, 0
        prev_min_idx = prev_c.values.argmin()
        p1, p2 = prev_c.iloc[prev_min_idx], c.iloc[min_idx]
        i1, i2 = ind.iloc[prev_min_idx], ind.iloc[min_idx]
        if p2 < p1 and i2 > i1: return True, bars_ago
    except Exception: pass
    return False, 0

def check_bearish_divergence(close: pd.Series, indicator: pd.Series, lookback: int = 30) -> Tuple[bool, int]:
    if len(close) < lookback: return False, 0
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        max_idx = c.values.argmax()
        bars_ago = (lookback - 1) - max_idx
        prev_c = c.iloc[:max_idx-2]
        if len(prev_c) < 3: return False, 0
        prev_max_idx = prev_c.values.argmax()
        p1, p2 = prev_c.iloc[prev_max_idx], c.iloc[max_idx]
        i1, i2 = ind.iloc[prev_max_idx], ind.iloc[max_idx]
        if p2 > p1 and i2 < i1: return True, bars_ago
    except Exception: pass
    return False, 0

def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)
    tr_smooth = pd.Series(tr, index=high.index).ewm(alpha=1/period, adjust=False).mean()
    pdm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    mdm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * (pdm_smooth / tr_smooth.replace(0, np.nan))
    mdi = 100 * (mdm_smooth / tr_smooth.replace(0, np.nan))
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)

# =============================
# KANGAROO TAIL
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

# =============================
# PRICE ACTION PATTERNS
# =============================
def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    O, H, L, C = df["Open"], df["High"], df["Low"], df["Close"]
    Body = (C - O).abs()
    Range = H - L
    UpperWick = H - df[["Open", "Close"]].max(axis=1)
    LowerWick = df[["Open", "Close"]].min(axis=1) - L
    AvgRange = Range.rolling(10).mean()

    is_bull = C > O
    is_bear = C < O

    df["PATTERN_DOJI"] = (Body <= 0.1 * Range) & (Range > 0)
    df["PATTERN_LL_DOJI"] = df["PATTERN_DOJI"] & (UpperWick >= 0.35 * Range) & (LowerWick >= 0.35 * Range) & (Range > AvgRange * 0.8)

    shape_hammer = (LowerWick >= 2 * Body) & (UpperWick <= 0.2 * Range) & (Body > 0.02 * Range)
    df["PATTERN_HAMMER"] = shape_hammer & (C < df.get("EMA50", C)) 
    df["PATTERN_HANGING_MAN"] = shape_hammer & (C > df.get("EMA50", C)) 

    shape_star = (UpperWick >= 2 * Body) & (LowerWick <= 0.2 * Range) & (Body > 0.02 * Range)
    df["PATTERN_SHOOTING_STAR"] = shape_star & (C > df.get("EMA50", C)) 
    df["PATTERN_INV_HAMMER"] = shape_star & (C < df.get("EMA50", C)) 

    df["PATTERN_MARUBOZU_BULL"] = is_bull & (Body >= 0.85 * Range) & (Range > AvgRange * 0.5)
    df["PATTERN_MARUBOZU_BEAR"] = is_bear & (Body >= 0.85 * Range) & (Range > AvgRange * 0.5)

    prev_is_bear = is_bear.shift(1); prev_is_bull = is_bull.shift(1); prev_O = O.shift(1); prev_C = C.shift(1)

    df["PATTERN_ENGULFING_BULL"] = is_bull & prev_is_bear & (O <= prev_C) & (C >= prev_O) & (Body > (prev_O - prev_C))
    df["PATTERN_ENGULFING_BEAR"] = is_bear & prev_is_bull & (O >= prev_C) & (C <= prev_O) & (Body > (prev_C - prev_O))

    df["PATTERN_HARAMI_BULL"] = is_bull & prev_is_bear & (O > prev_C) & (C < prev_O) & ((prev_O - prev_C) > AvgRange * 0.5)
    df["PATTERN_HARAMI_BEAR"] = is_bear & prev_is_bull & (O < prev_C) & (C > prev_O) & ((prev_C - prev_O) > AvgRange * 0.5)

    prev_H = H.shift(1); prev_L = L.shift(1)
    df["PATTERN_TWEEZER_TOP"] = (abs(H - prev_H) <= 0.002 * C) & is_bear & prev_is_bull & (H > df.get("EMA50", C))
    df["PATTERN_TWEEZER_BOTTOM"] = (abs(L - prev_L) <= 0.002 * C) & is_bull & prev_is_bear & (L < df.get("EMA50", C))

    df["PATTERN_PIERCING"] = is_bull & prev_is_bear & (O < L.shift(1)) & (C > (prev_O + prev_C)/2) & (C < prev_O)
    df["PATTERN_DARK_CLOUD"] = is_bear & prev_is_bull & (O > H.shift(1)) & (C < (prev_O + prev_C)/2) & (C > prev_O)

    prev2_is_bear = is_bear.shift(2); prev2_is_bull = is_bull.shift(2); prev2_O = O.shift(2); prev2_C = C.shift(2)

    df["PATTERN_MORNING_STAR"] = is_bull & prev2_is_bear & (prev_C < prev2_C) & (O > prev_C) & (C > (prev2_O + prev2_C)/2)
    df["PATTERN_EVENING_STAR"] = is_bear & prev2_is_bull & (prev_C > prev2_C) & (O < prev_C) & (C < (prev2_O + prev2_C)/2)

    return df

# =============================
# Overbought / Speculation
# =============================
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
        min_rsi = series.rolling(period).min()
        max_rsi = series.rolling(period).max()
        den = (max_rsi - min_rsi).replace(0, np.nan)
        stoch = 100 * (series - min_rsi) / den
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
    result = {"overbought_score": 0, "oversold_score": 0, "speculation_score": 0, "details": {}}

    if last["RSI"] > 70:
        result["overbought_score"] += 40; result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        result["oversold_score"] += 50; result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"

    if bool(last["BB_OVERBOUGHT"]):
        result["overbought_score"] += 20; result["details"]["bb"] = "Fiyat Bollinger üst bandında"
    elif bool(last["BB_OVERSOLD"]):
        result["oversold_score"] += 50; result["details"]["bb"] = "Fiyat Bollinger alt bandında"

    if bool(last["STOCH_OVERBOUGHT"]):
        result["overbought_score"] += 20; result["details"]["stoch"] = "Stokastik RSI aşırı alımda"

    if bool(last["VOLUME_SPIKE"]):
        result["speculation_score"] += 60; result["details"]["volume"] = "Ani hacim artışı (spekülasyon)"

    if bool(last["PRICE_EXTREME"]):
        result["overbought_score"] += 20; result["details"]["price_extreme"] = f"Fiyat EMA'dan çok uzak (EMA50: %{last['PRICE_TO_EMA50']:.1f})"

    if bool(last["WEAK_UPTREND"]):
        result["speculation_score"] += 40; result["details"]["weak_trend"] = "Fiyat yükselirken hacim düşüyor (zayıflama)"

    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])

    if result["overbought_score"] >= 60: result["verdict"] = "AŞIRI DEĞERLİ (SAT bölgesi)"
    elif result["oversold_score"] >= 60: result["verdict"] = "AŞIRI DEĞERSİZ (AL bölgesi)"
    elif result["speculation_score"] >= 60: result["verdict"] = "SPEKÜLATİF HAREKET (dikkatli olunmalı)"
    else: result["verdict"] = "NÖTR (normal değer aralığı)"
    return result

# =============================
# Feature builder
# =============================
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

    # Ekstra Sinyaller (Screener İçin)
    adx, pdi, mdi = adx_indicator(df["High"], df["Low"], df["Close"], 14)
    df["ADX"] = adx; df["PLUS_DI"] = pdi; df["MINUS_DI"] = mdi
    stoch_k, stoch_d = stochastic(df["High"], df["Low"], df["Close"], 5, 3)
    df["STOCH_K"] = stoch_k; df["STOCH_D"] = stoch_d

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
# Cached data loader (Zaman Makinesi Uyumlu)
# =============================
@st.cache_data(ttl=300, show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str, target_dt: pd.Timestamp = None) -> pd.DataFrame:
    tf_map = {"1m": "1m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1w"}
    ccxt_tf = tf_map.get(interval, "1d")

    limit = 1000
    if period == "45d": limit = 45
    elif period == "3mo": limit = 90
    elif period == "6mo": limit = 180
    elif period == "1y": limit = 365
    elif period == "2y": limit = 730
    
    if ccxt_tf == "1h": limit *= 24
    elif ccxt_tf == "4h": limit *= 6
    elif ccxt_tf == "1w": limit = max(limit // 7, 100)
    
    limit = min(limit, 1500) # KuCoin limit safe bound

    since_ms = None
    if target_dt is not None:
        fetch_limit = limit + 50
        mins_map = {"1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
        mins = mins_map.get(ccxt_tf, 1440)
        since_dt = target_dt - pd.Timedelta(minutes=mins * fetch_limit)
        since_ms = int(since_dt.timestamp() * 1000)

    try:
        bars = exchange.fetch_ohlcv(ticker, timeframe=ccxt_tf, limit=limit if since_ms is None else 1500, since=since_ms)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        if target_dt is not None:
            df = df[df.index <= target_dt]
            df = df.tail(limit) # Get exactly the period requested ending at target date

        return df
    except Exception:
        return pd.DataFrame()

# =============================
# Market regime filters
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_btc_regime_series(target_dt=None) -> pd.Series:
    try:
        df = load_data_cached("BTC/USDT", "1y", "1d", target_dt)
        if df.empty: return pd.Series(dtype=bool)
        df["EMA200"] = ema(df["Close"], 200)
        return df["Close"] > df["EMA200"]
    except Exception:
        return pd.Series(dtype=bool)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_higher_tf_trend_series(ticker: str, higher_tf_interval: str = "1w", ema_period: int = 200, target_dt=None) -> pd.Series:
    try:
        df = load_data_cached(ticker, "2y", higher_tf_interval, target_dt)
        if df.empty: return pd.Series(dtype=bool)
        df["EMA"] = ema(df["Close"], ema_period)
        return df["Close"] > df["EMA"]
    except Exception:
        return pd.Series(dtype=bool)

# =============================
# Strategy: scoring + checkpoints
# =============================
def signal_with_checkpoints(df: pd.DataFrame, cfg: dict, market_filter_series: pd.Series = None, higher_tf_filter_series: pd.Series = None):
    df = df.copy()
    liq_ok = (df["Volume"] > df["VOL_SMA"]).fillna(False)
    trend_ok = (df["Close"] > df["EMA200"]) & (df["EMA50"] > df["EMA200"])

    if market_filter_series is not None and not market_filter_series.empty:
        aligned_market = market_filter_series.reindex(df.index).ffill().fillna(True)
    else: aligned_market = pd.Series(True, index=df.index)

    if higher_tf_filter_series is not None and not higher_tf_filter_series.empty:
        aligned_htf = higher_tf_filter_series.reindex(df.index).ffill().fillna(True)
    else: aligned_htf = pd.Series(True, index=df.index)

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
        w["liq"] * liq_ok.astype(int) + w["trend"] * trend_ok.astype(int) + w["rsi"] * rsi_ok.astype(int)
        + w["macd"] * macd_ok.astype(int) + w["vol"] * vol_ok.astype(int) + w["bb"] * (bb_ok | bb_break).astype(int)
        + w["obv"] * obv_ok.astype(int)
    ).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 1
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & aligned_market & aligned_htf
    exit_ = ((df["Close"] < df["EMA50"]) | (df["MACD_hist"] < 0) | (df["RSI"] < cfg["rsi_exit_level"]) | (df["Close"] < df["BB_mid"]))

    df["SCORE"] = score; df["ENTRY"] = entry.astype(int); df["EXIT"] = exit_.astype(int)

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

# =============================
# Backtest
# =============================
def backtest_long_only(df: pd.DataFrame, cfg: dict, benchmark_returns: Optional[pd.Series] = None):
    df = df.copy()
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int); exit_sig = df["EXIT"].shift(1).fillna(0).astype(int)
    cash = float(cfg["initial_capital"]); shares = 0.0; stop = np.nan; entry_price = np.nan; target_price = np.nan
    bars_held = 0; half_sold = False; trades = []; equity_curve = []

    commission = cfg["commission_bps"] / 10000.0; slippage = cfg["slippage_bps"] / 10000.0
    time_stop_bars = cfg.get("time_stop_bars", 10); tp_mult = cfg.get("take_profit_mult", 2.0); risk_pct = float(cfg.get("risk_per_trade", 0.01)) 

    for i in range(len(df)):
        row = df.iloc[i]; date = df.index[i]; price = float(row["Close"])
        if shares > 0 and pd.notna(row["ATR"]) and row["ATR"] > 0:
            new_stop = price - cfg["atr_stop_mult"] * float(row["ATR"])
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        if shares == 0 and entry_sig.iloc[i] == 1:
            atrv = float(row.get("ATR", np.nan))
            if pd.notna(atrv) and atrv > 0:
                risk_amount = cash * risk_pct
                is_kangaroo = int(row.get("KANGAROO_BULL", 0)) == 1
                stop_dist = price - (float(row["Low"]) - (0.5 * atrv)) if is_kangaroo else cfg["atr_stop_mult"] * atrv
                stop_price = price - stop_dist
                shares_to_buy = min(risk_amount / stop_dist, cash / (price * (1 + slippage + commission)))
                if shares_to_buy > 0.001: 
                    shares = shares_to_buy; entry_price = price * (1 + slippage)
                    cash -= ((shares * entry_price) + (shares * entry_price) * commission)
                    stop = stop_price; target_price = entry_price + (tp_mult * stop_dist)
                    trades.append({"entry_date": date, "entry_price": entry_price, "equity_before": cash + (shares * price)})

        if shares > 0:
            bars_held += 1
            stop_hit = pd.notna(stop) and (price <= stop)
            target_hit = (not half_sold) and pd.notna(target_price) and (price >= target_price)
            time_stop_hit = (bars_held >= time_stop_bars) and (price < entry_price)

            if target_hit:
                sell_shares = shares * 0.5; sell_price = price * (1 - slippage)
                cash += (sell_shares * sell_price) - (sell_shares * sell_price * commission)
                shares -= sell_shares; half_sold = True; stop = max(stop, entry_price)
                if len(trades) > 0: trades[-1]["pnl"] = cash + (shares * price * (1 - slippage)) - trades[-1]["equity_before"]

            if exit_sig.iloc[i] == 1 or stop_hit or time_stop_hit:
                sell_price = price * (1 - slippage)
                cash += (shares * sell_price) - (shares * sell_price * commission)
                trades[-1]["exit_date"] = date; trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = "STOP" if stop_hit else ("TIME_STOP" if time_stop_hit else "RULE_EXIT")
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]
                shares = 0.0; stop = np.nan; entry_price = np.nan; target_price = np.nan; bars_held = 0; half_sold = False

        equity_curve.append((date, cash + (shares * price * (1 - slippage))))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    excess = ret - ((1 + 0.0) ** (1 / 252) - 1)
    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    mdd = max_drawdown(eq)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        if "pnl" not in tdf.columns: tdf["pnl"] = np.nan
        tdf["pnl"] = tdf["pnl"].astype(float)

    metrics = {
        "Total Return": float(total_return), "Annualized Return": float(ann_return), "Sharpe": float(sharpe), "Max Drawdown": float(mdd),
        "Trades": int(len(tdf)) if not tdf.empty else 0, "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty and "pnl" in tdf.columns else 0.0
    }
    return eq, tdf, metrics

# =============================
# Target price band / SR Levels
# =============================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs = []; ls = []; n = len(high)
    for i in range(left, n - right):
        if high.iloc[i] == high.iloc[i - left : i + right + 1].max(): hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == low.iloc[i - left : i + right + 1].min(): ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls

def analyze_sr_levels(df: pd.DataFrame, lookback: int = 200, tol=0.02) -> List[dict]:
    h = df["High"].tail(lookback).dropna(); l = df["Low"].tail(lookback).dropna(); c = df["Close"].tail(lookback).dropna()
    if len(c) < 10: return []
    v = df["Volume"].tail(lookback) if "Volume" in df.columns else pd.Series(dtype=float)

    hs, ls = _swing_points(h, l, left=3, right=3)
    raw_levels = sorted(list(set([round(float(x), 2) for x in [val for _, val in hs] + [val for _, val in ls] + [float(c.tail(20).max()), float(c.tail(20).min())] if np.isfinite(x)])))

    if not raw_levels: return []
    clusters = []
    for rl in raw_levels:
        placed = False
        for cl in clusters:
            if abs(rl - cl['center']) / cl['center'] <= tol: cl['points'].append(rl); placed = True; break
        if not placed: clusters.append({'center': rl, 'points': [rl]})

    avg_vol_normal = max(float(v.mean()), 1.0) if not v.empty else 1.0
    details = []
    df_lookback = df.tail(lookback)
    for cl in clusters:
        touches = df_lookback[(df_lookback["High"] >= cl['center'] * (1 - tol/2)) & (df_lookback["Low"] <= cl['center'] * (1 + tol/2))]
        if len(touches) == 0: continue
        vol_at_level = float(touches["Volume"].mean()) if "Volume" in df_lookback.columns and not touches.empty else avg_vol_normal
        dur = len(df_lookback) - df_lookback.index.get_loc(touches.index[0])
        strength_pct = min(min(len(touches) * 10, 40) + min(max(((vol_at_level / avg_vol_normal - 1.0) * 100.0) / 2.0, 0), 35) + min(dur / 2.0, 25), 99.0)
        details.append({"price": round(cl['center'], 2), "duration_bars": int(dur), "vol_at_level": float(vol_at_level), "vol_diff_pct": float((vol_at_level / avg_vol_normal - 1.0) * 100.0), "strength_pct": float(strength_pct), "touches": int(len(touches))})
    return sorted(details, key=lambda x: x["price"])

def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]; px_close = float(last["Close"]); atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan
    lv_details = analyze_sr_levels(df)

    if not np.isfinite(atrv) or atrv <= 0: return {"base": px_close, "bull": None, "bear": None, "levels": lv_details, "r1_dict": None, "s1_dict": None}

    bull1, bull2 = px_close + 1.5 * atrv, px_close + 3.0 * atrv
    bear1, bear2 = px_close - 1.5 * atrv, px_close - 3.0 * atrv

    valid_above = [x for x in lv_details if x["price"] >= px_close * 1.005 and x["duration_bars"] >= 10 and x["touches"] >= 2]
    valid_below = [x for x in lv_details if x["price"] <= px_close * 0.995 and x["duration_bars"] >= 10 and x["touches"] >= 2]
    strong_above = [x for x in valid_above if x["strength_pct"] >= 25]
    strong_below = [x for x in valid_below if x["strength_pct"] >= 35]
    
    r1_dict = min(strong_above, key=lambda x: x["price"]) if strong_above else (min(valid_above, key=lambda x: x["price"]) if valid_above else None)
    s1_dict = max(strong_below, key=lambda x: x["price"]) if strong_below else (max(valid_below, key=lambda x: x["price"]) if valid_below else None)

    r1 = r1_dict["price"] if r1_dict else None
    s1 = s1_dict["price"] if s1_dict else None
    
    pivot = (float(last["High"]) + float(last["Low"]) + px_close) / 3.0
    if r1 is None and ((2 * pivot) - float(last["Low"])) > px_close:
        r1 = (2 * pivot) - float(last["Low"])
        r1_dict = {"price": r1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    if s1 is None and px_close > ((2 * pivot) - float(last["High"])) > 0:
        s1 = (2 * pivot) - float(last["High"])
        s1_dict = {"price": s1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    return {"base": px_close, "bull": (bull1, bull2, r1), "bear": (bear1, bear2, s1), "levels": lv_details, "r1_dict": r1_dict, "s1_dict": s1_dict}

# =============================
# Live price
# =============================
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker: str) -> dict:
    try: return {"last_price": safe_float(exchange.fetch_ticker(ticker)['last']), "currency": "USDT", "exchange": "KuCoin"}
    except Exception: return {"last_price": np.nan, "currency": "USDT", "exchange": "KuCoin"}

# =============================
# Gemini helpers
# =============================
def _get_secret(name: str, default: str = "") -> str:
    try: return str(st.secrets.get(name, default)).strip() if st.secrets.get(name) is not None else default
    except Exception: return default

def gemini_generate_text(*, prompt: str, model: str = "gemini-1.5-flash", temperature: float = 0.2, max_output_tokens: int = 2048, image_bytes: Optional[bytes] = None) -> str:
    api_key = _get_secret("GEMINI_API_KEY", "")
    if not api_key: return "GEMINI_API_KEY bulunamadı."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts = [{"text": prompt}]
    if image_bytes: parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image_bytes).decode("utf-8")}})
    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)}}
    try:
        r = requests.post(url, json=payload, headers={"x-goog-api-key": api_key}, timeout=90).json()
        return "\n".join([p["text"] for p in r.get("candidates", [{}])[0].get("content", {}).get("parts", []) if "text" in p]) or "Cevap üretilemedi."
    except Exception as e: return str(e)

# =============================
# Export HTML / PDF (Condensed)
# =============================
def build_html_report(title, meta, checkpoints, metrics, tp, rr_info, figs, gemini_insight=None) -> bytes:
    def esc(x): return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    fig_blocks = "".join([f"<h3>{esc(name)}</h3>{fig.to_html(full_html=False, include_plotlyjs=('cdn' if i==0 else False))}" for i, (name, fig) in enumerate(figs.items())])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title></head><body style="font-family:Arial; margin:24px;"><h1>{esc(title)}</h1>{fig_blocks}</body></html>""".encode("utf-8")

def _plotly_fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    try: return fig.to_image(format="png", scale=2)
    except Exception: return None

# =============================
# RR helper 
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"]); atrv = float(latest_row.get("ATR", np.nan))
    if not np.isfinite(atrv) or atrv <= 0: return {"rr": None, "stop": None, "risk": None, "reward": None}

    stop = float(latest_row["Low"]) - (0.5 * atrv) if latest_row.get("KANGAROO_BULL", 0) == 1 else close - (float(cfg["atr_stop_mult"]) * atrv)
    risk = close - stop
    
    r1 = tp_dict["bull"][2] if tp_dict and tp_dict.get("bull") else None
    if r1 is not None and np.isfinite(r1) and r1 > close: target = float(r1); target_type = "Resistance (R1)"
    else: target = close + (cfg.get("take_profit_mult", 2.0) * cfg["atr_stop_mult"] * atrv); target_type = "ATR Target"

    reward = target - close
    return {"rr": None if risk <= 0 or reward <= 0 else float(reward / risk), "stop": float(stop), "risk": float(risk), "reward": reward, "target_type": target_type}

def pct_dist(level: float, base: float): return None if not level or not np.isfinite(level) or base == 0 else (level / base - 1.0) * 100.0

# =============================
# UI STATE
# =============================
if "app_errors" not in st.session_state: st.session_state.app_errors = []
if "ta_ran" not in st.session_state: st.session_state.ta_ran = False
if "gemini_text" not in st.session_state: st.session_state.gemini_text = ""
if "show_ema13_channel" not in st.session_state: st.session_state.show_ema13_channel = False

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/cryptocurrency.png", width=80)
    st.header("1) Piyasa & Sembol")
    ticker = st.selectbox("Kripto Parite", get_crypto_universe(), index=0)

    st.subheader("⏱️ Zaman Makinesi (Geçmiş Analiz)")
    use_timemachine = st.checkbox("Geçmiş bir tarihe/saate göre analiz yap", value=False, help="Örn: Dün gece 00:00 itibarıyla günlük ve haftalık kapanışa göre analiz.")
    if use_timemachine:
        hist_d = st.date_input("Tarih", datetime.date.today())
        hist_t = st.time_input("Saat", datetime.time(0, 0))
        target_datetime = pd.to_datetime(f"{hist_d} {hist_t}")
        st.warning(f"Analiz Hedefi:\n**{target_datetime}**")
    else:
        target_datetime = None

    st.header("2) Teknik Parametreler")
    preset_name = st.selectbox("Risk Modu", ["Defansif", "Dengeli", "Agresif"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "4h", "1h"], index=0)
    period = st.selectbox("Periyot", ["45d", "3mo", "6mo", "1y", "2y"], index=3)

    st.divider()
    ema_fast = st.number_input("EMA Fast", value=50); ema_slow = st.number_input("EMA Slow", value=200)
    rsi_period = st.number_input("RSI Period", value=14); bb_period = st.number_input("BB Period", value=20)
    bb_std = st.number_input("BB Std", value=2.0); atr_period = st.number_input("ATR Period", value=14)
    vol_sma = st.number_input("Volume SMA", value=20)

    use_btc_filter = st.checkbox("BTC > EMA200 filtresi", value=True)
    use_higher_tf_filter = st.checkbox("Haftalık trend filtresi", value=True)

    initial_capital = st.number_input("Sermaye ($)", value=1000.0); risk_per_trade = st.slider("Risk (%)", 0.002, 0.05, 0.01)
    commission_bps = st.number_input("Komisyon (bps)", value=10.0); slippage_bps = st.number_input("Slippage (bps)", value=5.0)

    st.header("3) AI Ayarları")
    ai_on = st.checkbox("Gemini AI aktif", value=True)

    run_btn = st.button("🚀 Analizi Çalıştır", type="primary")
    if run_btn: st.session_state.ta_ran = True

PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 2.0, "time_stop_bars": 15, "take_profit_mult": 2.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 1.5, "time_stop_bars": 10, "take_profit_mult": 2.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 1.2, "time_stop_bars": 7, "take_profit_mult": 1.5},
}
cfg = {"ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_period": rsi_period, "bb_period": bb_period, "bb_std": bb_std, "atr_period": atr_period, "vol_sma": vol_sma, "initial_capital": initial_capital, "risk_per_trade": risk_per_trade, "commission_bps": commission_bps, "slippage_bps": slippage_bps}
cfg.update(PRESETS[preset_name])

st.title("📈 Kripto Master 5 AI Pro | FA→TA Trading")

if not st.session_state.ta_ran:
    st.info("👈 Sol menüden Kripto Sembolünü, gerekirse 'Zaman Makinesi' tarihini seçip **Analizi Çalıştır**'a basın.")
    st.stop()

# =============================
# Run TA pipeline
# =============================
market_filter_series = get_btc_regime_series(target_datetime) if use_btc_filter else None
higher_tf_filter_series = get_higher_tf_trend_series(ticker, "1w", 200, target_datetime) if use_higher_tf_filter else None

with st.spinner(f"Veri indiriliyor: {ticker} (Hedef: {'Şimdi' if target_datetime is None else target_datetime})"):
    df_raw = load_data_cached(ticker, period, interval, target_datetime)

if df_raw.empty:
    st.error(f"Seçilen tarih aralığı için veri bulunamadı: {ticker}")
    st.stop()

df = build_features(df_raw, cfg)
df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_series, higher_tf_filter_series)
latest = df.iloc[-1]

# Zaman makinesi kullanılıyorsa "Canlı Fiyat" yerine o anki kapanış fiyatını gösteririz.
if use_timemachine:
    live_price = float(latest["Close"])
else:
    live = get_live_price(ticker)
    live_price = live.get("last_price", float(latest["Close"]))

if int(latest["ENTRY"]) == 1: rec = "AL"
elif int(latest["EXIT"]) == 1: rec = "SAT"
else: rec = "AL (Güçlü Trend)" if latest["SCORE"] >= 80 else ("İZLE" if latest["SCORE"] >= 60 else "UZAK DUR")

eq, tdf, metrics = backtest_long_only(df, cfg)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)

df["EMA13_High"] = ema(df["High"], 13); df["EMA13_Low"] = ema(df["Low"], 13); df["EMA13_Close"] = ema(df["Close"], 13)

# =============================
# Build figures
# =============================
if "show_chart_patterns" not in st.session_state: st.session_state.show_chart_patterns = True

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast")); fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot"))); fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))

if st.session_state.show_ema13_channel:
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_High"], name="13 EMA High", line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Low"], name="13 EMA Low", fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Close"], name="13 EMA Close", line=dict(color='darkorange', width=2)))

if st.session_state.show_chart_patterns:
    for pat, name in {"KANGAROO_BULL": "🟩🦘", "PATTERN_ENGULFING_BULL": "🟢 ENG", "PATTERN_HAMMER": "🟩🔨"}.items():
        if pat in df.columns: fig_price.add_trace(go.Scatter(x=df.index[df[pat]==1], y=df["Low"][df[pat]==1], mode="markers+text", name=name, text=name, textposition="bottom center", marker=dict(symbol="triangle-up", size=10, color="green")))
    for pat, name in {"KANGAROO_BEAR": "🟥🦘", "PATTERN_ENGULFING_BEAR": "🔴 ENG", "PATTERN_SHOOTING_STAR": "🟥🌠"}.items():
        if pat in df.columns: fig_price.add_trace(go.Scatter(x=df.index[df[pat]==1], y=df["High"][df[pat]==1], mode="markers+text", name=name, text=name, textposition="top center", marker=dict(symbol="triangle-down", size=10, color="red")))

fig_price.update_layout(height=600, xaxis_rangeslider_visible=False, title="Fiyat Grafiği")

fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df["RSI"], name="RSI")])
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

# =============================
# Tabs
# =============================
tab_dash, tab_triple, tab_screener = st.tabs(["📊 Dashboard", "📺 3 Ekranlı Sistem", "🔍 Gelişmiş Çoklu Tarayıcı (Screener)"])

with tab_dash:
    if use_timemachine:
        st.error(f"⚠️ DİKKAT: Şu an Zaman Makinesi aktif. Tüm grafikler ve skorlar **{target_datetime}** tarihine göre hesaplanmıştır.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market & Sembol", ticker)
    c2.metric("Güncel Close", f"{latest['Close']:.4f}")
    c3.metric("Skor & Sinyal", f"{latest['SCORE']:.0f}/100", rec)
    c4.metric("Risk Ödül (RR)", fmt_rr(rr_info.get("rr")))

    st.subheader("Fiyat + EMA + Bollinger")
    st.plotly_chart(fig_price, use_container_width=True)

    tools_col1, tools_col2, tools_col3 = st.columns(3)
    if tools_col1.button("Formasyonları Gizle/Aç", use_container_width=True):
        st.session_state.show_chart_patterns = not st.session_state.show_chart_patterns
        st.rerun()
    if tools_col2.button("13 EMA Kanalını Aç / Kapat", use_container_width=True):
        st.session_state.show_ema13_channel = not st.session_state.show_ema13_channel
        st.rerun()

    st.subheader("RSI Endeksi")
    st.plotly_chart(fig_rsi, use_container_width=True)

# =============================
# TRIPLE SCREEN TAB
# =============================
with tab_triple:
    st.header("📺 Üçlü Ekran Trading Sistemi")
    st.caption("Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan, trend ve giriş seviyesi analizleri.")
    
    if st.button("Üçlü Ekran Verilerini Getir ve Analiz Et", key="run_triple"):
        with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 1H)..."):
            df_1w = load_data_cached(ticker, "2y", "1wk", target_datetime)
            df_1d = load_data_cached(ticker, "1y", "1d", target_datetime)
            df_1h = load_data_cached(ticker, "60d", "1h", target_datetime)
            
            if df_1w.empty or df_1d.empty or df_1h.empty:
                st.error("Bazı zaman dilimleri için veri çekilemedi.")
            else:
                t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])
                
                with t_screen1:
                    m_line, m_sig, m_hist = macd(df_1w["Close"]); adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"])
                    st.metric("MACD Hist Eğimi", "YUKARI" if float(m_hist.iloc[-1]) > float(m_hist.iloc[-2]) else "AŞAĞI")
                    st.metric("Haftalık ADX", f"ADX: {adx_1w.iloc[-1]:.1f}")
                    
                    fig1 = go.Figure(data=[go.Bar(x=df_1w.index, y=m_hist, marker_color=['green' if x > 0 else 'red' for x in m_hist.diff()])])
                    st.plotly_chart(fig1, use_container_width=True)

                with t_screen2:
                    rsi13 = rsi(df_1d["Close"], 13); stoch_k, stoch_d = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], 5, 3)
                    er_ema, bull_p, bear_p = elder_ray(df_1d["High"], df_1d["Low"], df_1d["Close"], 13)
                    
                    st.metric("RSI (13)", f"{rsi13.iloc[-1]:.1f}")
                    st.metric("Stokastik", f"K: {stoch_k.iloc[-1]:.1f}")
                    
                    fig2_er = go.Figure()
                    fig2_er.add_trace(go.Bar(x=df_1d.index, y=bull_p, name="Bull Power", marker_color='green'))
                    fig2_er.add_trace(go.Bar(x=df_1d.index, y=bear_p, name="Bear Power", marker_color='red'))
                    st.plotly_chart(fig2_er, use_container_width=True)

                with t_screen3:
                    ema_1h = ema(df_1h["Close"], 13); atr_1h = atr(df_1h["High"], df_1h["Low"], df_1h["Close"], 14)
                    ema_est_tmrw = float(ema_1h.iloc[-1]) + (float(ema_1h.iloc[-1]) - float(ema_1h.iloc[-2]))
                    pens = ema_1h - df_1h["Low"]; avg_pen = float(pens[pens > 0].mean()) if not pens[pens > 0].empty else 0.0
                    buy_level = ema_est_tmrw - avg_pen
                    
                    st.markdown(f"**Önerilen Alış (Buy Limit):** {buy_level:.4f} \n\n**Stop-Loss:** {buy_level - (1.5 * float(atr_1h.iloc[-1])):.4f}")

# =============================
# YENİ EKLENEN 5. SEKME: GELİŞMİŞ ÇOKLU TARAYICI (SCREENER)
# =============================
with tab_screener:
    st.header("🔍 Gelişmiş Çoklu Kripto Tarayıcı (AI Destekli Skorlama)")
    st.markdown("Onlarca coini; RSI, MACD, StochRSI, ADX, Bollinger, Hacim, OBV ve Formasyonlara göre analiz edip 100 üzerinden skorlar.")
    
    if use_timemachine:
        st.warning(f"Zaman Makinesi Aktif: Tarama **{target_datetime}** verileri baz alınarak yapılacaktır.")

    sc_col1, sc_col2 = st.columns(2)
    scan_count = sc_col1.slider("Taranacak Coin Sayısı (Hıza etki eder)", 10, 150, 30, step=10)
    scan_tf = sc_col2.selectbox("Tarama Periyodu", ["1h", "4h", "1d"], index=2)
    
    if st.button("🚀 Kapsamlı Taramayı Başlat"):
        all_coins = get_crypto_universe()[:scan_count]
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        def scan_single_coin(coin):
            try:
                # 200 EMA ve diğer göstergeler için en az 1 yıllık (veya 300 barlık) data çekiyoruz
                d = load_data_cached(coin, "1y", scan_tf, target_datetime) 
                if d.empty or len(d) < 50: return None
                
                d = build_features(d, cfg) 
                last_row = d.iloc[-1]
                
                # --- KOMPLEKS SKORLAMA (Max 100) ---
                score = 0
                
                # Trend (Max 20)
                if last_row["Close"] > last_row["EMA200"]: score += 10
                if last_row["EMA50"] > last_row["EMA200"]: score += 10
                
                # RSI (Max 15)
                rsi_val = last_row["RSI"]
                if 50 < rsi_val < 70: score += 10
                elif rsi_val <= 30: score += 15 # Aşırı satım (dip) fırsatı
                
                # MACD (Max 10)
                if last_row["MACD_hist"] > 0: score += 10
                
                # ADX (Max 15)
                if last_row["ADX"] > 25 and last_row["PLUS_DI"] > last_row["MINUS_DI"]: score += 15
                
                # Stochastic RSI (Max 10)
                if last_row["STOCH_K"] > last_row["STOCH_D"] and last_row["STOCH_K"] < 80: score += 10
                
                # Bollinger (Max 10)
                if last_row["Close"] > last_row["BB_mid"]: score += 5
                if last_row["BB_WIDTH"] < 0.10: score += 5 # Squeeze (sıkışma patlama potansiyeli)
                
                # OBV & Hacim (Max 10)
                if last_row["OBV"] > last_row.get("OBV_EMA", last_row["OBV"]): score += 5
                if last_row["Volume"] > last_row.get("VOL_SMA", 0): score += 5
                
                # Mum Formasyonları (Max +10 / Min -15)
                bull_patterns = ["KANGAROO_BULL", "PATTERN_ENGULFING_BULL", "PATTERN_HAMMER", "PATTERN_MORNING_STAR", "PATTERN_PIERCING", "PATTERN_MARUBOZU_BULL"]
                bear_patterns = ["KANGAROO_BEAR", "PATTERN_ENGULFING_BEAR", "PATTERN_SHOOTING_STAR", "PATTERN_EVENING_STAR", "PATTERN_DARK_CLOUD", "PATTERN_MARUBOZU_BEAR"]
                
                has_bull = any(last_row.get(p, 0) == 1 for p in bull_patterns)
                has_bear = any(last_row.get(p, 0) == 1 for p in bear_patterns)
                
                if has_bull: score += 10
                if has_bear: score -= 15
                
                # Puanı 0-100 arasına sabitle
                score = max(0, min(100, score))
                
                # Tablo Görselleştirme Hazırlığı
                return {
                    "Sembol": coin,
                    "Fiyat": f"{last_row['Close']:.4f}",
                    "Skor (100)": score,
                    "Durum": "Güçlü Al 🚀" if score >= 80 else ("Al 🟢" if score >= 60 else ("Nötr ⚪" if score >= 40 else "Sat 🔴")),
                    "RSI": round(rsi_val, 2),
                    "MACD Hist": "Pozitif 🟢" if last_row["MACD_hist"] > 0 else "Negatif 🔴",
                    "ADX Trend": "Güçlü Boğa 🟢" if (last_row["ADX"] > 25 and last_row["PLUS_DI"] > last_row["MINUS_DI"]) else ("Güçlü Ayı 🔴" if last_row["ADX"] > 25 else "Zayıf/Yatay ⚪"),
                    "StochRSI": "Alışta 🟢" if (last_row["STOCH_K"] > last_row["STOCH_D"] and last_row["STOCH_K"] < 80) else "Satış/Aşırı 🔴",
                    "BB Genişlik": f"%{last_row['BB_WIDTH']*100:.1f}",
                    "Hacim": "Yüksek 🟢" if last_row["Volume"] > last_row["VOL_SMA"] else "Düşük 🔴",
                    "OBV": "Pozitif 🟢" if last_row["OBV"] > last_row["OBV_EMA"] else "Negatif 🔴",
                    "Formasyon": "Boğa 🟢" if has_bull else ("Ayı 🔴" if has_bear else "-")
                }
            except Exception as e:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in all_coins}
            
            for i, future in enumerate(as_completed(future_to_coin)):
                coin_name = future_to_coin[future]
                res = future.result()
                if res: results.append(res)
                
                progress_bar.progress((i + 1) / len(all_coins))
                status.text(f"Taranıyor: {coin_name}")
                time.sleep(0.02) 
        
        status.empty()
        if results:
            df_results = pd.DataFrame(results)
            # Skora göre büyükten küçüğe sırala
            df_results = df_results.sort_values(by="Skor (100)", ascending=False).reset_index(drop=True)
            
            st.success(f"✅ {len(results)} coin başarıyla tarandı ve puanlandı!")
            st.dataframe(df_results, use_container_width=True, height=600)
        else:
            st.error("Tarama başarısız oldu. Rate limit'e takılmış olabilirsiniz, 1 dakika bekleyip tekrar deneyin.")
