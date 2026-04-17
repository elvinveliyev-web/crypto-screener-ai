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

st.set_page_config(page_title="Crypto Master 5 AI Pro | FA→TA Trading", layout="wide")

# =============================
# BASE DIR & BORSA MOTORU (ccxt)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# KuCoin kullanarak kısıtlamaları aşıyoruz
exchange = ccxt.kucoin({'enableRateLimit': True})

def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)

# =============================
# Universe Loader (Kripto için)
# =============================
@st.cache_data(ttl=3600, show_spinner=False)
def get_crypto_universe() -> List[str]:
    try:
        tickers = exchange.fetch_tickers()
        symbols = [s for s, t in tickers.items() if s.endswith('/USDT') and ':' not in s]
        prio = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "BNB/USDT", "XRP/USDT"]
        return prio + sorted([p for p in symbols if p not in prio])
    except Exception:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

# =============================
# Helpers
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
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

# =============================
# Indicators (BİREBİR AYNI)
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
        if p2 < p1 and i2 > i1:
            return True, bars_ago
    except Exception:
        pass
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
    tr_smooth = pd.Series(tr, index=high.index).ewm(alpha=1/period, adjust=False).mean()
    pdm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    mdm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * (pdm_smooth / tr_smooth.replace(0, np.nan))
    mdi = 100 * (mdm_smooth / tr_smooth.replace(0, np.nan))
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)

# =============================
# KANGAROO TAIL (KANGURU KUYRUĞU)
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
    O = df["Open"]; H = df["High"]; L = df["Low"]; C = df["Close"]
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

    prev_is_bear = is_bear.shift(1); prev_is_bull = is_bull.shift(1)
    prev_O = O.shift(1); prev_C = C.shift(1)

    df["PATTERN_ENGULFING_BULL"] = is_bull & prev_is_bear & (O <= prev_C) & (C >= prev_O) & (Body > (prev_O - prev_C))
    df["PATTERN_ENGULFING_BEAR"] = is_bear & prev_is_bull & (O >= prev_C) & (C <= prev_O) & (Body > (prev_C - prev_O))

    df["PATTERN_HARAMI_BULL"] = is_bull & prev_is_bear & (O > prev_C) & (C < prev_O) & ((prev_O - prev_C) > AvgRange * 0.5)
    df["PATTERN_HARAMI_BEAR"] = is_bear & prev_is_bull & (O < prev_C) & (C > prev_O) & ((prev_C - prev_O) > AvgRange * 0.5)

    prev_H = H.shift(1); prev_L = L.shift(1)
    df["PATTERN_TWEEZER_TOP"] = (abs(H - prev_H) <= 0.002 * C) & is_bear & prev_is_bull & (H > df.get("EMA50", C))
    df["PATTERN_TWEEZER_BOTTOM"] = (abs(L - prev_L) <= 0.002 * C) & is_bull & prev_is_bear & (L < df.get("EMA50", C))

    df["PATTERN_PIERCING"] = is_bull & prev_is_bear & (O < L.shift(1)) & (C > (prev_O + prev_C)/2) & (C < prev_O)
    df["PATTERN_DARK_CLOUD"] = is_bear & prev_is_bull & (O > H.shift(1)) & (C < (prev_O + prev_C)/2) & (C > prev_O)

    prev2_is_bear = is_bear.shift(2); prev2_is_bull = is_bull.shift(2)
    prev2_O = O.shift(2); prev2_C = C.shift(2)

    df["PATTERN_MORNING_STAR"] = is_bull & prev2_is_bear & (prev_C < prev2_C) & (O > prev_C) & (C > (prev2_O + prev2_C)/2)
    df["PATTERN_EVENING_STAR"] = is_bear & prev2_is_bull & (prev_C > prev2_C) & (O < prev_C) & (C < (prev2_O + prev2_C)/2)

    return df

# =============================
# Overbought / Speculation Indicators
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
# Market regime filters (KRİPTO İÇİN)
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_btc_regime_series() -> pd.Series:
    try:
        bars = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=300)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["EMA200"] = ema(df["Close"], 200)
        return df["Close"] > df["EMA200"]
    except Exception:
        return pd.Series(dtype=bool)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_higher_tf_trend_series(ticker: str, higher_tf_interval: str = "1w", ema_period: int = 200) -> pd.Series:
    try:
        bars = exchange.fetch_ohlcv(ticker, timeframe=higher_tf_interval, limit=min(ema_period + 50, 1000))
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
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
    score = (w["liq"] * liq_ok.astype(int) + w["trend"] * trend_ok.astype(int) + w["rsi"] * rsi_ok.astype(int) +
             w["macd"] * macd_ok.astype(int) + w["vol"] * vol_ok.astype(int) + w["bb"] * (bb_ok | bb_break).astype(int) +
             w["obv"] * obv_ok.astype(int)).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 1
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & aligned_market & aligned_htf

    exit_ = ((df["Close"] < df["EMA50"]) | (df["MACD_hist"] < 0) | (df["RSI"] < cfg["rsi_exit_level"]) | (df["Close"] < df["BB_mid"]))

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

# =============================
# Backtest (long-only) + advanced exits
# =============================
def backtest_long_only(df: pd.DataFrame, cfg: dict, risk_free_annual: float, benchmark_returns: Optional[pd.Series] = None):
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
                
                potential_shares = risk_amount / stop_dist
                max_shares = cash / (price * (1 + slippage + commission))
                shares_to_buy = min(potential_shares, max_shares)
                
                if shares_to_buy > 0.001: 
                    shares = shares_to_buy
                    entry_price = price * (1 + slippage)
                    fee = (shares * entry_price) * commission
                    cash -= ((shares * entry_price) + fee)
                    stop = stop_price  
                    target_price = entry_price + (tp_mult * stop_dist)
                    trades.append({
                        "entry_date": date, 
                        "entry_price": entry_price, 
                        "equity_before": cash + (shares * price)
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

                if stop_hit: reason = "STOP"
                elif time_stop_hit: reason = "TIME_STOP"
                else: reason = "RULE_EXIT"

                trades[-1]["exit_reason"] = reason
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0; stop = np.nan; entry_price = np.nan; target_price = np.nan; bars_held = 0; half_sold = False

        position_value = shares * price * (1 - slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0
    rf_daily = (1 + float(risk_free_annual)) ** (1 / 252) - 1
    excess = ret - rf_daily

    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * 252) / downside_dev) if downside_dev > 0 else 0.0

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
            mean_r = r_aligned.mean() * 252
            mean_b = b_aligned.mean() * 252
            alpha = (mean_r - risk_free_annual) - beta * (mean_b - risk_free_annual)
            diff = r_aligned - b_aligned
            info_ratio = (diff.mean() * 252) / (diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0
        else:
            beta = 1.0; alpha = 0.0; info_ratio = 0.0
    else:
        beta = 1.0; alpha = 0.0; info_ratio = 0.0

    peak = eq.cummax()
    drawdown_pct = (eq - peak) / peak
    ulcer_index = np.sqrt((drawdown_pct**2).mean()) if len(drawdown_pct) > 0 else 0.0

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        if "pnl" not in tdf.columns: tdf["pnl"] = np.nan
        if "exit_date" not in tdf.columns: tdf["exit_date"] = pd.NaT
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["return_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["holding_days"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    profit_factor = 0.0
    if not tdf.empty and "pnl" in tdf.columns:
        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0: profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0: profit_factor = float("inf")

    if not tdf.empty and len(tdf) > 5 and "pnl" in tdf.columns:
        win_rate = (tdf["pnl"] > 0).mean()
        avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if win_rate > 0 else 0
        avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if win_rate < 1 else 0
        if avg_loss > 0 and win_rate > 0 and win_rate < 1:
            b = avg_win / avg_loss
            p = win_rate
            kelly = (p * b - (1 - p)) / b
            kelly = max(0, min(kelly, 0.10))
        else: kelly = 0.0
    else: kelly = 0.0

    metrics = {
        "Total Return": float(total_return), "Annualized Return": float(ann_return), "Annualized Volatility": float(ann_vol),
        "Sharpe": float(sharpe), "Sortino": float(sortino), "Calmar": float(calmar), "Max Drawdown": float(mdd),
        "Trades": int(len(tdf)) if not tdf.empty else 0, "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty and "pnl" in tdf.columns else 0.0,
        "Profit Factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "Beta": float(beta), "Alpha": float(alpha), "Information Ratio": float(info_ratio),
        "Ulcer Index": float(ulcer_index), "Kelly % (öneri)": float(kelly * 100),
    }
    return eq, tdf, metrics

# =============================
# Target price band / SR Levels (BİREBİR AYNI)
# =============================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs = []; ls = []; n = len(high)
    for i in range(left, n - right):
        hwin = high.iloc[i - left : i + right + 1]
        lwin = low.iloc[i - left : i + right + 1]
        if high.iloc[i] == hwin.max(): hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == lwin.min(): ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls

def analyze_sr_levels(df: pd.DataFrame, lookback: int = 200, tol=0.02) -> List[dict]:
    h = df["High"].tail(lookback).dropna(); l = df["Low"].tail(lookback).dropna(); c = df["Close"].tail(lookback).dropna()
    if len(c) < 10: return []
    v = df["Volume"].tail(lookback) if "Volume" in df.columns else pd.Series(dtype=float)
    hs, ls = _swing_points(h, l, left=3, right=3)
    raw_levels = [val for _, val in hs] + [val for _, val in ls] + [float(c.tail(20).max()), float(c.tail(20).min())]
    raw_levels = sorted(list(set([round(float(x), 2) for x in raw_levels if np.isfinite(x)])))
    if not raw_levels: return []

    clusters = []
    for rl in raw_levels:
        placed = False
        for cl in clusters:
            if abs(rl - cl['center']) / cl['center'] <= tol:
                cl['points'].append(rl); placed = True; break
        if not placed: clusters.append({'center': rl, 'points': [rl]})

    avg_vol_normal = float(v.mean()) if not v.empty else 1.0
    if avg_vol_normal <= 0: avg_vol_normal = 1.0

    details = []
    df_lookback = df.tail(lookback)
    for cl in clusters:
        level_px = cl['center']
        lower_bound = level_px * (1 - tol/2)
        upper_bound = level_px * (1 + tol/2)
        touches = df_lookback[(df_lookback["High"] >= lower_bound) & (df_lookback["Low"] <= upper_bound)]
        num_touches = len(touches)
        if num_touches == 0: continue
        first_touch_idx = touches.index[0]
        first_idx_num = df_lookback.index.get_loc(first_touch_idx)
        duration_bars = len(df_lookback) - first_idx_num
        if "Volume" in df_lookback.columns and not touches.empty: vol_at_level = float(touches["Volume"].mean())
        else: vol_at_level = avg_vol_normal
        vol_diff_pct = (vol_at_level / avg_vol_normal - 1.0) * 100.0
        score_touches = min(num_touches * 10, 40) 
        score_vol = min(max(vol_diff_pct / 2.0, 0), 35) 
        score_dur = min(duration_bars / 2.0, 25) 
        strength_pct = min(score_touches + score_vol + score_dur, 99.0)
        details.append({"price": round(level_px, 2), "duration_bars": int(duration_bars), "vol_at_level": float(vol_at_level), "vol_diff_pct": float(vol_diff_pct), "strength_pct": float(strength_pct), "touches": int(num_touches)})

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

    return {"base": px_close, "bull": (bull1, bull2, r1), "bear": (bear1, bear2, s1), "levels": lv_details, "r1_dict": r1_dict, "s1_dict": s1_dict}


# =============================
# Live price & Short Info (KRİPTO UYARLI)
# =============================
@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker: str) -> dict:
    out = {"last_price": np.nan, "currency": "USDT", "exchange": "KuCoin", "asof": ""}
    try:
        ticker_data = exchange.fetch_ticker(ticker)
        out["last_price"] = safe_float(ticker_data['last'])
        out["asof"] = str(datetime.datetime.now())
    except Exception:
        pass
    return out

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_short_info(ticker: str) -> dict:
    return {"short_percent_float": np.nan, "short_ratio": np.nan}


# =============================
# Gemini helpers
# =============================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        if v is None: return default
        return str(v).strip()
    except Exception:
        return default

def _http_post_json(url: str, payload: dict, headers: dict = None, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try: data = r.json()
    except Exception: data = {"error": {"message": f"Non-JSON response (status={r.status_code})", "text": r.text[:500]}}
    if r.status_code >= 400:
        if "error" not in data: data["error"] = {"message": f"HTTP {r.status_code}", "text": str(data)[:500]}
    return data

def _extract_gemini_text(resp: dict) -> str:
    if not isinstance(resp, dict): return str(resp)
    if resp.get("error"): return f"Gemini API error: {resp['error'].get('message','')}"
    cands = resp.get("candidates") or []
    if not cands: return "Gemini: boş cevap döndü."
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts: return "Gemini: boş cevap döndü."
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p: texts.append(p["text"])
    return "\n".join(texts).strip() if texts else "Gemini: metin üretmedi."

def gemini_generate_text(*, prompt: str, model: str = "gemini-1.5-flash", temperature: float = 0.2, max_output_tokens: int = 2048, image_bytes: Optional[bytes] = None) -> str:
    api_key = _get_secret("GEMINI_API_KEY", "")
    if not api_key: return "GEMINI_API_KEY bulunamadı. Streamlit Cloud > Settings > Secrets içine GEMINI_API_KEY=... ekleyin."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}
    parts = [{"text": prompt}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64_img}})
    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)}}
    resp = _http_post_json(url, payload, headers=headers, timeout=90)
    return _extract_gemini_text(resp)

# =============================
# Sentiment Analysis (Haber Botu)
# =============================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_news_sentiment(ticker: str, gemini_model: str = "gemini-1.5-flash", gemini_temp: float = 0.2, max_tokens: int = 2048) -> Dict[str, Any]:
    try:
        # Stock yerine Crypto aratıyoruz
        query = f"{ticker.replace('/USDT', '')} crypto"
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200: return {"error": f"Haberler çekilemedi (HTTP {resp.status_code})", "sentiment": None, "summary": ""}
            
        root = ET.fromstring(resp.content)
        news_items = []
        for item in root.findall(".//item")[:10]:
            title_node = item.find("title")
            link_node = item.find("link")
            if title_node is not None and title_node.text:
                t = title_node.text
                l = link_node.text if (link_node is not None and link_node.text) else ""
                news_items.append({"title": t, "link": l})

        if not news_items: return {"error": "Haber bulunamadı", "sentiment": None, "summary": ""}

        prompt_titles = [item["title"] for item in news_items]
        prompt = f"""Aşağıdaki kripto para haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
Pozitif: [sayı]
Negatif: [sayı]
Nötr: [sayı]
- Bileşik skor: (pozitif - negatif) / toplam (örneğin 0.35)
- Kısa bir özet (2 cümle)

Haber Başlıkları:
{chr(10).join([f"- {t}" for t in prompt_titles])}
"""
        response = gemini_generate_text(prompt=prompt, model=gemini_model, temperature=gemini_temp, max_output_tokens=max_tokens, image_bytes=None)

        pos_match = re.search(r"Pozitif:?\s*(\d+)", response, re.IGNORECASE)
        neg_match = re.search(r"Negatif:?\s*(\d+)", response, re.IGNORECASE)
        neu_match = re.search(r"Nötr:?\s*(\d+)", response, re.IGNORECASE)

        pos = int(pos_match.group(1)) if pos_match else 0
        neg = int(neg_match.group(1)) if neg_match else 0
        neu = int(neu_match.group(1)) if neu_match else 0
        total = pos + neg + neu
        compound = (pos - neg) / total if total > 0 else 0

        return {"error": None, "sentiment": compound, "summary": response, "pos": pos / total if total > 0 else 0, "neg": neg / total if total > 0 else 0, "neu": neu / total if total > 0 else 0, "news_items": news_items[:5]}
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": ""}

# =============================
# Price Action Pack
# =============================
def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if use.empty or len(use) < 10: return {"note": "insufficient_bars", "last_n": int(len(use))}

    o = use["Open"].astype(float); h = use["High"].astype(float); l = use["Low"].astype(float); c = use["Close"].astype(float)
    swing_highs, swing_lows = _swing_points(h, l, left=2, right=2)

    q20 = float(np.quantile(c.values, 0.20)); q50 = float(np.quantile(c.values, 0.50)); q80 = float(np.quantile(c.values, 0.80))
    recent_highs = [v for _, v in swing_highs[-5:]] if swing_highs else []
    recent_lows = [v for _, v in swing_lows[-5:]] if swing_lows else []
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())

    last_close = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
    last_high = float(h.iloc[-1]); last_low = float(l.iloc[-1])

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

    pack = {"last_n": int(len(use)), "q20": q20, "q50": q50, "q80": q80, "support": sup, "resistance": res, "bull_breakout": bool(bull_break), "bear_breakout": bool(bear_break), "vol_confirm": (None if vol_ok is None else bool(vol_ok)), "last_bar": {"t": str(use.index[-1]), "open": float(o.iloc[-1]), "high": last_high, "low": last_low, "close": last_close}, "swing_highs": [{"t": str(t), "p": float(p)} for t, p in swing_highs[-6:]], "swing_lows": [{"t": str(t), "p": float(p)} for t, p in swing_lows[-6:]], "order_block_proxy": ob}
    return pack

def df_snapshot_for_llm(df: pd.DataFrame, n: int = 25) -> dict:
    use_cols = ["Open", "High", "Low", "Close", "Volume", "EMA50", "EMA200", "RSI", "MACD", "MACD_signal", "MACD_hist", "BB_mid", "BB_upper", "BB_lower", "ATR", "ATR_PCT", "VOL_SMA", "VOL_RATIO", "BB_WIDTH", "SCORE", "ENTRY", "EXIT", "RSI_OVERBOUGHT", "BB_OVERBOUGHT", "BB_OVERSOLD", "VOLUME_SPIKE", "PRICE_EXTREME", "STOCH_OVERBOUGHT", "WEAK_UPTREND", "KANGAROO_BULL", "KANGAROO_BEAR"]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    summary = {}
    if not df.empty:
        summary["rsi_last"] = float(df["RSI"].iloc[-1]) if "RSI" in df else None
        summary["rsi_5d_avg"] = float(df["RSI"].tail(5).mean()) if "RSI" in df else None
        if "EMA50" in df and "EMA200" in df: summary["trend"] = "up" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "down"

    return {"cols": cols, "n": int(len(tail)), "last_index": str(tail.index[-1]) if len(tail) else None, "rows": tail.to_dict(orient="records"), "summary": summary}

# =============================
# Presets
# =============================
PRESETS = {
    "Defansif": {"rsi_entry_level": 52, "rsi_exit_level": 46, "atr_pct_max": 0.06, "atr_stop_mult": 2.0, "time_stop_bars": 15, "take_profit_mult": 2.5},
    "Dengeli": {"rsi_entry_level": 50, "rsi_exit_level": 45, "atr_pct_max": 0.08, "atr_stop_mult": 1.5, "time_stop_bars": 10, "take_profit_mult": 2.0},
    "Agresif": {"rsi_entry_level": 48, "rsi_exit_level": 43, "atr_pct_max": 0.10, "atr_stop_mult": 1.2, "time_stop_bars": 7, "take_profit_mult": 1.5},
}

# =============================
# REPORT EXPORT (BİREBİR AYNI)
# =============================
def build_html_report(title: str, meta: dict, checkpoints: dict, metrics: dict, tp: dict, rr_info: dict, figs: Dict[str, go.Figure], gemini_insight: Optional[str] = None, pa_pack: Optional[Dict[str, Any]] = None, sentiment_summary: Optional[str] = None, sentiment_items: Optional[List[dict]] = None, overbought_result: Optional[Dict[str, Any]] = None) -> bytes:
    def esc(x): return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    fig_blocks = []
    first = True
    for name, fig in (figs or {}).items():
        fig_html = fig.to_html(full_html=False, include_plotlyjs=("cdn" if first else False))
        first = False
        fig_blocks.append(f"<h3>{esc(name)}</h3>{fig_html}")

    cp_list = "".join([f"<li>{'✅' if v else '❌'} {esc(k)}</li>" for k, v in checkpoints.items()])

    bull = tp.get("bull")
    bear = tp.get("bear")
    levels = tp.get("levels", []) or []
    levels_txt = "<br>".join([f"{x['price']:.2f} (Güç: %{x['strength_pct']:.0f}, Uzunluk: {x['duration_bars']} Bar, Hacim: %{x['vol_diff_pct']:+.1f})" for x in levels[:120]]) if levels else "N/A"

    overbought_html = ""
    if overbought_result:
        ob_details = "<ul>"
        for _, v in overbought_result.get("details", {}).items(): ob_details += f"<li>{esc(v)}</li>"
        ob_details += "</ul>"
        overbought_html = f"""
        <div class="card" style="margin-top:16px;">
            <h2>📊 Aşırı Alım / Spekülasyon Analizi</h2>
            <div><b>Karar:</b> {esc(overbought_result['verdict'])}</div>
            <div><b>Aşırı Alım Skoru:</b> {overbought_result['overbought_score']}/100</div>
            <div><b>Aşırı Satım Skoru:</b> {overbought_result['oversold_score']}/100</div>
            <div><b>Spekülasyon Skoru:</b> {overbought_result['speculation_score']}/100</div>
            <div><b>Detaylar:</b> {ob_details}</div>
        </div>
        """

    gemini_block = ""
    if gemini_insight:
        gemini_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Gemini — Chart & Price Action Insight</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(gemini_insight)}</pre>
        </div>
        """

    pa_block = ""
    if pa_pack:
        pa_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Price Action Pack (Last {esc(pa_pack.get('last_n',''))} Bars)</h2>
            <pre style="white-space:pre-wrap; font-family:monospace; font-size:12px;">{esc(json.dumps(pa_pack, ensure_ascii=False, indent=2))}</pre>
        </div>
        """

    sentiment_block = ""
    if sentiment_summary:
        links_html = ""
        if sentiment_items:
            links_html = "<br><br><b>Kaynak Haberler:</b><ul>"
            for item in sentiment_items: links_html += f"<li><a href='{esc(item['link'])}' target='_blank'>{esc(item['title'])}</a></li>"
            links_html += "</ul>"
        sentiment_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Haber Duygu Analizi (Google News + Gemini)</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(sentiment_summary)}</pre>
            {links_html}
        </div>
        """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
    h1,h2,h3 {{ margin: 0 0 8px 0; }}
    ul {{ margin: 8px 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid #eee; padding: 6px 8px; vertical-align: top; }}
    @media print {{
      .no-print {{ display: none; }}
      body {{ margin: 10mm; }}
    }}
  </style>
</head>
<body>
  <div class="no-print card" style="background:#fff7e6;border-color:#ffd591;">
    <b>PDF yapmak için:</b> Bu dosyayı indir → tarayıcıda aç → <b>Ctrl+P</b> → <b>Save as PDF</b>.
  </div>
  <h1>{esc(title)}</h1>
  <div class="muted">
    Generated: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}<br>
    Market: Kripto | Ticker: {esc(meta.get('ticker'))} | Interval: {esc(meta.get('interval'))} | Period: {esc(meta.get('period'))}<br>
    Preset: {esc(meta.get('preset'))} | EMA: {esc(meta.get('ema_fast'))}/{esc(meta.get('ema_slow'))} | RSI: {esc(meta.get('rsi_period'))} | BB: {esc(meta.get('bb_period'))}/{esc(meta.get('bb_std'))} | ATR: {esc(meta.get('atr_period'))} | VolSMA: {esc(meta.get('vol_sma'))}
  </div>
  <div class="grid" style="margin-top:14px;">
    <div class="card">
      <h2>Checkpoints</h2>
      <ul>{cp_list}</ul>
    </div>
    <div class="card">
      <h2>Backtest</h2>
      <div>Total Return: {metrics.get('Total Return',0)*100:.1f}%</div>
      <div>Ann Return: {metrics.get('Annualized Return',0)*100:.1f}%</div>
      <div>Sharpe: {metrics.get('Sharpe',0):.2f}</div>
      <div>Max DD: {metrics.get('Max Drawdown',0)*100:.1f}%</div>
      <div>Trades: {metrics.get('Trades',0)}</div>
      <div>Win Rate: {metrics.get('Win Rate',0)*100:.1f}%</div>
      <div>Kelly Önerisi: {metrics.get('Kelly % (öneri)',0):.1f}%</div>
    </div>
  </div>
  {overbought_html}
  <div class="card" style="margin-top:16px;">
    <h2>Target Band</h2>
    <div>Base: {tp.get('base',0):.2f}</div>
    <div>Bull: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {(bull[2] if bull else 'N/A')}</div>
    <div>Bear: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {(bear[2] if bear else 'N/A')}</div>
    <div>RR: {('N/A' if rr_info.get('rr') is None else f"1:{rr_info.get('rr'):.2f}")}</div>
    <div class="muted"><br>Seviyeler ve Güçleri:<br>{levels_txt}</div>
  </div>
  {gemini_block}
  {sentiment_block}
  {pa_block}
  <div style="margin-top:18px;">
    {''.join(fig_blocks)}
  </div>
</body>
</html>
"""
    return html.encode("utf-8")

def _plotly_fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    try: return fig.to_image(format="png", scale=2)
    except Exception: return None

def _pdf_write_lines(c, lines: List[str], x: float, y: float, lh: float, bottom: float):
    for line in lines:
        if y <= bottom:
            c.showPage()
            y = A4[1] - 2.0 * cm
        c.drawString(x, y, (line or "")[:220])
        y -= lh
    return y

def generate_pdf_report(*, title: str, subtitle: str, meta: dict, checkpoints: dict, ta_summary: dict, target_band: dict, rr_info: dict, backtest_metrics: dict, levels: Optional[List[dict]], trades_df: Optional[pd.DataFrame], figs: Optional[Dict[str, go.Figure]], include_charts: bool = True, gemini_insight: Optional[str] = None, pa_pack: Optional[Dict[str, Any]] = None, sentiment_summary: Optional[str] = None, sentiment_items: Optional[List[dict]] = None, overbought_result: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
    if not REPORTLAB_OK: return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    left = 1.6 * cm; right = W - 1.6 * cm; top = H - 1.6 * cm; bottom = 1.6 * cm; y = top

    c.setFont("Helvetica-Bold", 16); c.drawString(left, y, title[:90]); y -= 18
    c.setFont("Helvetica", 10); c.drawString(left, y, subtitle[:140]); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(c, [
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Market: Crypto | Ticker: {meta.get('ticker','')} | Interval: {meta.get('interval','')} | Period: {meta.get('period','')}",
        ], left, y, 12, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Technical Summary"); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(c, [
            f"Recommendation: {ta_summary.get('rec','')}",
            f"Last Close: {ta_summary.get('close','N/A')} | Live: {ta_summary.get('live','N/A')}",
            f"Score: {ta_summary.get('score','N/A')} | RSI: {ta_summary.get('rsi','N/A')} | EMA50: {ta_summary.get('ema50','N/A')}"
        ], left, y, 12, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Checkpoints"); y -= 14
    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(c, [f"[{'OK' if v else 'NO'}] {k}" for k, v in checkpoints.items()], left, y, 11, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Backtest Summary"); y -= 14
    c.setFont("Helvetica", 9)
    bm = backtest_metrics or {}
    y = _pdf_write_lines(c, [
            f"Total Return: {fmt_pct(bm.get('Total Return'))} | Ann Return: {fmt_pct(bm.get('Annualized Return'))}",
            f"Max DD: {fmt_pct(bm.get('Max Drawdown'))} | Win Rate: {fmt_pct(bm.get('Win Rate'))}",
        ], left, y, 12, bottom)
    y -= 6

    if sentiment_summary:
        c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Haber Duygu Analizi"); y -= 14
        c.setFont("Helvetica", 9)
        y = _pdf_write_lines(c, sentiment_summary.splitlines(), left, y, 11, bottom)
        y -= 6

    if include_charts and figs:
        for name, fig in figs.items():
            img = _plotly_fig_to_png_bytes(fig)
            if not img: continue
            c.showPage()
            c.setFont("Helvetica-Bold", 14); c.drawString(left, top, f"Chart: {name}")
            img_reader = ImageReader(BytesIO(img))
            c.drawImage(img_reader, left, 2.0 * cm, width=(right-left), height=(H-5.2*cm), preserveAspectRatio=True, anchor="c")

    c.save(); buf.seek(0)
    return buf.read()

# =============================
# RR helper 
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"])
    atrv = float(latest_row.get("ATR", np.nan)) if pd.notna(latest_row.get("ATR", np.nan)) else np.nan
    if not np.isfinite(atrv) or atrv <= 0: return {"rr": None, "stop": None, "risk": None, "reward": None}

    if latest_row.get("KANGAROO_BULL", 0) == 1: stop = float(latest_row["Low"]) - (0.5 * atrv)
    else: stop = close - (float(cfg["atr_stop_mult"]) * atrv)
        
    risk = close - stop
    r1 = None
    if tp_dict and tp_dict.get("bull"): r1 = tp_dict["bull"][2] 
        
    if r1 is not None and np.isfinite(r1) and r1 > close: target = float(r1); target_type = "Resistance (R1)"
    else:
        tp_mult = cfg.get("take_profit_mult", 2.0)
        target = close + (tp_mult * cfg["atr_stop_mult"] * atrv)
        target_type = f"ATR-based Target ({tp_mult}x)"

    reward = target - close
    if risk <= 0 or reward <= 0: return {"rr": None, "stop": stop, "risk": risk, "reward": reward, "target_type": target_type}
    return {"rr": float(reward / risk), "stop": float(stop), "risk": float(risk), "reward": reward, "target_type": target_type}

def fmt_rr(rr):
    if rr is None or (isinstance(rr, float) and (not np.isfinite(rr))): return "N/A"
    return f"1:{rr:.2f}"

def pct_dist(level: float, base: float):
    if level is None or not np.isfinite(level) or base == 0: return None
    return (level / base - 1.0) * 100.0

# =============================
# Cached data loader (KRİPTO UYARLI - ccxt)
# =============================
@st.cache_data(ttl=300, show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str, end_date=None, force_latest: bool = False) -> pd.DataFrame:
    # Timeframe mapping for ccxt
    tf_map = {"1m": "1m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1w"}
    ccxt_tf = tf_map.get(interval, "1d")

    # Limit mapping logic based on period
    limit = 200
    if period == "45d": limit = 45
    elif period == "3mo": limit = 90
    elif period == "6mo": limit = 180
    elif period == "1y": limit = 365
    elif period == "2y": limit = 730
    
    if ccxt_tf == "1h": limit *= 24
    elif ccxt_tf == "4h": limit *= 6
    elif ccxt_tf == "1w": limit = max(limit // 7, 100)
    
    limit = min(limit, 1500) # KuCoin limit safe bound

    try:
        # ccxt ohlcv fetch
        bars = exchange.fetch_ohlcv(ticker, timeframe=ccxt_tf, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# =============================
# UI STATE
# =============================
st.title("📈 Kripto Master 5 AI Pro | FA→TA Trading")
st.caption("Efsanevi hisse senedi stratejisinin Kripto versiyonu. Otomatik emir göndermez.")

if "app_errors" not in st.session_state: st.session_state.app_errors = []
if "ta_ran" not in st.session_state: st.session_state.ta_ran = False
if "gemini_text" not in st.session_state: st.session_state.gemini_text = ""
if "pa_pack" not in st.session_state: st.session_state.pa_pack = {}
if "sentiment_summary" not in st.session_state: st.session_state.sentiment_summary = ""
if "sentiment_items" not in st.session_state: st.session_state.sentiment_items = []
if "show_ema13_channel" not in st.session_state: st.session_state.show_ema13_channel = False

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/cryptocurrency.png", width=80)
    st.header("1) Piyasa & Sembol")
    ticker = st.selectbox("Kripto Parite Seçin", get_crypto_universe(), index=0)

    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox(
        "Teknik Mod",
        list(PRESETS.keys()),
        index=1,
        help="Önceden tanımlı risk profilleri. Defansif: düşük risk, Agresif: yüksek risk.",
    )

    st.subheader("Zaman Aralığı")
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "4h", "1h"],
        index=0,
        help="Mum zaman dilimi. 1d günlük, 1wk haftalık, 4h 4 saatlik, 1h saatlik analiz için. Backtest için 1d önerilir.",
    )
    period = st.selectbox("Periyot", ["45d", "3mo", "6mo", "1y", "2y"], index=3)

    st.divider()
    st.subheader("Teknik Parametreler")
    ema_fast = st.number_input("EMA Fast", min_value=5, max_value=100, value=50, step=1)
    ema_slow = st.number_input("EMA Slow", min_value=50, max_value=400, value=200, step=1)
    rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, step=1)
    bb_period = st.number_input("Bollinger Period", min_value=10, max_value=50, value=20, step=1)
    bb_std = st.number_input("Bollinger Std", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
    atr_period = st.number_input("ATR Period", min_value=5, max_value=30, value=14, step=1)
    vol_sma = st.number_input("Volume SMA", min_value=5, max_value=60, value=20, step=1)

    st.subheader("Market Filtreleri")
    use_btc_filter = st.checkbox(
        "BTC > EMA200 filtresi",
        value=True,
        help="Bitcoin 200 günlük ortalamanın altındaysa (ayı piyasası) alım sinyallerini engeller.",
    )
    use_higher_tf_filter = st.checkbox(
        "Haftalık trend filtresi (Fiyat > EMA200)",
        value=True,
        help="Haftalık grafikte fiyatın 200 haftalık ortalamanın üzerinde olması gerekir.",
    )

    st.subheader("Risk / Backtest Ayarları")
    initial_capital = st.number_input("Başlangıç Sermayesi ($)", min_value=100.0, value=1000.0, step=100.0)
    risk_per_trade = st.slider("Trade başı risk (equity %)", 0.002, 0.05, 0.01, 0.001)
    commission_bps = st.number_input("Komisyon (bps)", min_value=0.0, value=10.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)

    st.divider()
    st.header("3) AI Ayarları (Gemini)")
    ai_on = st.checkbox("Gemini AI aktif", value=True)
    gemini_model = st.text_input("Gemini Model", value="gemini-1.5-flash")
    gemini_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    gemini_max_tokens = st.slider("Max Output Tokens", 256, 8192, 2048, 128)

    st.divider()
    st.header("4) Haber Duygu Analizi")
    use_sentiment = st.checkbox("Haber duygu analizini aktifleştir", value=True)

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn:
        st.session_state.ta_ran = True

# -----------------------------
# Config
# -----------------------------
cfg = {
    "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_period": rsi_period, "bb_period": bb_period,
    "bb_std": bb_std, "atr_period": atr_period, "vol_sma": vol_sma, "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade, "commission_bps": commission_bps, "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])


# -----------------------------
# If TA not ran yet: show message and stop
# -----------------------------
if not st.session_state.ta_ran:
    st.info("Sol menüden Kripto Sembolünü seçip **Teknik Analizi Çalıştır**’a basın.")
    st.stop()


# =============================
# Run TA pipeline
# =============================
market_filter_series = None
if use_btc_filter:
    with st.spinner("BTC rejimi kontrol ediliyor..."):
        market_filter_series = get_btc_regime_series()

higher_tf_filter_series = None
if use_higher_tf_filter:
    with st.spinner("Haftalık trend kontrol ediliyor..."):
        higher_tf_filter_series = get_higher_tf_trend_series(ticker, higher_tf_interval="1w", ema_period=200)

sentiment_summary = ""
if use_sentiment and ai_on:
    with st.spinner("Google News'ten kripto haberleri çekiliyor ve Gemini ile analiz ediliyor..."):
        sent = get_news_sentiment(ticker, gemini_model, gemini_temp, gemini_max_tokens)
        if sent.get("error") is None:
            sentiment_summary = sent["summary"]
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = sent.get("news_items", [])
        else:
            sentiment_summary = f"Haber analizi başarısız: {sent['error']}"
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = []
elif use_sentiment and not ai_on:
    st.warning("Haber duygu analizi için Gemini'nin açık olması gerekir.")


with st.spinner(f"Veri indiriliyor: {ticker}"):
    df_raw = load_data_cached(ticker, period, interval)

if df_raw.empty:
    st.error(f"Veri gelmedi: {ticker}")
    st.stop()

required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(df_raw.columns)):
    st.error("Veri setinde gerekli OHLCV kolonları eksik.")
    st.stop()

df = build_features(df_raw, cfg)

# Benchmark returns (BTC/USDT)
benchmark_df = load_data_cached("BTC/USDT", period, interval)
benchmark_returns = benchmark_df["Close"].pct_change().dropna() if not benchmark_df.empty else None

df, checkpoints = signal_with_checkpoints(df, cfg, market_filter_series=market_filter_series, higher_tf_filter_series=higher_tf_filter_series)
latest = df.iloc[-1]

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

if int(latest["ENTRY"]) == 1: rec = "AL"
elif int(latest["EXIT"]) == 1: rec = "SAT"
else: rec = "AL (Güçlü Trend)" if latest["SCORE"] >= 80 else ("İZLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=0.0, benchmark_returns=benchmark_returns)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)

df["EMA13_High"] = ema(df["High"], 13)
df["EMA13_Low"] = ema(df["Low"], 13)
df["EMA13_Close"] = ema(df["Close"], 13)

# =============================
# Build figures
# =============================
if "show_chart_patterns" not in st.session_state:
    st.session_state.show_chart_patterns = True

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

if st.session_state.show_ema13_channel:
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_High"], name="13 EMA High", line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Low"], name="13 EMA Low", fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Close"], name="13 EMA Close", line=dict(color='darkorange', width=2)))

entries = df[df["ENTRY"] == 1]; exits = df[df["EXIT"] == 1]
fig_price.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=10)))
fig_price.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=10)))

if st.session_state.show_chart_patterns:
    bull_patterns = {
        "KANGAROO_BULL": "🟩🦘 LONG KANGURU", "PATTERN_HAMMER": "🟩🔨 HAMMER", "PATTERN_INV_HAMMER": "🟩🔨 INV HAMMER",
        "PATTERN_ENGULFING_BULL": "🟢 ENGULFING", "PATTERN_HARAMI_BULL": "🟢🤰 HARAMI", "PATTERN_MARUBOZU_BULL": "🟩 MARUBOZU",
        "PATTERN_TWEEZER_BOTTOM": "🟢✌️ TWEEZER", "PATTERN_PIERCING": "🟢🗡️ PIERCING", "PATTERN_MORNING_STAR": "🟢🌅 M.STAR", "PATTERN_LL_DOJI": "🟢⚖️ LL DOJI",
    }
    bear_patterns = {
        "KANGAROO_BEAR": "🟥🦘 SHORT KANGURU", "PATTERN_HANGING_MAN": "🟥🪢 HANGING M.", "PATTERN_SHOOTING_STAR": "🟥🌠 S.STAR",
        "PATTERN_ENGULFING_BEAR": "🔴 ENGULFING", "PATTERN_HARAMI_BEAR": "🔴🤰 HARAMI", "PATTERN_MARUBOZU_BEAR": "🟥 MARUBOZU",
        "PATTERN_TWEEZER_TOP": "🔴✌️ TWEEZER", "PATTERN_DARK_CLOUD": "🔴🌩️ D.CLOUD", "PATTERN_EVENING_STAR": "🔴🌃 E.STAR"
    }
    bull_texts = pd.Series("", index=df.index); bear_texts = pd.Series("", index=df.index)

    for col, name in bull_patterns.items():
        if col in df.columns: mask = df[col] == 1; bull_texts[mask] += name + "<br>"
    for col, name in bear_patterns.items():
        if col in df.columns: mask = df[col] == 1; bear_texts[mask] += name + "<br>"

    bull_texts = bull_texts.str.rstrip("<br>"); bear_texts = bear_texts.str.rstrip("<br>")
    bull_mask = bull_texts != ""
    if bull_mask.any():
        fig_price.add_trace(go.Scatter(x=df.index[bull_mask], y=df["Low"][bull_mask], mode="markers+text", name="Boğa Formasyonları", text=bull_texts[bull_mask], textposition="bottom center", textfont=dict(color="green", size=10, family="Arial Black"), marker=dict(symbol="triangle-up", size=10, color="green", line=dict(width=1, color="DarkSlateGrey"))))
    bear_mask = bear_texts != ""
    if bear_mask.any():
        fig_price.add_trace(go.Scatter(x=df.index[bear_mask], y=df["High"][bear_mask], mode="markers+text", name="Ayı Formasyonları", text=bear_texts[bear_mask], textposition="top center", textfont=dict(color="red", size=10, family="Arial Black"), marker=dict(symbol="triangle-down", size=10, color="red", line=dict(width=1, color="DarkSlateGrey"))))

fig_price.update_layout(height=600, xaxis_rangeslider_visible=False, title="Fiyat Grafiği + EMA + Bollinger + Sinyaller & Formasyonlar", yaxis_title="Fiyat", xaxis_title="Tarih", template="plotly_dark")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
fig_rsi.update_layout(height=260, title="RSI", template="plotly_dark")

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(height=260, title="MACD", template="plotly_dark")

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(height=260, title="ATR %", template="plotly_dark")

fig_stoch = go.Figure()
if "STOCH_RSI_K" in df.columns and "STOCH_RSI_D" in df.columns:
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="Stochastic RSI K"))
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="Stochastic RSI D"))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
fig_stoch.update_layout(height=260, title="Stochastic RSI (K & D)", template="plotly_dark")

fig_bbwidth = go.Figure()
if "BB_WIDTH" in df.columns:
    fig_bbwidth.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"] * 100, name="BB % Genişlik"))
fig_bbwidth.update_layout(height=260, title="Bollinger Bandı Genişliği %", template="plotly_dark")

fig_volratio = go.Figure()
if "VOL_RATIO" in df.columns:
    fig_volratio.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı"))
fig_volratio.update_layout(height=260, title="Hacim Oranı", template="plotly_dark")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
fig_eq.update_layout(height=320, title="Backtest Sermaye Eğrisi", template="plotly_dark")

figs_for_report = {
    "Price + EMA + Bollinger + Signals": fig_price, "RSI": fig_rsi, "MACD": fig_macd,
    "ATR%": fig_atr, "Stochastic RSI": fig_stoch, "Bollinger Band Width": fig_bbwidth,
    "Volume Ratio": fig_volratio, "Equity Curve": fig_eq,
}

# =============================
# Tabs (5. Sekme: Kripto Tarayıcı Eklendi)
# =============================
tab_dash, tab_export, tab_heatmap, tab_triple, tab_screener = st.tabs(["📊 Dashboard", "📄 Rapor (PDF/HTML)", "🔥 Kripto Heatmap", "📺 3 Ekranlı Sistem", "🔍 Çoklu Tarayıcı (Screener)"])

with tab_dash:
    st.subheader("📊 Aşırı Alım / Spekülasyon Göstergeleri")
    col_ob1, col_ob2, col_ob3, col_ob4 = st.columns(4)
    col_ob1.metric("Aşırı Alım Skoru", f"{overbought_result['overbought_score']}/100")
    col_ob2.metric("Aşırı Satım Skoru", f"{overbought_result['oversold_score']}/100")
    col_ob3.metric("Spekülasyon Skoru", f"{overbought_result['speculation_score']}/100")
    col_ob4.metric("Genel Karar", overbought_result["verdict"])

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Sembol", ticker)
    c2.metric("Daily Close", f"{latest['Close']:.4f}")
    c3.metric("Live/Last", f"{live_price:.4f}" if np.isfinite(live_price) else "N/A")
    c4.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c5.metric("Sinyal", rec)
    c6.metric("BTC Filtresi", "BULL ✅" if checkpoints.get("Market Filter OK", True) else "BEAR ❌")
    c7.metric("Haftalık Trend", "BULL ✅" if checkpoints.get("Higher TF Filter OK", True) else "BEAR ❌")
    
    st.subheader("🕯️ Fiyat Aksiyonu (Price Action) Mum Formasyonları - Son Bar")
    is_bull_tail = latest.get("KANGAROO_BULL", 0) == 1
    is_bear_tail = latest.get("KANGAROO_BEAR", 0) == 1
    tail_val = "BOĞA 🦘" if is_bull_tail else ("AYI 🦘" if is_bear_tail else "YOK")
    
    pa_c1, pa_c2, pa_c3, pa_c4, pa_c5, pa_c6 = st.columns(6)
    pa_c1.metric("1. Kanguru", tail_val)
    pa_c2.metric("2. Engulfing", "Boğa 🟢" if latest.get("PATTERN_ENGULFING_BULL") else ("Ayı 🔴" if latest.get("PATTERN_ENGULFING_BEAR") else "Yok"))
    pa_c3.metric("3. Hammer / Star", "Çekiç 🟢" if latest.get("PATTERN_HAMMER") else ("Kayan Yıldız 🔴" if latest.get("PATTERN_SHOOTING_STAR") else "Yok"))
    pa_c4.metric("4. Doji", "Uzun Bacak ⚪" if latest.get("PATTERN_LL_DOJI") else ("Doji ⚪" if latest.get("PATTERN_DOJI") else "Yok"))
    pa_c5.metric("5. Marubozu", "Boğa 🟢" if latest.get("PATTERN_MARUBOZU_BULL") else ("Ayı 🔴" if latest.get("PATTERN_MARUBOZU_BEAR") else "Yok"))
    pa_c6.metric("6. Harami", "Boğa 🟢" if latest.get("PATTERN_HARAMI_BULL") else ("Ayı 🔴" if latest.get("PATTERN_HARAMI_BEAR") else "Yok"))

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    for i, (k, v) in enumerate(list(checkpoints.items())):
        with cp_cols[i % 3]:
            st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp["base"])
    rr_str = fmt_rr(rr_info.get("rr"))

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", f"{base_px:.4f}")

    if tp.get("bull"):
        bull1, bull2, r1 = tp["bull"]
        bcol2.metric("Bull Band", f"{bull1:.4f} → {bull2:.4f}")
    else: bcol2.metric("Bull Band", "N/A")

    if tp.get("bear"):
        bear1, bear2, s1 = tp["bear"]
        bcol3.metric("Bear Band", f"{bear1:.4f} → {bear2:.4f}  |  RR {rr_str}")
    else: bcol3.metric("Bear Band", f"N/A  |  RR {rr_str}")

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    with colA: st.plotly_chart(fig_rsi, use_container_width=True)
    with colB: st.plotly_chart(fig_macd, use_container_width=True)
    with colC: st.plotly_chart(fig_atr, use_container_width=True)

    st.subheader("🧪 Backtest Özeti (Long-only + Scale Out + Time Stop)")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
    m2.metric("Ann Return", f"{metrics['Annualized Return']*100:.1f}%")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%")
    m5.metric("Trades", f"{metrics['Trades']}")
    m6.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")
    m7.metric("Kelly Önerisi", f"{metrics['Kelly % (öneri)']:.1f}%")

    with st.expander("Trade listesi"):
        st.dataframe(tdf, use_container_width=True, height=240)
    with st.expander("Equity curve (Sermaye Eğrisi)"):
        st.plotly_chart(fig_eq, use_container_width=True)

    if sentiment_summary:
        st.subheader("📰 Haber Duygu Analizi (Google News + Gemini)")
        st.info(sentiment_summary)
        if st.session_state.sentiment_items:
            for item in st.session_state.sentiment_items:
                st.markdown(f"- [{item['title']}]({item['link']})")

    st.subheader("🤖 Gemini Multimodal AI — Grafik + Price Action + Spekülasyon Analizi")
    if not ai_on:
        st.info("Gemini kapalı.")
    else:
        pa = price_action_pack(df, last_n=20)
        st.session_state.pa_pack = pa

        if st.button("🖼️ Gemini'ye Sor (Görsel + Tüm Veriler)", use_container_width=True):
            snap20 = df_snapshot_for_llm(df, n=25)
            prompt = f"""
Sen profesyonel bir kripto para analistisin.
Analiz edilen parite: {ticker}
Aşağıdaki adımları takip et:
1. Genel trendi değerlendir.
2. Temel destek/direnç seviyelerini belirle.
3. Aşırı alım/spekülasyon göstergelerini incele.
4. Hedef bant: {tp}
5. Haber Özeti: {sentiment_summary}
Ekteki grafiği incele ve AL/SAT/İZLE önerisi ver.
Tablo eklemeyi unutma: Alış, Hedef, Stop.
"""
            image_bytes = _plotly_fig_to_png_bytes(fig_price)
            text = gemini_generate_text(prompt=prompt, model=gemini_model, temperature=gemini_temp, max_output_tokens=gemini_max_tokens, image_bytes=image_bytes)
            st.session_state.gemini_text = text

        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)

# =============================
# HEATMAP TAB (Kripto Uyarlı)
# =============================
with tab_heatmap:
    st.header("🔥 Kripto Piyasası Heatmap")
    if st.button("Verileri Getir ve Heatmap Oluştur"):
        with st.spinner("Toplu veri çekiliyor..."):
            universe_hm = get_crypto_universe()[:50] # Rate limit için ilk 50
            hm_data = []
            
            def fetch_hm_coin(coin):
                try:
                    bars = exchange.fetch_ohlcv(coin, timeframe="1d", limit=30)
                    if len(bars) > 5:
                        c_last = bars[-1][4]
                        c_1d = bars[-2][4]
                        c_1w = bars[-8][4] if len(bars)>7 else bars[0][4]
                        c_1m = bars[0][4]
                        return {
                            "Ticker": coin, "Sector": "Kripto",
                            "1 Günlük %": (c_last/c_1d-1)*100, "1 Haftalık %": (c_last/c_1w-1)*100, "1 Aylık %": (c_last/c_1m-1)*100
                        }
                except: pass
                return None

            with ThreadPoolExecutor(max_workers=5) as ex:
                for res in ex.map(fetch_hm_coin, universe_hm):
                    if res: hm_data.append(res)
                    
            df_hm = pd.DataFrame(hm_data)
            if not df_hm.empty:
                df_hm["Abs_1D"] = df_hm["1 Günlük %"].abs()
                fig_hm_1d = px.treemap(df_hm, path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"], values="Abs_1D", color="1 Günlük %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
                st.plotly_chart(fig_hm_1d, use_container_width=True)

# =============================
# EXPORT TAB
# =============================
with tab_export:
    st.subheader("📄 Rapor İndir")
    html_bytes = build_html_report(title=f"FA→TA Trading Report - {ticker}", meta=cfg, checkpoints=checkpoints, metrics=metrics, tp=tp, rr_info=rr_info, figs=figs_for_report, overbought_result=overbought_result)
    st.download_button("⬇️ HTML İndir", data=html_bytes, file_name=f"{ticker.replace('/', '_')}_report.html", mime="text/html", use_container_width=True)

# =============================
# TRIPLE SCREEN TAB (BİREBİR AYNI)
# =============================
with tab_triple:
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    if st.button("Üçlü Ekran Verilerini Analiz Et"):
        with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 1H)..."):
            df_1w = load_data_cached(ticker, "2y", "1wk")
            df_1d = load_data_cached(ticker, "1y", "1d")
            df_1h = load_data_cached(ticker, "60d", "1h")
            
            if df_1w.empty or df_1d.empty or df_1h.empty:
                st.error("Veri çekilemedi.")
            else:
                t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])
                
                with t_screen1:
                    m_line, m_sig, m_hist = macd(df_1w["Close"])
                    ema_1w_13 = ema(df_1w["Close"], 13)
                    ema_1w_26 = ema(df_1w["Close"], 26)
                    st.metric("MACD Histogram Eğimi", "YUKARI" if float(m_hist.iloc[-1]) > float(m_hist.iloc[-2]) else "AŞAĞI")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(x=df_1w.index, y=m_hist, name="MACD Hist"))
                    st.plotly_chart(fig1, use_container_width=True)
                
                with t_screen2:
                    ema_1d_11 = ema(df_1d["Close"], 11)
                    ema_1d_22 = ema(df_1d["Close"], 22)
                    st.metric("Günlük EMA (11-22)", "Trend Devam" if ema_1d_11.iloc[-1] > ema_1d_22.iloc[-1] else "Bekle")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Candlestick(x=df_1d.index, open=df_1d["Open"], high=df_1d["High"], low=df_1d["Low"], close=df_1d["Close"]))
                    fig2.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_11, name="EMA11"))
                    st.plotly_chart(fig2, use_container_width=True)
                
                with t_screen3:
                    st.info("Kısa vadeli giriş seviyesi arayışı.")
                    st.line_chart(df_1h["Close"])

# =============================
# YENİ EKLENEN 5. SEKME: ÇOKLU TARAYICI (SCREENER)
# =============================
with tab_screener:
    st.header("🔍 Çoklu Kripto Tarayıcı")
    st.markdown("Onlarca coini eş zamanlı olarak algoritmalarla tarayın.")
    
    sc_col1, sc_col2 = st.columns(2)
    scan_count = sc_col1.slider("Taranacak Coin Sayısı", 10, 100, 30)
    scan_tf = sc_col2.selectbox("Tarama Periyodu", ["1h", "4h", "1d"], index=2)
    
    if st.button("🚀 Taramayı Başlat"):
        all_coins = get_crypto_universe()[:scan_count]
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        def scan_single_coin(coin):
            try:
                # CCXT mapping for screener
                tf_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
                bars = exchange.fetch_ohlcv(coin, timeframe=tf_map[scan_tf], limit=100)
                d = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                if d.empty: return None
                
                d = build_features(d, cfg) # Senin tüm formasyonları çalıştırır
                last_row = d.iloc[-1]
                
                return {
                    "Sembol": coin,
                    "Fiyat": last_row["Close"],
                    "RSI": round(last_row["RSI"], 2),
                    "Trend": "YUKARI 🟢" if last_row["Close"] > last_row["EMA200"] else "AŞAĞI 🔴",
                    "🦘 Sinyal": "AL ✅" if last_row.get("KANGAROO_BULL") else ("SAT ❌" if last_row.get("KANGAROO_BEAR") else "-"),
                    "Formasyon": "Yutan Boğa 📈" if last_row.get("PATTERN_ENGULFING_BULL") else "-"
                }
            except:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {executor.submit(scan_single_coin, coin): coin for coin in all_coins}
            
            for i, future in enumerate(as_completed(future_to_coin)):
                coin_name = future_to_coin[future]
                res = future.result()
                if res:
                    results.append(res)
                
                progress_bar.progress((i + 1) / len(all_coins))
                status.text(f"İşleniyor: {coin_name}")
                time.sleep(0.05) 
        
        status.empty()
        if results:
            st.success(f"{len(results)} coin başarıyla tarandı.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
