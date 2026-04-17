
import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

st.set_page_config(
    page_title="Bitcoin Master 5 AI Pro | By Valiyev",
    page_icon="₿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #1e2130;
        color: white;
        border: 1px solid #30363d;
    }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    .stTextInput > div > div > input { background-color: #111827; color: white; }
    .btc-box {
        padding: 12px 14px;
        border: 1px solid #273244;
        border-radius: 10px;
        background: rgba(20, 24, 35, 0.75);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TIMEFRAME_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
SCAN_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]

if "show_chart_patterns" not in st.session_state:
    st.session_state.show_chart_patterns = True
if "show_ema13_channel" not in st.session_state:
    st.session_state.show_ema13_channel = False


def get_exchange_client(exchange_name: str):
    if exchange_name == "Binance":
        return ccxt.binance({"enableRateLimit": True})
    if exchange_name == "KuCoin":
        return ccxt.kucoin({"enableRateLimit": True})
    if exchange_name == "Kraken":
        return ccxt.kraken({"enableRateLimit": True})
    return ccxt.binance({"enableRateLimit": True})


@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(exchange_name: str, symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 300) -> pd.DataFrame:
    client = get_exchange_client(exchange_name)
    try:
        bars = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Europe/Istanbul")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def safe_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return cols[0] if cols else None


def pct_dist(level: Optional[float], base: float) -> Optional[float]:
    if level is None or base == 0:
        return None
    return ((level / base) - 1) * 100


def rsi_status(value: float) -> str:
    if pd.isna(value):
        return "Yetersiz veri"
    if value >= 70:
        return "Aşırı Alım"
    if value <= 30:
        return "Aşırı Satım"
    return "Normal"


def stoch_status(value: float) -> str:
    if pd.isna(value):
        return "Yetersiz veri"
    if value >= 80:
        return "Aşırı Alım"
    if value <= 20:
        return "Aşırı Satım"
    return "Normal"


def fmt_num(v: Optional[float], nd: int = 2) -> str:
    try:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{float(v):,.{nd}f}"
    except Exception:
        return "N/A"


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


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
    if len(close) < lookback:
        return False, 0
    c = close.tail(lookback)
    ind = indicator.tail(lookback)
    try:
        min_idx = c.values.argmin()
        bars_ago = (lookback - 1) - min_idx
        prev_c = c.iloc[: max(min_idx - 2, 0)]
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
        prev_c = c.iloc[: max(max_idx - 2, 0)]
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
    adx = ta.adx(high, low, close, length=period)
    if adx is None or adx.empty:
        idx = close.index
        return pd.Series(index=idx, data=0.0), pd.Series(index=idx, data=0.0), pd.Series(index=idx, data=0.0)
    adx_col = safe_col(adx, "ADX_")
    pdi_col = safe_col(adx, "DMP_")
    mdi_col = safe_col(adx, "DMN_")
    return adx[adx_col].fillna(0), adx[pdi_col].fillna(0), adx[mdi_col].fillna(0)


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    O, H, L, C = df["Open"], df["High"], df["Low"], df["Close"]
    body = (C - O).abs()
    rng = (H - L).replace(0, pd.NA)
    upper = H - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - L
    avg_rng = (H - L).rolling(10).mean()

    is_bull = C > O
    is_bear = C < O
    prev_O = O.shift(1)
    prev_C = C.shift(1)
    prev_H = H.shift(1)
    prev_L = L.shift(1)
    prev_is_bull = is_bull.shift(1)
    prev_is_bear = is_bear.shift(1)

    df["PATTERN_DOJI"] = (body / rng) <= 0.1
    df["PATTERN_LL_DOJI"] = df["PATTERN_DOJI"] & (upper >= 0.35 * rng) & (lower >= 0.35 * rng)

    shape_hammer = (lower >= 2 * body) & (upper <= 0.2 * rng) & (body > 0.02 * rng)
    shape_star = (upper >= 2 * body) & (lower <= 0.2 * rng) & (body > 0.02 * rng)
    ema50_ref = df["SMA50"] if "SMA50" in df.columns else C

    df["PATTERN_HAMMER"] = shape_hammer & (C < ema50_ref)
    df["PATTERN_HANGING_MAN"] = shape_hammer & (C > ema50_ref)
    df["PATTERN_SHOOTING_STAR"] = shape_star & (C > ema50_ref)
    df["PATTERN_INV_HAMMER"] = shape_star & (C < ema50_ref)

    df["PATTERN_MARUBOZU_BULL"] = is_bull & (body >= 0.85 * rng) & ((H - L) > avg_rng * 0.5)
    df["PATTERN_MARUBOZU_BEAR"] = is_bear & (body >= 0.85 * rng) & ((H - L) > avg_rng * 0.5)

    df["PATTERN_ENGULFING_BULL"] = is_bull & prev_is_bear & (O <= prev_C) & (C >= prev_O)
    df["PATTERN_ENGULFING_BEAR"] = is_bear & prev_is_bull & (O >= prev_C) & (C <= prev_O)

    df["PATTERN_HARAMI_BULL"] = is_bull & prev_is_bear & (O > prev_C) & (C < prev_O)
    df["PATTERN_HARAMI_BEAR"] = is_bear & prev_is_bull & (O < prev_C) & (C > prev_O)

    df["PATTERN_TWEEZER_TOP"] = (abs(H - prev_H) <= 0.002 * C) & is_bear & prev_is_bull & (H > ema50_ref)
    df["PATTERN_TWEEZER_BOTTOM"] = (abs(L - prev_L) <= 0.002 * C) & is_bull & prev_is_bear & (L < ema50_ref)

    df["PATTERN_PIERCING"] = is_bull & prev_is_bear & (O < prev_L) & (C > (prev_O + prev_C) / 2) & (C < prev_O)
    df["PATTERN_DARK_CLOUD"] = is_bear & prev_is_bull & (O > prev_H) & (C < (prev_O + prev_C) / 2) & (C > prev_O)

    prev2_O = O.shift(2)
    prev2_C = C.shift(2)
    prev2_is_bear = is_bear.shift(2)
    prev2_is_bull = is_bull.shift(2)
    df["PATTERN_MORNING_STAR"] = is_bull & prev2_is_bear & (prev_C < prev2_C) & (O > prev_C) & (C > (prev2_O + prev2_C) / 2)
    df["PATTERN_EVENING_STAR"] = is_bear & prev2_is_bull & (prev_C > prev2_C) & (O < prev_C) & (C < (prev2_O + prev2_C) / 2)

    df["Body"] = body
    df["Lower_Wick"] = lower
    df["Upper_Wick"] = upper
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 60:
        return df

    df = df.copy()
    df["EMA9"] = ta.ema(df["Close"], length=9)
    df["EMA21"] = ta.ema(df["Close"], length=21)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)

    df["EMA13_High"] = ta.ema(df["High"], length=13)
    df["EMA13_Low"] = ta.ema(df["Low"], length=13)
    df["EMA13_Close"] = ta.ema(df["Close"], length=13)

    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR_PCT"] = (df["ATR"] / df["Close"]) * 100

    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=5, d=3)
    if stoch is not None and not stoch.empty:
        df["STOCH_K"] = stoch.iloc[:, 0]
        df["STOCH_D"] = stoch.iloc[:, 1]

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)

    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df = pd.concat([df, bb], axis=1)
        bbu = safe_col(df, "BBU_")
        bbm = safe_col(df, "BBM_")
        bbl = safe_col(df, "BBL_")
        if bbu and bbm and bbl:
            df["BB_WIDTH"] = ((df[bbu] - df[bbl]) / df[bbm]) * 100

    df["VOL_SMA_10"] = df["Volume"].rolling(10).mean()
    df["VOL_SMA_20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA_20"]

    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["OBV_EMA"] = ta.ema(df["OBV"], length=21)

    df["RSI_OVERBOUGHT"] = (df["RSI"] > 70).astype(int)
    df["RSI_OVERSOLD"] = (df["RSI"] < 30).astype(int)
    bbu = safe_col(df, "BBU_")
    bbl = safe_col(df, "BBL_")
    if bbu and bbl:
        df["BB_OVERBOUGHT"] = (df["Close"] > df[bbu]).astype(int)
        df["BB_OVERSOLD"] = (df["Close"] < df[bbl]).astype(int)
    else:
        df["BB_OVERBOUGHT"] = 0
        df["BB_OVERSOLD"] = 0
    df["VOLUME_SPIKE"] = (df["Volume"] > df["VOL_SMA_20"] * 1.5).astype(int)
    df["PRICE_TO_SMA50"] = ((df["Close"] / df["SMA50"]) - 1) * 100
    df["PRICE_TO_SMA200"] = ((df["Close"] / df["SMA200"]) - 1) * 100
    df["PRICE_EXTREME"] = ((df["PRICE_TO_SMA50"] > 12) | (df["PRICE_TO_SMA200"] > 20)).astype(int)
    df["WEAK_UPTREND"] = ((df["Close"].diff() > 0) & (df["Volume"].diff() < 0)).astype(int)

    df = add_candle_patterns(df)
    df["KANGAROO_BULL"] = (df["Lower_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] < 35)
    df["KANGAROO_BEAR"] = (df["Upper_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] > 65)
    return df.dropna(how="all")


@st.cache_data(ttl=120, show_spinner=False)
def get_btc_data(exchange_name: str, timeframe: str, symbol: str = "BTC/USDT", limit: int = 300) -> pd.DataFrame:
    raw = fetch_ohlcv(exchange_name, symbol=symbol, timeframe=timeframe, limit=limit)
    return calculate_indicators(raw)


def trend_status(last_row: pd.Series) -> str:
    if pd.isna(last_row.get("SMA50")):
        return "Belirsiz"
    if last_row["Close"] > last_row["SMA50"] and last_row["EMA9"] > last_row["EMA21"]:
        return "Yükseliş 🟢"
    if last_row["Close"] < last_row["SMA50"] and last_row["EMA9"] < last_row["EMA21"]:
        return "Düşüş 🔴"
    return "Kararsız 🟡"


def detect_active_patterns(last_row: pd.Series) -> List[str]:
    found = []
    pattern_map = {
        "PATTERN_ENGULFING_BULL": "Bullish Engulfing",
        "PATTERN_ENGULFING_BEAR": "Bearish Engulfing",
        "PATTERN_DOJI": "Doji",
        "PATTERN_LL_DOJI": "Long-Legged Doji",
        "PATTERN_HAMMER": "Hammer",
        "PATTERN_HANGING_MAN": "Hanging Man",
        "PATTERN_SHOOTING_STAR": "Shooting Star",
        "PATTERN_INV_HAMMER": "Inverted Hammer",
        "PATTERN_MARUBOZU_BULL": "Bull Marubozu",
        "PATTERN_MARUBOZU_BEAR": "Bear Marubozu",
        "PATTERN_HARAMI_BULL": "Bull Harami",
        "PATTERN_HARAMI_BEAR": "Bear Harami",
        "PATTERN_TWEEZER_TOP": "Tweezer Top",
        "PATTERN_TWEEZER_BOTTOM": "Tweezer Bottom",
        "PATTERN_PIERCING": "Piercing",
        "PATTERN_DARK_CLOUD": "Dark Cloud",
        "PATTERN_MORNING_STAR": "Morning Star",
        "PATTERN_EVENING_STAR": "Evening Star",
        "KANGAROO_BULL": "Kangaroo Bull",
        "KANGAROO_BEAR": "Kangaroo Bear",
    }
    for col, label in pattern_map.items():
        if bool(last_row.get(col, False)):
            found.append(label)
    return found


def master_signal(df: pd.DataFrame) -> Tuple[str, int, List[str]]:
    if df.empty or len(df) < 3:
        return "NÖTR", 0, ["Yetersiz veri"]

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    reasons = []

    macd_col = safe_col(df, "MACD_")
    macds_col = safe_col(df, "MACDs_")
    hist_col = safe_col(df, "MACDh_")
    bbu_col = safe_col(df, "BBU_")
    bbm_col = safe_col(df, "BBM_")

    if last["Close"] > last["SMA50"]:
        score += 1
        reasons.append("Fiyat SMA50 üzerinde")
    else:
        score -= 1
        reasons.append("Fiyat SMA50 altında")

    if last["EMA9"] > last["EMA21"]:
        score += 1
        reasons.append("EMA9 > EMA21")
    else:
        score -= 1
        reasons.append("EMA9 < EMA21")

    if last["RSI"] < 30:
        score += 1
        reasons.append("RSI aşırı satım bölgesinde")
    elif last["RSI"] > 70:
        score -= 1
        reasons.append("RSI aşırı alım bölgesinde")

    if macd_col and macds_col and hist_col:
        if last[macd_col] > last[macds_col] and prev[macd_col] <= prev[macds_col]:
            score += 2
            reasons.append("MACD yukarı kesişim")
        elif last[macd_col] < last[macds_col] and prev[macd_col] >= prev[macds_col]:
            score -= 2
            reasons.append("MACD aşağı kesişim")
        elif last[hist_col] > 0:
            score += 1
            reasons.append("MACD histogram pozitif")
        else:
            score -= 1
            reasons.append("MACD histogram negatif")
    else:
        reasons.append("MACD verisi sınırlı")

    if bbu_col and bbm_col:
        if last["Close"] > last[bbm_col]:
            score += 1
            reasons.append("Fiyat Bollinger orta bandın üstünde")
        else:
            score -= 1
            reasons.append("Fiyat Bollinger orta bandın altında")

    if pd.notna(last.get("OBV_EMA")):
        if last["OBV"] > last["OBV_EMA"]:
            score += 1
            reasons.append("OBV, OBV EMA üstünde")
        else:
            score -= 1
            reasons.append("OBV, OBV EMA altında")

    if bool(last.get("PATTERN_ENGULFING_BULL", False)) or bool(last.get("PATTERN_HAMMER", False)) or bool(last.get("KANGAROO_BULL", False)):
        score += 2
        reasons.append("Boğa mum/formasyon desteği")

    if bool(last.get("PATTERN_ENGULFING_BEAR", False)) or bool(last.get("PATTERN_SHOOTING_STAR", False)) or bool(last.get("KANGAROO_BEAR", False)):
        score -= 2
        reasons.append("Ayı mum/formasyon baskısı")

    if score >= 5:
        return "GÜÇLÜ AL ✅", score, reasons
    if score >= 2:
        return "AL 🟢", score, reasons
    if score <= -5:
        return "GÜÇLÜ SAT ❌", score, reasons
    if score <= -2:
        return "SAT 🔴", score, reasons
    return "NÖTR 🟡", score, reasons


def detect_speculation(df: pd.DataFrame) -> Dict[str, object]:
    last = df.iloc[-1]
    result = {"overbought_score": 0, "oversold_score": 0, "speculation_score": 0, "details": {}, "verdict": "NÖTR"}
    if last["RSI"] > 70:
        result["overbought_score"] += 40
        result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        result["oversold_score"] += 40
        result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"
    if int(last.get("BB_OVERBOUGHT", 0)) == 1:
        result["overbought_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger üst bandı üzerinde"
    if int(last.get("BB_OVERSOLD", 0)) == 1:
        result["oversold_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger alt bandı altında"
    if int(last.get("VOLUME_SPIKE", 0)) == 1:
        result["speculation_score"] += 35
        result["details"]["vol"] = "Ani hacim sıçraması var"
    if int(last.get("PRICE_EXTREME", 0)) == 1:
        result["overbought_score"] += 20
        result["details"]["price_extreme"] = f"Fiyat SMA50'den uzaklaştı (%{last.get('PRICE_TO_SMA50', 0):.1f})"
    if int(last.get("WEAK_UPTREND", 0)) == 1:
        result["speculation_score"] += 25
        result["details"]["weak"] = "Fiyat yükselirken hacim zayıflıyor"
    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])
    if result["overbought_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERLİ (SAT bölgesi)"
    elif result["oversold_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERSİZ (AL bölgesi)"
    elif result["speculation_score"] >= 60:
        result["verdict"] = "SPEKÜLATİF HAREKET (dikkat)"
    else:
        result["verdict"] = "NÖTR (normal değer aralığı)"
    return result


def simple_checkpoints(df: pd.DataFrame) -> Dict[str, bool]:
    last = df.iloc[-1]
    macd_col = safe_col(df, "MACD_")
    bbm_col = safe_col(df, "BBM_")
    return {
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA_20"]) if pd.notna(last["VOL_SMA_20"]) else False,
        "Trend (Close>SMA200 & EMA9>EMA21)": bool((last["Close"] > last["SMA200"]) and (last["EMA9"] > last["EMA21"])) if pd.notna(last["SMA200"]) else False,
        "RSI > 50": bool(last["RSI"] > 50) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last[macd_col] > 0) if macd_col else False,
        "ATR% < 8": bool(last["ATR_PCT"] < 8) if pd.notna(last["ATR_PCT"]) else False,
        "Bollinger (Close>BB_mid)": bool(last["Close"] > last[bbm_col]) if bbm_col else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last.get("OBV_EMA")) else False,
    }


def weekly_trend_info(exchange_name: str) -> Tuple[bool, bool]:
    df_1d = get_btc_data(exchange_name, "1d", limit=260)
    df_1w = get_btc_data(exchange_name, "1w", limit=260)
    market_ok = False
    weekly_ok = False
    if not df_1d.empty and pd.notna(df_1d["SMA200"].iloc[-1]):
        market_ok = bool(df_1d["Close"].iloc[-1] > df_1d["SMA200"].iloc[-1])
    if not df_1w.empty:
        ema13 = ema(df_1w["Close"], 13)
        ema26 = ema(df_1w["Close"], 26)
        weekly_ok = bool((ema13.iloc[-1] > ema26.iloc[-1]) and (df_1w["Close"].iloc[-1] > ema13.iloc[-1]))
    return market_ok, weekly_ok


def nearest_levels(df: pd.DataFrame) -> Dict[str, object]:
    use = df.tail(200).copy()
    if use.empty:
        return {"base": None, "bull": None, "bear": None, "levels": []}
    base = float(use["Close"].iloc[-1])
    atr = float(use["ATR"].iloc[-1]) if pd.notna(use["ATR"].iloc[-1]) else 0.0
    highs = use["High"][(use["High"] == use["High"].rolling(5, center=True).max())].dropna().tolist()
    lows = use["Low"][(use["Low"] == use["Low"].rolling(5, center=True).min())].dropna().tolist()
    raw = sorted(set([round(float(x), 2) for x in highs + lows if pd.notna(x)]))
    levels = []
    for lv in raw:
        mask = (use["Low"] <= lv) & (use["High"] >= lv)
        touches = int(mask.sum())
        if touches == 0:
            continue
        first_idx = mask[mask].index[0]
        duration_bars = len(use.loc[first_idx:])
        vol_at = float(use.loc[mask, "Volume"].mean()) if "Volume" in use.columns else 0
        vol_avg = float(use["Volume"].mean()) if "Volume" in use.columns else 1
        vol_diff_pct = ((vol_at / vol_avg) - 1) * 100 if vol_avg else 0
        strength_pct = min(99.0, touches * 12 + max(0, vol_diff_pct / 2) + min(duration_bars / 2, 25))
        levels.append({
            "price": lv,
            "duration_bars": duration_bars,
            "vol_diff_pct": vol_diff_pct,
            "strength_pct": strength_pct,
            "touches": touches,
        })
    above = [x for x in levels if x["price"] > base]
    below = [x for x in levels if x["price"] < base]
    r1 = min(above, key=lambda x: x["price"]) if above else None
    s1 = max(below, key=lambda x: x["price"]) if below else None
    return {
        "base": base,
        "bull": (base + 1.5 * atr, base + 3.0 * atr, r1["price"] if r1 else None),
        "bear": (base - 1.5 * atr, base - 3.0 * atr, s1["price"] if s1 else None),
        "levels": levels,
        "r1_dict": r1,
        "s1_dict": s1,
    }


def build_price_chart(df: pd.DataFrame, show_indicators: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="BTC Fiyat", increasing_line_color="#00ffcc", decreasing_line_color="#ff4d6d"
        )
    )
    if show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA 9", line=dict(color="#00d4ff", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA 21", line=dict(color="yellow", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(color="orange", width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", line=dict(color="red", width=2)))
        bbu_col = safe_col(df, "BBU_")
        bbm_col = safe_col(df, "BBM_")
        bbl_col = safe_col(df, "BBL_")
        if bbu_col and bbm_col and bbl_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], name="BB Üst", line=dict(color="gray", dash="dash", width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df[bbm_col], name="BB Orta", line=dict(color="gray", dash="dot", width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], name="BB Alt", line=dict(color="gray", dash="dash", width=1)))
    if st.session_state.show_ema13_channel:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_High"], name="13 EMA High", line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_Low"], name="13 EMA Low", fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_Close"], name="13 EMA Close", line=dict(color='darkorange', width=2)))

    if st.session_state.show_chart_patterns:
        bull_patterns = {
            "KANGAROO_BULL": "🟩🦘 LONG KANGURU",
            "PATTERN_HAMMER": "🟩🔨 HAMMER",
            "PATTERN_INV_HAMMER": "🟩🔨 INV HAMMER",
            "PATTERN_ENGULFING_BULL": "🟢 ENGULFING",
            "PATTERN_HARAMI_BULL": "🟢🤰 HARAMI",
            "PATTERN_MARUBOZU_BULL": "🟩 MARUBOZU",
            "PATTERN_TWEEZER_BOTTOM": "🟢✌️ TWEEZER",
            "PATTERN_PIERCING": "🟢🗡️ PIERCING",
            "PATTERN_MORNING_STAR": "🟢🌅 M.STAR",
            "PATTERN_LL_DOJI": "🟢⚖️ LL DOJI",
        }
        bear_patterns = {
            "KANGAROO_BEAR": "🟥🦘 SHORT KANGURU",
            "PATTERN_HANGING_MAN": "🟥🪢 HANGING M.",
            "PATTERN_SHOOTING_STAR": "🟥🌠 S.STAR",
            "PATTERN_ENGULFING_BEAR": "🔴 ENGULFING",
            "PATTERN_HARAMI_BEAR": "🔴🤰 HARAMI",
            "PATTERN_MARUBOZU_BEAR": "🟥 MARUBOZU",
            "PATTERN_TWEEZER_TOP": "🔴✌️ TWEEZER",
            "PATTERN_DARK_CLOUD": "🔴🌩️ D.CLOUD",
            "PATTERN_EVENING_STAR": "🔴🌃 E.STAR",
        }
        bull_texts = pd.Series("", index=df.index)
        bear_texts = pd.Series("", index=df.index)
        for col, name in bull_patterns.items():
            if col in df.columns:
                mask = df[col].astype(bool)
                bull_texts[mask] += name + "<br>"
        for col, name in bear_patterns.items():
            if col in df.columns:
                mask = df[col].astype(bool)
                bear_texts[mask] += name + "<br>"
        bull_texts = bull_texts.str.rstrip("<br>")
        bear_texts = bear_texts.str.rstrip("<br>")
        bull_mask = bull_texts != ""
        bear_mask = bear_texts != ""
        if bull_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[bull_mask], y=df["Low"][bull_mask], mode="markers+text", name="Boğa Formasyonları",
                text=bull_texts[bull_mask], textposition="bottom center",
                textfont=dict(color="green", size=10, family="Arial Black"),
                marker=dict(symbol="triangle-up", size=9, color="green")
            ))
        if bear_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[bear_mask], y=df["High"][bear_mask], mode="markers+text", name="Ayı Formasyonları",
                text=bear_texts[bear_mask], textposition="top center",
                textfont=dict(color="red", size=10, family="Arial Black"),
                marker=dict(symbol="triangle-down", size=9, color="red")
            ))

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Fiyat Grafiği + EMA + Bollinger + Sinyaller & Formasyonlar",
    )
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", line=dict(color="#00ffcc", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
    fig.update_layout(height=240, template="plotly_dark", title="RSI (Göreceli Güç Endeksi)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_macd_chart(df: pd.DataFrame) -> go.Figure:
    macd_col = safe_col(df, "MACD_")
    signal_col = safe_col(df, "MACDs_")
    hist_col = safe_col(df, "MACDh_")
    fig = go.Figure()
    if macd_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_col], name="MACD", line=dict(color="#00d4ff", width=2)))
    if signal_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[signal_col], name="Signal", line=dict(color="orange", width=2)))
    if hist_col:
        fig.add_trace(go.Bar(x=df.index, y=df[hist_col], name="Histogram"))
    fig.update_layout(height=260, template="plotly_dark", title="MACD (Moving Average Convergence Divergence)", margin=dict(l=10, r=10, t=30, b=10), barmode="relative")
    return fig


def build_atr_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"], name="ATR %"))
    fig.update_layout(height=260, template="plotly_dark", title="ATR % (Ortalama Gerçek Aralık / Fiyat)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_stoch_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"], name="Stochastic K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"], name="Stochastic D"))
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
    fig.update_layout(height=260, template="plotly_dark", title="Stochastic RSI (K & D)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_bbwidth_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "BB_WIDTH" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"], name="BB Width"))
        fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Sıkışma Bölgesi")
    fig.update_layout(height=260, template="plotly_dark", title="Bollinger Bandı Genişliği %", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_volratio_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı"))
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anormal Hacim")
    fig.update_layout(height=260, template="plotly_dark", title="Hacim Oranı (Son Hacim / SMA)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_vol_market_chart(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> go.Figure:
    bench_vol = benchmark_df["Volume"].reindex(df.index).ffill().fillna(0) if not benchmark_df.empty else pd.Series(0, index=df.index)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="BTC Hacmi", marker_color='lightblue', opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=bench_vol, name="Kıyas Benchmark Hacmi", line=dict(color='orange', width=2)), secondary_y=True)
    fig.update_layout(height=260, template="plotly_dark", title="Bitcoin vs Benchmark Hacmi", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_vol_2wk_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim", marker_color='cadetblue', opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA_10"], name="2 Haftalık Ort. (10 Bar)", line=dict(color='red', width=2)))
    fig.update_layout(height=260, template="plotly_dark", title="Bitcoin Hacmi vs 2 Haftalık Ortalama", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_obv_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV_EMA"], name="OBV EMA (21)", line=dict(color='orange', dash='dot')))
    fig.update_layout(height=260, template="plotly_dark", title="On-Balance Volume (OBV)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def summarize_timeframe(df: pd.DataFrame, label: str) -> Dict[str, str]:
    if df.empty or len(df) < 2:
        return {"Zaman": label, "Trend": "Veri yok", "RSI": "-", "Sinyal": "-", "Skor": "-"}
    last = df.iloc[-1]
    signal, score, _ = master_signal(df)
    return {"Zaman": label, "Trend": trend_status(last), "RSI": f"{last['RSI']:.1f} ({rsi_status(last['RSI'])})", "Sinyal": signal, "Skor": str(score)}


def render_bias_box(title: str, text: str):
    st.markdown(f"<div class='btc-box'><strong>{title}</strong><br>{text}</div>", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_bitcoin_news() -> List[Dict[str, str]]:
    feeds = [
        "https://news.google.com/rss/search?q=Bitcoin+OR+BTC&hl=en-US&gl=US&ceid=US:en",
        "https://cointelegraph.com/rss/tag/bitcoin",
    ]
    seen = set()
    items: List[Dict[str, str]] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:12]:
                title = getattr(entry, "title", "Başlıksız")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")
                key = (title.strip(), link.strip())
                if key in seen:
                    continue
                seen.add(key)
                items.append({"title": title, "link": link, "date": published})
        except Exception:
            continue
    return items[:15]


def ai_market_context(df: pd.DataFrame, exchange_name: str, timeframe: str) -> str:
    last = df.iloc[-1]
    signal, score, reasons = master_signal(df)
    patterns = detect_active_patterns(last)
    pattern_text = ", ".join(patterns) if patterns else "Aktif formasyon yok"
    return f"""
Varlık: Bitcoin (BTC/USDT)
Borsa: {exchange_name}
Ana zaman dilimi: {timeframe}
Son fiyat: {last['Close']:.2f}
RSI: {last['RSI']:.2f}
Stochastic K: {last['STOCH_K']:.2f}
ATR: {last['ATR']:.2f}
EMA9: {last['EMA9']:.2f}
EMA21: {last['EMA21']:.2f}
SMA50: {last['SMA50']:.2f}
SMA200: {last['SMA200']:.2f}
OBV: {last['OBV']:.2f}
Trend: {trend_status(last)}
Master sinyal: {signal}
Master skor: {score}
Aktif formasyonlar: {pattern_text}
Sinyal gerekçeleri: {', '.join(reasons)}
""".strip()


def ask_gemini(api_key: str, system_prompt: str, user_prompt: str) -> str:
    try:
        from google import genai
    except Exception:
        return "Gemini SDK bulunamadı. `pip install -U google-genai` komutunu çalıştırıp tekrar deneyin."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\nKullanıcı sorusu:\n{user_prompt}",
        )
        return getattr(response, "text", "Yanıt alınamadı.") or "Yanıt alınamadı."
    except Exception as e:
        return f"Gemini hatası: {e}"


st.sidebar.image("https://img.icons8.com/color/96/bitcoin--v1.png", width=84)
st.sidebar.title("Bitcoin Master 5 AI Pro")
st.sidebar.markdown("---")
exchange_name = st.sidebar.selectbox("Borsa", ["Binance", "KuCoin", "Kraken"], index=0)
symbol = "BTC/USDT"
selected_tf = st.sidebar.selectbox("Ana Zaman Dilimi", ["15m", "1h", "4h", "1d", "1w"], index=3)
bar_limit = st.sidebar.slider("Çubuk Sayısı", 150, 500, 300, 25)

st.sidebar.markdown("---")
st.sidebar.subheader("Analiz Ayarları")
show_indicators = st.sidebar.checkbox("İndikatörleri Göster", value=True)
show_stats = st.sidebar.checkbox("Piyasa İstatistiklerini Göster", value=True)
show_patterns_toggle = st.sidebar.checkbox("Formasyonları Göster", value=st.session_state.show_chart_patterns)
st.session_state.show_chart_patterns = show_patterns_toggle

st.sidebar.markdown("---")
st.sidebar.subheader("Google AI")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

with st.spinner("Bitcoin verisi yükleniyor..."):
    df = get_btc_data(exchange_name, selected_tf, symbol=symbol, limit=bar_limit)
    benchmark_df = get_btc_data(exchange_name, "1d", symbol=symbol, limit=bar_limit)

if df.empty or len(df) < 50:
    st.error("Veri alınamadı veya indikatörler için yeterli veri yok. Borsa/zaman dilimini değiştirip tekrar deneyin.")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]
change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
signal, score, reasons = master_signal(df)
patterns = detect_active_patterns(last)
market_ok, weekly_ok = weekly_trend_info(exchange_name)
overbought = detect_speculation(df)
checkpoints = simple_checkpoints(df)
levels = nearest_levels(df)
fig_price = build_price_chart(df, show_indicators=show_indicators)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🤖 Google AI Chat",
    "📰 Haber Analizi",
    "📺 3 Ekranlı Sistem",
])

with tab1:
    st.subheader(f"₿ Bitcoin Dashboard | {exchange_name} | {selected_tf}")

    col_ob1, col_ob2, col_ob3, col_ob4 = st.columns(4)
    col_ob1.metric("Aşırı Alım Skoru", f"{overbought['overbought_score']}/100")
    col_ob2.metric("Aşırı Satım Skoru", f"{overbought['oversold_score']}/100")
    col_ob3.metric("Spekülasyon Skoru", f"{overbought['speculation_score']}/100")
    col_ob4.metric("Genel Karar", overbought["verdict"])

    with st.expander("Detaylı Aşırı Alım/Spekülasyon Analizi"):
        if overbought["details"]:
            for _, v in overbought["details"].items():
                st.write(f"• {v}")
        else:
            st.write("Öne çıkan aşırılaşma sinyali yok.")

    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Son Fiyat", f"{last['Close']:,.2f}", f"{change_pct:.2f}%")
    m2.metric("RSI (14)", f"{last['RSI']:.1f}", rsi_status(last["RSI"]))
    m3.metric("Stoch K", f"{last['STOCH_K']:.1f}", stoch_status(last["STOCH_K"]))
    m4.metric("Trend", trend_status(last))
    m5.metric("ATR", f"{last['ATR']:.2f}", "Volatilite Ölçüsü")
    m6.metric("Master Sinyal", signal, f"Skor: {score}")
    m7.metric("Piyasa Filtresi", "BULL ✅" if market_ok else "BEAR ❌")
    m8.metric("Haftalık Trend", "BULL ✅" if weekly_ok else "BEAR ❌")

    st.subheader("🕯️ Fiyat Aksiyonu (Price Action) Mum Formasyonları - Son Bar")
    pa_c1, pa_c2, pa_c3, pa_c4, pa_c5, pa_c6 = st.columns(6)
    pa_c1.metric("1. Kanguru", "BOĞA 🦘" if bool(last.get("KANGAROO_BULL")) else ("AYI 🦘" if bool(last.get("KANGAROO_BEAR")) else "YOK"))
    pa_c2.metric("2. Engulfing", "Boğa 🟢" if bool(last.get("PATTERN_ENGULFING_BULL")) else ("Ayı 🔴" if bool(last.get("PATTERN_ENGULFING_BEAR")) else "Yok"))
    pa_c3.metric("3. Hammer / Star", "Çekiç 🟢" if bool(last.get("PATTERN_HAMMER")) else ("Kayan Yıldız 🔴" if bool(last.get("PATTERN_SHOOTING_STAR")) else "Yok"))
    pa_c4.metric("4. Doji", "Uzun Bacak ⚪" if bool(last.get("PATTERN_LL_DOJI")) else ("Doji ⚪" if bool(last.get("PATTERN_DOJI")) else "Yok"))
    pa_c5.metric("5. Marubozu", "Boğa 🟢" if bool(last.get("PATTERN_MARUBOZU_BULL")) else ("Ayı 🔴" if bool(last.get("PATTERN_MARUBOZU_BEAR")) else "Yok"))
    pa_c6.metric("6. Harami", "Boğa 🟢" if bool(last.get("PATTERN_HARAMI_BULL")) else ("Ayı 🔴" if bool(last.get("PATTERN_HARAMI_BEAR")) else "Yok"))

    pa2_c1, pa2_c2, pa2_c3, pa2_c4, pa2_c5, pa2_c6 = st.columns(6)
    pa2_c1.metric("7. Tweezer", "Dip 🟢" if bool(last.get("PATTERN_TWEEZER_BOTTOM")) else ("Tepe 🔴" if bool(last.get("PATTERN_TWEEZER_TOP")) else "Yok"))
    pa2_c2.metric("8. M./E. Star", "Sabah 🟢" if bool(last.get("PATTERN_MORNING_STAR")) else ("Akşam 🔴" if bool(last.get("PATTERN_EVENING_STAR")) else "Yok"))
    pa2_c3.metric("9. Piercing / Dark", "Delen 🟢" if bool(last.get("PATTERN_PIERCING")) else ("Kara Bulut 🔴" if bool(last.get("PATTERN_DARK_CLOUD")) else "Yok"))
    pa2_c4.metric("10. Inv. H / Hang", "Ters Çekiç 🟢" if bool(last.get("PATTERN_INV_HAMMER")) else ("Asılı Adam 🔴" if bool(last.get("PATTERN_HANGING_MAN")) else "Yok"))
    pa2_c5.metric("11. Filtre Durumu", "Aktif ✅")
    pa2_c6.write("")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    for i, (k, v) in enumerate(checkpoints.items()):
        with cp_cols[i % 3]:
            st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = levels["base"]
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", f"{base_px:.2f}" if base_px else "N/A")

    r1 = levels["bull"][2] if levels["bull"] else None
    s1 = levels["bear"][2] if levels["bear"] else None
    if levels["bull"]:
        bull1, bull2, r1 = levels["bull"]
        bcol2.metric("Bull Band", f"{bull1:.2f} → {bull2:.2f}")
        if r1 is not None:
            info = levels.get("r1_dict") or {}
            bcol2.caption(f"Yakın direnç: {r1:.2f} ({pct_dist(r1, base_px):+.2f}%)\n\n**Güç:** %{info.get('strength_pct', 0):.0f} | **Uzunluk:** {info.get('duration_bars', 0)} Bar | **Hacim:** %{info.get('vol_diff_pct', 0):+.1f}")
        else:
            bcol2.caption("Yakın direnç: YOK")
    if levels["bear"]:
        bear1, bear2, s1 = levels["bear"]
        bcol3.metric("Bear Band", f"{bear1:.2f} → {bear2:.2f}")
        if s1 is not None:
            info = levels.get("s1_dict") or {}
            bcol3.caption(f"Yakın destek: {s1:.2f} ({pct_dist(s1, base_px):+.2f}%)\n\n**Güç:** %{info.get('strength_pct', 0):.0f} | **Uzunluk:** {info.get('duration_bars', 0)} Bar | **Hacim:** %{info.get('vol_diff_pct', 0):+.1f}")
        else:
            bcol3.caption("Yakın destek: YOK")

    def render_levels_marked():
        lines = []
        for lv_dict in levels.get("levels", []):
            lv = float(lv_dict["price"])
            tag = ""
            if s1 is not None and abs(lv - float(s1)) < 1e-9:
                tag = " 🟩 Yakın Destek"
            if r1 is not None and abs(lv - float(r1)) < 1e-9:
                tag = " 🟥 Yakın Direnç"
            dist = pct_dist(lv, base_px)
            lines.append(f"- **{lv:.2f}** ({dist:+.2f}%) | Güç: %{lv_dict['strength_pct']:.0f} | Uzunluk: {lv_dict['duration_bars']} Bar | Hacim: %{lv_dict['vol_diff_pct']:+.1f}{tag}")
        return "\n".join(lines) if lines else "_Seviye yok_"

    with st.expander("Seviye listesi (yaklaşık) — işaretli + fiyata uzaklık %", expanded=False):
        st.markdown(render_levels_marked())

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("🛠️ Grafik Analiz Araçları")
    tools_col1, tools_col2, tools_col3 = st.columns(3)
    if tools_col1.button("Sadece Grafiği Analiz Et", use_container_width=True):
        st.session_state.show_chart_patterns = False
        st.rerun()
    if tools_col2.button("Formasyonları Geri Getir", use_container_width=True):
        st.session_state.show_chart_patterns = True
        st.rerun()
    if tools_col3.button("13 EMA Kanalını Aç / Kapat", use_container_width=True):
        st.session_state.show_ema13_channel = not st.session_state.show_ema13_channel
        st.rerun()

    st.subheader("📉 RSI / MACD / ATR%")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(build_macd_chart(df), use_container_width=True)
    with c3:
        st.plotly_chart(build_atr_chart(df), use_container_width=True)

    st.subheader("📊 Stochastic RSI / Bollinger Genişliği / Hacim Oranı")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.plotly_chart(build_stoch_chart(df), use_container_width=True)
    with c5:
        st.plotly_chart(build_bbwidth_chart(df), use_container_width=True)
    with c6:
        st.plotly_chart(build_volratio_chart(df), use_container_width=True)

    st.subheader("📊 Hacim ve Trend Karşılaştırmaları")
    c7, c8, c9 = st.columns(3)
    with c7:
        st.plotly_chart(build_vol_market_chart(df, benchmark_df), use_container_width=True)
    with c8:
        st.plotly_chart(build_vol_2wk_chart(df), use_container_width=True)
    with c9:
        st.plotly_chart(build_obv_chart(df), use_container_width=True)

    if show_stats:
        st.markdown("---")
        st.subheader("📋 Bitcoin Piyasa İstatistikleri")
        s1c, s2c, s3c, s4c = st.columns(4)
        s1c.info(f"**Hacim:** {last['Volume']:,.2f}")
        s2c.info(f"**Bar Aralığı:** {last['Low']:,.2f} - {last['High']:,.2f}")
        s3c.info(f"**EMA9 / EMA21:** {last['EMA9']:,.2f} / {last['EMA21']:,.2f}")
        s4c.info(f"**SMA50 / SMA200:** {last['SMA50']:,.2f} / {last['SMA200']:,.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("🟢 Al / Sat Gerekçeleri")
        for item in reasons:
            st.write(f"- {item}")
    with right:
        st.subheader("🕯 Tespit Edilen Formasyonlar")
        if patterns:
            for p in patterns:
                st.write(f"- {p}")
        else:
            st.write("- Son mumda güçlü bir formasyon tespit edilmedi.")

with tab2:
    st.subheader("🤖 Google AI ile Bitcoin Sohbet / Analiz")
    st.caption("Güncel teknik veriyi prompt içine otomatik ekler.")

    if "btc_chat_history" not in st.session_state:
        st.session_state["btc_chat_history"] = []

    system_context = ai_market_context(df, exchange_name, selected_tf)

    st.text_area("AI'ye giden piyasa özeti", value=system_context, height=220, disabled=True)
    user_question = st.text_area("Sorunuzu yazın", placeholder="Örn: Bu görünüm kısa vadede düzeltme mi yoksa trend devamı mı gösteriyor? Riskler neler?", height=100)

    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        ask_button = st.button("🧠 Analiz Sor")
    with col_ai2:
        clear_button = st.button("🧹 Sohbeti Temizle")

    if clear_button:
        st.session_state["btc_chat_history"] = []
        st.success("Sohbet temizlendi.")

    if not gemini_api_key:
        st.warning("Gemini API anahtarını sol menüden girin.")
    elif ask_button:
        if not user_question.strip():
            st.warning("Önce bir soru yazın.")
        else:
            history_text = "\n".join([f"Soru: {item['q']}\nYanıt: {item['a']}" for item in st.session_state["btc_chat_history"][-3:]])
            full_prompt = (
                "Sen deneyimli bir teknik analiz uzmanısın. Yanıtın Türkçe olsun. Kesin yatırım tavsiyesi verme. "
                f"Piyasa Bağlamı:\n{system_context}\n\nÖnceki Kısa Sohbet:\n{history_text if history_text else 'Yok'}\n"
            )
            answer = ask_gemini(gemini_api_key, full_prompt, user_question)
            st.session_state["btc_chat_history"].append({"q": user_question, "a": answer})

    if st.session_state["btc_chat_history"]:
        st.markdown("---")
        st.subheader("Sohbet Geçmişi")
        for i, item in enumerate(reversed(st.session_state["btc_chat_history"]), 1):
            st.markdown(f"**Soru {i}:** {item['q']}")
            st.markdown(item["a"])
            st.markdown("---")

with tab3:
    st.subheader("📰 Bitcoin Haber Analizi")
    st.caption("Google News ve Cointelegraph RSS kaynakları birleştirilir.")
    if st.button("🔄 Bitcoin Haberlerini Yenile"):
        st.cache_data.clear()
    news_items = fetch_bitcoin_news()
    if news_items:
        for item in news_items:
            st.markdown(f"- **[{item['title']}]({item['link']})**  ")
            st.caption(item["date"])
    else:
        st.info("Şu anda haber çekilemedi.")

with tab4:
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    st.caption("Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan, trend, osilatör ve giriş seviyesi analizleri.")

    df_1w = get_btc_data(exchange_name, "1w", symbol=symbol, limit=250)
    df_1d = get_btc_data(exchange_name, "1d", symbol=symbol, limit=250)
    df_1h = get_btc_data(exchange_name, "1h", symbol=symbol, limit=400)

    if df_1w.empty or df_1d.empty or df_1h.empty:
        st.error("Bazı zaman dilimleri için veri çekilemedi.")
    else:
        t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])

        with t_screen1:
            st.subheader("1. Ekran: Haftalık (Ana Trend)")
            m_line = ta.macd(df_1w["Close"], fast=12, slow=26, signal=9)
            hist_col = safe_col(m_line, "MACDh_")
            hist = m_line[hist_col] if hist_col else pd.Series(index=df_1w.index, data=0.0)

            ema_1w_13 = ema(df_1w["Close"], 13)
            ema_1w_26 = ema(df_1w["Close"], 26)
            last_close_1w = df_1w["Close"].iloc[-1]

            if ema_1w_13.iloc[-1] > ema_1w_26.iloc[-1] and last_close_1w > ema_1w_13.iloc[-1]:
                ema1w_sig = "AL"
            elif ema_1w_13.iloc[-1] < ema_1w_26.iloc[-1] and last_close_1w < ema_1w_13.iloc[-1]:
                ema1w_sig = "SAT"
            else:
                ema1w_sig = "BEKLE"

            last_hist = float(hist.iloc[-1])
            prev_hist = float(hist.iloc[-2])
            slope_up = last_hist > prev_hist
            div_macd, macd_ago = check_bullish_divergence(df_1w["Close"], hist)

            adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"])
            adx_val_1w = adx_1w.iloc[-1]
            pdi_val_1w = pdi_1w.iloc[-1]
            mdi_val_1w = mdi_1w.iloc[-1]

            if adx_val_1w >= 25 and pdi_val_1w > mdi_val_1w:
                adx_sig_1w = "AL (Güçlü Trend)"
            elif adx_val_1w >= 25 and mdi_val_1w > pdi_val_1w:
                adx_sig_1w = "SAT (Güçlü Trend)"
            else:
                adx_sig_1w = "BEKLE (Zayıf Trend)"

            c1w_1, c1w_2, c1w_3 = st.columns(3)
            c1w_1.metric("MACD Histogram Eğimi", "YUKARI (AL Sinyali)" if slope_up else "AŞAĞI (SAT Sinyali)", f"{last_hist - prev_hist:.2f}")
            c1w_2.metric("Haftalık EMA (13-26)", ema1w_sig, f"EMA13: {ema_1w_13.iloc[-1]:.2f} | EMA26: {ema_1w_26.iloc[-1]:.2f}")
            c1w_3.metric("ADX (14)", adx_sig_1w, f"ADX: {adx_val_1w:.1f} | +DI: {pdi_val_1w:.1f} | -DI: {mdi_val_1w:.1f}")

            if div_macd:
                st.success(f"🚀 Sistem Haftalık MACD Histogramında **Pozitif Uyumsuzluk** tespit etti! ({macd_ago} bar önce)")

            fig1_price = go.Figure()
            fig1_price.add_trace(go.Candlestick(x=df_1w.index, open=df_1w["Open"], high=df_1w["High"], low=df_1w["Low"], close=df_1w["Close"], name="Fiyat"))
            fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_13, name="EMA 13", line=dict(color='blue')))
            fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_26, name="EMA 26", line=dict(color='red')))
            fig1_price.update_layout(title="Haftalık Fiyat ve EMA (13 & 26)", height=350, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig1_price, use_container_width=True)

            fig1 = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in hist.diff().fillna(0)]
            fig1.add_trace(go.Bar(x=df_1w.index, y=hist, name="MACD Hist", marker_color=colors))
            fig1.update_layout(title="Haftalık MACD Histogramı", height=250, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)

            fig1_adx = go.Figure()
            fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=adx_1w, name="ADX", line=dict(color='black', width=2.5)))
            fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=pdi_1w, name="+DI", line=dict(color='green')))
            fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=mdi_1w, name="-DI", line=dict(color='red')))
            fig1_adx.add_hline(y=25, line_dash="dash", line_color="gray")
            fig1_adx.add_hline(y=50, line_dash="dot", line_color="purple")
            fig1_adx.update_layout(title="Haftalık ADX ve Yön Göstergeleri (+DI / -DI)", height=250, template="plotly_dark")
            st.plotly_chart(fig1_adx, use_container_width=True)

        with t_screen2:
            st.subheader("2. Ekran: Günlük (Osilatörler ve Sapmalar)")
            ema_1d_11 = ema(df_1d["Close"], 11)
            ema_1d_22 = ema(df_1d["Close"], 22)
            last_close_1d = df_1d["Close"].iloc[-1]
            if ema_1d_11.iloc[-1] > ema_1d_22.iloc[-1] and last_close_1d > ema_1d_11.iloc[-1]:
                ema1d_sig = "AL"
            elif ema_1d_11.iloc[-1] < ema_1d_22.iloc[-1] and last_close_1d < ema_1d_11.iloc[-1]:
                ema1d_sig = "SAT"
            else:
                ema1d_sig = "BEKLE"
            st.metric("Günlük EMA (11-22)", ema1d_sig, f"EMA11: {ema_1d_11.iloc[-1]:.2f} | EMA22: {ema_1d_22.iloc[-1]:.2f}")

            fi = force_index(df_1d["Close"], df_1d["Volume"])
            fi_ema13 = ema(fi, 13)
            fi_ema2 = ema(fi, 2)
            rsi13 = ta.rsi(df_1d["Close"], length=13)
            stoch_k, stoch_d = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], k_period=5, d_period=3)
            er_ema, bull_p, bear_p = elder_ray(df_1d["High"], df_1d["Low"], df_1d["Close"], 13)

            fi_al = (fi_ema2.iloc[-1] < 0) and (fi_ema2.iloc[-1] > fi_ema2.iloc[-2])
            rsi_al = (rsi13.iloc[-1] < 30)
            stoch_al = (stoch_k.iloc[-1] < 20)
            er_ema_up = (er_ema.iloc[-1] > er_ema.iloc[-2])
            bp_neg_but_rising = (bear_p.iloc[-1] < 0) and (bear_p.iloc[-1] > bear_p.iloc[-2])
            er_al = er_ema_up and bp_neg_but_rising

            div_rsi, rsi_ago = check_bullish_divergence(df_1d["Close"], rsi13)
            div_stoch, stoch_ago = check_bullish_divergence(df_1d["Close"], stoch_k)
            div_er, er_ago = check_bullish_divergence(df_1d["Close"], bear_p)
            div_er_bear, er_bear_ago = check_bearish_divergence(df_1d["Close"], bull_p)

            adx_1d, pdi_1d, mdi_1d = adx_indicator(df_1d["High"], df_1d["Low"], df_1d["Close"])
            adx_val_1d = adx_1d.iloc[-1]
            pdi_val_1d = pdi_1d.iloc[-1]
            mdi_val_1d = mdi_1d.iloc[-1]
            if adx_val_1d >= 25 and pdi_val_1d > mdi_val_1d:
                adx_sig_1d = "AL (Güçlü Trend)"
            elif adx_val_1d >= 25 and mdi_val_1d > pdi_val_1d:
                adx_sig_1d = "SAT (Güçlü Trend)"
            else:
                adx_sig_1d = "BEKLE (Zayıf Trend)"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Kuvvet Endeksi (FI)", "AL" if fi_al else "BEKLE", "2 EMA Negatif & Yukarı Dönüş" if fi_al else "")
            c2.metric("RSI (13)", "AL" if rsi_al else "BEKLE", f"{rsi13.iloc[-1]:.1f}")
            c3.metric("Stokastik (5)", "AL" if stoch_al else "BEKLE", f"{stoch_k.iloc[-1]:.1f}")
            c4.metric("Elder-Ray", "AL" if er_al else "BEKLE")
            c5.metric("ADX (14)", adx_sig_1d, f"ADX: {adx_val_1d:.1f} | +DI: {pdi_val_1d:.1f}")

            if div_rsi:
                st.success(f"🚀 RSI(13)'te **Pozitif Uyumsuzluk** tespit edildi! ({rsi_ago} bar önce)")
            if div_stoch:
                st.success(f"🚀 Stokastik(5)'te **Pozitif Uyumsuzluk** tespit edildi! ({stoch_ago} bar önce)")
            if div_er:
                st.success(f"🚀 Elder-Ray Bear Power'da **Pozitif Uyumsuzluk (Boğa Uyumsuzluğu)** tespit edildi! ({er_ago} bar önce)")
            if div_er_bear:
                st.warning(f"⚠️ Elder-Ray Bull Power'da **Negatif Uyumsuzluk (Ayı Uyumsuzluğu)** tespit edildi! ({er_bear_ago} bar önce)")

            fig2_price = go.Figure()
            fig2_price.add_trace(go.Candlestick(x=df_1d.index, open=df_1d["Open"], high=df_1d["High"], low=df_1d["Low"], close=df_1d["Close"], name="Fiyat"))
            fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_11, name="EMA 11", line=dict(color='blue')))
            fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_22, name="EMA 22", line=dict(color='red')))
            fig2_price.update_layout(title="Günlük Fiyat ve EMA (11 & 22)", height=350, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig2_price, use_container_width=True)

            fig2_fi = go.Figure()
            fig2_fi.add_trace(go.Scatter(x=df_1d.index, y=fi_ema13, name="FI 13 EMA", line=dict(color='orange')))
            fig2_fi.add_trace(go.Bar(x=df_1d.index, y=fi_ema2, name="FI 2 EMA", marker_color='gray'))
            fig2_fi.update_layout(title="Kuvvet Endeksi (Force Index)", height=250, template="plotly_dark")
            st.plotly_chart(fig2_fi, use_container_width=True)

            fig2_er = go.Figure()
            fig2_er.add_trace(go.Bar(x=df_1d.index, y=bull_p, name="Bull Power", marker_color='green'))
            fig2_er.add_trace(go.Bar(x=df_1d.index, y=bear_p, name="Bear Power", marker_color='red'))
            fig2_er.update_layout(title="Elder-Ray (Bull & Bear Power)", height=250, template="plotly_dark")
            st.plotly_chart(fig2_er, use_container_width=True)

            fig2_adx = go.Figure()
            fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=adx_1d, name="ADX", line=dict(color='black', width=2.5)))
            fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=pdi_1d, name="+DI", line=dict(color='green')))
            fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=mdi_1d, name="-DI", line=dict(color='red')))
            fig2_adx.add_hline(y=25, line_dash="dash", line_color="gray")
            fig2_adx.add_hline(y=50, line_dash="dot", line_color="purple")
            fig2_adx.update_layout(title="Günlük ADX ve Yön Göstergeleri (+DI / -DI)", height=250, template="plotly_dark")
            st.plotly_chart(fig2_adx, use_container_width=True)

        with t_screen3:
            st.subheader("3. Ekran: 1 Saatlik (Giriş / Çıkış ve Hedefler)")
            adx_1h, pdi_1h, mdi_1h = adx_indicator(df_1h["High"], df_1h["Low"], df_1h["Close"])
            adx_val_1h = adx_1h.iloc[-1]
            pdi_val_1h = pdi_1h.iloc[-1]
            mdi_val_1h = mdi_1h.iloc[-1]

            if adx_val_1h >= 25 and pdi_val_1h > mdi_val_1h:
                adx_sig_1h = "AL (Güçlü Trend)"
            elif adx_val_1h >= 25 and mdi_val_1h > pdi_val_1h:
                adx_sig_1h = "SAT (Güçlü Trend)"
            else:
                adx_sig_1h = "BEKLE (Zayıf Trend)"

            st.metric("1 Saatlik ADX (14)", adx_sig_1h, f"ADX: {adx_val_1h:.1f} | +DI: {pdi_val_1h:.1f} | -DI: {mdi_val_1h:.1f}")

            ema_1h = ema(df_1h["Close"], 13)
            atr_1h = ta.atr(df_1h["High"], df_1h["Low"], df_1h["Close"], length=14)
            last_atr_1h = float(atr_1h.iloc[-1]) if not pd.isna(atr_1h.iloc[-1]) else 0.0

            pens = ema_1h - df_1h["Low"]
            pens_positive = pens[pens > 0]
            avg_pen = float(pens_positive.mean()) if not pens_positive.empty else 0.0

            up_pens = df_1h["High"] - ema_1h
            up_pens_positive = up_pens[up_pens > 0]
            avg_up_pen = float(up_pens_positive.mean()) if not up_pens_positive.empty else 0.0

            ema_today = float(ema_1h.iloc[-1])
            ema_yest = float(ema_1h.iloc[-2])
            ema_delta = ema_today - ema_yest
            ema_est_tmrw = ema_today + ema_delta

            buy_level = ema_est_tmrw - avg_pen
            stop_loss = buy_level - (1.5 * last_atr_1h) if last_atr_1h > 0 else buy_level * 0.98
            risk = buy_level - stop_loss
            target_1 = ema_est_tmrw + avg_up_pen
            target_2 = buy_level + (risk * 2)

            st.markdown(f"""
            **Hesaplamalar ve Strateji (Buy Limit & Hedefler):**
            * 📌 **Güncel EMA (13):** {ema_today:.2f} | **Bir Sonraki Tahmini EMA:** {ema_est_tmrw:.2f}
            * 🟢 **Önerilen Alış Seviyesi (Buy Limit): {buy_level:.2f}**
            * 🔴 **Zarar Kes (Stop-Loss): {stop_loss:.2f}** *(Risk: {risk:.2f})*
            * 🎯 **Hedef 1 (Kısa Vade): {target_1:.2f}**
            * 🚀 **Hedef 2 (Trend - 1:2 RR): {target_2:.2f}**
            """)

            fig3 = go.Figure()
            fig3.add_trace(go.Candlestick(x=df_1h.index, open=df_1h["Open"], high=df_1h["High"], low=df_1h["Low"], close=df_1h["Close"], name="Price"))
            fig3.add_trace(go.Scatter(x=df_1h.index, y=ema_1h, name="EMA 13", line=dict(color='blue')))
            next_time = df_1h.index[-1] + pd.Timedelta(hours=1)
            fig3.add_trace(go.Scatter(x=[next_time], y=[ema_est_tmrw], mode='markers', marker=dict(size=10, color='orange'), name="Tahmini EMA"))
            fig3.add_hline(y=target_2, line_dash="dash", line_color="darkgreen", annotation_text="Hedef 2 (1:2 RR)")
            fig3.add_hline(y=target_1, line_dash="dashdot", line_color="cyan", annotation_text="Hedef 1 (Simetrik)")
            fig3.add_hline(y=buy_level, line_dash="dash", line_color="lime", annotation_text="Limit Alış Seviyesi")
            fig3.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop-Loss (1.5 ATR)")
            fig3.update_layout(title="1 Saatlik Giriş/Çıkış Stratejisi (Alış, Hedef ve Stop)", height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig3, use_container_width=True)

            fig3_adx = go.Figure()
            fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=adx_1h, name="ADX", line=dict(color='black', width=2.5)))
            fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=pdi_1h, name="+DI", line=dict(color='green')))
            fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=mdi_1h, name="-DI", line=dict(color='red')))
            fig3_adx.add_hline(y=25, line_dash="dash", line_color="gray")
            fig3_adx.add_hline(y=50, line_dash="dot", line_color="purple")
            fig3_adx.update_layout(title="1 Saatlik ADX ve Yön Göstergeleri (+DI / -DI)", height=250, template="plotly_dark")
            st.plotly_chart(fig3_adx, use_container_width=True)

st.markdown("---")
st.caption(
    f"Son güncelleme: {datetime.now(timezone.utc).astimezone().strftime('%d.%m.%Y %H:%M:%S')} | "
    "Bu uygulama eğitim/analiz amaçlıdır, yatırım tavsiyesi değildir."
)
