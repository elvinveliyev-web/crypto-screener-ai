import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import concurrent.futures
import time

# --- SAYFA VE MARKA AYARLARI ---
st.set_page_config(page_title="MarketPlus By Valiyev | Crypto AI", layout="wide", page_icon="💹")

# Özel CSS ile daha profesyonel bir görünüm
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("💹 MarketPlus By Valiyev | Crypto AI Pro")
st.caption("Master 5 AI Algoritması ile Kripto Para Teknik Analiz ve Tarama Sistemi")

# --- BORSA BAĞLANTISI ---
exchange = ccxt.kucoin({'enableRateLimit': True})

# --- ANALİZ MOTORU (MASTER 5 LOGIC) ---
def apply_master_indicators(df):
    """Tüm teknik göstergeleri ve formasyonları hesaplar."""
    if df.empty or len(df) < 50:
        return df
    
    # Hareketli Ortalamalar
    df["EMA9"] = ta.ema(df["Close"], length=9)
    df["EMA21"] = ta.ema(df["Close"], length=21)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    
    # Momentum ve Volatilite
    df["RSI"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Özel Formasyon: Kanguru Kuyruğu (Kangaroo Tail) Tespiti
    # Basit mantık: Mumun iğnesi gövdesinden en az 2 kat büyükse ve uç noktadaysa
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    
    df["Kangaroo_Bull"] = (df["Lower_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] < 35)
    df["Kangaroo_Bear"] = (df["Upper_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] > 65)
    
    return df

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, timeframe="1d", limit=200):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return apply_master_indicators(df)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_market_symbols():
    try:
        tickers = exchange.fetch_tickers()
        pairs = [s for s, t in tickers.items() if s.endswith('/USDT') and ':' not in s]
        # Hacme göre sıralama yapılabilir, şimdilik popüler olanları başa alalım
        prio = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "BNB/USDT"]
        return prio + [p for p in pairs if p not in prio]
    except:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# --- SEKMELER ---
tab1, tab2 = st.tabs(["📊 Master Analiz", "🔍 Çoklu Tarayıcı (Screener)"])

with tab1:
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        symbols = get_market_symbols()
        selected_coin = st.selectbox("Sembol Seçin", symbols)
        selected_tf = st.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d", "1w"], index=3)
    
    df_ana = fetch_crypto_data(selected_coin, selected_tf)
    
    if not df_ana.empty:
        last = df_ana.iloc[-1]
        
        # Metrik Paneli
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fiyat", f"{last['Close']:.4f}")
        m2.metric("RSI (14)", f"{last['RSI']:.1f}")
        m3.metric("Trend", "Yükseliş 🟢" if last['Close'] > last['SMA50'] else "Düşüş 🔴")
        
        # Sinyal Durumu
        sig = "Nötr"
        if last['Kangaroo_Bull']: sig = "KANGURU KUYRUĞU (AL) 🦘"
        elif last['Kangaroo_Bear']: sig = "KANGURU KUYRUĞU (SAT) 🦘"
        m4.metric("Özel Sinyal", sig)

        # Gelişmiş Grafik
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_ana.index, open=df_ana['Open'], high=df_ana['High'], low=df_ana['Low'], close=df_ana['Close'], name="Fiyat"))
        fig.add_trace(go.Scatter(x=df_ana.index, y=df_ana['EMA21'], name="EMA 21", line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=df_ana.index, y=df_ana['SMA50'], name="SMA 50", line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df_ana.index, y=df_ana['SMA200'], name="SMA 200", line=dict(color='red', width=2)))
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Master AI Piyasa Taraması")
    scan_col1, scan_col2 = st.columns(2)
    count = scan_col1.slider("Taranacak Coin Sayısı", 10, 100, 30)
    stf = scan_col2.selectbox("Tarama Dilimi", ["1h", "4h", "1d"], index=2)
    
    if st.button("🚀 Piyasayı Tara"):
        all_symbols = get_market_symbols()[:count]
        results = []
        progress = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {executor.submit(fetch_crypto_data, coin, stf): coin for coin in all_symbols}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_coin)):
                coin = future_to_coin[future]
                df_res = future.result()
                if not df_res.empty:
                    last_r = df_res.iloc[-1]
                    results.append({
                        "Sembol": coin,
                        "Fiyat": last_r["Close"],
                        "RSI": round(last_r["RSI"], 2),
                        "EMA 21/50": "Üstünde" if last_r["Close"] > last_r["SMA50"] else "Altında",
                        "Kangaroo": "AL 🦘" if last_r["Kangaroo_Bull"] else ("SAT 🦘" if last_r["Kangaroo_Bear"] else "-")
                    })
                progress.progress((i + 1) / len(all_symbols))
                time.sleep(0.05)
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
