import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="Kripto Tarayıcı & AI", layout="wide")

st.title("🚀 Kripto Piyasa Tarayıcısı (V1)")

# --- 1. VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=300) # 5 dakikada bir yenile
def fetch_data(symbol="BTC/USDT", timeframe="1d", limit=100):
    exchange = ccxt.binance()
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Hata: {e}")
        return pd.DataFrame()

# --- 2. ANA UYGULAMA ---
symbol = st.sidebar.selectbox("Coin Seç", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
timeframe = st.sidebar.selectbox("Zaman Dilimi", ["1d", "4h", "1h"], index=0)

with st.spinner(f"{symbol} verileri çekiliyor..."):
    df = fetch_data(symbol, timeframe)

if not df.empty:
    # Pandas-TA ile tek satırda RSI ve SMA hesaplama
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    st.subheader(f"{symbol} - Son Fiyat: {df['Close'].iloc[-1]:.2f}")
    
    # Hızlı Metrikler
    c1, c2 = st.columns(2)
    c1.metric("Güncel RSI", f"{df['RSI_14'].iloc[-1]:.1f}")
    c2.metric("Güncel Fiyat Farkedilebilirliği", "SMA 50 Üstünde ✅" if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] else "SMA 50 Altında ❌")

    # Plotly ile Fiyat Grafiği
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name="Fiyat")])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange')))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Ham Veriyi Gör"):
        st.dataframe(df.tail(10))
