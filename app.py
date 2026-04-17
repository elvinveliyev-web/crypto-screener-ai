import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import concurrent.futures
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kripto Tarayıcı & AI", layout="wide", page_icon="🚀")
st.title("🚀 Kripto Piyasa Tarayıcısı (V1)")

# --- BORSA BAĞLANTISI ---
# Binance coğrafi engelini aşmak için KuCoin kullanıyoruz. API Key gerekmez.
exchange = ccxt.kucoin({
    'enableRateLimit': True, # CCXT'nin kendi hız koruma sistemi
})

# --- VERİ ÇEKME FONKSİYONLARI ---
@st.cache_data(ttl=3600) # En yüksek hacimli coinleri saatte bir yenile
def get_top_usdt_symbols(limit=50):
    """Borsadaki en yüksek hacimli USDT paritelerini bulur."""
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            # Sadece USDT spot paritelerini al ve hacim verisi olanları filtrele
            if symbol.endswith('/USDT') and ':' not in symbol and ticker.get('quoteVolume') is not None:
                usdt_pairs.append({
                    'symbol': symbol,
                    'volume': ticker['quoteVolume']
                })
        
        # Hacme göre büyükten küçüğe sırala ve en baştakileri al
        df = pd.DataFrame(usdt_pairs)
        df = df.sort_values(by='volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except Exception as e:
        st.error(f"Sembol listesi çekilemedi: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

def fetch_data(symbol, timeframe="1d", limit=100):
    """Tek bir coinin OHLCV mum verisini çeker."""
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def analyze_symbol(symbol, timeframe):
    """Screener için tek bir coini analiz edip özet metrikleri döndürür."""
    df = fetch_data(symbol, timeframe, limit=100)
    if df.empty or len(df) < 50:
        return None
    
    # İndikatörleri hesapla
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    
    last_close = df["Close"].iloc[-1]
    last_rsi = df["RSI_14"].iloc[-1]
    last_sma = df["SMA_50"].iloc[-1]
    
    # RSI Sinyali
    rsi_signal = "Nötr"
    if last_rsi < 30:
        rsi_signal = "Aşırı Satım (Alış Fırsatı) 🟢"
    elif last_rsi > 70:
        rsi_signal = "Aşırı Alım (Satış Riski) 🔴"
        
    # Trend Sinyali
    trend = "Yükseliş ↗" if last_close > last_sma else "Düşüş ↘"
    
    return {
        "Sembol": symbol,
        "Fiyat": round(last_close, 4),
        "RSI": round(last_rsi, 1),
        "RSI Sinyali": rsi_signal,
        "Trend (SMA 50)": trend
    }

# --- ARAYÜZ (SEKMELER) ---
tab1, tab2 = st.tabs(["🔎 Tekil Analiz", "🌐 Toplu Tarama (Screener)"])

with tab1:
    st.header("Detaylı Grafik ve Analiz")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        top_coins = get_top_usdt_symbols(limit=100)
        selected_symbol = st.selectbox("Coin Seç", top_coins, index=0)
        selected_timeframe = st.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d", "1w"], index=3)
    
    with col2:
        with st.spinner(f"{selected_symbol} grafiği yükleniyor..."):
            df_single = fetch_data(selected_symbol, selected_timeframe, limit=150)
            
            if not df_single.empty:
                df_single["SMA_50"] = ta.sma(df_single["Close"], length=50)
                
                # Plotly Grafiği
                fig = go.Figure(data=[go.Candlestick(x=df_single.index,
                                open=df_single['Open'], high=df_single['High'],
                                low=df_single['Low'], close=df_single['Close'], name="Fiyat")])
                fig.add_trace(go.Scatter(x=df_single.index, y=df_single['SMA_50'], name="SMA 50", line=dict(color='orange', width=2)))
                fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, template="plotly_dark")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Veri çekilemedi. Başka bir zaman dilimi veya coin deneyin.")

with tab2:
    st.header("Piyasa Tarayıcısı")
    st.markdown("En yüksek hacimli coinleri aynı anda tarar ve belirlediğin indikatörlere göre listeler.")
    
    s_col1, s_col2, s_col3 = st.columns(3)
    scan_limit = s_col1.slider("Kaç Coin Taranacak?", min_value=10, max_value=100, value=30, step=10)
    scan_timeframe = s_col2.selectbox("Tarama Zaman Dilimi", ["15m", "1h", "4h", "1d"], index=3)
    
    if st.button("🚀 Tarayıcıyı Çalıştır"):
        scan_symbols = get_top_usdt_symbols(limit=scan_limit)
        results = []
        
        # İlerleme çubuğu (Progress bar)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Hız sınırlarına takılmamak ve hızlı sonuç almak için ThreadPool (Paralel İşlem)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(analyze_symbol, sym, scan_timeframe): sym for sym in scan_symbols}
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                completed += 1
                progress_bar.progress(completed / len(scan_symbols))
                status_text.text(f"Taranıyor: %{int((completed / len(scan_symbols)) * 100)} tamamlandı.")
                
                res = future.result()
                if res:
                    results.append(res)
                
                # API limitlerini korumak için çok hafif bir fren
                time.sleep(0.05)
                
        status_text.empty() # İşlem bitince yazıyı sil
        
        if results:
            df_results = pd.DataFrame(results)
            
            # Sonuçları DataFrame olarak göster
            st.success("Tarama Tamamlandı!")
            
            # Sadece aşırı satım olanları öne çıkarma butonu
            if st.checkbox("Sadece Aşırı Satımda Olanları (RSI < 30) Göster"):
                df_results = df_results[df_results["RSI"] < 30]
                if df_results.empty:
                    st.info("Şu an RSI'ı 30'un altında olan coin bulunmuyor.")
            
            st.dataframe(df_results, use_container_width=True)
        else:
            st.error("Tarama sonucunda veri bulunamadı.")
