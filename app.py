import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import concurrent.futures
import time
import feedparser
import google.generativeai as genai

# --- MARKA VE SAYFA AYARLARI ---
st.set_page_config(page_title="MarketPlus By Valiyev | Master 5 AI Pro", layout="wide", page_icon="💹")

# Orijinal Profesyonel Görünüm için CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1e2130; color: white; border: 1px solid #30363d; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- BORSA MOTORU (Sadece burası yfinance yerine ccxt oldu) ---
exchange = ccxt.kucoin({'enableRateLimit': True})

# --- TEKNİK ANALİZ MOTORU (Orijinal formasyonlar ve indikatörler) ---
def calculate_master_indicators(df):
    if df.empty or len(df) < 50:
        return df
    
    df["EMA9"] = ta.ema(df["Close"], length=9)
    df["EMA21"] = ta.ema(df["Close"], length=21)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    
    df["RSI"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)
    
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Kangaroo_Bull"] = (df["Lower_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] < 35)
    df["Kangaroo_Bear"] = (df["Upper_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] > 65)
    
    return df

@st.cache_data(ttl=120)
def get_crypto_data(symbol, timeframe="1d", limit=200):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return calculate_master_indicators(df)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_top_symbols():
    try:
        tickers = exchange.fetch_tickers()
        symbols = [s for s, t in tickers.items() if s.endswith('/USDT') and ':' not in s]
        return sorted(symbols)
    except:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

def fetch_crypto_news(symbol):
    base_coin = symbol.split('/')[0]
    feed = feedparser.parse("https://tr.cointelegraph.com/rss")
    news_list = []
    for entry in feed.entries[:15]:
        if base_coin in entry.title or base_coin in entry.description or "Kripto" in entry.title:
            news_list.append({"title": entry.title, "link": entry.link, "date": entry.published})
    return news_list if news_list else feed.entries[:5]

# --- YAN MENÜ (Birebir Aynı) ---
st.sidebar.image("https://img.icons8.com/fluent/96/000000/cryptocurrency.png", width=80)
st.sidebar.title("Master 5 AI Pro")
st.sidebar.markdown("---")

all_symbols = get_top_symbols()
selected_symbol = st.sidebar.selectbox("Sembol Seçimi", all_symbols, index=all_symbols.index("BTC/USDT") if "BTC/USDT" in all_symbols else 0)
selected_tf = st.sidebar.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d", "1w"], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("Analiz Ayarları")
use_fa = st.sidebar.checkbox("Temel Verileri Göster", value=True)
use_ta = st.sidebar.checkbox("Teknik İndikatörleri Çiz", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Entegrasyonu")
ai_api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- ANA PANEL SEKMELERİ ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Master Analiz", "🤖 AI Yorumu", "📰 Haber Analizi", "🔍 Çoklu Tarayıcı (Screener)"])

# SEKME 1: ANALİZ
with tab1:
    with st.spinner(f"{selected_symbol} Analiz Ediliyor..."):
        df = get_crypto_data(selected_symbol, selected_tf)
        
        if not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            m1, m2, m3, m4, m5 = st.columns(5)
            change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
            
            m1.metric("Son Fiyat", f"{last['Close']:.4f}", f"{change:.2f}%")
            m2.metric("RSI (14)", f"{last['RSI']:.1f}", "Aşırı Satım" if last['RSI'] < 30 else ("Aşırı Alım" if last['RSI'] > 70 else "Normal"))
            
            trend_status = "Yükseliş 🟢" if last['Close'] > last['SMA50'] else "Düşüş 🔴"
            m3.metric("Trend (SMA50)", trend_status)
            
            vol_status = "Yüksek" if last['ATR'] > df['ATR'].mean() else "Düşük"
            m4.metric("Volatilite", vol_status)
            
            sig = "Nötr"
            if last['Kangaroo_Bull']: sig = "AL 🦘"
            elif last['Kangaroo_Bear']: sig = "SAT 🦘"
            m5.metric("Master Sinyal", sig)

            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="Fiyat", increasing_line_color='#00ffcc', decreasing_line_color='#ff3366'
            ))
            
            if use_ta:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name="EMA 21", line=dict(color='yellow', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color='orange', width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA 200", line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name="Bant Üst", line=dict(color='gray', dash='dash', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name="Bant Alt", line=dict(color='gray', dash='dash', width=1)))

            fig.update_layout(
                height=700, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if use_fa:
                st.markdown("---")
                st.subheader("📋 MarketPlus İstatistikleri")
                f1, f2, f3 = st.columns(3)
                f1.info(f"**24s Hacim:** {last['Volume']:.2f}")
                f2.info(f"**Günlük Aralık:** {last['Low']:.4f} - {last['High']:.4f}")
                f3.info(f"**ATR Değeri:** {last['ATR']:.6f}")

# SEKME 2: AI YORUMU
with tab2:
    st.subheader(f"🤖 {selected_symbol} AI Analizi")
    if not ai_api_key:
        st.warning("Lütfen sol menüdeki 'AI Entegrasyonu' bölümüne API anahtarınızı girin.")
    elif not df.empty:
        if st.button("🧠 Yorum İste"):
            with st.spinner("AI analiz ediyor..."):
                try:
                    genai.configure(api_key=ai_api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"Sen bir finans analistisin. {selected_symbol} fiyatı {df['Close'].iloc[-1]}, RSI değeri {df['RSI'].iloc[-1]:.2f}. Buna göre kısa bir teknik analiz yorumu yap."
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Hata: {e}")

# SEKME 3: HABER ANALİZİ
with tab3:
    st.subheader(f"📰 {selected_symbol} Haberleri")
    if st.button("🔄 Haberleri Çek"):
        with st.spinner("Haberler yükleniyor..."):
            news = fetch_crypto_news(selected_symbol)
            for item in news:
                st.markdown(f"- **[{item['title']}]({item['link']})** ({item['date']})")

# SEKME 4: ÇOKLU TARAYICI (SCREENER)
with tab4:
    st.subheader("🌐 Kripto Piyasa Tarayıcısı")
    st.markdown("Aynı anda onlarca coini 'Master 5 AI' algoritmalarıyla tarayın.")
    
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    scan_count = sc_col1.slider("Taranacak Coin Sayısı", 10, 100, 40)
    scan_tf = sc_col2.selectbox("Tarama Periyodu", ["1h", "4h", "1d"], index=2)
    
    if st.button("🚀 Taramayı Başlat"):
        symbols_to_scan = all_symbols[:scan_count]
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {executor.submit(get_crypto_data, coin, scan_tf): coin for coin in symbols_to_scan}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_coin)):
                coin_name = future_to_coin[future]
                df_res = future.result()
                
                if not df_res.empty:
                    last_row = df_res.iloc[-1]
                    results.append({
                        "Sembol": coin_name,
                        "Fiyat": last_row["Close"],
                        "RSI": round(last_row["RSI"], 2),
                        "SMA50 Trend": "YUKARI 🟢" if last_row["Close"] > last_row["SMA50"] else "AŞAĞI 🔴",
                        "EMA21 Durum": "Üstünde" if last_row["Close"] > last_row["EMA21"] else "Altında",
                        "🦘 Sinyal": "AL ✅" if last_row["Kangaroo_Bull"] else ("SAT ❌" if last_row["Kangaroo_Bear"] else "-")
                    })
                
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                status.text(f"İşleniyor: {coin_name}")
                time.sleep(0.05) 
        
        status.empty()
        if results:
            st.success(f"{len(results)} coin başarıyla tarandı.")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
