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

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1e2130; color: white; border: 1px solid #30363d; font-weight: bold; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    .news-box { padding: 15px; border-radius: 8px; background-color: #1e2130; margin-bottom: 10px; border-left: 4px solid #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- BORSA MOTORU (KuCoin) ---
exchange = ccxt.kucoin({'enableRateLimit': True})

# --- TEKNİK ANALİZ VE FORMASYON MOTORU ---
def calculate_master_indicators(df):
    if df.empty or len(df) < 50:
        return df
    
    # 1. Trend ve Momentum (Master 5 Standartları)
    df["EMA9"] = ta.ema(df["Close"], length=9)
    df["EMA21"] = ta.ema(df["Close"], length=21)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # 2. Mum Formasyonları (Candlestick Patterns)
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    
    # Kanguru Kuyruğu
    df["Kangaroo_Bull"] = (df["Lower_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] < 35)
    df["Kangaroo_Bear"] = (df["Upper_Wick"] > (df["Body"] * 2.5)) & (df["RSI"] > 65)
    
    # Doji
    df['Doji'] = df['Body'] <= (df['High'] - df['Low']) * 0.1
    
    # Yutan Boğa / Ayı (Engulfing)
    df['Bullish_Engulfing'] = (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open']) & (df['Close'] >= df['Open'].shift(1)) & (df['Open'] <= df['Close'].shift(1))
    df['Bearish_Engulfing'] = (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'] < df['Open']) & (df['Close'] <= df['Open'].shift(1)) & (df['Open'] >= df['Close'].shift(1))
    
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
        prio = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "AVAX/USDT"]
        return prio + sorted([p for p in symbols if p not in prio])
    except:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

def fetch_crypto_news(symbol):
    """CoinTelegraph Türkiye RSS'inden kripto haberlerini çeker."""
    base_coin = symbol.split('/')[0]
    feed = feedparser.parse("https://tr.cointelegraph.com/rss")
    news_list = []
    for entry in feed.entries[:15]:
        # Eğer spesifik coin adı geçiyorsa veya genel kripto haberiyse al
        if base_coin in entry.title or base_coin in entry.description or "Kripto" in entry.title:
            news_list.append({"title": entry.title, "link": entry.link, "date": entry.published})
    return news_list if news_list else feed.entries[:5] # Eşleşme yoksa en son 5 haberi ver

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.image("https://img.icons8.com/fluent/96/000000/cryptocurrency.png", width=80)
st.sidebar.title("Master 5 AI Pro")
st.sidebar.caption("Kripto Piyasa Terminali")
st.sidebar.markdown("---")

all_symbols = get_top_symbols()
selected_symbol = st.sidebar.selectbox("Sembol Seçimi", all_symbols)
selected_tf = st.sidebar.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d", "1w"], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("Görünüm Ayarları")
use_fa = st.sidebar.checkbox("Market İstatistikleri", value=True)
use_ta = st.sidebar.checkbox("Teknik Çizgiler", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Entegrasyonu")
ai_api_key = st.sidebar.text_input("Gemini API Key (Yorum için)", type="password")

# --- ANA PANEL ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Master Analiz", "🤖 AI Yorumu", "📰 Haber Analizi", "🔍 Çoklu Tarayıcı"])

df = get_crypto_data(selected_symbol, selected_tf)

# SEKME 1: TEKNİK ANALİZ
with tab1:
    if not df.empty:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Metrikler
        m1, m2, m3, m4, m5 = st.columns(5)
        change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
        
        m1.metric("Son Fiyat", f"{last['Close']:.4f}", f"{change:.2f}%")
        m2.metric("RSI (14)", f"{last['RSI']:.1f}", "Aşırı Satım" if last['RSI'] < 30 else ("Aşırı Alım" if last['RSI'] > 70 else "Nötr"))
        m3.metric("Trend (SMA50)", "Yükseliş 🟢" if last['Close'] > last['SMA50'] else "Düşüş 🔴")
        m4.metric("MACD Sinyali", "AL 🟢" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "SAT 🔴")
        
        # Formasyon Tespiti
        formation = "Nötr"
        if last['Kangaroo_Bull']: formation = "AL 🦘 (Kanguru)"
        elif last['Kangaroo_Bear']: formation = "SAT 🦘 (Kanguru)"
        elif last['Bullish_Engulfing']: formation = "Yutan Boğa 📈"
        elif last['Bearish_Engulfing']: formation = "Yutan Ayı 📉"
        elif last['Doji']: formation = "Kararsız (Doji) ⚖️"
        m5.metric("Aktif Formasyon", formation)

        # Grafik
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Fiyat", increasing_line_color='#00ffcc', decreasing_line_color='#ff3366'
        ))
        
        if use_ta:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name="EMA 21", line=dict(color='yellow', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name="Bant Üst", line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name="Bant Alt", line=dict(color='gray', dash='dash')))

        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        if use_fa:
            st.markdown("---")
            f1, f2, f3 = st.columns(3)
            f1.info(f"**24s Hacim:** {last['Volume']:.2f}")
            f2.info(f"**Günlük Aralık:** {last['Low']:.4f} - {last['High']:.4f}")
            f3.info(f"**Volatilite (ATR):** {last['ATR']:.4f}")

# SEKME 2: AI YORUMU
with tab2:
    st.subheader(f"🤖 {selected_symbol} İçin Yapay Zeka Yorumu")
    
    if not ai_api_key:
        st.warning("Lütfen sol menüdeki 'AI Entegrasyonu' bölümüne Gemini API anahtarınızı girin.")
    elif not df.empty:
        if st.button("🧠 Master AI'den Yorum İste"):
            with st.spinner("AI piyasa verilerini analiz ediyor..."):
                try:
                    genai.configure(api_key=ai_api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    last = df.iloc[-1]
                    prompt = f"""
                    Sen profesyonel bir kripto para analistisin (MarketPlus By Valiyev Baş Analisti). 
                    Aşağıdaki teknik analiz verilerine göre {selected_symbol} için kısa, net ve yatırımcıyı yönlendirecek profesyonel bir piyasa yorumu yap.
                    
                    Veriler:
                    - Fiyat: {last['Close']}
                    - RSI: {last['RSI']:.2f}
                    - SMA50 Durumu: {'Fiyatın Altında (Destek)' if last['Close'] > last['SMA50'] else 'Fiyatın Üstünde (Direnç)'}
                    - MACD: {'Pozitif' if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else 'Negatif'}
                    - Tespit Edilen Formasyon: {'Kanguru Kuyruğu' if last['Kangaroo_Bull'] else 'Yutan Boğa' if last['Bullish_Engulfing'] else 'Nötr'}
                    
                    Yorumunda çok fazla teknik terime boğmadan ne anlama geldiğini açıkla. Sorumluluk reddi beyanı ekle.
                    """
                    response = model.generate_content(prompt)
                    st.success("Analiz Tamamlandı!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI bağlantı hatası: API anahtarınızı kontrol edin. Detay: {e}")

# SEKME 3: HABER ANALİZİ
with tab3:
    st.subheader(f"📰 {selected_symbol.split('/')[0]} ve Piyasa Haberleri")
    if st.button("🔄 Haberleri Güncelle"):
        with st.spinner("Güncel haberler çekiliyor..."):
            news = fetch_crypto_news(selected_symbol)
            for item in news:
                st.markdown(f"""
                <div class="news-box">
                    <h4 style='margin:0;'><a href="{item.get('link', '#')}" target="_blank" style="color: #00ffcc; text-decoration: none;">{item.get('title', 'Başlık Yok')}</a></h4>
                    <small style='color: #888;'>📅 {item.get('date', 'Tarih Yok')}</small>
                </div>
                """, unsafe_allow_html=True)

# SEKME 4: ÇOKLU TARAYICI (SCREENER)
with tab4:
    st.subheader("🌐 Master AI Kripto Tarayıcı")
    sc_col1, sc_col2 = st.columns(2)
    scan_count = sc_col1.slider("Taranacak Coin Sayısı", 10, 100, 30)
    scan_tf = sc_col2.selectbox("Tarama Periyodu", ["15m", "1h", "4h", "1d"], index=2)
    
    if st.button("🚀 Taramayı Başlat"):
        symbols_to_scan = all_symbols[:scan_count]
        results = []
        progress_bar = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {executor.submit(get_crypto_data, coin, scan_tf): coin for coin in symbols_to_scan}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_coin)):
                coin_name = future_to_coin[future]
                df_res = future.result()
                
                if not df_res.empty:
                    lr = df_res.iloc[-1]
                    sig = "AL ✅" if lr["Kangaroo_Bull"] or lr["Bullish_Engulfing"] else ("SAT ❌" if lr["Kangaroo_Bear"] or lr["Bearish_Engulfing"] else "-")
                    
                    results.append({
                        "Sembol": coin_name,
                        "Fiyat": lr["Close"],
                        "RSI": round(lr["RSI"], 2),
                        "Trend": "YUKARI 🟢" if lr["Close"] > lr["SMA50"] else "AŞAĞI 🔴",
                        "Formasyon Sinyali": sig
                    })
                progress_bar.progress((i + 1) / len(symbols_to_scan))
                time.sleep(0.05)
                
        if results:
            st.success("Tarama Başarılı!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
