import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Advanced Stock Volatility Predictor", layout="wide")
st.title("🚀 Advanced Stock Volatility Predictor")
st.caption("Live News + Portfolio Comparison + Risk Forecast + Technical Indicators")

@st.cache_resource
def load_model():
    model = joblib.load('volatility_model.pkl')
    config = joblib.load('model_config.pkl')
    return model, config['features']

model, features = load_model()
analyzer = SentimentIntensityAnalyzer()

def clean_yfinance_df(df):
    if len(df) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if str(x) != '']).strip() for col in df.columns.values]
    rename_map = {}
    for col in df.columns:
        c = str(col).lower()
        if 'adj close' in c:
            rename_map[col] = 'Adj Close'
        elif c.startswith('close') or 'close_' in c:
            rename_map[col] = 'Close'
        elif c.startswith('open') or 'open_' in c:
            rename_map[col] = 'Open'
        elif c.startswith('high') or 'high_' in c:
            rename_map[col] = 'High'
        elif c.startswith('low') or 'low_' in c:
            rename_map[col] = 'Low'
        elif 'volume' in c:
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    return df

st.sidebar.header("📊 Stock Analysis")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()

st.sidebar.subheader("📰 Live News")
use_demo_news = st.sidebar.checkbox("Use demo live news")
if use_demo_news:
    demo_news = {
        "AAPL": "Apple unveils AI-powered iPhone 17. Shares surge on strong demand.",
        "TSLA": "Tesla delays robotaxi launch amid regulatory hurdles.",
        "MSFT": "Microsoft cloud revenue beats expectations again.",
        "NVDA": "NVIDIA expands AI chip sales as demand stays strong."
    }
    news_text = demo_news.get(ticker, "Company reports strong quarterly earnings and positive outlook.")
    st.sidebar.text_area("News Text", news_text, height=100)
else:
    news_text = st.sidebar.text_area(
        "Paste News / Social Media",
        "Apple announces record earnings. Investors are optimistic.",
        height=100
    )

def get_stock_data(tick, period="6mo"):
    df = yf.download(tick, period=period, progress=False, threads=False)
    df = clean_yfinance_df(df)
    return df

def add_indicators(df):
    df = df.copy()
    if 'Close' in df.columns:
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

def buy_sell_hold_signal(rsi, predicted_vol):
    if pd.isna(rsi) or pd.isna(predicted_vol):
        return "HOLD"
    if rsi < 30 and predicted_vol < 0.25:
        return "BUY"
    elif rsi > 70 or predicted_vol > 0.40:
        return "SELL"
    return "HOLD"

if st.sidebar.button("🔮 Predict Volatility", use_container_width=True):
    with st.spinner("Fetching live data..."):
        df = get_stock_data(ticker, period="6mo")

    if len(df) == 0:
        st.error("Invalid ticker or no data found.")
        st.stop()

    df = add_indicators(df)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    df['Returns'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(5).std() * np.sqrt(252)

    sentiment_score = analyzer.polarity_scores(news_text)['compound']

    latest_return = df['Returns'].dropna().iloc[-1] if len(df['Returns'].dropna()) > 0 else 0
    latest_volume = df['Volume'].dropna().iloc[-1] if len(df['Volume'].dropna()) > 0 else 0

    input_df = pd.DataFrame([{
        'Sentiment': sentiment_score,
        'Returns': latest_return,
        'Volume': latest_volume
    }])
    input_df = input_df[features]

    prediction = float(model.predict(input_df)[0])

    rsi_value = df['RSI'].dropna().iloc[-1] if 'RSI' in df.columns and len(df['RSI'].dropna()) > 0 else np.nan
    signal = buy_sell_hold_signal(rsi_value, prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Predicted Volatility", f"{prediction:.2%}")
    col2.metric("😐 Sentiment Score", f"{sentiment_score:.2f}")
    risk_score = min(prediction * 100, 100)
    col3.metric("⚠️ Risk Score", f"{risk_score:.0f}/100")

    col4, col5 = st.columns(2)
    col4.metric("📌 RSI", f"{rsi_value:.2f}" if not pd.isna(rsi_value) else "N/A")
    col5.metric("🟢 Signal", signal)

    st.subheader(f"📉 {ticker} Historical vs Predicted Volatility")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['Volatility'].dropna(), name='Historical Volatility', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(y=[prediction], x=[len(df)], name='Predicted', mode='markers+text',
                             marker=dict(size=18, color='red', symbol='star'),
                             text=[f"{prediction:.1%}"], textposition="top center"))
    fig.update_layout(xaxis_title="Days", yaxis_title="Volatility", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 30-Day Volatility Forecast")
    forecast_days = np.arange(30)
    forecast_vol = np.maximum(prediction + np.random.normal(0, prediction * 0.15, 30), 0.01)
    fig_forecast = px.line(x=forecast_days, y=forecast_vol * 100,
                           labels={'x': 'Days Ahead', 'y': 'Volatility %'},
                           title=f"{ticker} Expected Volatility Trend")
    fig_forecast.add_hline(y=prediction * 100, line_dash="dash", annotation_text="Today Prediction")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📊 Technical Indicators")
    tech_cols = st.columns(3)
    tech_cols[0].metric("SMA 20", f"{df['SMA_20'].dropna().iloc[-1]:.2f}" if 'SMA_20' in df.columns and len(df['SMA_20'].dropna()) > 0 else "N/A")
    tech_cols[1].metric("SMA 50", f"{df['SMA_50'].dropna().iloc[-1]:.2f}" if 'SMA_50' in df.columns and len(df['SMA_50'].dropna()) > 0 else "N/A")
    tech_cols[2].metric("RSI", f"{rsi_value:.2f}" if not pd.isna(rsi_value) else "N/A")

    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        st.line_chart(df[['Close', 'SMA_20', 'SMA_50']].dropna())

    st.subheader("📝 Detailed Sentiment Analysis")
    scores = analyzer.polarity_scores(news_text)
    s1, s2, s3 = st.columns(3)
    s1.metric("😊 Positive", f"{scores['pos']:.1%}")
    s2.metric("😠 Negative", f"{scores['neg']:.1%}")
    s3.metric("😐 Neutral", f"{scores['neu']:.1%}")

st.subheader("🏆 Live Tech Portfolio Comparison")
col1, col2, col3, col4 = st.columns(4)
tech_stocks = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA"
}

for i, (tick, name) in enumerate(tech_stocks.items()):
    with [col1, col2, col3, col4][i]:
        try:
            data = get_stock_data(tick, period="3mo")
            if len(data) > 15 and 'Close' in data.columns:
                vol = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric(f"{tick}\n{name}", f"{vol:.1f}%")
            else:
                st.metric(f"{tick}\n{name}", "N/A")
        except:
            st.metric(f"{tick}\n{name}", "Retry")

st.caption("⚡ Real-time volatility comparison | Green = low risk, Red = high risk")
st.markdown("---")




