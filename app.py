import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Stock Volatility Predictor", layout="wide")
st.title("📈 Stock Volatility Predictor using Sentiment Analysis")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
news_text = st.sidebar.text_area(
    "Paste Recent News / Social Media Text",
    "Apple announces record earnings. Investors are optimistic about future growth."
)

# Load model (cached)
@st.cache_resource
def load_model():
    model = joblib.load('volatility_model.pkl')
    config = joblib.load('model_config.pkl')
    return model, config['features']

try:
    model, features = load_model()
except FileNotFoundError:
    st.error("❌ Model files not found! Upload `volatility_model.pkl` and `model_config.pkl` to your repo.")
    st.stop()

if st.sidebar.button("🔮 Predict Volatility"):
    # Fetch recent stock data
    with st.spinner("Fetching stock data..."):
        df = yf.download(ticker, period="6mo")
    
    if len(df) == 0:
        st.error("Invalid ticker or no data found.")
        st.stop()
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    
    # Map columns to standard names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'adj close' in cl:
            col_map[col] = 'Adj Close'
        elif cl.startswith('close_') or cl == 'close':
            col_map[col] = 'Close'
        elif 'volume' in cl:
            col_map[col] = 'Volume'
    df = df.rename(columns=col_map)
    
    # Fallback if Adj Close missing
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    df['Returns'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(5).std() * np.sqrt(252)
    
    # Compute sentiment
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(news_text)['compound']
    
    # Latest features
    latest_return = df['Returns'].iloc[-1] if not pd.isna(df['Returns'].iloc[-1]) else 0
    
    # Handle Volume column name variation
    vol_col = 'Volume' if 'Volume' in df.columns else 'Vol'
    latest_volume = df[vol_col].iloc[-1]
    
    input_df = pd.DataFrame([{'Sentiment': sentiment_score, 'Returns': latest_return, 'Volume': latest_volume}])
    input_df = input_df[features]
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Predicted Next-Day Volatility (Annualized)", f"{prediction:.2%}")
    with col2:
        st.metric("😐 Sentiment Score", f"{sentiment_score:.2f} (-1 to 1)")
    
    # Plot
    st.subheader("📉 Historical vs Predicted Volatility")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['Volatility'],
        name='Historical Volatility',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=[prediction],
        name='Predicted',
        mode='markers',
        marker=dict(size=14, color='red', symbol='star')
    ))
    fig.update_layout(xaxis_title="Days", yaxis_title="Volatility", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment breakdown
    st.subheader("📝 Sentiment Breakdown")
    scores = analyzer.polarity_scores(news_text)
    st.json(scores)
