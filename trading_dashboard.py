import yfinance as yf
import pandas as pd
import ta
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =================
SYMBOL = "RELIANCE.NS"
INTERVAL = "5m"
PERIOD = "5d"

RSI_BUY = 30
RSI_SELL = 70
# ==========================================

st.set_page_config(
    page_title="Trading Bot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(f"📊 AI Trading Dashboard - {SYMBOL}")

# ---------- DATA ----------
@st.cache_data
def get_data():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)
    df.dropna(inplace=True)
    return df

# ---------- INDICATORS ----------
def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['returns'] = df['Close'].pct_change()

    return df

# ---------- AI MODEL ----------
def train_model(df):
    df = df.dropna()

    X = df[['rsi', 'ma50', 'returns']]
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    y = df['target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X[:-1], y[:-1])

    return model

def predict(model, df):
    latest = df.iloc[-1]
    X = [[latest['rsi'], latest['ma50'], latest['returns']]]
    prob = model.predict_proba(X)[0][1]
    return prob

# ---------- SIGNAL ----------
def get_signal(df, prob):
    latest = df.iloc[-1]
    price = latest['Close']
    rsi = latest['rsi']
    ma50 = latest['ma50']

    signal = "HOLD"

    if rsi < RSI_BUY and price > ma50 and prob > 0.6:
        signal = "BUY"
    elif rsi > RSI_SELL and prob < 0.4:
        signal = "SELL"

    return signal, price, rsi, ma50

# ---------- RUN ----------
df = get_data()
df = add_indicators(df)

model = train_model(df)
prob = predict(model, df)

signal, price, rsi, ma50 = get_signal(df, prob)

# ---------- METRICS ----------
col1, col2 = st.columns(2)

col1.metric("Price", f"{price:.2f}")
col2.metric("Signal", signal)

st.metric("RSI", f"{rsi:.2f}")
st.metric("AI Up Probability", f"{prob*100:.2f}%")

# ---------- CHART ----------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['ma50'],
    name="MA50"
))

st.plotly_chart(fig, use_container_width=True)

# ---------- RSI ----------
st.subheader("RSI Indicator")
st.line_chart(df['rsi'])
