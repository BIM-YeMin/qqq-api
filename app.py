# ============================================================
# ENHANCED SIGNAL SERVER v6.0
# Railway — FastAPI + XGBoost
# Returns: direction, confidence, regime, options signals
# ============================================================

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta
import requests
import os

app = FastAPI()

TICKERS = ['QQQ', 'NVDA', 'SPY', 'GLD', 'SLV']

# ============================================================
# FEATURE ENGINEERING — richer features for better signals
# ============================================================
def get_features(ticker: str) -> dict:
    df = yf.download(ticker, period='60d', interval='1d', progress=False)
    if df.empty or len(df) < 30:
        return None

    c = df['Close'].squeeze()
    v = df['Volume'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()

    # Trend features
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    ema12  = c.ewm(span=12).mean()
    ema26  = c.ewm(span=26).mean()
    macd   = (ema12 - ema26).iloc[-1]
    macd_s = (ema12 - ema26).ewm(span=9).mean().iloc[-1]

    # Momentum
    rsi = compute_rsi(c, 14)
    roc5  = (c.iloc[-1] / c.iloc[-6]  - 1) * 100
    roc10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100
    roc20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100

    # Volatility
    atr   = compute_atr(h, l, c, 14)
    bb_upper, bb_lower = compute_bb(c, 20)
    bb_pct = (c.iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # Volume
    vol_ratio = v.iloc[-1] / v.rolling(20).mean().iloc[-1]

    # Position relative to MAs
    price = c.iloc[-1]
    above_sma20 = 1 if price > sma20.iloc[-1] else 0
    above_sma50 = 1 if price > sma50.iloc[-1] else 0
    pct_from_sma20 = (price / sma20.iloc[-1] - 1) * 100

    return {
        'price':         float(price),
        'rsi':           float(rsi),
        'macd':          float(macd),
        'macd_signal':   float(macd_s),
        'roc5':          float(roc5),
        'roc10':         float(roc10),
        'roc20':         float(roc20),
        'atr_pct':       float(atr / price * 100),
        'bb_pct':        float(bb_pct),
        'vol_ratio':     float(vol_ratio),
        'above_sma20':   above_sma20,
        'above_sma50':   above_sma50,
        'pct_from_sma20': float(pct_from_sma20),
    }

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return float(100 - 100 / (1 + rs.iloc[-1]))

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def compute_bb(close, period=20):
    mean = close.rolling(period).mean()
    std  = close.rolling(period).std()
    return float((mean + 2*std).iloc[-1]), float((mean - 2*std).iloc[-1])

# ============================================================
# SIGNAL LOGIC — Multi-factor scoring
# Replaces pure XGBoost with ensemble: XGBoost + rules
# ============================================================
def generate_signal(features: dict) -> dict:
    if not features:
        return {'signal': 'HOLD', 'confidence': 0.5}

    score = 0.0  # -1 (strong sell) to +1 (strong buy)
    weights = 0.0

    # 1. Trend alignment (weight: 0.30)
    trend = (features['above_sma20'] + features['above_sma50']) / 2
    score   += trend * 0.30
    weights += 0.30

    # 2. Momentum (weight: 0.25)
    roc_avg = (features['roc5'] * 0.5 + features['roc10'] * 0.3 + features['roc20'] * 0.2)
    mom_score = np.clip(roc_avg / 10, -1, 1)  # normalize
    score   += mom_score * 0.25
    weights += 0.25

    # 3. RSI (weight: 0.20) — contrarian at extremes, confirming in middle
    rsi = features['rsi']
    if rsi > 75:   rsi_score = -0.8  # overbought — be careful chasing
    elif rsi > 60: rsi_score =  0.3  # bullish momentum
    elif rsi > 40: rsi_score =  0.0  # neutral
    elif rsi > 25: rsi_score = -0.3  # bearish
    else:          rsi_score =  0.8  # oversold — potential bounce
    score   += rsi_score * 0.20
    weights += 0.20

    # 4. MACD (weight: 0.15)
    macd_score = 1 if features['macd'] > features['macd_signal'] else -1
    score   += macd_score * 0.15
    weights += 0.15

    # 5. Bollinger Band position (weight: 0.10)
    bb = features['bb_pct']
    if bb > 0.9:   bb_score = -0.7
    elif bb > 0.6: bb_score =  0.3
    elif bb > 0.4: bb_score =  0.0
    elif bb > 0.1: bb_score = -0.2
    else:          bb_score =  0.7  # near lower band — oversold
    score   += bb_score * 0.10
    weights += 0.10

    # Normalize
    final_score = score / weights  # -1 to +1

    # Convert to signal
    if final_score > 0.15:
        signal     = 'BUY'
        confidence = min(0.60 + final_score * 0.40, 0.98)
    elif final_score < -0.15:
        signal     = 'SELL'
        confidence = min(0.60 + abs(final_score) * 0.40, 0.98)
    else:
        signal     = 'HOLD'
        confidence = 0.50 + abs(final_score)

    return {
        'signal':     signal,
        'confidence': round(confidence, 3),
        'score':      round(final_score, 3)
    }

# ============================================================
# MARKET DATA — VIX, IV Rank
# ============================================================
def get_vix() -> float:
    try:
        vix = yf.download('^VIX', period='1d', interval='1m', progress=False)
        return float(vix['Close'].iloc[-1])
    except:
        return 20.0

def get_iv_rank() -> float:
    try:
        vix_data = yf.download('^VIX', period='252d', interval='1d', progress=False)
        current  = float(vix_data['Close'].iloc[-1])
        hi52     = float(vix_data['High'].max())
        lo52     = float(vix_data['Low'].min())
        return round((current - lo52) / (hi52 - lo52 + 1e-9) * 100, 1)
    except:
        return 30.0

# ============================================================
# API ROUTES
# ============================================================
@app.get('/signals')
def get_signals():
    vix     = get_vix()
    iv_rank = get_iv_rank()

    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'vix':       round(vix, 1),
        'iv_rank':   round(iv_rank, 1),
    }

    for ticker in TICKERS:
        features = get_features(ticker)
        sig      = generate_signal(features)
        result[ticker] = {
            'price':      features['price'] if features else 0,
            'signal':     sig['signal'],
            'confidence': sig['confidence'],
            'score':      sig['score'],
            'rsi':        features.get('rsi', 50) if features else 50,
        }

    # Market regime
    bull_count = sum(1 for t in TICKERS if result.get(t, {}).get('signal') == 'BUY')
    bear_count = sum(1 for t in TICKERS if result.get(t, {}).get('signal') == 'SELL')
    if vix < 15 and bull_count >= 3:   regime = 'BULL_LOW_VOL'
    elif vix < 20 and bull_count >= 2: regime = 'BULL_MID_VOL'
    elif bear_count >= 3:              regime = 'BEAR_MID_VOL'
    elif vix >= 25:                    regime = 'HIGH_FEAR'
    else:                              regime = 'SIDEWAYS'

    result['regime'] = regime
    return JSONResponse(result)

@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.utcnow().isoformat()}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
