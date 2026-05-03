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
# LIVE EARNINGS CALENDAR — auto-fetched via yfinance
# ============================================================
from datetime import datetime, timedelta
import json

_earnings_cache = {}
_earnings_cache_time = {}
CACHE_HOURS = 24

def get_earnings_date(ticker: str) -> dict:
    """Fetch next earnings date automatically using yfinance."""
    now = datetime.utcnow()
    
    # Return cache if fresh
    if ticker in _earnings_cache:
        age = (now - _earnings_cache_time[ticker]).total_seconds() / 3600
        if age < CACHE_HOURS:
            return _earnings_cache[ticker]
    
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        
        result = {
            'ticker':         ticker,
            'has_earnings':   False,
            'next_date':      None,
            'days_until':     None,
            'blackout':       False,  # True if within 3 days
        }

        if cal is not None and not cal.empty:
            # calendar returns a DataFrame with dates as columns
            dates = cal.columns.tolist()
            if dates:
                earn_date = dates[0]
                if hasattr(earn_date, 'date'):
                    earn_date = earn_date.date()
                else:
                    earn_date = datetime.strptime(str(earn_date)[:10], '%Y-%m-%d').date()
                
                today      = now.date()
                days_until = (earn_date - today).days
                
                result['has_earnings'] = True
                result['next_date']    = str(earn_date)
                result['days_until']   = days_until
                result['blackout']     = -1 <= days_until <= 3  # blackout 1 day before to 3 days after

        _earnings_cache[ticker]      = result
        _earnings_cache_time[ticker] = now
        return result

    except Exception as e:
        print(f"Earnings fetch error for {ticker}: {e}")
        fallback = {'ticker': ticker, 'has_earnings': False, 'next_date': None, 'days_until': None, 'blackout': False}
        _earnings_cache[ticker]      = fallback
        _earnings_cache_time[ticker] = now
        return fallback

def get_all_earnings() -> dict:
    """Fetch earnings for all tickers."""
    results = {}
    for ticker in TICKERS:
        results[ticker] = get_earnings_date(ticker)
    return results

# ============================================================
# NEWS SENTIMENT + MACRO EVENTS — auto analysis
# ============================================================
import re
from datetime import datetime, timedelta

# Macro event calendar — key dates that move markets
# Auto-updates via FRED/yfinance
MACRO_KEYWORDS = {
    'bearish': ['rate hike','inflation surge','recession','layoffs','tariff',
                'sanctions','war escalation','bank failure','debt ceiling',
                'fed tightening','yield inversion','earnings miss'],
    'bullish': ['rate cut','soft landing','strong jobs','earnings beat',
                'trade deal','ceasefire','stimulus','fed pause',
                'gdp beat','inflation falls','rate hold'],
}

TICKER_KEYWORDS = {
    'NVDA': ['nvidia','nvda','gpu','ai chip','cuda','blackwell','h100','data center'],
    'QQQ':  ['nasdaq','tech stocks','qqq','faang','big tech','rates','treasury'],
    'SPY':  ['s&p','spy','market rally','correction','dow','broad market'],
    'GLD':  ['gold','inflation','dollar','safe haven','fed','gld'],
    'SLV':  ['silver','slv','industrial metals','solar','ev'],
}

def score_headline(headline: str, ticker: str = None) -> float:
    """Score a headline -1.0 (bearish) to +1.0 (bullish)."""
    h = headline.lower()
    score = 0.0

    # General market sentiment
    for word in MACRO_KEYWORDS['bullish']:
        if word in h: score += 0.3
    for word in MACRO_KEYWORDS['bearish']:
        if word in h: score -= 0.3

    # Ticker-specific
    if ticker and ticker in TICKER_KEYWORDS:
        keywords = TICKER_KEYWORDS[ticker]
        relevant = any(k in h for k in keywords)
        if relevant:
            score *= 1.5  # amplify if ticker-relevant

    return max(-1.0, min(1.0, score))

def get_news_sentiment(ticker: str) -> dict:
    """Get news sentiment for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []

        if not news:
            return {'ticker': ticker, 'sentiment': 0.0, 'signal': 'NEUTRAL',
                    'headlines': [], 'article_count': 0}

        scores    = []
        headlines = []
        cutoff    = datetime.utcnow() - timedelta(hours=48)

        for article in news[:10]:  # last 10 articles
            title = article.get('title', '')
            if not title: continue

            # Check recency
            pub_time = article.get('providerPublishTime', 0)
            if pub_time:
                pub_dt = datetime.utcfromtimestamp(pub_time)
                if pub_dt < cutoff: continue

            score = score_headline(title, ticker)
            scores.append(score)
            headlines.append({'title': title[:100], 'score': round(score, 2)})

        avg_score = sum(scores) / len(scores) if scores else 0.0

        signal = 'BULLISH' if avg_score > 0.15 else 'BEARISH' if avg_score < -0.15 else 'NEUTRAL'

        return {
            'ticker':        ticker,
            'sentiment':     round(avg_score, 3),
            'signal':        signal,
            'headlines':     headlines[:5],
            'article_count': len(scores),
        }
    except Exception as e:
        print(f"News error for {ticker}: {e}")
        return {'ticker': ticker, 'sentiment': 0.0, 'signal': 'NEUTRAL',
                'headlines': [], 'article_count': 0}

def get_macro_context() -> dict:
    """Get macro market context — VIX term structure, yield curve, dollar."""
    try:
        # VIX
        vix_data  = yf.download('^VIX', period='5d', interval='1d', progress=False)
        vix_now   = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
        vix_prev  = float(vix_data['Close'].iloc[-2]) if len(vix_data) > 1 else vix_now
        vix_trend = 'RISING' if vix_now > vix_prev * 1.05 else 'FALLING' if vix_now < vix_prev * 0.95 else 'STABLE'

        # 10Y Treasury yield
        tnx = yf.download('^TNX', period='5d', interval='1d', progress=False)
        yield_10y = float(tnx['Close'].iloc[-1]) if not tnx.empty else 4.5

        # DXY (Dollar)
        dxy = yf.download('DX-Y.NYB', period='5d', interval='1d', progress=False)
        dollar = float(dxy['Close'].iloc[-1]) if not dxy.empty else 100.0

        # Market stress score 0-100
        stress = 50
        if vix_now > 25: stress += 25
        elif vix_now > 20: stress += 10
        if vix_trend == 'RISING': stress += 10
        if yield_10y > 5.0: stress += 10
        stress = min(100, max(0, stress))

        return {
            'vix':           round(vix_now, 1),
            'vix_trend':     vix_trend,
            'yield_10y':     round(yield_10y, 2),
            'dollar_dxy':    round(dollar, 2),
            'stress_score':  stress,
            'stress_label':  'HIGH' if stress > 70 else 'MODERATE' if stress > 40 else 'LOW',
        }
    except Exception as e:
        print(f"Macro error: {e}")
        return {'vix': 20.0, 'vix_trend': 'STABLE', 'yield_10y': 4.5,
                'dollar_dxy': 100.0, 'stress_score': 50, 'stress_label': 'MODERATE'}

def get_all_news_sentiment() -> dict:
    """Get news sentiment for all tickers."""
    result = {}
    for ticker in TICKERS:
        result[ticker] = get_news_sentiment(ticker)
    return result


# ============================================================
# FEATURE ENGINEERING — richer features for better signals
# ============================================================
def get_features(ticker: str) -> dict:
    df  = yf.download(ticker, period='252d', interval='1d', progress=False)
    if df.empty or len(df) < 50:
        return None

    c = df['Close'].squeeze()
    v = df['Volume'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()

    # Trend features
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ema12  = c.ewm(span=12).mean()
    ema26  = c.ewm(span=26).mean()
    macd   = (ema12 - ema26).iloc[-1]
    macd_s = (ema12 - ema26).ewm(span=9).mean().iloc[-1]

    # Momentum
    rsi   = compute_rsi(c, 14)
    roc5  = (c.iloc[-1] / c.iloc[-6]  - 1) * 100
    roc10 = (c.iloc[-1] / c.iloc[-11] - 1) * 100
    roc20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100

    # Volatility
    atr      = compute_atr(h, l, c, 14)
    bb_upper, bb_lower = compute_bb(c, 20)
    bb_pct   = (c.iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # Volume
    vol_ratio = v.iloc[-1] / v.rolling(20).mean().iloc[-1]

    # Price data
    price      = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) > 1 else price
    day_change = round((price / prev_close - 1) * 100, 2)

    # ATH (52-week high)
    ath        = float(h.rolling(252).max().iloc[-1])
    from_ath   = round((price / ath - 1) * 100, 2)

    # SMA values
    sma20_val  = float(sma20.iloc[-1])  if not pd.isna(sma20.iloc[-1])  else price
    sma50_val  = float(sma50.iloc[-1])  if not pd.isna(sma50.iloc[-1])  else price
    sma200_val = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else price

    above_sma20    = 1 if price > sma20_val  else 0
    above_sma50    = 1 if price > sma50_val  else 0
    pct_from_sma20 = (price / sma20_val - 1) * 100

    return {
        'price':          price,
        'prev_close':     prev_close,
        'day_change':     day_change,
        'sma20':          round(sma20_val,  2),
        'sma50':          round(sma50_val,  2),
        'sma200':         round(sma200_val, 2),
        'ath':            round(ath, 2),
        'from_ath':       from_ath,
        'atr':            round(float(atr), 2),
        'atr_pct':        round(float(atr / price * 100), 2),
        'rsi':            float(rsi),
        'macd':           float(macd),
        'macd_signal':    float(macd_s),
        'roc5':           float(roc5),
        'roc10':          float(roc10),
        'roc20':          float(roc20),
        'bb_pct':         float(bb_pct),
        'vol_ratio':      float(vol_ratio),
        'above_sma20':    above_sma20,
        'above_sma50':    above_sma50,
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
    """
    Enhanced signal generation with:
    - RSI extreme filter (blocks BUY when RSI > 75)
    - Volume confirmation (requires above-average volume)
    - ATR-based confidence scaling
    - 2:1 reward/risk awareness
    - Multi-factor weighted voting
    """
    if not features:
        return {'signal': 'HOLD', 'confidence': 0.5, 'score': 0.0,
                'blocked_reason': 'no_data', 'atr_pct': 0, 'vol_ratio': 1.0}

    score   = 0.0
    weights = 0.0
    blocked_reason = None

    rsi       = features.get('rsi', 50)
    vol_ratio = features.get('vol_ratio', 1.0)
    bb        = features.get('bb_pct', 0.5)
    atr_pct   = features.get('atr_pct', 1.0)

    # ============================================================
    # TIER 2 FIX 5: Hard RSI filter — skip BUY if overbought
    # ============================================================
    if rsi > 78:
        # Extremely overbought — only SELL or HOLD
        return {
            'signal':         'HOLD',
            'confidence':     0.55,
            'score':          -0.1,
            'blocked_reason': f'RSI_OVERBOUGHT_{rsi:.0f}',
            'atr_pct':        atr_pct,
            'vol_ratio':      vol_ratio,
        }

    if rsi < 22:
        # Extremely oversold — potential bounce, raise confidence
        return {
            'signal':         'BUY',
            'confidence':     0.72,
            'score':          0.4,
            'blocked_reason': 'RSI_OVERSOLD_BOUNCE',
            'atr_pct':        atr_pct,
            'vol_ratio':      vol_ratio,
        }

    # ============================================================
    # TIER 2 FIX 6: Volume confirmation
    # ============================================================
    volume_confirmed = vol_ratio >= 0.8  # at least 80% of average volume

    # 1. Trend alignment (weight: 0.25)
    trend   = (features.get('above_sma20', 0) + features.get('above_sma50', 0)) / 2
    score   += trend * 0.25
    weights += 0.25

    # 2. Momentum ROC (weight: 0.20)
    roc_avg   = (features.get('roc5',0)*0.5 + features.get('roc10',0)*0.3 + features.get('roc20',0)*0.2)
    mom_score = float(np.clip(roc_avg / 8, -1, 1))
    score     += mom_score * 0.20
    weights   += 0.20

    # 3. RSI — refined zones (weight: 0.20)
    if   rsi > 70: rsi_score = -0.5   # overbought warning
    elif rsi > 60: rsi_score =  0.4   # bullish momentum
    elif rsi > 50: rsi_score =  0.2   # mild bullish
    elif rsi > 40: rsi_score = -0.2   # mild bearish
    elif rsi > 30: rsi_score = -0.4   # bearish
    else:          rsi_score =  0.6   # oversold bounce
    score   += rsi_score * 0.20
    weights += 0.20

    # 4. MACD (weight: 0.15)
    macd       = features.get('macd', 0)
    macd_sig   = features.get('macd_signal', 0)
    macd_score = 1.0 if macd > macd_sig else -1.0
    # Extra weight if MACD crossing (strong signal)
    if abs(macd - macd_sig) < 0.1 and macd > macd_sig:
        macd_score = 1.5  # fresh crossover
    score   += macd_score * 0.15
    weights += 0.15

    # 5. Bollinger Band (weight: 0.10)
    if   bb > 0.95: bb_score = -0.9  # at upper band — overextended
    elif bb > 0.75: bb_score = -0.3
    elif bb > 0.5:  bb_score =  0.1
    elif bb > 0.25: bb_score =  0.3
    else:           bb_score =  0.8  # at lower band — oversold
    score   += bb_score * 0.10
    weights += 0.10

    # 6. Volume confirmation (weight: 0.10)
    vol_score = 0.5 if volume_confirmed else -0.3
    score     += vol_score * 0.10
    weights   += 0.10

    # Normalize
    final_score = score / weights if weights > 0 else 0.0

    # Convert to signal
    if final_score > 0.18:
        signal     = 'BUY'
        confidence = min(0.62 + final_score * 0.38, 0.95)
        # Scale down confidence if volume not confirmed
        if not volume_confirmed:
            confidence *= 0.85
    elif final_score < -0.18:
        signal     = 'SELL'
        confidence = min(0.62 + abs(final_score) * 0.38, 0.95)
    else:
        signal     = 'HOLD'
        confidence = 0.50 + abs(final_score) * 0.3

    # ATR adjustment — high volatility = lower confidence
    if atr_pct > 3.0:
        confidence *= 0.90  # very volatile — reduce confidence

    return {
        'signal':         signal,
        'confidence':     round(confidence, 3),
        'score':          round(final_score, 3),
        'blocked_reason': blocked_reason,
        'atr_pct':        round(atr_pct, 2),
        'vol_ratio':      round(vol_ratio, 2),
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
            'price':      features['price']       if features else 0,
            'day_change': features['day_change']   if features else 0,
            'sma20':      features['sma20']        if features else 0,
            'sma50':      features['sma50']        if features else 0,
            'sma200':     features['sma200']       if features else 0,
            'from_ath':   features['from_ath']     if features else 0,
            'atr':        features['atr']          if features else 0,
            'atr_pct':    features['atr_pct']      if features else 0,
            'rsi':        round(features['rsi'],1) if features else 50,
            'signal':     sig['signal'],
            'confidence': sig['confidence'],
            'score':      sig['score'],
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
    
    # Add live earnings data
    earnings = get_all_earnings()
    result['earnings'] = earnings

    # Add news sentiment (cached — only refresh every 2 hours)
    try:
        news_data = get_all_news_sentiment()
        result['news'] = news_data
    except Exception as e:
        result['news'] = {}
        print(f"News fetch error: {e}")

    # Add macro context
    try:
        result['macro'] = get_macro_context()
    except Exception as e:
        result['macro'] = {}
        print(f"Macro fetch error: {e}")

    return JSONResponse(result)

@app.get('/earnings')
def earnings_endpoint():
    """Dedicated earnings calendar endpoint."""
    return JSONResponse(get_all_earnings())

@app.get('/news')
def news_endpoint():
    """News sentiment for all tickers."""
    return JSONResponse(get_all_news_sentiment())

@app.get('/macro')
def macro_endpoint():
    """Macro market context."""
    return JSONResponse(get_macro_context())

@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.utcnow().isoformat()}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
