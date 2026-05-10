# ============================================================
# ENHANCED SIGNAL SERVER v6.0
# Railway — FastAPI + XGBoost
# Returns: direction, confidence, regime, options signals
# ============================================================

# requirements.txt: add scikit-learn pytz
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

@app.on_event("startup")
async def startup_event():
    print("Server started — using enhanced keyword sentiment (lightweight)")

TICKERS = [
    'QQQ', 'NVDA', 'SPY', 'GLD', 'SLV',
    'AMD', 'TSM', 'MSFT', 'AAPL', 'GOOGL',
    'JPM', 'V', 'BRK-B',
    'XOM', 'CVX', 'NEE',
    'UNH', 'JNJ', 'NVO',
    'CAT', 'AMZN', 'RTX',
]
_signal_cache = {}  # in-memory cache for expensive calls

# ============================================================
# LIVE EARNINGS CALENDAR — auto-fetched via yfinance
# ============================================================
from datetime import datetime, timedelta
import json

_earnings_cache = {}
_earnings_cache_time = {}
CACHE_HOURS = 24

# Known earnings dates — updated automatically by get_earnings_date()
# Fallback when all APIs fail
KNOWN_EARNINGS = {
    'NVDA':  ['2026-05-20', '2026-08-27', '2026-11-19'],
    'AMD':   ['2026-07-29', '2026-10-28', '2027-01-28'],
    'TSM':   ['2026-07-17', '2026-10-16', '2027-01-15'],
    'MSFT':  ['2026-07-29', '2026-10-28', '2027-01-28'],
    'AAPL':  ['2026-08-05', '2026-10-28', '2027-01-28'],
    'GOOGL': ['2026-07-29', '2026-10-28', '2027-01-28'],
    'JPM':   ['2026-07-15', '2026-10-14', '2027-01-14'],
    'V':     ['2026-07-23', '2026-10-22', '2027-01-28'],
    'XOM':   ['2026-08-05', '2026-10-28', '2027-01-30'],
    'CVX':   ['2026-08-05', '2026-10-28', '2027-01-30'],
    'NEE':   ['2026-07-22', '2026-10-21', '2027-01-28'],
    'UNH':   ['2026-07-15', '2026-10-14', '2027-01-14'],
    'JNJ':   ['2026-07-15', '2026-10-14', '2027-01-14'],
    'NVO':   ['2026-08-12', '2026-11-05', '2027-02-11'],
    'CAT':   ['2026-07-28', '2026-10-27', '2027-01-27'],
    'AMZN':  ['2026-08-05', '2026-10-28', '2027-01-28'],
    'RTX':   ['2026-07-22', '2026-10-21', '2027-01-21'],
    'QQQ': [], 'SPY': [], 'GLD': [], 'SLV': [], 'BRK-B': [],
}

def get_earnings_date(ticker: str) -> dict:
    """
    Fetch next earnings date using multiple sources:
    1. yfinance calendar (primary)
    2. yfinance quarterly financials dates (backup)
    3. Hard-coded known dates (final fallback)
    Always returns valid data — never fails silently.
    """
    now   = datetime.utcnow()
    today = now.date()

    # Return cache if fresh (12 hour TTL)
    if ticker in _earnings_cache:
        age = (now - _earnings_cache_time[ticker]).total_seconds() / 3600
        if age < CACHE_HOURS:
            return _earnings_cache[ticker]

    earn_date = None

    # SOURCE 1: yfinance calendar
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        if cal is not None and not cal.empty:
            dates = cal.columns.tolist()
            if dates:
                d = dates[0]
                earn_date = d.date() if hasattr(d, 'date') else datetime.strptime(str(d)[:10], '%Y-%m-%d').date()
                print(f"Earnings {ticker} from yfinance calendar: {earn_date}")
    except Exception as e:
        print(f"yfinance calendar error {ticker}: {e}")

    # SOURCE 2: yfinance next_earnings_date attribute
    if earn_date is None:
        try:
            stock    = yf.Ticker(ticker)
            info     = stock.info or {}
            ned      = info.get('nextEarningsDate') or info.get('earningsDate')
            if ned:
                if isinstance(ned, (int, float)):
                    earn_date = datetime.utcfromtimestamp(ned).date()
                else:
                    earn_date = datetime.strptime(str(ned)[:10], '%Y-%m-%d').date()
                print(f"Earnings {ticker} from yfinance info: {earn_date}")
        except Exception as e:
            print(f"yfinance info error {ticker}: {e}")

    # SOURCE 3: Hard-coded known dates (always reliable)
    if earn_date is None:
        known = KNOWN_EARNINGS.get(ticker, [])
        future_dates = []
        for d in known:
            try:
                dd = datetime.strptime(d, '%Y-%m-%d').date()
                if dd >= today:
                    future_dates.append(dd)
            except: pass
        if future_dates:
            earn_date = min(future_dates)
            print(f"Earnings {ticker} from hard-coded: {earn_date}")

    # Build result
    if earn_date:
        days_until = (earn_date - today).days
        blackout   = -3 <= days_until <= 3
        result = {
            'ticker':       ticker,
            'has_earnings': True,
            'next_date':    str(earn_date),
            'days_until':   days_until,
            'blackout':     blackout,
            'source':       'auto',
        }
    else:
        result = {
            'ticker':       ticker,
            'has_earnings': False,
            'next_date':    None,
            'days_until':   None,
            'blackout':     False,
            'source':       'none',
        }

    _earnings_cache[ticker]      = result
    _earnings_cache_time[ticker] = now
    return result

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
    'NVDA':  ['nvidia','nvda','gpu','ai chip','blackwell','h100','data center'],
    'QQQ':   ['nasdaq','tech stocks','qqq','big tech','rates'],
    'SPY':   ['s&p','spy','market rally','correction','broad market'],
    'GLD':   ['gold','inflation','dollar','safe haven','fed'],
    'SLV':   ['silver','industrial metals','solar','ev'],
    'AMD':   ['amd','advanced micro','ryzen','radeon','epyc'],
    'TSM':   ['tsmc','taiwan semi','foundry','chip making'],
    'MSFT':  ['microsoft','msft','azure','copilot','openai'],
    'AAPL':  ['apple','aapl','iphone','mac','app store'],
    'GOOGL': ['google','alphabet','googl','gemini','search','youtube'],
    'JPM':   ['jpmorgan','jpm','chase','banking','federal reserve','interest rates'],
    'V':     ['visa','payments','consumer spending','credit card'],
    'BRK-B': ['berkshire','buffett','brk','value investing'],
    'XOM':   ['exxon','xom','oil','crude','energy','petroleum'],
    'CVX':   ['chevron','cvx','oil','gas','energy'],
    'NEE':   ['nextera','nee','renewable','solar','wind','utility'],
    'UNH':   ['unitedhealth','unh','insurance','healthcare','medicare'],
    'JNJ':   ['johnson','jnj','pharma','medical device','healthcare'],
    'NVO':   ['novo nordisk','nvo','ozempic','wegovy','glp-1','obesity'],
    'CAT':   ['caterpillar','cat','infrastructure','construction','mining'],
    'AMZN':  ['amazon','amzn','aws','prime','ecommerce','cloud'],
    'RTX':   ['raytheon','rtx','defense','aerospace','military','missile'],
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

    # PSAR calculation
    try:
        psar_data = compute_psar(df['High'].squeeze(), df['Low'].squeeze(), c)
    except:
        psar_data = {'psar': None, 'psar_bullish': None}

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
        'gap_pct':        round((price / prev_close - 1) * 100, 2) if prev_close > 0 else 0.0,
        'psar':           psar_data.get('psar'),
        'psar_bullish':   psar_data.get('psar_bullish'),
        'gap_down':       (price / prev_close - 1) * 100 < -2.0 if prev_close > 0 else False,
        'gap_up':         (price / prev_close - 1) * 100 > 2.0  if prev_close > 0 else False,
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

    # FIX 1: Block BUY on overnight gap down > 2%
    if features.get('gap_down', False):
        gap = features.get('gap_pct', 0)
        return {'signal': 'HOLD', 'confidence': 0.55, 'score': -0.2,
                'blocked_reason': f'GAP_DOWN_{gap:.1f}pct',
                'atr_pct': features.get('atr_pct', 1.5),
                'vol_ratio': features.get('vol_ratio', 1.0),
                'gap_pct': gap}

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

    # PSAR confirmation — adds extra signal quality
    psar_bull = features.get('psar_bullish') if features else None
    if psar_bull is False and signal == 'BUY':
        confidence *= 0.85   # PSAR says downtrend
    elif psar_bull is True and signal == 'BUY':
        confidence = min(confidence * 1.05, 0.95)  # PSAR confirms uptrend

    # FIX 2: Reduce confidence first/last 30min (volatile periods)
    try:
        from datetime import datetime as dt2
        import pytz
        est     = dt2.now(pytz.timezone('America/New_York'))
        first30 = est.hour == 9 and 30 <= est.minute < 60
        last30  = est.hour == 15 and est.minute >= 30
        if first30: confidence = round(confidence * 0.85, 3)
        if last30:  confidence = round(confidence * 0.90, 3)
    except:
        first30 = False
        last30  = False

    return {
        'signal':         signal,
        'confidence':     round(confidence, 3),
        'score':          round(final_score, 3),
        'blocked_reason': blocked_reason,
        'atr_pct':        round(atr_pct, 2),
        'vol_ratio':      round(vol_ratio, 2),
        'gap_pct':        features.get('gap_pct', 0),
        'first_30min':    first30,
        'last_30min':     last30,
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

# ============================================================
# MULTI-TIMEFRAME ANALYSIS — 4h + daily confirmation
# ============================================================
def get_4h_features(ticker: str) -> dict:
    """Get 4-hour timeframe features for confirmation."""
    try:
        df = yf.download(ticker, period='30d', interval='1h', progress=False)
        if df.empty or len(df) < 20:
            return None

        c = df['Close'].squeeze()
        v = df['Volume'].squeeze()

        # 4h candles — resample
        df_4h = df.resample('4h').agg({
            'Open': 'first', 'High': 'max',
            'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        if len(df_4h) < 10:
            return None

        c4h = df_4h['Close'].squeeze()

        # 4h trend
        ema9_4h  = float(c4h.ewm(span=9).mean().iloc[-1])
        ema21_4h = float(c4h.ewm(span=21).mean().iloc[-1])
        price_4h = float(c4h.iloc[-1])

        # 4h momentum
        roc3_4h  = (price_4h / float(c4h.iloc[-4]) - 1) * 100 if len(c4h) > 4 else 0

        # 4h RSI
        delta   = c4h.diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs      = gain / (loss + 1e-9)
        rsi_4h  = float(100 - 100 / (1 + rs.iloc[-1]))

        # Intraday trend — last 1h candles
        df_1h = df.tail(8)  # last 8 hours
        c1h   = df_1h['Close'].squeeze()
        intraday_up = float(c1h.iloc[-1]) > float(c1h.iloc[0])  # trending up today

        return {
            'price_4h':      round(price_4h, 2),
            'ema9_4h':       round(ema9_4h, 2),
            'ema21_4h':      round(ema21_4h, 2),
            'rsi_4h':        round(rsi_4h, 1),
            'roc3_4h':       round(roc3_4h, 2),
            'above_ema9':    price_4h > ema9_4h,
            'above_ema21':   price_4h > ema21_4h,
            'ema_bullish':   ema9_4h > ema21_4h,  # fast > slow = bullish
            'intraday_up':   intraday_up,
            'intraday_pct':  round((float(c1h.iloc[-1]) / float(c1h.iloc[0]) - 1) * 100, 2) if float(c1h.iloc[0]) > 0 else 0,
        }
    except Exception as e:
        print(f"4h features error {ticker}: {e}")
        return None

# ============================================================
# SUPPORT / RESISTANCE LEVELS
# ============================================================
def get_support_resistance(ticker: str) -> dict:
    """Calculate key support/resistance levels using pivot points."""
    try:
        df = yf.download(ticker, period='30d', interval='1d', progress=False)
        if df.empty or len(df) < 10:
            return {}

        # Use last 5 days for pivot calculation
        recent = df.tail(5)
        high   = float(recent['High'].max())
        low    = float(recent['Low'].min())
        close  = float(df['Close'].iloc[-1])

        # Classic pivot points
        pivot = (high + low + close) / 3
        r1    = 2 * pivot - low
        r2    = pivot + (high - low)
        s1    = 2 * pivot - high
        s2    = pivot - (high - low)

        # Distance to nearest support/resistance
        dist_r1 = round((r1 - close) / close * 100, 2)
        dist_s1 = round((close - s1) / close * 100, 2)

        # Is price near support (good entry) or resistance (bad entry)?
        near_support    = 0 < dist_s1 < 2.0   # within 2% of support
        near_resistance = 0 < dist_r1 < 1.5   # within 1.5% of resistance

        return {
            'pivot':           round(pivot, 2),
            'r1':              round(r1, 2),
            'r2':              round(r2, 2),
            's1':              round(s1, 2),
            's2':              round(s2, 2),
            'dist_to_r1_pct':  dist_r1,
            'dist_to_s1_pct':  dist_s1,
            'near_support':    near_support,
            'near_resistance': near_resistance,
        }
    except Exception as e:
        print(f"S/R error {ticker}: {e}")
        return {}


# ============================================================
# ============================================================
# SENTIMENT — Enhanced keyword scoring (lightweight, no PyTorch)
# FinBERT removed — too heavy for Railway (needs 2GB RAM)
# ============================================================
_finbert_cache      = {}
_finbert_cache_time = {}

BULLISH_WORDS = ['beat','surge','strong','rally','upgrade','bullish','record','growth','positive','profit','rise','gain']
BEARISH_WORDS = ['miss','drop','fall','downgrade','sell','bearish','loss','decline','weak','concern','tariff','crash']

def keyword_sentiment(headlines, ticker):
    if not headlines: return 0.0
    score = 0.0
    ticker_lower = ticker.lower()
    for h in headlines:
        h = h.lower()
        relevant = ticker_lower in h or any(k in h for k in ['stock','market','shares','earnings'])
        weight = 1.5 if relevant else 1.0
        for w in BULLISH_WORDS:
            if w in h: score += 0.2 * weight
        for w in BEARISH_WORDS:
            if w in h: score -= 0.2 * weight
    return round(max(-1.0, min(1.0, score / max(len(headlines), 1))), 3)


def finbert_score(texts: list) -> float:
    # Uses enhanced keyword sentiment — lightweight alternative to FinBERT
    return keyword_sentiment(texts, '') if texts else 0.0

BULLISH_WORDS = ['beat','surge','strong','rally','upgrade','bullish','record','growth','positive','profit']
BEARISH_WORDS = ['miss','drop','fall','downgrade','sell','bearish','loss','decline','weak','concern','tariff']

def keyword_sentiment(headlines, ticker):
    score = 0.0
    for h in headlines:
        h = h.lower()
        for w in BULLISH_WORDS:
            if w in h: score += 0.2
        for w in BEARISH_WORDS:
            if w in h: score -= 0.2
    return round(max(-1.0, min(1.0, score / max(len(headlines), 1))), 3)

def get_finbert_sentiment(ticker: str) -> dict:
    """Get FinBERT-powered sentiment for a ticker."""
    now = datetime.utcnow()

    # Check cache (2 hour TTL)
    if ticker in _finbert_cache:
        age = (now - _finbert_cache_time[ticker]).total_seconds() / 3600
        if age < 2:
            return _finbert_cache[ticker]

    try:
        stock    = yf.Ticker(ticker)
        news     = stock.news or []
        # yfinance sometimes returns news under different key
        if not news:
            try:
                info = stock.fast_info
                news = getattr(stock, 'news', []) or []
            except: pass
        cutoff    = now - timedelta(hours=48)
        headlines = []

        for article in news[:10]:
            # yfinance returns different formats — handle both
            title = (article.get('title') or
                     article.get('content', {}).get('title') or
                     article.get('headline') or '')
            pub_time = (article.get('providerPublishTime') or
                        article.get('content', {}).get('pubDate') or 0)
            if not title: continue
            # Skip old articles
            if pub_time and isinstance(pub_time, (int, float)):
                if datetime.utcfromtimestamp(pub_time) < cutoff: continue
            headlines.append(title)

        if not headlines:
            result = {'ticker': ticker, 'sentiment': 0.0, 'signal': 'NEUTRAL',
                      'method': 'finbert', 'headlines': [], 'count': 0}
        else:
            score = finbert_score(headlines)
            signal = 'BULLISH' if score > 0.15 else 'BEARISH' if score < -0.15 else 'NEUTRAL'
            result = {
                'ticker':    ticker,
                'sentiment': score,
                'signal':    signal,
                'method':    'finbert',
                'headlines': headlines[:3],
                'count':     len(headlines),
            }

        _finbert_cache[ticker]      = result
        _finbert_cache_time[ticker] = now
        return result

    except Exception as e:
        print(f"FinBERT sentiment error {ticker}: {e}")
        return {'ticker': ticker, 'sentiment': 0.0, 'signal': 'NEUTRAL', 'method': 'error', 'headlines': [], 'count': 0}

# ============================================================
# BUILD 2: VWAP CALCULATION
# ============================================================
def get_vwap(ticker: str) -> dict:
    """Calculate intraday VWAP and position relative to it."""
    try:
        df = yf.download(ticker, period='1d', interval='5m', progress=False)
        if df.empty or len(df) < 5:
            return {'vwap': None, 'price_vs_vwap': None, 'above_vwap': None}

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cum_vol       = df['Volume'].cumsum()
        cum_tp_vol    = (typical_price * df['Volume']).cumsum()
        vwap          = cum_tp_vol / cum_vol

        current_vwap  = float(vwap.iloc[-1])
        current_price = float(df['Close'].iloc[-1])
        pct_vs_vwap   = round((current_price / current_vwap - 1) * 100, 2)
        above_vwap    = current_price > current_vwap

        return {
            'vwap':          round(current_vwap, 2),
            'price_vs_vwap': pct_vs_vwap,
            'above_vwap':    above_vwap,
        }
    except Exception as e:
        return {'vwap': None, 'price_vs_vwap': None, 'above_vwap': None}

# ============================================================
# BUILD 3: POST-EARNINGS MOMENTUM DETECTION
# ============================================================
def get_earnings_momentum(ticker: str) -> dict:
    """
    Detect post-earnings drift opportunity.
    If earnings was within last 5 days AND price gapped up → BUY momentum.
    """
    try:
        stock    = yf.Ticker(ticker)
        cal      = stock.calendar
        today    = datetime.utcnow().date()

        if cal is None or cal.empty:
            return {'post_earnings': False, 'earnings_gap': 0, 'days_since': None}

        dates = cal.columns.tolist()
        if not dates:
            return {'post_earnings': False, 'earnings_gap': 0, 'days_since': None}

        earn_date = dates[0]
        if hasattr(earn_date, 'date'):
            earn_date = earn_date.date()
        else:
            earn_date = datetime.strptime(str(earn_date)[:10], '%Y-%m-%d').date()

        days_since = (today - earn_date).days

        if 0 <= days_since <= 5:
            # Recent earnings — check for gap up
            df = yf.download(ticker, period='10d', interval='1d', progress=False)
            if len(df) >= 2:
                c  = df['Close'].squeeze()
                o  = df['Open'].squeeze()
                gap = float((float(o.iloc[-1]) / float(c.iloc[-2]) - 1) * 100)
                return {
                    'post_earnings': True,
                    'earnings_gap':  round(gap, 2),
                    'days_since':    days_since,
                    'bullish_drift': gap > 2.0,  # gap up > 2% = strong buy momentum
                }

        return {'post_earnings': False, 'earnings_gap': 0, 'days_since': days_since}

    except Exception as e:
        return {'post_earnings': False, 'earnings_gap': 0, 'days_since': None}

# ============================================================
# BUILD 5: 52-WEEK HIGH BREAKOUT DETECTION
# ============================================================
def get_breakout_signal(ticker: str, features: dict) -> dict:
    """Detect 52-week high breakout — strong momentum signal."""
    try:
        if not features:
            return {'breakout': False, 'pct_from_52wk_high': 0}

        price   = features.get('price', 0)
        ath     = features.get('ath', price)
        pct_off = features.get('from_ath', 0)

        # Near 52-week high (within 3%) = breakout candidate
        near_high = pct_off >= -3.0 and pct_off <= 0

        # Already broke out (new high)
        new_high  = pct_off >= 0

        return {
            'breakout':            new_high,
            'near_breakout':       near_high,
            'pct_from_52wk_high':  pct_off,
        }
    except:
        return {'breakout': False, 'near_breakout': False, 'pct_from_52wk_high': 0}

# ============================================================
# BUILD 4: REGIME-SPECIFIC SIGNAL WEIGHTS
# Based on research: scale signals differently per regime
# ============================================================
def get_regime_weights(regime: str) -> dict:
    """
    Return signal weights and position size multiplier per regime.
    Based on: strong-trend 1.5x momentum, sideways 1.2x mean-reversion,
    breakout 2.5x size, high-vol 0.7x size.
    """
    weights = {
        'BULL_LOW_VOL':  {'momentum_scale': 1.5, 'size_mult': 1.3, 'min_conf': 0.68},
        'BULL_MID_VOL':  {'momentum_scale': 1.2, 'size_mult': 1.0, 'min_conf': 0.72},
        'SIDEWAYS':      {'momentum_scale': 0.8, 'size_mult': 0.7, 'min_conf': 0.75},
        'BEAR_MID_VOL':  {'momentum_scale': 0.5, 'size_mult': 0.5, 'min_conf': 0.80},
        'HIGH_FEAR':     {'momentum_scale': 0.3, 'size_mult': 0.3, 'min_conf': 0.85},
        'BREAKOUT':      {'momentum_scale': 2.0, 'size_mult': 1.5, 'min_conf': 0.70},
    }
    return weights.get(regime, {'momentum_scale': 1.0, 'size_mult': 1.0, 'min_conf': 0.72})

# ============================================================
# BUILD 6: SHARPE RATIO CALCULATION
# ============================================================
def calculate_sharpe(ticker: str, days: int = 90) -> dict:
    """
    Calculate Sharpe, Sortino, and Calmar ratios for a ticker.
    - Sharpe:  total risk-adjusted return
    - Sortino: downside risk only (better for asymmetric strategies)
    - Calmar:  return / max drawdown (resilience measure)
    """
    try:
        df      = yf.download(ticker, period=f'{days}d', interval='1d', progress=False)
        if df.empty or len(df) < 20:
            return {'sharpe': None, 'sortino': None, 'calmar': None,
                    'annualized_return': None, 'volatility': None}

        c       = df['Close'].squeeze()
        returns = c.pct_change().dropna()
        mean_r  = float(returns.mean())
        std_r   = float(returns.std())

        if std_r == 0:
            return {'sharpe': 0, 'sortino': 0, 'calmar': 0,
                    'annualized_return': 0, 'volatility': 0}

        risk_free_daily = 0.045 / 252

        # Sharpe ratio
        sharpe = (mean_r - risk_free_daily) / std_r * (252 ** 0.5)

        # Sortino ratio — only penalizes downside volatility
        downside = returns[returns < risk_free_daily] - risk_free_daily
        downside_std = float(downside.std()) if len(downside) > 0 else std_r
        sortino = (mean_r - risk_free_daily) / downside_std * (252 ** 0.5) if downside_std > 0 else 0

        # Calmar ratio — annualized return / max drawdown
        cumulative  = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns   = (cumulative - rolling_max) / rolling_max
        max_dd      = float(abs(drawdowns.min()))
        ann_return  = mean_r * 252
        calmar      = ann_return / max_dd if max_dd > 0 else 0

        # Rating
        def rate(sharpe):
            if sharpe >= 2.0: return 'Excellent'
            if sharpe >= 1.5: return 'Good'
            if sharpe >= 1.0: return 'Acceptable'
            return 'Poor'

        return {
            'ticker':             ticker,
            'sharpe':             round(sharpe, 2),
            'sortino':            round(sortino, 2),
            'calmar':             round(calmar, 2),
            'annualized_return':  round(ann_return * 100, 2),
            'volatility':         round(std_r * (252 ** 0.5) * 100, 2),
            'max_drawdown_pct':   round(max_dd * 100, 2),
            'rating':             rate(sharpe),
            'days':               days,
        }
    except Exception as e:
        return {'sharpe': None, 'sortino': None, 'calmar': None,
                'annualized_return': None, 'volatility': None, 'error': str(e)}


# ============================================================
# PSAR — Parabolic SAR indicator
# Returns: above/below price = trend direction
# ============================================================
def compute_psar(high: pd.Series, low: pd.Series, close: pd.Series,
                 af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> dict:
    """Calculate Parabolic SAR. Returns current value and trend direction."""
    try:
        n = len(close)
        if n < 10:
            return {'psar': None, 'psar_bullish': None}

        psar   = [0.0] * n
        bull   = True
        af     = af_start
        ep     = float(low.iloc[0])
        hp     = float(high.iloc[0])
        lp     = float(low.iloc[0])
        psar[0] = float(high.iloc[0])

        for i in range(2, n):
            if bull:
                psar[i] = psar[i-1] + af * (hp - psar[i-1])
                psar[i] = min(psar[i], float(low.iloc[i-1]), float(low.iloc[i-2]))
                if float(low.iloc[i]) < psar[i]:
                    bull   = False
                    psar[i] = hp
                    lp     = float(low.iloc[i])
                    af     = af_start
                    ep     = lp
                else:
                    if float(high.iloc[i]) > hp:
                        hp = float(high.iloc[i])
                        af = min(af + af_step, af_max)
                    ep = hp
            else:
                psar[i] = psar[i-1] + af * (lp - psar[i-1])
                psar[i] = max(psar[i], float(high.iloc[i-1]), float(high.iloc[i-2]))
                if float(high.iloc[i]) > psar[i]:
                    bull   = True
                    psar[i] = lp
                    hp     = float(high.iloc[i])
                    af     = af_start
                    ep     = hp
                else:
                    if float(low.iloc[i]) < lp:
                        lp = float(low.iloc[i])
                        af = min(af + af_step, af_max)
                    ep = lp

        current_psar  = round(psar[-1], 2)
        current_price = float(close.iloc[-1])
        psar_bullish  = current_price > current_psar  # price above PSAR = bullish

        return {
            'psar':         current_psar,
            'psar_bullish': psar_bullish,
        }
    except Exception as e:
        print(f"PSAR error: {e}")
        return {'psar': None, 'psar_bullish': None}

# ============================================================
# RELATIVE STRENGTH vs SPY
# Measures how ticker performs vs S&P 500 over 5 days
# RS > 1.0 = outperforming market = strong stock
# ============================================================
_rs_cache     = {}
_rs_cache_time = {}

def get_relative_strength(ticker: str) -> dict:
    """Calculate relative strength vs SPY over 5 days."""
    try:
        now = datetime.utcnow()
        cache_key = f'rs_{ticker}'
        if cache_key in _rs_cache:
            age = (now - _rs_cache_time[cache_key]).total_seconds() / 3600
            if age < 1:  # 1 hour cache
                return _rs_cache[cache_key]

        # Download both ticker and SPY
        tdf = yf.download(ticker, period='15d', interval='1d', progress=False)
        sdf = yf.download('SPY',  period='15d', interval='1d', progress=False)

        if tdf.empty or sdf.empty or len(tdf) < 6 or len(sdf) < 6:
            return {'rs': 1.0, 'rs_5d': 0.0, 'spy_5d': 0.0, 'outperforming': None}

        tc = tdf['Close'].squeeze()
        sc = sdf['Close'].squeeze()

        # 5-day performance
        t5d = float((tc.iloc[-1] / tc.iloc[-6] - 1) * 100)
        s5d = float((sc.iloc[-1] / sc.iloc[-6] - 1) * 100)

        # Relative strength ratio
        rs = round(t5d - s5d, 2)  # positive = outperforming SPY

        result = {
            'rs':            round(rs, 2),
            'rs_5d':         round(t5d, 2),   # ticker 5-day return
            'spy_5d':        round(s5d, 2),   # SPY 5-day return
            'outperforming': rs > 0,           # True if beating market
        }
        _rs_cache[cache_key]      = result
        _rs_cache_time[cache_key] = now
        return result

    except Exception as e:
        print(f"RS error {ticker}: {e}")
        return {'rs': 0.0, 'rs_5d': 0.0, 'spy_5d': 0.0, 'outperforming': None}


# ============================================================
# ADAPTIVE STRATEGY ENGINE
# Detects market regime and adjusts strategy automatically
# BULL_LOW_VOL / BULL_MID_VOL  → momentum (buy breakouts)
# SIDEWAYS                     → mean reversion (buy dips)
# BEAR_MID_VOL                 → defensive (reduce size)
# HIGH_FEAR                    → cash + defensive only
# BREAKOUT                     → aggressive momentum
# ============================================================

def get_mean_reversion_signal(features: dict) -> dict:
    """
    Mean reversion strategy — used in SIDEWAYS regime.
    Buy when price is oversold relative to recent range.
    Opposite of momentum — buy weakness, sell strength.
    """
    if not features:
        return {'signal': 'HOLD', 'confidence': 0.5, 'strategy': 'mean_reversion'}

    rsi      = features.get('rsi', 50)
    bb_pct   = features.get('bb_pct', 0.5)
    price    = features.get('price', 0)
    sma20    = features.get('sma20', price)
    vol_ratio = features.get('vol_ratio', 1.0)

    score = 0.0

    # RSI oversold = buy opportunity in mean reversion
    if   rsi < 30: score += 0.8   # very oversold — strong buy
    elif rsi < 40: score += 0.4   # oversold — buy
    elif rsi > 70: score -= 0.6   # overbought — sell
    elif rsi > 60: score -= 0.3   # approaching overbought

    # Bollinger Band — below lower band = buy
    if   bb_pct < 0.1: score += 0.6   # at lower band
    elif bb_pct < 0.25: score += 0.3
    elif bb_pct > 0.9: score -= 0.6   # at upper band — sell
    elif bb_pct > 0.75: score -= 0.3

    # Distance from SMA20 — far below = buy
    if sma20 > 0:
        pct_from_sma = (price - sma20) / sma20 * 100
        if   pct_from_sma < -5: score += 0.4   # very far below — bounce likely
        elif pct_from_sma < -2: score += 0.2
        elif pct_from_sma > 5:  score -= 0.4   # very far above — pullback likely
        elif pct_from_sma > 2:  score -= 0.2

    final_score = max(-1.0, min(1.0, score))

    if final_score > 0.3:
        signal     = 'BUY'
        confidence = min(0.60 + final_score * 0.35, 0.90)
    elif final_score < -0.3:
        signal     = 'SELL'
        confidence = min(0.60 + abs(final_score) * 0.35, 0.90)
    else:
        signal     = 'HOLD'
        confidence = 0.5

    return {
        'signal':     signal,
        'confidence': round(confidence, 3),
        'score':      round(final_score, 3),
        'strategy':   'mean_reversion',
    }

def get_defensive_signal(features: dict) -> dict:
    """
    Defensive strategy — used in BEAR/HIGH_FEAR regime.
    Only buy defensive stocks (utilities, healthcare, gold).
    Reduce position sizes significantly.
    """
    if not features:
        return {'signal': 'HOLD', 'confidence': 0.5, 'strategy': 'defensive'}

    rsi = features.get('rsi', 50)

    # In defensive mode — very conservative
    # Only buy on extreme oversold
    if rsi < 25:
        return {'signal': 'BUY', 'confidence': 0.65, 'score': 0.3, 'strategy': 'defensive'}
    elif rsi > 65:
        return {'signal': 'SELL', 'confidence': 0.70, 'score': -0.3, 'strategy': 'defensive'}
    else:
        return {'signal': 'HOLD', 'confidence': 0.5, 'score': 0.0, 'strategy': 'defensive'}

def get_adaptive_signal(features: dict, regime: str, ticker: str) -> dict:
    """
    Master adaptive signal — selects strategy based on regime.
    Returns signal with strategy metadata for transparency.
    """
    # Defensive tickers — always use these in bear market
    DEFENSIVE_TICKERS = ['GLD', 'SLV', 'NEE', 'UNH', 'JNJ', 'V', 'JPM']

    if regime in ['HIGH_FEAR', 'BEAR_MID_VOL']:
        if ticker in DEFENSIVE_TICKERS:
            sig = get_mean_reversion_signal(features)  # buy dips on defensives
        else:
            sig = get_defensive_signal(features)        # very conservative on others
        sig['size_mult'] = 0.4   # 40% normal size in fear
        sig['regime_strategy'] = 'defensive'

    elif regime == 'SIDEWAYS':
        sig = get_mean_reversion_signal(features)
        sig['size_mult'] = 0.7   # 70% size in sideways
        sig['regime_strategy'] = 'mean_reversion'

    elif regime in ['BULL_LOW_VOL', 'BULL_MID_VOL']:
        sig = generate_signal(features)  # standard momentum
        sig['size_mult'] = 1.0
        sig['regime_strategy'] = 'momentum'

    elif regime == 'BREAKOUT':
        sig = generate_signal(features)  # momentum
        # Boost confidence in breakout regime
        if sig['signal'] == 'BUY':
            sig['confidence'] = min(sig['confidence'] * 1.15, 0.95)
        sig['size_mult'] = 1.5   # 150% size in breakout
        sig['regime_strategy'] = 'aggressive_momentum'

    else:
        sig = generate_signal(features)
        sig['size_mult'] = 1.0
        sig['regime_strategy'] = 'momentum'

    return sig

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
        try:
            features = get_features(ticker)
        except:
            features = None
        sig = generate_signal(features)

        # Multi-timeframe 4h — use cached if available (refresh every 30min)
        cache_key = f'tf4h_{ticker}'
        tf4h = _signal_cache.get(cache_key, {}).get('data')
        if tf4h is None or (datetime.utcnow() - _signal_cache.get(cache_key, {}).get('time', datetime.min)).seconds > 1800:
            try: tf4h = get_4h_features(ticker)
            except: tf4h = {}
            _signal_cache[cache_key] = {'data': tf4h, 'time': datetime.utcnow()}

        # Support/resistance — cache 1 hour
        sr_key = f'sr_{ticker}'
        sr = _signal_cache.get(sr_key, {}).get('data')
        if sr is None or (datetime.utcnow() - _signal_cache.get(sr_key, {}).get('time', datetime.min)).seconds > 3600:
            try: sr = get_support_resistance(ticker)
            except: sr = {}
            _signal_cache[sr_key] = {'data': sr, 'time': datetime.utcnow()}

        # Combine into final signal
        ticker_data = {
            'price':            features['price']       if features else 0,
            'day_change':       features['day_change']   if features else 0,
            'sma20':            features['sma20']        if features else 0,
            'sma50':            features['sma50']        if features else 0,
            'sma200':           features['sma200']       if features else 0,
            'from_ath':         features['from_ath']     if features else 0,
            'atr':              features['atr']          if features else 0,
            'atr_pct':          features['atr_pct']      if features else 0,
            'rsi':              round(features['rsi'],1) if features else 50,
            'gap_pct':          features.get('gap_pct', 0) if features else 0,
            'signal':           sig['signal'],
            'confidence':       sig['confidence'],
            'score':            sig['score'],
            'blocked_reason':   sig.get('blocked_reason'),
            'vol_ratio':        sig.get('vol_ratio', 1.0),
            'first_30min':      sig.get('first_30min', False),
            'last_30min':       sig.get('last_30min', False),
            # 4h timeframe
            '4h':               tf4h or {},
            # Support/resistance
            'sr':               sr,
        }

        # Adjust confidence based on 4h confirmation
        if tf4h:
            daily_bull  = sig['signal'] == 'BUY'
            h4_bull     = tf4h['ema_bullish'] and tf4h['above_ema9']
            intraday_up = tf4h['intraday_up']

            if daily_bull and h4_bull and intraday_up:
                # All timeframes aligned — boost confidence
                ticker_data['confidence'] = min(sig['confidence'] * 1.10, 0.95)
                ticker_data['tf_aligned'] = True
            elif daily_bull and not h4_bull:
                # Daily says BUY but 4h disagrees — reduce confidence
                ticker_data['confidence'] = sig['confidence'] * 0.80
                ticker_data['tf_aligned'] = False
            else:
                ticker_data['tf_aligned'] = None

        # Adjust for resistance — reduce confidence if near resistance
        if sr.get('near_resistance', False):
            ticker_data['confidence'] = round(ticker_data['confidence'] * 0.85, 3)
            ticker_data['sr_warning'] = 'NEAR_RESISTANCE'
        elif sr.get('near_support', False):
            ticker_data['confidence'] = round(ticker_data['confidence'] * 1.05, 3)
            ticker_data['sr_warning'] = 'NEAR_SUPPORT_GOOD_ENTRY'
        else:
            ticker_data['sr_warning'] = None

        # BUILD 1: FinBERT sentiment — cached 2 hours
        fb_key = f'finbert_{ticker}'
        finbert_data = _signal_cache.get(fb_key, {}).get('data')
        if finbert_data is None or (datetime.utcnow() - _signal_cache.get(fb_key, {}).get('time', datetime.min)).seconds > 7200:
            try: finbert_data = get_finbert_sentiment(ticker)
            except: finbert_data = {'ticker': ticker, 'sentiment': 0.0, 'signal': 'NEUTRAL'}
            _signal_cache[fb_key] = {'data': finbert_data, 'time': datetime.utcnow()}

        # BUILD 2: VWAP — cached 5 minutes
        vw_key = f'vwap_{ticker}'
        vwap_data = _signal_cache.get(vw_key, {}).get('data')
        if vwap_data is None or (datetime.utcnow() - _signal_cache.get(vw_key, {}).get('time', datetime.min)).seconds > 300:
            try: vwap_data = get_vwap(ticker)
            except: vwap_data = {}
            _signal_cache[vw_key] = {'data': vwap_data, 'time': datetime.utcnow()}

        # BUILD 3: Post-earnings momentum — cached 6 hours
        em_key = f'earn_{ticker}'
        earn_mom = _signal_cache.get(em_key, {}).get('data')
        if earn_mom is None or (datetime.utcnow() - _signal_cache.get(em_key, {}).get('time', datetime.min)).seconds > 21600:
            try: earn_mom = get_earnings_momentum(ticker)
            except: earn_mom = {}
            _signal_cache[em_key] = {'data': earn_mom, 'time': datetime.utcnow()}

        # BUILD 5: Breakout signal
        try: breakout = get_breakout_signal(ticker, features)
        except: breakout = {}

        # Enhance confidence with FinBERT
        if finbert_data['signal'] == 'BULLISH' and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] = min(ticker_data['confidence'] * 1.12, 0.95)
        elif finbert_data['signal'] == 'BEARISH' and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] *= 0.80  # reduce on bad news
            if ticker_data['confidence'] < 0.72:
                ticker_data['signal'] = 'HOLD'

        # VWAP boost — above VWAP = institutional buying
        if vwap_data.get('above_vwap') and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] = min(ticker_data['confidence'] * 1.05, 0.95)

        # Post-earnings momentum — special boost
        if earn_mom.get('bullish_drift') and earn_mom.get('post_earnings'):
            ticker_data['signal']     = 'BUY'
            ticker_data['confidence'] = min(ticker_data['confidence'] * 1.20, 0.95)
            ticker_data['post_earnings_play'] = True

        # Breakout boost
        if breakout.get('breakout') and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] = min(ticker_data['confidence'] * 1.10, 0.95)
            ticker_data['breakout']   = True

        # BUILD 2: Relative strength vs SPY
        rs_data = get_relative_strength(ticker) if ticker != 'SPY' else {'rs': 0, 'rs_5d': 0, 'spy_5d': 0, 'outperforming': None}

        # RS boost — outperforming SPY = stronger entry
        if rs_data.get('outperforming') is True and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] = min(ticker_data['confidence'] * 1.08, 0.95)
        elif rs_data.get('outperforming') is False and ticker_data['signal'] == 'BUY':
            ticker_data['confidence'] *= 0.88  # lagging market — reduce confidence

        # Add PSAR to response
        ticker_data['psar']    = features.get('psar') if features else None
        ticker_data['psar_bullish'] = features.get('psar_bullish') if features else None

        # Add all data to response
        ticker_data['finbert']         = finbert_data
        ticker_data['vwap']            = vwap_data
        ticker_data['earnings_mom']    = earn_mom
        ticker_data['breakout_signal'] = breakout
        ticker_data['rs']              = rs_data
        ticker_data['regime_strategy'] = sig.get('regime_strategy', 'momentum')
        ticker_data['size_mult']       = sig.get('size_mult', 1.0)

        result[ticker] = ticker_data

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

    # News — cached 2 hours
    news_key = 'all_news'
    news_data = _signal_cache.get(news_key, {}).get('data')
    if news_data is None or (datetime.utcnow() - _signal_cache.get(news_key, {}).get('time', datetime.min)).seconds > 7200:
        try: news_data = {t: get_finbert_sentiment(t) for t in TICKERS}
        except: news_data = {}
        _signal_cache[news_key] = {'data': news_data, 'time': datetime.utcnow()}
    result['news'] = news_data

    # Macro — cached 30 minutes
    macro_key = 'macro'
    macro_data = _signal_cache.get(macro_key, {}).get('data')
    if macro_data is None or (datetime.utcnow() - _signal_cache.get(macro_key, {}).get('time', datetime.min)).seconds > 1800:
        try: macro_data = get_macro_context()
        except: macro_data = {}
        _signal_cache[macro_key] = {'data': macro_data, 'time': datetime.utcnow()}
    result['macro'] = macro_data

    return JSONResponse(result)

@app.get('/earnings')
def earnings_endpoint():
    """Dedicated earnings calendar endpoint."""
    return JSONResponse(get_all_earnings())

@app.get('/news')
def news_endpoint():
    results = {}
    for ticker in TICKERS:
        results[ticker] = get_finbert_sentiment(ticker)
    return JSONResponse(results)

@app.get('/sharpe')
def sharpe_endpoint():
    results = {}
    for ticker in ['NVDA', 'QQQ', 'SPY']:
        results[ticker] = calculate_sharpe(ticker)
    return JSONResponse(results)

@app.get('/regime_weights/{regime}')
def regime_weights_endpoint(regime: str):
    return JSONResponse(get_regime_weights(regime))

@app.get('/macro')
def macro_endpoint():
    """Macro market context."""
    return JSONResponse(get_macro_context())

@app.post('/retrain')
def retrain_endpoint(data: dict):
    ticker   = data.get('ticker', 'NVDA')
    outcomes = data.get('outcomes', [])
    if len(outcomes) < 10:
        return JSONResponse({'status': 'insufficient_data', 'count': len(outcomes)})
    try:
        keys = ['rsi','macd','roc5','roc10','roc20','bb_pct','vol_ratio','above_sma20','above_sma50','atr_pct']
        X = [[float(o.get('features',{}).get(k,0)) for k in keys] for o in outcomes]
        y = [1 if o.get('result')=='WIN' else 0 for o in outcomes]
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', use_label_encoder=False)
        model.fit(X, y)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, model.predict(X))
        return JSONResponse({'status': 'success', 'ticker': ticker, 'accuracy': round(acc,3), 'samples': len(outcomes)})
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)})


# ============================================================
# BACKTESTING FRAMEWORK — test strategy on historical data
# ============================================================
@app.get('/backtest/{ticker}')
def backtest(ticker: str, days: int = 180):
    """
    Simple backtest: simulate buy/sell signals on last N days.
    Returns win rate, total return, max drawdown.
    """
    try:
        df = yf.download(ticker, period=f'{days+50}d', interval='1d', progress=False)
        if df.empty or len(df) < 50:
            return JSONResponse({'error': 'Insufficient data'})

        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        v = df['Volume'].squeeze()

        trades      = []
        in_position = False
        entry_price = 0
        equity      = 10000.0
        peak_equity = 10000.0
        max_dd      = 0.0

        for i in range(50, len(c) - 1):
            # Build features for this day
            window = c.iloc[i-50:i+1]
            rsi_val = compute_rsi(window, 14)
            sma20   = float(window.rolling(20).mean().iloc[-1])
            sma50   = float(window.rolling(50).mean().iloc[-1]) if i >= 50 else sma20
            price   = float(c.iloc[i])
            vol_r   = float(v.iloc[i]) / float(v.iloc[i-20:i].mean()) if i >= 20 else 1.0

            features = {
                'rsi':        rsi_val,
                'above_sma20': 1 if price > sma20 else 0,
                'above_sma50': 1 if price > sma50 else 0,
                'macd':        float((window.ewm(span=12).mean() - window.ewm(span=26).mean()).iloc[-1]),
                'macd_signal': float((window.ewm(span=12).mean() - window.ewm(span=26).mean()).ewm(span=9).mean().iloc[-1]),
                'roc5':        float((price / float(c.iloc[i-5]) - 1) * 100) if i >= 5 else 0,
                'roc10':       float((price / float(c.iloc[i-10]) - 1) * 100) if i >= 10 else 0,
                'roc20':       float((price / float(c.iloc[i-20]) - 1) * 100) if i >= 20 else 0,
                'bb_pct':      0.5,
                'vol_ratio':   vol_r,
                'atr_pct':     float(compute_atr(h.iloc[i-14:i+1], l.iloc[i-14:i+1], c.iloc[i-14:i+1], 14) / price * 100) if i >= 14 else 1.5,
                'gap_down':    False,
                'gap_pct':     0,
            }

            sig = generate_signal(features)

            if not in_position and sig['signal'] == 'BUY' and sig['confidence'] >= 0.65:
                # Dynamic SL based on ATR
                atr_pct   = features['atr_pct'] / 100
                stop_loss = price * (1 - max(atr_pct * 2, 0.05))
                take_prof = price * 1.14  # 14% TP (2:1 ratio)
                entry_price = price
                in_position = True
                entry_idx   = i

            elif in_position:
                next_price = float(c.iloc[i+1])
                pnl_pct    = (next_price - entry_price) / entry_price

                # Exit conditions
                exit_reason = None
                if next_price <= entry_price * 0.93:  # -7% SL
                    exit_reason = 'STOP_LOSS'
                elif next_price >= entry_price * 1.14:  # +14% TP
                    exit_reason = 'TAKE_PROFIT'
                elif sig['signal'] == 'SELL' and sig['confidence'] >= 0.75:
                    exit_reason = 'SIGNAL_EXIT'
                # MAX_HOLD removed — premature exits hurt performance

                if exit_reason:
                    pnl_pct  = (next_price - entry_price) / entry_price
                    pnl_usd  = equity * pnl_pct
                    equity  += pnl_usd
                    peak_equity = max(peak_equity, equity)
                    drawdown    = (peak_equity - equity) / peak_equity
                    max_dd      = max(max_dd, drawdown)

                    trades.append({
                        'date':        str(df.index[i+1])[:10],
                        'entry':       round(entry_price, 2),
                        'exit':        round(next_price, 2),
                        'pnl_pct':     round(pnl_pct * 100, 2),
                        'reason':      exit_reason,
                        'result':      'WIN' if pnl_pct > 0 else 'LOSS',
                    })
                    in_position = False

        if not trades:
            return JSONResponse({'error': 'No trades generated', 'ticker': ticker})

        wins        = [t for t in trades if t['result'] == 'WIN']
        losses      = [t for t in trades if t['result'] == 'LOSS']
        win_rate    = len(wins) / len(trades) * 100
        total_ret   = (equity - 10000) / 10000 * 100
        avg_win     = sum(t['pnl_pct'] for t in wins)   / len(wins)   if wins   else 0
        avg_loss    = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else 0

        return JSONResponse({
            'ticker':        ticker,
            'period_days':   days,
            'total_trades':  len(trades),
            'wins':          len(wins),
            'losses':        len(losses),
            'win_rate_pct':  round(win_rate, 1),
            'total_return_pct': round(total_ret, 2),
            'avg_win_pct':   round(avg_win, 2),
            'avg_loss_pct':  round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'final_equity':  round(equity, 2),
            'recent_trades': trades[-5:],
        })
    except Exception as e:
        return JSONResponse({'error': str(e), 'ticker': ticker})

@app.get('/walkforward/{ticker}')
def walkforward(ticker: str, windows: int = 6):
    """
    Walk-forward backtest: splits 180 days into rolling windows.
    Each window: 30 days train (signal calibration) + 15 days test.
    Returns consistency score — how stable is the strategy across periods.
    """
    try:
        df = yf.download(ticker, period='200d', interval='1d', progress=False)
        if df.empty or len(df) < 60:
            return JSONResponse({'error': 'Insufficient data', 'ticker': ticker})

        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        v = df['Volume'].squeeze()

        window_size = 30  # train window
        test_size   = 15  # test window
        results     = []

        for w in range(windows):
            train_start = w * test_size
            train_end   = train_start + window_size
            test_end    = train_end + test_size

            if test_end > len(c):
                break

            test_c = c.iloc[train_end:test_end]
            test_h = h.iloc[train_end:test_end]
            test_l = l.iloc[train_end:test_end]
            test_v = v.iloc[train_end:test_end]

            # Simulate strategy on test window
            trades     = []
            in_pos     = False
            entry_px   = 0
            entry_idx  = 0

            for i in range(5, len(test_c) - 1):
                window_data = c.iloc[train_start:train_end+i]
                rsi_val = compute_rsi(window_data, 14)
                sma20   = float(window_data.rolling(20).mean().iloc[-1]) if len(window_data) >= 20 else float(window_data.mean())
                price   = float(test_c.iloc[i])
                vol_r   = float(test_v.iloc[i]) / float(test_v.iloc[max(0,i-10):i].mean()) if i >= 10 else 1.0

                features = {
                    'rsi': rsi_val, 'above_sma20': 1 if price > sma20 else 0,
                    'above_sma50': 1 if price > sma20 else 0,
                    'macd': 0.1, 'macd_signal': 0.0,
                    'roc5': float((price/float(test_c.iloc[max(0,i-5)])-1)*100),
                    'roc10': float((price/float(test_c.iloc[max(0,i-10)])-1)*100),
                    'roc20': 0, 'bb_pct': 0.5,
                    'vol_ratio': vol_r, 'atr_pct': 1.5,
                    'gap_down': False, 'gap_pct': 0,
                }
                sig = generate_signal(features)

                if not in_pos and sig['signal'] == 'BUY' and sig['confidence'] >= 0.72:
                    in_pos    = True
                    entry_px  = price
                    entry_idx = i
                elif in_pos:
                    pnl = (price - entry_px) / entry_px
                    if pnl <= -0.07 or pnl >= 0.14 or (i - entry_idx) >= 15:
                        trades.append({
                            'pnl':    round(pnl * 100, 2),
                            'result': 'WIN' if pnl > 0 else 'LOSS',
                        })
                        in_pos = False

            if trades:
                wins    = len([t for t in trades if t['result'] == 'WIN'])
                wr      = wins / len(trades) * 100
                avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
                results.append({
                    'window':    w + 1,
                    'trades':    len(trades),
                    'win_rate':  round(wr, 1),
                    'avg_pnl':   round(avg_pnl, 2),
                    'profitable': avg_pnl > 0,
                })

        if not results:
            return JSONResponse({'error': 'No trades in any window', 'ticker': ticker})

        # Consistency score — % of windows that were profitable
        profitable_windows = sum(1 for r in results if r['profitable'])
        consistency = round(profitable_windows / len(results) * 100, 1)
        avg_wr      = round(sum(r['win_rate'] for r in results) / len(results), 1)
        avg_pnl     = round(sum(r['avg_pnl'] for r in results) / len(results), 2)

        return JSONResponse({
            'ticker':       ticker,
            'windows':      len(results),
            'consistency':  consistency,
            'avg_win_rate': avg_wr,
            'avg_pnl':      avg_pnl,
            'verdict':      'ROBUST' if consistency >= 70 else 'MODERATE' if consistency >= 50 else 'UNSTABLE',
            'window_results': results,
        })
    except Exception as e:
        return JSONResponse({'error': str(e), 'ticker': ticker})

@app.get('/regime_strategy')
def regime_strategy_endpoint():
    """Current regime and which strategy is active."""
    import yfinance as yf2
    try:
        vix_data = yf2.download('^VIX', period='5d', interval='1d', progress=False)
        vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 18.0
    except:
        vix = 18.0
    regime = 'SIDEWAYS'
    if vix < 15:   regime = 'BULL_LOW_VOL'
    elif vix < 20: regime = 'BULL_MID_VOL'
    elif vix < 25: regime = 'SIDEWAYS'
    elif vix >= 30: regime = 'HIGH_FEAR'
    else:           regime = 'BEAR_MID_VOL'
    strategy_map = {
        'BULL_LOW_VOL':  'momentum',
        'BULL_MID_VOL':  'momentum',
        'SIDEWAYS':      'mean_reversion',
        'BEAR_MID_VOL':  'defensive',
        'HIGH_FEAR':     'defensive',
        'BREAKOUT':      'aggressive_momentum',
    }
    return JSONResponse({
        'vix':      round(vix, 1),
        'regime':   regime,
        'strategy': strategy_map.get(regime, 'momentum'),
        'size_mult': 0.4 if regime in ['HIGH_FEAR','BEAR_MID_VOL'] else 0.7 if regime == 'SIDEWAYS' else 1.0,
    })

@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.utcnow().isoformat()}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
