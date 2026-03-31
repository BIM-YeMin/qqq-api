from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import ta
import os

app = Flask(__name__)

# ── LOAD STOCK MODELS ────────────────────────────────────────
STOCK_TICKERS = ['QQQ', 'NVDA', 'SPY', 'GLD', 'SLV']
STOCK_MODELS  = {}

for ticker in STOCK_TICKERS:
    mp = f'{ticker}_model.pkl'
    sp = f'{ticker}_scaler.pkl'
    fp = f'{ticker}_features.json'
    if os.path.exists(mp):
        STOCK_MODELS[ticker] = {
            'model'   : joblib.load(mp),
            'scaler'  : joblib.load(sp),
            'features': json.load(open(fp))
        }
        print(f"Loaded stock model: {ticker}")

# ── LOAD OPTIONS MODELS ──────────────────────────────────────
OPTIONS_TICKERS = ['0DTE', '1DTE', '3DTE', '7DTE']
OPTIONS_MODELS  = {}

for dte in OPTIONS_TICKERS:
    mp = f'SPY_{dte}_model.pkl'
    sp = f'SPY_{dte}_scaler.pkl'
    if os.path.exists(mp):
        OPTIONS_MODELS[dte] = {
            'model' : joblib.load(mp),
            'scaler': joblib.load(sp),
        }
        print(f"Loaded options model: {dte}")

# Load options config
OPTIONS_FEATURES = []
VIX_RULES = {}
if os.path.exists('options_features.json'):
    with open('options_features.json') as f:
        OPTIONS_FEATURES = json.load(f)
if os.path.exists('vix_rules.json'):
    with open('vix_rules.json') as f:
        VIX_RULES = json.load(f)

STRIKE_MULT = {
    '0DTE': 0.8, '1DTE': 1.0,
    '3DTE': 1.5, '7DTE': 2.0
}

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def build_stock_features(ticker):
    df     = yf.Ticker(ticker).history(
             period='6mo', auto_adjust=True)
    df.index = pd.to_datetime(
               df.index).tz_localize(None)
    close  = df['Close']
    volume = df['Volume']
    df['sma_20']      = close.rolling(20).mean()
    df['sma_50']      = close.rolling(50).mean()
    df['ema_12']      = close.ewm(span=12).mean()
    df['rsi']         = ta.momentum.RSIIndicator(
                         close).rsi()
    df['macd']        = ta.trend.MACD(close).macd()
    df['macd_sig']    = ta.trend.MACD(
                         close).macd_signal()
    df['bb_high']     = ta.volatility.BollingerBands(
                         close).bollinger_hband()
    df['bb_low']      = ta.volatility.BollingerBands(
                         close).bollinger_lband()
    df['volume_ma']   = volume.rolling(20).mean()
    df['returns']     = close.pct_change()
    df['mom_5']       = close.pct_change(5)
    df['mom_10']      = close.pct_change(10)
    df['mom_20']      = close.pct_change(20)
    df['volatility']  = close.rolling(10).std()
    df['volume_spike']= volume / df['volume_ma']
    df['dist_sma20']  = (close - df['sma_20']) / \
                         df['sma_20']
    df['dist_sma50']  = (close - df['sma_50']) / \
                         df['sma_50']
    df.dropna(inplace=True)
    return df

def build_options_features(vix_now, vix_prev5):
    spy = yf.Ticker("SPY").history(
          period='1y', auto_adjust=True)
    spy.index = pd.to_datetime(
                spy.index).tz_localize(None)
    vix_tk = yf.Ticker("^VIX").history(
             period='1y')
    vix_tk.index = pd.to_datetime(
                   vix_tk.index).tz_localize(None)
    vix = vix_tk['Close'].reindex(
          spy.index).ffill()

    close = spy['Close']
    high  = spy['High']
    low   = spy['Low']

    spy['hv5']  = close.pct_change().rolling(5).std() \
                  * np.sqrt(252) * 100
    spy['hv10'] = close.pct_change().rolling(10).std() \
                  * np.sqrt(252) * 100
    spy['hv20'] = close.pct_change().rolling(20).std() \
                  * np.sqrt(252) * 100
    spy['vix']      = vix
    spy['vix_ma10'] = vix.rolling(10).mean()
    spy['vix_ma20'] = vix.rolling(20).mean()

    vix_52h = vix.rolling(252).max()
    vix_52l = vix.rolling(252).min()
    spy['iv_rank']      = (vix - vix_52l) / \
                           (vix_52h - vix_52l) * 100
    spy['iv_pct']       = vix.rolling(252).apply(
        lambda x: (x < x.iloc[-1]).sum() /
                  len(x) * 100, raw=False)
    spy['vix_hv_ratio'] = vix / (spy['hv20'] + 0.001)
    spy['vix_chg1']     = vix.pct_change(1) * 100
    spy['vix_chg5']     = vix.pct_change(5) * 100
    spy['sma20']        = close.rolling(20).mean()
    spy['sma50']        = close.rolling(50).mean()
    spy['rsi']          = ta.momentum.RSIIndicator(
                           close).rsi()
    spy['returns1']     = close.pct_change(1) * 100
    spy['returns5']     = close.pct_change(5) * 100
    spy['dist_sma20']   = (close - spy['sma20']) / \
                           spy['sma20'] * 100
    spy['atr5']         = ta.volatility.AverageTrueRange(
        high, low, close, window=5).average_true_range()
    spy['atr14']        = ta.volatility.AverageTrueRange(
        high, low, close, window=14).average_true_range()
    spy['atr_pct']      = spy['atr14'] / close * 100
    spy['day_of_week']  = pd.to_datetime(
                           spy.index).dayofweek
    spy['month']        = pd.to_datetime(
                           spy.index).month

    # KEY FIX: only drop NaN in feature columns
    # do NOT dropna on all columns
    # this keeps the latest rows intact
    feature_cols = [
        'hv5','hv10','hv20','vix','vix_ma10',
        'vix_ma20','iv_rank','iv_pct',
        'vix_hv_ratio','vix_chg1','vix_chg5',
        'sma20','sma50','rsi','returns1',
        'returns5','dist_sma20','atr_pct',
        'day_of_week','month'
    ]
    spy.dropna(subset=feature_cols, inplace=True)
    return spy


# ── STOCK SIGNAL ENDPOINT ────────────────────────────────────
@app.route('/signal')
def signal():
    ticker = request.args.get('ticker', 'QQQ').upper()
    if ticker not in STOCK_MODELS:
        return jsonify({
            'status': 'error',
            'error' : f'No model for {ticker}'
        }), 404
    try:
        df      = build_stock_features(ticker)
        close   = df['Close']
        price   = float(close.iloc[-1])
        prev    = float(close.iloc[-2])
        day_chg = round((price-prev)/prev*100, 2)
        rsi_val = round(float(df['rsi'].iloc[-1]), 1)
        sma20   = round(float(
                    df['sma_20'].iloc[-1]), 2)
        sma50   = round(float(
                    df['sma_50'].iloc[-1]), 2)
        m       = STOCK_MODELS[ticker]
        latest  = df[m['features']].iloc[-1:]
        scaled  = m['scaler'].transform(latest)
        proba   = float(
            m['model'].predict_proba(scaled)[0][1])
        if   proba >= 0.55:
            sig  = "BUY / HOLD"
            conf = round(proba*100)
        elif proba <= 0.45:
            sig  = "SELL / STAY OUT"
            conf = round((1-proba)*100)
        else:
            sig  = "HOLD / NEUTRAL"
            conf = 50
        return jsonify({
            'ticker'     : ticker,
            'signal'     : sig,
            'confidence' : conf,
            'probability': round(proba, 4),
            'price'      : round(price, 2),
            'day_change' : day_chg,
            'rsi'        : rsi_val,
            'sma20'      : sma20,
            'sma50'      : sma50,
            'model'      : f'XGBoost AI ({ticker})',
            'status'     : 'ok'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error' : str(e)
        }), 500

# ── ALL STOCK SIGNALS ────────────────────────────────────────
@app.route('/signals/all')
def all_signals():
    results = {}
    for ticker in STOCK_MODELS:
        try:
            df    = build_stock_features(ticker)
            close = df['Close']
            price = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            m     = STOCK_MODELS[ticker]
            latest= df[m['features']].iloc[-1:]
            scaled= m['scaler'].transform(latest)
            proba = float(
                m['model'].predict_proba(
                    scaled)[0][1])
            if   proba >= 0.55:
                sig  = "BUY / HOLD"
                conf = round(proba*100)
            elif proba <= 0.45:
                sig  = "SELL / STAY OUT"
                conf = round((1-proba)*100)
            else:
                sig  = "HOLD / NEUTRAL"
                conf = 50
            results[ticker] = {
                'signal'     : sig,
                'confidence' : conf,
                'probability': round(proba, 4),
                'price'      : round(price, 2),
                'rsi'        : round(float(
                    df['rsi'].iloc[-1]), 1),
                'status'     : 'ok'
            }
        except Exception as e:
            results[ticker] = {
                'status': 'error',
                'error' : str(e)
            }
    return jsonify(results)

# ── CONDOR SIGNAL ENDPOINT ───────────────────────────────────
@app.route('/condor')
def condor():
    try:
        df  = build_options_features(None, None)
        latest   = df[OPTIONS_FEATURES].iloc[-1:]
        close_now= float(df['Close'].iloc[-1])
        atr_now  = float(df['atr_pct'].iloc[-1])
        vix_now  = float(df['vix'].iloc[-1])
        ivr_now  = float(df['iv_rank'].iloc[-1])
        hv20_now = float(df['hv20'].iloc[-1])
        date_now = df.index[-1].strftime('%Y-%m-%d')

        signals = {}
        for dte in OPTIONS_TICKERS:
            if dte not in OPTIONS_MODELS:
                continue
            cfg    = VIX_RULES.get(dte, {})
            m      = OPTIONS_MODELS[dte]
            scaled = m['scaler'].transform(latest)
            proba  = float(
                m['model'].predict_proba(
                    scaled)[0][1])
            conf     = round(proba*100, 1)
            mult     = STRIKE_MULT.get(dte, 1.0)
            rng      = mult * atr_now
            call_s   = round(
                close_now*(1+rng/100), 0)
            put_s    = round(
                close_now*(1-rng/100), 0)
            vix_max  = cfg.get('vix_max', 25)
            thresh   = cfg.get('threshold', 0.55)
            prem_pct = cfg.get('premium_pct', 0.35)
            stop_m   = cfg.get('stop_mult', 2.0)
            tp_m     = cfg.get('take_profit', 0.60)
            wing     = 5.00
            prem     = wing * prem_pct
            stop_d   = round(prem * stop_m, 2)
            tp_d     = round(prem * tp_m, 2)
            be_wr    = round(
                stop_m/(stop_m+tp_m)*100, 1)
            vix_ok   = vix_now < vix_max
            conf_ok  = proba >= thresh

            if vix_ok and conf_ok:
                sig = "SELL CONDOR"
            elif not vix_ok:
                sig = "WAIT — VIX HIGH"
            else:
                sig = "WAIT — LOW CONF"

            signals[dte] = {
                'signal'      : sig,
                'confidence'  : conf,
                'probability' : round(proba, 4),
                'vix_ok'      : vix_ok,
                'conf_ok'     : conf_ok,
                'call_strike' : call_s,
                'put_strike'  : put_s,
                'range_pct'   : round(rng, 2),
                'premium_est' : round(prem, 2),
                'take_profit' : tp_d,
                'stop_loss'   : stop_d,
                'breakeven_wr': be_wr,
                'status'      : 'ok'
            }

        return jsonify({
            'date'      : date_now,
            'spy_price' : round(close_now, 2),
            'vix'       : round(vix_now, 1),
            'iv_rank'   : round(ivr_now, 1),
            'hv20'      : round(hv20_now, 1),
            'atr_pct'   : round(atr_now, 2),
            'signals'   : signals,
            'status'    : 'ok'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error' : str(e)
        }), 500

# ── HEALTH CHECK ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status'         : 'running',
        'stock_models'   : list(STOCK_MODELS.keys()),
        'options_models' : list(OPTIONS_MODELS.keys()),
        'stock_count'    : len(STOCK_MODELS),
        'options_count'  : len(OPTIONS_MODELS)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
