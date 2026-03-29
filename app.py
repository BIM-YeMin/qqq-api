from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import ta
import os

app = Flask(__name__)

TICKERS = ['QQQ', 'NVDA', 'SPY', 'GLD', 'SLV']
MODELS = {}

for ticker in TICKERS:
    model_path = f'{ticker}_model.pkl'
    scaler_path = f'{ticker}_scaler.pkl'
    feature_path = f'{ticker}_features.json'
    if os.path.exists(model_path):
        MODELS[ticker] = {
            'model': joblib.load(model_path),
            'scaler': joblib.load(scaler_path),
            'features': json.load(open(feature_path))
        }
        print(f"Loaded model for {ticker}")
    else:
        print(f"No model found for {ticker}")

def build_features(ticker):
    df = yf.download(ticker, period='6mo', auto_adjust=True, progress=False)
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['ema_12'] = close.ewm(span=12).mean()
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['macd'] = ta.trend.MACD(close).macd()
    df['macd_sig'] = ta.trend.MACD(close).macd_signal()
    df['bb_high'] = ta.volatility.BollingerBands(close).bollinger_hband()
    df['bb_low'] = ta.volatility.BollingerBands(close).bollinger_lband()
    df['volume_ma'] = volume.rolling(20).mean()
    df['returns'] = close.pct_change()
    df['mom_5'] = close.pct_change(5)
    df['mom_10'] = close.pct_change(10)
    df['mom_20'] = close.pct_change(20)
    df['volatility'] = close.rolling(10).std()
    df['volume_spike'] = volume / df['volume_ma']
    df['dist_sma20'] = (close - df['sma_20']) / df['sma_20']
    df['dist_sma50'] = (close - df['sma_50']) / df['sma_50']
    df.dropna(inplace=True)
    return df

@app.route('/signal')
def signal():
    ticker = request.args.get('ticker', 'QQQ').upper()
    if ticker not in MODELS:
        return jsonify({'status': 'error', 'error': f'No AI model for {ticker}'}), 404
    try:
        df = build_features(ticker)
        close = df['Close'].squeeze()
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        day_chg = round((price - prev) / prev * 100, 2)
        rsi_val = round(float(df['rsi'].iloc[-1]), 1)
        sma20 = round(float(df['sma_20'].iloc[-1]), 2)
        sma50 = round(float(df['sma_50'].iloc[-1]), 2)
        m = MODELS[ticker]
        latest = df[m['features']].iloc[-1:]
        scaled = m['scaler'].transform(latest)
        proba = float(m['model'].predict_proba(scaled)[0][1])
        if proba >= 0.55:
            sig = "BUY / HOLD"
            conf = round(proba * 100)
        elif proba <= 0.45:
            sig = "SELL / STAY OUT"
            conf = round((1 - proba) * 100)
        else:
            sig = "HOLD / NEUTRAL"
            conf = 50
        return jsonify({
            'ticker': ticker,
            'signal': sig,
            'confidence': conf,
            'probability': round(proba, 4),
            'price': round(price, 2),
            'day_change': day_chg,
            'rsi': rsi_val,
            'sma20': sma20,
            'sma50': sma50,
            'model': f'XGBoost AI v1 ({ticker})',
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/signals/all')
def all_signals():
    results = {}
    for ticker in MODELS.keys():
        try:
            df = build_features(ticker)
            close = df['Close'].squeeze()
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            m = MODELS[ticker]
            latest = df[m['features']].iloc[-1:]
            scaled = m['scaler'].transform(latest)
            proba = float(m['model'].predict_proba(scaled)[0][1])
            if proba >= 0.55:
                sig = "BUY / HOLD"
                conf = round(proba * 100)
            elif proba <= 0.45:
                sig = "SELL / STAY OUT"
                conf = round((1 - proba) * 100)
            else:
                sig = "HOLD / NEUTRAL"
                conf = 50
            results[ticker] = {
                'signal': sig,
                'confidence': conf,
                'probability': round(proba, 4),
                'price': round(float(price), 2),
                'rsi': round(float(df['rsi'].iloc[-1]), 1),
                'status': 'ok'
            }
        except Exception as e:
            results[ticker] = {'status': 'error', 'error': str(e)}
    return jsonify(results)

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'models': list(MODELS.keys()),
        'count': len(MODELS)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
