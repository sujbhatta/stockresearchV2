import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (optional)
# If GOOGLE_API_KEY is already set in system environment, it will be used
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Configure Gemini - check both .env file and system environment
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"✓ Google Gemini API configured successfully")
else:
    print("⚠ WARNING: GOOGLE_API_KEY not found. AI report generation will not work.")
    print("  Set GOOGLE_API_KEY in system environment or create a .env file")

# Cache directory
CACHE_DIR = 'cache'
CACHE_TTL_HOURS = 24

# NIFTY 50 Symbols with .NS suffix for NSE
NIFTY_50_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'WIPRO', 'ASIANPAINT', 'HCLTECH',
    'MARUTI', 'BAJFINANCE', 'TITAN', 'SUNPHARMA', 'TECHM', 'NESTLEIND', 'POWERGRID',
    'ULTRACEMCO', 'ADANIENT', 'TATAMOTORS', 'ONGC', 'TATASTEEL', 'JSWSTEEL', 'NTPC',
    'INDUSINDBK', 'M&M', 'COALINDIA', 'BAJAJFINSV', 'HINDALCO', 'DRREDDY', 'GRASIM',
    'DIVISLAB', 'BAJAJ-AUTO', 'BRITANNIA', 'HEROMOTOCO', 'ADANIPORTS', 'CIPLA', 'UPL',
    'SBILIFE', 'EICHERMOT', 'BPCL', 'TATACONSUM', 'APOLLOHOSP', 'SHREECEM', 'HDFC'
]


def get_cache_path(symbol):
    """Get cache file path for a symbol"""
    return os.path.join(CACHE_DIR, f"{symbol}.NS.csv")


def is_cache_valid(cache_path):
    """Check if cache file exists and is less than 24 hours old"""
    if not os.path.exists(cache_path):
        return False

    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    return age < timedelta(hours=CACHE_TTL_HOURS)


def fetch_stock_data(symbol):
    """Fetch stock data from yfinance or cache"""
    cache_path = get_cache_path(symbol)

    # Check cache first
    if is_cache_valid(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}")

    # Fetch from yfinance
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="1y")

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path)

        return df
    except Exception as e:
        print(f"Error fetching data from yfinance for {symbol}: {e}")

        # Try to load from cache as fallback
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df
            except Exception as e2:
                print(f"Error reading fallback cache for {symbol}: {e2}")

        raise Exception(f"Could not fetch data for {symbol}")


def calculate_moving_average(df, window):
    """Calculate moving average"""
    return df['Close'].rolling(window=window).mean()


def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_stock_info(symbol):
    """Get comprehensive stock information"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        return info
    except Exception as e:
        print(f"Error fetching stock info for {symbol}: {e}")
        return {}


def determine_trend(df):
    """Determine 3-month price trend"""
    if len(df) < 60:
        return "Insufficient data"

    three_months_ago = df['Close'].iloc[-60]
    current = df['Close'].iloc[-1]
    change_pct = ((current - three_months_ago) / three_months_ago) * 100

    if change_pct > 10:
        return f"Strong Uptrend (+{change_pct:.1f}%)"
    elif change_pct > 0:
        return f"Uptrend (+{change_pct:.1f}%)"
    elif change_pct > -10:
        return f"Downtrend ({change_pct:.1f}%)"
    else:
        return f"Strong Downtrend ({change_pct:.1f}%)"


@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html', stocks=NIFTY_50_STOCKS)


@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """Get stock data with technical indicators"""
    try:
        # Fetch OHLCV data
        df = fetch_stock_data(symbol)

        # Calculate technical indicators
        df['MA20'] = calculate_moving_average(df, 20)
        df['MA50'] = calculate_moving_average(df, 50)
        df['MA200'] = calculate_moving_average(df, 200)
        df['RSI'] = calculate_rsi(df)

        # Get stock info
        info = get_stock_info(symbol)

        # Prepare response
        current_price = df['Close'].iloc[-1]
        high_52w = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
        low_52w = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()

        # Get latest indicator values
        ma20 = df['MA20'].iloc[-1] if not pd.isna(df['MA20'].iloc[-1]) else None
        ma50 = df['MA50'].iloc[-1] if not pd.isna(df['MA50'].iloc[-1]) else None
        ma200 = df['MA200'].iloc[-1] if not pd.isna(df['MA200'].iloc[-1]) else None
        rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else None

        # Prepare OHLCV data for chart
        ohlcv_data = []
        for date, row in df.iterrows():
            ohlcv_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
                'ma20': float(row['MA20']) if not pd.isna(row['MA20']) else None,
                'ma50': float(row['MA50']) if not pd.isna(row['MA50']) else None,
                'ma200': float(row['MA200']) if not pd.isna(row['MA200']) else None,
                'rsi': float(row['RSI']) if not pd.isna(row['RSI']) else None,
            })

        response_data = {
            'symbol': symbol,
            'price': round(current_price, 2),
            'high52w': round(high_52w, 2),
            'low52w': round(low_52w, 2),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'ma20': round(ma20, 2) if ma20 else None,
            'ma50': round(ma50, 2) if ma50 else None,
            'ma200': round(ma200, 2) if ma200 else None,
            'rsi': round(rsi, 2) if rsi else None,
            'ohlcv_data': ohlcv_data
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/<symbol>')
def generate_report(symbol):
    """Generate AI-powered investment report using Gemini"""
    try:
        if not GOOGLE_API_KEY:
            return jsonify({'error': 'GOOGLE_API_KEY not configured'}), 500

        # Fetch stock data
        df = fetch_stock_data(symbol)

        # Calculate technical indicators
        df['MA20'] = calculate_moving_average(df, 20)
        df['MA50'] = calculate_moving_average(df, 50)
        df['MA200'] = calculate_moving_average(df, 200)
        df['RSI'] = calculate_rsi(df)

        # Get stock info
        info = get_stock_info(symbol)

        # Prepare data for prompt
        current_price = df['Close'].iloc[-1]
        high_52w = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
        low_52w = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')

        ma20 = df['MA20'].iloc[-1] if not pd.isna(df['MA20'].iloc[-1]) else 'N/A'
        ma50 = df['MA50'].iloc[-1] if not pd.isna(df['MA50'].iloc[-1]) else 'N/A'
        ma200 = df['MA200'].iloc[-1] if not pd.isna(df['MA200'].iloc[-1]) else 'N/A'
        rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 'N/A'

        trend = determine_trend(df)

        # Format market cap
        if isinstance(market_cap, (int, float)):
            market_cap_cr = market_cap / 10000000  # Convert to crores
            market_cap_str = f"{market_cap_cr:,.0f} Cr"
        else:
            market_cap_str = "N/A"

        # Create prompt
        prompt = f"""You are a professional equity research analyst specializing in Indian large-cap stocks.
Analyze the stock data below and write a concise, actionable investment report.

STOCK: {symbol}
Current Price: ₹{current_price:.2f}
52-Week Range: ₹{low_52w:.2f} - ₹{high_52w:.2f}
P/E Ratio: {pe_ratio if pe_ratio != 'N/A' else 'N/A'}
Market Cap: ₹{market_cap_str}

Technical Indicators:
- MA20: ₹{ma20 if ma20 != 'N/A' else 'N/A'} | MA50: ₹{ma50 if ma50 != 'N/A' else 'N/A'} | MA200: ₹{ma200 if ma200 != 'N/A' else 'N/A'}
- RSI(14): {rsi if rsi != 'N/A' else 'N/A'}
- Price Trend (3mo): {trend}

Write a 200-300 word investment report with these sections:
1. **Investment Thesis** - Why buy/avoid based on valuation and technicals
2. **Valuation Analysis** - Is the P/E justified?
3. **Technical Setup** - What do the MAs and RSI tell us?
4. **Key Risks** - What could go wrong?
5. **Recommendation** - BUY/HOLD/SELL with 12-month price target

Be direct, analytical, avoid hype. Assume reader understands stocks."""

        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)

        report = response.text

        return jsonify({'report': report})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
