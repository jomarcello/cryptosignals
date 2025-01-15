import ccxt
import requests
import time
import numpy as np
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuratie
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Exchange configuratie
EXCHANGE_ID = 'kraken'
EXCHANGE_CONFIG = {
    'enableRateLimit': True,
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'options': {
        'adjustForTimeDifference': True
    }
}

# N8N Webhook configuratie
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def calculate_ema(prices, period):
    """Calculate EMA technical indicator"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD technical indicator"""
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    
    return macd_line, signal_line

def initialize_exchange():
    """Initialize exchange connection"""
    try:
        # Verify that API credentials are set
        if not EXCHANGE_CONFIG['apiKey'] or not EXCHANGE_CONFIG['secret']:
            raise ValueError("API credentials not found in environment variables")
            
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class(EXCHANGE_CONFIG)
        exchange.load_markets()
        logger.info(f"Connected to {EXCHANGE_ID}")
        return exchange
    except Exception as e:
        logger.error(f"Exchange initialization failed: {str(e)}")
        return None

def fetch_and_analyze_data(exchange, symbol="BTC/USD", timeframe='1m'):
    """Fetch and analyze market data"""
    try:
        # Kraken-specific symbol adjustment
        if symbol == "BTC/USD":
            symbol = "XBT/USD"  # Changed from XBT/ZUSD to XBT/USD
        
        # Fetch data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
        if not ohlcv or len(ohlcv) < 50:
            logger.error(f"No data received for {symbol}")
            return None

        # Convert to numpy arrays
        close_prices = np.array([candle[4] for candle in ohlcv])
        
        # Calculate technical indicators
        rsi = calculate_rsi(close_prices)
        macd, macd_signal = calculate_macd(close_prices)
        
        return {
            'rsi': rsi[-1],
            'macd': macd[-1],
            'macd_signal': macd_signal[-1],
            'price': close_prices[-1],
            'timestamp': ohlcv[-1][0]
        }
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}")
        return None

def generate_signal(analysis):
    """Generate trading signal based on analysis"""
    if analysis is None:
        return None
        
    signal = None
    reason = []
    
    # RSI Signals
    if analysis['rsi'] > 70:
        signal = "SELL"
        reason.append(f"RSI overbought ({analysis['rsi']:.2f})")
    elif analysis['rsi'] < 30:
        signal = "BUY"
        reason.append(f"RSI oversold ({analysis['rsi']:.2f})")
        
    # MACD Signals
    if analysis['macd'] > analysis['macd_signal']:
        if signal != "SELL":  # Only if RSI doesn't say SELL
            signal = "BUY"
            reason.append("MACD bullish crossover")
    elif analysis['macd'] < analysis['macd_signal']:
        if signal != "BUY":  # Only if RSI doesn't say BUY
            signal = "SELL"
            reason.append("MACD bearish crossover")
            
    if signal:
        return {
            'signal': signal,
            'reason': " & ".join(reason),
            'indicators': analysis
        }
    return None

def send_to_n8n(signal_data, symbol):
    """Send signal to N8N webhook"""
    try:
        # Verify webhook URL is set
        if not WEBHOOK_URL:
            raise ValueError("Webhook URL not found in environment variables")
            
        params = {
            "symbol": symbol,
            "timeframe": "1m",  # Updated timeframe
            "signal": signal_data['signal'],
            "reason": signal_data['reason'],
            "price": str(signal_data['indicators']['price']),
            "rsi": str(signal_data['indicators']['rsi']),
            "timestamp": datetime.fromtimestamp(signal_data['indicators']['timestamp']/1000).isoformat()
        }
        
        response = requests.get(WEBHOOK_URL, params=params)
        if response.status_code == 200:
            logger.info(f"Signal sent successfully: {params['signal']} for {symbol}")
            logger.info(f"Current price: {params['price']}, RSI: {params['rsi']}")
            logger.info(f"Reason: {params['reason']}")  # Added reason to logging
        else:
            logger.error(f"Error sending signal: {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"Error sending to n8n: {str(e)}")

def main():
    """Main function to run the signal generator"""
    # Verify environment variables
    required_env_vars = ['KRAKEN_API_KEY', 'KRAKEN_API_SECRET', 'WEBHOOK_URL']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    exchange = initialize_exchange()
    if not exchange:
        return

    # Only analyze BTC/USD
    symbol = "BTC/USD"  # Will be converted to XBT/USD for Kraken
    timeframe = "1m"  # Changed to 1 minute timeframe
    
    try:
        while True:
            try:
                logger.info(f"Analyzing {symbol} on {timeframe} timeframe...")
                
                # Analyze data
                analysis = fetch_and_analyze_data(exchange, symbol, timeframe)
                if analysis:
                    # Generate signal
                    signal = generate_signal(analysis)
                    if signal:
                        # Send signal to n8n
                        send_to_n8n(signal, symbol)
                    else:
                        logger.info(f"No trading signal generated for {symbol}")
                        logger.info(f"Current RSI: {analysis['rsi']:.2f}, Price: {analysis['price']}")
                
                # Wait 1 minute before next check
                logger.info(f"Waiting 1 minute for next {symbol} analysis...")
                time.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)  # Wait 10 seconds before retrying if there's an error
            
    except KeyboardInterrupt:
        logger.info("Script stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
