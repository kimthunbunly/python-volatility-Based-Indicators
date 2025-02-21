import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class XRPVolatilityAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.symbol = 'XRP/USDT'
        
    def fetch_data(self, timeframe='1h', limit=500):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_indicators(self, df):
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Bollinger Bands
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        
        # Keltner Channels
        df['kc_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['kc_mid'] = ta.volatility.keltner_channel_mband(df['high'], df['low'], df['close'], window=20)
        df['kc_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
        
        # Volatility-based signals
        df['volatility_breakout'] = np.where(
            (df['close'] > df['bb_high']) & (df['close'] > df['kc_high']),
            1,  # Strong upward volatility
            np.where(
                (df['close'] < df['bb_low']) & (df['close'] < df['kc_low']),
                -1,  # Strong downward volatility
                0  # No significant volatility
            )
        )
        
        return df
    
    def generate_signals(self, df):
        signals = []
        for index, row in df.iterrows():
            if row['volatility_breakout'] == 1:
                signals.append({
                    'timestamp': row['timestamp'],
                    'signal': 'LONG',
                    'price': row['close'],
                    'atr': row['atr']
                })
            elif row['volatility_breakout'] == -1:
                signals.append({
                    'timestamp': row['timestamp'],
                    'signal': 'SHORT',
                    'price': row['close'],
                    'atr': row['atr']
                })
        return signals

    def plot_analysis(self, df):
        plt.figure(figsize=(15, 10))
        
        # Plot price and indicators
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['close'], label='Price', color='blue')
        plt.plot(df['timestamp'], df['bb_high'], label='BB High', color='red', alpha=0.7)
        plt.plot(df['timestamp'], df['bb_low'], label='BB Low', color='red', alpha=0.7)
        plt.plot(df['timestamp'], df['kc_high'], label='KC High', color='green', alpha=0.7)
        plt.plot(df['timestamp'], df['kc_low'], label='KC Low', color='green', alpha=0.7)
        
        # Plot ATR
        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['atr'], label='ATR', color='purple')
        
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        df = self.fetch_data()
        df = self.calculate_indicators(df)
        signals = self.generate_signals(df)
        
        print("\nLatest Volatility Signals:")
        for signal in signals[-3:]:  # Show last 3 signals
            print(f"Time: {signal['timestamp']}, Signal: {signal['signal']}, "
                  f"Price: {signal['price']:.4f}, ATR: {signal['atr']:.4f}")
        
        self.plot_analysis(df)

if __name__ == "__main__":
    analyzer = XRPVolatilityAnalyzer()
    analyzer.run_analysis()
