import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from datetime import datetime

class SOLTrendAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.symbol = 'SOL/USDT'
        
    def fetch_data(self, timeframe='4h', limit=500):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_indicators(self, df):
        # EMAs
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_55'] = ta.trend.ema_indicator(df['close'], window=55)
        
        # MACD
        df['macd_line'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_hist'] = ta.trend.macd_diff(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # Trend Strength Signal
        df['trend_strength'] = np.where(
            (df['ema_9'] > df['ema_21']) & 
            (df['ema_21'] > df['ema_55']) & 
            (df['adx'] > 25),
            1,  # Strong uptrend
            np.where(
                (df['ema_9'] < df['ema_21']) & 
                (df['ema_21'] < df['ema_55']) & 
                (df['adx'] > 25),
                -1,  # Strong downtrend
                0  # No clear trend
            )
        )
        
        return df
    
    def generate_signals(self, df):
        signals = []
        for i in range(1, len(df)):
            if (df['trend_strength'].iloc[i] == 1 and 
                df['rsi'].iloc[i] < 70 and 
                df['macd_hist'].iloc[i] > 0 and 
                df['macd_hist'].iloc[i-1] <= 0):
                
                signals.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'signal': 'LONG',
                    'price': df['close'].iloc[i],
                    'strength': df['adx'].iloc[i]
                })
            elif (df['trend_strength'].iloc[i] == -1 and 
                  df['rsi'].iloc[i] > 30 and 
                  df['macd_hist'].iloc[i] < 0 and 
                  df['macd_hist'].iloc[i-1] >= 0):
                
                signals.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'signal': 'SHORT',
                    'price': df['close'].iloc[i],
                    'strength': df['adx'].iloc[i]
                })
        return signals

    def plot_analysis(self, df):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price and EMAs
        ax1.plot(df['timestamp'], df['close'], label='Price', color='blue')
        ax1.plot(df['timestamp'], df['ema_9'], label='EMA 9', color='red', alpha=0.7)
        ax1.plot(df['timestamp'], df['ema_21'], label='EMA 21', color='green', alpha=0.7)
        ax1.plot(df['timestamp'], df['ema_55'], label='EMA 55', color='purple', alpha=0.7)
        ax1.set_title('SOL Price and EMAs')
        ax1.legend()
        
        # MACD
        ax2.plot(df['timestamp'], df['macd_line'], label='MACD', color='blue')
        ax2.plot(df['timestamp'], df['macd_signal'], label='Signal', color='orange')
        ax2.bar(df['timestamp'], df['macd_hist'], label='Histogram', color='gray', alpha=0.5)
        ax2.set_title('MACD')
        ax2.legend()
        
        # ADX and RSI
        ax3.plot(df['timestamp'], df['adx'], label='ADX', color='purple')
        ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='green')
        ax3.axhline(y=25, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='r', linestyle='--', alpha=0.5)
        ax3.set_title('ADX and RSI')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        df = self.fetch_data()
        df = self.calculate_indicators(df)
        signals = self.generate_signals(df)
        
        print("\nLatest Trend Signals for SOL:")
        for signal in signals[-3:]:  # Show last 3 signals
            print(f"Time: {signal['timestamp']}, Signal: {signal['signal']}, "
                  f"Price: {signal['price']:.4f}, Trend Strength (ADX): {signal['strength']:.2f}")
        
        self.plot_analysis(df)

if __name__ == "__main__":
    analyzer = SOLTrendAnalyzer()
    analyzer.run_analysis()
