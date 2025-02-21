import time
from datetime import datetime
import pandas as pd
import os
from btc_news_prediction import BTCNewsPricePredictor

class TradingMonitor:
    def __init__(self, api_key, interval_seconds=5):
        self.predictor = BTCNewsPricePredictor(api_key)
        self.interval_seconds = interval_seconds
        self.prediction_count = 0
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.RESET = '\033[0m'
        self.results_file = 'trading_signals.csv'
        self.initialize_results_file()

    def initialize_results_file(self):
        if not os.path.exists(self.results_file):
            columns = ['timestamp', 'price', 'position', 'entry', 'stop_loss', 
                      'take_profit', 'confidence', 'prediction']
            pd.DataFrame(columns=columns).to_csv(self.results_file, index=False)

    def save_result(self, prediction, trading_signal):
        result = {
            'timestamp': datetime.now(),
            'price': prediction['current_price'],
            'position': trading_signal['position'],
            'entry': trading_signal['entry'],
            'stop_loss': trading_signal['stop_loss'],
            'take_profit': trading_signal['take_profit'],
            'confidence': prediction['confidence'],
            'prediction': prediction['prediction']
        }
        pd.DataFrame([result]).to_csv(self.results_file, mode='a', header=False, index=False)
        return result

    def print_summary(self, prediction, trading_signal):
        print("\n" + "="*50)
        print(f"Timestamp: {prediction['timestamp']}")
        print(f"Current BTC Price: ${prediction['current_price']:,.2f}")
        print(f"Prediction: {prediction['prediction']}")
        
        # Add color to position
        position_color = self.GREEN if trading_signal['position'] == 'LONG' else self.RED
        print(f"Position: {position_color}{trading_signal['position']}{self.RESET}")
        
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Risk/Reward: {trading_signal['risk_reward']:.2f}")
        
        # Calculate percentages
        stop_loss_pct = abs((trading_signal['stop_loss'] - trading_signal['entry']) / trading_signal['entry'] * 100)
        take_profit_pct = abs((trading_signal['take_profit'] - trading_signal['entry']) / trading_signal['entry'] * 100)
        
        # Add color to entry, stop loss, and take profit with percentages
        print(f"Entry: {position_color}${trading_signal['entry']:,.2f}{self.RESET}")
        print(f"Stop Loss: {position_color}${trading_signal['stop_loss']:,.2f} ({stop_loss_pct:.2f}%){self.RESET}")
        print(f"Take Profit: {position_color}${trading_signal['take_profit']:,.2f} ({take_profit_pct:.2f}%){self.RESET}")
        print("="*50 + "\n")

    def run(self):
        print(f"Starting BTC Trading Monitor")
        print(f"Interval: {self.interval_seconds} seconds")
        print("Press Ctrl+C to stop\n")
        print(f"Saving results to: {self.results_file}")

        try:
            while True:
                prediction, trading_signal = self.predictor.run_prediction()
                self.save_result(prediction, trading_signal)
                self.print_summary(prediction, trading_signal)
                self.prediction_count += 1
                print(f"Total predictions made: {self.prediction_count}")
                print(f"Next update in {self.interval_seconds} seconds...")
                time.sleep(self.interval_seconds)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            print(f"Results saved to: {self.results_file}")

if __name__ == "__main__":
    NEWS_API_KEY = '89a28001bf9549b091031ad555b24a2d'
    INTERVAL_SECONDS = 5
    
    monitor = TradingMonitor(NEWS_API_KEY, INTERVAL_SECONDS)
    monitor.run()
