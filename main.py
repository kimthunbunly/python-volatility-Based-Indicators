import time
from datetime import datetime
import pandas as pd
import os
from btc_news_prediction import BTCNewsPricePredictor
from eth_prediction import ETHPricePredictor


class TradingMonitor:
    def __init__(self, api_key, interval_seconds=5, symbol='BTC'):
        self.interval_seconds = interval_seconds
        self.prediction_count = 0
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.RESET = '\033[0m'

        # Initialize predictor based on symbol
        self.symbol = symbol
        if symbol == 'BTC':
            self.predictor = BTCNewsPricePredictor(api_key)
        else:
            self.predictor = ETHPricePredictor(api_key)

        # Setup data directories
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.signals_dir = os.path.join(self.data_dir, 'signals')
        self.history_dir = os.path.join(self.data_dir, 'history')

        # Create directories if they don't exist
        os.makedirs(self.signals_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        # Update file paths
        self.results_file = os.path.join(
            self.signals_dir, f'{symbol.lower()}_trading_signals.csv')
        self.trade_history_file = os.path.join(
            self.history_dir, f'{symbol.lower()}_trade_history.csv')

        self.initialize_results_file()
        self.initialize_trade_history()

        self.active_trades = []
        self.completed_trades = []

        self.current_prediction = None
        self.prediction_start_time = None
        self.prediction_duration = 300  # 5 minutes in seconds

    def initialize_results_file(self):
        if not os.path.exists(self.results_file):
            columns = ['timestamp', 'price', 'position', 'entry', 'stop_loss',
                       'take_profit', 'confidence', 'prediction']
            pd.DataFrame(columns=columns).to_csv(
                self.results_file, index=False)

    def initialize_trade_history(self):
        if not os.path.exists(self.trade_history_file):
            columns = [
                'timestamp', 'entry_time', 'exit_time', 'position', 
                'entry_price', 'exit_price', 'current_price',
                'stop_loss', 'take_profit', 'outcome', 'profit_loss_pct',
                'target_price', 'prediction', 'confidence', 'predicted_change_pct',
                'actual_change_pct', 'accuracy'
            ]
            pd.DataFrame(columns=columns).to_csv(self.trade_history_file, index=False)

    def save_result(self, prediction, trading_signal):
        """Save only activated trades to signals history"""
        if any(trade['position'] == trading_signal['position']
               for trade in self.active_trades):
            current_time = datetime.now()
            result = {
                'timestamp': current_time,
                'price': prediction['current_price'],
                'position': trading_signal['position'],
                'entry': trading_signal['entry'],
                'stop_loss': trading_signal['stop_loss'],
                'take_profit': trading_signal['take_profit'],
                'confidence': prediction['confidence'],
                'prediction': prediction['prediction']
            }
            pd.DataFrame([result]).to_csv(
                self.results_file, mode='a', header=False, index=False)
            return result
        return None

    def save_completed_trade(self, trade, current_prediction=None):
        current_time = datetime.now()
        current_price = trade['exit_price']
        
        trade_data = {
            'timestamp': current_time,
            'entry_time': trade['entry_time'],
            'exit_time': trade['exit_time'],
            'position': trade['position'],
            'entry_price': trade['entry'],
            'exit_price': trade['exit_price'],
            'current_price': current_price,
            'stop_loss': trade['stop_loss'],
            'take_profit': trade['take_profit'],
            'outcome': trade['outcome'],
            'profit_loss_pct': trade['profit_loss_pct']
        }
        
        # Add prediction data if available
        if current_prediction:
            accuracy = self.calculate_prediction_accuracy(current_price, current_prediction)
            trade_data.update({
                'target_price': current_prediction['price'],
                'prediction': 'BULLISH' if current_prediction['change_percent'] > 0 else 'BEARISH',
                'confidence': trade.get('confidence', 0),
                'predicted_change_pct': current_prediction['change_percent'],
                'actual_change_pct': (current_price - current_prediction['initial_price']) / current_prediction['initial_price'] * 100,
                'accuracy': accuracy
            })
        
        pd.DataFrame([trade_data]).to_csv(self.trade_history_file, mode='a', header=False, index=False)

    def save_full_history(self, prediction, trading_signal, pred_target):
        """Save every prediction to full history"""
        current_time = datetime.now()
        history_entry = {
            'timestamp': current_time,
            'price': prediction['current_price'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'position': trading_signal['position'],
            'target_price': pred_target['price'],
            'predicted_change_pct': pred_target['change_percent']
        }
        pd.DataFrame([history_entry]).to_csv(
            self.full_history_file, mode='a', header=False, index=False)

    def track_active_trades(self, current_price):
        # Create copy to allow modification during iteration
        for trade in self.active_trades[:]:
            entry_price = trade['entry']
            position = trade['position']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            # Calculate profit/loss percentage
            if position == 'LONG':
                profit_pct = (current_price - entry_price) / entry_price * 100
                triggered = (current_price >= take_profit) or (
                    current_price <= stop_loss)
                outcome = 'WIN' if current_price >= take_profit else 'LOSS'
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price * 100
                triggered = (current_price <= take_profit) or (
                    current_price >= stop_loss)
                outcome = 'WIN' if current_price <= take_profit else 'LOSS'

            # Check if stop loss or take profit is hit
            if triggered:
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = current_price
                trade['outcome'] = outcome
                trade['profit_loss_pct'] = profit_pct

                # Save to history and remove from active trades
                self.save_completed_trade(trade)
                self.active_trades.remove(trade)

                print(
                    f"\n{self.RED if outcome == 'LOSS' else self.GREEN}Trade Completed:")
                print(f"Position: {position}")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Exit: ${current_price:,.2f}")
                print(f"Profit/Loss: {profit_pct:.2f}%")
                print(f"Outcome: {outcome}{self.RESET}\n")

    def print_trade_status(self, current_price):
        if self.active_trades:
            print("\nActive Trades Status:")
            for trade in self.active_trades:
                position = trade['position']
                entry_price = trade['entry']
                entry_time = trade['entry_time']

                # Calculate current P/L
                if position == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    profit_pct = (entry_price - current_price) / entry_price * 100

                # Calculate time elapsed since trade entry
                elapsed = (datetime.now() - entry_time).total_seconds()
                remaining = max(0, self.prediction_duration - elapsed)

                # Color coding
                status_color = self.GREEN if profit_pct > 0 else self.RED
                
                # Basic trade info always shown
                print(f"{status_color}Position: {position}")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Current: ${current_price:,.2f}")
                print(f"P/L: {profit_pct:.2f}%")
                
                # Show extended info only during first 5 minutes
                if remaining > 0 and self.current_prediction and trade.get('initial_prediction'):
                    target_price = trade['initial_prediction']['price']
                    target_pct = (target_price - entry_price) / entry_price * 100
                    mins_remaining = int(remaining // 60)
                    secs_remaining = int(remaining % 60)
                    
                    print(f"Target Price: ${target_price:,.2f} ({target_pct:+.2f}%)")
                    print(f"Time to Target: {mins_remaining}m {secs_remaining}s")
                    print(f"Stop Loss: ${trade['stop_loss']:.2f}")
                    print(f"Take Profit: ${trade['take_profit']:.2f}")
                print(f"{self.RESET}")

    def add_new_trade(self, trading_signal):
        # Add entry time and initial prediction to trading signal
        trading_signal['entry_time'] = datetime.now()
        trading_signal['initial_prediction'] = self.current_prediction
        self.active_trades.append(trading_signal)

        position_color = self.GREEN if trading_signal['position'] == 'LONG' else self.RED
        print(f"\n{position_color}New Trade Opened:")
        print(f"Position: {trading_signal['position']}")
        print(f"Entry: ${trading_signal['entry']:,.2f}")
        print(f"Stop Loss: ${trading_signal['stop_loss']:,.2f}")
        print(
            f"Take Profit: ${trading_signal['take_profit']:,.2f}{self.RESET}\n")

    def print_summary(self, prediction, trading_signal):
        current_time = datetime.now()
        current_price = prediction['current_price']

        print("\n" + "="*50)
        print(f"Timestamp: {current_time}")
        print(
            f"Current {self.symbol} Price: ${prediction['current_price']:,.2f}")
        print(f"Prediction: {prediction['prediction']}")

        position_color = self.GREEN if trading_signal['position'] == 'LONG' else self.RED
        print(
            f"Position: {position_color}{trading_signal['position']}{self.RESET}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Risk/Reward: {trading_signal['risk_reward']:.2f}")

        # Entry, Stop Loss, Take Profit with percentages
        stop_loss_pct = abs(
            (trading_signal['stop_loss'] - trading_signal['entry']) / trading_signal['entry'] * 100)
        take_profit_pct = abs(
            (trading_signal['take_profit'] - trading_signal['entry']) / trading_signal['entry'] * 100)

        print(f"\nTrade Levels:")
        print(
            f"Entry: {position_color}${trading_signal['entry']:,.2f}{self.RESET}")
        print(
            f"Stop Loss: {position_color}${trading_signal['stop_loss']:,.2f} ({stop_loss_pct:.2f}%){self.RESET}")
        print(
            f"Take Profit: {position_color}${trading_signal['take_profit']:,.2f} ({take_profit_pct:.2f}%){self.RESET}")

        # 5-minute prediction with countdown
        print("\n5-Minute Price Prediction:")
        pred = self.calculate_five_minute_prediction(
            current_price, prediction, trading_signal)

        # Calculate time remaining and progress
        elapsed_seconds = (
            current_time - self.prediction_start_time).total_seconds()
        remaining_seconds = max(0, self.prediction_duration - elapsed_seconds)
        minutes_remaining = int(remaining_seconds // 60)
        seconds_remaining = int(remaining_seconds % 60)

        # Calculate current accuracy
        if self.current_prediction:
            initial_price = self.current_prediction['initial_price']
            target_price = self.current_prediction['price']
            current_deviation = abs(current_price - target_price)
            max_deviation = abs(target_price - initial_price)
            accuracy = max(0, (1 - current_deviation/max_deviation)
                           * 100) if max_deviation > 0 else 0

            change_color = self.GREEN if pred['change_percent'] > 0 else self.RED
            print(f"Time Remaining: {minutes_remaining}m {seconds_remaining}s")
            print(f"Initial Price: ${initial_price:,.2f}")
            print(f"Current Price: ${current_price:,.2f}")
            print(
                f"Target Price: {change_color}${pred['price']:,.2f} ({pred['change_percent']:+.2f}%){self.RESET}")
            print(
                f"Range: ${pred['lower']:,.2f} ({pred['lower_percent']:+.2f}%) - ${pred['upper']:,.2f} ({pred['upper_percent']:+.2f}%)")
            print(f"Current Accuracy: {accuracy:.2f}%")

        print("="*50 + "\n")

    def should_enter_trade(self, trading_signal, prediction):
        """More aggressive entry conditions"""
        confidence_threshold = 0.2  # Lowered threshold further
        max_active_trades = 3
        min_risk_reward = 1.5  # Minimum risk/reward ratio

        # Enhanced entry conditions
        basic_conditions = (
            len(self.active_trades) < max_active_trades and
            trading_signal['confidence'] > confidence_threshold and
            trading_signal['risk_reward'] >= min_risk_reward
        )

        # Check if we already have a similar position
        has_similar_position = any(
            trade['position'] == trading_signal['position']
            for trade in self.active_trades
        )

        return basic_conditions and not has_similar_position

    def save_signal_result(self, prediction, trading_signal, current_prediction, current_price):
        """Save signal results after 5-minute countdown completion"""
        current_time = datetime.now()
        
        if current_prediction and self.prediction_start_time:
            elapsed = (current_time - self.prediction_start_time).total_seconds()
            if elapsed >= self.prediction_duration:  # 5 minutes passed
                result = {
                    'timestamp': current_time,
                    'initial_price': current_prediction['initial_price'],
                    'final_price': current_price,
                    'target_price': current_prediction['price'],
                    'prediction': prediction['prediction'],
                    'position': trading_signal['position'],
                    'confidence': prediction['confidence'],
                    'risk_reward': trading_signal['risk_reward'],
                    'entry': trading_signal['entry'],
                    'stop_loss': trading_signal['stop_loss'],
                    'take_profit': trading_signal['take_profit'],
                    'actual_change_pct': (current_price - current_prediction['initial_price']) / current_prediction['initial_price'] * 100,
                    'predicted_change_pct': current_prediction['change_percent'],
                    'accuracy': self.calculate_prediction_accuracy(current_price, current_prediction)
                }
                
                # Initialize the file with headers if it doesn't exist
                if not os.path.exists(self.results_file):
                    pd.DataFrame(columns=list(result.keys())).to_csv(self.results_file, index=False)
                
                # Save to CSV
                pd.DataFrame([result]).to_csv(self.results_file, mode='a', header=False, index=False)
                return result
        return None

    def calculate_prediction_accuracy(self, current_price, prediction):
        """Calculate accuracy of prediction"""
        initial_price = prediction['initial_price']
        target_price = prediction['price']
        current_deviation = abs(current_price - target_price)
        max_deviation = abs(target_price - initial_price)
        return max(0, (1 - current_deviation/max_deviation) * 100) if max_deviation > 0 else 0

    def calculate_five_minute_prediction(self, current_price, prediction, trading_signal):
        """Calculate 5-minute price prediction"""
        current_time = datetime.now()

        # If we don't have a prediction or the current one has expired
        if (self.current_prediction is None or
            self.prediction_start_time is None or
            (current_time - self.prediction_start_time).total_seconds() >= self.prediction_duration):

            # Calculate new prediction
            atr = abs(trading_signal['take_profit'] - trading_signal['entry']) / 3
            trend_direction = 1 if prediction['prediction'] == 'BULLISH' else -1
            confidence = prediction['confidence']

            estimated_move = atr * confidence * trend_direction
            predicted_price = current_price + estimated_move

            volatility = atr * 0.3
            upper_bound = predicted_price + volatility
            lower_bound = predicted_price - volatility

            self.current_prediction = {
                'price': predicted_price,
                'upper': upper_bound,
                'lower': lower_bound,
                'change_percent': (predicted_price - current_price) / current_price * 100,
                'upper_percent': (upper_bound - current_price) / current_price * 100,
                'lower_percent': (lower_bound - current_price) / current_price * 100,
                'initial_price': current_price
            }
            self.prediction_start_time = current_time

        return self.current_prediction

    def run(self):
        print(f"Starting {self.symbol} Trading Monitor")
        print(f"Interval: {self.interval_seconds} seconds")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Get predictions
                if self.symbol == 'BTC':
                    prediction, trading_signal = self.predictor.run_prediction()
                else:
                    prediction, trading_signal = self.predictor.predict()

                current_price = prediction['current_price']
                # Update method call to use new name
                pred_target = self.calculate_five_minute_prediction(
                    current_price, prediction, trading_signal)

                # Process trades
                self.track_active_trades(current_price)
                
                # Save completed 5-minute predictions
                self.save_signal_result(prediction, trading_signal, self.current_prediction, current_price)

                if self.should_enter_trade(trading_signal, prediction):
                    self.add_new_trade(trading_signal)
                    print(f"{self.GREEN if trading_signal['position'] == 'LONG' else self.RED}"
                          f"New trade entered with confidence: {trading_signal['confidence']:.2%}{self.RESET}")

                self.print_summary(prediction, trading_signal)
                self.print_trade_status(current_price)

                # Save completed trades with prediction data
                if self.current_prediction and self.prediction_start_time:
                    elapsed = (datetime.now() - self.prediction_start_time).total_seconds()
                    if elapsed >= self.prediction_duration:
                        for trade in self.completed_trades:
                            self.save_completed_trade(trade, self.current_prediction)
                        self.completed_trades = []  # Clear completed trades

                self.prediction_count += 1
                print(f"\nTotal predictions made: {self.prediction_count}")
                print(f"Active trades: {len(self.active_trades)}")
                print(f"Next update in {self.interval_seconds} seconds...")
                time.sleep(self.interval_seconds)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")


if __name__ == "__main__":
    NEWS_API_KEY = '89a28001bf9549b091031ad555b24a2d'
    INTERVAL_SECONDS = 5

    # Monitor BTC
    # btc_monitor = TradingMonitor(NEWS_API_KEY, INTERVAL_SECONDS, 'BTC')
    # btc_monitor.run()

    # To monitor ETH instead, use:
    eth_monitor = TradingMonitor(NEWS_API_KEY, INTERVAL_SECONDS, 'ETH')
    eth_monitor.run()
