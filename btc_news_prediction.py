import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
import requests
import time
import ta  # Add this import for technical analysis indicators
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class BTCNewsPricePredictor:
    def __init__(self, news_api_key):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.newsapi = NewsApiClient(api_key=news_api_key)
        self.symbol = 'BTC/USDT'
        
    def fetch_price_data(self, timeframe='1h', limit=168):  # 7 days of hourly data
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_news_data(self, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news_data = []
        try:
            response = self.newsapi.get_everything(
                q='bitcoin OR crypto OR cryptocurrency',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt'
            )
            
            for article in response['articles']:
                sentiment = TextBlob(article['title'] + ' ' + (article['description'] or '')).sentiment
                news_data.append({
                    'timestamp': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    'sentiment_polarity': sentiment.polarity,
                    'sentiment_subjectivity': sentiment.subjectivity
                })
        except Exception as e:
            print(f"Error fetching news: {e}")
        
        return pd.DataFrame(news_data)

    def prepare_features(self, price_df, news_df):
        # Aggregate news sentiment by hour
        news_df.set_index('timestamp', inplace=True)
        hourly_sentiment = news_df.resample('1H').mean().fillna(method='ffill')
        
        # Merge price and sentiment data
        df = price_df.set_index('timestamp').join(hourly_sentiment)
        df.fillna(method='ffill', inplace=True)
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(24).std()
        df['price_ma'] = df['close'].rolling(24).mean()
        df['volume_ma'] = df['volume'].rolling(24).mean()
        
        # Sentiment features
        df['sentiment_ma'] = df['sentiment_polarity'].rolling(24).mean()
        df['sentiment_std'] = df['sentiment_polarity'].rolling(24).std()
        
        df.dropna(inplace=True)
        return df

    def train_model(self, df):
        # Prepare features and target
        features = ['returns', 'volatility', 'sentiment_polarity', 
                   'sentiment_subjectivity', 'sentiment_ma', 'sentiment_std']
        X = df[features].values
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]  # Remove last row
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[:-1])  # Remove last row to match y
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler, features

    def generate_prediction(self):
        # Fetch data
        price_df = self.fetch_price_data()
        news_df = self.fetch_news_data()
        
        # Prepare data
        df = self.prepare_features(price_df, news_df)
        
        # Train model
        model, scaler, features = self.train_model(df)
        
        # Generate prediction for next period
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        prediction_prob = model.predict(latest_features_scaled)[0]
        
        # Generate signal and confidence
        current_price = df['close'].iloc[-1]
        prediction = {
            'timestamp': df.index[-1],
            'current_price': current_price,
            # BEARISH: Price expected to fall (< current price)
            # BULLISH: Price expected to rise (> current price)
            'prediction': "BULLISH" if prediction_prob > 0.5 else "BEARISH",
            'confidence': abs(prediction_prob - 0.5) * 2,  # Scale to 0-1
            'sentiment_score': df['sentiment_ma'].iloc[-1]
        }
        
        return prediction

    def generate_trading_signals(self, prediction, df):
        current_price = prediction['current_price']
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
        
        if prediction['prediction'] == "BULLISH":
            entry_price = current_price
            stop_loss = entry_price - (2 * atr)  # 2 ATR for stop loss
            take_profit = entry_price + (3 * atr)  # 3 ATR for take profit
            signal = {
                'position': 'LONG',
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs((take_profit - entry_price) / (entry_price - stop_loss)),
                'confidence': prediction['confidence'],
                'sentiment': prediction['sentiment_score']
            }
        else:  # BEARISH
            entry_price = current_price
            stop_loss = entry_price + (2 * atr)  # 2 ATR for stop loss
            take_profit = entry_price - (3 * atr)  # 3 ATR for take profit
            signal = {
                'position': 'SHORT',
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs((take_profit - entry_price) / (stop_loss - entry_price)),
                'confidence': prediction['confidence'],
                'sentiment': prediction['sentiment_score']
            }
        
        return signal

    def run_prediction(self):
        prediction = self.generate_prediction()
        # Get last 168 hours of price data for ATR calculation
        df = self.fetch_price_data(timeframe='1h', limit=168)
        trading_signal = self.generate_trading_signals(prediction, df)
        
        print("\nBTC Price Prediction with News Sentiment:")
        print(f"Timestamp: {prediction['timestamp']}")
        print(f"Current Price: ${prediction['current_price']:,.2f}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Sentiment Score: {prediction['sentiment_score']:.3f}")
        
        print("\nTrading Signal:")
        print(f"Position: {trading_signal['position']}")
        print(f"Entry Price: ${trading_signal['entry']:,.2f}")
        print(f"Stop Loss: ${trading_signal['stop_loss']:,.2f}")
        print(f"Take Profit: ${trading_signal['take_profit']:,.2f}")
        print(f"Risk/Reward Ratio: {trading_signal['risk_reward']:.2f}")
        
        return prediction, trading_signal

if __name__ == "__main__":
    # Replace 'YOUR_NEWS_API_KEY' with your actual NewsAPI key
    predictor = BTCNewsPricePredictor('89a28001bf9549b091031ad555b24a2d')
    predictor.run_prediction()
