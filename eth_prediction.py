import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ETHPricePredictor:
    def __init__(self, news_api_key):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.newsapi = NewsApiClient(api_key=news_api_key)
        self.symbol = 'ETH/USDT'

    def fetch_price_data(self, timeframe='1h', limit=168):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_news_data(self, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news_data = []
        try:
            # Include Ethereum-specific keywords
            response = self.newsapi.get_everything(
                q='ethereum OR eth OR defi OR "smart contract"',
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
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        
        # Sentiment features
        df['sentiment_ma'] = df['sentiment_polarity'].rolling(24).mean()
        df['sentiment_std'] = df['sentiment_polarity'].rolling(24).std()
        
        df.dropna(inplace=True)
        return df

    def generate_trading_signals(self, prediction, df):
        current_price = prediction['current_price']
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
        
        if prediction['prediction'] == "BULLISH":
            entry_price = current_price
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
            signal = {
                'position': 'LONG',
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs((take_profit - entry_price) / (entry_price - stop_loss)),
                'confidence': prediction['confidence'],
                'sentiment': prediction['sentiment_score']
            }
        else:
            entry_price = current_price
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)
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

    def predict(self):
        # Fetch and prepare data
        price_df = self.fetch_price_data()
        news_df = self.fetch_news_data()
        df = self.prepare_features(price_df, news_df)
        
        # Prepare features for prediction
        features = ['returns', 'volatility', 'rsi', 'macd', 
                   'sentiment_polarity', 'sentiment_ma', 'sentiment_std']
        
        # Train model on historical data
        X = df[features].values[:-1]
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Generate prediction
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        prediction_prob = model.predict(latest_features_scaled)[0]
        
        prediction = {
            'timestamp': df.index[-1],
            'current_price': df['close'].iloc[-1],
            'prediction': "BULLISH" if prediction_prob > 0.5 else "BEARISH",
            'confidence': abs(prediction_prob - 0.5) * 2,
            'sentiment_score': df['sentiment_ma'].iloc[-1]
        }
        
        trading_signal = self.generate_trading_signals(prediction, price_df)
        
        return prediction, trading_signal

if __name__ == "__main__":
    NEWS_API_KEY = '89a28001bf9549b091031ad555b24a2d'
    predictor = ETHPricePredictor(NEWS_API_KEY)
    prediction, trading_signal = predictor.predict()
    
    print("\nETH Price Prediction:")
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
