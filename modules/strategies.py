import logging
import numpy as np
import pandas as pd
import ta
from modules.config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    FAST_EMA, SLOW_EMA
)

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def get_signal(self, klines):
        """
        Should be implemented by subclasses.
        Returns 'BUY', 'SELL', or None.
        """
        raise NotImplementedError("Each strategy must implement get_signal method")


class RSIStrategy(TradingStrategy):
    """Simple RSI-based strategy"""
    def __init__(self):
        super().__init__('RSI')
        self.rsi_period = RSI_PERIOD
        self.rsi_overbought = RSI_OVERBOUGHT
        self.rsi_oversold = RSI_OVERSOLD
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Get the last two RSI values to check for crossing
        last_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        
        # Generate signals
        if last_rsi < self.rsi_oversold and prev_rsi >= self.rsi_oversold:
            logger.info(f"RSI Strategy: BUY signal - RSI crossed below oversold level ({last_rsi:.2f})")
            return "BUY"
        elif last_rsi > self.rsi_overbought and prev_rsi <= self.rsi_overbought:
            logger.info(f"RSI Strategy: SELL signal - RSI crossed above overbought level ({last_rsi:.2f})")
            return "SELL"
        else:
            return None


class EMAStrategy(TradingStrategy):
    """EMA Crossover strategy"""
    def __init__(self):
        super().__init__('EMA_Cross')
        self.fast_ema = FAST_EMA
        self.slow_ema = SLOW_EMA
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate EMAs
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.fast_ema
        ).ema_indicator()
        
        df['slow_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.slow_ema
        ).ema_indicator()
        
        # Check for crossovers
        current_fast = df['fast_ema'].iloc[-1]
        current_slow = df['slow_ema'].iloc[-1]
        prev_fast = df['fast_ema'].iloc[-2]
        prev_slow = df['slow_ema'].iloc[-2]
        
        # Generate signals
        if current_fast > current_slow and prev_fast <= prev_slow:
            logger.info(f"EMA Strategy: BUY signal - Fast EMA crossed above Slow EMA")
            return "BUY"
        elif current_fast < current_slow and prev_fast >= prev_slow:
            logger.info(f"EMA Strategy: SELL signal - Fast EMA crossed below Slow EMA")
            return "SELL"
        else:
            return None


class RSIEMAStrategy(TradingStrategy):
    """Combined RSI and EMA strategy"""
    def __init__(self):
        super().__init__('RSI_EMA')
        self.rsi_strategy = RSIStrategy()
        self.ema_strategy = EMAStrategy()
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=RSI_PERIOD
        ).rsi()
        
        # Calculate EMAs
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=FAST_EMA
        ).ema_indicator()
        
        df['slow_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=SLOW_EMA
        ).ema_indicator()
        
        # Check both indicators
        rsi_signal = self.rsi_strategy.get_signal(klines)
        ema_signal = self.ema_strategy.get_signal(klines)
        
        # Only return BUY if both strategies agree or if EMA gives buy and RSI is not selling
        if rsi_signal == "BUY" and (ema_signal == "BUY" or ema_signal is None):
            logger.info("RSI_EMA Strategy: Strong BUY signal - Both indicators align")
            return "BUY"
        # Only return SELL if both strategies agree or if EMA gives sell and RSI is not buying
        elif rsi_signal == "SELL" and (ema_signal == "SELL" or ema_signal is None):
            logger.info("RSI_EMA Strategy: Strong SELL signal - Both indicators align")
            return "SELL"
        else:
            return None


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands strategy"""
    def __init__(self):
        super().__init__('Bollinger_Bands')
        self.window = 20
        self.window_dev = 2.0
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(
            close=df["close"], 
            window=self.window, 
            window_dev=self.window_dev
        )
        
        df['bb_high'] = indicator_bb.bollinger_hband()
        df['bb_low'] = indicator_bb.bollinger_lband()
        df['bb_mid'] = indicator_bb.bollinger_mavg()
        
        # Get current price and bands
        current_price = df['close'].iloc[-1]
        current_bb_low = df['bb_low'].iloc[-1]
        current_bb_high = df['bb_high'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_bb_low = df['bb_low'].iloc[-2]
        prev_bb_high = df['bb_high'].iloc[-2]
        
        # Generate signals
        if current_price < current_bb_low and prev_price >= prev_bb_low:
            logger.info(f"BB Strategy: BUY signal - Price crossed below lower band")
            return "BUY"
        elif current_price > current_bb_high and prev_price <= prev_bb_high:
            logger.info(f"BB Strategy: SELL signal - Price crossed above upper band")
            return "SELL"
        else:
            return None


def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    strategies = {
        'RSI': RSIStrategy(),
        'EMA_Cross': EMAStrategy(),
        'RSI_EMA': RSIEMAStrategy(),
        'Bollinger_Bands': BollingerBandsStrategy(),
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    else:
        logger.warning(f"Strategy {strategy_name} not found. Using default RSI_EMA strategy.")
        return strategies['RSI_EMA']