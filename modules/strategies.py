import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.risk_manager = None
        
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
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for the strategy"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for {self.strategy_name} strategy")
    
    def get_signal(self, klines):
        """
        Should be implemented by subclasses.
        Returns 'BUY', 'SELL', or None.
        """
        raise NotImplementedError("Each strategy must implement get_signal method")


class XRPDynamicGridStrategy(TradingStrategy):
    """
    Dynamic XRP Grid Trading Strategy that adapts to market trends
    and different market conditions (bullish, bearish, and sideways).
    """
    def __init__(self, 
                 grid_levels=5, 
                 grid_spacing_pct=1.0,
                 trend_ema_fast=8,
                 trend_ema_slow=21,
                 volatility_lookback=20,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30,
                 volume_ma_period=20,
                 adx_period=14,
                 adx_threshold=25,
                 sideways_threshold=15):
        
        super().__init__('XRPDynamicGrid')
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.volatility_lookback = volatility_lookback
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sideways_threshold = sideways_threshold
        self.grids = None
        self.current_trend = None
        self.current_market_condition = None
        self.last_grid_update = None
        
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame"""
        # Trend indicators
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_slow)
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 'UPTREND', 'DOWNTREND')
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                    df['close'], 
                                                    window=self.volatility_lookback)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # ADX for trend strength
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 
                                             window=self.adx_period)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        # Volume indicators
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], 
                                                window=self.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df['close'], 
                                                   window=20, 
                                                   window_dev=2)
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_lower'] = indicator_bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # MACD for additional trend confirmation
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Market condition classification
        df['market_condition'] = self.classify_market_condition(df)
        
        return df
    
    def classify_market_condition(self, df):
        """
        Classify market condition as BULLISH, BEARISH, or SIDEWAYS
        based on indicators like ADX, RSI, and price action
        """
        conditions = []
        
        for i in range(len(df)):
            if i < self.adx_period:
                conditions.append('SIDEWAYS')  # Default for initial rows
                continue
                
            adx = df['adx'].iloc[i]
            di_plus = df['di_plus'].iloc[i]
            di_minus = df['di_minus'].iloc[i]
            rsi = df['rsi'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            
            # Strong trend with ADX above threshold
            if adx > self.adx_threshold:
                if di_plus > di_minus:
                    if rsi > 50:  # Confirming bullish with RSI
                        conditions.append('BULLISH')
                    else:
                        conditions.append('SIDEWAYS')  # Conflicting signals
                else:
                    if rsi < 50:  # Confirming bearish with RSI
                        conditions.append('BEARISH')
                    else:
                        conditions.append('SIDEWAYS')  # Conflicting signals
            # Weak trend or consolidation
            elif adx < self.sideways_threshold:
                conditions.append('SIDEWAYS')
            # Moderate trend strength
            else:
                # Check if price is in a clear pattern
                if di_plus > di_minus and rsi > 50:
                    conditions.append('BULLISH')
                elif di_minus > di_plus and rsi < 50:
                    conditions.append('BEARISH')
                else:
                    conditions.append('SIDEWAYS')
        
        return pd.Series(conditions, index=df.index)
    
    def calculate_grid_spacing(self, df):
        """Dynamically calculate grid spacing based on volatility and market condition"""
        # Get the latest row
        latest = df.iloc[-1]
        
        # Base grid spacing on ATR percentage
        base_spacing = latest['atr_pct']
        
        # Adjust based on Bollinger Band width
        bb_multiplier = min(max(latest['bb_width'] * 5, 0.5), 3.0)
        
        # Adjust based on market condition
        market_condition = latest['market_condition']
        if market_condition == 'SIDEWAYS':
            # Tighter grid spacing in sideways markets
            condition_multiplier = 0.8
        elif market_condition == 'BULLISH' or market_condition == 'BEARISH':
            # Wider grid spacing in trending markets
            condition_multiplier = 1.2
        else:
            condition_multiplier = 1.0
        
        # Calculate final grid spacing
        dynamic_spacing = base_spacing * bb_multiplier * condition_multiplier
        
        # Ensure minimum and maximum spacing
        return min(max(dynamic_spacing, 0.5), 3.0)
    
    def generate_grids(self, df):
        """Generate dynamic grid levels based on current price and market conditions"""
        # Get latest price and indicators
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        market_condition = latest['market_condition']
        
        # Update risk manager with market condition if available
        if self.risk_manager and market_condition:
            self.risk_manager.set_market_condition(market_condition)
            logger.info(f"Updated risk manager with market condition: {market_condition}")
        
        # Determine grid bias based on market condition
        if market_condition == 'BULLISH':
            grid_bias = 0.7  # More levels above current price
        elif market_condition == 'BEARISH':
            grid_bias = 0.3  # More levels below current price
        else:  # SIDEWAYS
            grid_bias = 0.5  # Equal distribution
        
        # Calculate dynamic grid spacing
        dynamic_spacing = self.calculate_grid_spacing(df)
        
        # Generate grid levels
        grid_levels = []
        
        # Calculate number of levels above and below current price
        levels_above = int(self.grid_levels * grid_bias)
        levels_below = self.grid_levels - levels_above
        
        # Generate grid levels below current price
        for i in range(1, levels_below + 1):
            grid_price = current_price * (1 - (dynamic_spacing / 100) * i)
            grid_levels.append({
                'price': grid_price,
                'type': 'BUY',
                'status': 'ACTIVE'
            })
        
        # Generate grid levels above current price
        for i in range(1, levels_above + 1):
            grid_price = current_price * (1 + (dynamic_spacing / 100) * i)
            grid_levels.append({
                'price': grid_price,
                'type': 'SELL',
                'status': 'ACTIVE'
            })
        
        # Sort grid levels by price
        grid_levels.sort(key=lambda x: x['price'])
        
        return grid_levels
    
    def should_update_grids(self, df):
        """Determine if grids should be updated based on market conditions"""
        if self.grids is None or len(self.grids) == 0:
            return True
            
        latest = df.iloc[-1]
        current_trend = latest['trend']
        current_market_condition = latest['market_condition']
        
        # Update risk manager with new market condition if it changed
        if self.risk_manager and self.current_market_condition != current_market_condition:
            self.risk_manager.set_market_condition(current_market_condition)
            logger.info(f"Updated risk manager with market condition: {current_market_condition}")
        
        # Update grids if trend or market condition changed
        if (self.current_trend != current_trend or 
            self.current_market_condition != current_market_condition):
            logger.info(f"Market conditions changed. Trend: {self.current_trend}->{current_trend}, "
                       f"Condition: {self.current_market_condition}->{current_market_condition}. "
                       f"Updating grids.")
            return True
            
        # Check if price moved significantly outside grid range
        current_price = latest['close']
        min_grid = min(grid['price'] for grid in self.grids)
        max_grid = max(grid['price'] for grid in self.grids)
        
        # If price is outside grid range by more than 2%, update grids
        if current_price < min_grid * 0.98 or current_price > max_grid * 1.02:
            logger.info(f"Price moved outside grid range. Updating grids.")
            return True
            
        return False
    
    def get_grid_signal(self, df):
        """Get trading signal based on grid levels and current price"""
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # If no grids, generate them first
        if self.grids is None or len(self.grids) == 0 or self.should_update_grids(df):
            self.grids = self.generate_grids(df)
            self.current_trend = latest['trend']
            self.current_market_condition = latest['market_condition']
            self.last_grid_update = latest['open_time']
            logger.info(f"Generated new grids for {self.current_market_condition} market condition")
            return None  # No signal on grid initialization
        
        # Find the nearest grid levels
        buy_grids = [grid for grid in self.grids if grid['type'] == 'BUY' and grid['status'] == 'ACTIVE']
        sell_grids = [grid for grid in self.grids if grid['type'] == 'SELL' and grid['status'] == 'ACTIVE']
        
        # Find closest buy and sell grids
        closest_buy = None
        closest_sell = None
        
        if buy_grids:
            closest_buy = max(buy_grids, key=lambda x: x['price'])
            
        if sell_grids:
            closest_sell = min(sell_grids, key=lambda x: x['price'])
        
        # Determine signal based on price position relative to grids
        if closest_buy and current_price <= closest_buy['price'] * 1.001:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_buy['price']:
                    grid['status'] = 'TRIGGERED'
            return 'BUY'
            
        elif closest_sell and current_price >= closest_sell['price'] * 0.999:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_sell['price']:
                    grid['status'] = 'TRIGGERED'
            return 'SELL'
            
        return None
    
    def get_sideways_signal(self, df):
        """Get signal optimized for sideways market conditions"""
        latest = df.iloc[-1]
        
        # In sideways markets, look for overbought/oversold conditions
        # and mean reversion opportunities
        
        # Buy near lower Bollinger Band
        if latest['close'] < latest['bb_lower'] * 1.01:
            return 'BUY'
            
        # Sell near upper Bollinger Band
        elif latest['close'] > latest['bb_upper'] * 0.99:
            return 'SELL'
            
        # RSI overbought/oversold in sideways market
        elif latest['rsi'] < 30:
            return 'BUY'
        elif latest['rsi'] > 70:
            return 'SELL'
            
        return None
    
    def get_bullish_signal(self, df):
        """Get signal optimized for bullish market conditions"""
        if len(df) < 3:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # In bullish markets, look for dips to buy and continuation patterns
        
        # Buy on RSI oversold conditions (buy the dip)
        if latest['rsi'] < 40:
            return 'BUY'
            
        # Buy on MACD crossover to the upside
        if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            return 'BUY'
            
        # Sell on extreme overbought conditions
        if latest['rsi'] > 80 and latest['close'] > latest['bb_upper']:
            return 'SELL'
            
        return None
    
    def get_bearish_signal(self, df):
        """Get signal optimized for bearish market conditions"""
        if len(df) < 3:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # In bearish markets, look for rallies to sell and continuation patterns
        
        # Sell on RSI overbought conditions (sell the rally)
        if latest['rsi'] > 60:
            return 'SELL'
            
        # Sell on MACD crossover to the downside
        if prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            return 'SELL'
            
        # Buy on extreme oversold conditions
        if latest['rsi'] < 20 and latest['close'] < latest['bb_lower']:
            return 'BUY'
            
        return None
    
    def get_signal(self, klines):
        """
        Get trading signal combining multiple signals based on market condition:
        - BULLISH market: Focus on buying dips and trend continuation
        - BEARISH market: Focus on selling rallies and trend continuation
        - SIDEWAYS market: Focus on range trading and mean reversion
        
        Returns 'BUY', 'SELL', or None
        """
        # Prepare and add indicators to the data
        df = self.prepare_data(klines)
        df = self.add_indicators(df)
        
        if len(df) < self.trend_ema_slow + 5:
            # Not enough data to generate reliable signals
            return None
        
        # Get latest market condition
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Update risk manager with current market condition
        if self.risk_manager:
            self.risk_manager.set_market_condition(market_condition)
        
        # Get grid signal (works in all market conditions)
        grid_signal = self.get_grid_signal(df)
        
        # Get condition-specific signals
        if market_condition == 'SIDEWAYS':
            condition_signal = self.get_sideways_signal(df)
            logger.debug(f"SIDEWAYS market detected. Grid signal: {grid_signal}, Sideways signal: {condition_signal}")
            
            # In sideways markets, prioritize mean reversion signals
            if condition_signal:
                return condition_signal
            elif grid_signal:
                return grid_signal
                
        elif market_condition == 'BULLISH':
            condition_signal = self.get_bullish_signal(df)
            logger.debug(f"BULLISH market detected. Grid signal: {grid_signal}, Bullish signal: {condition_signal}")
            
            # In bullish markets, prioritize buy signals
            if condition_signal == 'BUY' or grid_signal == 'BUY':
                return 'BUY'
            elif condition_signal == 'SELL':  # Only sell on strong sell signals in bullish market
                return 'SELL'
                
        elif market_condition == 'BEARISH':
            condition_signal = self.get_bearish_signal(df)
            logger.debug(f"BEARISH market detected. Grid signal: {grid_signal}, Bearish signal: {condition_signal}")
            
            # In bearish markets, prioritize sell signals
            if condition_signal == 'SELL' or grid_signal == 'SELL':
                return 'SELL'
            elif condition_signal == 'BUY':  # Only buy on strong buy signals in bearish market
                return 'BUY'
        
        # Default to grid signal if no condition-specific signal
        return grid_signal


def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    from modules.config import (
        XRP_GRID_LEVELS, XRP_GRID_SPACING_PCT, XRP_TREND_EMA_FAST, XRP_TREND_EMA_SLOW,
        XRP_VOLATILITY_LOOKBACK, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
        XRP_VOLUME_MA_PERIOD, XRP_ADX_PERIOD, XRP_ADX_THRESHOLD, XRP_SIDEWAYS_THRESHOLD
    )
    
    strategies = {
        'XRPDynamicGrid': XRPDynamicGridStrategy(
            grid_levels=XRP_GRID_LEVELS,
            grid_spacing_pct=XRP_GRID_SPACING_PCT,
            trend_ema_fast=XRP_TREND_EMA_FAST,
            trend_ema_slow=XRP_TREND_EMA_SLOW,
            volatility_lookback=XRP_VOLATILITY_LOOKBACK,
            rsi_period=RSI_PERIOD,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_oversold=RSI_OVERSOLD,
            volume_ma_period=XRP_VOLUME_MA_PERIOD,
            adx_period=XRP_ADX_PERIOD,
            adx_threshold=XRP_ADX_THRESHOLD,
            sideways_threshold=XRP_SIDEWAYS_THRESHOLD
        )
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to base trading strategy.")
    return TradingStrategy(strategy_name)


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default strategies based on symbol
    symbol_strategies = {
        'XRPUSDT': XRPDynamicGridStrategy()
    }
    
    if symbol in symbol_strategies:
        return symbol_strategies[symbol]
    
    # Default to base strategy
    return TradingStrategy(symbol)