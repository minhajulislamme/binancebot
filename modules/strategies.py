import logging
import numpy as np
import pandas as pd
import ta

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


class BTCScalpingStrategy(TradingStrategy):
    """EMA 9/21 + RSI Scalping strategy optimized for BTC"""
    def __init__(self):
        super().__init__('BTC_Scalping')
        self.fast_ema = 9
        self.slow_ema = 21
        self.rsi_period = 7  # Shorter period for scalping
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volume_window = 10
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate indicators
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.fast_ema
        ).ema_indicator()
        
        df['slow_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.slow_ema
        ).ema_indicator()

        # Volume trend
        df['volume_change'] = df['volume'].pct_change() * 100
        df['volume_ma'] = df['volume'].rolling(window=self.volume_window).mean()
        df['is_volume_spike'] = df['volume'] > (df['volume_ma'] * 1.2)
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1] 
        current_fast_ema = df['fast_ema'].iloc[-1]
        current_slow_ema = df['slow_ema'].iloc[-1]
        volume_spike = df['is_volume_spike'].iloc[-1]
        
        # Previous values
        prev_fast_ema = df['fast_ema'].iloc[-2] 
        prev_slow_ema = df['slow_ema'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        
        # Signal logic for BTC scalping
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # BUY conditions for BTC (optimized for 1m-5m timeframes)
        # Condition 1: EMA crossover with RSI confirming uptrend
        if (current_fast_ema > current_slow_ema and prev_fast_ema <= prev_slow_ema):
            if current_rsi > prev_rsi and current_rsi > 40:
                buy_signal = True
                reason = "EMA 9/21 bullish crossover with RSI momentum"
        
        # Condition 2: RSI oversold bounce
        elif (current_rsi > prev_rsi and prev_rsi < self.rsi_oversold and current_rsi < 45):
            buy_signal = True
            reason = "RSI oversold bounce"
        
        # SELL conditions for BTC
        # Condition 1: EMA bearish crossover
        if (current_fast_ema < current_slow_ema and prev_fast_ema >= prev_slow_ema):
            sell_signal = True
            reason = "EMA 9/21 bearish crossover"
        
        # Condition 2: RSI overbought exit
        elif (current_rsi < prev_rsi and current_rsi > self.rsi_overbought):
            sell_signal = True
            reason = "RSI overbought exit signal"
            
        # Generate signals
        if buy_signal and (volume_spike or len(df) > 100):
            logger.info(f"BTC Scalping: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"BTC Scalping: SELL signal - {reason}")
            return "SELL"
            
        return None


class ETHStochMACD(TradingStrategy):
    """MACD + Stochastic Reversal strategy optimized for ETH"""
    def __init__(self):
        super().__init__('ETH_StochMACD')
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_smooth = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate MACD for trend strength
        macd = ta.trend.MACD(
            close=df['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()  # Histogram
        
        # Calculate Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.stoch_k,
            smooth_window=self.stoch_smooth
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        current_macd_hist = df['macd_hist'].iloc[-1]
        current_stoch_k = df['stoch_k'].iloc[-1]
        current_stoch_d = df['stoch_d'].iloc[-1]
        
        # Previous values
        prev_macd = df['macd'].iloc[-2]
        prev_macd_signal = df['macd_signal'].iloc[-2]
        prev_macd_hist = df['macd_hist'].iloc[-2]
        prev_stoch_k = df['stoch_k'].iloc[-2]
        prev_stoch_d = df['stoch_d'].iloc[-2]
        
        # Signal logic for ETH reversal strategy
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # BUY conditions for ETH
        # Condition 1: MACD bullish crossover with stochastic confirming
        if (current_macd > current_macd_signal and prev_macd <= prev_macd_signal):
            if current_stoch_k < self.stoch_oversold and current_stoch_k > current_stoch_d:
                buy_signal = True
                reason = "MACD bullish crossover with stochastic confirmation"
        
        # Condition 2: Stochastic oversold reversal with positive MACD histogram
        elif (prev_stoch_k < self.stoch_oversold and current_stoch_k > current_stoch_d and
              current_stoch_k > prev_stoch_k and current_macd_hist > 0):
            buy_signal = True
            reason = "Stochastic oversold reversal with positive MACD"
        
        # SELL conditions for ETH
        # Condition 1: MACD bearish crossover with stochastic confirming
        if (current_macd < current_macd_signal and prev_macd >= prev_macd_signal):
            if current_stoch_k > self.stoch_overbought and current_stoch_k < current_stoch_d:
                sell_signal = True
                reason = "MACD bearish crossover with stochastic confirmation"
        
        # Condition 2: Stochastic overbought reversal with negative MACD histogram
        elif (prev_stoch_k > self.stoch_overbought and current_stoch_k < current_stoch_d and
              current_stoch_k < prev_stoch_k and current_macd_hist < 0):
            sell_signal = True
            reason = "Stochastic overbought reversal with negative MACD"
            
        # Generate signals
        if buy_signal:
            logger.info(f"ETH StochMACD: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"ETH StochMACD: SELL signal - {reason}")
            return "SELL"
            
        return None


class BNBGridStrategy(TradingStrategy):
    """Grid Trading strategy optimized for BNB sideways markets"""
    def __init__(self):
        super().__init__('BNB_Grid')
        self.bb_window = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_middle = 50
        self.grid_levels = 5  # Number of grid levels within BB range
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate Bollinger Bands for grid levels
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_window,
            window_dev=self.bb_std
        )
        df['bb_high'] = bb_indicator.bollinger_hband()
        df['bb_low'] = bb_indicator.bollinger_lband()
        df['bb_mid'] = bb_indicator.bollinger_mavg()
        
        # Calculate RSI to detect ranging market
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate volatility
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range()
        
        df['volatility'] = (df['atr'] / df['close']) * 100
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_bb_high = df['bb_high'].iloc[-1]
        current_bb_low = df['bb_low'].iloc[-1]
        current_bb_mid = df['bb_mid'].iloc[-1]
        current_volatility = df['volatility'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        
        # Calculate grid levels
        grid_range = current_bb_high - current_bb_low
        grid_step = grid_range / self.grid_levels
        
        # Signal logic for BNB grid trading - optimized for sideways markets
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # Check if we're in a sideways market (low volatility + RSI near middle)
        is_sideways = (abs(current_rsi - self.rsi_middle) < 15) and (current_volatility < 2.0)
        
        # BUY conditions for BNB
        # Condition 1: Price near lower grid level in sideways market
        if is_sideways and (current_price < (current_bb_low + grid_step)):
            buy_signal = True
            reason = "Grid buy signal at lower band"
        
        # Condition 2: Price drop with RSI indicating no strong downtrend
        elif (current_price < prev_price and 
              current_price < current_bb_mid and 
              current_rsi > 40 and 
              current_rsi > prev_rsi):
            buy_signal = True
            reason = "Support level buy in range-bound market"
        
        # SELL conditions for BNB
        # Condition 1: Price near upper grid level in sideways market
        if is_sideways and (current_price > (current_bb_high - grid_step)):
            sell_signal = True
            reason = "Grid sell signal at upper band"
        
        # Condition 2: Price rise with RSI indicating no strong uptrend
        elif (current_price > prev_price and 
              current_price > current_bb_mid and 
              current_rsi < 60 and 
              current_rsi < prev_rsi):
            sell_signal = True
            reason = "Resistance level sell in range-bound market"
            
        # Generate signals
        if buy_signal:
            logger.info(f"BNB Grid: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"BNB Grid: SELL signal - {reason}")
            return "SELL"
            
        return None


class SOLSqueezeStrategy(TradingStrategy):
    """Bollinger Squeeze strategy optimized for SOL's explosive breakouts"""
    def __init__(self):
        super().__init__('SOL_Squeeze')
        self.bb_window = 20
        self.bb_std = 2.0
        self.kc_window = 20
        self.kc_mult = 1.5
        self.atr_period = 14
        self.rsi_period = 14
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_window,
            window_dev=self.bb_std
        )
        df['bb_high'] = bb_indicator.bollinger_hband()
        df['bb_low'] = bb_indicator.bollinger_lband()
        df['bb_mid'] = bb_indicator.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Calculate ATR for Keltner Channels
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        ).average_true_range()
        
        # Calculate Keltner Channels
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['kc_mid'] = typical_price.rolling(window=self.kc_window).mean()
        df['kc_high'] = df['kc_mid'] + self.kc_mult * df['atr']
        df['kc_low'] = df['kc_mid'] - self.kc_mult * df['atr']
        
        # Calculate RSI for trend direction
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['is_volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(5) * 100
        
        # Bollinger Squeeze occurs when Bollinger Bands are inside Keltner Channels
        df['squeeze_on'] = (df['bb_high'] < df['kc_high']) & (df['bb_low'] > df['kc_low'])
        df['squeeze_off'] = ~df['squeeze_on']
        
        # Detect squeeze release
        df['squeeze_release'] = df['squeeze_off'] & df['squeeze_on'].shift(1)
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_momentum = df['momentum'].iloc[-1]
        current_squeeze_on = df['squeeze_on'].iloc[-1]
        current_squeeze_release = df['squeeze_release'].iloc[-1]
        volume_spike = df['is_volume_spike'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        prev_momentum = df['momentum'].iloc[-2] if len(df) > 2 else 0
        
        # Signal logic for SOL squeeze strategy
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # BUY conditions for SOL
        # Condition 1: Squeeze release with bullish momentum
        if current_squeeze_release and current_price > prev_price and current_momentum > 0:
            buy_signal = True
            reason = "Bullish squeeze release"
        
        # Condition 2: Squeeze building with early momentum
        elif (current_squeeze_on and 
              current_momentum > prev_momentum and 
              current_rsi > prev_rsi and 
              current_rsi > 50):
            buy_signal = True
            reason = "Early squeeze momentum building"
        
        # SELL conditions for SOL
        # Condition 1: Squeeze release with bearish momentum
        if current_squeeze_release and current_price < prev_price and current_momentum < 0:
            sell_signal = True
            reason = "Bearish squeeze release"
        
        # Condition 2: RSI overbought after squeeze
        elif not current_squeeze_on and current_rsi > 70 and current_rsi < prev_rsi:
            sell_signal = True
            reason = "Overbought after squeeze release"
            
        # Generate signals
        if buy_signal and (volume_spike or len(df) > 100):
            logger.info(f"SOL Squeeze: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"SOL Squeeze: SELL signal - {reason}")
            return "SELL"
            
        return None


class ADAEMATrendStrategy(TradingStrategy):
    """RSI + EMA Trend Riding strategy optimized for ADA"""
    def __init__(self):
        super().__init__('ADA_EMATrend')
        self.fast_ema = 8
        self.medium_ema = 21
        self.slow_ema = 55
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate EMAs
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.fast_ema
        ).ema_indicator()
        
        df['medium_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.medium_ema
        ).ema_indicator()
        
        df['slow_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.slow_ema
        ).ema_indicator()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_fast_ema = df['fast_ema'].iloc[-1]
        current_medium_ema = df['medium_ema'].iloc[-1]
        current_slow_ema = df['slow_ema'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_fast_ema = df['fast_ema'].iloc[-2]
        prev_medium_ema = df['medium_ema'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        
        # Signal logic for ADA trend riding
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # Check for EMA alignment (trend strength)
        bullish_alignment = current_fast_ema > current_medium_ema > current_slow_ema
        bearish_alignment = current_fast_ema < current_medium_ema < current_slow_ema
        
        # BUY conditions for ADA
        # Condition 1: RSI crosses above 30 with bullish EMA alignment
        if current_rsi > 30 and prev_rsi <= 30 and bullish_alignment:
            buy_signal = True
            reason = "RSI bounce from oversold with bullish EMA alignment"
        
        # Condition 2: Fast EMA crosses above medium EMA with rising RSI
        elif (current_fast_ema > current_medium_ema and prev_fast_ema <= prev_medium_ema and
              current_rsi > prev_rsi and current_rsi > 40):
            buy_signal = True
            reason = "Fast EMA crosses above medium with rising RSI"
        
        # Condition 3: Price pullback to fast EMA in uptrend
        elif (bullish_alignment and
              abs(current_price - current_fast_ema) / current_price < 0.005 and  # Price near fast EMA
              current_price > prev_price and current_rsi > 45):
            buy_signal = True
            reason = "Price pullback to fast EMA in uptrend"
        
        # SELL conditions for ADA
        # Condition 1: RSI crosses below 70 after being overbought
        if current_rsi < 70 and prev_rsi >= 70:
            sell_signal = True
            reason = "RSI crosses down from overbought"
        
        # Condition 2: Fast EMA crosses below medium EMA with falling RSI
        elif (current_fast_ema < current_medium_ema and prev_fast_ema >= prev_medium_ema and
              current_rsi < prev_rsi and current_rsi < 60):
            sell_signal = True
            reason = "Fast EMA crosses below medium with falling RSI"
        
        # Condition 3: Price in bearish alignment with strong downward momentum
        elif bearish_alignment and current_rsi < 40 and current_rsi < prev_rsi:
            sell_signal = True
            reason = "Strong bearish alignment with momentum"
            
        # Generate signals
        if buy_signal:
            logger.info(f"ADA EMATrend: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"ADA EMATrend: SELL signal - {reason}")
            return "SELL"
            
        return None


class XRPScalpingStrategy(TradingStrategy):
    """Aggressive Scalping strategy optimized for XRP's fast movements"""
    def __init__(self):
        super().__init__('XRP_Scalping')
        self.fast_ema = 5
        self.medium_ema = 13
        self.slow_ema = 21
        self.rsi_period = 6  # Very short for fast response
        self.rsi_overbought = 75
        self.rsi_oversold = 25
        self.bb_window = 15
        self.bb_std = 2.5
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate EMAs for multiple confirmations
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.fast_ema
        ).ema_indicator()
        
        df['medium_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.medium_ema
        ).ema_indicator()
        
        df['slow_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.slow_ema
        ).ema_indicator()
        
        # Calculate RSI with very short period
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate Bollinger Bands 
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_window,
            window_dev=self.bb_std
        )
        df['bb_high'] = bb_indicator.bollinger_hband()
        df['bb_low'] = bb_indicator.bollinger_lband()
        
        # Calculate momentum
        df['price_change'] = df['close'].pct_change(3) * 100
        df['volume_change'] = df['volume'].pct_change(3) * 100
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_fast_ema = df['fast_ema'].iloc[-1]
        current_medium_ema = df['medium_ema'].iloc[-1]
        current_slow_ema = df['slow_ema'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_bb_high = df['bb_high'].iloc[-1]
        current_bb_low = df['bb_low'].iloc[-1]
        current_price_change = df['price_change'].iloc[-1]
        current_volume_change = df['volume_change'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_fast_ema = df['fast_ema'].iloc[-2]
        prev_medium_ema = df['medium_ema'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        
        # Signal logic for XRP aggressive scalping
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # XRP often moves quickly, so we want to catch early moves
        
        # BUY conditions for XRP
        # Condition 1: Quick RSI reversal from oversold
        if current_rsi > prev_rsi + 5 and prev_rsi < self.rsi_oversold:
            buy_signal = True
            reason = "Quick RSI reversal from oversold"
        
        # Condition 2: Fast EMA crossing above medium with strong momentum
        elif (current_fast_ema > current_medium_ema and prev_fast_ema <= prev_medium_ema and
              current_price_change > 1.0 and current_volume_change > 15):
            buy_signal = True
            reason = "Fast EMA crosses with strong momentum"
        
        # Condition 3: Price near lower BB with positive momentum shift
        elif (current_price < current_bb_low * 1.02 and  # Price near or below lower BB
              current_price > prev_price and current_rsi > prev_rsi):
            buy_signal = True
            reason = "Price bounce from lower BB"
        
        # SELL conditions for XRP
        # Condition 1: Quick RSI reversal from overbought
        if current_rsi < prev_rsi - 5 and prev_rsi > self.rsi_overbought:
            sell_signal = True
            reason = "Quick RSI reversal from overbought"
        
        # Condition 2: Fast EMA crossing below medium with momentum
        elif (current_fast_ema < current_medium_ema and prev_fast_ema >= prev_medium_ema and
              current_price_change < -1.0):
            sell_signal = True
            reason = "Fast EMA crosses below medium with momentum"
        
        # Condition 3: Price near upper BB with momentum shift
        elif (current_price > current_bb_high * 0.98 and  # Price near or above upper BB
              current_price < prev_price and current_rsi < prev_rsi):
            sell_signal = True
            reason = "Price rejection from upper BB"
            
        # Generate signals - for XRP we want to be more aggressive with entries
        if buy_signal:
            logger.info(f"XRP Scalping: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"XRP Scalping: SELL signal - {reason}")
            return "SELL"
            
        return None


class DOGEScalpingStrategy(TradingStrategy):
    """VWAP + EMA 5 Scalping strategy optimized for DOGE's meme volatility"""
    def __init__(self):
        super().__init__('DOGE_Scalping')
        self.fast_ema = 5
        self.medium_ema = 10
        self.rsi_period = 8
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.vwap_window = 14
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate EMAs
        df['fast_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.fast_ema
        ).ema_indicator()
        
        df['medium_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.medium_ema
        ).ema_indicator()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate VWAP (approximation since we don't have intraday data)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Create cumulative sums for the specified window
        df['cum_tp_volume'] = df['tp_volume'].rolling(window=self.vwap_window).sum()
        df['cum_volume'] = df['volume'].rolling(window=self.vwap_window).sum()
        
        # Calculate VWAP
        df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
        
        # Calculate volume and price change
        df['volume_change'] = df['volume'].pct_change(3) * 100
        df['price_change_1'] = df['close'].pct_change(1) * 100
        df['price_change_3'] = df['close'].pct_change(3) * 100
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_fast_ema = df['fast_ema'].iloc[-1]
        current_medium_ema = df['medium_ema'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        current_volume_change = df['volume_change'].iloc[-1]
        current_price_change_1 = df['price_change_1'].iloc[-1]
        current_price_change_3 = df['price_change_3'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_fast_ema = df['fast_ema'].iloc[-2]
        prev_medium_ema = df['medium_ema'].iloc[-2]
        prev_rsi = df['rsi'].iloc[-2]
        prev_vwap = df['vwap'].iloc[-2]
        
        # Signal logic for DOGE meme volatility scalping
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # BUY conditions for DOGE
        # Condition 1: Price crosses above VWAP with volume spike
        if current_price > current_vwap and prev_price <= prev_vwap and current_volume_change > 20:
            buy_signal = True
            reason = "Price crossed above VWAP with volume spike"
        
        # Condition 2: Fast EMA crossing above medium with momentum
        elif (current_fast_ema > current_medium_ema and prev_fast_ema <= prev_medium_ema and
              current_price_change_1 > 1.0):
            buy_signal = True
            reason = "Fast EMA 5 cross with strong momentum"
        
        # Condition 3: Price bounces off EMA 5 in uptrend
        elif (current_price > current_fast_ema and
              abs(prev_price - prev_fast_ema)/prev_price < 0.005 and  # Previous bar touched EMA
              current_price > prev_price and current_rsi > 50):
            buy_signal = True
            reason = "Price bounce from EMA 5 support"
        
        # SELL conditions for DOGE
        # Condition 1: Price crosses below VWAP with momentum
        if current_price < current_vwap and prev_price >= prev_vwap and current_price_change_1 < -1.0:
            sell_signal = True
            reason = "Price crossed below VWAP with momentum"
        
        # Condition 2: Fast EMA crossing below medium
        elif (current_fast_ema < current_medium_ema and prev_fast_ema >= prev_medium_ema):
            sell_signal = True
            reason = "Fast EMA crossed below medium EMA"
        
        # Condition 3: RSI overbought with price rejection
        elif current_rsi > self.rsi_overbought and current_price < prev_price:
            sell_signal = True
            reason = "RSI overbought with price rejection"
            
        # Generate signals
        if buy_signal:
            logger.info(f"DOGE Scalping: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"DOGE Scalping: SELL signal - {reason}")
            return "SELL"
            
        return None


class SHIBBreakoutStrategy(TradingStrategy):
    """Breakout Trading strategy optimized for SHIB's extreme moves"""
    def __init__(self):
        super().__init__('SHIB_Breakout')
        self.atr_period = 14
        self.rsi_period = 14
        self.lookback_period = 10
        self.volume_multiplier = 2.0
        
    def get_signal(self, klines):
        df = self.prepare_data(klines)
        
        # Calculate ATR for volatility
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.atr_period
        ).average_true_range()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.rsi_period
        ).rsi()
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate price ranges for breakout detection
        df['highest_high'] = df['high'].rolling(window=self.lookback_period).max()
        df['lowest_low'] = df['low'].rolling(window=self.lookback_period).min()
        
        # Calculate momentum and volatility features
        df['price_change'] = df['close'].pct_change() * 100
        df['volatility'] = (df['atr'] / df['close']) * 100
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_highest_high = df['highest_high'].iloc[-1]
        current_lowest_low = df['lowest_low'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        current_volatility = df['volatility'].iloc[-1]
        
        # Previous values
        prev_price = df['close'].iloc[-2]
        prev_high = df['high'].iloc[-2]
        prev_highest_high = df['highest_high'].iloc[-2]
        prev_lowest_low = df['lowest_low'].iloc[-2]
        
        # Signal logic for SHIB breakout trading
        buy_signal = False
        sell_signal = False
        reason = ""
        
        # BUY conditions for SHIB
        # Condition 1: Price breaks above previous high with volume
        if (current_high > prev_highest_high and 
            current_price > prev_price and 
            current_volume_ratio > self.volume_multiplier):
            buy_signal = True
            reason = "Bullish breakout with high volume"
        
        # Condition 2: Strong upward momentum on high volume
        elif (current_price > prev_price * 1.03 and  # 3% price jump
              current_volume_ratio > self.volume_multiplier):
            buy_signal = True
            reason = "Strong momentum breakout"
        
        # SELL conditions for SHIB
        # Condition 1: Price breaks below previous low with volume
        if (current_low < prev_lowest_low and 
            current_price < prev_price and 
            current_volume_ratio > self.volume_multiplier):
            sell_signal = True
            reason = "Bearish breakdown with high volume"
        
        # Condition 2: Strong downward momentum on high volume
        elif (current_price < prev_price * 0.97 and  # 3% price drop
              current_volume_ratio > self.volume_multiplier):
            sell_signal = True
            reason = "Strong momentum breakdown"
            
        # Generate signals
        if buy_signal:
            logger.info(f"SHIB Breakout: BUY signal - {reason}")
            return "BUY"
        elif sell_signal:
            logger.info(f"SHIB Breakout: SELL signal - {reason}")
            return "SELL"
            
        return None


def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    strategies = {
        'BTC_Scalping': BTCScalpingStrategy(),
        'ETH_StochMACD': ETHStochMACD(),
        'BNB_Grid': BNBGridStrategy(),
        'SOL_Squeeze': SOLSqueezeStrategy(),
        'ADA_EMATrend': ADAEMATrendStrategy(),
        'XRP_Scalping': XRPScalpingStrategy(),
        'DOGE_Scalping': DOGEScalpingStrategy(),
        'SHIB_Breakout': SHIBBreakoutStrategy(),
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    else:
        logger.warning(f"Strategy {strategy_name} not found. Using default ETH_StochMACD strategy.")
        return strategies['ETH_StochMACD']


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Otherwise, map symbols to optimized strategies
    symbol = symbol.upper()
    if 'BTCUSDT' in symbol:
        return BTCScalpingStrategy()
    elif 'ETHUSDT' in symbol:
        return ETHStochMACD()
    elif 'BNBUSDT' in symbol:
        return BNBGridStrategy()
    elif 'SOLUSDT' in symbol:
        return SOLSqueezeStrategy()
    elif 'ADAUSDT' in symbol:
        return ADAEMATrendStrategy()
    elif 'XRPUSDT' in symbol:
        return XRPScalpingStrategy()
    elif 'DOGEUSDT' in symbol:
        return DOGEScalpingStrategy()
    elif 'SHIBUSDT' in symbol:
        return SHIBBreakoutStrategy()
    else:
        # Default to ETH_StochMACD for other tokens as a reasonable choice
        return ETHStochMACD()