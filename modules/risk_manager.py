import logging
import math
from modules.config import (
    INITIAL_BALANCE, RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, USE_TAKE_PROFIT, 
    TAKE_PROFIT_PCT, TRAILING_TAKE_PROFIT, TRAILING_TAKE_PROFIT_PCT, TRAILING_STOP, TRAILING_STOP_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT,
    # Adaptive risk management settings
    STOP_LOSS_PCT_BULLISH, STOP_LOSS_PCT_BEARISH, STOP_LOSS_PCT_SIDEWAYS,
    TAKE_PROFIT_PCT_BULLISH, TAKE_PROFIT_PCT_BEARISH, TAKE_PROFIT_PCT_SIDEWAYS,
    TRAILING_STOP_PCT_BULLISH, TRAILING_STOP_PCT_BEARISH, TRAILING_STOP_PCT_SIDEWAYS,
    TRAILING_TAKE_PROFIT_PCT_BULLISH, TRAILING_TAKE_PROFIT_PCT_BEARISH, TRAILING_TAKE_PROFIT_PCT_SIDEWAYS
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, binance_client):
        """Initialize risk manager with a reference to binance client"""
        self.binance_client = binance_client
        self.initial_balance = None
        self.last_known_balance = None
        self.current_market_condition = None  # Will be set to 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        
    def set_market_condition(self, market_condition):
        """Set the current market condition for adaptive risk management"""
        if market_condition in ['BULLISH', 'BEARISH', 'SIDEWAYS']:
            if self.current_market_condition != market_condition:
                logger.info(f"Market condition changed to {market_condition}")
                self.current_market_condition = market_condition
        else:
            logger.warning(f"Invalid market condition: {market_condition}. Using default risk parameters.")
        
    def calculate_position_size(self, symbol, side, price, stop_loss_price=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            price: Current market price
            stop_loss_price: Optional stop loss price for calculating risk
            
        Returns:
            quantity: The position size
        """
        # Get account balance
        balance = self.binance_client.get_account_balance()
        
        # Initialize initial balance if not set
        if self.initial_balance is None:
            self.initial_balance = balance
            self.last_known_balance = balance
            
        # Auto compound logic
        if AUTO_COMPOUND and self.last_known_balance is not None:
            profit = balance - self.last_known_balance
            if profit > 0:
                # We've made profit, apply compounding by increasing risk amount
                logger.info(f"Auto-compounding profit of {profit:.2f} USDT")
                # Update the last known balance
                self.last_known_balance = balance
            
        if balance <= 0:
            logger.error("Insufficient balance to open a position")
            return 0
            
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {symbol}")
            return 0
            
        # Calculate risk amount
        risk_amount = balance * RISK_PER_TRADE
        
        # Calculate position size based on risk and stop loss
        if stop_loss_price and USE_STOP_LOSS:
            # If stop loss is provided, calculate size based on it
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error("Stop loss too close to entry price")
                return 0
                
            # Calculate max quantity based on risk
            max_quantity = risk_amount / risk_per_unit
        else:
            # If no stop loss, use a percentage of balance with leverage
            leverage = self.get_current_leverage(symbol)
            max_quantity = (balance * RISK_PER_TRADE * leverage) / price
        
        # Apply precision to quantity
        quantity_precision = symbol_info['quantity_precision']
        quantity = round_step_size(max_quantity, get_step_size(symbol_info['min_qty']))
        
        # Check minimum notional
        min_notional = symbol_info['min_notional']
        if quantity * price < min_notional:
            logger.warning(f"Position size too small - below minimum notional of {min_notional}")
            if min_notional / price <= max_quantity:
                quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
                logger.info(f"Adjusted position size to meet minimum notional: {quantity}")
            else:
                logger.error(f"Cannot meet minimum notional with current risk settings")
                return 0
                
        logger.info(f"Calculated position size: {quantity} units at {price} per unit")
        return quantity
        
    def get_current_leverage(self, symbol):
        """Get the current leverage for a symbol"""
        position_info = self.binance_client.get_position_info(symbol)
        if position_info:
            return position_info['leverage']
        return 1  # Default to 1x if no position info
        
    def should_open_position(self, symbol):
        """Check if a new position should be opened based on risk rules"""
        # Check if we already have an open position
        position_info = self.binance_client.get_position_info(symbol)
        if position_info and abs(position_info['position_amount']) > 0:
            logger.info(f"Already have an open position for {symbol}")
            return False
            
        # Check maximum number of open positions
        positions = self.binance_client.client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            logger.info(f"Maximum number of open positions ({MAX_OPEN_POSITIONS}) reached")
            return False
            
        return True
        
    def calculate_stop_loss(self, symbol, side, entry_price):
        """Calculate stop loss price based on configuration and market condition"""
        if not USE_STOP_LOSS:
            return None
            
        # Choose stop loss percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            stop_loss_pct = STOP_LOSS_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            stop_loss_pct = STOP_LOSS_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            stop_loss_pct = STOP_LOSS_PCT_SIDEWAYS
        else:
            stop_loss_pct = STOP_LOSS_PCT  # Default
            
        if side == "BUY":  # Long position
            stop_price = entry_price * (1 - stop_loss_pct)
        else:  # Short position
            stop_price = entry_price * (1 + stop_loss_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_price = round(stop_price, price_precision)
            
        logger.info(f"Calculated {self.current_market_condition} stop loss at {stop_price} ({stop_loss_pct*100}%)")
        return stop_price
        
    def calculate_take_profit(self, symbol, side, entry_price):
        """Calculate take profit price based on configuration and market condition"""
        if not USE_TAKE_PROFIT:
            return None
            
        # Choose take profit percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            take_profit_pct = TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            take_profit_pct = TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            take_profit_pct = TAKE_PROFIT_PCT_SIDEWAYS
        else:
            take_profit_pct = TAKE_PROFIT_PCT  # Default
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # Short position
            take_profit_price = entry_price * (1 - take_profit_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        logger.info(f"Calculated {self.current_market_condition} take profit at {take_profit_price} ({take_profit_pct*100}%)")
        return take_profit_price
        
    def adjust_stop_loss_for_trailing(self, symbol, side, current_price, position_info=None):
        """Adjust stop loss for trailing stop if needed"""
        if not TRAILING_STOP:
            return None
            
        if not position_info:
            position_info = self.binance_client.get_position_info(symbol)
            
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        entry_price = position_info['entry_price']
        
        # Choose trailing stop percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            trailing_stop_pct = TRAILING_STOP_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            trailing_stop_pct = TRAILING_STOP_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_stop_pct = TRAILING_STOP_PCT_SIDEWAYS
        else:
            trailing_stop_pct = TRAILING_STOP_PCT  # Default
        
        # Calculate new stop loss based on current price
        if side == "BUY":  # Long position
            new_stop = current_price * (1 - trailing_stop_pct)
            # Only move stop loss up, never down
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop <= current_stop:
                return None
        else:  # Short position
            new_stop = current_price * (1 + trailing_stop_pct)
            # Only move stop loss down, never up
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop >= current_stop:
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop = round(new_stop, price_precision)
            
        logger.info(f"Adjusted {self.current_market_condition} trailing stop loss to {new_stop} ({trailing_stop_pct*100}%)")
        return new_stop
        
    def adjust_take_profit_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust take profit price based on trailing settings
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('BUY' or 'SELL')
            current_price: Current market price
            position_info: Position information including entry_price
            
        Returns:
            new_take_profit: New take profit price if it should be adjusted, None otherwise
        """
        if not USE_TAKE_PROFIT or not TRAILING_TAKE_PROFIT:
            return None
            
        if not position_info:
            return None
            
        entry_price = float(position_info.get('entry_price', 0))
        if entry_price <= 0:
            return None
        
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            return None
            
        price_precision = symbol_info.get('price_precision', 2)
        
        # Choose trailing take profit percentage based on market condition
        if self.current_market_condition == 'BULLISH':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition == 'BEARISH':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_SIDEWAYS
        else:
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT  # Default
        
        # Calculate the current dynamic take profit level based on the current price
        if side == 'BUY':  # Long position
            # For long positions, we want take profit to trail above the price
            current_take_profit = current_price * (1 + trailing_take_profit_pct)
            current_take_profit = math.floor(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists
            existing_take_profit = None
            for order in open_orders:
                if order['type'] == 'TAKE_PROFIT_MARKET' and order['side'] == 'SELL':
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit or our new one is better (higher for long), return the new one
            if not existing_take_profit or current_take_profit > existing_take_profit:
                logger.info(f"Long position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                return current_take_profit
                
        elif side == 'SELL':  # Short position
            # For short positions, we want take profit to trail below the price
            current_take_profit = current_price * (1 - trailing_take_profit_pct)
            current_take_profit = math.ceil(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists
            existing_take_profit = None
            for order in open_orders:
                if order['type'] == 'TAKE_PROFIT_MARKET' and order['side'] == 'BUY':
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit or our new one is better (lower), return the new one
            if not existing_take_profit or current_take_profit < existing_take_profit:
                logger.info(f"Short position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100}%)")
                return current_take_profit
        
        return None
        
    def update_balance_for_compounding(self):
        """Update balance tracking for auto-compounding"""
        if not AUTO_COMPOUND:
            return False
            
        current_balance = self.binance_client.get_account_balance()
        
        # First time initialization
        if self.last_known_balance is None:
            self.last_known_balance = current_balance
            self.initial_balance = current_balance
            return False
        
        profit = current_balance - self.last_known_balance
        
        if profit > 0:
            # We've made profits since last update
            reinvest_amount = profit * COMPOUND_REINVEST_PERCENT
            logger.info(f"Auto-compounding: {reinvest_amount:.2f} USDT from recent {profit:.2f} USDT profit")
            self.last_known_balance = current_balance
            return True
            
        return False


def round_step_size(quantity, step_size):
    """Round quantity based on step size"""
    precision = int(round(-math.log10(step_size)))
    return round(math.floor(quantity * 10**precision) / 10**precision, precision)


def get_step_size(min_qty):
    """Get step size from min_qty"""
    step_size = min_qty
    # Handle cases where min_qty is not the step size (common in Binance)
    if float(min_qty) > 0:
        step_size = float(min_qty)
        
    return step_size