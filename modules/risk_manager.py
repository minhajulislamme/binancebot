import logging
import math
from modules.config import (
    INITIAL_BALANCE, RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, USE_TAKE_PROFIT, 
    TAKE_PROFIT_PCT, TRAILING_STOP, TRAILING_STOP_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, binance_client):
        """Initialize risk manager with a reference to binance client"""
        self.binance_client = binance_client
        self.initial_balance = None
        self.last_known_balance = None
        
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
        """Calculate stop loss price based on configuration"""
        if not USE_STOP_LOSS:
            return None
            
        if side == "BUY":  # Long position
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
        else:  # Short position
            stop_price = entry_price * (1 + STOP_LOSS_PCT)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_price = round(stop_price, price_precision)
            
        logger.info(f"Calculated stop loss at {stop_price}")
        return stop_price
        
    def calculate_take_profit(self, symbol, side, entry_price):
        """Calculate take profit price based on configuration"""
        if not USE_TAKE_PROFIT:
            return None
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
        else:  # Short position
            take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        logger.info(f"Calculated take profit at {take_profit_price}")
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
        
        # Calculate new stop loss based on current price
        if side == "BUY":  # Long position
            new_stop = current_price * (1 - TRAILING_STOP_PCT)
            # Only move stop loss up, never down
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop <= current_stop:
                return None
        else:  # Short position
            new_stop = current_price * (1 + TRAILING_STOP_PCT)
            # Only move stop loss down, never up
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop >= current_stop:
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop = round(new_stop, price_precision)
            
        logger.info(f"Adjusted trailing stop loss to {new_stop}")
        return new_stop
        
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