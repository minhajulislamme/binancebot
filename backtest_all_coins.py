#!/usr/bin/env python3
"""
Backtest script for testing all cryptocurrency-specific strategies
"""
import sys
import os
import logging
import argparse
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the backtest module
from main import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('backtest_all_coins')

def main():
    parser = argparse.ArgumentParser(description='Run backtests for multiple cryptocurrencies with their specialized strategies')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--coins', nargs='+', default=None, help='Specific coins to backtest (e.g., BTC ETH)')
    args = parser.parse_args()
    
    # Define the coins and their optimized strategies
    coins_strategies = {
        'BTCUSDT': 'BTC_Scalping',
        'ETHUSDT': 'ETH_StochMACD',
        'BNBUSDT': 'BNB_Grid',
        'SOLUSDT': 'SOL_Squeeze',
        'ADAUSDT': 'ADA_EMATrend',
        'XRPUSDT': 'XRP_Scalping',
        'DOGEUSDT': 'DOGE_Scalping',
        'SHIBUSDT': 'SHIB_Breakout'
    }
    
    # If specific coins are provided, filter the dictionary
    if args.coins:
        filtered_coins = {}
        for coin in args.coins:
            coin_upper = coin.upper()
            if not coin_upper.endswith('USDT'):
                coin_upper += 'USDT'
            if coin_upper in coins_strategies:
                filtered_coins[coin_upper] = coins_strategies[coin_upper]
            else:
                logger.warning(f"Strategy not found for {coin_upper}")
        coins_strategies = filtered_coins
    
    # Dictionary to store backtest results
    results = {}
    
    # Calculate start date
    start_date = f"{args.days} days ago"
    
    # Run backtest for each coin with its specialized strategy
    logger.info(f"Starting backtest for {len(coins_strategies)} coins over {args.days} days")
    for symbol, strategy_name in coins_strategies.items():
        logger.info(f"Backtesting {symbol} with {strategy_name} strategy")
        
        # Define timeframes based on the coin's strategy
        if symbol in ['BTCUSDT', 'XRPUSDT', 'DOGEUSDT']:
            timeframes = ['1m', '3m', '5m']
        elif symbol in ['ETHUSDT', 'BNBUSDT']:
            timeframes = ['1m', '3m', '5m','15m', '30m', '1h']
        elif symbol in ['SOLUSDT', 'ADAUSDT', 'SHIBUSDT']:
            timeframes = ['5m', '15m']
        else:
            timeframes = ['15m']
        
        coin_results = {}
        
        # Run backtest for each timeframe
        for timeframe in timeframes:
            logger.info(f"Running {symbol} on {timeframe} timeframe")
            result = run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                start_date=start_date,
                save_results=True
            )
            
            if result:
                coin_results[timeframe] = {
                    'total_return': result.get('total_return', 0),
                    'win_rate': result.get('win_rate', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'total_trades': result.get('total_trades', 0)
                }
                
                logger.info(f"{symbol} {timeframe} backtest: Return: {result.get('total_return', 0):.2f}%, "
                           f"Win Rate: {result.get('win_rate', 0):.2f}%, "
                           f"Max DD: {result.get('max_drawdown', 0):.2f}%, "
                           f"Sharpe: {result.get('sharpe_ratio', 0):.2f}")
            else:
                logger.error(f"Backtest failed for {symbol} on {timeframe}")
        
        results[symbol] = coin_results
    
    # Print summary of results
    logger.info("\n\nBacktest Results Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Symbol':<10} {'Timeframe':<10} {'Return%':<10} {'Win Rate':<10} {'Max DD%':<10} {'Sharpe':<10} {'Trades':<10}")
    logger.info("=" * 80)
    
    for symbol in results:
        for timeframe, metrics in results[symbol].items():
            logger.info(f"{symbol:<10} {timeframe:<10} "
                      f"{metrics['total_return']:<10.2f} "
                      f"{metrics['win_rate']:<10.2f} "
                      f"{metrics['max_drawdown']:<10.2f} "
                      f"{metrics['sharpe_ratio']:<10.2f} "
                      f"{metrics['total_trades']:<10}")
    
    logger.info("\nBacktesting completed!")

if __name__ == "__main__":
    main()