import pandas as pd
from typing import Dict, Callable, List, Any # Consolidated typing imports
import os # For path joining
import numpy as np # Keep numpy 

# Remove duplicate/unused imports like math, jsonpickle, string etc. unless needed elsewhere

# --- 1. Import custom classes ---
# Make sure this matches your trader script filename (BS3.py?)
from BS3 import Trader
from backtester import Backtester
# Assuming datamodel.py contains these definitions
from datamodel import Listing, OrderDepth, Trade, Order # Add other necessary imports from datamodel

# --- Configuration ---

# *** Base directory containing the day CSV files ***
DATA_SUBDIR = 'round-3-island-data-bottle'
BASE_DATA_DIR = os.path.join('data', DATA_SUBDIR) # 'data/round-3-island-data-bottle'

# Define the products your strategy trades
VOLCANIC_PRODUCTS = [
    "VOLCANIC_ROCK",
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500",
]

# --- Configurations valid for all days ---
# Using the listings and limits from your provided code
listings: Dict[str, Listing] = {
    "VOLCANIC_ROCK": Listing(symbol="VOLCANIC_ROCK", product="VOLCANIC_ROCK", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_9500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_9500", product="VOLCANIC_ROCK_VOUCHER_9500", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_9750": Listing(symbol="VOLCANIC_ROCK_VOUCHER_9750", product="VOLCANIC_ROCK_VOUCHER_9750", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10000": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10000", product="VOLCANIC_ROCK_VOUCHER_10000", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10250": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10250", product="VOLCANIC_ROCK_VOUCHER_10250", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10500", product="VOLCANIC_ROCK_VOUCHER_10500", denomination="SEASHELLS"),
}

position_limit: Dict[str, int] = {
    # *** NOTE: Your code had 200 for VOLCANIC_ROCK, but you mentioned fixing it to 400 earlier. Adjust if needed. ***
    "VOLCANIC_ROCK": 400,  # Set to 400 based on previous fix
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
}

# Using the calculate_mid_price from your code
def calculate_mid_price(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders:
         if not order_depth.buy_orders and not order_depth.sell_orders: return None
         elif not order_depth.buy_orders: return min(order_depth.sell_orders.keys())
         else: return max(order_depth.buy_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

# Using the fair_marks definition from your code
fair_marks: Dict[str, Callable[[OrderDepth], float | None]] = {
    # Filter fair marks only for products we are trading
    prod: calculate_mid_price for prod in VOLCANIC_PRODUCTS
}
# Or use fair_marks = {}

# --- Multi-Day Setup ---
DAYS_TO_RUN = [0, 1, 2] # Define which days to run
daily_results = [] # To store results from each day
log_dir = 'logs' # Directory to store output logs
os.makedirs(log_dir, exist_ok=True) # Create log directory if it doesn't exist

# --- Main Loop ---
for day in DAYS_TO_RUN:
    print(f"\n========== RUNNING BACKTEST FOR DAY {day} ==========")

    # Construct dynamic filenames
    market_data_file = os.path.join(BASE_DATA_DIR, f"prices_round_3_day_{day}.csv")
    trade_history_file = os.path.join(BASE_DATA_DIR, f"trades_round_3_day_{day}.csv")
    output_log_file = os.path.join(log_dir, f"backtest_log_volcanic_day_{day}.log")

    # --- Load Data for the current day ---
    print(f"Loading market data from: {market_data_file}")
    try:
        market_data_df_all = pd.read_csv(market_data_file, sep=';')
        # Filter Market Data
        market_data_df = market_data_df_all[market_data_df_all['product'].isin(VOLCANIC_PRODUCTS)].copy()
        print(f"Market data loaded. Filtered {len(market_data_df)} rows for Volcanic products.")
        if market_data_df.empty:
            print(f"Warning: No data found for Volcanic products on Day {day}. Skipping.")
            continue
    except FileNotFoundError:
        print(f"Error: Market data file not found for Day {day} at {market_data_file}. Skipping.")
        continue
    except Exception as e:
        print(f"Error loading market data for Day {day}: {e}. Skipping.")
        continue

    print(f"Loading trade history from: {trade_history_file}")
    try:
        trade_history_df_all = pd.read_csv(trade_history_file, sep=';')
        # Filter Trade History
        trade_history_df = trade_history_df_all[trade_history_df_all['symbol'].isin(VOLCANIC_PRODUCTS)].copy()
        print(f"Trade history loaded. Filtered {len(trade_history_df)} rows for Volcanic products.")
    except FileNotFoundError:
        print(f"Warning: Trade history file not found for Day {day}. Proceeding without market trades.")
        trade_history_df = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller'])
    except Exception as e:
        print(f"Error loading trade history for Day {day}: {e}. Proceeding with empty trades.")
        trade_history_df = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller'])

    # --- Instantiate NEW Trader and Backtester for EACH day ---
    print("Instantiating Trader...")
    trader = Trader() # Create a new trader instance for the day
    print("Trader instantiated.")

    print("Instantiating Backtester...")
    active_listings = {p: l for p, l in listings.items() if p in VOLCANIC_PRODUCTS}
    backtester = Backtester(
        trader=trader, # Use the new trader instance
        listings=active_listings,
        position_limit=position_limit,
        fair_marks=fair_marks,
        market_data=market_data_df,
        trade_history=trade_history_df,
        file_name=output_log_file
    )
    print("Backtester instantiated.")

    # --- Run Backtest for the current day ---
    print(f"Running backtest for Day {day}...")
    try:
        backtester.run()
        print(f"Backtest finished for Day {day}. Results logged to: {output_log_file}")

        # --- Collect results for summary ---
        final_pnl = sum(pnl for pnl in backtester.pnl.values() if pnl is not None)
        final_positions = backtester.current_position.copy()
        avg_runtime = (sum(backtester.run_times) / len(backtester.run_times) * 1000) if backtester.run_times else 0
        daily_results.append({
            "day": day,
            "pnl": final_pnl,
            "avg_runtime_ms": avg_runtime,
            "final_positions": final_positions
        })
        print(f"Day {day} Summary: PnL = {final_pnl:.2f}")

    except Exception as e:
        import traceback
        print(f"!!! ERROR during backtest run for Day {day}: {e}")
        traceback.print_exc()


# --- Overall Summary ---
print("\n========== OVERALL BACKTEST SUMMARY ==========")
total_pnl_all_days = 0
if not daily_results:
    print("No days were successfully backtested.")
else:
    successful_days = len(daily_results)
    print(f"Results across {successful_days} successfully backtested day(s):")
    for result in daily_results:
        print(f"Day {result['day']}: PnL = {result['pnl']:.2f}, Avg Runtime = {result['avg_runtime_ms']:.4f} ms")
        total_pnl_all_days += result['pnl']

    print(f"\nTotal PnL across {successful_days} days: {total_pnl_all_days:.2f}")