import datetime
import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import io
from datamodel import Listing, ConversionObservation
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation
from BS3 import Trader
from backtester import Backtester
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import pandas as pd
from typing import Dict, Callable

# --- 1. Import your classes ---
# Assuming your trader class is named Trader and is in blackscholes_round3.py
from BS3 import Trader # Make sure you've fixed the self.LIMIT issue inside this file
from backtester import Backtester
# Assuming datamodel.py contains these definitions
from datamodel import Listing, OrderDepth, Trade, Order # Add other necessary imports from datamodel

# --- Configuration ---
# Adjust file paths and names based on your actual files

MARKET_DATA_FILE = 'data/round-3-island-data-bottle/prices_round_3_day_0.csv' # <-- Added 'data/' prefix
TRADE_HISTORY_FILE = 'data/round-3-island-data-bottle/trades_round_3_day_0.csv' # <-- Added 'data/' prefix
OUTPUT_LOG_FILE = 'backtest_log_volcanic_day_0.log'

# --- 2. Define Listings ---
# Define the products based on your CSV and blackscholes_round3.py
listings: Dict[str, Listing] = {
    "VOLCANIC_ROCK": Listing(symbol="VOLCANIC_ROCK", product="VOLCANIC_ROCK", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_9500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_9500", product="VOLCANIC_ROCK_VOUCHER_9500", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_9750": Listing(symbol="VOLCANIC_ROCK_VOUCHER_9750", product="VOLCANIC_ROCK_VOUCHER_9750", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10000": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10000", product="VOLCANIC_ROCK_VOUCHER_10000", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10250": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10250", product="VOLCANIC_ROCK_VOUCHER_10250", denomination="SEASHELLS"),
    "VOLCANIC_ROCK_VOUCHER_10500": Listing(symbol="VOLCANIC_ROCK_VOUCHER_10500", product="VOLCANIC_ROCK_VOUCHER_10500", denomination="SEASHELLS"),
    # Add other products from your CSV if your trader interacts with them
}

# --- 3. Define Position Limits for the Backtester ---
# These limits are used by the Backtester to enforce rules during execution.
# They should ideally match or be consistent with the limits used inside your Trader logic.
position_limit: Dict[str, int] = {
    "VOLCANIC_ROCK": 200,  # *** Verify this limit for Volcanic Rock ***
    "VOLCANIC_ROCK_VOUCHER_9500": 200, # Matches the limit in blackscholes_round3.py
    "VOLCANIC_ROCK_VOUCHER_9750": 200, # Matches the limit in blackscholes_round3.py
    "VOLCANIC_ROCK_VOUCHER_10000": 200,# Matches the limit in blackscholes_round3.py
    "VOLCANIC_ROCK_VOUCHER_10250": 200,# Matches the limit in blackscholes_round3.py
    "VOLCANIC_ROCK_VOUCHER_10500": 200,# Matches the limit in blackscholes_round3.py
    # Add limits for other products if needed
}


def calculate_mid_price(order_depth: OrderDepth) -> float | None:
    # Handle cases with empty books to avoid errors
    if not order_depth.buy_orders or not order_depth.sell_orders:
        # Maybe return None, or use the last known price if available elsewhere?
        # Returning None might cause issues in the PnL calculation if not handled.
        # Let's try to return None and see if the Backtester handles it.
         if not order_depth.buy_orders and not order_depth.sell_orders:
             return None # No price available
         elif not order_depth.buy_orders:
             # Only sell orders exist - maybe use best ask? Or None?
             return min(order_depth.sell_orders.keys()) # Or return None
         else: # Only buy orders exist
             return max(order_depth.buy_orders.keys()) # Or return None
    
    # Both sides have orders
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

# Assign the mid-price function to all relevant products
fair_marks: Dict[str, Callable[[OrderDepth], float | None]] = {
    prod: calculate_mid_price for prod in listings.keys()
}

# --- In run_backtest.py ---

# --- (Previous code: imports, listings, limits, fair_marks) ---

# Define the products your strategy trades
VOLCANIC_PRODUCTS = [
    "VOLCANIC_ROCK",
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500",
]

# --- 5. Load Data ---
print(f"Loading market data from: {MARKET_DATA_FILE}")
try:
    market_data_df_all = pd.read_csv(MARKET_DATA_FILE, sep=';')
    # *** Filter Market Data ***
    market_data_df = market_data_df_all[market_data_df_all['product'].isin(VOLCANIC_PRODUCTS)].copy()
    print(f"Market data loaded successfully. Filtered {len(market_data_df)} rows for Volcanic products.")
    if market_data_df.empty:
        print(f"Error: No data found for Volcanic products in {MARKET_DATA_FILE}")
        exit()
except FileNotFoundError:
    print(f"Error: Market data file not found at {MARKET_DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error loading market data: {e}")
    exit()

print(f"Loading trade history from: {TRADE_HISTORY_FILE}")
try:
    trade_history_df_all = pd.read_csv(TRADE_HISTORY_FILE, sep=';')
    # *** Filter Trade History ***
    trade_history_df = trade_history_df_all[trade_history_df_all['symbol'].isin(VOLCANIC_PRODUCTS)].copy()
    print(f"Trade history loaded successfully. Filtered {len(trade_history_df)} rows for Volcanic products.")
except FileNotFoundError:
    print(f"Warning: Trade history file not found at {TRADE_HISTORY_FILE}. Proceeding without market trades.")
    trade_history_df = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller'])
    # exit() # Or proceed without trades
except Exception as e:
    print(f"Error loading trade history: {e}")
    exit()

# --- 6. Instantiate Trader ---
# ... (rest of the script remains the same, but passes the *filtered* dataframes) ...

# --- 7. Instantiate Backtester ---
print("Instantiating Backtester...")
backtester = Backtester(
    trader=Trader(),
    # Pass only the listings relevant to the filtered data
    listings={p: l for p, l in listings.items() if p in VOLCANIC_PRODUCTS},
    position_limit=position_limit, # Keep all limits, backtester needs them by symbol if accessed
    fair_marks=fair_marks,      # Keep all fair marks
    market_data=market_data_df, # Pass filtered market data
    trade_history=trade_history_df, # Pass filtered trade data
    file_name=OUTPUT_LOG_FILE
)
# ... (rest of script)

print("Backtester instantiated.")

# --- 8. Run Backtest ---
print("Running backtest...")
backtester.run()
print(f"Backtest finished. Results logged to: {OUTPUT_LOG_FILE}")

# --- Optional: Print Summary Stats ---
print("\n--- Backtest Summary ---")
# Filter out None values from PnL before summing
total_pnl = sum(pnl for pnl in backtester.pnl.values() if pnl is not None)
print(f"Final Total PnL across all products: {total_pnl:.2f}")

print("Final Positions:")
# Ensure all products potentially traded are shown, even if final position is 0
all_products = set(listings.keys()) | set(backtester.current_position.keys())
for product in sorted(list(all_products)):
    position = backtester.current_position.get(product, 0)
    print(f"  {product}: {position}")

print("Final Cash per product:")
for product in sorted(list(all_products)):
     cash_val = backtester.cash.get(product, 0)
     print(f"  {product}: {cash_val:.2f}")

if backtester.run_times:
     avg_runtime_ms = sum(backtester.run_times) / len(backtester.run_times) * 1000
     max_runtime_ms = max(backtester.run_times) * 1000
     print(f"\nAverage Trader.run() time: {avg_runtime_ms:.4f} ms")
     print(f"Maximum Trader.run() time: {max_runtime_ms:.4f} ms")
else:
    print("\nNo run time data collected.")