from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    SPREAD = "SPREAD"


PARAMS = {
    
    Product.VOLCANIC_ROCK: {
    },
    Product.SPREAD: {
        "default_spread_mean": 379.50439988484239,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 58,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 9500,
        "starting_time_to_expiry": 7,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 9750,
        "starting_time_to_expiry": 7,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10000,
        "starting_time_to_expiry": 7,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10250,
        "starting_time_to_expiry": 7,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.15959997370608378,
        "threshold": 0.00163,
        "strike": 10500,
        "starting_time_to_expiry": 7,
        "std_window": 6,
        "zscore_threshold": 21,
    },

}

from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))
 
    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400, 
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
        # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume
    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) ->(List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_voucher_mid_price(
        self,
        product: str,
        order_depth: OrderDepth,
        traderData: Dict[str, Any] # Takes the dictionary as argument
    ) -> float | None:
        product_state = traderData.get(product, {})
        prev_price = product_state.get('prev_voucher_price', None)

        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            # Update the passed dictionary
            # Ensure product key exists before assigning sub-key (robustness)
            if product not in traderData:
                 traderData[product] = {}
            traderData[product]['prev_voucher_price'] = mid_price

            return mid_price
        elif prev_price is not None:
            return prev_price
        else:
            return None


    def generate_hedge_orders( # Keep this as before
        self,
        rock_trade_qty: float,
        rock_position: int,
        rock_order_depth: OrderDepth
    ) -> List[Order]:
        # ... (implementation as before) ...
        orders: List[Order] = []
        if rock_order_depth is None: return orders
        rock_limit = self.LIMIT[Product.VOLCANIC_ROCK]
        rock_trade_qty_int = int(round(rock_trade_qty))
        if rock_trade_qty_int > 0:
            max_buy_vol = rock_limit - rock_position
            if max_buy_vol <= 0: return orders
            quantity_to_buy = min(rock_trade_qty_int, max_buy_vol)
            if quantity_to_buy > 0 and len(rock_order_depth.sell_orders) > 0:
                best_ask = min(rock_order_depth.sell_orders.keys())
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity_to_buy))
        elif rock_trade_qty_int < 0:
            max_sell_vol = rock_limit + rock_position
            if max_sell_vol <= 0: return orders
            quantity_to_sell = min(abs(rock_trade_qty_int), max_sell_vol)
            if quantity_to_sell > 0 and len(rock_order_depth.buy_orders) > 0:
                best_bid = max(rock_order_depth.buy_orders.keys())
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity_to_sell))
        return orders
    # --- End Helper Functions ---


    # Renamed from rock_voucher_orders
    def get_voucher_orders(
        self,
        product: str, # Added parameter
        order_depth: OrderDepth,
        position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> (List[Order], List[Order]): # Return take_orders, make_orders
        
        # Access product-specific params and state
        params = self.params[product]
        limit = self.LIMIT[product]
        product_state = traderData.get(product, {}) # Get state for this product
        past_vol = product_state.get('past_coupon_vol', []) # Get past vol for this product

        past_vol.append(volatility)
        if len(past_vol) > params['std_window']:
            past_vol.pop(0)
        
        # Update state in traderData
        traderData[product]['past_coupon_vol'] = past_vol 

        if len(past_vol) < params['std_window']:
            return None, None # Not enough data yet

        vol_std_dev = np.std(past_vol)

        if vol_std_dev < 1e-8:
            vol_z_score = 0.0
        else:
            # Use product-specific mean volatility
            vol_z_score = (volatility - params['mean_volatility']) / vol_std_dev 

        take_orders = []
        make_orders = []
        
        # Use product-specific zscore threshold
        if vol_z_score >= params['zscore_threshold']:
            target_position = -limit
            if position > target_position: # Need to sell more
                if len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    needed_quantity = abs(target_position - position) # How many to sell
                    available_quantity = order_depth.buy_orders[best_bid]
                    
                    take_quantity = min(needed_quantity, available_quantity)
                    if take_quantity > 0:
                        take_orders.append(Order(product, best_bid, -take_quantity))
                        
                    quote_quantity = needed_quantity - take_quantity
                    if quote_quantity > 0:
                         # Quote remaining at the same price for simplicity
                         make_orders.append(Order(product, best_bid, -quote_quantity))

        elif vol_z_score <= -params['zscore_threshold']:
            target_position = limit
            if position < target_position: # Need to buy more
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    needed_quantity = abs(target_position - position) # How many to buy
                    available_quantity = abs(order_depth.sell_orders[best_ask])

                    take_quantity = min(needed_quantity, available_quantity)
                    if take_quantity > 0:
                        take_orders.append(Order(product, best_ask, take_quantity))
                        
                    quote_quantity = needed_quantity - take_quantity
                    if quote_quantity > 0:
                        # Quote remaining at the same price for simplicity
                        make_orders.append(Order(product, best_ask, quote_quantity))

        return take_orders, make_orders
    
    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

 # --- RUN METHOD (Robust Smile Strategy Version) ---
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
                if not isinstance(traderObject, dict): traderObject = {}
            except Exception: traderObject = {}

        VOUCHER_PRODUCTS = [
            Product.VOLCANIC_ROCK_VOUCHER_9500, Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000, Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500,
        ]

        # Ensure keys exist in traderObject
        if 'base_iv_history' not in traderObject: traderObject['base_iv_history'] = []
        for product in VOUCHER_PRODUCTS:
             if product not in traderObject: traderObject[product] = {"prev_voucher_price": None}

        result = {}
        conversions = 0
        total_target_delta_exposure = 0.0

        rock_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        rock_order_depth = state.order_depths.get(Product.VOLCANIC_ROCK)
        rock_mid_price = None
        if rock_order_depth and rock_order_depth.buy_orders and rock_order_depth.sell_orders:
             rock_mid_price = (max(rock_order_depth.buy_orders.keys()) + min(rock_order_depth.sell_orders.keys())) / 2

        if rock_mid_price is None: return {}, conversions, jsonpickle.encode(traderObject)

        # --- Step 1: Gather Data ---
        smile_data_points = []
        example_params = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]
        tte = example_params["starting_time_to_expiry"] - (state.timestamp / 1000000 / 250)

        if tte <= 1e-6: return {}, conversions, jsonpickle.encode(traderObject)

        for voucher_product in VOUCHER_PRODUCTS:
            if voucher_product not in state.order_depths: continue
            voucher_order_depth = state.order_depths[voucher_product]
            params = self.params[voucher_product]
            strike = params["strike"]

            voucher_mid_price = self.get_voucher_mid_price(voucher_product, voucher_order_depth, traderObject)
            if voucher_mid_price is None or voucher_mid_price <= 0: continue

            try:
                 log_moneyness = math.log(strike / rock_mid_price)
                 sqrt_tte = math.sqrt(tte)
                 if sqrt_tte < 1e-6: continue
                 m_t = log_moneyness / sqrt_tte
            except ValueError: continue

            try:
                v_t_raw = BlackScholes.implied_volatility(voucher_mid_price, rock_mid_price, strike, tte)
                # *** Relaxed IV sanity check ***
                if not (0.0 <= v_t_raw < 1.5): continue # Check >= 0.0
                v_t = v_t_raw
            except Exception as e: continue

            smile_data_points.append({'m': m_t, 'v': v_t, 'product': voucher_product, 'K': strike, 'price': voucher_mid_price})

        # --- Step 2: Fit Parabola ---
        poly_func = None
        base_iv = None
        coeffs = None

        if len(smile_data_points) >= 3:
            m_values = np.array([p['m'] for p in smile_data_points])
            v_values = np.array([p['v'] for p in smile_data_points])

            # *** Added check for flat IVs ***
            if np.std(v_values) < 1e-5:
                pass # Skip fit if flat
            else:
                 try:
                     coeffs = np.polyfit(m_values, v_values, 2)
                     poly_func = np.poly1d(coeffs)
                     base_iv = coeffs[2]
                     if base_iv is not None:
                         traderObject['base_iv_history'].append(base_iv)
                         if len(traderObject['base_iv_history']) > 200: traderObject['base_iv_history'].pop(0)
                 except (np.linalg.LinAlgError, ValueError) as e:
                     poly_func = None; base_iv = None; coeffs = None

        # --- Step 3: Generate Orders ---
        active_orders = {}
        deviation_threshold = 0.000525 # -> TUNE THIS
        trade_size = 10       # -> TUNE THIS

        if poly_func is not None:
            for point in smile_data_points:
                voucher_product = point['product']
                m_t = point['m']
                v_t = point['v']
                v_fitted = poly_func(m_t)
                deviation = v_t - v_fitted

                current_position = state.position.get(voucher_product, 0)
                limit = self.LIMIT[voucher_product]
                target_quantity = 0

                if deviation < -deviation_threshold:
                    target_quantity = min(trade_size, limit - current_position)
                elif deviation > deviation_threshold:
                    target_quantity = max(-trade_size, -limit - current_position)

                if target_quantity != 0:
                    order_price = None
                    order_qty = 0
                    voucher_order_depth = state.order_depths.get(voucher_product)
                    if voucher_order_depth is None: continue

                    if target_quantity > 0: # Buy
                         if voucher_order_depth.sell_orders:
                              order_price = min(voucher_order_depth.sell_orders.keys())
                              order_qty = target_quantity
                    elif target_quantity < 0: # Sell
                         if voucher_order_depth.buy_orders:
                              order_price = max(voucher_order_depth.buy_orders.keys())
                              order_qty = target_quantity

                    if order_price is not None and order_qty != 0:
                         if voucher_product not in active_orders: active_orders[voucher_product] = []
                         active_orders[voucher_product].append(Order(voucher_product, order_price, order_qty))

        # --- Step 4: Calculate Total Delta and Generate Hedge ---
        total_target_delta_exposure = 0.0
        for voucher_product in VOUCHER_PRODUCTS:
            current_position = state.position.get(voucher_product, 0)
            intended_trade_qty = sum(o.quantity for o in active_orders.get(voucher_product, []))
            # *** Correctly assign target_position ***
            target_position = current_position + intended_trade_qty

            if target_position == 0: continue

            product_params = self.params[voucher_product]
            strike = product_params['strike']
            v_t_for_delta = None
            for point in smile_data_points:
                 if point['product'] == voucher_product:
                      v_t_for_delta = point['v']; break
            if v_t_for_delta is None: continue

            try:
                delta = BlackScholes.delta(rock_mid_price, strike, tte, v_t_for_delta)
                total_target_delta_exposure += target_position * delta
            except Exception: continue

        if rock_order_depth is not None:
             target_rock_position = -total_target_delta_exposure
             rock_trade_qty = target_rock_position - rock_position
             rock_hedge_orders = self.generate_hedge_orders(rock_trade_qty, rock_position, rock_order_depth)
             if rock_hedge_orders:
                 active_orders[Product.VOLCANIC_ROCK] = rock_hedge_orders

        # --- Final Step ---
        result = active_orders
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData

