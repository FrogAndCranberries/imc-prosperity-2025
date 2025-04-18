from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
from typing import List
import jsonpickle
from math import log, sqrt, exp
from statistics import NormalDist

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


# REMEMBER, WE DO IT THIS WAY NOT BECAUSE IT IS EASY!
# But because we thought it would be.

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

            return returns_dic


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


    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        # print("traderData: " + state.traderData)
        # print("Observations: " + str(state.observations))
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

        # Extract traderData
        if state.traderData != "":
            traderData : dict[str,list[int]] = jsonpickle.decode(state.traderData)
        else:
            traderData = {"RAINFOREST_RESIN":[],
                          "KELP":[],
                          "SQUID_INK":[], 
                          "JAMS":[], 
                          "CROISSANTS":[], 
                          "DJEMBES":[],
                          "PICNIC_BASKET1":[],
                          "PICNIC_BASKET2":[]}
            
        # Calculate current midprices for all products
        midprices = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            productPrices = traderData[product]

            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                
            else:
                continue

            midprices[product] = sum([best_ask,best_bid])/2.0


        # For each basket, check spread and position to either close positions or investigate opportunity for buy or sell arbitrage
        for basket in self.BASKETS:

            orders: List[Order] = []

            # Check all components and basket are available
            skip_arbitrage = False
            for product in list(self.BASKET_CONTENTS[basket].keys()) + [basket]:
                if len(state.order_depths[product].sell_orders) == 0 and len(state.order_depths[product].buy_orders) == 0:
                    skip_arbitrage = True
            if skip_arbitrage:
                continue

            # Get midprice spread
            assembled_midprice = sum([self.BASKET_CONTENTS[basket][component] * midprices[component] for component in self.BASKET_CONTENTS[basket].keys()])
            midprice_spread = midprices[basket] - assembled_midprice


            # If we have no position, check if spread is +ve enough to sell or -ve enough to buy
            if basket not in state.position.keys() or state.position[basket] == 0:
                if midprice_spread > self.OPENING_SPREAD_THRESHOLD[basket]:
                    # print(f"Midprice spread {midprice_spread} on {basket}, testing short")
                    orders = self.open_short_arbitrage(state, basket)
                elif midprice_spread < - self.OPENING_SPREAD_THRESHOLD[basket]:
                    # print(f"Midprice spread {midprice_spread} on {basket}, testing long")
                    orders = self.open_long_arbitrage(state, basket)

            # If we're long on basket and spread is closing, sell position. If spread is still negative, buy more
            elif state.position[basket] > 0:
                if midprice_spread > - self.CLOSING_SPREAD_THRESHOLD[basket]:
                    print(f"Midprice spread {midprice_spread} on {basket}, closing long")
                    orders = self.close_long_arbitrage(state, basket)
                elif midprice_spread < - self.OPENING_SPREAD_THRESHOLD[basket]:
                    # print(f"Midprice spread {midprice_spread} on {basket}, testing long")
                    orders = self.open_long_arbitrage(state, basket)

            # If we're short on basket and spread is closing, buy back position. If spread is still positive, sell more
            elif state.position[basket] < 0:
                if midprice_spread < self.CLOSING_SPREAD_THRESHOLD[basket]:
                    print(f"Midprice spread {midprice_spread} on {basket}, closing short")
                    orders = self.close_short_arbitrage(state, basket)
                elif midprice_spread > self.OPENING_SPREAD_THRESHOLD[basket]:
                    # print(f"Midprice spread {midprice_spread} on {basket}, testing short")                    
                    orders = self.open_short_arbitrage(state, basket)


            # Append orders to result
            for order in orders:
                if order.symbol in result.keys():
                    result[order.symbol].append(order)
                else:
                    result[order.symbol] = [order]


        for product in ["KELP", "RAINFOREST_RESIN"]:
            if product not in state.order_depths.keys():
                continue
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            productPrices = traderData[product]

                
    
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                
            else:
                continue
            midprice = sum([best_ask,best_bid])/2.0

            productPrices += [midprice]

            if len(productPrices) > 200:
                productPrices=productPrices[-200:]

            avg_price = sum(productPrices)/len(productPrices)
            factor = .6

            top10 = avg_price + (max(productPrices[-100:]) - avg_price) * factor
            bot10 = avg_price - (avg_price - min(productPrices[-100:]))*factor
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                for ask,amount in list(order_depth.sell_orders.items()):
                    if ask < bot10:
                        # print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, ask, -amount))
        
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                for bid,amount in list(order_depth.buy_orders.items()):
                    if bid > top10:
                        # print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, bid, -amount))
            if product in result.keys():
                result[product] += orders
            else:
                result[product] = orders
            traderData[product] = productPrices


        traderData = jsonpickle.encode(traderData) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        # for key, value in result.items():
        #     print(key)
        #     print(value)

        return result, conversions, traderData






    # Buying baskets and selling components
    def open_long_arbitrage(self, state: TradingState, basket: Symbol):

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows

        component_bids_per_basket = self.component_bids_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(bids) for bids in component_bids_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket.append(sum([bids[i] for bids in component_bids_per_basket.values()]))

        # Collate into order_depth list format
        assembled_bid_depth = [[bid, assembled_prices_per_basket.count(bid)] for bid in sorted(set(assembled_prices_per_basket), reverse=True)]
        basket_ask_depth = [list(item) for item in state.order_depths[basket].sell_orders.items()]

        top_assembled_bid = assembled_bid_depth[0][0]

        # print("Long arbitrage:")
        # print("Basket ask depth: ", basket_ask_depth)
        # print("Assembled bid depth: ", assembled_bid_depth)

        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_bid_depth) > 0 and len(basket_ask_depth) > 0:

            best_assembled_bid, best_assembled_bid_amount = assembled_bid_depth[0] # Hopefully high
            best_basket_ask, best_basket_ask_amount = basket_ask_depth[0] # Hopefully low

            best_spread = best_basket_ask - best_assembled_bid # Hopefully very negative
            if best_spread < - self.OPENING_SPREAD_THRESHOLD[basket]:
                max_order_volume += min(abs(best_assembled_bid_amount), abs(best_basket_ask_amount))
                if abs(best_assembled_bid_amount) < abs(best_basket_ask_amount):
                    basket_ask_depth[0][1] += best_assembled_bid_amount # Plus here since one value is always +ve, other -ve
                    assembled_bid_depth.pop(0)
                elif abs(best_assembled_bid_amount) > abs(best_basket_ask_amount):
                    assembled_bid_depth[0][1] += best_basket_ask_amount # Plus here since one value is always +ve, other -ve
                    basket_ask_depth.pop(0)
                else:
                    basket_ask_depth.pop(0)
                    assembled_bid_depth.pop(0)
            else:
                break

        order_volume = min(max_order_volume, self.arbitrage_volume_limit(state, basket)[0])


        # Order to buy baskets
        orders: List[Order] = []
        if order_volume != 0:
            price = self.worst_price_for_volume(state, basket, order_volume)
            print(f"Opening long on {basket}: {order_volume} at {price} or better against assembled basket at {top_assembled_bid}.")
            orders.append(Order(basket, price, order_volume))

            # Orders to sell components
            for component in self.BASKET_CONTENTS[basket].keys():
                component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
                orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders

    # Selling baskets and buying components
    def open_short_arbitrage(self, state: TradingState, basket: Symbol):

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows
        component_asks_per_basket = self.component_asks_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(asks) for asks in component_asks_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket.append(sum([asks[i] for asks in component_asks_per_basket.values()]))

        # Collate into order_depth list format
        assembled_ask_depth = [[ask, - assembled_prices_per_basket.count(ask)] for ask in sorted(set(assembled_prices_per_basket))]
        basket_bid_depth = [list(item) for item in state.order_depths[basket].buy_orders.items()]

        # print("Short arbitrage:")
        # print("Basket bid depth: ", basket_bid_depth)
        # print("Items: ", component_asks_per_basket)
        # print("max baskets: ", max_baskets)
        # print("Assembled individual prices: ", assembled_prices_per_basket)
        # print("Assembled price depth: ", assembled_ask_depth)
        top_assembled_ask = assembled_ask_depth[0][0]

        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_ask_depth) > 0 and len(basket_bid_depth) > 0:

            best_assembled_ask, best_assembled_ask_amount = assembled_ask_depth[0] # Hopefully low
            best_basket_bid, best_basket_bid_amount = basket_bid_depth[0] # Hopefully high

            best_spread = best_basket_bid - best_assembled_ask # Hopefully very positive
            if best_spread > self.OPENING_SPREAD_THRESHOLD[basket]:
                max_order_volume -= min(abs(best_assembled_ask_amount), abs(best_basket_bid_amount))
                if abs(best_assembled_ask_amount) < abs(best_basket_bid_amount):
                    basket_bid_depth[0][1] += best_assembled_ask_amount # Plus here since one value is always +ve, other -ve
                    assembled_ask_depth.pop(0)
                elif abs(best_assembled_ask_amount) > abs(best_basket_bid_amount):
                    assembled_ask_depth[0][1] += best_basket_bid_amount # Plus here since one value is always +ve, other -ve
                    basket_bid_depth.pop(0)
                else:
                    basket_bid_depth.pop(0)
                    assembled_ask_depth.pop(0)
            else:
                break

        # Both max_order_volume and arbitrage_volume_limits()[1] are negative -> get the smaller magnitude with max
        order_volume = max(max_order_volume, self.arbitrage_volume_limit(state, basket)[1])


        orders: List[Order] = []
        if order_volume != 0:
            # Order to sell baskets - volume is negative
            price = self.worst_price_for_volume(state, basket, order_volume)
            print(f"Opening short on {basket}: {order_volume} at {price} or better against assembled basket at {top_assembled_ask}.")
            orders.append(Order(basket, price, order_volume))

            # Orders to buy components
            for component in self.BASKET_CONTENTS[basket].keys():
                component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
                orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders

    def close_long_arbitrage(self, state: TradingState, basket: Symbol):
        orders: List[Order] = []
        position = state.position[basket]

        # print(f"Closing position {position} of {basket}")

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows
        component_asks_per_basket = self.component_asks_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(asks) for asks in component_asks_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket.append(sum([asks[i] for asks in component_asks_per_basket.values()]))

        # Collate into order_depth list format
        assembled_ask_depth = [[ask, - assembled_prices_per_basket.count(ask)] for ask in sorted(set(assembled_prices_per_basket))]
        basket_bid_depth = [list(item) for item in state.order_depths[basket].buy_orders.items()]

        top_assembled_ask = assembled_ask_depth[0][0]
        # Find max volume to trade within position closing spread
        max_order_volume = 0

        while len(assembled_ask_depth) > 0 and len(basket_bid_depth) > 0:

            best_assembled_ask, best_assembled_ask_amount = assembled_ask_depth[0] # Hopefully low
            best_basket_bid, best_basket_bid_amount = basket_bid_depth[0] # Hopefully high

            best_spread = best_basket_bid - best_assembled_ask # Hopefully very positive
            if best_spread > - self.CLOSING_SPREAD_THRESHOLD[basket]:
                max_order_volume -= min(abs(best_assembled_ask_amount), abs(best_basket_bid_amount))
                if abs(best_assembled_ask_amount) < abs(best_basket_bid_amount):
                    basket_bid_depth[0][1] += best_assembled_ask_amount # Plus here since one value is always +ve, other -ve
                    assembled_ask_depth.pop(0)
                elif abs(best_assembled_ask_amount) > abs(best_basket_bid_amount):
                    assembled_ask_depth[0][1] += best_basket_bid_amount # Plus here since one value is always +ve, other -ve
                    basket_bid_depth.pop(0)
                else:
                    basket_bid_depth.pop(0)
                    assembled_ask_depth.pop(0)
            else:
                break

        print(max_order_volume, "   -   ", -position)
        order_volume = max(max_order_volume, - position)


        orders: List[Order] = []

        if order_volume != 0:
            # Order to sell baskets
            price = self.worst_price_for_volume(state, basket, order_volume)
            print(f"Closing long on {basket}: {order_volume} at {price} or better against assembled basket at {top_assembled_ask}.")

            orders.append(Order(basket, price, order_volume))

            # Orders to buy components
            for component in self.BASKET_CONTENTS[basket].keys():
                component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
                orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders

    # Buy back baskets and sell components to close short arbitrage position
    def close_short_arbitrage(self, state: TradingState, basket: Symbol):
        orders: List[Order] = []
        position = state.position[basket]

        # print(f"Closing position {position} of {basket}")

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows

        component_bids_per_basket = self.component_bids_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(bids) for bids in component_bids_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket.append(sum([bids[i] for bids in component_bids_per_basket.values()]))

        # Collate into order_depth list format
        assembled_bid_depth = [[bid, assembled_prices_per_basket.count(bid)] for bid in sorted(set(assembled_prices_per_basket), reverse=True)]
        basket_ask_depth = [list(item) for item in state.order_depths[basket].sell_orders.items()]

        top_assembled_bid = assembled_bid_depth[0][0]

        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_bid_depth) > 0 and len(basket_ask_depth) > 0:

            best_assembled_bid, best_assembled_bid_amount = assembled_bid_depth[0] # Hopefully high
            best_basket_ask, best_basket_ask_amount = basket_ask_depth[0] # Hopefully low

            best_spread = best_basket_ask - best_assembled_bid # Hopefully very negative
            if best_spread < self.CLOSING_SPREAD_THRESHOLD[basket]:
                max_order_volume += min(abs(best_assembled_bid_amount), abs(best_basket_ask_amount))
                if abs(best_assembled_bid_amount) < abs(best_basket_ask_amount):
                    basket_ask_depth[0][1] += best_assembled_bid_amount # Plus here since one value is always +ve, other -ve
                    assembled_bid_depth.pop(0)
                elif abs(best_assembled_bid_amount) > abs(best_basket_ask_amount):
                    assembled_bid_depth[0][1] += best_basket_ask_amount # Plus here since one value is always +ve, other -ve
                    basket_ask_depth.pop(0)
                else:
                    basket_ask_depth.pop(0)
                    assembled_bid_depth.pop(0)
            else:
                break
        
        print(max_order_volume, "   -   ", -position)
        order_volume = min(max_order_volume, - position)

        orders: List[Order] = []

        if order_volume != 0:
            # Order to buy baskets
            price = self.worst_price_for_volume(state, basket, order_volume)
            print(f"Closing short on {basket}: {order_volume} at {price} or better against assembled basket at {top_assembled_bid}.")

            orders.append(Order(basket, price, order_volume))

            # Orders to sell components
            for component in self.BASKET_CONTENTS[basket].keys():
                component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
                orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders
        
                    
    # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
    # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows
    def component_bids_per_basket(self, state: TradingState, basket: Symbol):
        component_bids_per_basket = {}
        for component in self.BASKET_CONTENTS[basket]:
            buy_order_depth = [list(item) for item in state.order_depths[component].buy_orders.items()]
            bid_per_basket = [0]
            counter = 0
            while True:
                if buy_order_depth[0][1] > 0:
                    counter += 1
                    buy_order_depth[0][1] -= 1
                    bid_per_basket[-1] += buy_order_depth[0][0]
                    if counter == self.BASKET_CONTENTS[basket][component]:
                        counter = 0
                        bid_per_basket.append(0)
                elif len(buy_order_depth) > 1:
                    buy_order_depth.pop(0)
                else:
                    bid_per_basket.pop(-1)
                    break

            component_bids_per_basket[component] = bid_per_basket

            # for bid in bid_per_basket:
            #     print(f"Bid per basket {basket} on {component}: {bid}")
        return component_bids_per_baske

    # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
    # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows
    def component_asks_per_basket(self, state: TradingState, basket: Symbol):
        component_asks_per_basket = {}
        for component in self.BASKET_CONTENTS[basket]:
            sell_order_depth = [list(item) for item in state.order_depths[component].sell_orders.items()]
            ask_per_basket = [0]
            counter = 0
            while True:
                if sell_order_depth[0][1] < 0:
                    counter += 1
                    sell_order_depth[0][1] += 1
                    ask_per_basket[-1] += sell_order_depth[0][0]
                    if counter == self.BASKET_CONTENTS[basket][component]:
                        counter = 0
                        ask_per_basket.append(0)
                elif len(sell_order_depth) > 1:
                    sell_order_depth.pop(0)
                else:
                    ask_per_basket.pop(-1)
                    break

            component_asks_per_basket[component] = ask_per_basket

            # for ask in ask_per_basket:
            #     print(f"Ask per basket {basket} on {component}: {ask}")
        return component_asks_per_basket

    def arbitrage_volume_limit(self, state: TradingState, basket: Symbol):
        component_buy_limits = []
        component_sell_limits = []
        for product in self.BASKET_CONTENTS[basket].keys():
            if product in state.position.keys():
                position = state.position[product]
            else:
                position = 0
            component_buy_limits.append((self.LIMITS[product] - position) // self.BASKET_CONTENTS[basket][product])
            component_sell_limits.append((self.LIMITS[product] + position) // self.BASKET_CONTENTS[basket][product])

        if basket in state.position.keys():
            position = state.position[basket]
        else:
            position = 0

        basket_buy_limit = self.LIMITS[basket] - position
        basket_sell_limit = (self.LIMITS[basket] + position)
        max_buy_volume = min(component_sell_limits + [basket_buy_limit])
        max_sell_volume = min(component_buy_limits + [basket_sell_limit])
                            
        return [max_buy_volume, -max_sell_volume]

    # The least favourable price to be paid when requesting to trade a volume against the order book. Volume is +ve to buy and -ve to sell
    def worst_price_for_volume(self, state: TradingState, product: Symbol, volume: int):


        if volume > 0:
            order_depth = [list(item) for item in state.order_depths[product].sell_orders.items()]
            if volume > - sum([order_depth[i][1] for i in range(len(order_depth))]):
                raise ValueError(f"Not enough supply in order book. Volume: {volume}")
            worst_price = 0
            while volume > 0:
                worst_price = order_depth[0][0]
                volume += order_depth[0][1]
                order_depth[0][1] += 1
                if order_depth[0][1] == 0:
                    order_depth.pop(0)
        elif volume < 0:
            order_depth = [list(item) for item in state.order_depths[product].buy_orders.items()]
            if volume < - sum([order_depth[i][1] for i in range(len(order_depth))]):
                raise ValueError(f"Not enough supply in order book. Volume: {volume}")
            worst_price = 0
            while volume < 0:
                worst_price = order_depth[0][0]
                volume += order_depth[0][1]
                order_depth[0][1] -= 1
                if order_depth[0][1] == 0:
                    order_depth.pop(0)
        else:
            print("Requested price for volume 0")
            worst_price = 0
        return worst_price

        # --- Final Step ---
