from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
from typing import List
import string
import jsonpickle


# class TraderData:
#     def __init__(self, average_price:int, num_rounds:int):
#         self.average_price = average_price
        # self.num_rounds = num_rounds


# REMEMBER, WE DO IT THIS WAY NOT BECAUSE IT IS EASY!
# But because we thought it would be.

class Trader:
    
    # Component volume must be more than 0 
    BASKET_CONTENTS = {
        "PICNIC_BASKET1": {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1},

        "PICNIC_BASKET2": {
            "CROISSANTS": 4,
            "JAMS": 2}}
    
    LIMITS = {
        "RAINFOREST_RESIN": 50, 
        "SQUID_INK": 50,
        "KELP": 50,
        "JAMS": 350, 
        "CROISSANTS": 250, 
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100}
    
    BASKETS = ["PICNIC_BASKET1", "PICNIC_BASKET2"]

    MIN_SPREAD_TO_OPEN_ARBITRAGE = {
        "PICNIC_BASKET1": 50, 
        "PICNIC_BASKET2": 50}
    
    MAX_SPREAD_TO_CLOSE = {
        "PICNIC_BASKET1": 10, 
        "PICNIC_BASKET2": 10}

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        # print("traderData: " + state.traderData)
        # print("Observations: " + str(state.observations))

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
            if basket not in state.position.keys():
                if midprice_spread > self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
                    orders = self.open_short_arbitrage(state, basket)
                elif midprice_spread < - self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
                    orders = self.open_long_arbitrage(state, basket)

            # If we're long on basket and spread is closing, sell position. If spread is still negative, buy more
            elif state.position[basket] > 0:
                if midprice_spread > - self.MAX_SPREAD_TO_CLOSE[basket]:
                    orders = self.close_long_arbitrage(state, basket)
                elif midprice_spread < - self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
                    orders = self.open_long_arbitrage(state, basket)

            # If we're short on basket and spread is closing, buy back position. If spread is still positive, sell more
            elif state.position[basket] < 0:
                if midprice_spread < self.MAX_SPREAD_TO_CLOSE[basket]:
                    orders = self.close_short_arbitrage(state, basket)
                elif midprice_spread > self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
                    orders = self.open_short_arbitrage(state, basket)
            else:
                print("Position shouldn't be 0")


            # Append orders to result
            for order in orders:
                if order.symbol in result.keys():
                    result[order.symbol].append(order)
                else:
                    result[order.symbol] = [order]

        traderData = jsonpickle.encode(traderData) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
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
            assembled_prices_per_basket[i] = sum([bids[i] for bids in component_bids_per_basket.values()])

        # Collate into order_depth list format
        assembled_bid_depth = [[bid, assembled_prices_per_basket.count(bid)] for bid in sorted(set(assembled_prices_per_basket))]
        basket_ask_depth = [list(item) for item in state.order_depths[basket].sell_orders.items()]


        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_bid_depth) > 0 and len(basket_ask_depth) > 0:

            best_assembled_bid, best_assembled_bid_amount = assembled_bid_depth[0] # Hopefully high
            best_basket_ask, best_basket_ask_amount = basket_ask_depth[0] # Hopefully low

            best_spread = best_basket_ask - best_assembled_bid # Hopefully very negative
            if best_spread < - self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
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
        orders.append(Order(basket, self.worst_price_for_volume(state, basket, order_volume), order_volume))

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
            assembled_prices_per_basket[i] = sum([asks[i] for asks in component_asks_per_basket.values()])

        # Collate into order_depth list format
        assembled_ask_depth = [[ask, assembled_prices_per_basket.count(ask)] for ask in sorted(set(assembled_prices_per_basket))]
        basket_bid_depth = [list(item) for item in state.order_depths[basket].buy_orders.items()]


        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_ask_depth) > 0 and len(basket_bid_depth) > 0:

            best_assembled_ask, best_assembled_ask_amount = assembled_ask_depth[0] # Hopefully low
            best_basket_bid, best_basket_bid_amount = basket_bid_depth[0] # Hopefully high

            best_spread = best_basket_bid - best_assembled_ask # Hopefully very positive
            if best_spread > self.MIN_SPREAD_TO_OPEN_ARBITRAGE[basket]:
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


        # Order to sell baskets - volume is negative
        orders: List[Order] = []
        orders.append(Order(basket, self.worst_price_for_volume(state, basket, order_volume), order_volume))

        # Orders to buy components
        for component in self.BASKET_CONTENTS[basket].keys():
            component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
            orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders
    
    def close_long_arbitrage(self, state: TradingState, basket: Symbol):
        orders: List[Order] = []
        position = state.position[basket]

        print(f"Closing position {position} of {basket}")

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows
        component_asks_per_basket = self.component_asks_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(asks) for asks in component_asks_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket[i] = sum([asks[i] for asks in component_asks_per_basket.values()])

        # Collate into order_depth list format
        assembled_ask_depth = [[ask, assembled_prices_per_basket.count(ask)] for ask in sorted(set(assembled_prices_per_basket))]
        basket_bid_depth = [list(item) for item in state.order_depths[basket].buy_orders.items()]

        # Find max volume to trade within position closing spread
        max_order_volume = 0

        while len(assembled_ask_depth) > 0 and len(basket_bid_depth) > 0:

            best_assembled_ask, best_assembled_ask_amount = assembled_ask_depth[0] # Hopefully low
            best_basket_bid, best_basket_bid_amount = basket_bid_depth[0] # Hopefully high

            best_spread = best_basket_bid - best_assembled_ask # Hopefully very positive
            if best_spread > - self.MAX_SPREAD_TO_CLOSE[basket]:
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

        order_volume = max(max_order_volume, - position)


        # Order to sell baskets
        orders: List[Order] = []
        orders.append(Order(basket, self.worst_price_for_volume(state, basket, order_volume), order_volume))

        # Orders to buy components
        for component in self.BASKET_CONTENTS[basket].keys():
            component_volume = - order_volume * self.BASKET_CONTENTS[basket][component]
            orders.append(Order(component, self.worst_price_for_volume(state, component, component_volume), component_volume))

        return orders
    
    def close_short_arbitrage(self, state: TradingState, basket: Symbol):
        orders: List[Order] = []
        position = state.position[basket]
        print(f"Closing position {position} of {basket}")

        # Make a list of how expensive each component will be for each subsequent basket (prices get worse as we progress through the order book)
        # E.g. how much will first 6 croissants cost, then next 6, etc as far as market allows

        component_bids_per_basket = self.component_bids_per_basket(state, basket)

        # Get max number of baskets from shortest supply component
        max_baskets = min([len(bids) for bids in component_bids_per_basket.values()])

        # Get total prices for each assembled basket equivalent
        assembled_prices_per_basket = []

        for i in range(max_baskets):
            assembled_prices_per_basket[i] = sum([bids[i] for bids in component_bids_per_basket.values()])

        # Collate into order_depth list format
        assembled_bid_depth = [[bid, assembled_prices_per_basket.count(bid)] for bid in sorted(set(assembled_prices_per_basket))]
        basket_ask_depth = [list(item) for item in state.order_depths[basket].sell_orders.items()]


        # Find max volume to trade with favorable spread
        max_order_volume = 0

        while len(assembled_bid_depth) > 0 and len(basket_ask_depth) > 0:

            best_assembled_bid, best_assembled_bid_amount = assembled_bid_depth[0] # Hopefully high
            best_basket_ask, best_basket_ask_amount = basket_ask_depth[0] # Hopefully low

            best_spread = best_basket_ask - best_assembled_bid # Hopefully very negative
            if best_spread < self.MAX_SPREAD_TO_CLOSE[basket]:
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

        order_volume = min(max_order_volume, position)

        # Order to buy baskets
        orders: List[Order] = []
        orders.append(Order(basket, self.worst_price_for_volume(state, basket, order_volume), order_volume))

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

            for bid in bid_per_basket:
                print(f"Bid per basket {basket} on {component}: {bid}")
        return component_bids_per_basket


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

            for ask in ask_per_basket:
                print(f"Ask per basket {basket} on {component}: {ask}")
        return component_asks_per_basket

    def arbitrage_volume_limit(self, state: TradingState, basket: Symbol):
        component_buy_limits = []
        component_sell_limits = []
        for product in self.BASKET_CONTENTS[basket].keys():
            component_buy_limits.append((self.LIMITS[product] - state.position[product]) // self.BASKET_CONTENTS[basket][product])
            component_sell_limits.append((self.LIMITS[product] + state.position[product]) // self.BASKET_CONTENTS[basket][product])

        basket_buy_limit = self.LIMITS[basket] - state.position[basket]
        basket_sell_limit = (self.LIMITS[basket] + state.position[basket])
        max_buy_volume = min(component_sell_limits + [basket_buy_limit])
        max_sell_volume = min(component_buy_limits + [basket_sell_limit])
                              
        return [max_buy_volume, -max_sell_volume]

    # The least favourable price to be paid when requesting to trade a volume against the order book. Volume is +ve to buy and -ve to sell
    def worst_price_for_volume(self, state: TradingState, product: Symbol, volume: int):

        if volume > 0:
            order_depth = [list(item) for item in state.order_depths[product].sell_orders.items()]
            if volume > -sum([order_depth[i][1] for i in range(len(order_depth))]):
                raise ValueError("Not enough supply in order book.")
            worst_price = 0
            while volume > 0:
                worst_price = order_depth[0][0]
                volume += order_depth[0][1]
            return worst_price
        elif volume < 0:
            order_depth = [list(item) for item in state.order_depths[product].buy_orders.items()]
            if volume < -sum([order_depth[i][1] for i in range(len(order_depth))]):
                raise ValueError("Not enough supply in order book.")
            worst_price = 0
            while volume < 0:
                worst_price = order_depth[0][0]
                volume += order_depth[0][1]
        else:
            print("Requested price for volume 0")
            return 0

