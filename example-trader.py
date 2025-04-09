from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle


# class TraderData:
#     def __init__(self, average_price:int, num_rounds:int):
#         self.average_price = average_price
        # self.num_rounds = num_rounds

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        
        traderData : dict[str,list[int]] = jsonpickle.decode(traderData)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            curr_price = sum([x.price for x in state.market_trades[product]])/len(state.market_trades[product])
            productPrices = traderData[product] + [curr_price]

            
            avg_price = sum(productPrices ) / len(productPrices)



            acceptable_price = avg_price;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
            traderData[product] = productPrices

    
    
        traderData = jsonpickle.encode(traderData) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
