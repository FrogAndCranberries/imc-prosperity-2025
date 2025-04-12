# This code is the sole property and creation of the cuckboy programmer Ediz Ucar and the legendary team that carried him all the way


from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
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
        # print("traderData: " + state.traderData)
        # print("Observations: " + str(state.observations))
        result = {}

        if state.traderData != "":
            traderData : dict[str,list[int]] = jsonpickle.decode(state.traderData)
        else:
            traderData = {"RAINFOREST_RESIN":[],"KELP":[],"SQUID_INK":[]}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            productPrices = traderData[product]

                
    
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                
            else:
                continue

            midprice = sum([best_ask,best_bid])/2.0


            # print(midprice)

            
            productPrices += [midprice]

            if len(productPrices) > 1000:
                productPrices=productPrices[-1000:]

            avg_price = sum(productPrices)/len(productPrices)
            factor = .5

            top10 = avg_price + (max(productPrices[-50:]) - avg_price) * factor
            bot10 = avg_price - (avg_price - min(productPrices[-50:]))*factor

            if product == "RAINFOREST_RESIN":
                print(top10, bot10)



            acceptable_price = avg_price;  # Participant should calculate this value
            # print("Acceptable price : " + str(acceptable_price))
            # print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
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
            
            result[product] = orders
            traderData[product] = productPrices

    
    
        traderData = jsonpickle.encode(traderData) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData


