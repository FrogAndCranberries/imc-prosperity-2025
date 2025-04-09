import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

class Backtest():
    def __init__(self, algo,
                 days = [-2, -1, 0]):
        self.algo = algo
        self.load_data(days)

    def load_data(self, days: List):
        PRICE_CSV_PREFIX = "backtest-data/prices_round_1_day_"
        TRADE_CSV_PREFIX = "backtest-data/trades_round_1_day_"
        days.sort()
        price_csvs = [PRICE_CSV_PREFIX + f"{day}.csv" for day in days]
        trade_csvs = [TRADE_CSV_PREFIX + f"{day}.csv" for day in days]
        self.orders = pd.concat([pd.read_csv(price_csv) for price_csv in price_csvs], ignore_index=True)
        self.trades = pd.concat([pd.read_csv(trade_csv) for trade_csv in trade_csvs], ignore_index=True)

    def collate_orders(self):
        pass

    def backtest(self):
        pass


