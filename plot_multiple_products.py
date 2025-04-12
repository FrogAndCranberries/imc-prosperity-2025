import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import get_midprice, midprice_smoothing

# Variables
DAY = 1
PRODUCTS = ["SQUID_INK", "KELP"]
SMOOTHING = False
SMOOTHING_PERIOD = 3
SMOOTHING_SIGMA = 1


# Plot product midprice
for product in PRODUCTS:
    midprice = get_midprice(DAY, product)
    if midprice is None:
        raise Exception(f"Product history unknown on day {DAY}")
    if SMOOTHING:
        midprice = midprice_smoothing(midprice, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

    plt.plot(midprice, label=product)
    plt.title(f"{PRODUCTS}, DAY {DAY}")

plt.legend()
plt.savefig(f"plots/{PRODUCTS}_D{DAY}.png")
plt.show()