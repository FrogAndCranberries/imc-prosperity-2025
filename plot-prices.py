import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import get_midprice, get_full_midprice, midprice_smoothing

# Variables
ROUND = 2
DAY = 1
PRODUCT = "SQUID_INK"
SMOOTHING = False
SMOOTHING_PERIOD = 100
SMOOTHING_SIGMA = 20


midprice = get_midprice(DAY, PRODUCT)
if midprice is None:
    raise Exception(f"Product history unknown on day {DAY}")
if SMOOTHING:
    midprice = midprice_smoothing(midprice, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

plt.plot(midprice)
plt.title(f"{PRODUCT}, ROUND {ROUND}, DAY {DAY}")
plt.savefig(f"plots/{PRODUCT}_R{ROUND}_D{DAY}.png")
plt.show()