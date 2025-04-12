import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import get_midprice, get_full_midprice, midprice_smoothing

# Variables
PRODUCT = "PICNIC_BASKET2"

SMOOTHING = False
SMOOTHING_PERIOD = 120
SMOOTHING_SIGMA = 20

midprice_long = get_full_midprice(PRODUCT)

if SMOOTHING:
    midprice_long = midprice_smoothing(midprice_long, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

plt.plot(midprice_long)

plt.title(f"FULL HISTORY OF {PRODUCT}")
plt.savefig(f"plots/MULTIDAY_{PRODUCT}.png")
plt.show()