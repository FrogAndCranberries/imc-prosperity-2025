import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from helper import get_midprice, get_full_midprice, midprice_smoothing

# Variables
PRODUCTS = ["KELP", "SQUID_INK", "RAINFOREST_RESIN",
            "JAMS", "CROISSANTS", "DJEMBES", "PICNIC_BASKET2", "PICNIC_BASKET2",
            "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"]

SMOOTHING = False
SMOOTHING_PERIOD = 120
SMOOTHING_SIGMA = 20

for PRODUCT in tqdm(PRODUCTS):
    midprice_long = get_full_midprice(PRODUCT)

    if SMOOTHING:
        midprice_long = midprice_smoothing(midprice_long, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

    plt.plot(midprice_long)

    plt.title(f"FULL HISTORY OF {PRODUCT}")
    plt.savefig(f"plots/MULTIDAY_{PRODUCT}.png")
    plt.show(block=False)
    plt.clf()

plt.show()