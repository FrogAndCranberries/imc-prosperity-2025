import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import get_midprice, get_full_midprice, midprice_smoothing

# Variables
BASKET = "PICNIC_BASKET1"

SMOOTHING = False
SMOOTHING_PERIOD = 120
SMOOTHING_SIGMA = 20

# Basket plot
midprice_long_basket = get_full_midprice(BASKET)
if SMOOTHING:
    midprice_long_basket = midprice_smoothing(midprice_long_basket, SMOOTHING_PERIOD, SMOOTHING_SIGMA)
plt.plot(midprice_long_basket, label=f"{BASKET} price")

# Jams
midprice_long_jams = get_full_midprice("JAMS")
if SMOOTHING:
    midprice_long_jams = midprice_smoothing(midprice_long_jams, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

# Croissants
midprice_long_cro = get_full_midprice("CROISSANTS")
if SMOOTHING:
    midprice_long_cro = midprice_smoothing(midprice_long_cro, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

# Djembes
midprice_long_djembes = get_full_midprice("DJEMBES")
if SMOOTHING:
    midprice_long_djembes = midprice_smoothing(midprice_long_djembes, SMOOTHING_PERIOD, SMOOTHING_SIGMA)

if BASKET == "PICNIC_BASKET1":
    midprice_long_together = 6*np.array(midprice_long_cro) + 3*np.array(midprice_long_jams) + np.array(midprice_long_djembes)
elif BASKET == "PICNIC_BASKET2":
    midprice_long_together = 4*np.array(midprice_long_cro) + 2*np.array(midprice_long_jams)

plt.plot(midprice_long_together, label=f"Items needed price")


plt.title(f"Worthiness of {BASKET}")
plt.legend()
plt.savefig(f"plots/WORTHINESS_{BASKET}.png")
plt.show()