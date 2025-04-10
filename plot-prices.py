import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Variables
ROUND = 2
DAY = 1
PRODUCT = "SQUID_INK"
SMOOTHING = False
SMOOTHING_PERIOD = 100
SMOOTHING_SIGMA = 20

# Load dataframe
csv_path = f"./data/round-{ROUND}-island-data-bottle/prices_round_{ROUND}_day_{DAY}.csv"
df = pd.read_csv(csv_path, delimiter=';')

# Plot product midprice
midprice = df[(df["product"] == PRODUCT)]["mid_price"]

if SMOOTHING:
    x = np.linspace(-SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD)
    smoothing_kernel = np.exp(-x ** 2 / (2 * SMOOTHING_SIGMA ** 2))
    smoothing_kernel /= smoothing_kernel.sum()

    midprice = np.convolve(midprice.values, smoothing_kernel, mode="valid")

plt.plot(midprice)
plt.title(f"{PRODUCT}, ROUND {ROUND}, DAY {DAY}")
plt.savefig(f"plots/{PRODUCT}_R{ROUND}_D{DAY}.png")
plt.show()