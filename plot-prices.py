import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Variables
DAY = 0
PRODUCT = "KELP"
SMOOTHING_PERIOD = 100
SMOOTHING_SIGMA = 20

# Load dataframe
csv_path = f"./round-1-island-data-bottle/prices_round_1_day_{DAY}.csv"

df = pd.read_csv(csv_path, delimiter=';')

# Plot product midprice
midprice = df[(df["product"] == PRODUCT)]["mid_price"]
x = np.linspace(-SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD)
smoothing_kernel = np.exp(-x ** 2 / (2 * SMOOTHING_SIGMA ** 2))
smoothing_kernel /= smoothing_kernel.sum()

midprice_smooth = np.convolve(midprice.values, smoothing_kernel, mode="valid")

plt.plot(midprice_smooth)
plt.savefig('plots/plot.png')
plt.show()