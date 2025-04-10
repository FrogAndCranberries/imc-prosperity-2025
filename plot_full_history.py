import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Variables
PRODUCT = "CROISSANTS"

SMOOTHING = False
SMOOTHING_PERIOD = 3
SMOOTHING_SIGMA = 1

midprice_long = []
for DAY in [-2, -1, 0, 1, 2, 3, 4]:
    # Load dataframe
    df = None
    for round in [4, 3, 2, 1]:
        try:
            csv_path = f"./data/round-{round}-island-data-bottle/prices_round_{round}_day_{DAY}.csv"
            df = pd.read_csv(csv_path, delimiter=';')
            break
        except FileNotFoundError:
            continue

    if df is None:
        continue


    # Plot product midprice
    midprice = df[(df["product"] == PRODUCT)]["mid_price"]

    if SMOOTHING:
        x = np.linspace(-SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD // 2, SMOOTHING_PERIOD)
        smoothing_kernel = np.exp(-x ** 2 / (2 * SMOOTHING_SIGMA ** 2))
        smoothing_kernel /= smoothing_kernel.sum()

        midprice = np.convolve(midprice.values, smoothing_kernel, mode="valid")

    midprice_long += list(midprice)

plt.plot(midprice_long)

plt.title(f"FULL HISTORY OF {PRODUCT}")
plt.savefig(f"plots/MULTIDAY_{PRODUCT}.png")
plt.show()