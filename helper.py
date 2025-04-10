import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_midprice(day: int, product: str) -> list | None:
    """
    Returns list of midprices of a product in a given day, return None if not available
    """
    # Load dataframe searching in all rounds
    df = None
    for round in [4, 3, 2, 1]:
        try:
            csv_path = f"./data/round-{round}-island-data-bottle/prices_round_{round}_day_{day}.csv"
            df = pd.read_csv(csv_path, delimiter=';')
            break
        except FileNotFoundError:
            continue

    if df is None:
        return None

    midprice = list(df[(df["product"] == product)]["mid_price"])
    if len(midprice) == 0:
        return None
    else:
        return midprice


def get_full_midprice(product: str) -> list:
    midprice_long = []
    for day in [-2, -1, 0, 1, 2, 3, 4]:
        # Plot product midprice
        midprice = get_midprice(day, product)
        if midprice is None:
            continue

        midprice_long += list(midprice)

    return midprice_long


def midprice_smoothing(midprice: list, smoothing_period: int, smoothing_sigma: float) -> list:
    x = np.linspace(-smoothing_period // 2, smoothing_period // 2, smoothing_period)
    smoothing_kernel = np.exp(-x ** 2 / (2 * smoothing_sigma ** 2))
    smoothing_kernel /= smoothing_kernel.sum()

    return list(np.convolve(midprice, smoothing_kernel, mode="valid"))


if __name__ == "__main__":
    # Quick unit testing

    # get_midprice
    assert(get_midprice(-2, "DJEMBES") is None)
    assert (len(get_midprice(0, "DJEMBES")) == 10000)

    # midprice_smoothing

    print("No assertion error")