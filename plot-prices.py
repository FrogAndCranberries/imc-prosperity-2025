from matplotlib import pyplot
import pandas as pd
import numpy as np

DAY = 0

PRODUCT = "kelp"

csv_path = f"./round-1-island-data-bottle/prices_round_1_day_{DAY}.csv"

df = pd.read_csv(csv_path)

print(df.head())

df["product"] == PRODUCT