import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DAY = 0

PRODUCT = "KELP"

csv_path = f"./round-1-island-data-bottle/prices_round_1_day_{DAY}.csv"

df = pd.read_csv(csv_path, delimiter=';')

print(df.head())
print(df.keys())
midprice = df[(df["product"] == PRODUCT)]["mid_price"]
print(len(midprice))

midprice.plot()
plt.savefig('plot.png')
plt.show()