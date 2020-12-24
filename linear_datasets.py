from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv("train.csv")
a = a[["LotArea", "SalePrice"]]
x = a["LotArea"][: 50].values
y = a["SalePrice"][: 50].values

plt.scatter(x, y)
plt.show()
