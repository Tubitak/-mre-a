import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from itertools import combinations
from sklearn.neighbors import DistanceMetric
import pylab as pl
from matplotlib import collections as mc
from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/forestfires.csv')
data = data[data.area > 0]

X_train, X_test, y_train, y_test = train_test_split(data[['wind', 'ISI']].values, data['area'].values, test_size=0.1)

df = DataFrame(dict(x=X_train[:, 0], y=X_train[:, 1], value=y_train))

dist = DistanceMetric.get_metric('euclidean')
rastojanja = pd.DataFrame(dist.pairwise(df.value.values.reshape(-1, 1)))


# Pojacani gradijenti na ovom problemu
gb = RandomForestRegressor()
gb.fit(X_train, y_train)
rezultat = gb.predict(X_test)
print(mean_squared_error(y_test, rezultat))

figure = plt.figure(figsize=(27, 9))
ax = plt.axes()

EPSILON = 0.1

sta_povezati = rastojanja.values < EPSILON

linije = []
for i in range(sta_povezati.shape[0]):
    for j in range(sta_povezati.shape[1]):
        if sta_povezati[i, j]:
            linije.append([df.iloc[i][['x', 'y']], df.iloc[j][['x', 'y']]])

ax.scatter(df.x, df.y, c=df.value, cmap='gray')
lc = mc.LineCollection(linije, linewidths=1)
ax.add_collection(lc)

plt.tight_layout()
plt.show()
