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
from tqdm import tqdm


blob, _ = make_blobs(n_samples=1500, centers=1)
X_train = blob[:1350]
y_train = np.array([np.linalg.norm(x)**2 - np.linalg.norm(x) for x in X_train])

X_test = blob[1350:]
y_test = np.array([np.linalg.norm(x)**2 - np.linalg.norm(x) for x in X_test])

df = DataFrame(dict(x=X_train[:,0], y=X_train[:,1], value=y_train))


dist = DistanceMetric.get_metric('euclidean')
rastojanja = pd.DataFrame(dist.pairwise(df.value.values.reshape(-1, 1)))


'''
# Pojacani gradijenti na ovom problemu
for _ in range(10):
    gb = RandomForestRegressor()
    gb.fit(X_train, y_train)
    rezultat = gb.predict(X_test)
    print(mean_squared_error(y_test, rezultat))
'''

def nadji_tacke_u_okolini(x, y, epsilon):
    tacke_u_okolini = []
    prva_tacka = np.array([x, y])
    for tacka in df.itertuples():
        druga_tacka = np.array([getattr(tacka, "x"), getattr(tacka, "y")])
        if np.linalg.norm(prva_tacka - druga_tacka) < epsilon:
            tacke_u_okolini.append(getattr(tacka, "Index"))
    return tacke_u_okolini




EPSILON = 0.3  # najmanja razlika u vrednosti funkcije koja se uci da bi se ostvarila žveza.
DELTA = 2.  # Okolina oko nove test tačke u kojoj se traži ž-struktura za indukovanje informacija. Možda ovo učiti, za početak?

sta_povezati = rastojanja.values < EPSILON

print('Kreiranje podataka vezanih za linije u ž-geometriji...')
linije = []  # sadrzi parove povezanih koordinata
linije_indeksi = []  # sadrzi indekse svih tacaka povezanih za i-tom tackom u ž-geometriji
for i in tqdm(range(sta_povezati.shape[0])):
    linije_indeksi.append([])
    for j in range(sta_povezati.shape[1]):
        if sta_povezati[i, j]:
            linije.append([df.iloc[i][['x', 'y']], df.iloc[j][['x', 'y']]])
            linije_indeksi[i].append(nadji_tacke_u_okolini(df.iloc[i]['x'], df.iloc[i]['y'], DELTA))


figure = plt.figure(figsize=(27, 9))
ax = plt.axes()
ax.scatter(df.x, df.y, c=df.value, cmap='gray')
lc = mc.LineCollection(linije, linewidths=1)
ax.add_collection(lc)

plt.tight_layout()
plt.show()
