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


def nepoznata_funkcija(x):
    return np.linalg.norm(x) ** 2 - np.linalg.norm(x) - 2 * np.dot(x, x + 5.)**2 + 1

BROJ_TACAKA = 1000
BROJ_TACAKA_ZA_UCENJE = 900

blob, _ = make_blobs(n_samples=BROJ_TACAKA, centers=3)
X_train = blob[:BROJ_TACAKA_ZA_UCENJE]
y_train = np.array([nepoznata_funkcija(x) for x in X_train])

X_test = blob[BROJ_TACAKA_ZA_UCENJE:]
y_test = np.array([nepoznata_funkcija(x) for x in X_test])

df = DataFrame(dict(x=X_train[:, 0], y=X_train[:, 1], value=y_train))
df_test = DataFrame(dict(x=X_test[:, 0], y=X_test[:, 1], value=y_test))

'''
dist = DistanceMetric.get_metric('euclidean')
rastojanja = pd.DataFrame(dist.pairwise(df.value.values.reshape(-1, 1)))
'''

def nadji_tacke_u_okolini(x, y, epsilon, df):
    tacke_u_okolini = []
    prva_tacka = np.array([x, y])
    for tacka in df.itertuples():
        druga_tacka = np.array([getattr(tacka, "x"), getattr(tacka, "y")])
        if np.linalg.norm(prva_tacka - druga_tacka) < epsilon:
            tacke_u_okolini.append(getattr(tacka, "Index"))
    return tacke_u_okolini


def predvidi_u_tacki(x, y, epsilon, delta, df):
    ž_tacke_u_okolini = nadji_tacke_u_okolini(x, y, delta, df)
    vrednosti_u_okolini = df.iloc[ž_tacke_u_okolini].value.values

    while len(vrednosti_u_okolini) == 0:
        delta = delta + epsilon
        ž_tacke_u_okolini = nadji_tacke_u_okolini(x, y, delta, df)
        vrednosti_u_okolini = df.iloc[ž_tacke_u_okolini].value.values
    predikcija = np.average(vrednosti_u_okolini)
    return predikcija


EPSILON = 0.3  # najmanja razlika u vrednosti funkcije koja se uci da bi se ostvarila žveza.
DELTA = 2.  # Okolina oko nove test tačke u kojoj se traži ž-struktura za indukovanje informacija. Možda ovo učiti, za početak?

'''
sta_povezati = rastojanja.values < EPSILON

print('Kreiranje podataka vezanih za linije u ž-geometriji...')
linije = []  # sadrzi parove povezanih koordinata
linije_indeksi = []  # sadrzi indekse svih tacaka povezanih za i-tom tackom u ž-geometriji
for i in tqdm(range(sta_povezati.shape[0])):
    linije_indeksi.append([])
    for j in range(sta_povezati.shape[1]):
        if sta_povezati[i, j]:
            linije.append([df.iloc[i][['x', 'y']], df.iloc[j][['x', 'y']]])
            linije_indeksi[i].append(nadji_tacke_u_okolini(df.iloc[i]['x'], df.iloc[i]['y'], DELTA, df))
'''

for EPSILON in [0.001, 0.01, 0.1]:
    for DELTA in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 2., 5.]:
        greske = []
        for test_primer in df_test.itertuples():
            tacka = np.array([getattr(test_primer, "x"), getattr(test_primer, "y")])
            prava_vrednost = getattr(test_primer, "value")
            predikcija = predvidi_u_tacki(tacka[0], tacka[1], EPSILON, DELTA, df)
            greska = (predikcija - prava_vrednost) ** 2
            greske.append(greska)
        y = y_test
        pred_y = [predvidi_u_tacki(tacka[0], tacka[1], EPSILON, DELTA, df) for tacka in X_test]
        print('MSE:', sklearn.metrics.mean_squared_error(y, pred_y))
        print(EPSILON, DELTA, 'Finalna prosecna greska:', np.average(greske))

# Drugi modeli na ovom problemu
gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X_train, y_train)
rezultat = gb.predict(X_test)
print('Greska za Pojacane Gradijente sa 100 drveta:', mean_squared_error(y_test, rezultat))
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=5)
rf.fit(X_train, y_train)
rezultat = rf.predict(X_test)
print('Greska za Random Forest sa 5 drveta:', mean_squared_error(y_test, rezultat))
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rezultat = rf.predict(X_test)
print('Greska za Random Forest sa 100 drveta:', mean_squared_error(y_test, rezultat))
from sklearn.neural_network import MLPRegressor
dnn = MLPRegressor((200, 100, 10), max_iter=50000)
dnn.fit(X_train, y_train)
rezultat = dnn.predict(X_test)
print('Greska za DNN [200, 100, 10]:', mean_squared_error(y_test, rezultat))
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
rezultat = knn.predict(X_test)
print('Greska za knn sa 5 okolina:', mean_squared_error(y_test, rezultat))
knn = KNeighborsRegressor(200)
knn.fit(X_train, y_train)
rezultat = knn.predict(X_test)
print('Greska za knn sa 200 okolina:', mean_squared_error(y_test, rezultat))
knn = KNeighborsRegressor(500)
knn.fit(X_train, y_train)
rezultat = knn.predict(X_test)
print('Greska za knn sa 500 okolina:', mean_squared_error(y_test, rezultat))



figure = plt.figure(figsize=(27, 9))
ax = plt.axes()
ax.scatter(df.x, df.y, c=df.value, cmap='gray')
lc = mc.LineCollection(linije, linewidths=1)
ax.add_collection(lc)

plt.tight_layout()
plt.show()
