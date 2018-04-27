import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from žModeli import predŽRegressor, testiraj_na_drugim_modelima
from sklearn.model_selection import GridSearchCV


def nepoznata_funkcija(x):
    return np.linalg.norm(x) ** 2 - np.linalg.norm(x) - 2 * np.dot(x, x + 5.) ** 2 + 1


if __name__ == '__main__':

    GRID_SEARCH = True
    BROJ_TACAKA = 2000
    BROJ_TACAKA_ZA_UCENJE = 1900

    blob, _ = make_blobs(n_samples=BROJ_TACAKA, n_features=5, centers=3)
    X_train = blob[:BROJ_TACAKA_ZA_UCENJE]
    y_train = np.array([nepoznata_funkcija(x) for x in X_train])

    X_test = blob[BROJ_TACAKA_ZA_UCENJE:]
    y_test = np.array([nepoznata_funkcija(x) for x in X_test])

    parametri = {'epsilon': 1., 'delta': 2.}

    if GRID_SEARCH:
        tuned_parameters = {"epsilon": [0.01, 0.5, 1., 2.], "delta": [0.001, 0.01, 0.1, 1., 2., 5.]}
        gs = GridSearchCV(predŽRegressor(), tuned_parameters, n_jobs=10, verbose=10)
        gs.fit(X_train, y_train)
        print('Najbolji parametri:', gs.best_params_)
        parametri = gs.best_params_

    model = predŽRegressor(parametri['epsilon'], parametri['delta'])
    model.fit(X_train, y_train)
    rezultat = model.predict(X_test)
    greska = model.score(X_test, y_test)
    print('Greska:', greska)

    testiraj_na_drugim_modelima(X_train, y_train, X_test, y_test)