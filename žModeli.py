from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
import numpy as np


def testiraj_na_drugim_modelima(X_train, y_train, X_test, y_test):
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


class predŽRegressor(BaseEstimator):
    """Prva verzija Ž klasifikatora"""

    def __init__(self, epsilon=0.01, delta=0.2):
        """
        Called when initializing the classifier
        """
        self.epsilon = epsilon
        self.delta = delta

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.

        :param X: numpy array of training data, float
        :param y: numpy array of labels, int
        """

        assert (type(self.epsilon) == float), "epsilon parameter must be float"
        assert (self.epsilon > 0), "epsilon parameter must be > 0"
        assert (type(self.delta) == float), "delta parameter must be float"
        assert (self.delta > 0), "delta parameter must be > 0"
        # assert (len(X) == 20), "X must be list with numerical values." TODO: proveriti jesu li kolone numericke

        self.X_train = X
        self.y_train = y

        return self

    def _nadji_tacke_u_okolini(self, x, poluprecnik_lopte):
        """
        Vraca indekse tacaka iz self.X u okolini od x poluprecnika poluprecnik_lopte
        :param x: tacka
        :param poluprecnik_lopte: poluprecnik lopte gde se traze druge tacke
        :return: indeksi u X tacaka u okolini
        """
        tacke_u_okolini = []
        prva_tacka = np.array(x)
        for index, druga_tacka in enumerate(self.X_train):
            if np.linalg.norm(prva_tacka - druga_tacka) < poluprecnik_lopte:
                tacke_u_okolini.append(index)
        return tacke_u_okolini

    def predict(self, X):

        '''
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        '''

        predikcije = np.array([])

        for tacka in X:
            tacke_u_okolini = self._nadji_tacke_u_okolini(tacka, self.delta)
            vrednosti_u_okolini = self.y_train[tacke_u_okolini]
            delta = self.delta
            while len(vrednosti_u_okolini) == 0:
                delta = delta + self.epsilon
                tacke_u_okolini = self._nadji_tacke_u_okolini(tacka, delta)
                vrednosti_u_okolini = self.y_train[tacke_u_okolini]

            predikcija = np.average(vrednosti_u_okolini)
            predikcije = np.append(predikcije, [predikcija], axis=0)

        return predikcije

    def score(self, X, y):
        """
        Vraca MSE.
        :param X: Ulazni podaci
        :param y: Tacne vrednosti
        :return: srednja kvadratna greska
        """
        greska = mean_squared_error(y, self.predict(X))
        return greska
