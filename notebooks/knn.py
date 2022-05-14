import pandas as pd
import sys

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

sys.path.insert(1, '/home/pablo/UBA/comp2022/MN/mn-tp2/build')
from mnpkg import *

RANDOM_STATE = 42

class KNNClassifier(BaseEstimator):
    def __init__(self, k_neighbors: int = 5):
        """Constructor

        Parameters
        ----------
        k_neighbors : int, optional, by default 5.
        """
        self.k_neighbors = k_neighbors
        self.model = KNNClassifierCpp(k_neighbors)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores de entrenamiento.
        y : pd.Series
            Series con las etiquetas de entrenamiento.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0
        assert len(X) == len(y)
        
        labels = y.tolist()
        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        self.model.fit(imgs, labels, len(X), len(X.iloc[0]))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predictor

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con vectores a predecir.

        Returns
        -------
        pd.Series
            Etiquetas predecidas para cada vector.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        pred = self.model.predict(imgs, len(imgs), len(imgs.iloc[0]))
        return pd.Series(pred)


if __name__ == '__main__':
    knn = KNNClassifier(3)

    breakpoint()

    df = pd.read_csv("../data/train.csv")

    y = df["label"]
    X = df.drop(columns='label')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    knn.fit(X_train, y_train)

    print(knn)