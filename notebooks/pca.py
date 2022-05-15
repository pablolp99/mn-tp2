import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
import pathlib
import logging

sys.path.insert(1, f"{pathlib.Path(__file__).parent.resolve()}/../build")
from mnpkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42

class PCA(BaseEstimator):
    def __init__(self, k_neighbors: int = 5):
        """Constructor

        Parameters
        ----------
        k_neighbors : int, optional, by default 5.
        """
        self.k_neighbors = k_neighbors
        self.model = PCACpp(k_neighbors)

    def fit(self, X: pd.DataFrame):
        """Fit

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores de entrenamiento.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        breakpoint()

        self.model.fit(imgs)

    def transform(self, X: pd.DataFrame):
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

        pred = self.model.predict(imgs, len(imgs), len(imgs[0]))
        return pd.Series(pred).astype(int)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    pca = PCA(1)

    logger.info("Loading CSV")
    df = pd.read_csv("data/train.csv")

    y = df["label"]
    X = df.drop(columns='label')

    logger.info("Splitting")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=RANDOM_STATE)

    logger.info(f"X,y train: {len(X_train)} - X,y test: {len(X_test)}")

    logger.info("Training")
    pca.fit(X_test)

    # logger.info("Predicting")
    # results = knn.predict(X_test)
    #
    # logger.info("Accuracy: %f" % (accuracy_score(y_test, results)))