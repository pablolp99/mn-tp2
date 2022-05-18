import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split

import logging
import sys
import time

sys.path.insert(1, "../build")
from metnum_pkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42


class KNNClassifier(BaseEstimator):
    def __init__(self, k: int = 5):
        """Constructor

        Parameters
        ----------
        k : int, optional, by default 5.
        """
        self.k = k
        self.model = KNNClassifierCpp(k)

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

        self.model.fit(imgs, labels)

    def predict(self, X: pd.DataFrame):
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

        pred = self.model.predict(imgs)
        return pd.Series(pred).astype(int)

    def get_params(self, deep=True):
        return {"k": self.k}

    def get_model(self):
        return self.model


if __name__ == "__main__":
    start_time = time.time()

    logger.info("Loading CSV")
    df = pd.read_csv("../data/train.csv")[:5000]

    y = df["label"]
    X = df.drop(columns="label")

    logger.info("Splitting")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )

    logger.info(f"X,y train: {len(X_train)} - X,y test: {len(X_test)}")


    logger.info("Running cross-validation")
    scores = cross_val_score(
        KNNClassifier(10),
        X,
        y,
        cv=5,
        scoring=make_scorer(accuracy_score),
    )

    logger.info(scores)
