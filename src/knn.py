import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

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


    def set_params(self, k):
        if k is not None:
            self.model = KNNClassifierCpp(k)
        return self


    def get_model(self):
        return self.model


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")[:5000]

    X = df.drop(columns="label")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Test size: {len(X_test)}")

    knn = KNNClassifier(15)
    knn.fit(X_train, y_train)

    start_time = time.time()
    pred = knn.predict(X_test)
    end_time = time.time()

    logger.info(f"Prediction time: {end_time-start_time}")

    logger.info(f"Accuracy: {accuracy_score(y_test, pred)}")

    # grid = {
    #     'k': [2,3,4,5,10,13,15,20]
    # }

    # knn_cv = GridSearchCV(
    #     KNNClassifier(),
    #     param_grid=grid,
    #     scoring=make_scorer(accuracy_score),
    #     n_jobs=-1,
    #     cv=10,
    #     verbose=2,
    # )

    # logger.info("Running GridSearch Cross-Validation for KNN")
    # logger.info("Using grid:")
    # logger.info(grid)

    # knn_cv.fit(X, y)

    # logger.info(knn_cv.best_estimator_)
    # logger.info(knn_cv.best_score_)
    # logger.info(knn_cv.best_params_)
    # logger.info(knn_cv.scorer_)
