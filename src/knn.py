import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
# from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support
# from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

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
        self.k = k
        self.model = KNNClassifierCpp(k)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert len(X) > 0
        assert len(X[0]) > 0
        assert len(X) == len(y)
        self.model.fit(X.tolist(), y.tolist())

    def predict(self, X: np.ndarray):
        assert len(X) > 0
        assert len(X[0]) > 0

        pred = self.model.predict(X.tolist())
        return pd.Series(pred).astype(int)

    def get_model(self):
        return self.model

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, k):
        if k is not None:
            self.model = KNNClassifierCpp(k)
        return self
    
    def __repr__(self):
        return repr(f"k: {self.k} - model: {self.model}")

if __name__ == "__main__":

    train = pd.read_csv("../data/train.csv")

    X = train.drop(columns='label').to_numpy()
    y = train.label.to_numpy()


    _knn = KNNClassifier(2)
    _knn.fit(X[4200:41999], y[4200:41999])

    _knn_st = time.time()
                
    _knn.predict(X[:4200])

    _knn_et = time.time()

    print(_knn_et - _knn_st)