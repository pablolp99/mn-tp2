import numpy as np

from sklearn.base import BaseEstimator

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