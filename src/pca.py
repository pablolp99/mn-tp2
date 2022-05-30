import numpy as np

from sklearn.base import BaseEstimator

import sys
import pathlib
import logging

sys.path.insert(1, f"{pathlib.Path(__file__).parent.resolve()}/../build")
from metnum_pkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42


class PCA(BaseEstimator):
    def __init__(self, alpha: int = 5, epsilon=0.001):
        """Constructor

        Parameters
        ----------

        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = PCACpp(alpha, epsilon)

    def fit(self, X: np.ndarray):
        assert len(X) > 0
        assert len(X[0]) > 0

        self.model.fit(X.tolist())

    def transform(self, X: np.ndarray, truncate: int = 0):
        assert len(X) > 0
        assert len(X[0]) > 0

        return self.model.transform(X.tolist(), truncate)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def get_model(self):
        return self.model

    def __repr__(self):
        return repr(f"alpha: {self.alpha} - epsilon: {self.epsilon} - model: {self.model}")