import pandas as pd

from sklearn.base import BaseEstimator

import logging

import sys
sys.path.insert(1, "../build")
from metnum_pkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

class KnnPCAClassifier(BaseEstimator):
    def __init__(self, k: int = 5, alpha: int = 5, epsilon = 0.00001):

        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon

        print(k, alpha, epsilon)
        self.knn_model = KNNClassifierCpp(k)
        self.pca_model = PCACpp(alpha, epsilon)

    def fit(self, X: pd.DataFrame, y: pd.Series):

        assert len(X) > 0
        assert len(X.iloc[0]) > 0
        assert len(X) == len(y)

        labels = y.tolist()
        imgs = X.to_numpy().tolist()

        print("Fitting PCA Model")
        self.pca_model.fit(imgs)
        
        print("Transforming data")
        pca_transformed = self.pca_model.transform(imgs)
        pca_transformed = pca_transformed.tolist()
        
        print("Fitting kNN Model")
        self.knn_model.fit(pca_transformed, labels)

        return self

    def predict(self, X: pd.DataFrame):
        
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        print("Transforming data")
        transformed = self.pca_model.transform(imgs)
        transformed = transformed.tolist()

        print("Predicting in kNN Model")
        pred = self.knn_model.predict(transformed)
        return pd.Series(pred).astype(int)


    def get_params(self, deep=True):
        return {
            "k": self.k, 
            "alpha": self.alpha, 
            "epsilon": self.epsilon
        }


    def set_params(self, k, alpha, epsilon):
        # if k is not None:
        self.k = k or self.k
        # if alpha is not None:
        self.alpha = alpha or self.alpha
        # if epsilon is not None:
        self.epsilon = epsilon or self.epsilon
        
        self.knn_model = KNNClassifierCpp(self.k)
        self.pca_model = PCACpp(self.alpha, self.epsilon)

        return self
    
    def __repr__(self):
        return repr(f"K: {self.k} - Alpha: {self.alpha} - Epsilon: {self.epsilon}")