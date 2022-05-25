import pandas as pd

from sklearn.base import BaseEstimator

import logging

import sys
sys.path.insert(1, "../build")
from metnum_pkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

class KNN_with_PCA_Classifier(BaseEstimator):
    def __init__(self, k: int = 5, alpha: int = 5, epsilon = 0.00001):
        """Constructor

        Parameters
        ----------
        k : int, optional, by default 5.
        """
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon

        self.knn_model = KNNClassifierCpp(k)
        self.pca_model = PCACpp(alpha, epsilon)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores de entrenamiento.
        y : pd.Series
            Series con las etiquetas de entrenamiento.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0
        assert len(X) == len(y)

        labels = y.tolist()
        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        self.knn_model.fit(imgs, labels)
        self.pca_model.fit(imgs)

        return self

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

        transformed = self.pca_model.transform(imgs)

        asdf = []
        for i in range(len(transformed)):
            asdf.append(transformed[i].tolist())

        # logger.info(f"vector transformed: {transformed}")
        logger.info(f"vector asdf: {type(asdf)}")

        pred = self.knn_model.predict(asdf)
        return pd.Series(pred).astype(int)


    def get_params(self, deep=True):
        return {"k": self.k, 
                "alpha": self.alpha, 
                "epsilon": self.epsilon
        }


    def set_params(self, k, alpha, epsilon):
        if k is not None:
            self.k = k
        if alpha is not None:
            self.alpha = alpha
        if epsilon is not None:
            self.epsilon = epsilon
        
        self.knn_model = KNNClassifierCpp(self.k)
        self.pca_model = PCACpp(self.alpha, self.epsilon)

        return self