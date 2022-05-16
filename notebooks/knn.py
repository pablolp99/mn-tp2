import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import logging
import sys
import time

sys.path.insert(1, '/home/pablo/UBA/comp2022/MN/mn-tp2/build')
from mnpkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


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


    def get_params(self):
        return { "k": self.k }


    def get_model(self):
        return self.model


if __name__ == '__main__':
    start_time = time.time()
    
    knn = KNNClassifier(15)
    
    logger.info("Loading CSV")
    df = pd.read_csv("../data/train.csv")

    y = df["label"]
    X = df.drop(columns='label')

    logger.info("Splitting")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=RANDOM_STATE)

    logger.info(f"X,y train: {len(X_train)} - X,y test: {len(X_test)}")

    logger.info("Training")
    knn.fit(X_train, y_train)
    
    logger.info("Predicting")
    
    predict_s_time = time.time()
    results = knn.predict(X_test)
    end_time = time.time()

    logger.info(f"--- Total compute time: {end_time - start_time} seconds ---" )
    logger.info(f"--- Prediction compute time: {end_time - predict_s_time} seconds ---" )

    logger.info("Global accuracy: %f" % (accuracy_score(y_test, results)))
    
    metrics = zip(["precision", "recall", "fbeta_score", "support"], [i.tolist() for i in precision_recall_fscore_support(y_test, results)])

    for r in metrics:
        logger.info(f"{r[0]}: {r[1]}")