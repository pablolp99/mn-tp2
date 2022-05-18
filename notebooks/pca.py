import pandas as pd
import plotly.express as px

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.decomposition as skld

import sys
import pathlib
import logging

sys.path.insert(1, f"{pathlib.Path(__file__).parent.resolve()}/../build")
from mnpkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42

class PCA(BaseEstimator):
    def __init__(self, alpha: int = 5, epsilon = 0.00001):
        """Constructor

        Parameters
        ----------
        
        """
        self.model = PCACpp(alpha, epsilon)


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

        self.model.fit(imgs)


    def transform(self, X: pd.DataFrame):
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        transformed = self.model.transform(imgs)
        return transformed


    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)


    def get_model(self):
        return self.model


if __name__ == '__main__':
    pca_ours = PCA(3)

    logger.info("Loading CSV")
    df = pd.read_csv("../data/train.csv")[:1000]

    # logger.info("Training")
    # pca.fit(X)

    logger.info("Transforming")
    transformed = pca_ours.fit_transform(df.drop(columns='label'))

    df['pca_0_ours'] = transformed[:,0]
    df['pca_1_ours'] = transformed[:,1]
    df['pca_2_ours'] = transformed[:,2]

    logger.info("Plotting")

    df['label'] = df['label'].astype(str)
    fig = px.scatter_3d(df, x="pca_0_ours", y="pca_1_ours", z="pca_2_ours", color="label", title='Ours')
    fig.show()


    pca_sk = skld.PCA(n_components=3)
    transformed_sk = pca_sk.fit_transform(df.drop(columns='label'))
    df['pca_0_sk'] = transformed_sk[:,0]
    df['pca_1_sk'] = transformed_sk[:,1]
    df['pca_2_sk'] = transformed_sk[:,2]
    fig = px.scatter_3d(df, x="pca_0_sk", y="pca_1_sk", z="pca_2_sk", color="label", title="Sklearn")
    fig.show()