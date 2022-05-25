import pandas as pd
# import plotly.express as px

from sklearn.base import BaseEstimator
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#  import sklearn.decomposition as skld

import sys
import pathlib
import logging

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

sys.path.insert(1, f"{pathlib.Path(__file__).parent.resolve()}/../build")
from metnum_pkg import *

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


RANDOM_STATE = 42


class PCAOurs(BaseEstimator):
    def __init__(self, alpha: int = 5, epsilon=0.00001):
        """Constructor

        Parameters
        ----------
        alpha: int
            Cantidad de componentes principales a tomar
        epsilon: double
            Diferencia a evaluar para la decisión de eligibilidad de un autovector en el método de la potencia.

        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = PCACpp(self.alpha, self.epsilon)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores de entrenamiento.
        
        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        self.model.fit(imgs)

        return self

    def transform(self, X: pd.DataFrame):
        """Transform

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores del modelo a transformar.
        """
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        transformed = self.model.transform(imgs)
        return transformed

    def fit_transform(self, X: pd.DataFrame, y=None):
        """fit_transform

        Llama secuencialmente a fit y luego transform.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame con los vectores del modelo a transformar.

        y : Ignored
            Ignored.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, alpha=None, epsilon=None):
        if alpha is not None:
            self.alpha = alpha
        if epsilon is not None:
            self.epsilon = epsilon
        
        self.model = PCACpp(self.alpha, self.epsilon)

        return self

    @property
    def explained_variance_ratio_(self):
        total = self.model.covariance_by_component_.sum()
        return self.model.covariance_by_component_ / total

    def get_model(self):
        return self.model


if __name__ == "__main__":
    # pca_ours = PCAOurs(1)
    # pca_sklrn = skld.PCA(n_components=784)
    # breakpoint()


    # logger.info("Loading CSV")
    # df = pd.read_csv("../data/train.csv")

    # # logger.info("Training")
    # # pca.fit(X)

    # logger.info("Transforming")
    # transformed = pca_ours.fit(df.drop(columns="label"))
    # # transformed = pca_ours.fit_transform(df.drop(columns="label"))
    # # pca_sklrn.fit_transform(df.drop(columns="label"))
    # breakpoint();

    # df["pca_0_ours"] = transformed[:, 0]
    # df["pca_1_ours"] = transformed[:, 1]
    # df["pca_2_ours"] = transformed[:, 2]

    # logger.info("Plotting")

    # df["label"] = df["label"].astype(str)
    # fig = px.scatter_3d(
    #     df, x="pca_0_ours", y="pca_1_ours", z="pca_2_ours", color="label", title="Ours"
    # )
    # fig.show()

    # pca_sk = skld.PCA(n_components=3)
    # transformed_sk = pca_sk.fit_transform(df.drop(columns="label"))
    # df["pca_0_sk"] = transformed_sk[:, 0]
    # df["pca_1_sk"] = transformed_sk[:, 1]
    # df["pca_2_sk"] = transformed_sk[:, 2]
    # fig = px.scatter_3d(
    #     df, x="pca_0_sk", y="pca_1_sk", z="pca_2_sk", color="label", title="Sklearn"
    # )
    # fig.show()

    #Levantamos el dataset
    df = pd.read_csv("../data/train.csv")

    grid = {
        'pca__alpha': [i for i in range (5, 150, 5)]
    }

    pca_with_knn_pipe_exp1 = Pipeline([
        ('pca', PCAOurs())
    ])

    pca_knn_cv_exp1 = GridSearchCV(
        pca_with_knn_pipe_exp1,
        param_grid=grid,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,
        cv=10,
        verbose=10,
    )

    train_dataset_exp1 = df.drop(columns="label")
    train_label_exp1 = df["label"]

    pca_knn_cv_exp1.fit(train_dataset_exp1, train_label_exp1)