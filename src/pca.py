import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support, cohen_kappa_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import sys
import pathlib
import logging

sys.path.insert(1, f"{pathlib.Path(__file__).parent.resolve()}/../build")
sys.path.insert(1, "../src")
from knn import *
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


def grid_search_step(k, a, splits_dataset, metrics):
    score_values = []
    cohen_kappa_values = []
    precision_recall_values = []
    f1_values = []
    confusion_matrix_values = []
    knn_time_values = []
    pca_train_transform_time_values = []
    pca_test_transform_time_values = []
    print(f"Running gridsearch for: k={k} - alpha={a}")

    print("\tSplits:")
    for idx, _pca, _x_train, _x_test, _y_train, _y_test in splits_dataset:
        print(f"\t\t{idx} time:", end="\t\t")
        # print(idx, _pca, _x_train, _x_test, _y_train, _y_test)

        _knn = KNNClassifier(k=k)
        
        # PCA train transform
        _pca_train_transform_st = time.time()

        _x_train_transformed = _pca.transform(_x_train, truncate=a)

        _pca_train_transform_et = time.time()

        # PCA test transform
        _pca_test_transform_st = time.time()

        _x_test_transformed = _pca.transform(_x_test, truncate=a)

        _pca_test_transform_et = time.time()
        
        # KNN fit and predict
        _knn_st = time.time()
        
        _knn.fit(_x_train_transformed, _y_train)
        _pred = _knn.predict(_x_test_transformed)

        _knn_et = time.time()

        print(f"{(_knn_et - _knn_st):.4f}")

        _score_metric = accuracy_score(_y_test, _pred)
        _cohen_kappa_metric = cohen_kappa_score(_y_test, _pred)
        _precision_recall_metric = precision_recall_fscore_support(_y_test, _pred)
        _f1_metric = f1_score(_y_test, _pred, average='weighted')
        _confusion_matrix_metric = confusion_matrix(_y_test, _pred)      

        score_values.append(_score_metric)
        cohen_kappa_values.append(_cohen_kappa_metric)
        precision_recall_values.append(_precision_recall_metric)
        f1_values.append(_f1_metric)
        confusion_matrix_values.append(_confusion_matrix_metric)
        knn_time_values.append(_knn_et - _knn_st)
        pca_train_transform_time_values.append(_pca_train_transform_et - _pca_train_transform_st)
        pca_test_transform_time_values.append(_pca_test_transform_et - _pca_test_transform_st)

    print("")

    metrics[f"{k} - {a}"] = dict()
    metrics[f"{k} - {a}"]['score'] = np.mean(score_values)
    metrics[f"{k} - {a}"]['cohen-kappa'] = np.mean(cohen_kappa_values)
    metrics[f"{k} - {a}"]['recall'] = np.mean(precision_recall_values)
    metrics[f"{k} - {a}"]['f1'] = np.mean(f1_values)
    metrics[f"{k} - {a}"]['confusion-matrix'] = confusion_matrix_values
    metrics[f"{k} - {a}"]['knn-time'] = np.mean(knn_time_values)
    metrics[f"{k} - {a}"]['pca-train-transform-time'] = np.mean(pca_train_transform_time_values)
    metrics[f"{k} - {a}"]['pca-test-transform-time'] = np.mean(pca_test_transform_time_values)
    print(f"\tMean time: {np.mean(score_values)}")

if __name__ == "__main__":

    N_SPLITS = 5

    train = pd.read_csv("../data/train.csv")

    X = train.drop(columns='label').to_numpy()
    y = train.label.to_numpy()

    grid = {
#     "k": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#     "alpha": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 153]
    "k": [2],
    "alpha": [10]
    }

    skf = StratifiedKFold(n_splits=N_SPLITS)
    splits = list(skf.split(X, y))

    metrics = dict()

    splits_dataset = []
    pca_times_values = []
    print("\tCalculating PCA for splits:")
    for idx, (train_index, test_index) in enumerate(splits):
        print(f"\t\t{idx} time:", end="\t\t")
        
        _x_train, _y_train = X[train_index], y[train_index]
        _x_test, _y_test = X[test_index], y[test_index]
        
        # PCA fit with train
        _pca = PCA(alpha=10)
        
        _pca_st = time.time()
                
        _pca.fit(_x_train)
                
        _pca_et = time.time()
        
        pca_times_values.append(_pca_et - _pca_st)
        
        splits_dataset.append((idx, _pca, _x_train, _x_test, _y_train, _y_test))

        print(f"{(_pca_et - _pca_st):.4f}")
        
    # metrics['pca-fit-time'] = np.mean(pca_times_values)

    for a in grid["alpha"]:
        for k in grid["k"]:
            grid_search_step(k,a,splits_dataset,metrics)
