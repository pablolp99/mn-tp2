import pandas as pd

from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

import sys
import time

sys.path.insert(1, "../src")
from knn import *
from pca import *

N_SPLITS = 10

train = pd.read_csv("../data/train.csv")[:1000]

X = train.drop(columns='label').to_numpy()
y = train.label.to_numpy()

k, a = 5, 5

_pca = PCA(alpha=a)
_knn = KNNClassifier(k=k)

_x_train, _y_train = X[:900], y[:900]
_x_test, _y_test = X[900:], y[900:]

_pca = PCA(alpha=a)
_knn = KNNClassifier(k=k)

_pca.fit(_x_train)
_x_train_transformed = _pca.transform(_x_train)
_knn.fit(_x_train_transformed, _y_train)


_x_test_transformed = _pca.transform(_x_test)

breakpoint()

_pred = _knn.predict(_x_test_transformed)