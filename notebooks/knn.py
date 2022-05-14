from sklearn.base import BaseEstimator
import pandas as pd
import sys

sys.path.insert(1, '/home/pablo/UBA/comp2022/MN/mn-tp2/build')
from mnpkg import *

class KNNClassifier(BaseEstimator):
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.model = KNNClassifierCpp(k_neighbors)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        assert len(X) > 0
        assert len(X.iloc[0]) > 0
        assert len(X) == len(y)
        
        labels = y.tolist()
        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        self.model.fit(imgs, labels, len(X), len(X.iloc[0]))

    def predict(self, X):
        assert len(X) > 0
        assert len(X.iloc[0]) > 0

        imgs = []
        for i in range(len(X)):
            imgs.append(X.iloc[i].tolist())

        return self.model.predict(imgs, len(imgs), len(imgs.iloc[0]))


if __name__ == '__main__':
    knn = KNNClassifier(3)

    breakpoint()

    df = pd.read_csv("../data/train.csv")

    y = df["label"]
    X = df.drop(columns='label')

    knn.fit(X, y)

    print(knn)