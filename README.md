# mn-tp2
TP2 - Metodos Numericos

## Python 3.10
En el presente trabajo se utilizara `python3.10` para el desarrollo.

## Sklearn compliance
Para poder utilizar las funncionalidades de `sklearn`, tales como `GridSearch` y `CrossValidation`, se armaran objetos que sigan la interfaz de la misma libreria.

Para el caso de un estimador (KNN):
```python3
class KNNClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        ...
        return self
    
    def fit(self, X, y):
        ...
        return self

    def predict(self, X):
        ...
        return prediction
    
    def set_params(self, params):
        ...
        return self
    
    def get_params(self):
        ...
        return params
```

Para el caso de transformador (PCA):
```python3
class PCA(BaseEstimator):
    def __init__(self, **kwargs):
        ...
        return self
    
    def fit(self, X, y):
        ...
        return self

    def fit_transform(self, X):
        ...
        return transformed

    def transform(self, X):
        ...
        return transformed
```