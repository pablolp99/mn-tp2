# mn-tp2
TP2 - Metodos Numericos

## Sklearn compliance
Para poder utilizar las funncionalidades de `sklearn`, tales como `GridSearch` y `CrossValidation`, se armaran objetos que sigan la interfaz de la misma libreria.

```python3
class KNNClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        ...
        return self
    
    def fit(self, X, y):
        ...
        return self

    def predect(self, X):
        ...
        return prediction
    
    def set_params(self, params):
        ...
        return self
    
    def get_params(self):
        ...
        return params
```