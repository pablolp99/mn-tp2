# mn-tp2
TP2 - Metodos Numericos

## Python 3.10-dev
En el presente trabajo se utilizara `python3.10-dev` para el desarrollo y experimentacion.

### Instalacion de Python 3.10-dev
Para utilizar dicha version de Python recomendamos el uso de `pyenv` para instalar dicha version. De la siguiente forma se puede instalar.
```bash
$ curl https://pyenv.run | bash
$ exec $SHELL
```

Una vez pyenv instalado, para obtener la version `3.10-dev` basta con
```bash
pyenv install 3.10
```

Con el archivo que se encuentra en el root del proyecto (`.python-version`) el mismo entorno deberia entrar en funcionamiento. De ser necesario, se pueden utilizar entornos virtuales para relizar las ejecuciones.




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