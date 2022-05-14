from sklearn.base import BaseEstimator

class KNNClassifier(BaseEstimator):
    def __init__(self, k_neighbors=5):
        """Constructor

        Parameters
        ----------
        k_neighbors : int, optional
            by default 5
        """
        pass

    def fit(self, X, y=None):
        """Fiteo del modelo

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_, optional
            _description_, by default None
        """

    def predict(self, X):
        """Predictor

        Parameters
        ----------
        X : _type_
            _description_
        """
    
    def set_params(self, **kwargs):
        """Seter de parametros del Clasificador
        """
    
    def get_params(self):
        """Getter de parametros del Clasificador
        """