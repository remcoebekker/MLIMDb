import numpy as np
from sklearn.base import BaseEstimator


class BaselineEstimator(BaseEstimator):

    def __init__(self):
        self.average = None

    def fit(self, X, y):
        # We determine the average of the target value in the training set
        self.average = np.mean(y)
        return self

    def predict(self, X):
        # And return as our baseline prediction the average of the target value in the training set
        return np.full(shape=(len(X),), fill_value=self.average)

