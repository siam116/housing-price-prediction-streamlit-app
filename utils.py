import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

totalRooms_col, totalBedrooms_col, population_col, households_col = 3, 4, 5, 6


class CombinesAttrAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_household=True):
        self.add_bedrooms_per_household = add_bedrooms_per_household

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, totalRooms_col] / X[:, households_col]
        population_per_household = X[:, population_col] / X[:, households_col]
        bedrooms_per_room = X[:, totalBedrooms_col] / X[:, totalRooms_col]

        if self.add_bedrooms_per_household:
            bedrooms_per_household = X[:, totalBedrooms_col] / X[:, households_col]
            return np.c_[X, rooms_per_household,
                         population_per_household,
                         bedrooms_per_room,
                         bedrooms_per_household]
        else:
            return np.c_[X, rooms_per_household,
                         population_per_household,
                         bedrooms_per_room]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
