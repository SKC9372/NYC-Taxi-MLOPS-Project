import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

class OutlierRemover(TransformerMixin,OneToOneFeatureMixin,BaseEstimator):

    def __init__(self,percentile_values:list,col_subset:list):
        self.percentile_values = percentile_values
        self.col_subset = col_subset

    def fit(self,X,y=None):
        X = X.copy()

        self.quantiles_ = []

        for col in self.col_subset:
            lower_bound = X.loc[:,col].quantile(self.percentile_values[0])
            upper_bound = X.loc[:,col].quantile(self.percentile_values[1])

            self.quantiles_.append((lower_bound,upper_bound))

        return self
        
    def transform(self,X):
        X = X.copy()

        for ind,values in enumerate(self.col_subset):
            lower_bound,upper_bound = self.quantiles_[ind]
            filter_df = X[(X.loc[:,values]>=lower_bound) & (X.loc[:,values] <= upper_bound)]

            X = filter_df

        return X
