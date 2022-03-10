import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#Custom transformer we wrote to engineer features ( bathrooms per bedroom and/or how old the house is in 2019  )
#passed as boolen arguements to its constructor
class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, bath_per_bed = True, years_old = True ):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old

    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self

        #Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y = None):
        #Check if needed
        if self._bath_per_bed:
            #create new column
            X.loc[:,'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            #drop redundant column
            X.drop('bathrooms', axis = 1 )
        #Check if needed
        if self._years_old:
            #create new column
            X.loc[:,'years_old'] =  2019 - X['yr_built']
            #drop redundant column
            X.drop('yr_built', axis = 1)

        #Converting any infinity values in the dataset to Nan
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        #returns a numpy array
        return X.values