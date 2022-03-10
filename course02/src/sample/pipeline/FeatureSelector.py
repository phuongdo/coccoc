from sklearn.base import BaseEstimator, TransformerMixin


#Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor
    def __init__( self, feature_names ):
        self._feature_names = feature_names

        #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self

        #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ]