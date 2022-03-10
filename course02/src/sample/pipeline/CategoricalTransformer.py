from sklearn.base import BaseEstimator, TransformerMixin


#Custom transformer that breaks dates column into year, month and day into separate columns and
#converts certain features to binary
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self, use_dates = ['year', 'month', 'day'] ):
        self._use_dates = use_dates

    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    #Helper function to extract year from column 'dates'
    def get_year( self, obj ):
        return str(obj)[:4]

    #Helper function to extract month from column 'dates'
    def get_month( self, obj ):
        return str(obj)[4:6]

    #Helper function to extract day from column 'dates'
    def get_day(self, obj):
        return str(obj)[6:8]

    #Helper function that converts values to Binary depending on input
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    #Transformer method we wrote for this transformer
    def transform(self, X , y = None ):
        #Depending on constructor argument break dates column into specified units
        #using the helper functions written above
        for spec in self._use_dates:

            exec( "X.loc[:,'{}'] = X['date'].apply(self.get_{})".format( spec, spec ) )
        #Drop unusable column
        X = X.drop('date', axis = 1 )

        #Convert these columns to binary for one-hot-encoding later
        X.loc[:,'waterfront'] = X['waterfront'].apply( self.create_binary )

        X.loc[:,'view'] = X['view'].apply( self.create_binary )

        X.loc[:,'yr_renovated'] = X['yr_renovated'].apply( self.create_binary )
        #returns numpy array
        return X.values