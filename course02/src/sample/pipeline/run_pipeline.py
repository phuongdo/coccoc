import pandas as pd
# Categrical features to pass down the categorical pipeline
from course02.src.sample.pipeline.CategoricalTransformer import CategoricalTransformer
from course02.src.sample.pipeline.FeatureSelector import FeatureSelector
from course02.src.sample.pipeline.NumericalTransformer import NumericalTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ['date', 'waterfront', 'view', 'yr_renovated']

#Numerical features to pass down the numerical pipeline
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'condition', 'grade', 'sqft_basement', 'yr_built']

#Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ),

                                           ( 'cat_transformer', CategoricalTransformer() ),

                                           ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )

#Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ),

                                         ( 'num_transformer', NumericalTransformer() ),

                                         ('imputer', SimpleImputer(strategy = 'median') ),

                                         ( 'std_scaler', StandardScaler() ) ] )

#Combining numerical and categorical piepline into one full big pipeline horizontally
#using FeatureUnion
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ),

                                                   ( 'numerical_pipeline', numerical_pipeline ) ] )


## Dataset: https://www.kaggle.com/harlfoxem/housesalesprediction?source=post_page-----20ea2a7adb65----------------------
data = pd.read_csv('data/kc_house_data.csv')
#Leave it as a dataframe becuase our pipeline is called on a
#pandas dataframe to extract the appropriate columns, remember?
X = data.drop('price', axis = 1)
#You can covert the target variable to numpy
y = data['price'].values

X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )

#The full pipeline as a step in another pipeline with an estimator as the final step
full_pipeline_m = Pipeline( steps = [ ( 'full_pipeline', full_pipeline),

                                      ( 'model', LinearRegression() ) ] )

#Can call fit on it just like any other pipeline
full_pipeline_m.fit( X_train, y_train )

#Can predict with it like any other pipeline
y_pred = full_pipeline_m.predict( X_test )

print(y_pred)