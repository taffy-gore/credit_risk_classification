# load required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# Model Building

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint


# Model Evaluation 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score

# load data
credit_cust_data = pd.read_csv('credit_customers.csv')
df = credit_cust_data.copy()

# get info about columns types
df.info()

# summary stats of numeric variables
df.describe().transpose

# Check for missing values
df.isnull().sum()

# Distribution plots for Numeric variables
%matplotlib inline

df.hist(figsize=(25,25))
plt.show()

# Class distribution in target variable - how imbalanced are the classes?
df['class'].value_counts().plot(kind='bar')

# Check Unique classes in each categorical variables
n = 1
for (columnName, columnData) in df.iteritems():
    if columnData.dtype == 'object':
        print('Name   : ', columnName)
        print('Unique : ', columnData.unique())
        print('No     : ',len(columnData.unique()))
        print()
        n+=1
    else:
        pass


# get the categorical variables column names
df.select_dtypes(['object']).columns

# get the numerical variables column names
df.select_dtypes(['float64']).columns

# Class distribution in Categorical variables
categorical_columns=['checking_status', 'credit_history', 'purpose', 'savings_status','employment','personal_status',
                     'other_parties', 'property_magnitude','other_payment_plans', 'housing', 'job',
                     'own_telephone','foreign_worker', 'class']
plt.figure(figsize=(25,25),layout='constrained')
for i in range(len(categorical_columns)):
    plt.subplot(5,3,i+1)
    sns.countplot(data=df,x=categorical_columns[i],)
    plt.title(categorical_columns[i]+'_count',)
    
plt.show()


# Separate gender and marriage status form the personal status column
def split_and_drop(df, col_name):
    """
    Splits a column in a dataframe on the space character, creates two new columns,
    drops the original column, and returns the modified dataframe.
    
    Args:
    - df (pandas.DataFrame): The dataframe containing the column to split and drop.
    - col_name (str): The name of the column to split and drop.
    
    Returns:
    - df (pandas.DataFrame): The modified dataframe with the new columns and the original column dropped.
    """
    df[['gender', 'marital_status']] = df[col_name].str.split(" ", expand=True)
    df.drop([col_name], axis=1, inplace=True)
    return df

df = split_and_drop(df,'personal_status')
df

# convert object datatype to category
cat = df.select_dtypes('object')
for i in list(cat.columns):
    df[i] = df[i].astype('category')
    
df.info()

# define a dictionary to map Target values to binary values
class_map = {'bad': 0, 'good': 1}

# use the map() method to convert class values to binary values
df['class'] = df['class'].map(class_map)

# Spilt the data
X = df.drop('class',axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# get variable types by categorical vs numerical
data_char = X_train.loc[:,df.dtypes=='category']
data_num = X_train.loc[:,df.dtypes!='category']


# create a DataFrameSelector class
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

# pipeline for raw data pre-processing 
cat_attribs = list(data_char)
num_attribs = list(data_num)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('one_hot_encoder', OneHotEncoder(handle_unknown = 'ignore'))
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

x_train = full_pipeline.fit_transform(X_train)
x_test = full_pipeline.transform(X_test)

# train the models with cross validation and get scores

class BestModel:
    def __init__(self):
        self.version = 1

    def cross_validate_models(self, x_train,y_train):
        
        # cross validation on Logistic Regression
        lr_clf = LogisticRegression()
        
        lr_scores = cross_val_score
        lr_scores = cross_val_score(lr_clf, x_train, y_train,
                            scoring='f1_weighted', cv=5)
        
        lr_mean_score = lr_scores.mean()
        lr_score_stdev = lr_scores.std()
        
        # cross validation on random forest
        rf_clf = RandomForestClassifier()
        
        rf_scores = cross_val_score
        rf_scores = cross_val_score(rf_clf, x_train, y_train,
                            scoring='f1_weighted', cv=5) 
        
        rf_mean_score = rf_scores.mean()
        rf_score_stdev = rf_scores.std()   
        
        # cross validation on gradient boost
        GBoost_clf = GradientBoostingClassifier()
        
        GBoost_scores = cross_val_score
        GBoost_scores = cross_val_score(GBoost_clf, x_train, y_train,
                            scoring='f1_weighted', cv=5) 
        
        GBoost_mean_score = GBoost_scores.mean()
        GBoost_score_stdev = GBoost_scores.std()
        
        # cross validation on XGBoost
        xgb_clf = XGBClassifier(random_state=42)
        
        XGBoost_scores = cross_val_score
        XGBoost_scores = cross_val_score(xgb_clf, x_train, y_train,
                            scoring='f1_weighted', cv=5) 
        
        XGBoost_mean_score = XGBoost_scores.mean()
        XGBoost_score_stdev = XGBoost_scores.std()

        # Summary of model training scores
        logistic_scores_dict = {
            'Logistic_F1_score': lr_mean_score, 
            'Logistic_F1score_std': lr_score_stdev 
            }
        random_forest_scores_dict = {
            'RandomForest_F1_score': rf_mean_score, 
            'RandomForest_F1score_std': rf_score_stdev
            }
        GBoost_scores_dict = {
            'GBoost_scores_F1_score': GBoost_mean_score, 
            'GBoost_scores_F1score_std': GBoost_score_stdev 
            }
        XGBoost_scores_dict = {
            'XGBoost_scores_F1_score': XGBoost_mean_score, 
            'XGBoost_scores_F1score_std': XGBoost_score_stdev
        }
        
        # Return both dictionaries
        return logistic_scores_dict, random_forest_scores_dict,GBoost_scores_dict, XGBoost_scores_dict

# get final evaluation scores
analysis = BestModel()

logistic_scores_dict, random_forest_scores_dict,GBoost_scores_dict,XGBoost_scores_dict = analysis.cross_validate_models(x_test,y_test)

print(logistic_scores_dict)
print()
print(random_forest_scores_dict)
print()
print(GBoost_scores_dict)
print()
print(XGBoost_scores_dict)

# set parameter range for RandomSearchCV
params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}

xgb_clf = XGBClassifier(random_state=42)

model=RandomizedSearchCV(xgb_clf,param_distributions=params,
                            n_iter=5,scoring='f1_weighted',n_jobs=-1,cv=5,verbose=3)

model.fit(x_train,y_train)

# best combination of parameters
model.best_params_

from pprint import pprint

pprint(model.best_estimator_.get_params())

# check final model f1_weighted score on test set
xgboost_scores = cross_val_score
xgboost_scores = cross_val_score(final_model, x_test, y_test,
                            scoring='f1_weighted', cv=5) 
        
xgboost_mean_score = xgboost_scores.mean()
xgboost_score_stdev = xgboost_scores.std()

print(xgboost_mean_score)
print(xgboost_score_stdev)















