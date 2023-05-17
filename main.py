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
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Model Evaluation 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score

# load data
data = "/Users/tafadzwagoremusandu/Documents/Credit Risk Prediction/credit_customers.csv"

def load_data(data):
    
    df = pd.read_csv(data)
    
    # split personal_status column into gender and marital_status columns
    df[['gender', 'marital_status']] = df['personal_status'].str.split(" ", expand=True)
    df.drop(['personal_status'], axis=1, inplace=True)  
    
    #map class values to binary values
    class_map = {'bad': 0, 'good': 1}
    df['class'] = df['class'].map(class_map) 

    return df

df = load_data(data)

# Spilt the data
X = df.drop('class',axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Data preprocessing
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

def preprocess_data(X_train, X_test):
    data_char = X_train.loc[:,df.dtypes==object]
    data_num = X_train.loc[:,df.dtypes!=object]

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
    return x_train,x_test

x_train, x_test = preprocess_data(X_train, X_test)        

# Train the model

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

print("best combination of parameters for the model:")
pprint(model.best_estimator_.get_params())
print("\n")


# prediction scores on test set

final_model = model.best_estimator_
y_pred = final_model.predict(x_test)

conf_matrix = confusion_matrix(y_test,y_pred)
classf_report = classification_report(y_test,y_pred)

print('Confusion Matrix:\n',conf_matrix)
print("\n")
print('Classification Report :\n', classf_report)

