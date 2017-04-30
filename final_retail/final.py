import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('train.csv')

data.drop('User_ID',axis=1,inplace=True)

y = data.pop('Purchase')
X = data

categorical_variables = X.columns

for variable in categorical_variables:
    # Fill missing data with the word "Missing"
    X[variable].fillna("Missing", inplace=True)
    # Create array of dummies
    dummies = pd.get_dummies(X[variable], prefix=variable)
    # Update X to include dummies and drop the main variable
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)

X.drop('City_Category_A',axis=1,inplace=True)
X.drop('City_Category_C',axis=1,inplace=True)
X.drop('Gender_F',axis=1,inplace=True)
X.drop('Marital_Status_0',axis=1,inplace=True)

scaler = MaxAbsScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = LogisticRegression(penalty='l2', C=1, solver='newton-cg',fit_intercept=True,max_iter=100)
model.fit(X_train,y_train)