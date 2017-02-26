
# coding: utf-8

# In[4]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.grid_search import RandomizedSearchCV
get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')


# In[5]:

df = pd.read_csv('midterm_train.csv')
y = df.pop('y')


# In[6]:

# Look at all the columns in the dataset
def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))


# In[7]:

printall(df)


# In[8]:

df.describe()
# Things I see here include 45 variables summarized and 50 variables above.  That must mean 5 non numeric fields


# In[9]:

# numeric variables
numeric_variables = list(df.dtypes[df.dtypes != "object"].index)
df[numeric_variables].shape


# In[10]:

# non numeric variables
categorical_variables = list(df.dtypes[df.dtypes == 'object'].index)
df[categorical_variables].shape


# In[11]:

# taking a quick look at the categoricals
df[categorical_variables].head()


# In[12]:

df['x9'].value_counts()


# In[13]:

df['x16'].value_counts()


# In[14]:

df['x43'].value_counts()


# In[15]:

df['x19'].value_counts()


# In[16]:

df['x44'].value_counts()


# In[17]:

'''
# a key idea is that I want to know how good my model is BEFORE I submit it to kaggle, 
# so I'm going to take my training example and make my own test/train split.  I'll hold my test out and use
# that to judge my model before submitting.
random.seed = 42
df = pd.read_csv("midterm_train.csv")
df_train, df_test = train_test_split(df)
df_train.to_csv("my_midterm_train_split.csv", index=False)
df_test.to_csv("my_midterm_test_split.csv", index=False)

'''


# In[18]:

df["x19"].value_counts()


# In[19]:

df["x44"]= df.x44.str.replace('$', '').str.replace('%', '').astype(float)


# In[20]:

df["x9"] = df.x9.str.replace('$', '').str.replace('%', '').astype(float)


# In[21]:

printall(df)


# In[22]:

df['x19'].value_counts()


# In[23]:

df["x19"] = df.x19.str.replace('Jun', 'june').str.replace('July', 'july').astype(str)
df["x19"] = df.x19.str.replace('May', 'may').str.replace('Aug', 'august').astype(str)
df["x19"] = df.x19.str.replace('Apr', 'april').str.replace('sept.', 'september').astype(str)
df["x19"] = df.x19.str.replace('Mar', 'march').str.replace('Oct', 'october').astype(str)
df["x19"] = df.x19.str.replace('Feb', 'february').str.replace('Nov', 'november').astype(str)
df["x19"] = df.x19.str.replace('January', 'january').str.replace('Dev', 'december').astype(str)


# In[24]:

df["x19"].value_counts()


# In[25]:

'''
def handle_missing_categorical(categorical_var):
    for variable in categorical_variables:
        # Fill missing data with the word "Missing"
        df[variable]= df[variable].fillna("Missing", inplace=True)
        # Create array of dummies
        dummies = pd.get_dummies(df[variable], prefix=variable)
        # Update df to include dummies and drop the main variable
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([variable], axis=1, inplace=True)

'''


# In[26]:

df["x16"].value_counts().isnull()


# In[27]:

df[df.x16.isnull()].head()


# In[28]:

# Fill missing data with the word "Missing"
df["x16"] = df["x16"].fillna("Missing", inplace=True)


# In[29]:

# Create array of dummies
dummies = pd.get_dummies(df["x16"], prefix="x16")

# Update df to include dummies and drop the main variable
df = pd.concat([df, dummies], axis=1)
df.drop(["x16"], axis=1, inplace=True)


# In[30]:

df[df.x19.isnull()].head()


# In[31]:

df["x19"].value_counts()


# In[32]:

df["x19"] = df.x19.str.replace('nan', 'Missing').astype(str)


# In[33]:

df["x19"].value_counts()


# In[34]:

# Create array of dummies
dummies = pd.get_dummies(df["x19"], prefix="x19")

# Update df to include dummies and drop the main variable
df = pd.concat([df, dummies], axis=1)
df.drop(["x19"], axis=1, inplace=True)


# In[35]:

df["x43"].value_counts()


# In[36]:

printall(df)


# In[37]:

df["x43"] = df.x43.str.replace('nan', 'Missing').astype(str)


# In[38]:

df[df.x43.isnull()]


# In[39]:

# Create array of dummies
dummies = pd.get_dummies(df["x43"], prefix="x43")

# Update df to include dummies and drop the main variable
df = pd.concat([df, dummies], axis=1)
df.drop(["x43"], axis=1, inplace=True)


# In[40]:

for var in numeric_variables:
    df[var].fillna(df[var].mean(), inplace=True)


# In[41]:

df.isnull().any()


# In[42]:

df["x9"] = df["x9"].astype(float)


# In[43]:

df["x9"].fillna(df["x9"].mean(), inplace=True)


# In[44]:

df.isnull().any()


# In[45]:

df["x44"].fillna(df["x44"].mean(), inplace=True)


# In[46]:

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.2, random_state=42)


# In[53]:

# Params: list form ['v1','v2','v3'], and contain the possible values of parameters to consider in a grid search
# with n_job=-1 (full processing power). This returns the estimator object to be stored in memory. 
# Ex: estimator = gridSearchRFC(n_estimators_param, max_features_param, min_samples_split_param, min_samples_leaf_param)

def gridSearchRFC(n_estimators_param, max_features_param, min_samples_split_param, min_samples_leaf_param):

    ### Grid Search
    n_estimators = [n_estimators_param]
    max_features = [max_features_param]
    min_samples_split = [min_samples_split_param]
    min_samples_leaf = [min_samples_leaf_param]

    rfc = RandomForestClassifier(n_jobs=-1)
    
    estimator = GridSearchCV(rfc,
                             dict(n_estimators=n_estimators_param,
                                  max_features=max_features_param,
                                  min_samples_split=min_samples_split_param,
                                  min_samples_leaf=min_samples_leaf_param
                                  ), cv=None, n_jobs=-1)
    estimator.fit(X_train, y_train)
    return estimator


# In[54]:

n_estimators = [300,400,500,1000,2000,3000]
max_features = ['auto', 'sqrt','log2',0.9,0.2]
min_samples_split = [3,5,7]
min_samples_leaf = [1,2,3]


# In[55]:

gridSearchRFC(n_estimators, max_features, min_samples_split, min_samples_leaf)


# In[ ]:



