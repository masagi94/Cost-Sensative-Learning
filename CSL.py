#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing

import pandas as pd
from pandas.api.types import is_numeric_dtype


# In[2]:


# Returns a sorted numpy array containing the correlations between the features in the data and the target label
def getCorrelation(data, label):

    corr_list = np.empty((0,2))
    
    # Loop through every feature column
    for col in data.columns:
        
        # get the correlation between this feature and the truth label
        correlation = format(abs(data[col].corr(label)), '.5f')
        
        # ignore nans, append to array
        if correlation != 'nan':
            corr_list = np.append(corr_list, [[correlation,col]], axis=0)
    
    # return sorted array
    return corr_list[corr_list[:,0].argsort()]


# In[3]:


# loops through the data and encodes string values to numerical format
def preprocessData(data):
    le = LabelEncoder()

    # Loop through every feature column
    for col in data.columns:
        
        # If it's not numeric, encode it
        if not(is_numeric_dtype(data[col])):
            newdata = le.fit_transform(data[col].astype('str'))
            data[col] = newdata


# In[4]:


data = pd.read_csv('cup98lrn.txt', sep=",", header= 0)
# replacing null with: 1) Most frequent value, if column values are of type "Object", and 2) Mean of the column, if column values are numeric
data = data.fillna(pd.Series([data[c].value_counts().index[0] if data[c].dtype == np.dtype('O') else data[c].mean() for c in data], index = data.columns))
preprocessData(data)


# In[ ]:





# In[5]:


#selecting donors
donors = data[data['TARGET_B'] == 1]

#selecting non-donors and reducing it to the number of donors
non_donors = data[data['TARGET_B'] == 0]

# percentage of confirmed donors we want present in the train data.
# Example: .70 would mean 70% of donors in the train data, 30% in the test data
donor_percent = .70

split = round(len(donors)*donor_percent)

xtrain = donors[:split]
xtest = donors[split:]

# xtrain = donors
# xtest = donors


# In[6]:


# create a list of non donors, and split them into train/test sets
midpoint = round(len(non_donors)/2)
non_donor_train = non_donors[:midpoint]
non_donor_test = non_donors[midpoint:]

non_donor_train= non_donor_train.sample(n= xtrain.shape[0], random_state= 42)
non_donor_test= non_donor_test.sample(n= xtest.shape[0], random_state= 42)


# In[7]:


# concatenate donors and nondonors to create the final train/test data
xtrain = xtrain.append(non_donor_train, ignore_index = True)
xtest = xtest.append(non_donor_test, ignore_index = True)

# drop unnecessary columns
xtrain = xtrain.drop(['TARGET_D'], axis= 1)
xtest = xtest.drop(['TARGET_D'], axis= 1)

ytrain = xtrain.pop('TARGET_B')
ytest = xtest.pop('TARGET_B')


# In[8]:


# scale the data so regression performs better
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)

scaler = preprocessing.StandardScaler().fit(xtest)
xtest_scaled = scaler.transform(xtest)


# In[ ]:





# In[9]:


# Train 3 models
clf1 = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)
clf2 = DecisionTreeClassifier(random_state=0).fit(xtrain,ytrain)
clf3 = LogisticRegression(random_state=0, max_iter=10000).fit(xtrain_scaled, ytrain)


# In[ ]:





# In[10]:


print("Model 1 Accuracy:",metrics.accuracy_score(ytest, clf1.predict(xtest)))
print("Model 2 Accuracy:",metrics.accuracy_score(ytest, clf2.predict(xtest)))
print("Model 3 Accuracy:",metrics.accuracy_score(ytest, clf3.predict(xtest_scaled)))
