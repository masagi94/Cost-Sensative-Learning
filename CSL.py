#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# In[2]:


if __name__ == '__main__':

  data = pd.read_csv('cup98lrn.txt', sep=",", header= 0)


  # replacing null with: 1) Most frequent value, if column values are of type "Object", and 2) Mean of the column, if column values are numeric
  data = data.fillna(pd.Series([data[c].value_counts().index[0] if data[c].dtype == np.dtype('O') else data[c].mean() for c in data], index = data.columns))

  # selecting donors
  donors = data[data['TARGET_B'] == 1]

  # selecting non-donors and reducing it to the number of donors
  nondonors = data[data['TARGET_B'] == 0]
  nondonors = nondonors.sample(n= donors.shape[0], random_state= 42)

  # create training set and lables for the classification model
  train1 = donors.append(nondonors, ignore_index= True)
  y1 = train1.pop('TARGET_B')
  train1 = train1.drop(['TARGET_D'], axis= 1)

  # create training set and lables for the regression model
  train2 = donors.drop(['TARGET_B'], axis= 1)
  y2 = train2.pop('TARGET_D')

  # if we do nothing, the profit from learning dataset  
  BaselineProfit = sum(y2) - 0.68*len(data)
  print(BaselineProfit)
