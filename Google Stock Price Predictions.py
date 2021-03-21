#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import quandl, math
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
quandl.ApiConfig.api_key = 'yu2dbazpA4SXydS7mxJa'
df = quandl.get('WIKI/GOOGL')


# In[63]:


df = df [["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]


# In[64]:


#df.head()


# In[65]:


#Percentage Change
df["HL PCT"] = (df["Adj. High"]-df["Adj. Close"])*100/df["Adj. Close"]
df["PCT change"] = (df["Adj. Close"]-df["Adj. Open"])*100/df["Adj. Open"]

df = df[["Adj. Close","HL PCT","PCT change","Adj. Volume"]]


# In[66]:


#df.head()


# In[67]:


forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))
df.dropna(inplace=True)

df["label"] = df[forecast_col].shift(-forcast_out)
df.dropna(inplace =True)


# In[68]:


df.tail(5)


# In[69]:


X = np.array(df.drop(['label'],1))

X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]


df.dropna(inplace=True)
y = np.array(df["label"])
y = y[:-forcast_out]


# In[70]:


X.shape


# In[71]:


y.shape


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Pickling
import pickle
with open('linearregression','wb') as f:
    pickle.dump(clf,f)
pickle_in = open('linearregression','rb')

clf = pickle.load(pickle_in)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)


# In[80]:


forcast = clf.predict(X_lately)


# In[81]:


forcast.dtype


# In[82]:


accuracy


# In[77]:





# In[78]:





# In[ ]:




