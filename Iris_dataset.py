#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
#load the data from sklearn's repository
from sklearn.datasets import load_iris


# In[71]:


#save the data in a variable
iris = load_iris()
type(iris)


# In[72]:


#iris.data
#The data contains features (are the columns) and observations (the rows in the dataset, there's 150 of them)
#response is the solution to the problem (eg: the type of species)
iris.data


# In[73]:


#feature names of the dataset
iris.feature_names


# In[74]:


#print the integers representing the species type
iris.target


# In[75]:


#print the strings representing the species type
iris.target_names


# In[76]:


#Both are NumPy arrays
type(iris.data)
type(iris.target)


# In[77]:


#Size of the dataset
iris.data.shape
#Number of observations x number of features


# In[78]:


#Shape of the target dataset
iris.target.shape


# In[79]:


#Storing the features (is a matrix, and is capitalised)
X = iris.data

#Storing the response (is a vector, and by convention is lower case)
y = iris.target


# In[80]:


y


# In[81]:


#The K nearest neighbours method (https://scikit-learn.org/stable/modules/neighbors.html)

from sklearn.neighbors import KNeighborsClassifier


# In[82]:


#create an instance of the class KneighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn


# In[83]:


knn.fit(X,y)


# In[84]:


#Obtaining the response prediction for an out of sample database(1x4 matrix)
knn.predict([[3,5,4,2]])


# In[85]:


#Obtaining the response prediction for an out of sample database (2x4 matrix)
Xnew = [[3,5,4,2],[5,4,3,2]]
knn.predict(Xnew)


# In[86]:


y_pred = knn.predict(X)
y_pred


# #Logistic Regression
# from sklearn.linear_model import LinearRegression
# logreg = LinearRegression()
# 
# #Fit the dataset for Liner Regression
# logreg.fit(X,y)
# 
# #Predict
# y_pred = logreg.predict(X)
# y_pred.round()

# In[87]:


from sklearn import metrics
print (metrics.accuracy_score(y,y_pred))


# In[92]:


# Train test and split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.4)


# In[94]:


Xtrain.shape


# In[95]:


Xtest.shape


# In[ ]:




