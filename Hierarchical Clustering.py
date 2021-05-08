#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset


# In[4]:


X = dataset.iloc[:, [3,4]].values


# In[5]:


X


# In[6]:


#using the dendorgram method to find the optional number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()


# In[10]:


#applying the hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[11]:


y_hc


# In[13]:


#visulising the result 
plt.scatter(X[y_hc==0, 0], X[y_hc == 0, 1], s =20, c ='red', label = 'Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc == 1, 1], s =20, c ='blue', label = 'Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc == 2, 1], s =20, c ='green', label = 'Target')
plt.scatter(X[y_hc==3, 0], X[y_hc == 3, 1], s =20, c ='yellow', label = 'careless')
plt.scatter(X[y_hc==4, 0], X[y_hc == 4, 1], s =20, c ='cyan', label = 'Sensible')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




