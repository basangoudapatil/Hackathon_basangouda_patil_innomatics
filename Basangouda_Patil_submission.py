#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries to read the dataset
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import, read and explore the dataset
df = pd.read_csv("D:/Data_Science/dataframe_.csv")
df.head()


# In[3]:


#check for null values, outliers, feature datatype etc
df.info()


# In[4]:


df.shape


# In[5]:


#outlier detection using IQR method for input feature
Q1=df.input.quantile(0.25)
Q3=df.input.quantile(0.75)
IQR=df.input.quantile(0.75)-df.input.quantile(0.25)
L1=Q1-(1.5*IQR)
L2=Q3+(1.5*IQR)
print(Q1)
print(Q3)
print(IQR)
print(L1)
print(L2)


# In[6]:


L1=Q1-(1.5*IQR)
count=0
for i in df.input:
  if i<(L1):
    count+=1
print(count)


# In[7]:


L2=Q3+(1.5*IQR)
count=0
for j in df.input:
  if j>(L2):
    count+=1
print(count)
#from above it is clear that there are no outliers 


# In[8]:


#outlier detection using boxplot
sns.boxplot(y=df.input)


# In[9]:


#outlier detection using IQR method for input feature
q1=df.output.quantile(0.25)
q3=df.output.quantile(0.75)
iqr=df.output.quantile(0.75)-df.output.quantile(0.25)
l1=q1 - (1.5*iqr)
l2=q3 + (1.5*iqr)
print(q1)
print(q3)
print(iqr)
print(l1)
print(l2)


# In[10]:


l1= q1 - (1.5*iqr)
count=0
for i in df.output:
  if i<(l1):
    count+=1
print(count)


# In[11]:


l2=q3 + (1.5*iqr)
count=0
for j in df.output:
  if j>(l2):
    count+=1
print(count)

#from above it is clear that there are outliers and the count is large enough
#therefore replace with largest value or remove them completely


# In[12]:


sns.boxplot(y=df.output)


# In[13]:


df1=df.copy()


# In[14]:


df1.shape


# In[16]:


df1=df[df['output']<l2]


# In[17]:


df1.shape


# In[18]:


df1[['input','output']].corr()
#the features are higly correlated, i.e., correlation value>0.2


# # Model Building

# In[19]:


X=df1.loc[:,['input']].values
y=df1.loc[:,'output'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[20]:


X_train[0:5]


# In[21]:


y_train[0:5]


# In[22]:


# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[23]:


y_pred = regressor.predict(X_test)
y_pred[0:5]


# In[24]:


#Evaluation Metrics
from sklearn import metrics
print('R2 Score:', metrics.r2_score(y_test, y_pred))
print('Train_Score:', regressor.score(X_train, y_train))
print('Test_score:', regressor.score(X_test, y_test))


# In[25]:


#Lasso Regression
from sklearn.linear_model import Lasso
for i in range(0,10):
  lasso=Lasso(alpha=i)
  lasso.fit(X_train, y_train)
  y_pred=lasso.predict(X_test)
  print('Train_score:', lasso.score(X_train, y_train))
  print('Test_score:', lasso.score(X_test, y_test))


# In[26]:


#Ridge Regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (metrics.mean_squared_error(y_train, y_train_pred),
                metrics.mean_squared_error(y_test, y_test_pred)))

print('R2 train: %.3f, test: %.3f' % (metrics.r2_score(y_train, y_train_pred), metrics.r2_score(y_test, y_test_pred)))


# In[27]:


# Conclusion:
# All the ML models give the same output for the given dataset.
# Therefore Linear Regression is the best suited ML algorithm for this particular problem.
# Linear Regression model is easy and efficient to build, interpret results as well as easy to understand. 
# While other two models are bit complex in nature.
print(regressor.intercept_)
print(regressor.coef_)

