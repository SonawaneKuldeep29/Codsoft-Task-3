#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# In[1]:


# Import the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Load iris data
df=pd.read_csv("Iris.csv")
df


# In[3]:


df.info()


# ## Data Analysis

# In[4]:


df.info()


# In[5]:


# Find null values
df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[9]:


df['species'].value_counts()


# ## Data visualisation

# In[11]:


df.hist(bins=10, figsize=(10, 8),color="red")
plt.tight_layout()
plt.show()


# In[13]:


# Multiple plot
sns.pairplot(df,hue='species')


# In[14]:


df.corr()


# In[15]:


# heat map
corr=df.corr()
fig,ax=plt.subplots(figsize=(5,4))
hm=sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[16]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[18]:


df['species']=le.fit_transform(df['species'])
df.head()


# In[20]:


from sklearn.model_selection import train_test_split
x=df.drop(columns=['species'])
y=df['species']
# train=70
# test=30
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[21]:


from sklearn.svm import SVC
model1_svc=SVC()
model1_svc.fit(x_train,y_train)


# In[22]:


predection=model1_svc.predict(x_test)


# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


print(accuracy_score(y_test,predection)*100)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


model=LogisticRegression()


# In[27]:


model.fit(x_train,y_train)


# In[28]:


print("Accuracy: ",model.score(x_test,y_test)*100)

