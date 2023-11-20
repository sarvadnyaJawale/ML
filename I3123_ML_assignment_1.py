#!/usr/bin/env python
# coding: utf-8

# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


# In[40]:


df = pd.read_csv("heart.csv")


# In[41]:


df


# In[42]:


#df.isna()


# In[43]:


df.isna().sum()
df


# In[47]:


df=df.fillna(df.median())
df


# In[48]:


#df=df.drop_duplicates()


# In[72]:


df = df.astype({"oldpeak":"int"})


# In[73]:


df


# In[74]:


x=df.drop('target', axis = 'columns')
x.shape


# In[76]:


y = df['target']
y.shape


# In[77]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)


# In[78]:


x_train.shape


# In[79]:


y_train.shape


# In[80]:


Reg = LogisticRegression()


# In[59]:


Reg.fit(x_train,y_train)


# In[81]:


y_pred = Reg.predict(x_test)


# In[82]:


print(accuracy_score(y_test, y_pred))


# In[84]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[85]:


print(classification_report(y_test, y_pred))


# In[86]:


print(confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred),annot= True)


# In[ ]:





# In[ ]:




