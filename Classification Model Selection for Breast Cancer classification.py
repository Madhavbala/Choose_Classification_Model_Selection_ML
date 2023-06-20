#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd #useful for loading the dataset
import numpy as np #to perform array
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[9]:


dataset = pd.read_csv('C:\\Users\\madhavan.bala\\Downloads\\data.csv')
print(dataset.shape)
# print(dataset.head(5))


# In[7]:


dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1}).astype(int)
# print(dataset.head)


# In[11]:


X = dataset.iloc[:, 2:32].values
X


# In[12]:


Y = dataset.iloc[:,1].values
Y


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


# In[16]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[17]:


results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f' % (name, cv_results.mean()))


# In[19]:


pyplot.ylim(.900, .999)
pyplot.bar(names, res, color ='maroon', width = 0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()

from sklearn.svm import SVC
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:


model = LogisticRegression(solver='liblinear',multi_class='ovr')
model.fit(X_train,y_train)
value = [[11.42,20.3877.58,386.1,0.14250.2839,0.2414,0.10520.2597,0.09744,0.4956,1.156,3.445,27.23,0.00911,0.07458,0.05661,0.01867,.05963,0.009208,0.009208, 26.5,98.87.567.7,0.2098,0.86630.6869,0.2575,0.6638,0.173]]
y_pred = model.predict(value)
print(y_pred)

