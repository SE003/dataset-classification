#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('/content/iris.csv')
#1st step- input X
X = music_data.drop(columns=['Species'])
#2nd step - output dataset
y = music_data['Species']

#5th step (data splitting)
train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#3rd step-#library sklearn.tree download
model = DecisionTreeClassifier()
model.fit(X, y)


#4th step- add age and gender
predictions = model.predict([[5,3,1.6,0.2], [5,3.4,1.6,0.4], [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.4]])

predictions

#6th step- accuracy testing (import accuracy_score)

predictions = model.predict(X_test)
#predictions


score = accuracy_score(y_test, predictions)
score



# In[70]:


#accuracy of model
#70% to 80% data for training and other 20 to 30% for testing


# In[ ]:





# In[ ]:





# In[ ]:
