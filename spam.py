import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
#%%
df=pd.read_csv("emails.csv")

print(df.head())

print(df.info)
#%%
print(df.shape)

print(df.columns)
#%%
print(df.isnull().sum())

print(df.describe())
#%%
print(df.corr()) #statistical relationship
#%%
print(df.dropna(inplace = True)) #remove the missing value
#%%
df = df.drop(['Email No.'],axis=1)
X = df.drop(['Prediction'],axis = 1)
y = df['Prediction']

from sklearn.preprocessing import scale
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

#%%
print(X)
#%%
print(X_train)

#%%
print(X_test)
#%%
print(y_train)

#%%
print(y_test)

#%%
print(X_train.shape)

#%%
print(X_test.shape)

#%%
print(y_train.shape)

#%%
print(y_test.shape)

#%%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test) #perform prediction on the test set using predict()

print("Prediction",y_pred)

print("KNN accuracy = ",metrics.accuracy_score(y_test,y_pred))

#%%
print("Confusion matrix",metrics.confusion_matrix(y_test,y_pred)) #performance of classification algorithm

#%%
#support vector classifier

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("SVM accuracy = ",metrics.accuracy_score(y_test,y_pred))


cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

print(cm)




