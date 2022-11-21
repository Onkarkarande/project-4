import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("sales_data_sample.csv")

df.head()

#information of the data
df.info()

#get the statistical info of the data
df.describe()

#check for null values
df.isnull().sum()

#correlation matrix
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)
plt.show()

#K means Clustering
from sklearn.cluster import KMeans
X=df[["PRICEEACH","SALES"]]

wcss=[]
for i in range(1, 11):
        km = KMeans(n_clusters=i)
        km.fit(X)
        wcss.append(km.inertia_)
plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss,color="green",linewidth=2)
plt.xlabel("K")
plt.ylabel("WCSS")
plt.grid()
plt.show()

km_model=KMeans(n_clusters=4)
km_model.fit(X)
y_pred = km_model.predict(X)

X['Target']=y_pred
X.head()

sns.scatterplot(X.PRICEEACH,X.SALES, hue=X.Target,palette=['red','orange','blue','green'
])
plt.title("Price of Each vs Total Sales")
plt.show()

#Hierarchical clustering

#Find the optimal k value using dendrogram graph
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(X, method="ward"))
plt.title("Dendrogram Plot")
plt.ylabel("Euclidean Distances")
plt.xlabel("Sales")
plt.show()

#train tge model
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(X.drop("Target",axis='columns'))

X = X.drop("Target",axis='columns')

X = X.values

X

plt.figure(figsize=(10,7))
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c="blue",label="Cluster 2")
plt.title("Clusters of Sales")
plt.xlabel("PRICEEACH")
plt.xlabel("SALES")
plt.show()

