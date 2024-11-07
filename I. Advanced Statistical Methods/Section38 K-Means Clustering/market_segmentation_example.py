import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('3.12.+Example.csv')
print(data)

# Plot the data

s = data['Satisfaction']
l = data['Loyalty']

plt.scatter(s,l)
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

x = data.copy()
kmeans = KMeans(3)
kmeans.fit(x)

clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
print(x_scaled)

wcss = []

for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1, 10), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans_new = KMeans(2)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
print(clusters_new)

plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
