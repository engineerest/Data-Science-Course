from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('3.01.+Country+clusters.csv')

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

# DataFrame.iloc(row indices, column indices) slices the data frame, given rows and columns to be kept

x = data.iloc[:,1:3]

kmeans = KMeans(3)
kmeans.fit(x)

# Clustering results

indentified_clusters = kmeans.fit_predict(x)
print(indentified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = indentified_clusters
print(indentified_clusters)

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

# Map the data

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})
print(data_mapped)

x = data_mapped.iloc[:,3:4]
print(x)

kmeans = KMeans(3)
kmeans.fit(x)

# Clustering results

# Cluster

indentified_clusters = kmeans.fit_predict(x)
print(indentified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = indentified_clusters
print(indentified_clusters)

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

# WCSS
print(kmeans.inertia_)

wcss = []

for i in range(1, 7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
print(wcss)

number_clusters = range(1, 7)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')