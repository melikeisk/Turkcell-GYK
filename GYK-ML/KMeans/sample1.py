import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#müşteri gelir ve harcama listesi

x = np.array([[100, 200],
              [150, 250],
              [200, 300],
              [250, 350],
              [300, 400],
              [350, 450],
              [400, 500],
              [450, 550],
              [500, 600],
              [550, 650],
              [600, 700],
              [650, 750],
              [700, 800]])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)

labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='black')
plt.xlabel("Gelir")
plt.ylabel("Harcama")
plt.title("K-means ile Müşteri Segmentasyonu")
plt.show()
              