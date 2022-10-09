import numpy as np
from scipy.spatial.distance import cdist
import data_reader
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def build_graph():
    labels, data = data_reader.read_data()
    K = range(1, len(data) + 1)
    KM = (KMeans(n_clusters=k).fit(data) for k in K)
    centroids = (k.cluster_centers_ for k in KM)

    D_k = (cdist(data, cent, 'euclidean') for cent in centroids)  # расстояния до кластерных центров
    dist = (np.min(D, axis=1) for D in D_k)  # минимальные расстояния до кластерных центров
    avg_within = [sum(d) / data.shape[0] for d in dist]  # среднее расстояние до центра кластера

    plt.plot(K, avg_within, 'bs-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
