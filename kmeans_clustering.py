import data_reader
from scipy.spatial.distance import squareform, pdist
from sklearn import manifold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def build_graph():
    labels, data = data_reader.read_data()
    data_d = squareform(pdist(data))

    mds = manifold.MDS(dissimilarity='precomputed')
    coords = mds.fit_transform(data_d)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(coords)
    y_kmeans = kmeans.predict(coords)

    plt.scatter(coords[:, 0], coords[:, 1], c=y_kmeans, s=30, cmap='Set1')
    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=50, alpha=0.7)

    for c, coord in zip(labels, coords):
        plt.annotate(c, xy=coord, xytext=coord + 10)

    plt.show()

    print_info(kmeans)


def print_info(kmeans):
    row_data = data_reader.get_row_data()
    row_data[:, 0] = kmeans.labels_
    for i in range(kmeans.n_clusters):
        members = []
        n = 0
        sums = []
        for z in range(len(row_data[0]) - 2):
            sums.append(0)
        print('Cluster {}:'.format(i))
        for val in row_data:
            if int(val[0]) == i:
                members.append(val[1])
                n += 1
                res = (s + int(v) for s, v in zip(sums, val[2:]))
        print('Members: {}'.format(members))
        print('Avg values: {}'.format(list(r / n for r in res)))
        print('-' * 100)
