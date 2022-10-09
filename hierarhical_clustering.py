import data_reader
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def build_graph():
    labels, data = data_reader.read_data()
    df = pdist(data)

    Z = linkage(df, method='ward')

    dendro = dendrogram(Z, labels=labels)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.show()

