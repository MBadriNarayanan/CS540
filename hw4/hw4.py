import csv
import itertools
import scipy
import matplotlib.pyplot as plt
import numpy as np


def get_complete_linkage_dist(cluster1, cluster2, distance_matrix):
    dist = -1
    for e1 in cluster1:
        for e2 in cluster2:
            dist = max(dist, distance_matrix[e1][e2])
    return dist


def load_data(filepath):
    country_data = []
    with open(filepath, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            country_data.append(row)
    return country_data


def calc_features(row):
    feature_arr = np.zeros(6)
    feature_arr[0] = float(row["Population"])
    feature_arr[1] = float(row["Net migration"])
    feature_arr[2] = float(row["GDP ($ per capita)"])
    feature_arr[3] = float(row["Literacy (%)"])
    feature_arr[4] = float(row["Phones (per 1000)"])
    feature_arr[5] = float(row["Infant mortality (per 1000 births)"])
    return feature_arr


def hac(features):
    n = np.shape(features)[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = scipy.spatial.distance.euclidean(
                features[i], features[j]
            )
    clusters = {}
    for i in range(n):
        clusters[i] = []
        clusters[i].append(i)
    hac_array = np.zeros((n - 1, 4))
    for i in range(n - 1):
        min_complete_linkage_dist = 1e30
        id1 = -1
        id2 = -1
        tie_indexes = set()
        distances = {}

        for (clusterID1, cluster1), (clusterID2, cluster2) in itertools.combinations(
            clusters.items(), 2
        ):
            complete_linkage_dist = get_complete_linkage_dist(
                cluster1, cluster2, distance_matrix
            )
            distances[(clusterID1, clusterID2)] = complete_linkage_dist
        min_complete_linkage_dist = min(distances.values())

        tie_indexes = {
            (min(clusterID1, clusterID2), max(clusterID1, clusterID2))
            for (clusterID1, clusterID2), dist in distances.items()
            if dist == min_complete_linkage_dist
        }

        tie_list = list(tie_indexes)
        tie_list.sort()
        id1 = tie_list[0][0]
        id2 = tie_list[0][1]
        l1 = clusters[id1]
        l2 = clusters[id2]
        clusters[n + i] = np.concatenate((np.array(l1), np.array(l2)))
        del clusters[id1]
        del clusters[id2]
        hac_array[i][0] = int(id1)
        hac_array[i][1] = int(id2)
        hac_array[i][2] = min_complete_linkage_dist
        hac_array[i][3] = int(len(clusters[n + i]))
    return hac_array


def fig_hac(Z, names):
    fig = plt.figure()
    dn = scipy.cluster.hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()
    plt.show()
    return fig


def normalize_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    features_normalized = (features - means) / std_devs
    return list(features_normalized)


if __name__ == "__main__":
    country_data = load_data("./countries.csv")
    feature_arr = calc_features(country_data[0])
    features = [calc_features(data) for data in country_data]
    features = np.array(features)

    features_normalized = np.array(normalize_features(features))
    n = 20
    h_base = hac(features[:n])
    names = [row["Country"] for row in country_data[:n]]
    h_normalized = hac(features_normalized[:n])
    fig_hac(h_normalized, names)
