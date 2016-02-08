__author__ = 'chachalaca'

import numpy as np
from functools import reduce

class KMeans:

    clusters_count = None
    cluster_centers = None

    def __init__(self, clusters_count):
        self.clusters_count = clusters_count

    def fit(self, data):
        self.cluster_centers = self._lloyd_k_means(data)
        return self

    def predict(self, data):
        if self.cluster_centers is None:
            raise RuntimeError(
                "fit() must be called before predict()."
            )
        return list(
            map(
                lambda item: self._get_nearest_center(item, self.cluster_centers),
                data
            )
        )

    def _check_data(self, data):
        if issubclass(np.dtype("float64").type, data.dtype.type) is not True:
            raise ValueError(
                "Data must be instance of float64, %d given." % (
                    data.dtype
                )
            )
        if data.shape[0] < self.clusters_count:
            raise ValueError(
                "Samples count (%d) must be higher or equal to clusters count (%d)." % (
                    data.shape[0],
                    self.clusters_count
                )
            )
        if len(data.shape) != 2:
            raise ValueError(
                "Wrong data shape (%d)." % (
                    data.shape
                )
            )

    def _lloyd_k_means(self, data):
        cluster_centers = {}
        for i in range(self.clusters_count):
            init_center = []
            for j in range(data.shape[1]):
                init_center.append(
                    np.random.uniform(
                        np.min(data[:,j]),
                        np.max(data[:,j])
                    )
                )
            cluster_centers[i] = np.array(init_center)

        clusters = {k: [] for k, v in cluster_centers.items()}

        for x in data:
            nearest_center = self._get_nearest_center(x, cluster_centers)
            clusters[nearest_center].append(x)

        cluster_centers = {c: np.mean(values, axis=0) for c, values in clusters.items()}

        while True:
            best_change = {
                "delta": 0,
                "cluster_to": None,
                "cluster_from": None,
                "x": None
            }
            for cluster_to in clusters.keys():
                for x in data:
                    if np.array(x) in np.array(clusters[cluster_to]):
                        continue

                    cluster_from = self._get_nearest_center(x, cluster_centers)

                    _clusters = {k: list(v) for k, v in clusters.items()}
                    _clusters[cluster_from] = [y for y in _clusters[cluster_from] if not np.array_equal(x, y)]
                    _clusters[cluster_to].append(x)

                    delta = self._get_objective_function_value(clusters, cluster_centers) \
                        - self._get_objective_function_value(_clusters, cluster_centers)

                    if delta > 0 and delta > best_change["delta"]:
                        best_change = {
                            "delta": delta,
                            "cluster_to": cluster_to,
                            "cluster_from": cluster_from,
                            "x": x
                        }

            if best_change["delta"] > 0:
                clusters[best_change["cluster_from"]].remove(best_change["x"])
                clusters[best_change["cluster_to"]].append(best_change["x"])

                cluster_centers[best_change["cluster_from"]] = np.mean(clusters[best_change["cluster_from"]], axis=0)
                cluster_centers[best_change["cluster_to"]] = np.mean(clusters[best_change["cluster_to"]], axis=0)
            else:
                return cluster_centers

    @staticmethod
    def _get_nearest_center(x, cluster_centers):
        distance = {}
        for k, c in cluster_centers.items():
            distance[k] = np.linalg.norm(x-c)

        return min(distance, key=distance.get)

    @staticmethod
    def _get_objective_function_value(clusters, cluster_centers):
        return sum([
            reduce(
                lambda x, y: x + np.linalg.norm(y-cluster_centers[c])**2,
                values,
                0
            )
            for c, values in clusters.items()
        ]) / len(clusters)




