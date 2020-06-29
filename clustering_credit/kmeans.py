import numpy, random, copy, math


class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []

    def recalculate_center(self):
        new_center = [0 for i in range(len(self.center))]
        for d in self.data:
            for i in range(len(d)):
                new_center[i] += d[i]

        n = len(self.data)
        if n != 0:
            self.center = [x / n for x in new_center]


class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []
        self.dimensions = 0

    def fit(self, data, normalize=True, plus_plus=True):
        if normalize:
            data = self.normalize_data(data)
        self.data = data

        self.dimensions = len(self.data[0])
        if plus_plus:
            self.init_centers_plus_plus()
        else:
            self.init_centers_random()
        self.iterate()

    def init_centers_random(self):
        for i in range(self.n_clusters):
            point = [random.random() for x in range(self.dimensions)]
            self.clusters.append(Cluster(point))

    def init_centers_plus_plus(self):
        point = [random.random() for x in range(self.dimensions)]
        self.clusters.append(Cluster(point))

        for i in range(self.n_clusters - 1):
            max_dist = -math.inf
            max_idx = -1
            for (i, d) in enumerate(self.data):
                closest_center_dist = min(self.euclidean_distance(d, c.center) for c in self.clusters)
                if closest_center_dist > max_dist:
                    max_dist = closest_center_dist
                    max_idx = i
            self.clusters.append(Cluster(self.data[max_idx]))

    def iterate(self):
        iter_no = 0
        not_changed = False
        while iter_no <= self.max_iter and not not_changed:
            for cluster in self.clusters:
                cluster.data = []

            for d in self.data:
                cluster_index = self.predict(d)
                self.clusters[cluster_index].data.append(d)

            not_changed = True
            for cluster in self.clusters:
                old_center = copy.deepcopy(cluster.center)
                cluster.recalculate_center()
                not_changed = not_changed and (cluster.center == old_center)
            iter_no += 1

        print("Num of iterations: " + str(iter_no))

    def predict(self, datum):
        min_distance = math.inf
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = self.euclidean_distance(datum, self.clusters[index].center)
            if distance < min_distance:
                cluster_index = index
                min_distance = distance

        return cluster_index
    
    @staticmethod
    def euclidean_distance(x, y):
        return sum((yi - xi)**2 for xi, yi in zip(x, y)) ** 0.5

    @staticmethod
    def normalize_data(data):
        cols = len(data[0])
        for col in range(cols):
            column_data = []
            for row in data:
                column_data.append(row[col])
            mean = numpy.mean(column_data)
            std = numpy.std(column_data)
            for row in data:
                row[col] = (row[col] - mean) / std

        return data

    def sum_squared_error(self):
        return sum(self.euclidean_distance(cluster.center, d) for cluster in self.clusters for d in cluster.data) ** 2
