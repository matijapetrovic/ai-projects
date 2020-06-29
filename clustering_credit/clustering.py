from kmeans import KMeans
import matplotlib.pyplot as plt
import pandas
import seaborn as sns


class DataReader():
    def __init__(self, path):
        self.path = path

    def read_data(self):
        pandas.set_option("display.max_rows", None, "display.max_columns", None)
        print("READING DATA")
        print("=============")
        return pandas.read_csv(self.path, sep=',')


class DataVisualizer():
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def describe_data(self):
        print(self.data_frame.head())
        print(self.data_frame.shape)
        print(self.data_frame.info())
        print(self.data_frame.describe())

    def plot_correlation(self):
        k = 18  # number of variables for heatmap
        cm = self.data_frame.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, cmap='viridis')
        plt.show()

    def plot_outliers(self):
        l = self.data_frame.columns.values
        number_of_columns = 18
        number_of_rows = len(l) - 1 / number_of_columns
        plt.figure(figsize=(number_of_columns, 5 * number_of_rows))
        for i in range(1, len(l)):
            plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
            sns.set_style('whitegrid')
            sns.boxplot(self.data_frame[l[i]], color='green', orient='v')
            plt.tight_layout()
        plt.show()


    def plot_skewness(self):
        l = self.data_frame.columns.values
        number_of_columns = 18
        number_of_rows = len(l) - 1 / number_of_columns
        plt.figure(figsize=(2 * number_of_columns, 5 * number_of_rows))
        for i in range(1, len(l)):
            plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
            sns.distplot(self.data_frame[l[i]], kde=True)
        plt.show()


def kmeans_cluster(data, num_clusters):
    print("CLUSTERING DATA")
    print("===============")
    kmeans = KMeans(n_clusters=num_clusters, max_iter=100)
    kmeans.fit(data, normalize=True)
    return kmeans


def plot_clusters(kmeans):
    print("PLOTTING RESULTS")
    print("================")
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'black'}
    plt.figure()
    for idx, cluster in enumerate(kmeans.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
        for datum in cluster.data:  # iscrtavanje tacaka
            plt.scatter(datum[0], datum[1], c=colors[idx])
    plt.xlabel('Balance')
    plt.ylabel('Purchases')
    plt.show()


def plot_sse_for_cluster_num(data, start, end):
    print("PLOTTING SSE AGAINST NUM OF CLUSTERS")
    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(start, end):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
        kmeans.fit(data)
        sse = kmeans.sum_squared_error()
        sum_squared_errors.append(sse)

    plt.plot(sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('SSE')
    plt.show()


def extract_data(data_frame):
    return data_frame.values[:, 1:3]


def cluster(n_clusters, data):
    kmeans = kmeans_cluster(data, n_clusters)
    plot_clusters(kmeans)


if __name__ == '__main__':
    dr = DataReader('credit_card_data.csv')
    df = dr.read_data().dropna(axis=0)

    data = extract_data(df)

    dv = DataVisualizer(df)
    dv.describe_data()
    dv.plot_correlation()

    #plot_sse_for_cluster_num(data, 2, 10)
    cluster(3, data)
