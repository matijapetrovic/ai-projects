
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier, _tree


class DataVisualizer:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def describe_data(self):
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


def read_data(path):
    """
    Ucitavamo podatke i izbacujemo ID customer-a
    """
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("READING DATA")
    print("=============")
    df = pd.read_csv(path, sep=',')
    return df.drop(columns=['CUST_ID'])


def fill_na(df):
    """
    Uocavamo da kolona MINIMUM_PAYMENTS jedina ima na vrednosti
    S obzirom da broj na vrednosti nije prevelik, dopunjavamo medijanom za vrednosti tog atributa
    Biramo medijanu jer ovaj atribut ima velik broj outlier-a
    """
    missing = df.isna().sum()
    print(missing)
    return df.fillna(df.median())


def normalize_data(df):
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('dim_reduction', PCA(n_components=2, random_state=0))
    ])
    return pipeline.fit_transform(df)


def find_optimal_n_clusters(df):
    """
    Pomocu elbow metode trazimo optimalan broj klastera
    Plottujemo ukupnu gresku za brojeve klastera od 1 do 10 i gledamo u kom trenutku je najveci pad greske
    Zakljucujemo da je optimalan broj klastera 7/8 ?
    """
    #data = df.values
    plot_sse_for_cluster_num(df, 1, 15)


def plot_sse_for_cluster_num(data, start, end):
    print("PLOTTING SSE AGAINST NUM OF CLUSTERS")
    print("====================================")
    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(start, end):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=100, init="k-means++", n_init=10)
        kmeans.fit_predict(data)
        print("Num of iterations: ", kmeans.n_iter_)
        sum_squared_errors.append(kmeans.inertia_)

    plt.plot(sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('SSE')
    plt.show()


def describe_data(df):
    """
    Radimo eksplorativnu analizu podataka
    Tabelarno prikazujemo srednju vrednost, devijaciju i kvantile
    Plotujemo korelaciju izmedju atributa kako bismo videli koje mozemo da izbacimo
    """
    dv = DataVisualizer(df)
    dv.describe_data()
    # df.boxplot(rot=90, figsize=(20, 10))
    # plt.show()
    dv.plot_correlation()


def cluster(df, n_clusters):
    """
    Pomocu K-MEANS algoritma vrsimo klasterovanje na prethodno odredjen broj klastera
    Koristimo K-MEANS++ inicijalizaciju radi brze i tacnije konvergencije algoritma
    """
    #data = df.values
    return kmeans_cluster(df, n_clusters)


def kmeans_cluster(data, num_clusters):
    print("CLUSTERING DATA")
    print("===============")
    kmeans = KMeans(n_clusters=num_clusters, max_iter=100, init="k-means++", n_init=10)
    clusters = kmeans.fit_predict(data)
    print("Clustering finished in ", kmeans.n_iter_, " iterations")
    return clusters


def plot_clusters(df, clusters, n_clusters):
    print("PLOTTING CLUSTERS")
    print("================")
    sns.countplot(clusters).set_title('Cluster sizes', fontsize=20)
    plt.show()
    df["cluster"] = clusters
    #pairplot_clusters(df)
    distplot_clusters(df, n_clusters)


def pairplot_clusters(df):
    best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    best_cols.append("cluster")
    # cols1 = cols[:len(cols) // 2]
    # cols.append("clusters")
    # cols2 = cols[len(cols)//2:]
    # sns.pairplot(df[cols1], hue="clusters")
    sns.pairplot(df[best_cols], hue="cluster")
    plt.show()


def distplot_clusters(df, n_clusters):
    cols = df.columns

    n_cols = 17
    n_rows = len(cols) - 1 / n_cols
    for i in range(n_clusters):
        fig = plt.figure(figsize=(n_cols + 3, 5*n_rows + 3))
        dff = df[df['cluster'] == i]
        plt.subplots_adjust(0.06, 0.06, 0.90, 0.95, 0.33, 0.20)
        for j in range(len(cols)):
            plt.subplot(3, 6, j + 1)
            sns.distplot(pd.DataFrame(dff[cols.values[j]]))
            plt.xlabel(cols[j])
        fig.suptitle('Cluster # ' + str(i))
        plt.show()


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))
    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])


if __name__ == '__main__':
    df = read_data('credit_card_data.csv')
    df = fill_na(df)
    describe_data(df)

    X = normalize_data(df)
    find_optimal_n_clusters(X)
    clusters = cluster(X, n_clusters=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette='bright', ax=ax2)
    ax2.set(xlabel="pc1", ylabel="pc2", title="Wine clustering result")
    ax2.legend(title='cluster')
    cluster_report(df, clusters, min_samples_leaf=20, pruning_level=0.05)
    plt.show()
