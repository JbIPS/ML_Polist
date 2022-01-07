#!/usr/bin/env python
# coding: utf-8
import datetime
import math
import os
import sys
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    calinski_harabasz_score, # Coherence in cluster, higher the better
    davies_bouldin_score, # Similarity between clusters, minimum 0, lower the better
    silhouette_score, # Intra-cluster distance vs inter-cluster distance. Between -1 and 1. higher the better
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

import mlflow
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.features import PCA as PCAViz

# Global constants
class Algorithms(Enum):
    KMEANS = 'kmeans'
    DBSCAN =  'dbscan'
    AGGLO = 'agglo'

SELECTED_COLUMNS = ['recency', 'frequency', 'monetary', 'mean_review']
AGGLO_SAMPLING_RATIO = 0.1 # Due to memory consumption, we need to sample the data

def preprocess(dataset: pd.DataFrame, scale: bool = True, log: bool = True) -> pd.DataFrame:
    """Add features engineering to the dataset"""
    data = dataset.dropna()
    dropped_features = data.columns.drop(SELECTED_COLUMNS)
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler() if scale else 'passthrough')])
    log_features = ['monetary']
    data = data.drop(data[(data[log_features] == 0).any(axis=1)].index)
    log_transformer = Pipeline(steps=[('log', FunctionTransformer(np.log) if log else 'passthrough'),
                                      ('numeric', numeric_transformer)])

    preprocessor = ColumnTransformer(
        transformers=[
            ('log', log_transformer, log_features),
            ('dropped', 'drop', dropped_features)
        ],
        remainder=numeric_transformer,
        n_jobs=-1
    )

    preprocessor.fit(data)
    columns = data.columns.drop(dropped_features)
    return pd.DataFrame(preprocessor.transform(data), columns=columns)

def draw_radar_plot(data: tuple[typing.Any], labels: list[str], merged: bool = True, nb_columns: int = 2, figsize: tuple[int, int] = (10,10)) -> plt.Figure:
    """ Draw a radar plot """
    scaler = MinMaxScaler()
    stats = scaler.fit_transform(data)

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # close the shape
    angles=np.concatenate((angles,[angles[0]]))
    first_stats = np.array([stat[0] for stat in stats]).reshape(-1,1)
    stats = np.append(stats, first_stats, axis=1)

    fig=plt.figure(figsize=figsize)
    for i, _ in enumerate(stats):
        stat = stats[i]
        if merged:
            ax = plt.subplot(polar=True)
            plt.legend(range(0, len(data)))
        else:
            ax = plt.subplot(math.ceil(len(stats)/nb_columns), nb_columns, i+1, polar=True)
        ax.plot(angles, stat, 'o-', linewidth=2)
        ax.fill(angles, stat, alpha=0.25)
        plt.thetagrids(angles[0:-1] * 180/np.pi, labels)
        ax.set_title(f'Cluster {i}')
    fig.suptitle("Clusters stats")
    fig.tight_layout()
    plt.close(fig)
    return fig

def create_clusters(X: pd.DataFrame, algorithm: Algorithms) -> None:
    """ Cluster data in X into nb_cluster """
    with mlflow.start_run(run_name=f"Clustering with {algorithm}", nested=True):
        if algorithm == Algorithms.KMEANS or algorithm == Algorithms.AGGLO:
            data = X if algorithm == Algorithms.KMEANS else X.sample(frac=AGGLO_SAMPLING_RATIO)
            visualizer = KElbowVisualizer(KMeans() if algorithm == Algorithms.KMEANS else AgglomerativeClustering(), k=(2,10))
            visualizer.fit(data)
            mlflow.log_figure(visualizer.fig, ARTIFACTS_FOLDER + f'/{algorithm} elbow.png')
            plt.close()
            elbow_value = visualizer.elbow_value_
            if not elbow_value:
                raise Exception(f"Cannot find {algorithm} elbow")

            for nb_cluster in range(elbow_value - 2, elbow_value + 3):
                exp_artifacts_folder = f'{ARTIFACTS_FOLDER}'
                if not os.path.exists(exp_artifacts_folder):
                    os.makedirs(exp_artifacts_folder)

                with mlflow.start_run(run_name=f'{algorithm} - {nb_cluster} clusters', nested=True):
                    mlflow.log_param('nb_cluster', nb_cluster)
                    model = KMeans(nb_cluster, random_state=3) if algorithm == Algorithms.KMEANS else AgglomerativeClustering(nb_cluster)
                    model.fit(data)

                    if algorithm == Algorithms.KMEANS:
                        # Visualize clusters silhouette
                        silhouette_visualizer = SilhouetteVisualizer(model, is_fitted=True)
                        silhouette_visualizer.fit(data)
                        mlflow.log_figure(silhouette_visualizer.fig, exp_artifacts_folder + f'/{algorithm} silhouttes.png')
                        plt.close()
                        mlflow.log_metric(f'{algorithm} silhouettes score', silhouette_visualizer.silhouette_score_)
                    else:
                        mlflow.log_metric(f'{algorithm} silhouettes score', silhouette_score(data, model.labels_))

                    # Visualize clusters on PCA projection
                    pca_visualizer = PCAViz(scale=False)
                    pca_visualizer.fit_transform(data, model.labels_)
                    mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER + f'/{algorithm} - {nb_cluster} cluster viz.png')
                    plt.close()

                    mlflow.log_metric(f'{algorithm} calinski score', calinski_harabasz_score(data, model.labels_))
                    mlflow.log_metric(f'{algorithm} davies score', davies_bouldin_score(data, model.labels_))

        else:
            # Get information on the distance between neighbors
            nb_neighbors = 10
            nearest_neighbors = NearestNeighbors(n_neighbors=nb_neighbors)
            neighbors = nearest_neighbors.fit(X)
            distances, _ = neighbors.kneighbors(X)

            # Get max distance between neighbors
            max_distances = np.sort(distances[:, nb_neighbors - 1])

            # Find an elbow
            index = np.arange(len(max_distances))
            knee = KneeLocator(index, max_distances, curve='convex', direction='increasing', interp_method='polynomial')
            knee.plot_knee(figsize=(10,10))
            plt.xlabel("Points")
            plt.ylabel("Distance")
            mlflow.log_figure(plt.gcf(), ARTIFACTS_FOLDER + '/DBSCAN elbow.png')
            plt.close()
            mlflow.log_metric('dbscan elbow', knee.elbow_y)
            model = DBSCAN(min_samples=100, eps=knee.elbow_y)
            model.fit(X)

            # Visualize clusters on PCA projection
            pca_visualizer = PCAViz(scale=False)
            pca_visualizer.fit_transform(X, model.labels_)
            mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER + f'/{algorithm} - {len(np.unique(model.labels_))} cluster viz.png')
            plt.close()

            mlflow.log_metric(f'{algorithm} silhouettes score', silhouette_score(X, model.labels_))
            mlflow.log_metric(f'{algorithm} calinski score', calinski_harabasz_score(X, model.labels_))
            mlflow.log_metric(f'{algorithm} davies score', davies_bouldin_score(X, model.labels_))

def analyze_clusters(data: pd.DataFrame, labels: pd.Series, title: str ="") -> None:
    """ Draw boxplots for each cluster and each features """
    data_labeled = data.reset_index(drop=True).merge(labels, left_index=True, right_index=True)
    grouped = data_labeled.groupby(labels.name)
    mlflow.log_dict(grouped.count().iloc[:, 0].to_dict(), ARTIFACTS_FOLDER + '/Clusters size.json')
    # Boxplots per cluster
    for i in range(0, model.n_clusters):
        fig, ax = plt.subplots()
        cluster = grouped.get_group(i)
        cluster.drop(columns=labels.name).plot.box(ax=ax, vert=False, subplots=True, layout=(4,1), sharex=False, title=f'Cluster {i} : {cluster.shape[0]} individus')
        plt.tight_layout()
        mlflow.log_figure(fig, ARTIFACTS_FOLDER + f'/[{title}] Cluster {i} boxplot.png')
        plt.close(fig)
        for idx, value in cluster.mean().drop(columns=labels.name).items():
            mlflow.log_metric(idx, value)

    # Boxplots per features
    for i in data_labeled.drop(columns=labels.name).columns:
        fig, ax = plt.subplots()
        grouped.boxplot(column=i, ax=ax, vert=False, subplots=False)
        mlflow.log_figure(fig, ARTIFACTS_FOLDER + f'/[{title}] {i} boxplot.png')
        plt.close(fig)

def split_periods(dataset: pd.DataFrame, period: int, time_column: str ="order_purchase_timestamp") -> list[pd.DataFrame]:
    """
    Split the dataset into periods based on periods given

        Parameters:
            data (DataFrame): DataFrame to split
            period (int): Number of days for each period
            time_column (str): Column label containing time info. Default to 'total_recency'

        Returns:
            data_periods (DataFrame[]): A list of DataFrame group by period defined by input
    """
    data = dataset.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    data_periods = []
    delta = datetime.timedelta(period)
    last_threshold = data[time_column].min()
    max_date = data[time_column].max()
    while(last_threshold < max_date):
        data_periods.append(data[(data[time_column] >= last_threshold)
                                 & (data[time_column] < last_threshold + delta)])
        last_threshold += delta

    return data_periods

def get_rfm(data: pd.DataFrame, time_column: str = 'order_purchase_timestamp', id_column: str = 'customer_unique_id', customer_id_column: str = 'customer_id', price_column: str = 'price') -> pd.DataFrame :
    """ Get RFM marketing data from input dataset """
    max_date = max(data[time_column]) + datetime.timedelta(days=1)
    rfm_data = data.groupby(id_column).agg({
            time_column: lambda x: (max_date - x.max()).days,
            customer_id_column: 'count',
            price_column: 'sum'
    })
    rfm_data.columns = ['recency','frequency','monetary']
    return rfm_data

def get_mean_review(dataset: pd.DataFrame, review_column: str = 'review_score', id_column: str = 'customer_unique_id', customer_id_column: str = 'customer_id') -> pd.Series:
    """ Get mean review for each unique customer """
    mean_review = dataset.dropna(subset=[review_column]).groupby([id_column, customer_id_column])[review_column].first()\
        .groupby(id_column).mean()
    mean_review.name = 'mean_review'
    return mean_review

if __name__ == "__main__":
    mlflow.set_experiment('POLIST')
    TIME_ANALYSIS = len(sys.argv) > 1 and sys.argv[1] == 'time_analysis'
    with mlflow.start_run():
        ARTIFACTS_FOLDER = 'outputs'
        if not os.path.exists(ARTIFACTS_FOLDER):
            os.makedirs(ARTIFACTS_FOLDER)

        if not TIME_ANALYSIS:
            dataset = pd.read_csv('./clean_dataset.csv', index_col=0)

            X = preprocess(dataset, False)

            NB_CLUSTER = int(sys.argv[1]) if len(sys.argv) > 1 else None

            if NB_CLUSTER is None:
              # Exploration
                pca_visualizer = PCAViz(proj_features=True)
                pca_visualizer.fit_transform(X)
                mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER + '/PCA.png')
                plt.close()

                for algo in Algorithms:
                    print(f'Creating clusters with {algo}')
                    create_clusters(X, algo)

                # TODO get dynamic cluster number
                NB_CLUSTER = 5

            # Get clusters details
            with mlflow.start_run(run_name="Clusters details", nested=True):
                print(f'Analyzing clusters with selected algo')
                mlflow.log_param('nb_cluster', NB_CLUSTER)
                model = KMeans(n_clusters=NB_CLUSTER, random_state=3)
                X = preprocess(dataset, False)

                step = 0
                for scale, log in [(False, False), (False, True), (True, False), (True, True)]:
                    data = preprocess(dataset, scale, log)
                    model.fit(data)

                    mlflow.log_metric(f'silhouettes score', silhouette_score(data, model.labels_))
                    mlflow.log_metric(f'calinski score', calinski_harabasz_score(data, model.labels_))
                    mlflow.log_metric(f'davies score', davies_bouldin_score(data, model.labels_))

                    silhouette_visualizer = SilhouetteVisualizer(model, is_fitted=True)
                    silhouette_visualizer.fit(data)
                    mlflow.log_figure(silhouette_visualizer.fig, ARTIFACTS_FOLDER + f'/{"Std " if scale else ""}{"Log " if log else ""}silhouttes.png')
                    plt.close(silhouette_visualizer.fig)

                # Visualize clusters on PCA projection
                pca_visualizer = PCAViz()
                pca_visualizer.fit_transform(X, model.labels_)
                mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER + '/Cluster viz.png')
                plt.close()

                # Get clusters stats
                fig = draw_radar_plot(model.cluster_centers_, X.columns.values, merged=False, figsize=(20,20))
                mlflow.log_figure(fig, ARTIFACTS_FOLDER + '/clusters_stats.png')

                # Draw boxplots for each features in cluster
                labels = pd.Series(model.labels_)
                labels.name = 'label'
                analyze_clusters(dataset[SELECTED_COLUMNS], labels, "Raw")
                analyze_clusters(X, labels, "Standard")

                minMaxed = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=SELECTED_COLUMNS)
                analyze_clusters(minMaxed, labels, "MinMaxed")

        else:
            # Temporal analysis
            NB_CLUSTER = 5
            dataset = pd.read_csv('./full_dataset.csv', index_col=0)
            periods = split_periods(dataset, 30)
            first_period = pd.concat(periods[0:6])
            periods = [first_period] + periods[6:]
            models = []
            for i,_ in enumerate(periods):
                period_data = pd.concat(periods[:i+1])
                period_rfm_data = get_rfm(period_data)
                period_mean_review = get_mean_review(period_data)

                period_dataset = pd.concat([period_rfm_data, period_mean_review], axis=1)
                X = preprocess(period_dataset, False)
                period_model = KMeans(n_clusters=NB_CLUSTER, random_state=3)
                period_model.fit(X)

                for model,aris in models:
                    ari = adjusted_rand_score(model.predict(X), period_model.labels_)
                    aris.append(ari)

                models.append((period_model, [None] * i))

            fig, ax = plt.subplots(figsize=(20,20))
            periods_index = [i for i,_ in enumerate(periods[1:])]
            for model, aris in models:
                plt.plot(periods_index, aris, '-o', figure=fig)
            ax.set_xticks([ str(i) + f'\n{periods[i].shape[0]}' for i in periods_index])
            plt.legend([i for i,_ in enumerate(models)])
            mlflow.log_figure(fig, ARTIFACTS_FOLDER + '/ARI.png')
            plt.close(fig)

        for file in os.listdir(ARTIFACTS_FOLDER):
            os.unlink(os.path.join(ARTIFACTS_FOLDER, file))
