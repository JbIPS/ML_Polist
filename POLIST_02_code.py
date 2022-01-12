#!/usr/bin/env python
# coding: utf-8

import datetime
from enum import Enum
import math
import os
import sys
import typing
from typing import cast

from kneed import KneeLocator
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    v_measure_score,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    StandardScaler
)
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.features import PCA as PCAViz


# Global constants
class Algorithms(Enum):
    KMEANS = 'kmeans'
    DBSCAN = 'dbscan'
    AGGLO = 'agglo'


SELECTED_COLUMNS = ['recency', 'frequency', 'monetary', 'mean_review']
# Due to memory consumption, we need to sample the data
AGGLO_SAMPLING_RATIO = 0.1


def preprocess(dataset: pd.DataFrame, scale: bool = True, log: bool = True) -> pd.DataFrame:
    """
        Add features engineering to the dataset.

        Parameters:
            dataset: Dataset to preprocess
            scale: Wether to use a StandardScaler on the data
            log: Wether to use a log transformation on data

        Returns:
            Preprocess data
    """
    # Extract RFM data from dataset
    rfm_data = get_rfm(dataset)
    mean_review = get_mean_review(dataset)

    data = pd.concat([rfm_data, mean_review], axis=1)
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler() if scale else 'passthrough')])
    log_features = ['monetary']
    data = data.drop(data[(data[log_features] == 0).any(axis=1)].index)
    data.dropna(inplace=True)
    log_transformer = Pipeline(steps=[('log', FunctionTransformer(np.log) if log else 'passthrough'),
                                      ('numeric', numeric_transformer)])
    preprocessor = log_transformer

    preprocessor.fit(data.values)
    return pd.DataFrame(preprocessor.transform(data.values), columns=SELECTED_COLUMNS)


def draw_radar_plot(data: tuple[typing.Any], labels: list[str], merged: bool = True, nb_columns: int = 2, figsize: tuple[int, int]=(10, 10)) -> plt.Figure:
    """
        Draw a radar plot.

        Parameters:
            data: Point coordinates to draw on radar
            labels: Labels of the axes
            merged: Wether to draw all points on the same radar plot
            nb_columns: If multiple plot, select a number of columns for the layout
            figsize: Tuple of figure size passed to pyplot

        Returns:
            The radar plot figure
    """
    scaler = MinMaxScaler()
    stats = scaler.fit_transform(data)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # Close the shape
    angles = np.concatenate((angles, np.array([angles[0]])))
    first_stats = np.array([stat[0] for stat in stats]).reshape(-1, 1)
    stats = np.append(stats, first_stats, axis=1)

    fig = plt.figure(figsize=figsize)
    for i, _ in enumerate(stats):
        stat = stats[i]
        if merged:
            ax = plt.subplot(polar=True)
            plt.legend(range(0, len(data)))
        else:
            ax = plt.subplot(math.ceil(len(stats)/nb_columns),
                             nb_columns, i+1, polar=True)
        ax.plot(angles, stat, 'o-', linewidth=2)
        ax.fill(angles, stat, alpha=0.25)
        plt.thetagrids(angles[0:-1] * 180/np.pi, labels)
        ax.set_title(f'Cluster {i}')
    fig.suptitle("Clusters stats")
    fig.tight_layout()
    return fig


def create_clusters(X: pd.DataFrame, algorithm: Algorithms) -> None:
    """
        Use selected algorithm to cluster the data in X.

        Parameters:
            X: Data to run the algorithm on
            algorithm: Clustering algorithm. Currently support K-Means, AgglomerativeClustering and DBSCAN
    """
    with mlflow.start_run(run_name=f"Clustering with {algorithm.value}", nested=True):
        if algorithm == Algorithms.KMEANS or algorithm == Algorithms.AGGLO:
            data = X if algorithm == Algorithms.KMEANS else X.sample(
                frac=AGGLO_SAMPLING_RATIO)
            visualizer = KElbowVisualizer(KMeans(
            ) if algorithm == Algorithms.KMEANS else AgglomerativeClustering(), k=(2, 10))
            visualizer.fit(data.values)
            mlflow.log_figure(
                visualizer.fig, ARTIFACTS_FOLDER + f'/{algorithm.value} elbow.png')
            plt.close()
            elbow_value = visualizer.elbow_value_
            if not elbow_value:
                raise Exception(f"Cannot find {algorithm.value} elbow")

            for nb_cluster in range(min(elbow_value - 2, 2), max(elbow_value + 3, 8)):
                exp_artifacts_folder = f'{ARTIFACTS_FOLDER}'
                if not os.path.exists(exp_artifacts_folder):
                    os.makedirs(exp_artifacts_folder)

                with mlflow.start_run(run_name=f'{algorithm.value} - {nb_cluster} clusters', nested=True):
                    mlflow.log_param('nb_cluster', nb_cluster)
                    model = KMeans(
                        nb_cluster, random_state=3) if algorithm == Algorithms.KMEANS else AgglomerativeClustering(nb_cluster)
                    model.fit(data.values)

                    if algorithm == Algorithms.KMEANS:
                        # Visualize clusters silhouette
                        silhouette_visualizer = SilhouetteVisualizer(
                            model, is_fitted=True)
                        silhouette_visualizer.fit(data.values)
                        mlflow.log_figure(
                            silhouette_visualizer.fig, exp_artifacts_folder + f'/{algorithm.value} silhouttes.png')
                        plt.close()
                        mlflow.log_metric(
                            'silhouettes score', silhouette_visualizer.silhouette_score_)
                    else:
                        mlflow.log_metric(
                            'silhouettes score', silhouette_score(data, model.labels_))

                    # Visualize clusters on PCA projection
                    pca_visualizer = PCAViz()
                    pca_visualizer.fit_transform(data.values, model.labels_)
                    mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER +
                                      f'/{algorithm.value} - {nb_cluster} cluster viz.png')
                    plt.close()

                    mlflow.log_metric(
                        f'calinski score', calinski_harabasz_score(data, model.labels_))
                    mlflow.log_metric(
                        f'davies score', davies_bouldin_score(data, model.labels_))

        else:
            # Get information on the distance between neighbors
            nb_neighbors = 10
            nearest_neighbors = NearestNeighbors(n_neighbors=nb_neighbors)
            nearest_neighbors.fit(X.values)
            distances, _ = nearest_neighbors.kneighbors(X.values)

            # Get max distance between neighbors
            max_distances = np.sort(distances[:, nb_neighbors - 1])

            # Find an elbow
            index = np.arange(len(max_distances))
            knee = KneeLocator(index, max_distances, curve='convex',
                               direction='increasing', interp_method='polynomial')
            knee.plot_knee(figsize=(10, 10))
            plt.xlabel("Points")
            plt.ylabel("Distance")
            mlflow.log_figure(plt.gcf(), ARTIFACTS_FOLDER +
                              '/DBSCAN elbow.png')
            plt.close()
            mlflow.log_metric('dbscan elbow', cast(float, knee.elbow_y))
            model = DBSCAN(min_samples=100, eps=knee.elbow_y)
            model.fit(X.values)

            # Visualize clusters on PCA projection
            pca_visualizer = PCAViz()
            pca_visualizer.fit_transform(X.values, model.labels_)
            mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER +
                              f'/{algorithm.value} - {len(np.unique(model.labels_))} cluster viz.png')
            plt.close()

            mlflow.log_metric('silhouettes score',
                              silhouette_score(X, model.labels_))
            mlflow.log_metric('calinski score',
                              calinski_harabasz_score(X, model.labels_))
            mlflow.log_metric('davies score',
                              davies_bouldin_score(X, model.labels_))


def analyze_clusters(data: pd.DataFrame, labels: pd.Series, title: str = "") -> None:
    """
        Draw boxplots for each cluster and each features.

        Parameters:
            data: Data to analyze
            labels: Clusters labels assign to data
            title: Title of the plots. Default to ""
    """
    data_labeled = data.reset_index(drop=True).merge(
        labels, left_index=True, right_index=True)
    grouped = data_labeled.groupby(labels.name)
    mlflow.log_dict(grouped.count().iloc[:, 0].to_dict(
    ), ARTIFACTS_FOLDER + '/Clusters size.json')
    # Boxplots per cluster
    for i in range(0, model.n_clusters):
        # fig, ax = plt.subplots()
        cluster = grouped.get_group(i)
        cluster.drop(columns=labels.name).plot.box(vert=False, subplots=True, layout=(
            4, 1), sharex=True, title=f'Cluster {i} : {cluster.shape[0]} individus')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), ARTIFACTS_FOLDER +
                          f'/[{title}] Cluster {i} boxplot.png')
        plt.close(plt.gcf())
        for idx, value in cluster.mean().drop(columns=labels.name).items():
            mlflow.log_metric(idx, value)

    # Boxplots per features
    for i in data_labeled.drop(columns=labels.name).columns:
        fig, ax = plt.subplots()
        grouped.boxplot(column=i, ax=ax, vert=False, subplots=False)
        mlflow.log_figure(fig, ARTIFACTS_FOLDER +
                          f'/[{title}] {i} boxplot.png')
        plt.close(fig)


def split_periods(dataset: pd.DataFrame, period: int, time_column: str = "order_purchase_timestamp") -> list[pd.DataFrame]:
    """
        Split the dataset into periods based on periods given

        Parameters:
            data (DataFrame): DataFrame to split
            period (int): Number of days for each period
            time_column (str): Column label containing time info. Default to 'total_recency'

        Returns:
            data_periods (DataFrame[]): A list of DataFrame group by period defined by input
    """
    data_periods = []
    delta = datetime.timedelta(period)
    last_threshold = dataset[time_column].min()
    max_date = dataset[time_column].max()
    while(last_threshold < max_date):
        data_periods.append(dataset[(dataset[time_column] >= last_threshold) & (dataset[time_column] < last_threshold + delta)])
        last_threshold += delta

    return data_periods


def get_rfm(dataset: pd.DataFrame, time_column: str = 'order_purchase_timestamp', id_column: str = 'customer_unique_id', customer_id_column: str = 'customer_id', price_column: str = 'price') -> pd.DataFrame:
    """
        Get RFM marketing data from input dataset

        Parameters:
            dataset: Dataset to extract RFM data from
            time_column: Name of the column containing time data. Default to `order_purchase_timestamp`
            id_column: Name of the column containing a unique ID. Default to `customer_unique_id`
            customer_id_column: Name of the column containing ordering ID. Default to `customer_id`
            price_column: Name of the column containing price data. Default to `price`

        Returns:
            A DataFrame with recency, frequency and monetary data
    """
    max_date = max(dataset[time_column]) + datetime.timedelta(days=1)
    rfm_data = dataset.groupby(id_column).agg({
        time_column: lambda x: (max_date - x.max()).days,
        customer_id_column: 'count',
        price_column: 'sum'
    })
    rfm_data.columns = ['recency', 'frequency', 'monetary']
    return rfm_data


def get_mean_review(dataset: pd.DataFrame, review_column: str = 'review_score', id_column: str = 'customer_unique_id', customer_id_column: str = 'customer_id') -> pd.Series:
    """
        Get mean review for each unique customer

        Parameters:
            dataset: Dataset to extract mean review from
            review_column: Name of the column containing review data. Default to `review_score`
            id_column: Name of the column containing a unique ID. Default to `customer_unique_id`
            customer_id_column: Name of the column containing ordering ID. Default to `customer_id`

        Returns:
            A Series with the mean review for each customer
    """
    mean_review = dataset.dropna(subset=[review_column]).groupby([id_column, customer_id_column])[review_column].first()\
        .groupby(id_column).mean()
    mean_review.name = 'mean_review'
    return mean_review


if __name__ == "__main__":
    TIME_ANALYSIS = len(sys.argv) > 1 and sys.argv[1] == 'time_analysis'
    with mlflow.start_run():
        ARTIFACTS_FOLDER = 'outputs'
        if not os.path.exists(ARTIFACTS_FOLDER):
            os.makedirs(ARTIFACTS_FOLDER)

        # Load dataset and parse timestamps
        dataset = pd.read_csv('./full_dataset.csv', index_col=0,
                              parse_dates=['order_purchase_timestamp'])

        if not TIME_ANALYSIS:
            # Try to read cluster number from CLI
            NB_CLUSTER = int(sys.argv[1]) if len(sys.argv) > 1 else None

            if NB_CLUSTER is None:
                X = preprocess(dataset)

                # Exploration
                pca_visualizer = PCAViz(proj_features=True)
                pca_visualizer.fit_transform(X.values)
                mlflow.log_figure(pca_visualizer.fig,
                                  ARTIFACTS_FOLDER + '/PCA.png')
                plt.close()

                for algo in Algorithms:
                    print(f'Creating clusters with {algo.value}')
                    create_clusters(X, algo)

                # TODO get dynamic cluster number
                NB_CLUSTER = 5

            # Get clusters details
            with mlflow.start_run(run_name="Clusters details", nested=True):
                print(f'Analyzing clusters with selected algo')
                mlflow.log_param('nb_cluster', NB_CLUSTER)
                model = KMeans(n_clusters=NB_CLUSTER, random_state=3)
                X = preprocess(dataset)

                for scale, log in [(True, False), (True, True)]:
                    data = preprocess(dataset, scale, log)
                    model.fit(data.values)

                    mlflow.log_metric(f'silhouettes score',
                                      silhouette_score(data, model.labels_))
                    mlflow.log_metric(
                        f'calinski score', calinski_harabasz_score(data, model.labels_))
                    mlflow.log_metric(
                        f'davies score', davies_bouldin_score(data, model.labels_))

                    silhouette_visualizer = SilhouetteVisualizer(
                        model, is_fitted=True)
                    silhouette_visualizer.fit(data.values)
                    mlflow.log_figure(silhouette_visualizer.fig, ARTIFACTS_FOLDER +
                                      f'/{"Std " if scale else ""}{"Log " if log else ""}silhouttes.png')
                    plt.close(silhouette_visualizer.fig)

                # Visualize clusters on PCA projection
                pca_visualizer = PCAViz()
                pca_visualizer.fit_transform(X.values, model.labels_)
                mlflow.log_figure(pca_visualizer.fig,
                                  ARTIFACTS_FOLDER + '/Cluster viz.png')
                plt.close(pca_visualizer.fig)

                # Get clusters stats
                fig = draw_radar_plot(
                    model.cluster_centers_, X.columns.values, merged=False, figsize=(20, 20))
                mlflow.log_figure(fig, ARTIFACTS_FOLDER +
                                  '/clusters_stats.png')
                plt.close(fig)

                # Draw boxplots for each features in cluster
                labels = pd.Series(model.labels_)
                labels.name = 'label'
                analyze_clusters(preprocess(dataset, False, False), labels, "Raw")
                analyze_clusters(preprocess(dataset), labels, "Standard")

                minMaxed = pd.DataFrame(
                    MinMaxScaler().fit_transform(X.values), columns=SELECTED_COLUMNS)
                analyze_clusters(minMaxed, labels, "MinMaxed")

        # Temporal analysis
        NB_CLUSTER = 5
        periods = split_periods(dataset, 30)
        first_period = pd.concat(periods[0:6])
        periods = [first_period] + periods[6:]
        models = []
        Xs = []

        # Compute consistency scores between periods
        for i, _ in enumerate(periods):
            period_data = pd.concat(periods[:i+1])
            X = preprocess(period_data)
            period_model = KMeans(n_clusters=NB_CLUSTER, random_state=3)
            period_model.fit(X.values)

            for j, (model, aris, v_mesures, accuracies) in enumerate(models):
                y_true = period_model.labels_
                y_pred = model.predict(X.values)
                # Adjusted Rand Score, label agnostic
                ari = adjusted_rand_score(y_true, y_pred)
                # mlflow.log_metric('ARI', ari, j)
                aris.append(ari)

                # V-measure, also label agnostic
                v_measure = v_measure_score(y_true, y_pred)
                # mlflow.log_metric('V-mesure', v_measure, j)
                v_mesures.append(v_measure)

                # Accuracy, we need to match clusters label between period
                conf_matrix = confusion_matrix(y_true, y_pred)
                rows = []
                for k, row in enumerate(conf_matrix):
                    argmax = row.argmax()
                    copy = y_pred.copy()
                    rows.append(np.where(copy == k, argmax, 0))

                y_matched = np.array(rows).sum(axis=0)
                acc = cast(float, accuracy_score(y_true, y_matched))
                # mlflow.log_metric('Accuracy', acc, j)
                accuracies.append(acc)

            models.append((period_model, [None] * i, [None] * i, [None] * i))
            Xs.append(X)

        fig, ax = plt.subplots(figsize=(40, 20))
        periods_index = [i for i, _ in enumerate(periods[1:])]
        for model, aris, _, _ in models:
            plt.plot(periods_index, aris, '-o', figure=fig)

        ax.set_ylim(0, 1)
        ax.set_xticks(periods_index)
        for i, xpos in enumerate(ax.get_xticks()):
            ax.text(
                xpos, -0.02, f"Period pop.\n{str(Xs[i].shape[0])}", size=10, ha='center')

        plt.legend([i for i, _ in enumerate(models)])
        plt.title("Evolution of ARI over time")
        mlflow.log_figure(fig, ARTIFACTS_FOLDER + '/ARI.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(40, 20))
        periods_index = [i for i, _ in enumerate(periods[1:])]
        for model, _, v_measure, _ in models:
            plt.plot(periods_index, v_measure, '-o', figure=fig)

        ax.set_ylim(0, 1)
        plt.legend([i for i, _ in enumerate(models)])
        ax.set_xticks(periods_index)
        plt.title("Evolution of V-mesure over time")
        mlflow.log_figure(fig, ARTIFACTS_FOLDER + '/V-mesure.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(40, 20))
        periods_index = [i for i, _ in enumerate(periods[1:])]
        for model, _, _, accuracy in models:
            plt.plot(periods_index, accuracy, '-o', figure=fig)

        ax.set_ylim(0, 1)
        plt.legend([i for i, _ in enumerate(models)])
        ax.set_xticks(periods_index)
        plt.title("Evolution of Accuracy over time")
        mlflow.log_figure(fig, ARTIFACTS_FOLDER + '/accuracy.png')
        plt.close(fig)

    for file in os.listdir(ARTIFACTS_FOLDER):
        os.unlink(os.path.join(ARTIFACTS_FOLDER, file))
