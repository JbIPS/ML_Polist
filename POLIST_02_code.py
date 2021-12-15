#!/usr/bin/env python
# coding: utf-8
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, StandardScaler)
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.features import PCA as PCAViz

def preprocess(data, nb_category):
    """Add features engineering to the dataset"""
    selected_columns = ['recency', 'frequency', 'monetary', 'mean_review']
    cleaned_dataset = data[data['mean_command_freight_value'] != 0]
    categorical_features = cleaned_dataset.select_dtypes(include=['category', 'object']).columns
    dropped_features = categorical_features[nb_category:].append(cleaned_dataset.columns.drop(selected_columns))
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    log_features = list(set(['mean_command_price', 'mean_command_freight_value', 'monetary']).intersection(selected_columns))
    log_transformer = Pipeline(steps=[('log', FunctionTransformer(np.log)), ('numeric', numeric_transformer)])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features[:nb_category]),
            ('log', log_transformer, log_features),
            ('dropped', 'drop', dropped_features)
        ],
        remainder=numeric_transformer,
        n_jobs=-1
    )

    preprocessor.fit(cleaned_dataset)
    # new_columns = preprocessor.named_transformers_['cat'].get_feature_names(categorical_features)
    # columns = pd.Index(new_columns).append(cleaned_dataset.columns.drop(dropped_features))
    columns = cleaned_dataset.columns.drop(dropped_features)
    return pd.DataFrame(preprocessor.transform(cleaned_dataset), columns=columns)

def draw_radar_plot(data, labels):
    """ Draw a radar plot """
    # centroids = model.cluster_centers_
    scaler = MinMaxScaler()
    # labels = X.columns.values
    # stats = centroids
    stats = scaler.fit_transform(data)

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # close the shape
    angles=np.concatenate((angles,[angles[0]]))
    first_stats = np.array([stat[0] for stat in stats]).reshape(-1,1)
    stats = np.append(stats, first_stats, axis=1)

    fig=plt.figure()
    for i in enumerate(stats):
        stat = stats[i]
        ax = plt.subplot(polar=True)
        ax.plot(angles, stat, 'o-', linewidth=2)
        ax.fill(angles, stat, alpha=0.25)
    plt.thetagrids(angles[0:-1] * 180/np.pi, labels)
    fig.suptitle("Clusters stats")
    plt.legend(range(0, len(data)))
    mlflow.log_figure(fig, exp_artifacts_folder + '/clusters_stats.png')
    plt.close()

if __name__ == "__main__":
    mlflow.set_experiment('POLIST')
    with mlflow.start_run():
        ARTIFACTS_FOLDER = 'outputs'
        if not os.path.exists(ARTIFACTS_FOLDER):
            os.makedirs(ARTIFACTS_FOLDER)
        dataset = pd.read_csv('./clean_dataset.csv', index_col=0)

        X = preprocess(dataset, 0)

        NB_CLUSTER = int(sys.argv[1]) if len(sys.argv) > 1 else None

        if NB_CLUSTER is None:
          # Exploration
            pca_visualizer = PCAViz(proj_features=True, scale=False)
            pca_visualizer.fit_transform(X)
            mlflow.log_figure(pca_visualizer.fig, ARTIFACTS_FOLDER + '/PCA.png')
            plt.close()

            visualizer = KElbowVisualizer(KMeans(), k=(2,10))
            visualizer.fit(X)
            mlflow.log_figure(visualizer.fig, ARTIFACTS_FOLDER + '/KMeans elbow.png')
            plt.close()
            elbow_value = visualizer.elbow_value_ if visualizer.elbow_value_ is not None else 4

            for nb_cluster in range(elbow_value - 2, elbow_value + 3):
                exp_artifacts_folder = f'{ARTIFACTS_FOLDER}'
                if not os.path.exists(exp_artifacts_folder):
                    os.makedirs(exp_artifacts_folder)
                with mlflow.start_run(run_name=f"Exploration {nb_cluster} clusters", nested=True):
                    mlflow.log_param('nb_cluster', nb_cluster)
                    model = KMeans(nb_cluster, random_state=3)
                    model.fit(X)

                    silhouette_visualizer = SilhouetteVisualizer(model)
                    silhouette_visualizer.fit(X)
                    mlflow.log_figure(silhouette_visualizer.fig, exp_artifacts_folder + '/silhouttes.png')
                    mlflow.log_metric('silhouettes score', silhouette_visualizer.silhouette_score_)
                    plt.close()

                    draw_radar_plot(model.cluster_centers_, X.columns.values)
        else:
            # Get clusters details
            with mlflow.start_run(run_name="Clusters details", nested=True):
                mlflow.log_param('nb_cluster', NB_CLUSTER)

        for file in os.listdir(ARTIFACTS_FOLDER):
            os.unlink(os.path.join(ARTIFACTS_FOLDER, file))
