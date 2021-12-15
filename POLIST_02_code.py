#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans

from functions import display_scree_plot
from yellowbrick.features import PCA as PCAViz
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import mlflow


def preprocess(dataset, nb_category):
		"""Add features engineering to the dataset"""
		selected_columns = ['recency', 'frequency', 'monetary', 'mean_review']
		cleaned_dataset = dataset[dataset['mean_command_freight_value'] != 0]
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

if __name__ == "__main__":
		ARTIFACTS_FOLDER = 'outputs'
		if not os.path.exists(ARTIFACTS_FOLDER):
				os.makedirs(ARTIFACTS_FOLDER)
		mlflow.set_experiment('POLIST')
		dataset = pd.read_csv('./clean_dataset.csv', index_col=0)
		# for i in range(0, 7):
		X = preprocess(dataset, 0)
		# with mlflow.start_run():
				# mlflow.log_param("nb_categorial_feature", 0)

		# Log an artifact (output file)

		pca_visualizer = PCAViz(proj_features=True, scale=False)
		pca_visualizer.fit_transform(X)
		pca_visualizer.show(outpath=ARTIFACTS_FOLDER + '/PCA.png')
		plt.close()

		visualizer = KElbowVisualizer(KMeans(), k=(2,10))
		visualizer.fit(X)
		visualizer.show(outpath=ARTIFACTS_FOLDER + '/KMeans elbow.png')
		# mlflow.log_metric('elbow value', visualizer.elbow_value_)
		plt.close()

		for nb_cluster in range(visualizer.elbow_value_ - 2, visualizer.elbow_value_ + 3):
				exp_artifacts_folder = f'{ARTIFACTS_FOLDER}'
				if not os.path.exists(exp_artifacts_folder):
						os.makedirs(exp_artifacts_folder)
				with mlflow.start_run():
						mlflow.log_param('nb_cluster', nb_cluster)
						model = KMeans(nb_cluster, random_state=3)
						model.fit(X)

						silhouette_visualizer = SilhouetteVisualizer(model)
						silhouette_visualizer.fit(X)
						silhouette_visualizer.show(outpath=exp_artifacts_folder + '/silhouttes.png')
						mlflow.log_metric('silhouettes score', silhouette_visualizer.silhouette_score_)
						plt.close()

						centroids = model.cluster_centers_
						scaler = MinMaxScaler()
						labels = X.columns.values
						stats = centroids
						stats = scaler.fit_transform(stats)

						angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
						# close the plot
						angles=np.concatenate((angles,[angles[0]]))
						first_stats = np.array([stat[0] for stat in stats]).reshape(-1,1)
						stats = np.append(stats, first_stats, axis=1)

						fig=plt.figure()
						for i in range(0, len(stats)):
								stat = stats[i]
								ax = plt.subplot(polar=True)
								ax.plot(angles, stat, 'o-', linewidth=2)
								ax.fill(angles, stat, alpha=0.25)
						plt.thetagrids(angles[0:-1] * 180/np.pi, labels)
						fig.suptitle("Clusters stats")
						plt.legend(range(0, len(centroids)))
						plt.savefig(exp_artifacts_folder + '/clusters_stats.png')
						plt.close()

						mlflow.log_artifacts(ARTIFACTS_FOLDER)
						mlflow.log_artifacts(exp_artifacts_folder)
						for file in os.listdir(exp_artifacts_folder):
							os.unlink(os.path.join(exp_artifacts_folder, file))





						# À première vue, nous pouvons découper les clients en 5 groupes :
						# 1. Les clients "habitués" effectuant beaucoup de petites commandes peu coûteuses (en jaune)
						# 2. Les clients "mécontants" effectuant peu de commande d'un prix modéré mais à fort risque d'annulation (en vert)
						# 3. Les clients "à crédit" effectuant peu de commande d'un prix moyen mais payant à crédit (en rouge)
						# 4. Les clients "dépensier" effectuant peu de commande mais contenant beaucoup d'article pour un prix conséquent (en bleu)

						# # TODO
						# * Spider graph pour tracer les centroids de chaque cluster
						# * Utiliser la PCA pour la visualisation mais pas pour le clustering
						# * Faire le clustering avec que les variables numérique puis ajouter une à une les categorielles et voir le comportement
