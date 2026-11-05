import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocessing(df,features):
  X = df[features].copy()

  numerical = X.select_dtypes(include='number').columns
  categorical = X.select_dtypes(exclude='number').columns

  encoder = LabelEncoder()
  for col in categorical:
    X[col] = encoder.fit_transform(X[col].astype(str))

  scaler = StandardScaler()
  X[numerical] = scaler.fit_transform(X[numerical])

  return X

def pca_reduction(X, n_components):
  pca = PCA(n_components=n_components)
  columns = ['PC' + str(i+1) for i in range(n_components)]
  X_pca = pd.DataFrame(pca.fit_transform(X), columns=columns)
  return X_pca, pca.explained_variance_ratio_

def kmeans_clustering(X, n_clusters):
  model = KMeans(n_clusters=n_clusters)
  model.fit(X)
  return model.labels_

def kmedoids_clustering(X, n_clusters, metric):
  model = KMedoids(n_clusters=n_clusters, metric=metric)
  model.fit(X)
  return model.labels_

def dbscan_clustering(X, eps, min_samples):
  model = DBSCAN(eps=eps,min_samples=min_samples)
  model.fit(X)
  return model.labels_

def clustering(X, n_clusters,linkage):
  model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
  model.fit(X)
  return model.labels_

def evaluate(X,labels):
  return silhouette_score(X, labels), davies_bouldin_score(X, labels)
