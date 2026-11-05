import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, pca_reduction, kmeans_clustering, kmedoids_clustering, dbscan_clustering, clustering, evaluate

st.title('Clustering App')

st.header('Data Input')
file = st.file_uploader('upload your dataset',type=['csv'])

if file is not None:
    try:
        df = pd.read_csv(file, sep=None, engine='python')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep=None, engine='python', encoding='latin-1')
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

st.header('Feature Selection')
mode = st.radio('Select feature mode',['Manual Selection','PCA Reduction'])

if mode == 'Manual Selection':
    features = st.multiselect('Select the features for clustering',df.columns.to_list())
    if len(features) < 2:
        st.warning('Select at least 2 features')
        st.stop()
else:
    features = st.multiselect('Select the features for PCA',df.columns.to_list(),default=df.columns.to_list())
    if len(features) < 2:
        st.warning('Select at least 2 features')
        st.stop()
    n_components = st.slider('Number of components',2,len(features),2)

st.header('Preprocessing')
df = df.dropna(subset=features).reset_index(drop=True)
X = preprocessing(df,features)

if mode == 'PCA Reduction':
    X, variance = pca_reduction(X,n_components)
    st.write('Explained variance',variance)
    st.write('Total explained variance',variance.sum())

st.dataframe(X.head())

st.header('Model Selection')
algorithm = st.selectbox('Select clustering algorithm',['K-Means','K-Medoids','DBSCAN','Hierarchical Clustering'])

st.header('Hyperparameter Tuning')
if algorithm == 'K-Means':
    n_cluster = st.slider('n_clusters',2,10,3)
    labels = kmeans_clustering(X,n_clusters=n_cluster)
elif algorithm == 'K-Medoids':
    n_cluster = st.slider('n_clusters',2,10,3)
    metric = st.selectbox('metric',['euclidean','manhattan'])
    labels = kmedoids_clustering(X,n_clusters=n_cluster,metric=metric)
elif algorithm == 'DBSCAN':
    eps = st.slider('eps',0.1,5.0,0.5,0.1)
    min_samples = st.slider('min_samples',2,20,5)
    labels = dbscan_clustering(X,eps=eps,min_samples=min_samples)
else:
    n_cluster = st.slider('n_clusters',2,10,3)
    linkage = st.selectbox('linkage',['ward','complete','average','single'])
    labels = clustering(X,n_clusters=n_cluster,linkage=linkage)

st.header('Output')
st.subheader('Cluster Labels')
st.write(labels)
st.write(pd.Series(labels).value_counts().sort_index())

df['cluster'] = labels
st.subheader('Clustered Dataset')
st.dataframe(df)

st.header('Visualization')
if X.shape[1] == 2:
    X_plot = X
else:
    X_plot, variance_plot = pca_reduction(X,2)

fig, ax = plt.subplots()
scatter = ax.scatter(X_plot.iloc[:,0], X_plot.iloc[:,1], c=labels, cmap='viridis')
ax.set_xlabel(X_plot.columns[0])
ax.set_ylabel(X_plot.columns[1])
ax.set_title(algorithm)
ax.legend(*scatter.legend_elements(), title='Cluster')
st.pyplot(fig)

st.header('Evaluation')
if len(set(labels)) < 2:
    st.warning('Only one cluster found, try other parameters')
    st.stop()

silhoutte, davis = evaluate(X,labels)
st.write('Silhouette',silhoutte)
st.write('Davies Bouldin',davis)
