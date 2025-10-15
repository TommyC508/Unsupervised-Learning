import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")

WORKDIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(WORKDIR, '..', "gym_members_exercise_tracking.csv")

def load_and_prepare(path):
    df = pd.read_csv(path, skipinitialspace=True)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Choose numeric features for clustering
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Numeric columns detected ({len(num_cols)}): {num_cols}")

    X = df[num_cols].copy()
    # Keep only complete rows for clustering but preserve original index for alignment
    X_clean = X.dropna()

    return df, X_clean, num_cols


def find_best_k(X_scaled, k_min=2, k_max=8):
    inertias = []
    silhouettes = []
    Ks = list(range(k_min, k_max+1))

    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if len(set(labels)) > 1:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(np.nan)
        print(f"k={k} inertia={km.inertia_:.2f} silhouette={silhouettes[-1]}")

    return Ks, inertias, silhouettes


def plot_elbow(Ks, inertias, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, inertias, '-o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow plot')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_silhouette(Ks, silhouettes, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, silhouettes, '-o')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_clusters(X_scaled, labels, km, out_path):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(km.cluster_centers_)

    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab10', n_colors=len(np.unique(labels)))
    # Create a mesh over the PCA space and predict cluster for each point to shade regions
    x_min, x_max = X2[:, 0].min() - 1.0, X2[:, 0].max() + 1.0
    y_min, y_max = X2[:, 1].min() - 1.0, X2[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Predict cluster label for each point in the grid by inverse transforming through PCA
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # map back to scaled feature space for KMeans prediction
    grid_in_scaled = pca.inverse_transform(grid_points)
    try:
        Z = km.predict(grid_in_scaled)
    except Exception:
        # If prediction fails for any reason, fallback to no background shading
        Z = None

    # Plot shaded regions if prediction succeeded
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([sns.color_palette('tab10')[i] for i in range(len(np.unique(labels)))])
    if Z is not None:
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=np.arange(-0.5, len(np.unique(labels)) + 0.5, 1), cmap=cmap, alpha=0.15)

    # Scatter the actual points on top
    for i, color in enumerate(palette):
        idx = labels == i
        plt.scatter(X2[idx,0], X2[idx,1], s=30, color=color, label=f'cluster {i}', edgecolor='k', linewidth=0.2, alpha=0.8)

    plt.scatter(centers_2d[:,0], centers_2d[:,1], s=200, c='k', marker='X', label='centers')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('KMeans clusters (PCA projection)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_clusters_feature_space(X_df, labels, km, scaler, feature_x, feature_y, out_path):
     """Plot clusters using two actual features from the dataset.
*** End Patch