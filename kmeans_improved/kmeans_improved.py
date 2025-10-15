import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats.mstats import winsorize

sns.set(style='whitegrid')
warnings.filterwarnings('ignore')

WORKDIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(WORKDIR, '..', 'gym_members_exercise_tracking.csv')

# Config
WINSORIZE_LIMIT = 0.005  # trim 0.5% on each tail
FEATURES = [
    'BMI',
    'Calories_Burned',
    'Session_Duration (hours)',
    'Workout_Frequency (days/week)',
    'Fat_Percentage',
    'Experience_Level'
]
K_RANGE = range(2, 7)
N_INIT = 50


def load_and_prepare(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # Keep only rows with the selected features present
    X = df[FEATURES].copy()
    X = X.dropna()

    return df, X


def winsorize_df(X, limits=WINSORIZE_LIMIT):
    Xw = X.copy()
    for col in X.columns:
        arr = X[col].values
        # winsorize requires finite numbers
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if np.all(np.isnan(arr)):
            continue
        # Use masked winsorize, then convert back
        try:
            warr = winsorize(arr, limits=(limits, limits))
            Xw[col] = np.array(warr)
        except Exception:
            # fallback: clip at percentiles
            lo = np.nanpercentile(arr, limits * 100)
            hi = np.nanpercentile(arr, 100 - limits * 100)
            Xw[col] = np.clip(arr, lo, hi)
    return Xw


def transform_features(X):
    Xt = X.copy()
    # reduce skew in Calories_Burned
    if 'Calories_Burned' in Xt.columns:
        Xt['Calories_Burned'] = np.log1p(Xt['Calories_Burned'])
    return Xt


def run_kmeans(X_scaled, k, n_init=N_INIT):
    km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    labels = km.fit_predict(X_scaled)
    return km, labels


def plot_elbow(Ks, inertias, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, inertias, '-o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow plot (improved)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_silhouette(Ks, silhouettes, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, silhouettes, '-o')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores (improved)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pca_clusters(X_scaled, labels, km, out_path):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(km.cluster_centers_)

    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab10', n_colors=len(np.unique(labels)))
    for i, color in enumerate(palette):
        idx = labels == i
        plt.scatter(X2[idx,0], X2[idx,1], s=30, color=color, label=f'cluster {i}', alpha=0.7, edgecolor='k', linewidth=0.2)
    plt.scatter(centers_2d[:,0], centers_2d[:,1], s=200, c='k', marker='X', label='centers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('KMeans (improved) PCA projection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_space(X_df, labels, km, scaler, fx, fy, out_path):
    # Scatter in original units
    x = X_df[fx].values
    y = X_df[fy].values

    # Build grid and predict using means for other features
    x_min, x_max = x.min() - 0.1*abs(x.max()), x.max() + 0.1*abs(x.max())
    y_min, y_max = y.min() - 0.1*abs(y.max()), y.max() + 0.1*abs(y.max())
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    mean_vals = X_df.mean()
    full_grid = np.tile(mean_vals.values, (grid_points.shape[0], 1))
    cols = list(X_df.columns)
    ix = cols.index(fx)
    iy = cols.index(fy)
    full_grid[:, ix] = grid_points[:,0]
    full_grid[:, iy] = grid_points[:,1]

    full_grid_scaled = scaler.transform(full_grid)
    Z = km.predict(full_grid_scaled).reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    palette = sns.color_palette('tab10', n_colors=len(np.unique(labels)))
    cmap = ListedColormap([palette[i] for i in range(len(np.unique(labels)))])

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, levels=np.arange(-0.5, len(np.unique(labels))+0.5,1), cmap=cmap, alpha=0.15)
    for i, color in enumerate(palette):
        idx = labels == i
        plt.scatter(x[idx], y[idx], s=30, color=color, label=f'cluster {i}', edgecolor='k', linewidth=0.2, alpha=0.85)

    # centroids in original units (inverse transform)
    cent_scaled = km.cluster_centers_
    cent_orig = scaler.inverse_transform(cent_scaled)
    centers_xy = np.array([[c[ix], c[iy]] for c in cent_orig])
    plt.scatter(centers_xy[:,0], centers_xy[:,1], s=200, c='k', marker='X', label='centers')

    plt.xlabel(fx)
    plt.ylabel(fy)
    plt.title(f'KMeans (improved): {fx} vs {fy}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print('Loading and preparing data...')
    df_full, X = load_and_prepare(CSV_PATH)

    print('Winsorizing outliers...')
    Xw = winsorize_df(X)

    print('Transforming features (log for calories)...')
    Xt = transform_features(Xw)

    print('Scaling with RobustScaler...')
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(Xt)

    results = []
    inertias = []
    silhouettes = []

    for k in K_RANGE:
        km, labels = run_kmeans(X_scaled, k)
        inertia = km.inertia_
        sil = None
        ch = None
        db = None
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
        inertias.append(inertia)
        silhouettes.append(sil if sil is not None else np.nan)
        results.append({'k': k, 'inertia': inertia, 'silhouette': sil, 'calinski_harabasz': ch, 'davies_bouldin': db, 'n_clusters': len(set(labels))})
        print(f'k={k} inertia={inertia:.2f} silhouette={sil} CH={ch} DB={db}')

    # Save results CSV
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(WORKDIR, 'kmeans_improved_results.csv')
    res_df.to_csv(out_csv, index=False)
    print(f'Results saved to {out_csv}')

    # Save diagnostic plots
    plot_elbow(list(K_RANGE), inertias, os.path.join(WORKDIR, 'elbow.png'))
    plot_silhouette(list(K_RANGE), silhouettes, os.path.join(WORKDIR, 'silhouette.png'))

    # Choose best k by silhouette (if available) otherwise inertia elbow
    sil_array = np.array([s if s is not None else np.nan for s in silhouettes])
    if not np.all(np.isnan(sil_array)):
        best_k = int(list(K_RANGE)[int(np.nanargmax(sil_array))])
    else:
        best_k = list(K_RANGE)[0]
    print(f'Chosen k = {best_k}')

    # Refit final model
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=N_INIT)
    labels_final = km_final.fit_predict(X_scaled)

    # Save labeled dataset
    df_full['cluster_improved'] = -1
    df_full.loc[X.index, 'cluster_improved'] = labels_final
    labeled_out = os.path.join(WORKDIR, 'gym_members_with_clusters_improved.csv')
    df_full.to_csv(labeled_out, index=False)
    print(f'Labeled data saved to {labeled_out}')

    # Save centroids (original units)
    cent_scaled = km_final.cluster_centers_
    cent_orig = scaler.inverse_transform(cent_scaled)
    cent_df = pd.DataFrame(cent_orig, columns=Xt.columns)
    cent_df['cluster'] = range(len(cent_df))
    cent_out = os.path.join(WORKDIR, f'centroids_k{best_k}_improved.csv')
    cent_df.to_csv(cent_out, index=False)
    print(f'Centroids saved to {cent_out}')

    # Save PCA + feature-space plots
    plot_pca_clusters(X_scaled, labels_final, km_final, os.path.join(WORKDIR, 'kmeans_pca.png'))
    plot_feature_space(Xt, labels_final, km_final, scaler, 'BMI', 'Calories_Burned', os.path.join(WORKDIR, 'kmeans_BMI_vs_Calories_Burned.png'))

    print('Done. Plots and CSVs are in the kmeans_improved folder.')

if __name__ == '__main__':
    main()
