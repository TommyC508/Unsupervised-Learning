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
CSV_PATH = os.path.join(WORKDIR, "gym_members_exercise_tracking.csv")

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

    To approximate decision regions, other features are fixed to their column mean.
    """
    # Ensure features exist
    if feature_x not in X_df.columns or feature_y not in X_df.columns:
        raise ValueError(f"Features {feature_x} and/or {feature_y} not found in data columns")

    # Prepare scatter data (original units)
    x_vals = X_df[feature_x].values
    y_vals = X_df[feature_y].values

    plt.figure(figsize=(8,6))

    # Create mesh in the two-feature space
    x_min, x_max = x_vals.min() - 0.1 * abs(x_vals.max()), x_vals.max() + 0.1 * abs(x_vals.max())
    y_min, y_max = y_vals.min() - 0.1 * abs(y_vals.max()), y_vals.max() + 0.1 * abs(y_vals.max())
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Build full feature vectors for each grid point by using mean values for other features
    mean_vals = X_df.mean()
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    full_grid = np.tile(mean_vals.values, (grid_points.shape[0], 1))
    # set the two requested feature columns to the grid values
    ix = list(X_df.columns).index(feature_x)
    iy = list(X_df.columns).index(feature_y)
    full_grid[:, ix] = grid_points[:, 0]
    full_grid[:, iy] = grid_points[:, 1]

    # Scale and predict
    full_grid_scaled = scaler.transform(full_grid)
    try:
        Z = km.predict(full_grid_scaled)
        Z = Z.reshape(xx.shape)
    except Exception:
        Z = None

    from matplotlib.colors import ListedColormap
    palette = sns.color_palette('tab10', n_colors=len(np.unique(labels)))
    cmap = ListedColormap([palette[i] for i in range(len(np.unique(labels)))])

    if Z is not None:
        plt.contourf(xx, yy, Z, levels=np.arange(-0.5, len(np.unique(labels)) + 0.5, 1), cmap=cmap, alpha=0.15)

    # Scatter actual points colored by cluster
    for i, color in enumerate(palette):
        idx = labels == i
        plt.scatter(x_vals[idx], y_vals[idx], s=30, color=color, label=f'cluster {i}', edgecolor='k', linewidth=0.2, alpha=0.85)

    # Plot centroids for these two features (inverse scale centroids)
    centroids_scaled = km.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centers_xy = np.array([[c[ix], c[iy]] for c in centroids_orig])
    plt.scatter(centers_xy[:,0], centers_xy[:,1], s=200, c='k', marker='X', label='centers')

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'KMeans clusters on {feature_x} vs {feature_y}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("Loading data and preparing features...")
    df_full, X, num_cols = load_and_prepare(CSV_PATH)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute diagnostics (kept for reference) but we will allow forcing k via env
    print("Finding diagnostics for k=2..8 (inertia + silhouette)")
    Ks, inertias, silhouettes = find_best_k(X_scaled, 2, 8)
    plot_elbow(Ks, inertias, os.path.join(WORKDIR, 'elbow.png'))
    plot_silhouette(Ks, silhouettes, os.path.join(WORKDIR, 'silhouette.png'))

    # Allow forcing k via environment variable KMEANS_K, otherwise default to 3
    env_k = os.environ.get('KMEANS_K')
    if env_k is not None:
        try:
            best_k = int(env_k)
            print(f"Using KMEANS_K from environment: {best_k}")
        except Exception:
            best_k = 3
            print(f"Invalid KMEANS_K value '{env_k}', falling back to k=3")
    else:
        best_k = 3
        print("No KMEANS_K set, forcing k=3 as requested")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    # save clusters plot
    plot_clusters(X_scaled, labels, km, os.path.join(WORKDIR, 'kmeans_clusters.png'))

    # also save a feature-space plot using two actual features (BMI vs Calories_Burned)
    try:
        feature_plot_path = os.path.join(WORKDIR, 'kmeans_clusters_BMI_vs_Calories_Burned.png')
        plot_clusters_feature_space(X, labels, km, scaler, 'BMI', 'Calories_Burned', feature_plot_path)
        print(f"Feature-space plot saved: {feature_plot_path}")
    except Exception as e:
        print(f"Could not create feature-space plot: {e}")

    # Create multiple feature-pair plots
    def sanitize_name(s: str) -> str:
        return s.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

    pairs = [
        ('Weight (kg)', 'BMI'),
        ('Session_Duration (hours)', 'Calories_Burned'),
        ('Fat_Percentage', 'BMI'),
        ('Workout_Frequency (days/week)', 'Session_Duration (hours)'),
        ('Avg_BPM', 'Max_BPM'),
        ('Water_Intake (liters)', 'Calories_Burned')
    ]

    for fx, fy in pairs:
        try:
            fname = f"kmeans_{sanitize_name(fx)}_vs_{sanitize_name(fy)}.png"
            outp = os.path.join(WORKDIR, fname)
            plot_clusters_feature_space(X, labels, km, scaler, fx, fy, outp)
            print(f"Saved feature plot: {outp}")
        except Exception as e:
            print(f"Skipping {fx} vs {fy}: {e}")

    # Attach cluster labels back to the original dataframe. Rows dropped due to NaN will get -1
    df_full['cluster'] = -1
    df_full.loc[X.index, 'cluster'] = labels
    out_csv = os.path.join(WORKDIR, 'gym_members_with_clusters.csv')
    df_full.to_csv(out_csv, index=False)

    # Save centroids in original feature space (inverse of scaling)
    centroids_scaled = km.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_orig, columns=num_cols)
    centroids_df['cluster'] = range(len(centroids_df))
    centroids_out = os.path.join(WORKDIR, f'centroids_k{best_k}.csv')
    centroids_df.to_csv(centroids_out, index=False)

    # Print brief cluster summaries
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes (for clustered rows):")
    for u, c in zip(unique, counts):
        print(f" - cluster {u}: {c} rows")

    print(f"Centroids saved to: {centroids_out}")
    print(f"Labeled dataset saved to: {out_csv}")
    print("Plots saved:")
    print(" - elbow.png")
    print(" - silhouette.png")
    print(" - kmeans_clusters.png")

if __name__ == '__main__':
    main()
