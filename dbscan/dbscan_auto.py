import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

sns.set(style='whitegrid')

WORKDIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(WORKDIR, '..', 'gym_members_exercise_tracking.csv')

# Feature subset to try (prefer these if present)
PREFERRED_FEATURES = ['BMI', 'Calories_Burned', 'Session_Duration (hours)']


def load_numeric(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.dropna()
    return df, X


def find_knee(y_sorted):
    # Use distance from line method to find elbow index
    n = len(y_sorted)
    if n < 3:
        return 0
    x = np.arange(n)
    # line between first and last
    x1, y1 = 0.0, float(y_sorted[0])
    x2, y2 = float(n - 1), float(y_sorted[-1])
    # distances
    num = np.abs((y2 - y1) * x - (x2 - x1) * y_sorted + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    d = num / (den + 1e-12)
    idx = int(np.argmax(d))
    return idx


def run():
    df, X_all = load_numeric(CSV_PATH)
    # choose subset
    features = [f for f in PREFERRED_FEATURES if f in X_all.columns]
    if len(features) < 2:
        # fallback to first two numeric features
        features = list(X_all.columns[:3])
    print(f"Using features: {features}")

    X = X_all[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k-distance plot (k = min_samples)
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8,4))
    plt.plot(k_distances)
    plt.xlabel('Points sorted')
    plt.ylabel(f'{k}-distance')
    plt.title(f'k-distance plot (k={k})')
    kdist_path = os.path.join(WORKDIR, 'dbscan_k_distance.png')
    plt.tight_layout()
    plt.savefig(kdist_path)
    plt.close()
    print(f"Saved k-distance plot to {kdist_path}")

    # find elbow
    idx = find_knee(k_distances)
    eps_candidate = float(k_distances[idx])
    print(f"Detected knee at index {idx} -> eps ~ {eps_candidate:.4f}")

    # Try a small set of eps around candidate
    eps_values = sorted(list({round(eps_candidate * factor, 6) for factor in [0.6, 0.8, 1.0, 1.2, 1.5]}))
    min_samples = k

    records = []
    results = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        pct_noise = 100.0 * n_noise / labels.size
        sil = None
        if n_clusters >= 2:
            try:
                sil = float(silhouette_score(X_scaled, labels))
            except Exception:
                sil = None
        records.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'n_noise': n_noise, 'pct_noise': pct_noise, 'silhouette': sil})
        results.append((eps, labels))
        print(f"eps={eps} clusters={n_clusters} noise={n_noise} pct_noise={pct_noise:.1f} silhouette={sil}")

    summary_df = pd.DataFrame(records)
    summary_out = os.path.join(WORKDIR, 'dbscan_auto_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved summary to {summary_out}")

    # choose best run by silhouette if available, else choose the run with >0 clusters and lowest pct_noise
    chosen_idx = None
    if summary_df['silhouette'].notnull().any():
        chosen_idx = int(summary_df['silhouette'].idxmax())
    else:
        # prefer runs with at least 1 cluster
        with_clusters = summary_df[summary_df['n_clusters'] > 0]
        if not with_clusters.empty:
            chosen_idx = int(with_clusters['pct_noise'].idxmin())
        else:
            chosen_idx = 0
    chosen_eps = float(summary_df.loc[chosen_idx, 'eps'])
    chosen_labels = results[chosen_idx][1]
    print(f"Chosen eps={chosen_eps} (index {chosen_idx})")

    # Save labeled dataset for chosen run
    out_df = df.copy()
    out_df['dbscan_auto_cluster'] = -1
    out_df.loc[X.index, 'dbscan_auto_cluster'] = chosen_labels
    out_csv = os.path.join(WORKDIR, 'gym_members_dbscan_auto_labels.csv')
    out_df.to_csv(out_csv, index=False)
    print(f"Saved labeled CSV to {out_csv}")

    # PCA visualization of chosen run
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab10', n_colors=max(3, len(set(chosen_labels))))
    for lab in sorted(set(chosen_labels)):
        mask = chosen_labels == lab
        color = 'k' if lab == -1 else palette[lab % len(palette)]
        label_name = 'noise' if lab == -1 else f'cluster {lab}'
        plt.scatter(X2[mask,0], X2[mask,1], s=30, color=color, label=label_name, alpha=0.7, edgecolor='k', linewidth=0.2)
    plt.legend()
    plt.title(f'DBSCAN (eps={chosen_eps}, min_samples={min_samples}) on features: {features}')
    out_pca = os.path.join(WORKDIR, 'dbscan_auto_pca.png')
    plt.tight_layout()
    plt.savefig(out_pca)
    plt.close()
    print(f"Saved PCA plot to {out_pca}")

    return summary_df, chosen_eps


if __name__ == '__main__':
    run()
