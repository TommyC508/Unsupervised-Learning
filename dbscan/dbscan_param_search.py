import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

sns.set(style='whitegrid')

WORKDIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(WORKDIR, '..', 'gym_members_exercise_tracking.csv')


def load_numeric(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.dropna()
    return df, X


def main():
    df, X = load_numeric(CSV_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_values = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    min_samples_vals = [3, 5, 8, 12]

    records = []
    for eps in eps_values:
        for ms in min_samples_vals:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            pct_noise = 100.0 * n_noise / labels.size
            sil = None
            if n_clusters >= 2:
                try:
                    sil = silhouette_score(X_scaled, labels)
                except Exception:
                    sil = None
            records.append({'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters, 'n_noise': int(n_noise), 'pct_noise': pct_noise, 'silhouette': sil})
            print(f"eps={eps} ms={ms} clusters={n_clusters} noise={n_noise} pct_noise={pct_noise:.1f} silhouette={sil}")

    res = pd.DataFrame(records)
    out_csv = os.path.join(WORKDIR, 'dbscan_param_search.csv')
    res.to_csv(out_csv, index=False)
    print(f"Wrote results to {out_csv}")

    # Pivot for heatmap (clusters)
    pivot = res.pivot(index='min_samples', columns='eps', values='n_clusters')
    plt.figure(figsize=(10,4))
    sns.heatmap(pivot, annot=True, fmt='g', cmap='viridis')
    plt.title('DBSCAN: number of clusters (min_samples vs eps)')
    plt.tight_layout()
    heatmap_path = os.path.join(WORKDIR, 'dbscan_param_search_heatmap_clusters.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved cluster heatmap to {heatmap_path}")

    pivot_noise = res.pivot(index='min_samples', columns='eps', values='pct_noise')
    plt.figure(figsize=(10,4))
    sns.heatmap(pivot_noise, annot=True, fmt='.1f', cmap='magma')
    plt.title('DBSCAN: % noise (min_samples vs eps)')
    plt.tight_layout()
    heatmap_noise_path = os.path.join(WORKDIR, 'dbscan_param_search_heatmap_noise.png')
    plt.savefig(heatmap_noise_path)
    plt.close()
    print(f"Saved noise heatmap to {heatmap_noise_path}")


if __name__ == '__main__':
    main()
