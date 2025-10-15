import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    # Example DBSCAN run — tune eps and min_samples
    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN produced {n_clusters} clusters (plus noise label -1)")

    # Save labeled dataset
    df['dbscan_cluster'] = -1
    df.loc[X.index, 'dbscan_cluster'] = labels
    out_csv = os.path.join(WORKDIR, 'gym_members_dbscan_labels.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved labeled CSV to {out_csv}")

    # Quick PCA 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab10', n_colors=len(set(labels)))
    for lab in sorted(set(labels)):
        mask = labels == lab
        color = 'k' if lab == -1 else palette[lab % len(palette)]
        label_name = 'noise' if lab == -1 else f'cluster {lab}'
        plt.scatter(X2[mask,0], X2[mask,1], s=30, color=color, label=label_name, alpha=0.7)
    plt.legend()
    plt.title('DBSCAN clusters (PCA projection)')
    plt.tight_layout()
    plt.savefig(os.path.join(WORKDIR, 'dbscan_pca.png'))
    plt.close()


if __name__ == '__main__':
    main()
