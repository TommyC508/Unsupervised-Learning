Improved KMeans pipeline (winsorize + RobustScaler + feature subset)

Run the script to run a small experiment (k=2..6, n_init=50) and save diagnostics:

Files produced by the script:
- `kmeans_improved_results.csv` — metrics per k
- `elbow.png`, `silhouette.png` — diagnostic plots
- `kmeans_pca.png` — PCA 2D cluster plot
- `kmeans_BMI_vs_Calories_Burned.png` — feature-space plot
- `gym_members_with_clusters_improved.csv` — original dataset with cluster labels
- `centroids_kX_improved.csv` — centroids for chosen k

How to run:

```bash
cd kmeans_improved
python3 kmeans_improved.py
```

The script uses RobustScaler and winsorizes numeric columns at 0.5%/99.5% by default. It also log-transforms `Calories_Burned` to reduce skew.