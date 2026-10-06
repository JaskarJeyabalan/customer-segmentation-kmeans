"""
Customer Segmentation using K-Means Clustering
==============================================
Segments Mall customers by Annual Income and Spending Score.

Run from the project root:
    python src/customer_segmentation.py

Outputs (saved to images/):
    elbow_method.png, silhouette_score.png, clusters.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed; figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 5
sns.set(style="whitegrid")
np.random.seed(RANDOM_STATE)

# Fixed segment IDs so cluster numbers always mean the same thing.
# K-Means numbers its clusters arbitrarily (it can change between runs or
# scikit-learn versions), so we name each cluster from its centroid values
# and then map it to these stable IDs.
SEGMENT_IDS = {
    "Moderate Income, Moderate Spending": 0,
    "High Income, Low Spending": 1,
    "Low Income, High Spending": 2,
    "Low Income, Low Spending": 3,
    "High Income, High Spending": 4,
}


def name_segment(income: float, spending: float) -> str:
    """Describe a cluster from its mean income (k$) and spending score."""
    if 40 <= income <= 70 and 40 <= spending <= 60:
        return "Moderate Income, Moderate Spending"
    inc = "High" if income > 70 else "Low"
    spend = "High" if spending > 50 else "Low"
    return f"{inc} Income, {spend} Spending"


def main() -> None:
    # ------------------------------------------------------------------------
    # 1. Load and validate
    # ------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    df = df.rename(
        columns={
            "Annual Income (k$)": "Annual Income",
            "Spending Score (1-100)": "Spending Score",
        }
    )

    # ------------------------------------------------------------------------
    # 2. Features and scaling
    # ------------------------------------------------------------------------
    features = ["Annual Income", "Spending Score"]
    X_scaled = StandardScaler().fit_transform(df[features])

    # ------------------------------------------------------------------------
    # 3. Elbow method + silhouette scores for k = 2..10
    # ------------------------------------------------------------------------
    k_range = range(2, 11)
    wcss, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette score: {best_k} ({max(sil_scores):.3f})")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker="o")
    plt.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"k = {N_CLUSTERS}")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.legend()
    plt.savefig(IMAGE_DIR / "elbow_method.png", bbox_inches="tight", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), sil_scores, marker="o")
    plt.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"k = {N_CLUSTERS}")
    plt.title("Silhouette Score for Different K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.savefig(IMAGE_DIR / "silhouette_score.png", bbox_inches="tight", dpi=120)
    plt.close()

    # ------------------------------------------------------------------------
    # 4. Final K-Means model
    # ------------------------------------------------------------------------
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    raw_labels = kmeans.fit_predict(X_scaled)
    centroids = pd.DataFrame(
        StandardScaler().fit(df[features]).inverse_transform(kmeans.cluster_centers_),
        columns=features,
    )

    # Map arbitrary K-Means labels -> named segments -> stable IDs
    raw_to_name = {
        i: name_segment(row["Annual Income"], row["Spending Score"])
        for i, row in centroids.iterrows()
    }
    if len(set(raw_to_name.values())) != N_CLUSTERS:
        raise RuntimeError(f"Segment naming was not unique: {raw_to_name}")

    df["Segment"] = pd.Series(raw_labels).map(raw_to_name)
    df["Cluster"] = df["Segment"].map(SEGMENT_IDS)
    centroids["Cluster"] = centroids.index.map(lambda i: SEGMENT_IDS[raw_to_name[i]])

    print("\nSilhouette score (k=5):", round(silhouette_score(X_scaled, raw_labels), 3))

    # ------------------------------------------------------------------------
    # 5. Cluster profile
    # ------------------------------------------------------------------------
    profile = (
        df.groupby(["Cluster", "Segment"])
        .agg(
            Age=("Age", "mean"),
            Annual_Income=("Annual Income", "mean"),
            Spending_Score=("Spending Score", "mean"),
            Customer_Count=("CustomerID", "count"),
        )
        .round(1)
    )
    print("\nCluster profile:\n", profile)

    # ------------------------------------------------------------------------
    # 6. Cluster plot with centroids
    # ------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="Annual Income",
        y="Spending Score",
        hue="Cluster",
        palette="Set2",
        s=90,
        edgecolor="black",
    )
    plt.scatter(
        centroids["Annual Income"],
        centroids["Spending Score"],
        color="black",
        marker="X",
        s=250,
        label="Centroids",
    )
    plt.title("Customer Segments", fontsize=14)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Cluster")
    plt.savefig(IMAGE_DIR / "clusters.png", bbox_inches="tight", dpi=120)
    plt.close()

    print(f"\nImages saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
