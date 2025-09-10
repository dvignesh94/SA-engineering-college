import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_iris_data():
    print(" Loading Iris dataset for unsupervised learning...")
    data_path = "/Users/vignesh/Documents/GitHub/Generative_ai/Datasets/neural_networks/Iris.csv"
    data = pd.read_csv(data_path)
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = data[features]
    y_true = data['Species']  # Only for reference/plotting; not used for training
    print(f" Loaded {len(X)} samples with {len(features)} features")
    return X, y_true


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    print("🧹 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(" Features scaled")
    return X_scaled


def reduce_dimensions(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    print(" Reducing dimensions with PCA to 2 components for visualization...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f" PCA complete. Variance explained: {explained:.2%}")
    return X_pca


def run_kmeans(X: np.ndarray, n_clusters: int = 3):
    print(f" Running KMeans with k={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    inertia = kmeans.inertia_
    sil = silhouette_score(X, labels)
    print(f" KMeans done. Inertia: {inertia:.2f}, Silhouette Score: {sil:.4f}")
    return kmeans, labels, inertia, sil


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, title: str, filename: str):
    print(f"📊 Saving cluster plot: {filename}")
    plt.figure(figsize=(8, 6))
    palette = np.array(['#ff6b6b', '#4dabf7', '#51cf66'])
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=palette[labels], s=50, alpha=0.8, edgecolor='k')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/{filename}", dpi=300, bbox_inches='tight')
    plt.show()
    print(" Plot saved")


def elbow_method(X_scaled: np.ndarray, k_min: int = 1, k_max: int = 10):
    print("🧪 Running elbow method to choose k...")
    inertias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(ks, inertias, marker='o')
    plt.xticks(ks)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within-cluster SSE)')
    plt.title('Elbow Method for Choosing k')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/vignesh/Documents/GitHub/Generative_ai/Machine Learning/elbow_method.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(" Elbow plot saved as 'elbow_method.png'")


def simple_unsupervised_iris():
    print(" Simple Unsupervised Learning: Iris Clustering (KMeans)")
    print("=" * 50)

    # Step 1: Load data
    X, y_true = load_iris_data()

    # Step 2: Scale features
    X_scaled = preprocess_features(X)

    # Step 3: Optional elbow method to choose k
    elbow_method(X_scaled, k_min=1, k_max=8)

    # Step 4: PCA to 2D for visualization
    X_pca = reduce_dimensions(X_scaled, n_components=2)

    # Step 5: Cluster with KMeans
    kmeans, labels, inertia, sil = run_kmeans(X_scaled, n_clusters=3)

    # Step 6: Visualize clusters in 2D PCA space
    plot_clusters_2d(
        X_pca,
        labels,
        title=f'Iris KMeans Clusters (silhouette={sil:.2f})',
        filename='iris_kmeans_pca.png'
    )

    print(" Unsupervised example completed!")
    print("Key takeaways:")
    print("1. Unsupervised learning finds structure without labels.")
    print("2. Scaling and PCA help visualize and improve clustering.")
    print("3. Use elbow and silhouette to choose/assess k.")

if __name__ == "__main__":
    simple_unsupervised_iris()
