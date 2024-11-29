import numpy as np
import matplotlib.pyplot as plt

# Data siswa
data = np.array([
    [60, 80],  # Siswa 1
    [63, 85],  # Siswa 2
    [70, 75],  # Siswa 3
    [65, 60],  # Siswa 4
    [80, 85],  # Siswa 5
    [75, 70]   # Siswa 6
])

# Centroid awal (siswa 1 dan siswa 4)
centroids = np.array([data[0], data[3]])

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-Means iterasi
def k_means(data, centroids, max_iter=100, tol=1e-4):
    for iteration in range(max_iter):
        # Step 1: Assign clusters
        clusters = []
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)  # Cluster terdekat
            clusters.append(cluster)
        
        # Step 2: Recompute centroids
        new_centroids = []
        clusters = np.array(clusters)  # Konversi ke array NumPy
        for cluster_idx in range(len(centroids)):
            cluster_points = data[clusters == cluster_idx]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[cluster_idx])  # Retain old centroid if no points
        
        new_centroids = np.array(new_centroids)
        
        # Check convergence (if centroids do not change significantly)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids

    return clusters, centroids

# Jalankan algoritma K-Means
clusters, final_centroids = k_means(data, centroids)

# Visualisasi hasil clustering
colors = ['red', 'blue']
for i, point in enumerate(data):
    plt.scatter(point[0], point[1], color=colors[clusters[i]], label=f"Siswa {i+1}")

for i, centroid in enumerate(final_centroids):
    plt.scatter(centroid[0], centroid[1], color=colors[i], marker='x', s=200, label=f"Centroid {i+1}")

plt.title("Hasil Clustering K-Means")
plt.xlabel("Nilai Matematika")
plt.ylabel("Nilai Bahasa")
plt.legend()
plt.grid()
plt.show()

# Hasil akhir cluster dan centroid
clusters, final_centroids
