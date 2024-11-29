import numpy as np

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
    x1, y1 = point1
    x2, y2 = point2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Fungsi K-Means dengan langkah-langkah terperinci
def k_means_with_details(data, centroids, max_iter=100, tol=1e-4):
    print(f"____ Iterasi K-Means ____")
    print(f"Data siswa:\n{data}\n")
    print(f"Centroid awal:\n{centroids}\n")
    
    for iteration in range(1, max_iter + 1):
        print(f"\n---- Iterasi {iteration} ----")
        
        # Step 1: Assign clusters
        clusters = []
        print("Langkah 1: Hitung jarak ke setiap centroid dan tentukan cluster")
        for i, point in enumerate(data):
            distances = []
            print(f"Siswa {i+1} dengan koordinat {point}:")
            for idx, centroid in enumerate(centroids):
                dist = euclidean_distance(point, centroid)
                distances.append(dist)
                # Menampilkan perhitungan jarak Euclidean
                print(f"  Jarak ke Centroid {idx+1}: d = sqrt(({point[0]} - {centroid[0]})^2 + ({point[1]} - {centroid[1]})^2) = {dist:.2f}")
            cluster = np.argmin(distances)  # Cluster terdekat
            clusters.append(cluster)
            print(f"  Cluster yang ditentukan: {cluster + 1}\n")
        
        # Step 2: Recompute centroids
        print("Langkah 2: Hitung centroid baru berdasarkan titik-titik di masing-masing cluster")
        new_centroids = []
        clusters = np.array(clusters)  # Konversi ke array NumPy
        
        for cluster_idx in range(len(centroids)):
            cluster_points = data[clusters == cluster_idx]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
                new_centroids.append(new_centroid)
                print(f"  Cluster {cluster_idx + 1} (Titik: {cluster_points}):")
                print(f"  Centroid baru: {new_centroid}\n")
            else:
                new_centroids.append(centroids[cluster_idx])  # Retain old centroid if no points
                print(f"  Cluster {cluster_idx + 1} tidak memiliki titik. Centroid tetap: {centroids[cluster_idx]}")
        
        new_centroids = np.array(new_centroids)
        
        # Check convergence (if centroids do not change significantly)
        print("\nLangkah 3: Cek konvergensi")
        if np.allclose(centroids, new_centroids, atol=tol):
            print("\nCentroid tidak berubah secara signifikan. Iterasi selesai.")
            break
        
        centroids = new_centroids
        print(f"\nCentroid diperbarui:\n{centroids}\n")
    
    print(f"Hasil akhir setelah iterasi {iteration}:")
    print(f"Clusters: {clusters+1}")
    print(f"Centroid akhir:\n{centroids}")
    return clusters, centroids

# Jalankan algoritma K-Means dengan penjelasan detail
clusters, final_centroids = k_means_with_details(data, centroids)
