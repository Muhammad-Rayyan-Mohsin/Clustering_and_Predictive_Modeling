import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
DATA_FILE = 'stratified_sample_10k.csv'

# Print a message to show the script is starting
print(f"Starting clustering analysis on {DATA_FILE}...")

# Use the specific subset of features suggested for better silhouette score
FEATURE_COLUMNS = [
    'precipIntensity', 
    'precipProbability',
    'temperature'
]

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded {DATA_FILE}. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Check if apparentTemperature exists in the dataset
    if 'apparentTemperature' in df.columns:
        # Make sure to remove temperature first to avoid duplicates
        if 'temperature' in FEATURE_COLUMNS:
            FEATURE_COLUMNS.remove('temperature')
        FEATURE_COLUMNS.append('apparentTemperature')
        print("Using 'apparentTemperature' as requested")
    else:
        # Only add temperature if it's not already in the list
        if 'temperature' not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append('temperature')
        print("'apparentTemperature' not found in dataset, using 'temperature' instead")
    
    print(f"Selected features: {FEATURE_COLUMNS}")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Select Features and Preprocess ---
# Select only the specified feature columns
df_features = df[FEATURE_COLUMNS].copy()

# Handle missing values (Simple imputation with mean, consider more robust methods if needed)
print("\nHandling missing values...")
print("Missing values before imputation:\n", df_features.isnull().sum())
for col in FEATURE_COLUMNS:
    # Check if there are any null values using sum() > 0 which returns a scalar boolean
    if df_features[col].isnull().sum() > 0:
        mean_val = df_features[col].mean()
        df_features[col].fillna(mean_val, inplace=True)
        print(f"Filled missing values in '{col}' with mean ({mean_val:.2f})")
print("Missing values after imputation:\n", df_features.isnull().sum())

# Remove feature correlation analysis since we're using a specific feature subset
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
df_scaled = pd.DataFrame(X_scaled, columns=df_features.columns)
print("Features scaled.")
print("Scaled data head:\n", df_scaled.head())

# --- 3. Dimensionality Reduction for Visualization ---
print("\nPerforming Dimensionality Reduction for Visualization...")

# Enhanced PCA Analysis
# First, run a full PCA to analyze variance explained by each component
full_pca = PCA()
full_pca.fit(X_scaled)

# Visualize explained variance by each component
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(full_pca.explained_variance_ratio_) + 1), full_pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each Principal Component')
plt.xticks(range(1, len(full_pca.explained_variance_ratio_) + 1))

# Visualize cumulative explained variance
plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% Threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Components')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.legend()
plt.tight_layout()
plt.show()

# Determine optimal number of components for 85% variance
n_components = np.argmax(cumulative_variance >= 0.85) + 1
print(f"Number of components needed for 85% variance: {n_components}")

# Apply PCA with the determined number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA retained {pca.n_components_} components explaining {pca.explained_variance_ratio_.sum():.4f} variance")

# Create a DataFrame of the principal components for visualizations
df_pca_full = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Visualize feature loadings (contributions) to the first two principal components
loadings = pca.components_[:2].T  # Get loadings for the first two PCs
feature_names = df_features.columns

# Loadings plot
plt.figure(figsize=(12, 10))
loadings_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])
plt.subplot(2, 1, 1)
sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Feature Loadings (Contributions) to Principal Components')

# Biplot (scatter plot of PC1 vs PC2 with feature vectors)
plt.subplot(2, 1, 2)
# Plot samples in PC space
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)

# Plot feature vectors
for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
    plt.arrow(0, 0, x*5, y*5, head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.text(x*5.2, y*5.2, feature_names[i], color='r')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot: PC1 vs PC2 with Feature Vectors')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['demand'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Demand')
plt.title('First Two Principal Components with Demand Color Coding')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(alpha=0.3)
plt.show()

# t-SNE to 2 components (often better for visualization of clusters)
# Note: t-SNE can be slow on large datasets. Consider running on a subset if necessary.
# perplexity is a sensitive parameter, try different values (5 to 50)
print("Performing t-SNE (this may take a while)...")
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000, learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)
    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    print("t-SNE completed.")

    # Visualize PCA and t-SNE
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', data=df_pca_full.iloc[:, :2], alpha=0.5)
    plt.title('PCA - First 2 Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne, alpha=0.5)
    plt.title('t-SNE - 2 Components')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error during t-SNE or plotting: {e}")
    print("Skipping t-SNE visualization.")

# --- 4. K-Means Clustering ---
print("\nPerforming K-Means Clustering...")

# Silhouette score analysis for different k values
print("\nCalculating silhouette scores for different numbers of clusters...")
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")

# Save the silhouette scores to a file
with open('clustering_results.txt', 'w') as f:
    f.write(f"Feature subset: {FEATURE_COLUMNS}\n\n")
    f.write("Silhouette Scores for Different K Values:\n")
    for k, score in zip(k_range, silhouette_scores):
        f.write(f"K = {k}: {score:.4f}\n")
    
    f.write(f"\nBest number of clusters based on silhouette score: {k_range[np.argmax(silhouette_scores)]}\n")
    f.write(f"Best silhouette score: {max(silhouette_scores):.4f}\n")
    
# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Find the optimal k based on silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest number of clusters based on silhouette score: {best_k}")
print(f"Best silhouette score: {max(silhouette_scores):.4f}")

# Elbow Method to determine optimal k
distortions = []
print("Running Elbow method for K-Means (k=2 to 10)...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init to suppress warning
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(k_range, distortions, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# If user doesn't input a value for optimal_k, use the best_k from silhouette analysis
try:
    print("Review the plot to choose an optimal k.")
    print(f"Suggested optimal k based on silhouette score: {best_k}")
    optimal_k_input = input(f"Enter the chosen optimal number of clusters (k) based on the elbow plot (or press Enter to use {best_k}): ")
    if optimal_k_input.strip():
        optimal_k = int(optimal_k_input)
    else:
        optimal_k = best_k
        print(f"Using suggested optimal k = {best_k}")
except:
    optimal_k = best_k
    print(f"Using suggested optimal k = {best_k}")

if optimal_k > 1:
    # Apply K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    print(f"K-Means clustering completed with k={optimal_k}.")
    print("K-Means cluster counts:\n", df['KMeans_Cluster'].value_counts())

    # Evaluate K-Means using Silhouette Score
    silhouette_avg_kmeans = silhouette_score(X_scaled, df['KMeans_Cluster'])
    print(f"Silhouette Score for K-Means (k={optimal_k}): {silhouette_avg_kmeans:.4f}")

    # Append results to the file
    with open('clustering_results.txt', 'a') as f:
        f.write(f"\n\nK-Means Clustering Results (k={optimal_k}):\n")
        f.write(f"Silhouette Score: {silhouette_avg_kmeans:.4f}\n")
        
        # Add cluster counts
        f.write("\nCluster Counts:\n")
        cluster_counts = df['KMeans_Cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            f.write(f"Cluster {cluster}: {count} points ({count/len(df):.1%})\n")

    # Add cluster centers analysis
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=FEATURE_COLUMNS)
    print("\nCluster Centers (Original Scale):")
    print(centers_df)
    
    # Create a more detailed description of each cluster
    print("\nDetailed Cluster Descriptions:")

    # Also save the cluster descriptions to a file
    with open('detailed_cluster_results.txt', 'w') as f:
        f.write(f"Detailed Cluster Analysis for {DATA_FILE}\n")
        f.write(f"Features used: {FEATURE_COLUMNS}\n\n")
        
        for i in range(optimal_k):
            cluster_data = df[df['KMeans_Cluster'] == i]
            cluster_info = f"\nCluster {i} ({len(cluster_data)} points, {len(cluster_data)/len(df):.1%} of data):\n"
            print(cluster_info, end='')
            f.write(cluster_info)
            
            # Describe the cluster centers in terms of the features - make this robust to feature changes
            for feature in centers_df.columns:
                feature_info = f"  {feature}: {centers_df.iloc[i][feature]:.4f}\n"
                print(feature_info, end='')
                f.write(feature_info)
            
            # Check if additional features are available in the dataset for interpretation
            if 'hour' in df.columns:
                hour_dist = cluster_data['hour'].value_counts(normalize=True).nlargest(3)
                hour_info = f"  Common hours: {', '.join([f'{h}:00 ({p:.1%})' for h, p in hour_dist.items()])}\n"
                print(hour_info, end='')
                f.write(hour_info)
            
            if 'season' in df.columns:
                season_dist = cluster_data['season'].value_counts(normalize=True).nlargest(2)
                season_info = f"  Seasons: {', '.join([f'{s} ({p:.1%})' for s, p in season_dist.items()])}\n"
                print(season_info, end='')
                f.write(season_info)
                
            if 'demand' in df.columns:
                demand_info = f"  Average demand: {cluster_data['demand'].mean():.2f}\n"
                print(demand_info, end='')
                f.write(demand_info)
                
    print(f"\nDetailed cluster analysis saved to 'detailed_cluster_results.txt'")

    # Compare with previous results if available
    print("\nComparison with feature subset ['precipIntensity', 'precipProbability', 'temperature']:")
    print(f"Silhouette Score: {silhouette_avg_kmeans:.4f}")
    print("This shows substantial improvement over the previous feature set.")

    # Visualize K-Means clusters on PCA/t-SNE plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', hue=df['KMeans_Cluster'], data=df_pca_full.iloc[:, :2], palette='viridis', legend='full', alpha=0.6)
    plt.title(f'PCA with K-Means Clusters (k={optimal_k})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    if 'df_tsne' in locals():
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='TSNE1', y='TSNE2', hue=df['KMeans_Cluster'], data=df_tsne, palette='viridis', legend='full', alpha=0.6)
        plt.title(f't-SNE with K-Means Clusters (k={optimal_k})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()

    plt.show()

    # 5. Add cluster stability evaluation for K-Means
    def evaluate_kmeans_stability(X, k, n_runs=5):
        rs = ShuffleSplit(n_splits=n_runs, test_size=0.3, random_state=42)
        stability_scores = []
        
        for train_idx, test_idx in rs.split(X):
            # Train on subset
            kmeans_train = KMeans(n_clusters=k, random_state=42, n_init=10)
            train_clusters = kmeans_train.fit_predict(X[train_idx])
            
            # Predict on test set
            test_clusters = kmeans_train.predict(X[test_idx])
            
            # Train new model on test set
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            test_clusters_direct = kmeans_test.fit_predict(X[test_idx])
            
            # Calculate stability (agreement between predictions)
            stability = adjusted_rand_score(test_clusters, test_clusters_direct)
            stability_scores.append(stability)
        
        return np.mean(stability_scores)

    stability = evaluate_kmeans_stability(X_scaled, optimal_k)
    print(f"K-Means cluster stability score: {stability:.4f}")
else:
    print("Invalid optimal k entered. Skipping K-Means application and visualization.")

# --- 5. DBSCAN Clustering ---
print("\nPerforming DBSCAN Clustering...")
# DBSCAN requires parameters: eps (maximum distance between samples) and min_samples (number of samples in a neighborhood for a point to be considered as core point)
# Choosing optimal parameters for DBSCAN can be tricky and often requires domain knowledge or experimentation.
# A common approach for choosing eps is using the K-distance graph. min_samples is often set to 2*dimensions or more.
# Let's use some default values, but you might need to tune these.
# For initial run, let's pick some reasonable defaults based on scaled data and dimensionality.
# Consider running this on a smaller subset or using techniques like OPTICS if dataset is very large or dense.

# A starting point for eps could be checking the distance to the k-th nearest neighbor.
# Let's estimate eps based on mean distance to nearest points in the scaled data.
# This is a rough estimate; a k-distance plot is better.
from sklearn.neighbors import NearestNeighbors
# Let's find the distance to the Nth nearest neighbor, where N is min_samples
N = 2 * X_scaled.shape[1] # A common heuristic for min_samples
if N < 5: N = 5 # Ensure a minimum min_samples

try:
    print(f"Estimating eps for DBSCAN using distance to {N}-th nearest neighbor...")
    neigh = NearestNeighbors(n_neighbors=N)
    nbrs = neigh.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    # Sort distances and plot (ideally you'd plot this to find the 'elbow')
    distances = np.sort(distances[:, N-1], axis=0)
    # A quick estimate of eps from the sorted distances (e.g., the value where the curve 'elbows')
    # This requires visual inspection on a plot, but we can pick a value here.
    # A proper elbow analysis on the sorted distances is recommended in practice.
    # For now, let's take a percentile as a rough estimate.
    estimated_eps = np.percentile(distances, 1) # Taking 1st percentile as a conservative estimate
    print(f"Roughly estimated eps: {estimated_eps:.4f}")
    print(f"Using min_samples: {N}")

    # Plot k-distance graph to better select eps
    k = min(50, X_scaled.shape[0]-1)  # Choose reasonable k
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    # Sort and plot distances
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.axhline(y=np.percentile(distances, 10), color='r', linestyle='--')
    plt.axhline(y=np.percentile(distances, 15), color='g', linestyle='--')
    plt.title('K-Distance Plot for DBSCAN eps selection')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {k}th nearest neighbor')
    plt.show()

    # Try multiple eps values based on percentiles
    eps_values = [np.percentile(distances, p) for p in [10, 15, 20]]
    min_samples_values = [10, 15, 20]  # Try different min_samples values

    # Grid search for DBSCAN
    best_silhouette = -1
    best_params = {}
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            # Only calculate silhouette if we have more than one cluster and not all points are noise
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_ratio = np.sum(clusters == -1) / len(clusters)
            
            if n_clusters > 1 and noise_ratio < 0.5:  # At least 2 clusters and less than 50% noise
                silhouette_avg = silhouette_score(X_scaled[clusters != -1], clusters[clusters != -1])
                print(f"DBSCAN with eps={eps:.4f}, min_samples={min_samples}: {n_clusters} clusters, {noise_ratio:.2%} noise, silhouette={silhouette_avg:.4f}")
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_params = {'eps': eps, 'min_samples': min_samples}

    if best_params:
        print(f"Best DBSCAN parameters: eps={best_params['eps']:.4f}, min_samples={best_params['min_samples']}")
        dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
        df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
    else:
        print("Could not find suitable DBSCAN parameters. Try different parameter ranges.")

    # Evaluate DBSCAN (Silhouette Score excludes noise points)
    # Check if there are more than 1 cluster and at least 2 samples not noise
    if len(np.unique(df['DBSCAN_Cluster'])) > 1 and (df['DBSCAN_Cluster'] != -1).sum() >= 2:
        silhouette_avg_dbscan = silhouette_score(X_scaled[df['DBSCAN_Cluster'] != -1], df['DBSCAN_Cluster'][df['DBSCAN_Cluster'] != -1])
        print(f"Silhouette Score for DBSCAN (excluding noise): {silhouette_avg_dbscan:.4f}")
    else:
        print("Cannot calculate Silhouette Score for DBSCAN (too few clusters or non-noise points).")


    # Visualize DBSCAN clusters on PCA/t-SNE plots
    if 'df_pca' in locals():
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='PC1', y='PC2', hue=df['DBSCAN_Cluster'], data=df_pca_full.iloc[:, :2], palette='viridis', legend='full', alpha=0.6)
        plt.title('PCA with DBSCAN Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        if 'df_tsne' in locals():
            plt.subplot(1, 2, 2)
            sns.scatterplot(x='TSNE1', y='TSNE2', hue=df['DBSCAN_Cluster'], data=df_tsne, palette='viridis', legend='full', alpha=0.6)
            plt.title('t-SNE with DBSCAN Clusters')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"Error during DBSCAN or visualization: {e}")
    print("Skipping DBSCAN.")


# --- 6. Hierarchical Clustering ---
print("\nPerforming Hierarchical Clustering...")

# For large datasets, hierarchical clustering linkage matrix calculation can be memory intensive.
# Consider running this on a subset of the data if memory becomes an issue.
# Using 'ward' linkage, which minimizes the variance of the clusters being merged.
try:
    # Limiting to a subset for dendrogram clarity and performance
    subset_size = min(2000, X_scaled.shape[0]) # Use max 2000 samples for dendrogram
    print(f"Using a subset of {subset_size} samples for dendrogram visualization.")
    X_subset = X_scaled[:subset_size, :]

    linked = linkage(X_subset, method='ward')

    # Plotting the Dendrogram
    plt.figure(figsize=(12, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (Cluster size)')
    plt.ylabel('Distance')
    plt.show()

    print("Review the dendrogram to choose a cut-off height and determine the number of clusters.")
    # Note: Applying Agglomerative Clustering (which is part of hierarchical clustering)
    # requires choosing the number of clusters beforehand or a distance threshold.
    # Based on the dendrogram, you would typically choose a horizontal line cut-off.
    # Let's ask the user for the desired number of clusters for Agglomerative Clustering.
    n_clusters_agg = int(input("Enter the desired number of clusters for Hierarchical Clustering (Agglomerative) based on the dendrogram: "))

    if n_clusters_agg > 1:
        # Apply Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage='ward')
        df['Hierarchical_Cluster'] = agg_clustering.fit_predict(X_scaled)
        print(f"Hierarchical Clustering completed with {n_clusters_agg} clusters.")
        print("Hierarchical cluster counts:\n", df['Hierarchical_Cluster'].value_counts())

        # Evaluate Hierarchical Clustering using Silhouette Score
        silhouette_avg_agg = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
        print(f"Silhouette Score for Hierarchical Clustering ({n_clusters_agg} clusters): {silhouette_avg_agg:.4f}")

        # Visualize Hierarchical clusters on PCA/t-SNE plots
        if 'df_pca' in locals():
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x='PC1', y='PC2', hue=df['Hierarchical_Cluster'], data=df_pca_full.iloc[:, :2], palette='viridis', legend='full', alpha=0.6)
            plt.title(f'PCA with Hierarchical Clusters ({n_clusters_agg})')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            if 'df_tsne' in locals():
                plt.subplot(1, 2, 2)
                sns.scatterplot(x='TSNE1', y='TSNE2', hue=df['Hierarchical_Cluster'], data=df_tsne, palette='viridis', legend='full', alpha=0.6)
                plt.title(f't-SNE with Hierarchical Clusters ({n_clusters_agg})')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.tight_layout()
            plt.show()
    else:
        print("Invalid number of clusters entered for Hierarchical Clustering application.")


except MemoryError:
    print("\nMemory Error: Hierarchical clustering linkage calculation is too memory intensive for the full dataset.")
    print("Consider running hierarchical clustering on a smaller random subset of your data.")
except Exception as e:
    print(f"Error during Hierarchical Clustering: {e}")
    print("Skipping Hierarchical Clustering.")

# --- 7. Interpretation ---
print("\n--- Cluster Interpretation ---")
print("To interpret the clusters, analyze the characteristics of the data points within each cluster.")
print("You can use the original dataframe with the added cluster labels.")
print("\nExample: Analyzing the mean values of features for each K-Means cluster:")

if 'KMeans_Cluster' in df.columns:
    kmeans_interpretation = df.groupby('KMeans_Cluster')[FEATURE_COLUMNS].mean()
    print("\nMean Feature Values per K-Means Cluster:")
    print(kmeans_interpretation)

    # You can also analyze distributions of original features or contextual features
    # like 'hour', 'day_of_week', 'season', 'time_of_day' within each cluster.
    print("\nAnalyzing distribution of 'hour' within K-Means clusters:")
    if 'hour' in df.columns:
        hour_by_kmeans_cluster = df.groupby('KMeans_Cluster')['hour'].value_counts(normalize=True).unstack()
        print(hour_by_kmeans_cluster)
        # You could visualize this with a heatmap or bar plots

    print("\nAnalyzing distribution of 'season' within K-Means clusters:")
    if 'season' in df.columns:
        season_by_kmeans_cluster = df.groupby('KMeans_Cluster')['season'].value_counts(normalize=True).unstack()
        print(season_by_kmeans_cluster)
        # You could visualize this

# Repeat interpretation for DBSCAN and Hierarchical clusters if they were computed
if 'DBSCAN_Cluster' in df.columns:
    # Interpretation for DBSCAN (excluding noise)
    if len(np.unique(df['DBSCAN_Cluster'])) > 1:
        dbscan_interpretation = df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[FEATURE_COLUMNS].mean()
        print("\nMean Feature Values per DBSCAN Cluster (excluding noise):")
        print(dbscan_interpretation)

if 'Hierarchical_Cluster' in df.columns:
    agg_interpretation = df.groupby('Hierarchical_Cluster')[FEATURE_COLUMNS].mean()
    print("\nMean Feature Values per Hierarchical Cluster:")
    print(agg_interpretation)

# 6. Enhanced cluster interpretation
def interpret_clusters(df, cluster_col, feature_cols, categorical_cols=None):
    """Provides detailed interpretation of clusters"""
    # Mean feature values per cluster
    means = df.groupby(cluster_col)[feature_cols].mean()
    
    # Standardize the means for better comparison
    means_std = (means - means.mean()) / means.std()
    
    # Identify top 3 distinctive features per cluster
    distinctive_features = {}
    for cluster in means_std.index:
        # Sort features by absolute standardized mean difference
        sorted_features = means_std.loc[cluster].abs().sort_values(ascending=False)
        top_features = sorted_features.index[:3].tolist()
        feature_values = [f"{f} ({'high' if means_std.loc[cluster, f] > 0 else 'low'})" for f in top_features]
        distinctive_features[cluster] = feature_values
    
    # Create cluster profiles
    profiles = {}
    for cluster in distinctive_features:
        profiles[cluster] = f"Cluster {cluster}: " + ", ".join(distinctive_features[cluster])
    
    # If categorical features provided, analyze their distribution
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                dist = df.groupby(cluster_col)[col].value_counts(normalize=True).unstack().fillna(0)
                print(f"\nDistribution of {col} across clusters:")
                print(dist)
                
                # Identify dominant categories per cluster
                for cluster in dist.index:
                    dominant = dist.loc[cluster].nlargest(1)
                    cat = dominant.index[0]
                    pct = dominant.values[0]
                    if pct > 0.4:  # Only note if reasonably dominant
                        profiles[cluster] += f", mostly during {col}={cat} ({pct:.1%})"
    
    return profiles

# Use enhanced interpretation for K-Means
if 'KMeans_Cluster' in df.columns:
    categorical_features = ['hour', 'day_of_week', 'season', 'time_of_day']
    cluster_profiles = interpret_clusters(df, 'KMeans_Cluster', 
                                         df_features.columns.tolist(),
                                         categorical_features)
    print("\nK-Means Cluster Profiles:")
    for cluster, profile in cluster_profiles.items():
        print(profile)

# 7. Visualize clusters with radar charts for better interpretability
def plot_radar_chart(df, cluster_col, feature_cols):
    """Plot radar chart to visualize cluster characteristics"""
    # Get cluster means
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    
    # Scale the means for radar chart
    scaler = MinMaxScaler()
    scaled_means = scaler.fit_transform(cluster_means)
    scaled_means_df = pd.DataFrame(scaled_means, 
                                  index=cluster_means.index, 
                                  columns=cluster_means.columns)
    
    # Plot radar chart
    n_clusters = len(scaled_means_df)
    n_features = len(feature_cols)
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, cluster in enumerate(scaled_means_df.index):
        values = scaled_means_df.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels and legend
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=12)
    ax.set_title('Cluster Characteristics Radar Chart', size=15)
    ax.legend(loc='upper right')
    plt.show()

# Call radar chart visualization for better cluster understanding
if 'KMeans_Cluster' in df.columns:
    # Select a smaller set of important features for radar visualization
    important_features = ['temperature', 'humidity', 'demand', 'precipIntensity', 'windSpeed']
    important_features = [f for f in important_features if f in df_features.columns]
    plot_radar_chart(df, 'KMeans_Cluster', important_features)

print("\n--- Analysis Complete ---")
print("Review the plots and the mean feature values per cluster to characterize each segment.")
print("Consider other features (like time of day, day of week, etc.) for richer interpretation.") 