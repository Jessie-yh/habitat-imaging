import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_cluster_2d(x, labels, n_clusters, filename_base, sequence_names):
    """
    Plot 2D cluster diagram
    
    Parameters:
    x - data points (2 x N)
    labels - cluster labels
    n_clusters - number of clusters
    filename_base - base output filename
    sequence_names - list of sequence names
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1 = axes[0]
    for i in range(n_clusters):
        mask = labels == i
        ax1.scatter(x[0, mask], x[1, mask], 
                   c=colors[i % len(colors)], 
                   marker=markers[i % len(markers)],
                   s=30, alpha=0.7, 
                   label=f'Cluster {i+1}',
                   edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel(sequence_names[0], fontsize=14, fontweight='bold')
    ax1.set_ylabel(sequence_names[1], fontsize=14, fontweight='bold')
    ax1.set_title(f'2D K-means Clustering (k={n_clusters})', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    try:
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42).fit(x.T)
        centers = kmeans_temp.cluster_centers_
        ax1.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='X', s=200, 
                   label='Cluster Centers',
                   edgecolors='black', linewidth=2)
        ax1.legend(fontsize=12)
    except:
        pass
    
    ax2 = axes[1]
    
    for i in range(n_clusters):
        mask = labels == i
        ax2.scatter(x[0, mask], x[1, mask], 
                   c=colors[i % len(colors)], 
                   marker=markers[i % len(markers)],
                   s=30, alpha=0.6, 
                   label=f'Cluster {i+1}',
                   edgecolors='white', linewidth=0.5)
    
    try:
        from scipy.stats import gaussian_kde
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) > 1:
                kde = gaussian_kde([x[0, mask], x[1, mask]])
                
                x_min, x_max = x[0, mask].min(), x[0, mask].max()
                y_min, y_max = x[1, mask].min(), x[1, mask].max()
                
                xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                
                z = np.reshape(kde(positions).T, xx.shape)
                
                ax2.contour(xx, yy, z, levels=3, 
                           colors=[colors[i % len(colors)]], 
                           linewidths=1, alpha=0.8)
    except:
        print("Note: Could not generate density contours")
    
    ax2.set_xlabel(sequence_names[0], fontsize=14, fontweight='bold')
    ax2.set_ylabel(sequence_names[1], fontsize=14, fontweight='bold')
    ax2.set_title('Clusters with Density Contours', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{filename_base}_2d_clusters.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved 2D cluster plot: {filename}")
    
    plt.close(fig)
    
    return filename


def find_elbow_point(distortions):
    """
    Find elbow point in distortion curve using maximum curvature method
    
    Parameters:
    distortions - list of distortion values for k=2 to k=5
    
    Returns:
    elbow_k - k value at elbow point
    """
    k_range = np.array(range(2, 10))
    distortions = np.array(distortions)
    
    k_normalized = (k_range - min(k_range)) / (max(k_range) - min(k_range))
    d_normalized = (distortions - min(distortions)) / (max(distortions) - min(distortions))
    
    curve = np.zeros(len(k_range))
    for i in range(1, len(k_range)-1):
        dx1 = k_normalized[i] - k_normalized[i-1]
        dy1 = d_normalized[i] - d_normalized[i-1]
        dx2 = k_normalized[i+1] - k_normalized[i]
        dy2 = d_normalized[i+1] - d_normalized[i]
        
        numerator = abs(dy2/dx2 - dy1/dx1)
        denominator = (1 + (dy1/dx1)**2)**1.5
        curve[i] = numerator / denominator if denominator != 0 else 0
    
    elbow_index = np.argmax(curve[1:-1]) + 1
    return k_range[elbow_index]


def plot_optimal_clusters_summary(k_range, distortions, silhouette_scores, ch_scores, db_scores,
                                 elbow_k, silhouette_k, ch_k, db_k, output_folder):
    """
    Create a summary plot showing all evaluation metrics together
    
    Parameters:
    k_range - range of k values tested
    distortions - distortion values for each k
    silhouette_scores - silhouette scores for each k
    ch_scores - Calinski-Harabasz scores for each k
    db_scores - Davies-Bouldin scores for each k
    elbow_k - optimal k from elbow method
    silhouette_k - optimal k from silhouette method
    ch_k - optimal k from Calinski-Harabasz method
    db_k - optimal k from Davies-Bouldin method
    output_folder - directory to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Elbow method
    ax1.plot(k_range, distortions, 'bo-', linewidth=2, markersize=8)
    elbow_idx = list(k_range).index(elbow_k)
    ax1.plot(elbow_k, distortions[elbow_idx], 'ro', markersize=12, 
             label=f'Elbow Point (k={elbow_k})')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Distortion (Inertia)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14)
    ax1.set_xticks(k_range)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Silhouette score
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    if not np.all(np.isnan(silhouette_scores)):
        silhouette_idx = list(k_range).index(silhouette_k)
        ax2.plot(silhouette_k, silhouette_scores[silhouette_idx], 'ro', markersize=12, 
                 label=f'Best Score (k={silhouette_k})')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score (Higher is better)', fontsize=14)
    ax2.set_xticks(k_range)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Calinski-Harabasz index
    ax3.plot(k_range, ch_scores, 'ro-', linewidth=2, markersize=8)
    ch_idx = list(k_range).index(ch_k)
    ax3.plot(ch_k, ch_scores[ch_idx], 'bo', markersize=12, 
             label=f'Best Score (k={ch_k})')
    ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=12)
    ax3.set_title('Calinski-Harabasz Index (Higher is better)', fontsize=14)
    ax3.set_xticks(k_range)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Davies-Bouldin index
    ax4.plot(k_range, db_scores, 'mo-', linewidth=2, markersize=8)
    db_idx = list(k_range).index(db_k)
    ax4.plot(db_k, db_scores[db_idx], 'go', markersize=12, 
             label=f'Best Score (k={db_k})')
    ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax4.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax4.set_title('Davies-Bouldin Index (Lower is better)', fontsize=14)
    ax4.set_xticks(k_range)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.suptitle('Optimal Cluster Number Evaluation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the summary plot
    summary_path = os.path.join(output_folder, 'optimal_clusters_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved optimal clusters summary plot: {summary_path}")
    
    plt.close(fig)
    
    # Create a recommendations table
    recommendations_path = os.path.join(output_folder, 'optimal_k_recommendations.txt')
    with open(recommendations_path, 'w') as f:
        f.write("Optimal Cluster Number Recommendations\n")
        f.write("="*50 + "\n\n")
        f.write(f"Elbow Method: k = {elbow_k}\n")
        f.write(f"Silhouette Score: k = {silhouette_k}\n")
        f.write(f"Calinski-Harabasz Index: k = {ch_k}\n")
        f.write(f"Davies-Bouldin Index: k = {db_k}\n\n")
        f.write("="*50 + "\n\n")
        f.write("Interpretation:\n")
        f.write("- Elbow method: Looks for the point where distortion decrease slows\n")
        f.write("- Silhouette score: Measures how similar points are to their own cluster\n")
        f.write("- Calinski-Harabasz: Ratio of between-cluster to within-cluster dispersion\n")
        f.write("- Davies-Bouldin: Average similarity between each cluster and its most similar\n")
    
    print(f"Saved optimal k recommendations: {recommendations_path}")


def evaluate_cluster_numbers(all_data, output_folder):
    """
    Evaluate different cluster numbers using multiple metrics and plot results
    
    Parameters:
    all_data - input data for clustering
    output_folder - directory to save evaluation plots
    """
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    k_range = range(2, 10)
    
    distortions = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    print("\nEvaluating cluster numbers...")
    for k in k_range:
        print(f"  Testing k = {k}")
        
        if np.isnan(all_data).any():
            print(f"    Warning: Data contains NaN values for k={k}")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            all_data_clean = imputer.fit_transform(all_data)
        else:
            all_data_clean = all_data
        
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20).fit(all_data_clean)
        
        distortions.append(kmeans.inertia_)
        
        sample_size = min(10000, len(all_data_clean))
        if sample_size >= k * 10:
            sample_indices = np.random.choice(len(all_data_clean), sample_size, replace=False)
            sample_data = all_data_clean[sample_indices]
            sample_labels = kmeans.labels_[sample_indices]
            silhouette_scores.append(silhouette_score(sample_data, sample_labels))
        else:
            silhouette_scores.append(np.nan)
        
        ch_scores.append(calinski_harabasz_score(all_data_clean, kmeans.labels_))
        
        db_scores.append(davies_bouldin_score(all_data_clean, kmeans.labels_))
    
    silhouette_scores = np.array(silhouette_scores)
    valid_silhouette = ~np.isnan(silhouette_scores)
    if np.any(valid_silhouette):
        silhouette_k = k_range[np.nanargmax(silhouette_scores)]
    else:
        silhouette_k = 3
    
    elbow_k = find_elbow_point(distortions)
    ch_k = k_range[np.argmax(ch_scores)]
    db_k = k_range[np.argmin(db_scores)]
    
    # Plot 1: Elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distortions, 'bo-', linewidth=2.5, markersize=10)
    elbow_idx = list(k_range).index(elbow_k)
    plt.plot(elbow_k, distortions[elbow_idx], 'ro', markersize=12, 
             label=f'Elbow Point (k={elbow_k})')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Distortion (Inertia)', fontsize=12)
    plt.title(f'Elbow Method for Optimal k (Suggested k={elbow_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    elbow_plot_path = os.path.join(output_folder, 'elbow_method.png')
    plt.savefig(elbow_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved elbow method plot: {elbow_plot_path}")
    
    # Plot 2: Silhouette score
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2.5, markersize=10)
    if np.any(valid_silhouette):
        silhouette_idx = list(k_range).index(silhouette_k)
        plt.plot(silhouette_k, silhouette_scores[silhouette_idx], 'ro', markersize=12, 
                 label=f'Best Silhouette (k={silhouette_k})')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title(f'Silhouette Score for Optimal k (Best k={silhouette_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    silhouette_plot_path = os.path.join(output_folder, 'silhouette_score.png')
    plt.savefig(silhouette_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved silhouette score plot: {silhouette_plot_path}")
    
    # Plot 3: Calinski-Harabasz index
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, ch_scores, 'ro-', linewidth=2.5, markersize=10)
    ch_idx = list(k_range).index(ch_k)
    plt.plot(ch_k, ch_scores[ch_idx], 'bo', markersize=12, 
             label=f'Best Calinski-Harabasz (k={ch_k})')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Calinski-Harabasz Index', fontsize=12)
    plt.title(f'Calinski-Harabasz Index for Optimal k (Best k={ch_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    ch_plot_path = os.path.join(output_folder, 'calinski_harabasz.png')
    plt.savefig(ch_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Calinski-Harabasz plot: {ch_plot_path}")
    
    # Plot 4: Davies-Bouldin index
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, db_scores, 'mo-', linewidth=2.5, markersize=10)
    db_idx = list(k_range).index(db_k)
    plt.plot(db_k, db_scores[db_idx], 'go', markersize=12, 
             label=f'Best Davies-Bouldin (k={db_k})')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Davies-Bouldin Index', fontsize=12)
    plt.title(f'Davies-Bouldin Index for Optimal k (Best k={db_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    db_plot_path = os.path.join(output_folder, 'davies_bouldin.png')
    plt.savefig(db_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Davies-Bouldin plot: {db_plot_path}")
    
    # Create summary plot
    plot_optimal_clusters_summary(k_range, distortions, silhouette_scores, ch_scores, db_scores,
                                 elbow_k, silhouette_k, ch_k, db_k, output_folder)
    
    print("\nCluster Evaluation Results:")
    print("k\tDistortion\tSilhouette\tCalinski-H.\tDavies-B.")
    for i, k in enumerate(k_range):
        silhouette_val = silhouette_scores[i] if not np.isnan(silhouette_scores[i]) else "N/A"
        if isinstance(silhouette_val, float):
            print(f"{k}\t{distortions[i]:.2f}\t{silhouette_val:.4f}\t{ch_scores[i]:.2f}\t{db_scores[i]:.4f}")
        else:
            print(f"{k}\t{distortions[i]:.2f}\t{silhouette_val}\t{ch_scores[i]:.2f}\t{db_scores[i]:.4f}")
    
    print("\nSuggested optimal cluster numbers:")
    print(f"  Elbow method: k = {elbow_k}")
    print(f"  Silhouette score: k = {silhouette_k}")
    print(f"  Calinski-Harabasz: k = {ch_k}")
    print(f"  Davies-Bouldin: k = {db_k}")
    
    return elbow_k, silhouette_k, ch_k, db_k


def perform_kmeans(all_data, output_folder, sequence_names, n_clusters=None):
    """
    Perform K-means clustering with specified number of clusters
    
    Parameters:
    all_data - input data for clustering
    output_folder - directory to save results
    sequence_names - sequence names
    n_clusters - number of clusters to use (default=None, auto-determine)
    """
    if np.isnan(all_data).any():
        print("Warning: Data contains NaN values. Using mean imputation...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        all_data = imputer.fit_transform(all_data)
    
    if n_clusters is None:
        elbow_k, silhouette_k, ch_k, db_k = evaluate_cluster_numbers(all_data, output_folder)
        n_clusters = elbow_k
    
    print(f"\nPerforming final clustering with k = {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=20).fit(all_data)
    labels = kmeans.labels_
    
    plot_cluster_2d(np.array(all_data).T, labels, n_clusters, 
                    os.path.join(output_folder, 'kmeans_cluster_all_patients'),
                    sequence_names)
    
    cluster_ranges = {i: {sequence_names[0]: [np.inf, -np.inf], 
                         sequence_names[1]: [np.inf, -np.inf]} 
                     for i in range(n_clusters)}
    
    all_data_np = np.array(all_data)
    
    for idx, point in enumerate(all_data_np):
        cluster_idx = labels[idx]
        for seq_idx, seq_name in enumerate(sequence_names):
            if point[seq_idx] < cluster_ranges[cluster_idx][seq_name][0]:
                cluster_ranges[cluster_idx][seq_name][0] = point[seq_idx]
            if point[seq_idx] > cluster_ranges[cluster_idx][seq_name][1]:
                cluster_ranges[cluster_idx][seq_name][1] = point[seq_idx]
    
    range_report_path = os.path.join(output_folder, "cluster_value_ranges.txt")
    with open(range_report_path, "w") as f:
        f.write(f"Final clustering with k = {n_clusters}\n")
        f.write(f"Number of data points: {len(all_data)}\n")
        f.write("="*60 + "\n")
        
        for cluster_idx in range(n_clusters):
            cluster_mask = labels == cluster_idx
            cluster_size = np.sum(cluster_mask)
            f.write(f"\nCluster {cluster_idx+1} (Size: {cluster_size}, {cluster_size/len(labels)*100:.1f}%):\n")
            
            for seq_name in sequence_names:
                seq_idx = sequence_names.index(seq_name)
                cluster_values = all_data_np[cluster_mask, seq_idx]
                
                if len(cluster_values) > 0:
                    mean_val = np.mean(cluster_values)
                    std_val = np.std(cluster_values)
                    median_val = np.median(cluster_values)
                    min_val = np.min(cluster_values)
                    max_val = np.max(cluster_values)
                    
                    f.write(f"  {seq_name}:\n")
                    f.write(f"    Range: [{min_val:.6f}, {max_val:.6f}]\n")
                    f.write(f"    Mean +- SD: {mean_val:.6f} +- {std_val:.6f}\n")
                    f.write(f"    Median: {median_val:.6f}\n")
                else:
                    f.write(f"  {seq_name}: No data\n")
            
            f.write("-"*60 + "\n")
        
        f.write("\nOverall Statistics:\n")
        for seq_name in sequence_names:
            seq_idx = sequence_names.index(seq_name)
            seq_values = all_data_np[:, seq_idx]
            f.write(f"\n{seq_name}:\n")
            f.write(f"  Global range: [{np.min(seq_values):.6f}, {np.max(seq_values):.6f}]\n")
            f.write(f"  Global mean +- SD: {np.mean(seq_values):.6f} +- {np.std(seq_values):.6f}\n")
            f.write(f"  Global median: {np.median(seq_values):.6f}\n")
    
    print("\n" + "="*60)
    print(f"Final clustering results (k={n_clusters}):")
    print(f"Total data points: {len(all_data)}")
    print("="*60)
    
    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_size = np.sum(cluster_mask)
        print(f"\nCluster {cluster_idx+1} (Size: {cluster_size}, {cluster_size/len(labels)*100:.1f}%):")
        
        for seq_name in sequence_names:
            seq_idx = sequence_names.index(seq_name)
            cluster_values = all_data_np[cluster_mask, seq_idx]
            
            if len(cluster_values) > 0:
                mean_val = np.mean(cluster_values)
                std_val = np.std(cluster_values)
                print(f"  {seq_name}: {mean_val:.6f} +- {std_val:.6f} "
                      f"[{cluster_ranges[cluster_idx][seq_name][0]:.6f}, "
                      f"{cluster_ranges[cluster_idx][seq_name][1]:.6f}]")
        
        print("-"*60)
    
    return labels, n_clusters


def process_image_data_2d(path_to_img_folder1, path_to_img_folder2, 
                         roi_folder, output_folder, sequence_names, n_clusters=3):
    """
    Process image data and perform clustering (2D version)
    
    Parameters:
    path_to_img_folder1 - D sequence image folder path
    path_to_img_folder2 - Dstar sequence image folder path
    roi_folder - ROI folder path
    output_folder - output folder path
    sequence_names - list of sequence names
    n_clusters - number of clusters to use (default=3)
    """
    
    filenames = [f for f in os.listdir(path_to_img_folder1) if f.endswith(".nii.gz")]
    
    if not filenames:
        print(f"Error: No .nii.gz files found in {path_to_img_folder1}")
        return
    
    print(f"Found {len(filenames)} image files")
    
    all_data = []
    rois = []
    
    for filename in filenames:
        print(f"Processing: {filename}")
        
        try:
            img1_path = os.path.join(path_to_img_folder1, filename)
            img1 = nib.load(img1_path)
            img1_data = img1.get_fdata()
            
            img2_path = os.path.join(path_to_img_folder2, filename)
            img2 = nib.load(img2_path)
            img2_data = img2.get_fdata()
            
            roi_path = os.path.join(roi_folder, filename)
            roi = nib.load(roi_path)
            roi_data = roi.get_fdata()
            
            roi_mask = roi_data != 0
            
            x = img1_data[roi_mask]
            y = img2_data[roi_mask]
            
            if len(x) == 0:
                print(f"  Warning: No ROI voxels for {filename}")
                continue
            
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) == 0:
                print(f"  Warning: No valid voxels after NaN/Inf removal for {filename}")
                continue
            
            rois.append({
                'filename': filename,
                'roi_data': roi_data, 
                'roi_mask': roi_mask, 
                'affine': roi.affine,
                'valid_voxels': len(x)
            })
            
            all_data.extend(np.column_stack((x, y)))
            
            print(f"  Extracted {len(x)} valid voxels")
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue
    
    if not all_data:
        print("Error: No valid data extracted from any file!")
        return
    
    all_data = np.array(all_data)
    print(f"\nTotal data points for clustering: {len(all_data)}")
    print(f"Data shape: {all_data.shape}")
    
    print("\nData quality check:")
    print(f"  NaN values: {np.isnan(all_data).sum()}")
    print(f"  Inf values: {np.isinf(all_data).sum()}")
    print(f"  D range: [{np.min(all_data[:, 0]):.6f}, {np.max(all_data[:, 0]):.6f}]")
    print(f"  Dstar range: [{np.min(all_data[:, 1]):.6f}, {np.max(all_data[:, 1]):.6f}]")
    
    labels, n_clusters = perform_kmeans(all_data, output_folder, sequence_names, n_clusters)
    
    start = 0
    for i, roi_info in enumerate(rois):
        filename = roi_info['filename']
        roi_data = roi_info['roi_data']
        roi_mask = roi_info['roi_mask']
        affine = roi_info['affine']
        valid_voxels = roi_info['valid_voxels']
        
        roi_kmeans = np.zeros_like(roi_data)
        
        if valid_voxels > 0:
            end = start + valid_voxels
            roi_kmeans[roi_mask] = 0
            
            temp_mask = roi_mask.copy()
            roi_kmeans[roi_mask] = labels[start:end] + 1
            start = end
        
        cluster_nii = nib.Nifti1Image(roi_kmeans, affine=affine)
        output_filename = filename.replace(".nii.gz", "_kmeans.nii.gz")
        nib.save(cluster_nii, os.path.join(output_folder, output_filename))
        print(f"Saved cluster result: {output_filename}")
        
        if valid_voxels > 0:
            patient_data = all_data[start-valid_voxels:start]
            patient_labels = labels[start-valid_voxels:start]
            
            if len(patient_data) > 0:
                patient_output_base = os.path.join(
                    output_folder, 
                    filename.replace(".nii.gz", "_kmeans")
                )
                
                plot_cluster_2d(
                    patient_data.T, 
                    patient_labels, 
                    n_clusters, 
                    patient_output_base,
                    sequence_names
                )


# ============== MAIN CONFIGURATION ==============
sequence_names = ["wash-in", "wash-out"]

N_CLUSTERS = 2

path_to_img_folder1 = "/data/yuwei/code/yanhong/data/DCE/wash-in_normalization/"
path_to_img_folder2 = "/data/yuwei/code/yanhong/data/DCE/wash-out_normalization/"
roi_folder = "/data/yuwei/code/yanhong/data/DCE/masks-DCE/"
output_folder = "/data/yuwei/code/yanhong/data/DCE_habitat/k-means2_wash-in_wash-out/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

print("="*60)
print("Starting 2D K-means Clustering (D and Dstar only)")
print("="*60)
print(f"D folder: {path_to_img_folder1}")
print(f"Dstar folder: {path_to_img_folder2}")
print(f"ROI folder: {roi_folder}")
print(f"Output folder: {output_folder}")
print(f"Number of clusters: {N_CLUSTERS}")
print("="*60)

process_image_data_2d(
    path_to_img_folder1, 
    path_to_img_folder2, 
    roi_folder, 
    output_folder,
    sequence_names,
    n_clusters=N_CLUSTERS
)

print("\n" + "="*60)
print("2D K-means clustering completed successfully!")
print(f"Results saved to: {output_folder}")
print("="*60)