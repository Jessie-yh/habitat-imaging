import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import imageio
import seaborn as sns


def plot_cluster(x, labels, n_clusters, filename_base, sequence_names):

    # Plot 3D cluster diagram from multiple angles with improved layout
    
    # Parameters:
    # x - data points (3 x N)
    # labels - cluster labels
    # n_clusters - number of clusters
    # filename_base - base output filename (without extension)
    # sequence_names - names of the three sequences

    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k', 'w']
    
    # Create larger figure with better layout
    fig = plt.figure(figsize=(14, 10))
    
    # Adjust subplot position to provide more space for labels
    ax = fig.add_axes([0.08, 0.08, 0.85, 0.85], projection='3d')
    
    # Plot each cluster
    for i in range(n_clusters):
        ax.scatter(x[0, labels == i], x[1, labels == i], x[2, labels == i], 
                   c=colors[i], label=f'Cluster {i+1}', s=8, alpha=0.5)
    
    # Set labels using sequence names with increased font size
    ax.set_xlabel(sequence_names[0], fontsize=14, labelpad=15)
    ax.set_ylabel(sequence_names[1], fontsize=14, labelpad=15)
    ax.set_zlabel(sequence_names[2], fontsize=14, labelpad=15)
    
    # Increase tick label size
    ax.tick_params(axis='x', labelsize=12, pad=8)
    ax.tick_params(axis='y', labelsize=12, pad=8)
    ax.tick_params(axis='z', labelsize=12, pad=8)
    
    # Position legend outside the plot area
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Save multiple view angles
    angles = [(30, 30), (30, 120), (30, 210), (30, 300), 
              (60, 30), (60, 120), (60, 210), (60, 300),
              (0, 0), (0, 90), (0, 180), (0, 270)]
    
    filenames = []
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        
        # Create filename
        filename = f"{filename_base}_angle_{i+1:02d}.png"
        
        # Save with tight layout and increased DPI
        plt.savefig(
            filename, 
            dpi=200,
            bbox_inches='tight',
            pad_inches=0.5
        )
        filenames.append(filename)
        print(f"Saved view: {filename}")
    
    # Create rotation animation
    create_rotation_gif(fig, ax, filename_base + "_rotation.gif")
    
    plt.close(fig)
    return filenames


def create_rotation_gif(fig, ax, gif_filename, num_frames=36, duration=100):

    # Create rotation animation GIF of 3D plot
    
    # Parameters:
    # fig - matplotlib figure object
    # ax - matplotlib axis object
    # gif_filename - output GIF filename
    # num_frames - number of frames
    # duration - frame duration in milliseconds

    print(f"Creating rotation animation: {gif_filename}")
    
    # Generate frames for rotation animation
    frames = []
    for angle in range(0, 360, 360 // num_frames):
        ax.view_init(elev=30, azim=angle)
        plt.draw()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
    
    # Save as GIF
    imageio.mimsave(gif_filename, frames, duration=duration, loop=0)
    print(f"Saved rotation GIF: {gif_filename}")


def find_elbow_point(distortions):

    # Find the elbow point in the distortion curve using the maximum curvature method
    
    # Parameters:
    # distortions - list of distortion values for k=2 to k=5
    
    # Returns:
    # elbow_k - the k value at the elbow point

    k_range = np.array(range(2, 10))
    distortions = np.array(distortions)
    
    # Calculate the normalized values
    k_normalized = (k_range - min(k_range)) / (max(k_range) - min(k_range))
    d_normalized = (distortions - min(distortions)) / (max(distortions) - min(distortions))
    
    # Calculate the curvature
    curve = np.zeros(len(k_range))
    for i in range(1, len(k_range)-1):
        dx1 = k_normalized[i] - k_normalized[i-1]
        dy1 = d_normalized[i] - d_normalized[i-1]
        dx2 = k_normalized[i+1] - k_normalized[i]
        dy2 = d_normalized[i+1] - d_normalized[i]
        
        # Calculate curvature using the formula
        numerator = abs(dy2/dx2 - dy1/dx1)
        denominator = (1 + (dy1/dx1)**2)**1.5
        curve[i] = numerator / denominator if denominator != 0 else 0
    
    # Find the point with maximum curvature
    elbow_index = np.argmax(curve[1:-1]) + 1
    return k_range[elbow_index]


def evaluate_cluster_numbers(all_data, output_folder):

    # Evaluate different cluster numbers using multiple metrics and plot results
    
    # Parameters:
    # all_data - input data for clustering
    # output_folder - directory to save evaluation plots

    # Set seaborn style for better looking plots
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    # Range of cluster numbers to evaluate
    k_range = range(2, 10)
    
    # Initialize metrics storage
    distortions = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    print("\nEvaluating cluster numbers...")
    for k in k_range:
        print(f"  Testing k = {k}")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20).fit(all_data)
        
        # Calculate distortion (inertia)
        distortions.append(kmeans.inertia_)
        
        # Calculate silhouette score (sampling if large dataset)
        sample_size = min(10000, len(all_data))
        sample_indices = np.random.choice(len(all_data), sample_size, replace=False)
        sample_data = all_data[sample_indices]
        sample_labels = kmeans.labels_[sample_indices]
        silhouette_scores.append(silhouette_score(sample_data, sample_labels))
        
        # Calculate Calinski-Harabasz index
        ch_scores.append(calinski_harabasz_score(all_data, kmeans.labels_))
        
        # Calculate Davies-Bouldin index
        db_scores.append(davies_bouldin_score(all_data, kmeans.labels_))
    
    # Find optimal k for each method
    elbow_k = find_elbow_point(distortions)
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    ch_k = k_range[np.argmax(ch_scores)]
    db_k = k_range[np.argmin(db_scores)]
    
    # Plot distortion (elbow method)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distortions, 'bo-', linewidth=2.5, markersize=10)
    
    # Highlight elbow point
    elbow_idx = k_range.index(elbow_k)
    plt.plot(elbow_k, distortions[elbow_idx], 'ro', markersize=12, 
             label=f'Elbow Point (k={elbow_k})')
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Distortion (Inertia)', fontsize=12)
    plt.title(f'Elbow Method for Optimal k (Suggested k={elbow_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'elbow_method.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2.5, markersize=10)
    
    # Highlight best silhouette score
    silhouette_idx = k_range.index(silhouette_k)
    plt.plot(silhouette_k, silhouette_scores[silhouette_idx], 'ro', markersize=12, 
             label=f'Best Silhouette (k={silhouette_k})')
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title(f'Silhouette Score for Optimal k (Best k={silhouette_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'silhouette_score.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot Calinski-Harabasz index
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, ch_scores, 'ro-', linewidth=2.5, markersize=10)
    
    # Highlight best CH score
    ch_idx = k_range.index(ch_k)
    plt.plot(ch_k, ch_scores[ch_idx], 'bo', markersize=12, 
             label=f'Best Calinski-Harabasz (k={ch_k})')
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Calinski-Harabasz Index', fontsize=12)
    plt.title(f'Calinski-Harabasz Index for Optimal k (Best k={ch_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'calinski_harabasz.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot Davies-Bouldin index
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, db_scores, 'mo-', linewidth=2.5, markersize=10)
    
    # Highlight best DB score
    db_idx = k_range.index(db_k)
    plt.plot(db_k, db_scores[db_idx], 'go', markersize=12, 
             label=f'Best Davies-Bouldin (k={db_k})')
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Davies-Bouldin Index', fontsize=12)
    plt.title(f'Davies-Bouldin Index for Optimal k (Best k={db_k})', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'davies_bouldin.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print evaluation results
    print("\nCluster Evaluation Results:")
    print("k\tDistortion\tSilhouette\tCalinski-H.\tDavies-B.")
    for i, k in enumerate(k_range):
        print(f"{k}\t{distortions[i]:.2f}\t{silhouette_scores[i]:.4f}\t{ch_scores[i]:.2f}\t{db_scores[i]:.4f}")
    
    # Print suggested optimal k values
    print("\nSuggested optimal cluster numbers:")
    print(f"  Elbow method: k = {elbow_k}")
    print(f"  Silhouette score: k = {silhouette_k}")
    print(f"  Calinski-Harabasz: k = {ch_k}")
    print(f"  Davies-Bouldin: k = {db_k}")
    
    return elbow_k, silhouette_k, ch_k, db_k


def perform_kmeans(all_data, output_folder, sequence_names, n_clusters=3):

    # Perform K-means clustering with specified number of clusters
    
    # Parameters:
    # all_data - input data for clustering
    # output_folder - directory to save results
    # sequence_names - names of the sequences
    # n_clusters - number of clusters to use (default=3)

    # First evaluate different cluster numbers
    elbow_k, silhouette_k, ch_k, db_k = evaluate_cluster_numbers(all_data, output_folder)
    
    print(f"\nPerforming final clustering with k = {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=20).fit(all_data)
    labels = kmeans.labels_
    
    # Plot with multiple angles using sequence names
    plot_cluster(np.array(all_data).T, labels, n_clusters, 
                os.path.join(output_folder, 'kmeans_cluster_all_patients'),
                sequence_names)
    
    # Calculate cluster value ranges
    cluster_ranges = {i: {sequence_names[0]: [np.inf, -np.inf], 
                         sequence_names[1]: [np.inf, -np.inf], 
                         sequence_names[2]: [np.inf, -np.inf]} 
                     for i in range(n_clusters)}
    
    all_data_np = np.array(all_data)
    
    for idx, point in enumerate(all_data_np):
        cluster_idx = labels[idx]
        # Update range for each sequence
        for seq_idx, seq_name in enumerate(sequence_names):
            if point[seq_idx] < cluster_ranges[cluster_idx][seq_name][0]:
                cluster_ranges[cluster_idx][seq_name][0] = point[seq_idx]
            if point[seq_idx] > cluster_ranges[cluster_idx][seq_name][1]:
                cluster_ranges[cluster_idx][seq_name][1] = point[seq_idx]
    
    # Save results to file
    range_report_path = os.path.join(output_folder, "cluster_value_ranges.txt")
    with open(range_report_path, "w") as f:
        f.write(f"Final clustering with k = {n_clusters}\n")
        f.write("Value ranges for each cluster across sequences:\n")
        f.write("="*50 + "\n")
        f.write(f"Suggested optimal k values:\n")
        f.write(f"  Elbow method: {elbow_k}\n")
        f.write(f"  Silhouette score: {silhouette_k}\n")
        f.write(f"  Calinski-Harabasz: {ch_k}\n")
        f.write(f"  Davies-Bouldin: {db_k}\n")
        f.write("="*50 + "\n")
        for cluster_idx, ranges in cluster_ranges.items():
            f.write(f"Cluster {cluster_idx+1}:\n")
            for seq_name in sequence_names:
                f.write(f"  {seq_name}:  [{ranges[seq_name][0]:.4f}, {ranges[seq_name][1]:.4f}]\n")
            f.write("-"*50 + "\n")
    
    # Print results to console
    print("\nCluster value ranges:")
    print("="*50)
    for cluster_idx, ranges in cluster_ranges.items():
        print(f"Cluster {cluster_idx+1}:")
        for seq_name in sequence_names:
            print(f"  {seq_name}:  [{ranges[seq_name][0]:.4f}, {ranges[seq_name][1]:.4f}]")
        print("-"*50)
    
    return labels, n_clusters


def process_image_data(path_to_img_folder1, path_to_img_folder2, path_to_img_folder3, 
                       roi_folder, output_folder, sequence_names, n_clusters=3):

    # Process image data and perform clustering
    
    # Parameters:
    # n_clusters - number of clusters to use (default=3)

    filenames = os.listdir(path_to_img_folder1)
    all_data = []
    rois = []
    
    for filename in filenames:
        if filename.endswith(".nii.gz"):
            # Load images
            img1 = nib.load(os.path.join(path_to_img_folder1, filename))
            print(os.path.join(path_to_img_folder1, filename))
            img2 = nib.load(os.path.join(path_to_img_folder2, filename))
            img3 = nib.load(os.path.join(path_to_img_folder3, filename))
            roi = nib.load(os.path.join(roi_folder, filename))
            
            # Get image data
            img1_data = img1.get_fdata()
            img2_data = img2.get_fdata()
            img3_data = img3.get_fdata()
            roi_data = roi.get_fdata()
            
            # Create ROI mask
            roi_mask = roi_data != 0
            
            # Extract ROI values
            x = img1_data[roi_mask]
            y = img2_data[roi_mask]
            z = img3_data[roi_mask]
            
            # Store ROI info
            rois.append({
                'roi_data': roi_data, 
                'roi_mask': roi_mask, 
                'affine': roi.affine
            })
            
            # Combine data
            all_data.extend(np.column_stack((x, y, z)))
    
    # Convert to numpy array for clustering
    all_data = np.array(all_data)
    
    # Perform clustering with specified number of clusters
    labels, n_clusters = perform_kmeans(all_data, output_folder, sequence_names, n_clusters)
          
    start = 0
    for i, filename in enumerate(filenames):
        if filename.endswith(".nii.gz"):
            roi_data = rois[i]['roi_data']
            roi_mask = rois[i]['roi_mask']
            affine = rois[i]['affine']
            
            # Apply cluster labels to ROI
            roi_kmeans = roi_data.copy()
            end = start + np.count_nonzero(roi_mask)
            roi_kmeans[roi_mask] = labels[start:end] + 1 
            
            # Save cluster result
            cluster_nii = nib.Nifti1Image(roi_kmeans, affine=affine)
            nib.save(cluster_nii, os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans.nii.gz")))
            
            # Plot per-patient clusters with sequence names
            patient_data = all_data[start:end]
            if len(patient_data) > 0:  # Ensure data exists
                plot_cluster(
                    patient_data.T, 
                    labels[start:end], 
                    n_clusters, 
                    os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans")),
                    sequence_names
                )
            start = end


# ============== MAIN CONFIGURATION ==============
# Define sequence names
sequence_names = ["ADC", "D", "Dstar"]  # Customize as needed

# Manually set the number of clusters to use
N_CLUSTERS = 4  # Change this value based on evaluation results

# Define paths
path_to_img_folder1 = "/data/yuwei/code/yanhong/data/IVIM/ADC-resp_normalization/"
path_to_img_folder2 = "/data/yuwei/code/yanhong/data/IVIM/D-reg-resp_normalization/"
path_to_img_folder3 = "/data/yuwei/code/yanhong/data/IVIM/Dstar-reg-resp_normalization/"
roi_folder = "/data/yuwei/code/yanhong/data/IVIM/masks/"
output_folder = "/data/yuwei/code/yanhong/data/IVIM_habitat/k-means4_ADC_D_Dstar/"

# Create output directory if needed
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process image data with specified number of clusters
process_image_data(
    path_to_img_folder1, 
    path_to_img_folder2, 
    path_to_img_folder3, 
    roi_folder, 
    output_folder,
    sequence_names,
    n_clusters=N_CLUSTERS  # Pass the manually specified cluster number
)