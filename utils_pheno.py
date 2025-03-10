import os
import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import openslide
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity



def extract_patches(idx):
    if "BRACS" in idx:
        # Extract roi_id and coordinates
        roi_id = '_'.join(idx.split('_')[:-2])
        x, y = int(idx.split('_')[-2]), int(idx.split('_')[-1])
        
        # Search for the corresponding ROI image file
        roi_paths = glob.glob(f'/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs/BRACS_ROI/*/*/{roi_id}.png')
        
        # Check if the file exists
        if not roi_paths:
            raise FileNotFoundError(f"ROI image for {roi_id} not found.")
        
        roi_pt = roi_paths[0]  # First match
        roi_im = Image.open(roi_pt)
        
        # Get image dimensions to ensure the patch is within bounds
        img_width, img_height = roi_im.size
        if x + 512 > img_width or y + 512 > img_height:
            raise ValueError(f"Patch coordinates ({x}, {y}) exceed image dimensions ({img_width}, {img_height}).")
        
        # Extract the 512x512 patch from the image
        patch_im = roi_im.crop((x, y, x + 512, y + 512))
                
        return patch_im

    elif "_FPE_" in idx:
        # Extract WSI ID, x, y, and patch size from idx
        wsi_id = '_'.join(idx.split('_')[:3])  # Assuming WSI ID is in the first three parts of idx
        x, y = int(idx.split('_')[-3]), int(idx.split('_')[-2])
        patch_size = int(idx.split('_')[-1])
        x, y = int(x * patch_size), int(y * patch_size)
        
        # Search for the corresponding WSI file (use the correct glob pattern)
        wsi_paths = glob.glob(f'/scratch_tmp/prj/cb_normalbreast/prj_BreastAgeNet/WSIs/*/{wsi_id}*.*')
        
        # Ensure that a WSI path was found
        if not wsi_paths:
            raise FileNotFoundError(f"WSI for {wsi_id} not found.")
        
        # Ensure only one file is found
        if len(wsi_paths) > 1:
            raise ValueError(f"Multiple WSI files found for {wsi_id}, expected only one.")
        
        # Open the WSI file
        wsi_pt = wsi_paths[0]  # Take the first matching file path
        wsi = openslide.OpenSlide(wsi_pt)
        
        # Read the region corresponding to the extracted coordinates
        patch_im = wsi.read_region((x, y), 0, (patch_size, patch_size))  # level=0 for full resolution

        return patch_im
    else:
        raise ValueError(f"Invalid idx format: {idx} does not contain 'BRACS' or '_FPE_'.")




def get_KNN_data(model_stain):
    pt = glob.glob(f'/scratch_tmp/prj/cb_normalbreast/prj_NBTPhenotyping/RESULTS/FoundationModel_features/KHP_RM_{model_stain}_features*.csv')[0]
    print(pt)
    KHP_RM_df = pd.read_csv(pt)
    KHP_RM_df.loc[KHP_RM_df['TC_epi'] > 0.9, ['patch_id', 'wsi_id', 'cohort'] + [col for col in KHP_RM_df.columns if col.startswith('embedding_')]]
    KHP_RM_df['source'] = 'RM'
    print(f'KHP_RM_df: {KHP_RM_df.shape}')

    
    pt = glob.glob(f'/scratch_tmp/prj/cb_normalbreast/prj_NBTPhenotyping/RESULTS/FoundationModel_features/KHP_RRM_{model_stain}_features*.csv')[0]
    print(pt)
    KHP_RRM_df = pd.read_csv(pt)
    KHP_RRM_df = KHP_RRM_df.loc[KHP_RRM_df['TC_epi'] > 0.9, ['patch_id', 'wsi_id', 'cohort'] + [col for col in KHP_RRM_df.columns if col.startswith('embedding_')]]
    KHP_RRM_df['source'] = 'RRM'
    print(f'KHP_RRM_df: {KHP_RRM_df.shape}')

    
    BRACS_df = pd.read_csv(f'/scratch_tmp/prj/cb_normalbreast/prj_NBTPhenotyping/RESULTS/FoundationModel_features/BRACS_sample_{model_stain}_features.csv')
    # num_embeddings = len(BRACS_df.columns)-3
    # column_names = ['roi_id', 'patch_id', 'source'] + [f'embedding_{i}' for i in range(num_embeddings)]
    # BRACS_df.columns = column_names
    BRACS_df['cohort'] = 'BRACS'
    BRACS_df['wsi_id'] = BRACS_df['roi_id']
    print(f'BRACS_df: {BRACS_df.shape}')

    
    df_merge = pd.concat([BRACS_df.loc[:, ['patch_id', 'cohort', 'wsi_id', 'source'] + [col for col in BRACS_df.columns if col.startswith('embedding_')]],
                         KHP_RM_df.loc[:, ['patch_id', 'cohort', 'wsi_id', 'source'] + [col for col in KHP_RM_df.columns if col.startswith('embedding_')]],
                         KHP_RRM_df.loc[:, ['patch_id', 'cohort', 'wsi_id', 'source'] + [col for col in KHP_RRM_df.columns if col.startswith('embedding_')]]])
    print(np.unique(df_merge['cohort'], return_counts=True))
    print(np.unique(df_merge['source'], return_counts=True))


    sampled_df1 = df_merge.groupby('source').apply(lambda x: x.sample(n=8700, replace=True, random_state=42))
    sampled_df2 = df_merge.groupby('cohort').apply(lambda x: x.sample(n=20000, replace=True, random_state=42))
    sampled_df = pd.concat([sampled_df1, sampled_df2])
    sampled_df = sampled_df.drop_duplicates()
    sampled_df = sampled_df.reset_index(drop=True)
    
    save_pt = f'/scratch_tmp/prj/cb_normalbreast/prj_NBTPhenotyping/RESULTS/FoundationModel_features/Combined_sampled{len(sampled_df)}_{model_stain}.csv'
    print(save_pt)
    sampled_df.to_csv(save_pt, index=False)
    return sampled_df



def run_KNN(knn_df, save_pt=None):
    features = knn_df.loc[:, [col for col in knn_df.columns if col.startswith('embedding')]]
    print(features.shape)
    knn_df['label'] = knn_df['source'] 
    labels = knn_df.loc[:, 'label'].replace({'RM': 'N', 'RRM': 'N'})
    print(labels.shape)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'n_neighbors': [8, 16, 32, 64, 128, 256, 512]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold, scoring='accuracy')
    grid_search.fit(features, labels)

    results = pd.DataFrame(grid_search.cv_results_)
    results['n_neighbors'] = results['params'].apply(lambda x: x['n_neighbors'])
    results = results.drop(columns=['params'])
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_}")

    if save_pt:
        results.to_csv(save_pt, index=False)



def knn_acc_lineplot(df, save_pt=None):
    custom_palette = {'UNI': '#E34B36', 'conch': '#EF9B80', 'gigapath': '#3C5587', 'iBOT': '#57BCCC', 'EXAONEPath': '#10A18B'}
    line_style_dict = {'aug': 'solid', 'rein': 'dashed',  'orig': 'dotted'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for i, stain in enumerate(['orig', 'rein', 'aug']):
        stain_df = df[df['stain'] == stain]
        for model_name in stain_df['model'].unique():
            model_df = stain_df[stain_df['model'] == model_name]  
            line_style = line_style_dict.get(stain, 'solid')  
            for model_name, model_data in model_df.groupby('model'):
                color = custom_palette.get(model_name, '#000000') 
                x = model_data['param_n_neighbors']
                y = model_data['mean_test_score']
                axes[i].plot(x, y, label=model_name, color=color, linestyle=line_style_dict[stain], marker='o', markersize=6, linewidth=2.5)
        x_values = [8, 16, 32, 64, 128, 256, 512]
        axes[i].set_xticks(x_values)
        axes[i].set_title(f'Line Style: {stain.capitalize()}', fontsize=14)
        axes[i].set_xlabel('Number of Neighbors (param_n_neighbors)', fontsize=12)
        axes[i].set_ylabel('Mean Test Accuracy', fontsize=12)
        axes[i].grid(alpha=0.3)
        
    plt.tight_layout() 
    handles, labels = axes[0].get_legend_handles_labels() 
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    fig.legend(handles=unique_handles, labels=unique_labels, title='Models', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
    if save_pt:
        plt.savefig(save_pt, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()




def knn_acc_boxplot(df, save_pt=None):
    model_order = ['UNI', 'iBOT', 'conch', 'gigapath', 'EXAONEPath']
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    param_values = [8, 16, 32, 64, 128, 256, 512]
    cmap = plt.cm.binary  
    norm = plt.Normalize(vmin=min(param_values), vmax=max(param_values))
    legend_handles = [mlines.Line2D([], [], marker='o', color='w', markerfacecolor=cmap(norm(param_value)), markersize=10, label=f'{param_value}') for param_value in param_values]

    for i, stain in enumerate(['orig', 'rein', 'aug']):
        ax = axes[i]
        stain_df = df[df['stain'] == stain]
        sns.boxplot(data=stain_df, x='model', y='mean_test_score', hue='model', palette=custom_palette, order=model_order, ax=ax, legend=False)
        for model_name in model_order:
            model_df = stain_df[stain_df['model'] == model_name]
            for param_value in param_values:
                param_df = model_df[model_df['param_n_neighbors'] == param_value]
                # Scatter plot for each point
                ax.scatter([model_name] * len(param_df), param_df['mean_test_score'], zorder=10, s=50, color=cmap(norm(param_value)))
        ax.set_title(f'{stain.capitalize()} Stain', fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticks(range(len(model_order)))  # Set the tick positions
        ax.set_xticklabels(model_order, rotation=45)  # Set the tick labels
        if i == 0:
            ax.set_ylabel('Mean Test Accuracy', fontsize=12)
        else:
            ax.set_ylabel('')
        ax.grid(alpha=0.3)
    
    fig.legend(handles=legend_handles, title="param_n_neighbors", loc='upper right', fontsize=12, bbox_to_anchor=(1.2, 0.8))
    fig.subplots_adjust(right=1.25)  
    plt.tight_layout()  
    if save_pt:
        plt.savefig(save_pt, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def get_adata(knn_df):
    adata = anndata.AnnData(X=knn_df.loc[:, [f'embedding_{i}' for i in range(512)]].to_numpy(), 
                        obs=knn_df.loc[:, ['patch_id', 'cohort', 'wsi_id', 'source']])
    sc.tl.pca(adata, svd_solver='arpack')
    # sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, method='umap', metric='euclidean', key_added='nn_15')
    return adata
    


def run_leiden_umap(resolution=0.5):
    groupby = f"leiden_{resolution}"
    sc.tl.leiden(adata, resolution=resolution, key_added=groupby, neighbors_key='nn_15')
    sc.tl.paga(adata, groups=groupby, neighbors_key=neighbors_key)
    # sc.pl.paga(adata, color=[groupby, 'source', 'cohort'], layout='fr', random_state=0, 
    #            threshold=0.29, node_size_scale=8, node_size_power=0.5, edge_width_scale=.05, fontsize=10, 
    #            fontoutline=2, cmap="Set1",frameon=False, show=False)
    sc.tl.umap(adata, init_pos="paga", neighbors_key=neighbors_key)
    return adata




def plot_umap_with_random_wsi(adata, n_iterations=10, ax=None, save_pt=None):
    wsi_counts = adata.obs['wsi_id'].value_counts()
    wsi_ids_more_than_100 = wsi_counts[wsi_counts > 100].index

    random_wsi_ids = np.random.choice(wsi_ids_more_than_100, n_iterations, replace=False)
    adata.obs['highlight'] = adata.obs['wsi_id'].apply(lambda x: x if x in random_wsi_ids else 'other')
    adata.obs['size'] = adata.obs['wsi_id'].apply(lambda x: 50 if x in random_wsi_ids else 5)
    adata.obs['alpha'] = adata.obs['wsi_id'].apply(lambda x: 1 if x in random_wsi_ids else 0.2)

    # Use the provided 'ax' for plotting
    sc.pl.umap(
        adata, 
        color='highlight', 
        size=adata.obs['size'], 
        alpha=adata.obs['alpha'], 
        ax=ax, 
        legend_loc=None,  # Disable Scanpy legend
        show=False
    )

    # Creating the custom legend for WSI IDs
    unique_labels = list(random_wsi_ids) + ['other']
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                          markerfacecolor='C{}'.format(i % 10), markersize=10)
               for i, label in enumerate(unique_labels)]
    ax.legend(handles=handles, title='WSI ID', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    ax.set_title("UMAP of 10 Randomly Selected WSIs", fontsize=16)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)

    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight', dpi=300)




def compute_centroids(adata):
    umap_coords = adata.obsm['X_umap']
    umap_df = pd.DataFrame(umap_coords, index=adata.obs.index, columns=['UMAP1', 'UMAP2'])
    centroids = adata.obs.groupby('source').apply(
        lambda group: np.mean(umap_df.loc[group.index, :], axis=0)
    )
    return centroids



def sample_closest_points(adata, tissue_type, centroid, num_samples=5000):
    data = adata.obsm['X_umap'][adata.obs['source'] == tissue_type]
    distances = np.linalg.norm(data - centroid, axis=1)
    closest_indices = np.argsort(distances)[:num_samples]
    return data[closest_indices]



def fit_kde_to_data(data_dict):
    kde_dict = {}
    for tissue_type, data in data_dict.items():
        kde_dict[tissue_type] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
    return kde_dict



def compute_log_density(kde_dict, x_grid):
    log_dens_dict = {}
    for tissue_type, kde in kde_dict.items():
        log_dens_dict[tissue_type] = kde.score_samples(x_grid)
    return log_dens_dict



def compute_kl_divergence_between_types(log_dens_dict, tissue_types):
    kl_results = {}
    for i, tissue_type_i in enumerate(tissue_types):
        for tissue_type_j in tissue_types[i + 1:]:  
            p = log_dens_dict[tissue_type_i]
            q = log_dens_dict[tissue_type_j]
            
            kl_div = np.sum(np.exp(p) * (p - q))  # Assuming p and q are log densities
            kl_results[(tissue_type_i, tissue_type_j)] = kl_div
    return kl_results



def compute_kl_divergences(adata, tissue_types, x_grid, num_samples=5000):
    centroids = compute_centroids(adata)
    
    data_dict = {}
    for tissue_type in tissue_types:
        centroid = centroids.loc[tissue_type].values  
        sampled_data = sample_closest_points(adata, tissue_type, centroid, num_samples)
        data_dict[tissue_type] = sampled_data

    kde_dict = fit_kde_to_data(data_dict)
    log_dens_dict = compute_log_density(kde_dict, x_grid)
    kl_results = compute_kl_divergence_between_types(log_dens_dict, tissue_types)
    
    kl_divergences_df = pd.DataFrame(np.nan, index=tissue_types, columns=tissue_types)
    for (tissue_type_i, tissue_type_j), kl_value in kl_results.items():
        kl_divergences_df.loc[tissue_type_i, tissue_type_j] = kl_value
        kl_divergences_df.loc[tissue_type_j, tissue_type_i] = kl_value 

    return kl_divergences_df



def normalize_kl_divergences(kl_divergences_df, method='minmax'):
    if method == 'minmax':
        kl_divergences_df_normalized = (kl_divergences_df - kl_divergences_df.min().min()) / (kl_divergences_df.max().max() - kl_divergences_df.min().min())
    elif method == 'zscore':
        kl_divergences_df_normalized = (kl_divergences_df - kl_divergences_df.mean().mean()) / kl_divergences_df.std().std()
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")
    return kl_divergences_df_normalized



def plot_kl_divergence_heatmap(adata, normalization='minmax', save_pt=None):
    umap_coords = adata.obsm['X_umap']  
    umap_range = np.array([ [umap_coords[:, 0].min(), umap_coords[:, 0].max()], [umap_coords[:, 1].min(), umap_coords[:, 1].max()] ])  
    x_grid, y_grid = np.meshgrid(np.linspace(umap_range[0, 0], umap_range[0, 1], 1000), np.linspace(umap_range[1, 0], umap_range[1, 1], 1000))
    x_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T  

    tissue_types = ['IC', 'DCIS', 'ADH', 'FEA', 'UDH', 'PB', 'N', 'RRM', 'RM']
    kl_divergences_df = compute_kl_divergences(adata, tissue_types, x_grid, num_samples=5000)
    kl_divergences_df_normalized = normalize_kl_divergences(kl_divergences_df, method=normalization)
    kl_divergences_df_normalized = kl_divergences_df_normalized.loc[tissue_types, tissue_types]
    mask = np.triu(np.ones_like(kl_divergences_df_normalized, dtype=bool), k=1)  
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(kl_divergences_df_normalized, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, mask=mask, linewidths=0.5)
    plt.title("Pairwise KL Divergence Between Cohort Centroids (Randomly Sampled Points)", fontsize=14)
    plt.xlabel("Cohorts", fontsize=12)
    plt.ylabel("Cohorts", fontsize=12)
    plt.grid(False)  # Disable grid lines
    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight', dpi=300, format='pdf')  # Save the plot as a PDF
    plt.show()




def plot_umap_highlight_each_source(adata, save_pt=None):
    # Define the desired order for the sources
    # ordered_sources = ['IC', 'DCIS', 'ADH', 'FEA', 'UDH', 'PB', 'N', 'RRM', 'RM']
    ordered_sources = ['N', 'RRM', 'RM']

    # Make sure that the sources in adata.obs['source'] are in the same set
    unique_sources = np.unique(adata.obs['source'])
    assert set(ordered_sources).issubset(unique_sources), "Some sources are missing from the data."
    
    # Create a 3x3 grid for the subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten axes to iterate over them easily
    
    # Define the custom color mapping
    # source_colors = {
    #     'IC': '#D62A28', 
    #     'DCIS': '#F57F20', 
    #     'ADH': '#2278B5', 
    #     'FEA': '#279E69', 
    #     'UDH': '#1FBFD0', 
    #     'PB': '#8D574C', 
    #     'N': '#855AA5', 
    #     'RM': '#D87AB2', 
    #     'RRM': '#B6BE62',
    #     'other': 'gray'  # Color for "other" group
    # }

    source_colors = {
        'N': '#855AA5', 
        'RM': '#D87AB2', 
        'RRM': '#B6BE62',
        'other': 'gray'  # Color for "other" group
    }
    
    # Loop over each source in the specified order and assign a subplot
    for i, source in enumerate(ordered_sources):
        ax = axes[i]  # Get corresponding axis for this source
        
        # Set the "highlight" column to show this source and "other" for the rest
        adata.obs['highlight'] = adata.obs['source'].apply(lambda x: x if x == source else 'other')
        
        # Set point size and alpha based on whether the source matches
        adata.obs['size'] = adata.obs['source'].apply(lambda x: 30 if x == source else 5)
        adata.obs['alpha'] = adata.obs['source'].apply(lambda x: 1 if x == source else 0.1)
        
        # Plot the UMAP with color assigned based on 'highlight' column
        sc.pl.umap(
            adata, 
            color='highlight', 
            size=adata.obs['size'], 
            alpha=adata.obs['alpha'], 
            ax=ax,  # Specify axis for this subplot
            legend_loc=None,  # Disable Scanpy legend
            show=False,  # Don't show immediately
            title=source,  # Title for each subplot
            palette=source_colors  # Set custom palette
        )
        
        # Customizing each subplot's background and colors
        ax.set_facecolor('white')  # Ensure the background color is consistent across all plots
        ax.set_xticks([])  # Optional: remove ticks for cleaner look
        ax.set_yticks([])  # Optional: remove ticks for cleaner look
    
    # Add the legend for all sources on the side of the grid
    unique_labels = list(ordered_sources) + ['other']
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                          markerfacecolor=source_colors.get(label, 'gray'), markersize=10)
               for i, label in enumerate(unique_labels)]
    
    # Add a unified legend outside the grid
    fig.legend(handles=handles, title='Source', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    
    # Set common labels for axes and title
    fig.suptitle("UMAP of Each Breast Tissue Source", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between subplots
    
    # Save or show the plot
    if save_pt:
        plt.savefig(save_pt, bbox_inches='tight', dpi=300)
    
    plt.show()




def calculate_top_samples(adata, groupby, top_n=50):
    features = adata.X  # Assuming this is the raw feature matrix
    cluster_labels = adata.obs[groupby].astype(int)  # Assuming 'leiden_1.0' contains cluster labels
    
    centroids = []
    for cluster in np.unique(cluster_labels):
        cluster_samples = features[cluster_labels == cluster]  # Get the samples in the current cluster
        centroid = np.mean(cluster_samples, axis=0)  # Calculate the mean of the cluster
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    distances = pairwise_distances(features, centroids[cluster_labels])
 
    top_samples_idx = []
    for cluster in np.unique(cluster_labels):
        cluster_sample_indices = np.where(cluster_labels == cluster)[0]
        cluster_distances = distances[cluster_sample_indices, cluster]  
        sorted_idx = np.argsort(cluster_distances)[:top_n] 
        top_samples_idx.extend(cluster_sample_indices[sorted_idx])
    
    top_samples = adata.obs.iloc[top_samples_idx]
    adata.obs[f'{groupby}_top_samples'] = False
    adata.obs.loc[top_samples.index, f'{groupby}_top_samples'] = True
    print(f"Top {top_n} central samples indices: {top_samples.index}")

    return adata



    
def plot_cluster_exps(adata, groupby, model_stain, sampled_number=36, save_dir=None, save_imgs=False):
    top_samples_dict = {}

    for cluster in np.unique(adata.obs[groupby]):
        top_samples = adata.obs.loc[(adata.obs[groupby] == cluster)]
        top_samples_dict[f'cluster_{cluster}'] = top_samples.index.tolist()

    os.makedirs(save_dir, exist_ok=True)

    if save_imgs:
        patch_dir = f"{save_dir}_imgs"
        os.makedirs(patch_dir, exist_ok=True)

    for cluster, patch_ids in top_samples_dict.items():
        img_list = []

        for idx in patch_ids:
            try:
                patch = extract_patch_from_roi(idx)  # You must define this function
                img_list.append(np.array(patch))

                if save_imgs:
                    patch.save(f"{patch_dir}/{cluster}_{idx}.png")

            except Exception as e:
                print(f"Warning: Skipping patch {idx} due to error: {e}")

            if len(img_list) == sampled_number:
                break

        if len(img_list) == 0:
            print(f"No valid patches found for cluster {cluster}, skipping plot.")
            continue

        grid_size = int(np.ceil(sqrt(sampled_number)))  # Use ceil to ensure space for all images

        save_pt = f'{save_dir}/{model_stain}_{groupby}_cluster{cluster}.png'
        plot_multiple(img_list=img_list, 
                      caption_list=[cluster]*len(img_list), 
                      grid_x=grid_size, grid_y=grid_size, 
                      figure_size=(10, 10), 
                      title=f'Cluster {cluster}', 
                      save_pt=save_pt)

        print(f"Saved cluster {cluster} plot with {len(img_list)} patches.")




