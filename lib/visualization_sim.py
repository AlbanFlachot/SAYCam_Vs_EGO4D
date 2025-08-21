import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
import os
import seaborn as sns


### DISPLAY FUNCTIONS


import math
def plot_similarities(SIMs, submodels):
    '''plot the similarities as a function of layer depth.
    Plot is a subplot of adaptative size, depending on the length of the list submodel given.
    '''
    nb_subs = len(submodels)*(len(submodels)-1)/2 # Number of subs
    sqrt = np.sqrt(nb_subs)
    cols = math.ceil(sqrt)
    rows = math.ceil(nb_subs // sqrt)
    while (cols*rows)<(nb_subs):
        rows =rows + 1 ## compute the number of columns and rows
    fig, subs = plt.subplots(rows,cols, sharex=True, sharey=True, figsize=(cols*2+1, rows*2+1)) # adaptative size
    count = 0
    minval = 1
    maxval = 0
    for i, model1 in enumerate(submodels):
        for j, model2 in enumerate(submodels[i+1:]):
            minval = min(minval, np.amin(SIMs[model1][model2]))
            maxval = max(maxval, np.amax(SIMs[model1][model2]))
            if rows ==1:
                subs[count%cols].plot(SIMs[model1][model2])
                subs[count%cols].set_title(f'{model1}_{model2}')
            else:
                subs[count//cols, count%cols].plot(SIMs[model1][model2])
                subs[count//cols, count%cols].set_title(f'{model1}_{model2}')
            count+=1
    plt.ylim(np.round(minval,1)-0.1, np.round(maxval,1)+0.1)
    if rows == 1:
        subs[0].set_ylabel('Correlation')
        for sub in subs:
            sub.set_xlabel('Layer')
    else:
        for sub in subs[-1]:
            sub.set_xlabel('Layer')
        for sub in subs[:,0]:
            sub.set_ylabel('Correlation')
    fig.tight_layout()
    plt.show()
    plt.close()



def display_RDM(RDM, model):
    '''
    Function to display RDM in a figure as heatmaps.
    '''
    fig, subs = plt.subplots(1,1)
    # Using a different colormap that goes from blue (similar) to red (dissimilar)
    sns.heatmap(RDM,
                annot=False,
                cmap='Greys',      # Blue to red colormap
                square=True,
                cbar=True,
                cbar_kws={'label': 'Dissimilarity'},
                #fmt='.2f',
                linewidths=0,
                ax = subs,
                vmin=0,               # Set minimum value for color scale
                vmax=1)               # Set maximum value for color scale

    subs.set_title(model)
    subs.axis('off')
    fig.tight_layout()
    plt.show()
    #fig.savefig(f'../figures/RDM_{model}.png', dpi=300, bbox_inches='tight')


# t-SNE functions


from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def tsne_from_dissimilarity_matrix(dissimilarity_matrix, labels, size = 92,
                                 perplexity=30, n_iter=1000, random_state=42):
    """
    Create t-SNE visualization from dissimilarity matrix

    Parameters:
    - dissimilarity_matrix: 100x100 dissimilarity matrix
    - labels: list/array of labels for the 100 images
    - perplexity: t-SNE perplexity parameter
    - n_iter: number of iterations
    - random_state: random seed for reproducibility
    """

    # Ensure dissimilarity matrix is symmetric and valid
    assert dissimilarity_matrix.shape == (size, size), f"Matrix must be {size}x {size}"
    assert len(labels) == size, "Must have 100 labels"


    # Run t-SNE with precomputed distances
    tsne = TSNE(n_components=2,
                metric='precomputed',
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=random_state,
                init='random')

    # Fit and transform
    tsne_results = tsne.fit_transform(dissimilarity_matrix)

    return tsne_results

def plot_tsne_results(tsne_results, labels, figsize=(4.5, 3),
                     title="t-SNE Visualization of Image Dissimilarity"):
    """
    Plot t-SNE results with colored labels
    """
    # Encode labels to numbers for coloring
    le = LabelEncoder()
    label_encoded = le.fit_transform(labels)
    unique_labels = le.classes_

    # Create figure
    plt.figure(figsize=figsize)

    # Create scatter plot
    cmap = 'autumn'
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=label_encoded, cmap=cmap,
                         alpha=0.7, s=50)

    # Add labels and title
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=scatter.cmap(scatter.norm(i)),
                         markersize=8, label=unique_labels[i])
               for i in range(len(unique_labels))]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

def analyze_clusters(tsne_results, labels):
    """
    Analyze cluster quality and separation
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    import numpy as np

    le = LabelEncoder()
    label_encoded = le.fit_transform(labels)
    n_clusters = len(np.unique(labels)) # for kmeans

    # Calculate silhouette score
    sil_score = silhouette_score(tsne_results, label_encoded)
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Number of unique labels: {n_clusters}")

    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} images")

    # Calculate ARI by comparing clustering results to ground truth

    ari_results = {}

    # Method 1: K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(tsne_results)
    ari_kmeans = adjusted_rand_score(label_encoded, kmeans_labels)
    ari_results['KMeans'] = ari_kmeans
    print(f"K-Means ARI: {ari_kmeans:.3f}")

    return sil_score

# Example usage and complete pipeline
def complete_tsne_pipeline(dissimilarity_matrix, labels, title):
    """
    Complete example with synthetic data
    """

    print("Running t-SNE on dissimilarity matrix...")
    size = len(labels)
    # Run t-SNE
    tsne_results = tsne_from_dissimilarity_matrix(dissimilarity_matrix, labels, size = size)

    # Plot results
    fig = plot_tsne_results(tsne_results, labels, title = title)
    plt.show()

    # Analyze clusters
    analyze_clusters(tsne_results, labels)


    return tsne_results, labels