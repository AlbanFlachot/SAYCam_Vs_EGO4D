import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
import os
import seaborn as sns



# dot product
def similarity_dotproduct(vectors):
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # Compute similarity matrix (cosine similarity)
    similarity_matrix = np.dot(vectors_normalized, vectors_normalized.T)
    # Convert to dissimilarity: 1 - similarity
    return 1 - similarity_matrix


## L2 norm

from scipy.spatial.distance import pdist, squareform

def similarity_L2norm(vectors):
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # Compute similarity matrix (l2 norm similarity)
    distances = pdist(vectors_normalized, 'euclidean')
    # Convert to dissimilarity: 1 - similarity
    return squareform(distances)


def compute_RDMs(model, dataset, listimages, display = True):
    path2activations = f'/data/alban/activations/{model}_{dataset}'
    cls_token = list()
    #patch_token = list()
    for i, im in enumerate(listimages):
        cls_token.append(np.load(join(path2activations, f'cls_token_{im[:-4]}.npy')))
        #patch_token.append(np.load(join(path2activations, f'patch_token_{im[:-4]}.npy')))
    cls_token = np.array(cls_token)
    #corr_cls = np.corrcoef(cls_token.reshape(len(listimages), -1))
    CORRs = list()
    for lay in range(cls_token.shape[1]):
        CORRs.append(1-np.corrcoef(cls_token[:,lay]))

    #corr_patch = np.corrcoef(patch_token)

    #l2_cls = similarity_L2norm(cls_token)
    #l2_patch = similarity_L2norm(patch_token)

    if display:
        fig, subs = plt.subplots(1,1)
        # Using a different colormap that goes from blue (similar) to red (dissimilar)
        sns.heatmap(CORRs[-1],
                    annot=False,
                    cmap='Greys',      # Blue to red colormap
                    square=True,
                    cbar=False,
                    #cbar_kws={'label': 'Dissimilarity'},
                    #fmt='.2f',
                    linewidths=0,
                    ax = subs,
                    vmin=0,               # Set minimum value for color scale
                    vmax=1)               # Set maximum value for color scale


        subs.set_title(f'Trained on {model} and tested on {dataset}')

        subs.axis('off')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'figures/RDM_{model}_{dataset}.png', dpi=300, bbox_inches='tight')
        return CORRs#, l2_cls

def Compute_sim_RDMs(RDM1, RDM2):#
    '''
    Function to compute correlational similarity between 2 RDMs.
    Only considers the upper triangular part, excluding the diagonal
    '''
    # Extract upper triangular part (excluding diagonal)
    n = len(RDM1)
    if n != len(RDM2):
        print('RDMs are of different sizes')
    upper_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    upper_RDM1 = RDM1[upper_indices]
    upper_RDM2 = RDM2[upper_indices]

    # compute correlations
    return np.corrcoef(upper_RDM1, upper_RDM2)[0,1]


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

    subs.set_title('correlation')
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
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=label_encoded, cmap='autumn',
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
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    label_encoded = le.fit_transform(labels)

    # Calculate silhouette score
    sil_score = silhouette_score(tsne_results, label_encoded)

    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} images")

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

def corrs_layers(RDMs, models):
    '''
    Function that computes the correlation between RDMs, layer per layer, for all models, and saves them in a dictionary.
    Note that the dictionary is ranked folling the order the of the model names given, to avoid redundancy.
    '''
    SIMs = {}
    for model in models:
        SIMs[model] = {}
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i:]):
            SIMs[model1][model2] = list()
            for lay in range(len(RDMs[model1])):
                SIMs[model1][model2].append(Compute_sim_RDMs(RDMs[model1][lay], RDMs[model2][lay]))
    return SIMs

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