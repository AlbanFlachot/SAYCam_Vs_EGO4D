import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
import os
import seaborn as sns
import lib.utils_CKA as cka


### METRICS
def cosine_similarity(vector1, vector2):
    """
    Compute the centered cosine similarity between two dissimilarity matrices.

    Parameters:
    -----------
    vector1, vector2 : numpy.ndarray
        1D dissimilarity vectors (they are the flattened upper quadrant of Dissimilarity matrix - likely centered (recommended))

    Returns:
    --------
    float
        cosine similarity
    """
    # Ensure matrices have the same shape
    assert vector1.shape == vector2.shape, "inputs must have the same shape"

    # Compute cosine similarity manually
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Handle division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_sim = dot_product / (norm1 * norm2)
    return cosine_sim

# dot product
def dissimilarity_dotproduct(vectors, normalize=False):
    if normalize:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # Compute similarity matrix (cosine similarity)
    similarity_matrix = np.dot(vectors, vectors.T)
    # Convert to dissimilarity: 1 - similarity
    return 1 - similarity_matrix

## L2 norm

from sklearn.metrics.pairwise import euclidean_distances ##

def similarity_L2norm(vectors, squared = False, normalize=False):
    if normalize:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # Compute similarity matrix (l2 norm similarity)
    return euclidean_distances(vectors , squared= squared) # We use sklear instead of scipy coz for large matrices it is much wafaster!


def compute_RDMs(activations, metric = 'pearson', display = True, title = '', torch = True):

    '''
    activations should have shape (nb_images, nb_features)
    '''

    if 'normalize' in metric:
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        activations = activations/norms


    if 'pearson' in metric:
        RDM = 1-np.corrcoef(activations)
    elif 'L2squared' in metric:
        RDM = similarity_L2norm(activations, squared = True)
    elif ('L2' in metric) and not ('squared' in metric):
        RDM = similarity_L2norm(activations)
    elif 'dotproduct' in metric:
        RDM = dissimilarity_dotproduct(activations)

    if display:
        fig, subs = plt.subplots(1,1)
        sns.heatmap(RDM,
                    annot=False,
                    cmap='Greys',      # Blue to red colormap
                    square=True,
                    cbar=True,
                    #cbar_kws={'label': 'Dissimilarity'},
                    #fmt='.2f',
                    linewidths=0,
                    ax = subs,
                    vmin=0,               # Set minimum value for color scale
                    vmax=np.max(RDM))               # Set maximum value for color scale

        subs.set_title(title)
        subs.axis('off')
        fig.tight_layout()
        plt.show()

    return RDM

#### Utils
def compute_RDMs_models(model, dataset, listimages, metric = 'pearson', normalize = False, display = True):
    path2activations = f'/data/alban/activations/{model}_{dataset}'
    activations = list()
    #patch_token = list()
    for i, im in enumerate(listimages):
        activations.append(np.load(join(path2activations, f'cls_token_{im[:-4]}.npy')))

    nb_layers = 12
    activations = np.array(activations).reshape(len(listimages), nb_layers, -1)
    if normalize:
        norms = np.linalg.norm(activations, axis=2, keepdims=True)
        activations = activations/norms

    RDMs = list()
    for lay in range(activations.shape[1]):
        if metric == 'pearson':
            RDMs.append(1-np.corrcoef(activations[:,lay]))
        elif metric == 'L2':
            RDMs.append(similarity_L2norm(activations[:,lay]))


    if display:
        fig, subs = plt.subplots(1,1)
        sns.heatmap(RDMs[-1],
                    annot=False,
                    cmap='Greys',      # Blue to red colormap
                    square=True,
                    cbar=True,
                    #cbar_kws={'label': 'Dissimilarity'},
                    #fmt='.2f',
                    linewidths=0,
                    ax = subs,
                    vmin=0,               # Set minimum value for color scale
                    vmax=np.max(RDMs[-1]))               # Set maximum value for color scale


        subs.set_title(f'Trained on {model} and tested on {dataset}')

        subs.axis('off')
        fig.tight_layout()
        plt.show()
        fig.savefig(f'figures/RDM_{model}_{dataset}.png', dpi=300, bbox_inches='tight')
        return RDMs



def Compute_sim_RDMs(RDM1, RDM2, center = False, metric = 'cosine', means = {}):#
    '''
    Function to compute correlational similarity between 2 RDMs.
    Only considers the upper triangular part, excluding the diagonal.
    Allows for centering of matrices (as in CKA) and several metrics of similarity ('cosine', 'pearson', 'spearman').)
    '''

    assert RDM1.shape == RDM2.shape # two RDMs should be of the same size
    assert metric in ['cosine', 'pearson', 'spearman', 'pearson_global']

    if center: ### center matrices
        RDM1 = cka.centering(RDM1)
        RDM2 = cka.centering(RDM2)

    n = len(RDM1)
    upper_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    upper_RDM1 = RDM1[upper_indices]
    upper_RDM2 = RDM2[upper_indices]

    # compute similarity
    if metric == 'cosine':
        sim = cosine_similarity(upper_RDM1, upper_RDM2)
    elif metric == 'pearson':
        #print(np.corrcoef(upper_RDM1, upper_RDM2))
        sim =np.corrcoef(upper_RDM1, upper_RDM2)[0,1]
    elif metric == 'pearson_global': # to use for a subset if want to predict similarity measure for a larger set of RDM entries
        RDM1_centered = RDM1[upper_indices] - means['x']
        RDM2_centered = RDM2[upper_indices] - means['y']
        n = len(RDM1_centered)
        cov = np.mean(RDM1_centered*RDM2_centered)
        #norm = np.std(RDM1_centered) * np.std(RDM2_centered)
        norm = means['norm']
        sim = cov/norm
    return sim

def corrs_layers(RDMs, models):
    '''
    Function that computes the correlation between RDMs, layer per layer, for all models, and saves them in a dictionary.
    Note that the dictionary is ranked following the order the of the model names given, to avoid redundancy.
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
