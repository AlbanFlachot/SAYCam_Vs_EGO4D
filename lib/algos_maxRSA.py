import numpy as np
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def categorical_silhouette_score(cat_activations, models, listcat,
                                                 metric='euclidean', squared=True):
    """
    Memory-efficient version using sklearn's euclidean_distances with squared distances.
    Best for very large datasets. Uses squared euclidean distance for speed.

    Args:
        cat_activations: dict[model] -> list of category activation arrays
        models: list of model names
        listcat: list of category names
        metric: distance metric ('euclidean' or others)
        squared: if True, use squared euclidean distance (faster, avoids sqrt)
    """
    compactness2 = {}
    compact_categories2 = {}

    for model in models:
        print(model)
        n_cats = len(cat_activations[model])
        compactness_ratios = []

        for i in range(n_cats):
            current_cat = cat_activations[model][i]

            # Compute intra-category distances
            if len(current_cat) > 1:
                # Use sklearn's optimized euclidean_distances with squared option
                intra_dist_matrix = euclidean_distances(current_cat, current_cat, squared=squared)
                # Get upper triangle (excluding diagonal) to avoid counting pairs twice
                mask = np.triu(np.ones_like(intra_dist_matrix, dtype=bool), k=1)
                intra_distances = intra_dist_matrix[mask]
                mean_intra_distance = np.mean(intra_distances)

            # Compute inter-category distances in batches to save memory
            inter_distances_sum = 0.0
            inter_count = 0

            for j in range(n_cats):
                if i != j:
                    other_cat = cat_activations[model][j]
                    # Use sklearn's optimized euclidean_distances with squared option
                    distances = euclidean_distances(current_cat, other_cat, squared=squared)

                    inter_distances_sum += np.sum(distances)
                    inter_count += distances.size

            mean_inter_distance = inter_distances_sum / inter_count if inter_count > 0 else 1.0

            # Compactness ratio: lower is better (tight within, far between)
            compactness_ratio = mean_intra_distance / mean_inter_distance
            compactness_ratios.append(compactness_ratio)

        compactness_ratios = np.array(compactness_ratios)

        # Sort by compactness (ascending - lower is better)
        sort_indices = np.argsort(compactness_ratios)
        compactness2[model] = compactness_ratios[sort_indices]
        compact_categories2[model] = np.array(listcat)[sort_indices]

    return compactness2, compact_categories2

def compute_compactness_memory_efficient(cat_activations, models, listcat):
    """
    Memory-efficient version that processes one model at a time and uses generators.
    Good for very large datasets.
    """
    compactness2 = {}
    compact_categories2 = {}

    for model in models:
        print(model)
        n_cats = len(cat_activations[model])

        # Pre-compute centroids
        centroids = np.array([np.mean(cat_act, axis=0) for cat_act in cat_activations[model]])

        # Compute metrics using generator expressions to save memory
        intra_vars = np.array([
            np.mean((cat_activations[model][i] - centroids[i])**2)
            for i in range(n_cats)
        ])

        inter_vars = np.array([
            np.mean([
                np.mean((cat_activations[model][j] - centroids[i])**2)
                for j in range(n_cats) if j != i
            ])
            for i in range(n_cats)
        ])

        # Compute normalized variances
        normalized_vars = intra_vars / inter_vars

        # Sort and store results
        sort_indices = np.argsort(normalized_vars)
        compactness2[model] = normalized_vars[sort_indices]
        compact_categories2[model] = np.array(listcat)[sort_indices]

    return compactness2, compact_categories2

def max_compactness_difference(compact_categories, compactness, cat_activations, listcat, models = ['saycam', 'ego'], nb_max_compactness = 10):
    '''
    Function that sorts categories following the maximum difference in compactness given 2 models
    '''
    nb_categories = cat_activations[models[0]].shape[0]
    ori_cat = np.arange(0,nb_categories)
    comp_cat = np.zeros(nb_categories)
    for c, cat in enumerate(compact_categories[models[0]]):
        comp_cat[c] = compact_categories[models[1]].tolist().index(cat) # rank for model2 the categories sorted for model1

    #diff = np.absolute(comp_cat - ori_cat) # distance in terms of rank
    diff = np.absolute(compactness[models[1]][comp_cat.astype(int)] - compactness[models[0]][ori_cat.astype(int)]) # distance in terms of compactness
    sortedmaxdiffcats = np.array(listcat)[np.argsort(-diff)]
    maxdiffs = np.sort(diff)[::-1]

    labels = np.argsort(-diff)
    max_compactness_cats = sortedmaxdiffcats[:nb_max_compactness]
    print(f'The {nb_max_compactness} categories leading to the max differences between {models[0]} and {models[1]} are {sortedmaxdiffcats[:nb_max_compactness]}')
    print(f'Category numbers are {labels[:nb_max_compactness]}')
    print(f'With differences in compactness of  {maxdiffs[:nb_max_compactness]}')

    return labels, sortedmaxdiffcats, maxdiffs

def find_max_dissimilarity_images(cat_activations, models, categories,
                                  compute_RDM, compute_similarity,
                                  images_per_subset=4, method='exhaustive'):
    """
    Find the subset of images per category that maximizes RDM dissimilarity between two models.

    Parameters:
    -----------
    cat_activations : dict
        Dictionary with structure: cat_activations[model][category] = array of activations (n_images, n_features)
    models : list
        List of two model names, e.g., ['model1', 'model2']
    categories : list
        List of category names/indices
    compute_RDM : function
        Function that takes activations and returns RDM: RDM = compute_RDM(activations)
    compute_similarity : function
        Function that takes two RDMs and returns similarity: sim = compute_similarity(RDM1, RDM2)
    images_per_subset : int
        Number of images to select per category (default: 4)
    method : str
        'exhaustive' or 'random' sampling of combinations

    Returns:
    --------
    results : dict
        Dictionary with results for each category:
        {
            category: {
                'best_indices': array of selected image indices,
                'max_dissimilarity': maximum dissimilarity value,
                'model1_rdm': RDM for model1 with selected images,
                'model2_rdm': RDM for model2 with selected images,
                'similarity': similarity between the two RDMs
            }
        }
    """

    if len(models) != 2:
        raise ValueError("This function requires exactly 2 models")

    results = {}

    for category in tqdm(categories, desc="Processing categories"):
        print(f"\nProcessing category: {category}")

        # Get activations for both models for this category
        model1_activations = cat_activations[models[0]][category]  # Shape: (50, n_features)
        model2_activations = cat_activations[models[1]][category]  # Shape: (50, n_features)

        n_images = model1_activations.shape[0]

        if n_images < images_per_subset:
            raise ValueError(f"Category {category} has only {n_images} images, need at least {images_per_subset}")

        # Generate combinations of image indices
        if method == 'exhaustive':
            # All possible combinations (can be large!)
            all_combinations = list(combinations(range(n_images), images_per_subset))
        elif method == 'random':
            # Random sample of combinations (for large datasets)
            n_combinations = min(1000, len(list(combinations(range(n_images), images_per_subset))))
            all_combinations = []
            while len(all_combinations) < n_combinations:
                indices = np.random.choice(n_images, images_per_subset, replace=False)
                indices_tuple = tuple(sorted(indices))
                if indices_tuple not in all_combinations:
                    all_combinations.append(indices_tuple)
        else:
            raise ValueError("method must be 'exhaustive' or 'random'")

        print(f"Testing {len(all_combinations)} combinations of {images_per_subset} images")

        max_dissimilarity = -np.inf
        best_indices = None
        best_model1_rdm = None
        best_model2_rdm = None
        best_similarity = None

        # Test each combination
        for combination in tqdm(all_combinations, desc="Testing combinations", leave=False):
            indices = np.array(combination)

            # Get subset of activations
            subset_model1 = model1_activations[indices]  # Shape: (4, n_features)
            subset_model2 = model2_activations[indices]  # Shape: (4, n_features)

            # Compute RDMs for this subset
            rdm1 = compute_RDM(subset_model1)  # Shape: (4, 4)
            rdm2 = compute_RDM(subset_model2)  # Shape: (4, 4)

            # Compute similarity between RDMs
            similarity = compute_similarity(rdm1, rdm2)

            # We want to maximize dissimilarity, so minimize similarity
            dissimilarity = -similarity  # or 1 - similarity, depending on your similarity metric

            # Update best if this is better
            if dissimilarity > max_dissimilarity:
                max_dissimilarity = dissimilarity
                best_indices = indices
                best_model1_rdm = rdm1
                best_model2_rdm = rdm2
                best_similarity = similarity

        # Store results for this category
        results[category] = {
            'best_indices': best_indices,
            'max_dissimilarity': max_dissimilarity,
            'model1_rdm': best_model1_rdm,
            'model2_rdm': best_model2_rdm,
            'similarity': best_similarity
        }

        print(f"Best indices for {category}: {best_indices}")
        print(f"Max dissimilarity: {max_dissimilarity:.4f}")
        print(f"Similarity: {best_similarity:.4f}")

    return results


def analyze_selected_images(results, categories):
    """
    Analyze the results to understand patterns in selected images.
    """
    print("\n=== Analysis of Selected Images ===")

    for category in categories:
        result = results[int(category)]
        indices = result['best_indices']
        dissimilarity = result['max_dissimilarity']
        similarity = result['similarity']

        print(f"\nCategory: {int(category)}")
        print(f"Selected images: {indices}")
        print(f"Dissimilarity: {dissimilarity:.4f}")
        print(f"Similarity: {similarity:.4f}")

        # Could add more analysis here:
        # - Distribution of selected indices
        # - Patterns across categories
        # - Statistics on dissimilarity values

import math
def plot_stats(SIMs, submodels, labels = ['label1', 'label2']):
    '''plot the compactness as a function of sorted image category.
    Plot is a subplot of adaptative size, depending on the length of the list submodel given.
    '''
    nb_subs = len(submodels) # Number of subs
    sqrt = np.sqrt(nb_subs)
    cols = math.ceil(sqrt)
    rows = math.ceil(nb_subs // sqrt)
    while (cols*rows)<(nb_subs):
        rows =rows + 1 ## compute the number of columns and rows
    fig, subs = plt.subplots(rows,cols, sharex=True, sharey=True, figsize=(cols*2+1, rows*2+1)) # adaptative size
    count = 0
    minval = 1
    maxval = 0
    for i, model in enumerate(submodels):
        minval = min(minval, np.amin(SIMs[model]))
        maxval = max(maxval, np.amax(SIMs[model]))
        if rows ==1:
            subs[count%cols].plot(SIMs[model])
            subs[count%cols].set_title(f'{model}')
        else:
            subs[count//cols, count%cols].plot(SIMs[model])
            subs[count//cols, count%cols].set_title(f'{model}')
            count+=1
    maxval = min(maxval, 1.1)
    plt.ylim(np.round(minval,1)-0.1, np.round(maxval,1)+0.1)
    if rows == 1:
        subs[0].set_ylabel(labels[1])
        for sub in subs:
            sub.set_xlabel(labels[0])
    else:
        for sub in subs[-1]:
            sub.set_xlabel(labels[0])
        for sub in subs[:,0]:
            sub.set_ylabel(labels[1])

    fig.tight_layout()
    plt.show()
    plt.close()