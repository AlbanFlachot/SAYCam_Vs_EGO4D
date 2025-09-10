import importlib
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from os.path import join
import os
import seaborn as sns
from tqdm import tqdm
import argparse

#### Custum libraries
import lib.algos_maxRSA as max_rsa
import lib.utils_RSA as rsa
import lib.utils_CKA as cka

from lib.algos import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'ecoVal', help='path to real images')
parser.add_argument('--models', nargs='+', default = ['ego', 'saycam'], help='List of models to compare, should at least be two')
parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--compactness_measure', default = 'Fisher_discriminant', help = "Measure for compactness, can be 'Fisher_discriminant', 'silhouette_score'")
parser.add_argument('--compactness_diff_measure', default = 'rank', help = "Measure for difference in compactness")
parser.add_argument('--nb_considered_categories', type = int, default = 12, help = 'Number of categories to consider for the subsets of images')
parser.add_argument('--dissimilarity_measure', type = str, default = 'L2squared', help = 'Measure to compute the RDMs')
parser.add_argument('--similarity_measure', type = str, default = 'pearson', help = 'Measure to find most dissimilar subset')
opt = parser.parse_args()

dataset = opt.dataset
models  = opt.models
print(models)
path2activations = f'/data/alban/activations_datadriven/%s_{dataset}/'

imagelists = {}
activations = {}
for model in models:
    with open(join(path2activations%model, 'imagepaths.txt'), 'r') as f:
        imagelists[model] = [line.strip() for line in f.readlines()]
    activations[model] = np.load(join(path2activations % model, 'cls_tokens.npy'))

activations[model].shape

#### Normalize vectors
if opt.normalize:
    for model in models:
        norms = np.linalg.norm(activations[model], axis=1, keepdims=True)
        activations[model] = activations[model]/norms # normalization

### check if images were shown in the same order
assert imagelists[models[0]] == imagelists[models[1]]
imagelist = imagelists[models[0]] # since they are the same, only consider one list

#### check if each category has the same number of images and list all categories in listcats
count = 0
cat = ''
listcat = list()
for i, imgp in enumerate(imagelist):
    current_cat = imgp.split('/')[-2]
    if i == 0:
        cat = current_cat
        listcat.append(current_cat)
    if cat != current_cat:
        cat = current_cat
        listcat.append(current_cat)
        count = 1
    else:
        count += 1

nb_per_cat = count # in val, 50 images per category

### reshape activations according to include categories
cat_activations = activations.copy()

for model in models:
    shape = activations[model].shape
    cat_activations[model] = activations[model].reshape(-1, nb_per_cat, shape[-1])

### Compute representational compactness for each category and model
compactness, compact_categories = max_rsa.compute_compactness(cat_activations, models, listcat, measure = opt.compactness_measure)

max_rsa.plot_stats_one(compactness,models,  ['Categories', 'Normalized var'], savename=f'compactness/{opt.normalize}normalize_{opt.compactness_measure}.png')

for model in models:
    print(f'\n Overlap {model} and the others')
    for model2 in models[:]:
        print(max_rsa.check_list_similarity(compact_categories[model][:50],compact_categories[model2][:50]))

### Compute 500 random sequence for significance
# initialize sequences
nb_trials = 500
idx_vec = np.array(range(len(listcat)))
mat_vec = np.zeros((nb_trials, len(idx_vec)))
for i in range(nb_trials):
    np.random.shuffle(idx_vec)
    mat_vec[i] = idx_vec

# Compute all possible similarity pairs
list_sim = []
for i in range(len(mat_vec) - 1):
    for j in range(i + 1, len(mat_vec)):
        list_sim.append(max_rsa.check_list_similarity(list(mat_vec[i][:50]), list(mat_vec[j][:50])))

# Compute 95 percentile
confinter = np.percentile(list_sim, 95)
print(f'The 95% confidence interval is:{confinter}')

#####---------------------------------------------------------------------------------------------------------------#####
#### START OF ALGO PROPER
#####---------------------------------------------------------------------------------------------------------------#####
nb_considered_categories = opt.nb_considered_categories
nb_categories = len(listcat)
labels = {} # Dictionary of labels sorted for each pairwise comparisons that shows the maximum compactness difference
sortedmaxdiffcats = {} # Dictionary of category indices sorted like for labels
maxdiffs = {} # Same but for differences (based on metric used)
max_dissimilarity_images = {} # Same but for the subset of images obtained with algorithm
similarity_dict = {}
imagespaths = {}

for i, model1 in enumerate(models[:-1]):
    for j, model2 in enumerate(models[i+1:]):
        savename = f'{opt.dataset}_{opt.normalize}normalize_{opt.compactness_measure}_{opt.compactness_diff_measure}_{opt.similarity_measure}_{model1}_{model2}'
        labels[model1 + '_' + model2], sortedmaxdiffcats[model1 + '_' + model2], maxdiffs[model1 + '_' + model2] = max_rsa.max_compactness_difference(
                compact_categories, compactness, nb_categories, listcat, models = [model1, model2],
                nb_considered_categories = nb_considered_categories, compactness_diff_measure = opt.compactness_diff_measure
            )

        max_dissimilarity_images[model1 + '_' + model2] = max_rsa.find_max_dissimilarity_images(
                cat_activations, [model1, model2], labels[model1 + '_' + model2][:nb_considered_categories], nb_per_cat,
                images_per_subset=4, similarity_metric=opt.similarity_measure, diff = maxdiffs[model1 + '_' + model2]
            )

        similarity_dict[model1 + '_' + model2] = max_rsa.compute_sub_rdm_similarity(
            max_dissimilarity_images[model1 + '_' + model2], cat_activations, [model1, model2], labels[model1 + '_' + model2][:nb_considered_categories],
            savename = f'figures/compactness/sub_RDMs/{savename}.png')

        #imagelist = [img.replace('/raid/shared/datasets/visoin/', '/home/alban/Documents/') for img in imagelist]
        images, imagespaths[model1 + '_' + model2] = max_rsa.display_low_similarity_images(imagelist, similarity_dict[model1 + '_' + model2]['selected_indices'], n_images=40,
                                                      grid_cols=8, figsize=(20, 10),
                                                      save_path=f'figures/compactness/subset/{savename}.png')

RESULTS = {}
RESULTS['max_dissimilarity_images'] = max_dissimilarity_images
RESULTS['similarity_dict'] = similarity_dict
RESULTS['imagepaths'] = imagespaths

import pickle
f = open(f"/data/alban/results_image_selection/{opt.dataset}_{opt.normalize}normalize_{opt.compactness_measure}_{opt.compactness_diff_measure}_{opt.similarity_measure}.pkl","wb")
pickle.dump(RESULTS,f)
f.close()