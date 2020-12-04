# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:22 2020

@author: Camilo Martínez
"""
import warnings

from numba.core.errors import NumbaWarning

warnings.simplefilter("ignore", category=NumbaWarning)

from copy import deepcopy

from model import SegmentationModel
from utils_functions import load_img, find_path_of_img
# Source directory
src = ""
# src = "C:\\Users\\Camilo Martínez\\Google Drive"

# Train/dev/test split
TRAIN_SIZE = 0.7
DEV_SIZE = 0.2

model = SegmentationModel(src=src, as_255=False)

# %%

K_range = list(range(100, 130, 10))
scales_range = range(0, 4)
precomputed_feature_vectors_of_label = None
for n_clusters in K_range:
    for scales in scales_range:
        print(f"CURRENT ITERATION WITH K={n_clusters}, ME={scales}")
        if scales != 0:
            current_precomputed = None
        else:
            current_precomputed = precomputed_feature_vectors_of_label

        model.train(
            K=n_clusters,
            train_size=TRAIN_SIZE,
            dev_size=DEV_SIZE,
            precomputed_feature_vectors=current_precomputed,
            multiscale_statistics_scales=scales,
            use_minibatch=True,
            minibatch_size=10000,
            verbose=False,
        )

        model.evaluate_classification_performance(
            train=False, dev=False, test=True, plot=False
        )
        model.evaluate_segmentation_performance(
            plot=False, small_test=True, max_test_number=10
        )

        if scales == 0:
            precomputed_feature_vectors_of_label = deepcopy(
                model.feature_vectors_of_label
            )

# %%
model.train(
    K=100,
    train_size=TRAIN_SIZE,
    dev_size=DEV_SIZE,
    precomputed_feature_vectors=None,
    multiscale_statistics_scales=0,
    use_minibatch=False,
    minibatch_size=10000,
    verbose=False,
)

# %%
precomputed_feature_vectors_of_label = deepcopy(
    model.feature_vectors_of_label
)

# %%
from utils_functions import find_path_of_img

original_img, superpixels, segmentation = model.segment(
    find_path_of_img("cs0327.png", model.PATH_LABELED),
    plot_original=True,
    plot_superpixels=True,
    n_segments=500,
    compactness=0.17,
)

model.visualize_segmentation(original_img, superpixels, segmentation)
