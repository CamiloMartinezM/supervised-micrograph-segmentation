# -*- coding: utf-8 -*-
"""
Created on Fri May 22:13:00 2022

@author: Camilo Martínez
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import os

from estimator import TextonEstimator
from utils.functions import (
    flatten_and_append_shape,
    load_from_labels_dictionary,
    load_variable_from_file,
    randomize_tuple,
    load_img,
    visualize_segmentation
)

from datasets.load_datasets import MetalDAM_dataset, LABEL_MAPPING

# Validate with sklearn
# check_estimator(TextonEstimator())

# %% PREPARE TRAINING DATA
# Load the training windows. Keys are labels/classes, values are lists of 2d numpy arrays
# which are the labeled windows
images, labels_imgs, training_dict = MetalDAM_dataset()

# %%
# Extract the windows and labels in order
windows, labels, label_mapping = load_from_labels_dictionary(training_dict)

# Shuffle both windows and labels
X_train, y_train = randomize_tuple(
    windows, labels, to_numpy=True, dtypes={0: object, 1: int}, random_state=1
)

# X_train is an object array which has 2d numpy arrays in each position. Those 2d arrays
# need to be flattened and their original shape appended to the flattened array (to not
# lose that information)
(X_train,) = flatten_and_append_shape(X_train)

# %% TRAIN THE ESTIMATOR
import time

tic = time.time()
model = TextonEstimator(K=6)  # Define the estimator using 6 clusters

# Fit the data
model.fit(X_train, y_train)

# %% EXAMPLE
# img = load_img(os.path.join("saved_images", "upscaled_test_image.jpg"), with_io=True)
img = images[2]
original_shape = img.shape
img = np.concatenate([img.ravel(), original_shape])
original_shape = img[-2::].astype(int)

tic1 = time.time()
segmented_img = model.predict([img])
toc1 = time.time()
print(toc1 - tic1)

plt.figure()
plt.imshow(img[:-2].reshape(original_shape), cmap="gray")
plt.imshow(segmented_img.reshape(original_shape), alpha=0.5)
plt.show()
plt.close()
toc = time.time()
print(toc - tic)

# %%
plt.figure()
plt.imshow(img[:-2].reshape(original_shape), cmap="gray")
plt.imshow(labels_imgs[2], alpha=0.5)
plt.show()
plt.close()
toc = time.time()
print(toc - tic)

# %%
visualize_segmentation(
    img[:-2].reshape(original_shape),
    np.array(list(LABEL_MAPPING.values())),
    segmented_img.reshape(original_shape),
)

# %%
visualize_segmentation(
    img[:-2].reshape(original_shape),
    # np.array([LABEL_MAPPING[class_] for class_ in np.unique(labels_imgs[2])]),
    np.array(list(LABEL_MAPPING.values())),
    labels_imgs[2],
)
