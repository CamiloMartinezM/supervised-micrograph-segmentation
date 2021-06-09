# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:31:52 2021

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
    load_img
)

# Validate with sklearn
# check_estimator(TextonEstimator())

# %% PREPARE TRAINING DATA
# Load the training windows. Keys are labels/classes, values are lists of 2d numpy arrays
# which are the labeled windows
training_dict = load_variable_from_file("training_windows.pickle", "saved_variables")

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

# %% PREPARE TESTING DATA
testing_dict = load_variable_from_file("testing_windows.pickle", "saved_variables")

# Extract the windows and labels in order
X_test, y_test, _ = load_from_labels_dictionary(testing_dict, to_numpy=True)

# Flatten the arrays
(X_test,) = flatten_and_append_shape(X_test)

# Load ground-truth images and original images (where the labeled windows came from).
# In this case, these arrays are of shape (51, 250002), because 51 images of shape (500, 500)
# were labeled
gt = load_variable_from_file("ground_truth_with_originals", "saved_variables")[:, 0]
imgs = (
    load_variable_from_file("ground_truth_with_originals", "saved_variables")[:, 1]
    / 255.0
)

# Flatten the arrays accordingly
imgs, gt = flatten_and_append_shape(imgs, gt)

# %% TRAIN THE ESTIMATOR
import time
tic = time.time()
model = TextonEstimator(K=6)  # Define the estimator using 6 clusters

# Fit the data
model.fit(X_train, y_train)

# %%
# model.visualize_clusters(sample=5000)

# %% VALIDATE CLASSIFICATION PERFORMANCE
# y_pred_train = model.predict_windows(X_train)
# y_pred_test = model.predict_windows(X_test)

# target_names = list(label_mapping.keys())
# print("On training:\n")
# print(classification_report(y_train, y_pred_train, target_names=target_names))

# print("On testing:\n")
# print(classification_report(y_test, y_pred_test, target_names=target_names))

# %%
# TODO: VALIDATE SEGMENTATION PERFORMANCE


# %% EXAMPLE
# img = load_img(os.path.join("saved_images", "upscaled_test_image.jpg"), with_io=True)
img = imgs[50]
# original_shape = img.shape
# img = np.concatenate([img.ravel(), original_shape])
original_shape = img[-2::].astype(int)

tic1 = time.time()
segmented_img = model.predict([img])
toc1 = time.time()
print(toc1 - tic1)

# %%
plt.figure()
plt.imshow(img[:-2].reshape(original_shape), cmap="gray")
plt.imshow(segmented_img.reshape(original_shape), alpha=0.5)
plt.show()
plt.close()
toc = time.time()
print(toc - tic)