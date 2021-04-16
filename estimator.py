# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:22:09 2021

@author: Camilo Martínez
"""
import os
import pickle
import random
import warnings
from itertools import chain

import cupy as cp
import cusignal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import scipy.fftpack
from numba import double, jit
from numba.core.errors import NumbaWarning
from scipy.signal import fftconvolve, oaconvolve
from skimage.filters import sobel
from skimage.segmentation import (felzenszwalb, mark_boundaries, quickshift,
                                  slic, watershed)
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import cross_val_score

from utils_functions import (highlight_class_in_img, img_to_binary,
                             load_variable_from_file, show_img)

warnings.simplefilter('ignore', category=NumbaWarning)

class CustomFFTConvolution(object):
    def __init__(self, A, B, domain, threads=8):
        MK =  B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        if domain =="same":
            self.Y = M
            self.X = N
        elif domain == "valid":
            self.Y = M - MK + 1
            self.X = N - NK + 1
        elif domain == "full":
            self.Y = M + MK - 1
            self.X = N + NK - 1

        self.M = M + MK - 1
        self.N = N + NK - 1

        a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        b = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

        self.fft_A_obj = pyfftw.builders.rfft2(a, s=(self.M, self.N), threads=threads)
        self.fft_B_obj = pyfftw.builders.rfft2(b, s=(self.M, self.N), threads=threads)
        self.ifft_obj = pyfftw.builders.irfft2(self.fft_A_obj.output_array, s=(self.M, self.N), threads=threads)


        self.offset_Y = int(np.floor((self.M - self.Y)/2))
        self.offset_X = int(np.floor((self.N - self.X)/2))

    def __call__(self, A, B):
        MK =  B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
        b = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

        return self.ifft_obj(self.fft_A_obj(a) * self.fft_B_obj(b))[self.offset_Y:self.offset_Y + self.Y, self.offset_X:self.offset_X + self.X]

class TextonSegmentation(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(
        self,
        K: int = 1,
        algorithm: str = "felzenszwalb",
        algorithm_parameters: tuple = (125, 0.8, 115),
        subsegment_class: tuple = None,
    ):
        """
        Called when initializing the classifier
        """
        self.K = K
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters
        self.subsegment_class = subsegment_class

        sigmas, n_orientations = [1, 2, 4], 6
        edge, bar, rot = self._makeRFSfilters(
            sigmas=sigmas, n_orientations=n_orientations
        )
        self.filterbank = list(chain(edge, bar, rot))

    def fit(self, X, y=None):
        """
        Args:
            X (list): [matrix()]
        """
        self.feature_vectors = self._feature_vectors_from_dict(self._to_dict(X, y))
        self.classes = np.array(list(self.feature_vectors.keys()))
        C = len(self.classes)

        textons = {}
        for label in self.feature_vectors:
            textons[label] = MiniBatchKMeans(n_clusters=self.K).fit(
                self.feature_vectors[label]
            )

        # Matrix of texture textons
        # Once the textons have been learned for each of the classes, it is possible to
        # construct a matrix T of shape (C, K, 8) where each of the rows is a class and
        # each column has the texton k for k < K. Note that said texton must have 8
        # dimensions, since the pixels were represented precisely by 8 dimensions.
        self.T = np.zeros((C, self.K, 8), dtype=np.float64)
        for i, label in enumerate(self.classes):
            self.T[i] = textons[label].cluster_centers_

        return self

    def predict(self, X, y=None):
        """Predicts the class/label given the feature vectors that describe an image or a
        window of an image (like a superpixel).
        Args:
            feature_vectors (np.ndarray): Feature vectors of the image or window.
            classes (np.ndarray): Array of classes/labels.
            T (np.ndarray): Texton matrix.
        Returns:
            str: Predicted class.
        """
        try:
            getattr(self, "T")
        except AttributeError:
            raise RuntimeError("Untrained classifier")

        return [self._segment(x) for x in X]

    def _segment(self, X: np.ndarray) -> tuple:
        original_shape = (int(X[-2]), int(X[-1]))
        X = X[:-2].reshape(original_shape)

        # The image is segmented using the given algorithm for superpixels generation
        segments = self._generate_superpixels(X)

        # Superpixels obtained from segments.
        # Every key in S is a superpixel and its values are the pixels belonging
        # to it.
        responses = self._get_response_vector(X)

        # The new segments are created, i.e, actual segmentation.
        S_segmented = {}
        class_matrix = np.zeros(X.shape, dtype=int)
        for superpixel in np.unique(segments):
            pixels = np.argwhere(segments == superpixel)
            i = pixels[:, 0]
            j = pixels[:, 1]

            feature_vectors = responses[i, j]
            predicted_class_idx = self._class_of(feature_vectors)
            S_segmented[superpixel] = self.classes[predicted_class_idx]
            class_matrix[i, j] = predicted_class_idx

        class_matrix = np.append(class_matrix.flatten(), [500, 500])
        return class_matrix

    def _class_of(self, feature_vector: np.ndarray) -> int:
        # Distance matrices.
        minimum_distance_vector, distance_matrix = self._get_closest_texton_vector(feature_vector)

        # Matrix which correlates texture texton distances and minimum distances of every
        # pixel.
        A = np.sum(
            np.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-16), axis=-1
        )

        A_i = A.sum(axis=1)  # Sum over rows (i.e, over all pixels).

        # Class with maximum probability of occurrence is chosen.
        return A_i.argmax(axis=0)

    def _get_closest_texton_vector(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Obtains a vector whose values are the minimum distances of each pixel of a
        superpixel. For example, if a superpixel has 300 pixels, this function returns a
        (300,) vector, where each value is the minimum distance of an enumerated pixel.
        Args:
            feature_vectors (np.ndarray): Output of get_feature_vectors_of_superpixel.
            T (np.ndarray): Texton matrix.
        Returns:
            np.ndarray: Minimum distance vector.
        """
        # Matrix of shape (C, NUM_PIXELS, K). Every (i, j, k) matrix value
        # corresponds to the distance from the i-th pixel to the k-th texton
        # of the j-th class. Obtains a matrix which has the information of all 
        # possible distances from a pixel of a superpixel to every texton of every class.
        distance_matrix = np.linalg.norm(
            feature_vectors[:, np.newaxis] - self.T[:, np.newaxis, :], axis=-1
        )
        minimum_distance_vector = np.min(distance_matrix[np.newaxis], axis=(-1, 1))
        return minimum_distance_vector, distance_matrix

    def _get_response_vector(self, x: np.ndarray) -> np.ndarray:
        """Convolves the input image with the MR8 Filter Bank to get its response as a
        numpy array.
        Args:
            img (np.ndarray): Input image as a numpy array.
        Returns:
            np.ndarray: Numpy array of shape (*img.shape, 8).
        """
        # 8 responses from image
        r = self._apply_filterbank(x)

        # Every response is stacked on top of the other in a single matrix whose last axis has
        # dimension 8. That means, there is now only one response image, in which each channel
        # contains the information of each of the 8 responses.
        response = np.concatenate(
            [np.expand_dims(r[i], axis=-1) for i in range(len(r))], axis=-1
        )
        return response

    def _generate_superpixels(self, image: np.ndarray) -> np.ndarray:
        """Segments the input image in superpixels.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Array of superpixels with 0 <= values < n_segments.
        """
        if self.algorithm == "slic":
            n_segments, sigma, compactness = self.algorithm_parameters
            s = slic(
                image,
                n_segments=n_segments,
                sigma=sigma,
                compactness=compactness,
                start_label=1,
            )
        elif self.algorithm == "felzenszwalb":
            scale, sigma, min_size = self.algorithm_parameters
            s = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
        elif self.algorithm == "quickshift":
            ratio, kernel_size, max_dist, sigma = self.algorithm_parameters
            s = quickshift(
                image,
                ratio=ratio,
                kernel_size=kernel_size,
                max_dist=max_dist,
                sigma=sigma,
                convert2lab=False,
            )
        elif self.algorithm == "watershed":
            gradient = sobel(image)
            markers, compactness = self.algorithm_parameters
            s = watershed(gradient, markers=markers, compactness=compactness)
        else:
            s = None

        return s

    def _apply_filterbank(self, img: np.ndarray) -> np.ndarray:
        img = cp.array(img.astype(np.float64))
        result = np.zeros((8, *img.shape))
        for i, battery in enumerate(self.filterbank):
            # response1 = [
            #     fftconvolve(img.get(), np.flip(filt.get()), mode="same") for filt in battery
            # ]
            response = [
                cusignal.fftconvolve(img, np.flip(filt), mode="same").get() for filt in battery
            ]
            result[i] = np.max(response, axis=0)

        return result

    @staticmethod
    def _makeRFSfilters(
        radius: int = 24, sigmas: list = [1, 2, 4], n_orientations: int = 6
    ):
        """Generates filters for RFS filterbank.

        Args:
            radius : int, default 28
                radius of all filters. Size will be 2 * radius + 1
            sigmas : list of floats, default [1, 2, 4]
                define scales on which the filters will be computed
            n_orientations : int
                number of fractions the half-angle will be divided in
        Returns:
            edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
                Contains edge filters on different scales and orientations
            bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
                Contains bar filters on different scales and orientations
            rot : ndarray (2, 2*radius+1, 2*radius+1)
                contains two rotation invariant filters, Gaussian and Laplacian of
                Gaussian
        """

        def make_gaussian_filter(x, sigma, order=0):
            if order > 2:
                raise ValueError("Only orders up to 2 are supported")
            # compute unnormalized Gaussian response
            response = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
            if order == 1:
                response = -response * x
            elif order == 2:
                response = response * (x ** 2 - sigma ** 2)
            # normalize
            response /= np.linalg.norm(response, 1)
            return response

        def makefilter(scale, phasey, pts, sup):
            gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
            gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
            f = (gx * gy).reshape(sup, sup)
            # normalize
            f /= np.linalg.norm(f, 1)
            return f

        support = 2 * radius + 1
        x, y = np.mgrid[-radius : radius + 1, radius : -radius - 1 : -1]
        orgpts = np.vstack([x.ravel(), y.ravel()])

        rot, edge, bar = [], [], []
        for sigma in sigmas:
            for orient in range(n_orientations):
                # Not 2pi as filters have symmetry
                angle = np.pi * orient / n_orientations
                c, s = np.cos(angle), np.sin(angle)
                rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
                edge.append(makefilter(sigma, 1, rotpts, support))
                bar.append(makefilter(sigma, 2, rotpts, support))

        length = np.sqrt(x ** 2 + y ** 2)
        rot.append(make_gaussian_filter(length, sigma=10))
        rot.append(make_gaussian_filter(length, sigma=10, order=2))

        # reshape rot and edge
        edge = cp.asarray(edge)
        edge = edge.reshape(len(sigmas), n_orientations, support, support).astype(np.float64)
        bar = cp.asarray(bar).reshape(edge.shape).astype(np.float64)
        rot = cp.asarray(rot)[:, np.newaxis, :, :].astype(np.float64)
        return edge, bar, rot

    @staticmethod
    def _to_dict(X, y) -> dict:
        result = {}
        labels = np.unique(y)
        for label in labels:
            result[label] = X[y == label]

        return result

    def _concatenate_responses(self, responses: np.ndarray) -> np.ndarray:
        """Helper function to obtain the complete feature vector of a label by concatenating
        all responses of images with the same label, so that a single matrix is obtained in
        which a row corresponds to a single pixel and each pixel possesses 8 dimensions,
        because of the MR8 Filter Bank.
        Args:
            responses (np.ndarray): Numpy array of responses.
        Returns:
            np.ndarray: Numpy array of all responses, where a row corresponds to a single
                        pixel feature vector.
        """
        return np.concatenate(
            [
                response[:, i]
                for response in responses
                for i in range(response.shape[1])
                if np.nan not in response[:, i]
            ]
        )

    def _feature_vectors_from_dict(self, d: dict) -> dict:
        """Each pixel of each annotated window has 8 responses associated with the filters
        used. These responses must be unified in some way, since they are part of the same
        class. Therefore, the following implementation transforms each of the responses
        obtained per window into a matrix where each row is a pixel of an annotation. And,
        since each of the annotations has 8 associated responses, each pixel is represented
        by an 8-dimensional vector. This means that each row will have 8 columns,
        corresponding to the value obtained from the filter. On the other hand, since there
        are several classes, said matrix will be stored in a dictionary, whose keys will be
        the classes found.
        Args:
            windows (dict): Dictionary of windows per label.
            verbose (bool): True is additional information is needed. Defaults to True.
        Returns:
            dict: Dictionary of feature vectors per label. Keys corresponds to labels and
                  values are the feature vectors of that label.
        """
        feature_vectors = {}
        for label in d:
            responses = []

            for i, x in enumerate(d[label]):
                # Every annotated window/every window of a label
                responses.append(self._get_response_vector(x))

            responses_arr = np.array(responses, dtype=object)

            # Every pixel of every single labeled window has 8 responses, which come from 8
            # response images. The following operations convert responses_arr to a matrix
            # where each row is a pixel. That means, each row will have 8 columns associated
            # with each pixel responses.
            feature_vectors[label] = self._concatenate_responses(responses_arr)

        return feature_vectors


def _load_variable(filename: str):
    with open(filename, "rb") as f:
        variable = pickle.load(f)
    return variable


def _load_training_set(filename: str = "training_windows.pickle") -> tuple:
    all_windows_per_label = _load_variable(filename)
    X = []
    y = []
    label_mapping = {}
    for label, windows in all_windows_per_label.items():
        if label not in label_mapping:
            existing_labels = list(label_mapping.values())
            if existing_labels:
                max_key = max(existing_labels)
            else:
                max_key = -1
            label_mapping[label] = max_key + 1

        for window in windows:
            X.append(window)
            y.append(label_mapping[label])

    return X, y, label_mapping


windows, labels, label_mapping = _load_training_set()
c = list(zip(windows, labels))
random.shuffle(c)
windows, labels = zip(*c)

X = np.array(windows, dtype="object")
y = np.array(labels, dtype=int)

# %%
model = TextonSegmentation(K=6)
model.fit(X, y)

# %%
gt = load_variable_from_file("ground_truth", os.getcwd())[:, 0, :, :]
imgs = load_variable_from_file("ground_truth", os.getcwd())[:, 1, :, :] / 255.0

n_samples = imgs.shape[0]
n_pixels = imgs.shape[1] * imgs.shape[2]

imgs = imgs.reshape(n_samples, n_pixels)
gt = gt.reshape(n_samples, n_pixels)

X = np.append(imgs, [[500, 500] for _ in range(51)], axis=1)
y = np.append(gt, [[500, 500] for _ in range(51)], axis=1)

# cross_val_score(TextonSegmentation(), X, y, scoring = 'neg_mean_squared_error')

# %%
img = X[20]

%timeit y = model.predict([img])

# plt.figure()
# plt.imshow(img[:-2].reshape((500, 500)), cmap="gray")
# plt.imshow(y[0][:-2].reshape((500, 500)), alpha=0.5)
# plt.show()
# plt.close()

# %%