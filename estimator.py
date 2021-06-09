# -*- coding: utf-8 -*-
import importlib
import itertools
from itertools import chain
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from matplotlib.patches import Patch
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from utils.functions import highlight_class_in_img, img_to_binary, random_colors

# Import KMeans from cuml if it exists, otherwise go for sklearn's MiniBatchKMeans
cuml_spec = importlib.util.find_spec("cuml")
if cuml_spec:
    from cuml import KMeans
else:
    from sklearn.cluster import MiniBatchKMeans as KMeans

# Import tsnecuda it it exists, otherwise check for fitsne, openTSNE, MultiCoreTSNE in that order.
# If none of those exist, go for sklearn's implementation
tsnecuda_spec = importlib.util.find_spec(
    "tsnecuda"
)  # https://github.com/CannyLab/tsne-cuda
fitsne_spec = importlib.util.find_spec("fitsne")  # https://github.com/KlugerLab/FIt-SNE
opentsne_spec = importlib.util.find_spec(
    "openTSNE"
)  # https://github.com/pavlin-policar/openTSNE
multicoretsne_spec = importlib.util.find_spec(
    "MulticoreTSNE"
)  # https://github.com/DmitryUlyanov/Multicore-TSNE
if tsnecuda_spec:
    from tsnecuda import TSNE
elif fitsne_spec:
    from fitsne import fast_tsne as TSNE
elif opentsne_spec:
    from openTSNE import TSNE
elif multicoretsne_spec:
    from MulticoreTSNE import MulticoreTSNE as TSNE
else:
    from sklearn.manifold import TSNE

# Import cupy if it exists, otherwise use numpy as np and cp
# If cupy exists, import cusignal if it exists. Otherwise use scipy.signal's fftconvolve.
cupy_spec = importlib.util.find_spec("cupy")
cupy_spec = False
cusignal_spec = importlib.util.find_spec("cusignal")
if cupy_spec:
    import cupy as cp

    if cusignal_spec:
        from cusignal import fftconvolve
    else:
        from scipy.signal import fftconvolve
else:
    cp = np
    from scipy.signal import fftconvolve


class TextonEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        K: int = 1,
        algorithm: str = "felzenszwalb",
        algorithm_parameters: tuple = (125, 0.8, 115),
        subsegment_class: tuple = None,
        random_state: int = None,
    ):
        """
        Called when initializing the classifier
        """
        self.K = K
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters
        self.subsegment_class = subsegment_class
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Args:
            X (list): [matrix()]
        """
        if scipy.sparse.issparse(X):
            raise ValueError("sparse input is not supported")

        self.random_state_ = check_random_state(self.random_state)

        # Perform sklearn's validation if it is possible
        try:
            X, y = check_X_y(X, y)
        except ValueError as e:
            if str(e) != "setting an array element with a sequence.":
                raise e

        # Classifier can't train with only one feature
        if any(x.shape[0] <= 1 for x in X):
            raise ValueError("Classifier can't train with n_features = 1")

        # Extract labels/classes
        self.classes_, y = cp.unique(y, return_inverse=True)

        # Perform simple validations
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present")
        if y.shape != (X.shape[0],):
            raise ValueError("X and y have incorrect shapes")

        # Raise ValueError if there are negative float values
        if self.classes_.dtype.kind == "f":
            if any(self.classes_[i] < 0 for i in range(self.classes_.shape[0])):
                raise ValueError("Unknown label type: negative values")
        # If it's not float, accept only integer or string labels/classes
        elif self.classes_.dtype.kind not in "iOU":
            raise ValueError(f"Unknown label type: {self.classes_.dtype}")

        # Maximum number of features in X
        if len(X.shape) > 1:
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = np.max([x.shape[0] for x in X])

        # Build the filterbank
        sigmas, n_orientations = [1, 2, 4], 6
        edge, bar, rot = self._makeRFSfilters(
            sigmas=sigmas, n_orientations=n_orientations
        )
        self.filterbank_ = list(chain(edge, bar, rot))

        # Get the feature vectors for each class
        self.feature_vectors_ = self._feature_vectors_from(X, y)

        # Learn the textons
        # Once the textons have been learned for each of the classes, it is possible to
        # construct a matrix T of shape (C, K, 8) where each of the rows is a class and
        # each column has the texton k for k < K. Note that said texton must have 8
        # dimensions, since the pixels were represented precisely by 8 dimensions.
        self.T_ = cp.zeros((self.classes_.shape[0], self.K, 8), dtype=np.float64)
        self.T_labels_ = np.zeros((self.classes_.shape[0],), dtype=object)
        for i, label in enumerate(self.feature_vectors_):
            # TODO: Cambiar a random_state
            textons = KMeans(n_clusters=self.K, random_state=0).fit(
                self.feature_vectors_[label]
            )

            # Textons cluster centers
            self.T_[i] = cp.asarray(textons.cluster_centers_)

            # Cluster labels (used for visualization)
            self.T_labels_[i] = textons.labels_

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
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return np.array([self._segment(x) for x in X])

    def score(self, X, y, sample_weight=None):
        # Calculate predicted label for each x
        y_pred = self.predict_windows(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def predict_windows(self, X: np.ndarray):
        y_pred = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            # Reshape to original shape
            x_2d = x[:-2].reshape(x[-2::].astype(int))

            # Obtain feature vector
            feature_vector = self._get_response_vector(x_2d)
            feature_vector = feature_vector.reshape(-1, feature_vector.shape[-1])

            # Predict the class for that feature vector
            y_pred[i] = self._class_of(feature_vector)

        return y_pred

    def visualize_clusters(self, sample: Optional[int] = None) -> None:
        # Check is fit had been called
        check_is_fitted(self)

        # Instantiate a t-SNE model
        tsne_model = TSNE(perplexity=50, learning_rate=10, n_components=2, n_jobs=-1)

        # Extract a sample from the data
        X_sample = np.zeros((self.classes_.shape[0]), dtype=object)
        clusters = np.zeros((self.classes_.shape[0]), dtype=object)
        for i, label in enumerate(self.feature_vectors_):
            idx = np.random.randint(0, self.feature_vectors_[label].shape[0], sample)
            X_sample[i], clusters[i] = (
                self.feature_vectors_[label].get()[idx],
                self.T_labels_[i][idx],
            )

        # N distinct random colors
        colors = random_colors(self.classes_.shape[0] * self.K)

        k = 0
        fig, ax = plt.subplots()
        patches = []

        # Iterate over every sampled data per label
        for i, X in enumerate(X_sample):
            # Embed in a 2 dimensional space
            X_embedded = tsne_model.fit_transform(X)

            patches.append([])
            # Scatter plot of points in X_embedded that were assigned to a cluster, with different
            # colors for each cluster
            for cluster in range(self.K):
                cluster_data = X_embedded[clusters[i] == cluster]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[k])
                patches[i].append(
                    Patch(facecolor=colors[k], edgecolor="k")
                )  # used for legends
                k += 1

        handles = list(itertools.chain.from_iterable(zip(patches[0], patches[1])))
        labels = [""] * self.classes_.shape[0] * self.K
        for i, class_ in enumerate(self.classes_):
            labels[-(self.classes_.shape[0] - i)] = class_

        ax.legend(
            handles=handles,
            labels=labels,
            ncol=self.K,
            handletextpad=0.5,
            handlelength=1.0,
            columnspacing=-0.5,
            fontsize=16,
        )

    def _segment(self, X: np.ndarray) -> tuple:
        original_shape = X[-2::].astype(int)
        X_2d = X[:-2].reshape(original_shape)

        # The image is segmented using the given algorithm for superpixels generation
        segments = cp.asarray(self._generate_superpixels(X_2d))
        superpixels = cp.unique(segments)

        # Responses of every pixel in the input image
        responses = self._get_response_vector(X_2d)

        class_matrix = cp.zeros(X_2d.shape, dtype=self.classes_.dtype)
        for superpixel in superpixels:
            pixels = cp.argwhere(segments == superpixel)
            i = pixels[:, 0]
            j = pixels[:, 1]
            feature_vectors = responses[i, j]
            class_matrix[i, j] = self._class_of(feature_vectors)

        if self.subsegment_class:
            # Check if the class to subsegment is present in the segmentation
            if self.subsegment_class in cp.unique(class_matrix):
                new_class = cp.max(self.classes_) + 1
                mapping = {0: self.subsegment_class, 1: new_class}  # 0: black, 1:

                subsegmented_X = highlight_class_in_img(
                    cp.asarray(img_to_binary(X_2d)),
                    class_matrix,
                    self.subsegment_class,
                    fill_value=-1,
                )

                for binary_value, corresponding_class in mapping.items():
                    class_matrix[subsegmented_X == binary_value] = corresponding_class

                cp.append(self.classes_, new_class)

        class_matrix = class_matrix.ravel()
        if cupy_spec:
            class_matrix = class_matrix.get()
        return class_matrix

    def _class_of(self, feature_vector: np.ndarray) -> int:
        # Distance matrices.
        minimum_distance_vector, distance_matrix = self._get_closest_texton_vector(
            feature_vector
        )

        # Matrix which correlates texture texton distances and minimum distances of every
        # pixel. Sum over axis=2 is the sum over rows, i.e, all pixels.
        A = cp.sum(
            cp.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-16),
            axis=(-1, 1),
        )

        # Class with maximum probability of occurrence is chosen.
        return self.classes_[cp.argmax(A, axis=0)]

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
        distance_matrix = cp.linalg.norm(
            feature_vectors[:, np.newaxis] - self.T_[:, np.newaxis, :], axis=-1
        )
        minimum_distance_vector = cp.min(distance_matrix[np.newaxis], axis=(-1, 1))
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
        return self._apply_filterbank(x)

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
        if cusignal_spec:
            img = cp.asarray(img).astype(np.float64)

        result = cp.zeros((*img.shape, 8))
        # Every response is stacked on top of the other in a single matrix whose last axis has
        # dimension 8. That means, there is now only one response image, in which each channel
        # contains the information of each of the 8 responses.
        for i, battery in enumerate(self.filterbank_):
            response = []
            for filt in battery:
                if not cusignal_spec and cupy_spec:
                    filt = filt.get()
                
                response.append(fftconvolve(img, filt, mode="same"))
            result[:, :, i] = cp.max(cp.array(response), axis=0)

        return result

    def _feature_vectors_from(self, X: np.ndarray, y: np.ndarray) -> dict:
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
        for label in cp.unique(y):
            idx = y == label
            if cupy_spec:
                idx = idx.get()
            windows = X[idx]
            responses = []
            for x in windows:
                # x is reshaped to its original shape using the last two values
                x_2d = x[:-2].reshape(x[-2::].astype(int))

                # Every annotated window/every window of a label
                responses.append(self._get_response_vector(x_2d))

            # Every pixel of every single labeled window has 8 responses, which come from 8
            # response images. The following operations convert responses_arr to a matrix
            # where each row is a pixel. That means, each row will have 8 columns associated
            # with each pixel responses.
            concatenated_responses = self._concatenate_responses(responses)
            if cupy_spec and not cuml_spec:
                concatenated_responses = concatenated_responses.get()
                
            feature_vectors[label.item()] = concatenated_responses

        return feature_vectors

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

            x = cp.array(x)
            # compute unnormalized Gaussian response
            response = cp.exp(-(x ** 2) / (2.0 * sigma ** 2))
            if order == 1:
                response = -response * x
            elif order == 2:
                response = response * (x ** 2 - sigma ** 2)
            # normalize
            response /= cp.linalg.norm(response, 1)
            return response

        def makefilter(scale, phasey, pts, sup):
            gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
            gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
            f = (gx * gy).reshape(sup, sup)
            # normalize
            f /= cp.linalg.norm(f, 1)
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
        edge = cp.asarray(edge, dtype=np.float64)
        edge = edge.reshape(len(sigmas), n_orientations, support, support)
        bar = cp.asarray(bar, dtype=np.float64).reshape(edge.shape)
        rot = cp.asarray(rot, dtype=np.float64)[:, np.newaxis, :, :]
        
        # Flip for correlation instead of convolution
        edge = cp.flip(edge, (2, 3)) 
        bar = cp.flip(bar, (2, 3))
        rot = cp.flip(rot, (2, 3))
        return edge, bar, rot

    @staticmethod
    def _concatenate_responses(responses: List[np.ndarray]) -> np.ndarray:
        """Helper function to obtain the complete feature vector of a label by concatenating
        all responses of images with the same label, so that a single matrix is obtained in
        which a row corresponds to a single pixel and each pixel possesses 8 dimensions,
        because of the MR8 Filter Bank.

        Args:
            responses (list): List of numpy arrays, where a numpy array corresponds to the
                              filter response of a single window.

        Returns:
            np.ndarray: Numpy array of all responses, where a row corresponds to the feature
                        vector of a single pixel.
        """
        return cp.concatenate(
            [response[~cp.isnan(response).any(axis=-1), :] for response in responses]
        )
