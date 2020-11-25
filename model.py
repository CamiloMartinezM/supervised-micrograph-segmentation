# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:59:47 2020

@author: Camilo Martínez
"""
import os
from collections import Counter
from random import randint, shuffle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cuml.cluster import KMeans
from numba import jit
from skimage.segmentation import mark_boundaries
from sklearn import metrics

from utils_classes import FilterBankMR8, MultiscaleStatistics, Scaler, SLICSegmentation
from utils_functions import (
    find_path_of_img,
    load_img,
    matrix_to_excel,
    np2cudf,
    plot_confusion_matrix,
    print_table_from_dict,
    train_dev_test_split,
    train_dev_test_split_table,
)


class SegmentationModel:
    def __init__(
        self,
        src: str,
        as_255: bool,
        labeled: str = "Anotadas",
        preprocessed: str = "Preprocesadas",
    ):
        """
        Args:
            LABELED (str, optional): Name of folder with labeled images. Defaults to "Anotadas".
            SRC (str): Path to source.
        """
        # The following constants represent directories of interest.
        # The folder where the annotated micrographs are and the one where all the
        # pre-processed micrographs are. The latter will have two images for each
        # of the micrographs: the first one represents the micrograph without the scale,
        # which generally appears in the lower section of the image, and the second one
        # is precisely the latter (which naturally has the information regarding the
        # scale).
        self.PATH_LABELED = os.path.join(src, labeled)
        self.PATH_PREPROCESSED = os.path.join(src, preprocessed)

        print(f"\nPath to labeled micrographs: {self.PATH_LABELED}")
        print(f"Path to preprocessed micrographs: {self.PATH_PREPROCESSED}")

        self.micrographs, self.index_to_name = self.load_imgs(as_255)
        self.micrographs_scales = self.load_scales()
        (
            self.labels,
            self.windows,
            self.windows_per_name,
        ) = self.extract_labeled_windows()

        # Feature vectors of labels which are calculated during training
        self.feature_vectors_of_label = None

    def load_scales(self) -> dict:
        """Loads the scale of the images by going to PREPROCESSED and finding the corresponding
        SCALE image of every image in LABELED.

        Returns:
            dict: Dictionary whose keys are names of images and values are their respective scale
                  pixel length.
        """
        print("\n[*] SCALES EXTRACTION:\n")
        myScaler = Scaler(self.PATH_LABELED, self.PATH_PREPROCESSED)
        myScaler.process()
        scales = myScaler.scales

        print_table_from_dict(
            scales,
            cols=["Name of micrograph", "Pixels in scale"],
            title="Pixel length scales",
        )

        return scales

    def load_imgs(self, as_255: bool) -> tuple:
        """Loads images in LABELED in a numpy array.

        Returns:
            tuple: Numpy array with all images, dictionary whose keys are names of images and values
                   are the corresponding indeces in the numpy array of images.
        """
        print("\n[*] IMAGES LOADING:\n")
        m = []
        index_to_name = {}  # Every name will have its corresponding position in m.
        count = 0
        for folder in os.listdir(self.PATH_LABELED):
            if os.path.isdir(os.path.join(self.PATH_LABELED, folder)):
                print(f"[?] Currently reading folder: {folder}")
                for f in os.listdir(os.path.join(self.PATH_LABELED, folder)):
                    if f.endswith(".png"):
                        print(f"\t Reading and loading {f}... ", end="")
                        img = load_img(
                            os.path.join(self.PATH_LABELED, folder, f), as_255
                        )
                        m.append(img)
                        index_to_name[f] = count
                        count += 1
                        print("Done")

        return np.array(m), index_to_name

    def get_array_of_micrograph(self, name: str) -> np.ndarray:
        """Gets the numpy array of an image with the given name.

        Args:
            name (str): Name of image.

        Returns:
            np.ndarray: Numpy array of image.
        """
        return self.micrographs[self.index_to_name[name]]

    def extract_labeled_windows(self) -> tuple:
        """Within LABELED, each of the images has a .txt file associated with it that contains the
        information of the position of each of its regions or windows that were annotated. This
        section is then in charge of extracting said regions by slicing the numpy array of an image
        accordingly to get each of its labeled windows.

        Returns:
            tuple: dictionary of label counts; dictionary of windows whose keys are the labels and
                   values are a list of numpy arrays, which are the windows associated with the
                   label; and dictionary of windows and respective labels per loaded image.
        """

        def check_label(label: str) -> str:
            translation_dictionary = {
                "perlita": "pearlite",
                "ferrita": "ferrite",
                "plate martensite": "plate martensite",
                "martensita": "lath martensite",
                "bainita superior": "superior bainite",
                "austenita": "austenite",
                "cementita": "cementite",
            }

            if label in translation_dictionary.keys():
                return translation_dictionary[label]
            elif label in translation_dictionary.values():
                return label
            else:  # Ignored labels
                if label == "grano" or label == "C":
                    return "cementite"
                else:
                    return None

        print("\n[*] LABELED WINDOWS EXTRACTION:\n")
        labels = {}  # Counts number of labels/annotations per label.
        windows_per_label = {}  # key = label, value = list of windows
        windows_per_name = {}  # key = filename, value = [(coords, window, label)]

        for folder in os.listdir(self.PATH_LABELED):
            if os.path.isdir(os.path.join(self.PATH_LABELED, folder)):
                print(f"[?] Currently reading folder: {folder}")
                for f in os.listdir(os.path.join(self.PATH_LABELED, folder)):
                    if f.endswith(".txt"):
                        img_name = f[:-4] + ".png"
                        print(f"\t Getting windows of {img_name}... ", end="")
                        full_img = self.get_array_of_micrograph(
                            img_name
                        )  # Loads full img from micrographs array

                        with open(
                            os.path.join(self.PATH_LABELED, folder, f), "r"
                        ) as annotations:
                            line = annotations.readline()
                            while len(line) > 0:
                                line_parts = line.split(" ")
                                label = line_parts[0]
                                for k in range(1, len(line_parts)):
                                    try:
                                        int(
                                            float(line_parts[k])
                                        )  # Added float because int('0.0') does not work
                                        break
                                    except:  # is string
                                        label += " " + line_parts[k]
                                offset = k - 1
                                label = check_label(label)
                                if label is not None:
                                    # print(f"\t   Found {label}. Splitting image... ", end="")
                                    labels[label] = labels.get(label, 0) + 1
                                    first_point = tuple(
                                        [
                                            int(x)
                                            for x in line_parts[4 + offset : 6 + offset]
                                        ]
                                    )
                                    second_point = tuple(
                                        [
                                            int(x)
                                            for x in line_parts[6 + offset : 8 + offset]
                                        ]
                                    )
                                    assert (first_point and second_point) != (0, 0)
                                    window = self.slice_by_corner_coords(
                                        full_img, first_point, second_point
                                    )

                                    if f not in windows_per_name:
                                        windows_per_name[img_name] = []
                                    windows_per_name[img_name].append(
                                        ((first_point, second_point), window, label)
                                    )

                                    if label not in windows_per_label:
                                        windows_per_label[label] = []
                                    windows_per_label[label].append(window)
                                # print("Done")
                                line = annotations.readline()

                        print("Done")

        print_table_from_dict(
            labels, cols=["Label", "Number"], title="Number of windows per label",
        )

        return labels, windows_per_label, windows_per_name

    @staticmethod
    def slice_by_corner_coords(
        img: np.ndarray, first_point: tuple, second_point: tuple
    ) -> np.ndarray:
        return img[first_point[1] : second_point[1], first_point[0] : second_point[0]]

    @staticmethod
    def get_response_vector(img: np.ndarray) -> np.ndarray:
        """Convolves the input image with the MR8 Filter Bank to get its response as a numpy array.

        Args:
            img (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Numpy array of shape (*img.shape, 8).
        """
        MR8 = FilterBankMR8([1, 2, 4], 6)  # MR8 Filter bank

        # 8 responses from image
        r = MR8.response(img)

        # Every response is stacked on top of the other in a single matrix whose last axis has
        # dimension 8. That means, there is now only one response image, in which each channel
        # contains the information of each of the 8 responses.
        response = np.concatenate(
            [np.expand_dims(r[i], axis=-1) for i in range(len(r))], axis=-1
        )
        assert response.shape == (*r[0].shape, 8)
        return response

    @staticmethod
    def concatenate_responses(responses: np.ndarray) -> np.ndarray:
        """Helper function to obtain the complete feature vector of a label by concatenating all
        responses of images with the same label, so that a single matrix is obtained in which a row
        corresponds to a single pixel and each pixel possesses 8 dimensions, because of the MR8
        Filter Bank.

        Args:
            responses (np.ndarray): Numpy array of responses.

        Returns:
            np.ndarray: Numpy array of all responses, where a row corresponds to a single pixel
                        feature vector.
        """
        return np.concatenate(
            [
                response[:, i]
                for response in responses
                for i in range(response.shape[1])
                if np.nan not in response[:, i]
            ]
        )

    def get_feature_vector_of_window(
        self, window: np.ndarray, ravel: bool = False
    ) -> tuple:
        """Obtains the feature vectors of an image or window.

        Args:
            window (np.ndarray): Image as a numpy array.
            ravel (bool, optional): Specifies whether to flatten the feature vectors of an image,
                                    so that each row is the feature vector of a single pixel. If
                                    this parameter is True, the output will be reshaped to the
                                    original image shape.
        Returns:
            tuple: Feature vector of the given window and the number of pixels whose feature vector
            was calculated.
        """
        response_1 = self.get_response_vector(window)
        response_2 = self.multiscale_statistics.process(window)
        num_pixels = window.size
        response_img = np.dstack((response_1, response_2))

        if ravel:
            return (
                response_img.reshape((window.size, response_img.shape[-1])),
                num_pixels,
            )
        else:
            return response_img, num_pixels

    def get_feature_vectors_of_labels(
        self, windows: dict, multiscale_statistics_scales: int, verbose: bool = True
    ) -> dict:
        """Each pixel of each annotated window has 8 responses associated with the filters used.
        These responses must be unified in some way, since they are part of the same class.
        Therefore, the following implementation transforms each of the responses obtained per window
        into a matrix where each row is a pixel of an annotation. And, since each of the annotations
        has 8 associated responses, each pixel is represented by an 8-dimensional vector. This means
        that each row will have 8 columns, corresponding to the value obtained from the filter. On
        the other hand, since there are several classes, said matrix will be stored in a dictionary,
        whose keys will be the classes found.

        Args:
            windows (dict): Dictionary of windows per label.
            multiscale_statistics_scales (int): Number of scales to consider when computing
                                                multiscales statistics.

        Returns:
            dict: Dictionary of feature vectors per label. Keys corresponds to labels and values are
                  the feature vectors of that label.
        """
        self.multiscale_statistics = MultiscaleStatistics(multiscale_statistics_scales)
        if (
            self.feature_vectors_of_label is None
            or list(self.feature_vectors_of_label.values())[0].shape[-1]
            != 8 + multiscale_statistics_scales * 3 * 4
        ):
            feature_vectors_of_label = {}
            for label in windows:
                responses = []
                print(f"[?] Working on label: {label}...")
                num_pixels = 0
                for i, window in enumerate(
                    windows[label]
                ):  # Every annotated window/every window of a label
                    if verbose:
                        print(f"\t  Calculating response {i+1}... ", end="")

                    response, current_num_pixels = self.get_feature_vector_of_window(
                        window
                    )
                    num_pixels += current_num_pixels
                    responses.append(response)

                    if verbose:
                        print("Done")

                responses_arr = np.array(responses, dtype=object)

                print("\t> Flattening responses to get feature vector... ", end="")

                # Every pixel of every single labeled window has 8 responses, which come from 8
                # response images. The following operations convert responses_arr to a matrix where
                # each row is a pixel. That means, each row will have 8 columns associated with each
                # pixel responses.
                feature_vector = SegmentationModel.concatenate_responses(responses_arr)
                assert feature_vector.shape == (
                    num_pixels,
                    8 + 4 * 3 * self.multiscale_statistics.scales,
                )
                feature_vectors_of_label[label] = feature_vector
                print("Done")

            print("")
            return feature_vectors_of_label
        else:
            print(
                "[+] Response vectors have been previously calculated for each label."
            )
            return self.feature_vectors_of_label

    def train(
        self,
        K: int = 100,
        multiscale_statistics_scales: int = 3,
        precomputed_feature_vectors: dict = None,
        train_size: float = 0.7,
        dev_size: float = 0.2,
        compute_silhouette_scores: bool = False,
        verbose: bool = True,
    ) -> None:
        """Trains the model by setting K equal to the number of clusters to be learned in K-means,
        i.e, the number of textons; and the number of scales to consider when computing multiscale
        statistics to 3.

        Args:
            K (int, optional): K-means algorithm parameter. Defaults to 100.
            multiscale_statistics_scales (int, optional): Scales to consider when computing
                                                          multiscales statistics. Defaults to 3.
            precomputed_feature_vectors (dict, optional): Precomputed eature vectors of labels to
                                                          use.
            train_size (float): Percentage of the dataset that will be used as training set.
            dev_size (float): Percentage of the dataset that will be used as development set.
        """
        self.K = K
        print("\n[*] TRAINING:\n")

        # Train/dev/test split
        self.windows_train, self.windows_dev, self.windows_test = train_dev_test_split(
            self.windows, train_size=train_size, dev_size=dev_size
        )

        train_dev_test_split_table(
            self.windows_train, self.windows_dev, self.windows_test
        )

        self.multiscale_statistics = MultiscaleStatistics(multiscale_statistics_scales)
        # Feature vector extraction per label on training set
        if (
            precomputed_feature_vectors is not None
            and list(precomputed_feature_vectors.values())[0].shape[-1]
            == 8 + multiscale_statistics_scales * 3 * 4
        ):
            self.feature_vectors_of_label = precomputed_feature_vectors
        else:
            self.feature_vectors_of_label = self.get_feature_vectors_of_labels(
                self.windows_train, multiscale_statistics_scales, verbose=verbose
            )

        self.classes = np.asarray(
            list(self.feature_vectors_of_label.keys())
        )  # Number of classes/labels
        self.C = len(self.classes)
        print_table_from_dict(
            self.feature_vectors_of_label, cols=["Label", "Shape of feature vector"]
        )

        self.textons = {}
        self.silhouette_coefficients = {}
        for label in self.feature_vectors_of_label:
            print(f"[?] Computing K-means on feature vector of label: {label}... ")

            self.textons[label] = KMeans(n_clusters=K).fit(
                np2cudf(self.feature_vectors_of_label[label])
            )

            print(
                "\tExample: ",
                self.textons[label]
                .cluster_centers_[randint(0, K - 1)]
                .astype(np.uint8),
            )

            if compute_silhouette_scores:
                print("\tComputing silhouette score: ", end="")
                self.silhouette_coefficients[label] = metrics.silhouette_score(
                    self.feature_vectors_of_label[label],
                    self.textons[label].labels_,
                    metric="euclidean",
                    n_jobs=-1,
                )
                print(f"{self.silhouette_coefficients[label]}")
            print("\tDone")

        # Matrix of texture textons
        # Once the textons have been learned for each of the classes, it is possible to construct a
        # matrix T of shape (C, K, 8) where each of the rows is a class and each column has the
        # texton k for k < K. Note that said texton must have 8 dimensions, since the pixels were
        # represented precisely by 8 dimensions.
        self.T = np.zeros(
            (self.C, K, 4 * 3 * multiscale_statistics_scales + 8), dtype=np.uint8
        )
        for i, label in enumerate(self.classes):
            for k in range(K):
                self.T[i, k] = self.textons[label].cluster_centers_[k]

    @staticmethod
    @jit
    def get_closest_texton_vector(feature_vectors: np.ndarray, T) -> np.ndarray:
        """Obtains a vector whose values are the minimum distances of each pixel of a
        superpixel. For example, if a superpixel has 300 pixels, this function returns a (300,)
        vector, where each value is the minimum distance of an enumerated pixel.

        Args:
            feature_vectors (np.ndarray): Output of get_feature_vectors_of_superpixel.

        Returns:
            np.ndarray: Minimum distance vector.
        """
        distance_matrix = np.linalg.norm(
            feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1
        )
        minimum_distance_vector = np.min(distance_matrix[np.newaxis], axis=(-1, 1))
        return minimum_distance_vector

    @staticmethod
    @jit
    def get_distance_matrix(feature_vectors: np.ndarray, T) -> np.ndarray:
        """Obtains a matrix which has the information of all possible distances from a pixel of
        a superpixel to every texton of every class.

        Args:
            feature_vectors (np.ndarray): Feature vectors of the superpixel.

        Returns:
            np.ndarray: Matrix of shape (10, NUM_PIXELS, K). Every (i, j, k) matrix value
                        corresponds to the distance from the i-th pixel to the k-th texton of
                        the j-th class.
        """
        return np.linalg.norm(
            feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1
        )

    def predict_class_of(self, feature_vectors: np.ndarray) -> str:
        """Predicts the class/label given the feature vectors that describe an image or a
        window of an image (like a superpixel).

        Args:
            feature_vectors (np.ndarray): Feature vectors of the image or window.

        Returns:
            str: Predicted class.
        """
        # Distance matrices.
        minimum_distance_vector = self.get_closest_texton_vector(
            feature_vectors, self.T
        )
        distance_matrix = self.get_distance_matrix(feature_vectors, self.T)

        # Matrix which correlates texture texton distances and minimum distances of every pixel.
        A = np.sum(
            np.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-09), axis=-1,
        )
        A_i = A.sum(axis=1)  # Sum over rows (i.e, over all pixels).
        ci = A_i.argmax(
            axis=0
        )  # Class with maximum probability of occurrence is chosen.

        return self.classes[ci]  # Assigned class is returned.

    def segment(
        self,
        img_name: str,
        n_segments: int = 500,
        sigma: int = 5,
        compactness: int = 0.1,
        plot_original: bool = False,
        plot_superpixels: bool = False,
        verbose: bool = False,
    ) -> tuple:
        """Segments an image. The model must have been trained before.

        Args:
            img_name (str): Name of image to be segmented.
            n_segments (int, optional): Maximum number of superpixels to generate. Defaults to 500.
            sigma (int, optional): SLIC algorithm parameter. Defaults to 5.
            compactness (int, optional): SLIC algorithm parameter. Defaults to 0.1.
            plot_original (bool, optional): True if a plot of the original micrograph is desired.
                                            Defaults to True.
            plot_superpixels (bool, optional): True if a plot of the superpixel generation is
                                               desired. Defaults to False.
            verbose (bool, optional): True if additional prints regarding the assignment of a class
                                      to a superpixel are desired. Defaults to False.
        """

        def get_superpixels(segments: np.ndarray) -> dict:
            """Creates a dictionary whose key corresponds to a superpixel and its value
            is a list of tuples which represent the coordinates of pixels belonging
            to the superpixel.

            Args:
                segments (np.ndarray): Output of SLIC algorithm.

            Returns:
                dict: Dictionary representation of the superpixels.
            """
            S = {}
            for key in np.unique(segments):
                S[key] = np.argwhere(segments == key)
            return S

        def get_feature_vectors_of_superpixel(
            responses: np.ndarray, superpixel: dict
        ) -> dict:
            """Creates a dictionary whose key corresponds to a pixel in tuple form and its value
            is the response/feature vector of that pixel.

            Args:
                responses (np.ndarray): Obtained from the function get_response_vector applied
                                        to the test image, whose superpixels are stored in the
                                        next arg.
                superpixel (dict): Ouput of the function get_superpixels.

            Returns:
                dict: Dictionary representation of the feature vectors of every pixel inside
                      every superpixel.
            """
            S = {}
            for pixel in superpixel:
                S[pixel] = {(i, j): responses[i, j] for i, j in superpixel[pixel]}
            return S

        test_img = load_img(img_name, as_255=False, with_io=True)  # Image is loaded.

        # The image is segmented using the SLIC algorithm.
        slic_model = SLICSegmentation(
            n_segments=n_segments, sigma=sigma, compactness=compactness
        )
        segments = slic_model.segment(test_img)

        if plot_original:
            print("Original image:")
            plt.figure()
            plt.imshow(test_img, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

        if plot_superpixels:
            print("\nSuperpixeles:")
            slic_model.plot_output(test_img, segments)

        S = get_superpixels(segments)  # Superpixels obtained from segments.
        responses, _ = self.get_feature_vector_of_window(test_img)
        S_feature_vectors = get_feature_vectors_of_superpixel(responses, S)

        def feature_vector_of(superpixel: int) -> np.ndarray:
            """Obtains the totality of the feature vectors of a superpixel as a numpy array
            (It includes all the pixels belonging to the given superpixel).

            Args:
                superpixel (int): Superpixel.

            Returns:
                np.ndarray: Feature vectors of a superpixel.
            """
            return np.array(list(S_feature_vectors[superpixel].values()))

        # The new segments are created, i.e, actual segmentation.
        S_segmented = {}
        for superpixel in S:
            current_feature_vectors = feature_vector_of(superpixel)
            S_segmented[superpixel] = self.predict_class_of(current_feature_vectors)

            if verbose:  # In case additional prints are desired.
                if superpixel % 25 == 0:
                    num_pixels = current_feature_vectors.shape[0]
                    print(
                        f"Superpixel {superpixel} (pixels={num_pixels}) assigned to "
                        + S_segmented[superpixel]
                    )

        return test_img, S, S_segmented

    def visualize_segmentation(
        self, original_img: np.ndarray, S: dict, S_segmented: dict
    ) -> None:
        """Plots a segmentation result on top of the original image.

        Args:
            original_img (np.ndarray): Numpy array associated with the original image.
            S (dict): Original dictionary of superpixels obtained from SLIC.
            S_segmented (dict): Segmentation result.
        """
        present_classes = list(Counter(S_segmented.values()))
        C_p = len(present_classes)

        colour_names = [
            "red",
            "blue",
            "yellow",
            "orange",
            "black",
            "purple",
            "green",
            "turquoise",
            "grey",
            "maroon",
            "silver",
        ]
        colour_dict = {
            class_: mpl.colors.to_rgb(colour_names[i])
            for i, class_ in enumerate(present_classes)
        }

        new_segments = np.zeros(original_img.shape, dtype=np.int16)
        overlay = np.zeros((*original_img.shape, 3), dtype=float)
        for superpixel in S:
            for i, j in S[superpixel]:
                new_segments[i, j] = np.where(self.classes == S_segmented[superpixel])[
                    0
                ].flatten()
                overlay[i, j] = colour_dict[self.classes[new_segments[i, j]]]

        present_colours = [
            colour_dict[present_class] for present_class in present_classes
        ]
        colours = mpl.colors.ListedColormap(present_colours)

        norm = mpl.colors.BoundaryNorm(np.arange(C_p + 1) - 0.5, C_p)

        plt.figure(figsize=(9, 6), dpi=90)
        plt.imshow(mark_boundaries(original_img, new_segments), cmap="gray")
        plt.imshow(overlay, cmap=colours, norm=norm, alpha=0.5)
        cb = plt.colorbar(ticks=np.arange(C_p))
        cb.ax.set_yticklabels(present_classes)

        plt.tight_layout(w_pad=100)
        plt.axis("off")
        plt.show()
        plt.close()

    def segmentation_to_class_matrix(
        self, S: dict, S_segmented: dict, shape: tuple
    ) -> np.ndarray:
        """Obtains a matrix of the given shape where each pixel of position i, j corresponds to the
        predicted class of that pixel.

        Args:
            S (dict): Dictionary of initially extracted superpixels.
            S_segmented (dict): Segmentation result.
            shape (tuple): Image shape.

        Returns:
            np.ndarray: Class matrix.
        """
        class_matrix = np.empty(shape, dtype=object)
        for superpixel in S:
            for i, j in S[superpixel]:
                index = np.where(self.classes == S_segmented[superpixel])[0][0]
                class_matrix[i, j] = self.classes[index]

        return class_matrix

    def classification_confusion_matrix(
        self, windows: dict, small_test: bool = False, max_test_number: int = 3
    ) -> np.ndarray:
        confusion_matrix = np.zeros((self.C, self.C))
        for i, label in enumerate(self.classes):
            for window in windows[label]:
                current_feature_vectors, _ = self.get_feature_vector_of_window(
                    window, ravel=True
                )
                predicted_class = self.predict_class_of(current_feature_vectors)
                predicted_class_index = np.where(self.classes == predicted_class)[0][0]
                confusion_matrix[i, predicted_class_index] += 1

            if small_test and i == max_test_number:
                break

        return confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    def evaluate_classification_performance(
        self,
        train: bool = True,
        dev: bool = False,
        test: bool = False,
        plot: bool = True,
        savefig: bool = True,
        small_test: bool = False,
        max_test_number: int = 3,
    ) -> None:
        def helper(
            windows: dict, category: str, img_filename: str, excel_filename: str
        ):
            print(
                f"[+] Computing classification performance on {category} set... ",
                end="",
            )
            confusion_matrix = self.classification_confusion_matrix(
                windows, small_test=small_test, max_test_number=max_test_number,
            )
            print("Done")
            plot_confusion_matrix(
                confusion_matrix,
                self.classes,
                title=f"K = {self.K}, ME = {self.multiscale_statistics.scales}",
                distinguishable_title=img_filename,
                savefig=savefig,
                showfig=plot,
            )
            print(" > Exporting to excel... ", end="")
            matrix_to_excel(
                confusion_matrix,
                self.classes.tolist(),
                sheetname=f"K = {self.K}, ME = {self.multiscale_statistics.scales}",
                filename=excel_filename,
            )
            print("Done")

        print("\n[*] CLASSIFICATION PERFORMANCE:\n")
        constant_title = (
            f"(classification) K={self.K}, ME={self.multiscale_statistics.scales}"
        )
        if train:
            helper(self.windows_train, "Training", f"Train, {constant_title}", "Train")
        if dev:
            helper(self.windows_dev, "Development", f"Dev, {constant_title}", "Dev")
        if test:
            helper(self.windows_test, "Testing", f"Test, {constant_title}", "Test")

    def segmentation_confusion_matrix(
        self, src: str = "", small_test: bool = False, max_test_number: int = 3,
    ) -> np.ndarray:
        if src == "":
            src = self.PATH_LABELED

        shuffled_windows_per_name = list(self.windows_per_name.keys())
        shuffle(shuffled_windows_per_name)
        matrix = np.zeros((self.C, self.C))
        for k, name in enumerate(shuffled_windows_per_name):
            print(f"\t[?] Segmenting {name}... ", end="")
            original_img, superpixels, segmented_img = self.segment(
                find_path_of_img(name, src=src)
            )
            class_matrix = self.segmentation_to_class_matrix(
                superpixels, segmented_img, original_img.shape
            )
            print("Done")
            print("\t    > Evaluating results... ", end="")
            for coords, _, true_class in self.windows_per_name[name]:
                i = np.where(self.classes == true_class)[0][0]
                segmented_window = self.slice_by_corner_coords(class_matrix, *coords)
                class_counts = dict(
                    zip(*np.unique(segmented_window.ravel(), return_counts=True))
                )
                for predicted_class, count in class_counts.items():
                    j = np.where(self.classes == predicted_class)[0][0]
                    matrix[i, j] = count
            print("Done")

            if small_test and k == max_test_number:
                break

        return matrix / matrix.sum(axis=1, keepdims=True)

    def evaluate_segmentation_performance(
        self,
        plot: bool = True,
        savefig: bool = True,
        small_test: bool = False,
        max_test_number: int = 3,
    ) -> None:
        print("\n[*] SEGMENTATION PERFORMANCE:")
        print("\n[+] Computing segmentation performance... ")
        confusion_matrix = self.segmentation_confusion_matrix(
            small_test=small_test, max_test_number=max_test_number
        )
        print("Done")
        filename = f"(segmentation) K={self.K}, ME={self.multiscale_statistics.scales}"
        plot_confusion_matrix(
            confusion_matrix,
            self.classes,
            title=f"K = {self.K}, ME = {self.multiscale_statistics.scales}",
            distinguishable_title=filename,
            savefig=savefig,
            showfig=plot,
            format_percentage=True,
        )
        print(" > Exporting to excel... ", end="")
        matrix_to_excel(
            confusion_matrix,
            self.classes.tolist(),
            sheetname=f"K = {self.K}, ME = {self.multiscale_statistics.scales}",
            filename="Segmentation",
        )
        print("Done")
