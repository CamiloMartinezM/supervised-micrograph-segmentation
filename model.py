# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:59:47 2020

@author: Camilo Martínez
"""
import json
import os
import warnings
from itertools import chain
from random import randint

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cuml import KMeans as CumlKMeans
from cuml.metrics.cluster.entropy import cython_entropy
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit
from numba.core.errors import NumbaWarning
from pycm import ConfusionMatrix
from skimage import io
from skimage.segmentation import mark_boundaries
from sklearn.cluster import MiniBatchKMeans

from utils_classes import FilterBank, Scaler, SuperpixelSegmentation
from utils_functions import (
    calculate_spacing,
    find_path_of_img,
    formatter,
    get_folder,
    highlight_class_in_img,
    img_to_binary,
    jaccard_index_from_ground_truth,
    load_img,
    matrix_to_excel,
    np2cudf,
    plot_confusion_matrix,
    print_table_from_dict,
    statistics_from_matrix,
)

warnings.simplefilter("ignore", category=NumbaWarning)


def load_scales(
    path_labeled: str, path_preprocessed: str, load_full_preprocessed: bool = False
) -> dict:
    """Loads the scale of the images by going to PREPROCESSED and finding the
    corresponding SCALE image of every image in LABELED.

    Returns:
        dict: Dictionary whose keys are names of images and values are their respective
              scale pixel length.
    """
    print("\n[*] SCALES EXTRACTION:\n")
    if load_full_preprocessed:
        myScaler = Scaler(path_preprocessed, path_preprocessed)
    else:
        myScaler = Scaler(path_labeled, path_preprocessed)

    myScaler.process()
    scales = myScaler.scales

    print_table_from_dict(
        scales,
        cols=["Name of micrograph", "Pixels in scale"],
        title="Pixel length scales",
    )

    return scales


def load_imgs(imgs_path: str, exclude: list = []) -> tuple:
    """Loads images in LABELED in a numpy array.

    Args:
        exclude (list, optional): Folders to exclude in loading. Defaults to [].

    Returns:
        tuple: Numpy array with all images, dictionary whose keys are names of images
               and values are the corresponding indeces in the numpy array of images.
    """
    m = []
    index_to_name = {}  # Every name will have its corresponding position in m.
    count = 0
    for path, _, files in os.walk(imgs_path):
        print(f"[+] Currently reading: {path}")
        for folder_to_exclude in exclude:
            if folder_to_exclude in path:
                print(" └── Excluded from search.")
                break
        else:
            filtered_files = [
                f for f in files if f.endswith(".png") and not f.startswith("SCALE")
            ]
            bullet = " ├── "
            for i, f in enumerate(filtered_files):
                if i == len(filtered_files) - 1:
                    bullet = " └── "

                print(bullet + f"Reading and loading {f}... ", end="")
                img = load_img(os.path.join(path, f))
                m.append(img)
                index_to_name[f] = count
                count += 1
                print("Done")

    return np.array(m), index_to_name


def preprocess_with_clahe(src: str) -> None:
    """Preprocess the images on the given path.

    Args:
        src (str): Path where to process images.
    """
    check_file = "already_preprocessed.txt"

    def already_preprocessed() -> bool:
        """Checks if the given path has previously been preprocessed."""
        if check_file in os.listdir(src):
            return True
        else:
            return False

    def mark_as_preprocessed() -> None:
        """Marks the given path as preprocessed by creating a file called
        'already_preprocessed.txt'.
        """
        try:
            with open(os.path.join(src, check_file), "w") as f:
                f.write("Already preprocessed.")
            print(f"[+] Directory {src} succesfully marked as preprocessed.")
        except:
            print(f"[*] Directory {src} could not be marked as preprocessed.")

    if not already_preprocessed():
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print(f"\n[+] Directory {src} will be preprocessed")
        for path, _, files in os.walk(src):
            for f in files:
                if f.endswith(".png") and not f.startswith("SCALE"):
                    print("[+] Preprocessing " + str(f) + "... ", end="")
                    folder = get_folder(path)
                    if folder is not None:
                        original_path = find_path_of_img(f, src, relative_path=True)
                        img = cv2.imread(r"" + original_path, 0)
                        final_img = clahe.apply(img)
                        cv2.imwrite(original_path, final_img)
                        print("Done")
                    else:
                        print("Failed. Got None as folder.")
        print("[+] Finished preprocessing.")
        mark_as_preprocessed()


def get_array_of_micrograph(
    name: str, micrographs: dict, index_to_name: dict
) -> np.ndarray:
    """Gets the numpy array of an image with the given name.

    Args:
        name (str): Name of image.
        micrographs (dict): Numpy array with all images.
        index_to_name (dict): Dictionary whose keys are names of images and values
                              are the corresponding indeces in the numpy array of
                              images.

    Returns:
        np.ndarray: Numpy array of image.
    """
    return micrographs[index_to_name[name]]


def extract_labeled_windows(
    path_labeled: str, micrographs: dict, index_to_name: dict, exclude: list = []
) -> tuple:
    """Within LABELED, each of the images has a .txt file associated with it that
    contains the information of the position of each of its regions or windows that
    were annotated. This section is then in charge of extracting said regions by slicing
    the numpy array of an image accordingly to get each of its labeled windows.

    Args:
        micrographs (dict): Numpy array with all images.
        index_to_name (dict): Dictionary whose keys are names of images and values
                              are the corresponding indeces in the numpy array of
                              images.
        exclude (list, optional): Folders to exclude in loading. Defaults to [].

    Returns:
        tuple: Dictionary of label counts; dictionary of windows whose keys are the
               labels and values are a list of numpy arrays, which are the windows
               associated with the label; and dictionary of windows and respective
               labels per loaded image.
    """

    def check_label(label: str) -> str:
        """Makes sure all label names are consistent.

        Args:
            label (str): Input label.

        Returns:
            str: Consistent label name.
        """
        translation_dictionary = {
            "perlita": "pearlite",
            "ferrita": "ferrite",
            "ferrita proeutectoide": "proeutectoid ferrite",
            "cementita proeutectoide": "proeutectoid cementite",
            "sulfuro de manganeso": "manganese sulfide",
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
            return None

    labels = {}  # Counts number of labels/annotations per label.
    windows_per_label = {}  # key = label, value = list of windows
    windows_per_name = {}  # key = filename, value = [(coords, window, label)]

    for path, _, files in os.walk(path_labeled):
        print(f"[+] Currently reading: {path}")
        for folder_to_exclude in exclude:
            if folder_to_exclude in path:
                print(" └── Excluded from search.")
                break
        else:
            filtered_files = [f for f in files if f.endswith(".txt")]
            bullet = " ├── "
            for i, f in enumerate(filtered_files):
                if i == len(filtered_files) - 1:
                    bullet = " └── "

                img_name = f[:-4] + ".png"
                print(bullet + f"Getting windows of {img_name}... ", end="")
                full_img = get_array_of_micrograph(
                    img_name, micrographs, index_to_name
                )  # Loads full img from micrographs array

                with open(os.path.join(path, f), "r") as annotations:
                    line = annotations.readline()
                    while len(line) > 0:
                        line_parts = line.split(" ")
                        label = line_parts[0]
                        k = 1
                        while k < len(line_parts):
                            try:
                                int(
                                    float(line_parts[k])
                                )  # Added float because int('0.0') does not work
                                break
                            except:  # is string
                                label += " " + line_parts[k]
                            k += 1

                        offset = k - 1
                        label = check_label(label)
                        if label is not None:
                            labels[label] = labels.get(label, 0) + 1
                            first_point = tuple(
                                [int(x) for x in line_parts[4 + offset : 6 + offset]]
                            )
                            second_point = tuple(
                                [int(x) for x in line_parts[6 + offset : 8 + offset]]
                            )
                            assert (first_point and second_point) != (0, 0)
                            window = slice_by_corner_coords(
                                full_img, first_point, second_point
                            )

                            if img_name not in windows_per_name:
                                windows_per_name[img_name] = []

                            windows_per_name[img_name].append(
                                ((first_point, second_point), window, label)
                            )

                            if label not in windows_per_label:
                                windows_per_label[label] = []
                            windows_per_label[label].append((img_name, window))
                        # print("Done")
                        line = annotations.readline()

                print("Done")

    print_table_from_dict(
        labels, cols=["Label", "Number"], title="Number of windows per label",
    )

    return labels, windows_per_label, windows_per_name


def filterbank_example(
    path_labeled: str,
    img: str = "cs0328.png",
    dpi: int = 80,
    filterbank_name: str = "MR8",
) -> None:
    """Plots an example of the chosen filterbank.

    Args:
        img (str, optional): Image to filter and show. Defaults to "cs0328.png".
        dpi (int, optional): DPI of plotted figure. Defaults to 80.
    """
    MR8 = FilterBank(name=filterbank_name)  # MR8 Filter bank
    print("\nFilters (RFS Filter Bank):")
    MR8.plot_filters()

    # Example
    img = load_img(find_path_of_img(img, path_labeled))
    response = MR8.response(img)

    # Original image
    print("")
    print("Original image:")
    plt.figure(dpi=dpi)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.pause(0.05)
    # plt.show()
    # plt.close()

    # Plot responses
    print("")
    print("Responses:")
    fig2, ax2 = plt.subplots(3, 3)
    fig2.set_dpi(dpi)
    for axes, res in zip(ax2.ravel(), response):
        axes.imshow(res, cmap=plt.cm.gray)
        axes.set_xticks(())
        axes.set_yticks(())
    ax2[-1, -1].set_visible(False)
    fig2.tight_layout()
    plt.pause(0.05)
    # plt.show()


def slice_by_corner_coords(
    img: np.ndarray, first_point: tuple, second_point: tuple
) -> np.ndarray:
    """Slices a numpy array using 2 coordinates: upper left and lower right.

    Args:
        img (np.ndarray): Image to slice.
        first_point (tuple): First coordinate.
        second_point (tuple). Second coordinate.

    Returns:
        np.ndarray: sliced image.
    """
    return img[first_point[1] : second_point[1], first_point[0] : second_point[0]]


def get_response_vector(img: np.ndarray, filterbank_name: str = "MR8") -> np.ndarray:
    """Convolves the input image with the MR8 Filter Bank to get its response as a
    numpy array.

    Args:
        img (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Numpy array of shape (*img.shape, 8).
    """
    filterbank = FilterBank(name=filterbank_name)  # MR8 Filter bank

    # 8 responses from image
    r = filterbank.response(img)

    # Every response is stacked on top of the other in a single matrix whose last axis has
    # dimension 8. That means, there is now only one response image, in which each channel
    # contains the information of each of the 8 responses.
    response = np.concatenate(
        [np.expand_dims(r[i], axis=-1) for i in range(len(r))], axis=-1
    )
    assert response.shape == (*r[0].shape, filterbank.n_filters)
    return response


def concatenate_responses(responses: np.ndarray) -> np.ndarray:
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


def get_feature_vector_of_window(
    window: np.ndarray, ravel: bool = False, filterbank_name: str = "MR8"
) -> tuple:
    """Obtains the feature vectors of an image or window.

    Args:
        window (np.ndarray): Image as a numpy array.
        ravel (bool, optional): Specifies whether to flatten the feature vectors of an
                                image, so that each row is the feature vector of a
                                single pixel. If this parameter is True, the output will
                                be reshaped to the original image shape.
    Returns:
        tuple: Feature vector of the given window and the number of pixels whose feature
               vector was calculated.
    """
    response_img = get_response_vector(window, filterbank_name)
    num_pixels = response_img.size

    if ravel:
        return (
            response_img.reshape((window.size, response_img.shape[-1])),
            num_pixels,
        )
    else:
        return response_img, num_pixels


def feature_vectors_from_windows(
    windows: dict, verbose: bool = True, filterbank_name: str = "MR8"
) -> dict:
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

            response, current_num_pixels = get_feature_vector_of_window(
                window, filterbank_name=filterbank_name
            )
            num_pixels += current_num_pixels
            responses.append(response)

            if verbose:
                print("Done")

        responses_arr = np.array(responses, dtype=object)

        print("\t> Flattening responses to get feature vector... ", end="")

        # Every pixel of every single labeled window has 8 responses, which come from 8
        # response images. The following operations convert responses_arr to a matrix
        # where each row is a pixel. That means, each row will have 8 columns associated
        # with each pixel responses.
        feature_vector = concatenate_responses(responses_arr)

        # assert feature_vector.shape == (
        #     num_pixels,
        #     8 + 4 * 3 * multiscale_statistics.scales,
        # )
        feature_vectors_of_label[label] = feature_vector
        print("Done")

    print("")
    return feature_vectors_of_label


def obtain_feature_vectors_of_labels(
    windows_train: dict, filterbank: str, windows_dev: dict = None, verbose: bool = True
) -> dict:
    """Obtains the feature vectors of the labels present in the given windows.

    Args:
        windows_train (dict): Training set.
        windows_dev (dict, optional): Development set. If it is not None, it is included
                                        on training. Defaults to None.

    Returns:
        dict: Feature_vectors of labels.
    """
    # Feature vector extraction per label on training set
    if windows_dev is not None:
        windows_to_train_on = {}
        for k, v in chain(windows_train.items(), windows_dev.items()):
            windows_to_train_on.setdefault(k, []).extend(v)

        windows_train = windows_to_train_on

    feature_vectors_of_label = feature_vectors_from_windows(
        windows_train, verbose=verbose, filterbank_name=filterbank
    )

    return feature_vectors_of_label


def train(
    K: int,
    filterbank_name: str,
    feature_vectors: dict,
    minibatch_size: int = None,
    compute_clustering_entropy: bool = False,
    verbose: bool = True,
) -> None:
    """Trains the model by setting K equal to the number of clusters to be learned in
    K-means, i.e, the number of textons.

    Args:
        K (int): K-Means algorithm parameter.
        windows_train (dict): Training set.
        windows_dev (dict, optional): Development set. If it is not None, it is included
                                        on training. Defaults to None.
        precomputed_feature_vectors (dict, optional): Precomputed feature vectors of
                                                        labels to use. Defaults to None.
        minibatch_size (int, optional): Minibatch size paraemter in MiniBatchKMeans in
                                        case this method is going to be used and not
                                        CumlKMeans. Defaults to None.
        compute_clustering_entropy (bool, optional): True if the clustering entropy is
                                                     to be computed. Defaults to False.
        verbose (bool, optional): Specifies whether to include aditional information on
                                    console. Defaults to True.

    Returns:
        tuple: feature_vectors (dict), classes (list), texton matrix (np.ndarray) and
                clustering entropy (dict).
    """
    classes = np.asarray(list(feature_vectors.keys()))  # Number of classes/labels
    C = len(classes)

    print_table_from_dict(feature_vectors, cols=["Label", "Shape of feature vector"])
    print("")

    textons = {}
    clustering_entropy = {}
    for label in feature_vectors:
        print(f"[?] Computing K-means on feature vector of label: {label}... ")
        if minibatch_size is not None:
            textons[label] = MiniBatchKMeans(n_clusters=K).fit(feature_vectors[label])
        else:
            textons[label] = CumlKMeans(n_clusters=K, output_type="numpy").fit(
                np2cudf(feature_vectors[label])
            )

        print(
            "\tExample: ",
            textons[label].cluster_centers_[randint(0, K - 1)].astype(np.uint8),
        )

        if compute_clustering_entropy:
            print("\tComputing clustering entropy: ", end="")
            clustering_entropy[label] = cython_entropy(textons[label].labels_)
            print(f"{clustering_entropy[label]}")
        print("\tDone")

    # Matrix of texture textons
    # Once the textons have been learned for each of the classes, it is possible to
    # construct a matrix T of shape (C, K, 8) where each of the rows is a class and
    # each column has the texton k for k < K. Note that said texton must have 8
    # dimensions, since the pixels were represented precisely by 8 dimensions.
    T = np.zeros((C, K, feature_vectors[classes[0]].shape[-1]), dtype=np.float64)
    for i, label in enumerate(classes):
        T[i] = textons[label].cluster_centers_

    return classes, T, clustering_entropy


@jit
def get_closest_texton_vector(feature_vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Obtains a vector whose values are the minimum distances of each pixel of a
    superpixel. For example, if a superpixel has 300 pixels, this function returns a
    (300,) vector, where each value is the minimum distance of an enumerated pixel.

    Args:
        feature_vectors (np.ndarray): Output of get_feature_vectors_of_superpixel.
        T (np.ndarray): Texton matrix.

    Returns:
        np.ndarray: Minimum distance vector.
    """
    distance_matrix = np.linalg.norm(
        feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1
    )
    minimum_distance_vector = np.min(distance_matrix[np.newaxis], axis=(-1, 1))
    return minimum_distance_vector


@jit
def get_distance_matrix(feature_vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Obtains a matrix which has the information of all possible distances from a pixel
    of a superpixel to every texton of every class.

    Args:
        feature_vectors (np.ndarray): Feature vectors of the superpixel.
        T (np.ndarray): Texton matrix.

    Returns:
        np.ndarray: Matrix of shape (C, NUM_PIXELS, K). Every (i, j, k) matrix value
                    corresponds to the distance from the i-th pixel to the k-th texton
                    of the j-th class.
    """
    return np.linalg.norm(feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1)


@jit
def predict_class_of(
    feature_vectors: np.ndarray, classes: np.ndarray, T: np.ndarray
) -> str:
    """Predicts the class/label given the feature vectors that describe an image or a
    window of an image (like a superpixel).

    Args:
        feature_vectors (np.ndarray): Feature vectors of the image or window.
        classes (np.ndarray): Array of classes/labels.
        T (np.ndarray): Texton matrix.

    Returns:
        str: Predicted class.
    """
    # Distance matrices.
    minimum_distance_vector = get_closest_texton_vector(feature_vectors, T)
    distance_matrix = get_distance_matrix(feature_vectors, T)

    # Matrix which correlates texture texton distances and minimum distances of every
    # pixel.
    A = np.sum(
        np.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-09), axis=-1,
    )
    A_i = A.sum(axis=1)  # Sum over rows (i.e, over all pixels).
    ci = A_i.argmax(axis=0)  # Class with maximum probability of occurrence is chosen.

    return ci, classes[ci]  # Assigned class is returned.


def segment(
    img: np.ndarray,
    classes: np.ndarray,
    T: np.ndarray,
    algorithm: str,
    algorithm_parameters: tuple,
    filterbank_name: str = "MR8",
    plot_original: bool = False,
    plot_superpixels: bool = False,
    verbose: bool = False,
    subsegment_class: tuple = None,
) -> tuple:
    """Segments an image. The model must have been trained before.

    Args:
        img_name (str): Name of image to be segmented.
        classes (np.ndarray): Array of classes/labels.
        T (np.ndarray): Texton matrix.
        n (int): Maximum number of superpixels to generate.
        sigma (int): SLIC algorithm parameter.
        compactness (int): SLIC algorithm parameter.
        plot_original (bool, optional): True if a plot of the original micrograph is
                                        desired. Defaults to True.
        plot_superpixels (bool, optional): True if a plot of the superpixel generation
                                            is desired. Defaults to False.
        verbose (bool, optional): True if additional prints regarding the assignment of
                                    a class to a superpixel are desired. Defaults to
                                    False.
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
        """Creates a dictionary whose key corresponds to a pixel in tuple form and its
        value is the response/feature vector of that pixel.

        Args:
            responses (np.ndarray): Obtained from the function get_response_vector
                                    applied to the test image, whose superpixels are
                                    stored in the next arg.
            superpixel (dict): Ouput of the function get_superpixels.

        Returns:
            dict: Dictionary representation of the feature vectors of every pixel inside
                    every superpixel.
        """
        S = {}
        for pixel in superpixel:
            S[pixel] = {(i, j): responses[i, j] for i, j in superpixel[pixel]}
        return S

    # The image is segmented using the given algorithm.
    superpixel_generation_model = SuperpixelSegmentation(
        algorithm, algorithm_parameters
    )
    segments = superpixel_generation_model.segment(img)

    if plot_original:
        print("\nOriginal image:")
        plt.figure(figsize=(10, 8), dpi=120)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.pause(0.05)
        # plt.show()
        # plt.close()

    if plot_superpixels:
        print("\nSuperpixels:")
        superpixel_generation_model.plot_output(img, segments, dpi=120)

    S = get_superpixels(segments)  # Superpixels obtained from segments.
    responses, _ = get_feature_vector_of_window(img, filterbank_name=filterbank_name)
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
    class_matrix = np.zeros(img.shape, dtype=int)
    for superpixel in S:
        current_feature_vectors = feature_vector_of(superpixel)
        predicted_class_idx, S_segmented[superpixel] = predict_class_of(
            current_feature_vectors, classes, T
        )
        idx = S[superpixel]
        rows, cols = zip(*idx)
        class_matrix[rows, cols] = predicted_class_idx

        if verbose:  # In case additional prints are desired.
            if superpixel % 25 == 0:
                num_pixels = current_feature_vectors.shape[0]
                print(
                    f"Superpixel {superpixel} (pixels={num_pixels}) assigned to "
                    + S_segmented[superpixel]
                )

    if subsegment_class is not None:
        class_to_subsegment, name_of_resulting_class = subsegment_class
        # Check if the class to subsegment is present in the segmentation
        if np.where(classes == class_to_subsegment)[0][0] in np.unique(class_matrix):
            idx = np.where(classes == class_to_subsegment)[0][0]
            mapping = {0: np.where(classes == class_to_subsegment)[0][0], 1: 2}
            img_255 = cv2.normalize(
                img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_64F,
            ).astype(
                np.uint8
            )  # Image is changed to [0, 255] range.

            class_matrix, new_classes = sub_segment(
                img_255, class_matrix, idx, name_of_resulting_class, classes, mapping
            )
        else:
            print(
                f"{class_to_subsegment} was not found in segmentation, so it won't be "
                "subsegmented into {name_of_resulting_class}"
            )
    else:
        new_classes = classes.copy()

    present_classes_idxs, pixel_counts = np.unique(class_matrix, return_counts=True)
    present_classes = new_classes[
        present_classes_idxs.min() : present_classes_idxs.max() + 1
    ]
    segmentation_pixel_counts = dict(zip(present_classes, pixel_counts))

    return img, class_matrix, new_classes, segmentation_pixel_counts


def sub_segment(
    img: np.ndarray,
    segments: np.ndarray,
    class_to_subsegment: int,
    name_of_resulting_class: str,
    classes: np.ndarray,
    binary_mapping: dict,
) -> tuple:
    """Subsegments a class of an input image. The resulting new class must be brighter
    than the original class, which is subsegmented.

    Args:
        img (np.ndarray): Input image.
        segments (np.ndarray): Class matrix; segmented array where each value
                               corresponds to a class/label.
        class_to_subsegment (int): Value in segments of the class which is going to be
                                   subsegmented.
        name_of_resulting_class (str): Name of the resulting new class.
        classes (np.ndarray): Array of classes/labels.
        binary_mapping (dict): Tells which value in segments will correspond to which
                               class.

    Returns
        tuple: new class matrix (or segments), and array of new classes/labels
    """
    subsegmented_img = highlight_class_in_img(
        img_to_binary(img), segments, class_to_subsegment, fill_value=-1
    )
    new_segments = segments.copy()
    for binary_value, corresponding_class in binary_mapping.items():
        new_segments[subsegmented_img == binary_value] = corresponding_class

    new_classes = classes.copy().tolist()
    new_classes.append(name_of_resulting_class)

    return new_segments, np.array(new_classes)


def calculate_interlamellar_spacing(
    original_img: np.ndarray,
    segmented_img: np.ndarray,
    classes: np.ndarray,
    preprocess: bool = True,
    highlight_pearlite: bool = True,
    img_name: str = "img",
    save_plots: bool = False,
    dpi: int = 120,
) -> float:
    pearlite_class_idx = np.where(classes == "pearlite")[0][0]

    if preprocess:
        preprocessed_img = original_img  # TODO
    else:
        preprocessed_img = original_img

    if highlight_pearlite:
        img = highlight_class_in_img(
            preprocessed_img, segmented_img, class_=pearlite_class_idx
        )
    else:
        img = preprocessed_img

    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.pause(0.05)

    return calculate_spacing(img, img_name=img_name, save_plots=save_plots, dpi=dpi)


def visualize_segmentation(
    original_img: np.ndarray,
    classes: np.ndarray,
    segments: np.ndarray,
    dpi: int = 120,
    save_png: bool = False,
    png_name: str = "segmentation.png",
) -> None:
    """Plots a segmentation result on top of the original image.

    Args:
        original_img (np.ndarray): Numpy array associated with the original image.
        classes (np.ndarray): Array of classes/labels.
        segments (np.ndarray): Class matrix; segmented array where each value
                               corresponds to a class/label.
        dpi (int, optional): DPI for plotted figure. Defaults to 120.
    """
    present_classes = np.unique(segments).tolist()
    C_p = len(present_classes)

    colour_names = [
        "blue",
        "red",
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

    k = np.array(list(colour_dict.keys()))
    v = np.array(list(colour_dict.values()))

    mapping_ar = np.zeros((k.max() + 1, 3), dtype=v.dtype)
    mapping_ar[k] = v
    overlay = mapping_ar[segments]

    present_colours = [colour_dict[present_class] for present_class in present_classes]
    colours = mpl.colors.ListedColormap(present_colours)

    norm = mpl.colors.BoundaryNorm(np.arange(C_p + 1) - 0.5, C_p)

    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.imshow(mark_boundaries(original_img, segments), cmap="gray")
    plt.imshow(overlay, cmap=colours, norm=norm, alpha=0.6)
    cb = plt.colorbar(ticks=np.arange(C_p))
    cb.ax.set_yticklabels(classes[min(present_classes) : max(present_classes) + 1])

    plt.tight_layout(w_pad=100)
    plt.axis("off")
    plt.pause(0.05)
    if save_png:
        plt.savefig(png_name, bbox_inches=0, dpi=dpi)

    # plt.show()
    # plt.close()


def segmentation_to_class_matrix(
    classes: np.ndarray, S: dict, S_segmented: dict, shape: tuple, as_int: bool = False
) -> np.ndarray:
    """Obtains a matrix of the given shape where each pixel of position i, j corresponds
    to the predicted class of that pixel.

    Args:
        classes (np.ndarray): Array of classes/labels.
        S (dict): Dictionary of initially extracted superpixels.
        S_segmented (dict): Segmentation result.
        shape (tuple): Image shape.
        as_int (bool): True if the desired matrix must have integers as the
                       corresponding class. Defaults to False.
    Returns:
        np.ndarray: Class matrix.
    """
    if as_int:
        class_matrix = np.zeros(shape, dtype=int)
    else:
        class_matrix = np.empty(shape, dtype=object)

    for superpixel in S:
        for i, j in S[superpixel]:
            index = np.where(classes == S_segmented[superpixel])[0][0]
            if as_int:
                class_matrix[i, j] = index
            else:
                class_matrix[i, j] = classes[index]

    return class_matrix


def classification_confusion_matrix(
    classes: np.ndarray,
    T: np.ndarray,
    windows: dict,
    max_test_number: int = -1,
    filterbank_name: str = "MR8",
) -> np.ndarray:
    """Obtains the confusion matrix for classification. Essentially, this confusion
    matrix evaluates how good the model classifies each labeled window. In other words,
    if the model predicts the correct label for a labeled window. Note that this is
    different from the segmentation problem and should not be used as a direct estimator
    for the model's segmentation performance.

    Args:
        classes (np.ndarray): Array of classes/labels.
        T (np.ndarray): Texton matrix.
        windows (dict): Dictionary of labeled windows. These labeled windows are
                        classified by the model and the result is tested against the
                        known label.
        max_test_number (int, optional): Maximum number of labeled windows to take into
                                         account for evaluating classification
                                         performance. If set to -1, all labeled windows
                                         are taken. Defaults to -1.

    Returns:
        np.ndarray: Confusion matrix (un-normalized).
    """
    y_true = []
    y_pred = []
    for i, label in enumerate(classes):
        for window in windows[label]:
            current_feature_vectors, _ = get_feature_vector_of_window(
                window, ravel=True, filterbank_name=filterbank_name
            )
            predicted_class_index, predicted_class = predict_class_of(
                current_feature_vectors, classes, T
            )
            y_pred.append(predicted_class_index)
            y_true.append(i)
        if i == max_test_number:
            break

    cm = ConfusionMatrix(y_true, y_pred, transpose=True)

    return cm, cm.to_array(normalized=True)


def evaluate_classification_performance(
    K: int,
    classes: np.ndarray,
    T: np.ndarray,
    filterbank_name: str,
    windows_train: dict = None,
    windows_dev: dict = None,
    windows_test: dict = None,
    save_png: bool = True,
    save_xlsx: bool = True,
    normalized: bool = True,
    max_test_number: int = -1,
) -> None:
    """Evaluates the model's classification performance by creating the corresponding
    classification confusion matrices for training, development and test set (if they
    are requested) and provides the option to plot them, save them as a .png and
    .xlsx file.

    Args:
        K (int): K-Means algorithm parameter.
        classes (np.ndarray): Array of classes/labels.
        T (np.ndarray): Texton matrix.
        windows_train (dict, optional): Training set. Defaults to None.
        windows_dev (dict, optional): Development set. Defaults to None.
        windows_test (dict, optional): Test set. Defaults to None.
        save_png (bool, optional): Specifies whether to save the generated figures to
                                   the filesystem as .png files. Defaults to True.
        save_xlsx (bool, optional): Specifies whether to save the generated figures to
                                    the filesystem as a .xlsx file. Defaults to True.
        normalized (bool, optional): True if matrix is to be normalized.
        max_test_number (int, optional): Maximum number of labeled windows to take into
                                         account for evaluating classification
                                         performance. If set to -1, all labeled windows
                                         are taken. Defaults to -1.
    """

    def helper(
        windows: dict,
        category: str,
        img_filename: str,
        excel_filename: str,
        sheetname: str,
    ) -> None:
        """Helper function to avoid repetitive evaluation of classification performance
        of training, development and test set and their corresponding windows dicts.

        Args:
            windows (dict): Dictionary of labeled windows.
            category (str): Name of category (Training, Development, Testing).
            img_filename (str): Filename of .png file.
            excel_filename (str): Filename of .xlsx file.
        """
        if windows is not None:
            print(
                f"[+] Computing classification performance on {category} set... ",
                end="",
            )
            cm, cm_array = classification_confusion_matrix(
                classes, T, windows, max_test_number, filterbank_name=filterbank_name
            )
            # Maps integer classes to actual names of classes.
            # Example: {'0': 'proeutectoid ferrite', '1': 'pearlite'}
            integer_classes = [str(i) for i in range(len(classes))]
            mapping = dict(zip(integer_classes, classes))
            cm.relabel(mapping=mapping)
            print("Done")
            plot_confusion_matrix(
                cm, normalized=normalized, title=img_filename, save_png=save_png,
            )
            if save_xlsx:
                print(" ├── Exporting to excel... ", end="")
                matrix_to_excel(
                    cm_array,
                    classes.tolist(),
                    sheetname=sheetname,
                    filename=excel_filename,
                )
                print("Done")
            print(" └── Computing metrics... ", end="")
            stats = statistics_from_matrix(cm)
            print("Done")
            return stats

    constant_title = f"(classification) K={K}"
    metrics = {}
    metrics["Train"] = helper(
        windows_train, "Training", f"Train, {constant_title}", "Train", f"K = {K}"
    )
    metrics["Dev"] = helper(
        windows_dev, "Development", f"Dev, {constant_title}", "Dev", f"K = {K}"
    )
    metrics["Test"] = helper(
        windows_test, "Testing", f"Test, {constant_title}", "Test", f"K = {K}"
    )
    return metrics


def load_ground_truth(
    path_ground_truth: str, labeled_folder: str, classes: np.ndarray
) -> dict:
    """Obtains a dictionary of ground truth images, where keys are names of images
    and values are the corresponding ground truth images. This is the ground truth
    for segmentation.

    Args:
        tif_file (str): Name of .tif file where the ground truth images are. Since
                        this file does not have labels for each image, the
                        alphabetical order of files in PATH_LABELED is used.
        classes (np.ndarray): Array of labels/classes.
        folder (str): Specifies the folder inside PATH_LABELED where the alphabetical
                      order of the images matches the ground truth images'.

    Returns:
        dict: Dictionary of ground truth images.
    """
    ground_truth_imgs = io.imread(path_ground_truth)[:, 0, :, :]
    ground_truth = np.zeros(ground_truth_imgs.shape, dtype=int)
    for i in range(ground_truth_imgs.shape[0]):
        ground_truth_img = ground_truth_imgs[i, :, :].astype(int)
        ground_truth[i, :, :] = make_classes_consistent(ground_truth_img, classes)

    l = [f for f in sorted(os.listdir(labeled_folder)) if f.endswith(".png")]
    return dict(zip(l, ground_truth))


def make_classes_consistent(
    ground_truth_img: np.ndarray, classes: np.ndarray, correspondence: dict = None
) -> np.ndarray:
    """Given an image as a numpy array, this method makes sure the classes' numbering
    is consistent. This means that each number in a ground truth image correspond to
    a correct class in classes.

    Args:
        ground_truth_img (np.ndarray): Raw ground truth image.
        classes (np.ndarray): Array of labels/classes.
        correspondence (dict, optional): Dictionary that maps classes (as str) to a
                                         unique number. This is the mapping present
                                         in the raw ground truth images. Defaults
                                         to None.

    Returns:
        np.ndarray: Correct ground truth image as a numpy array.
    """
    if correspondence is None:  # Default correspondence
        correspondence = {
            "pearlite": 0,
            "ferrite": 1,
            "recrystallized ferrite": 2,
            "proeutectoid ferrite": 3,
        }

    idx = np.nonzero(np.array(list(correspondence.keys())) == classes[:, None])[1]
    # Dictionary that matches classes (as integers) -> correspondence.values()
    # Example:
    # >> classes = ["pearlite", "proeutectoid ferrite"]
    # >> range(len(classes)) = [0, 1]; idx = [0, 3]
    # >> d = {0: 0, 1: 3}
    d = dict(zip(idx, range(len(classes))))
    indexer = np.array(
        [
            d.get(i, -1)
            for i in range(ground_truth_img.min(), ground_truth_img.max() + 1)
        ]
    )
    print(np.unique(ground_truth_img))
    return indexer[(ground_truth_img - ground_truth_img.min())].astype(int)


def plot_image_with_ground_truth(
    name: str, ground_truth: dict, src: str, dpi: int = 80, alpha: int = 0.5,
) -> None:
    """Plots an image given its name with its ground truth segmentation as an overlay.

    Args:
        name (str): Image filename.
        ground_truth (dict): Dictionary of ground truth images.
        src (str, optional): Source; where to look for images. Defaults to PATH_LABELED.
        dpi (int, optional): DPI for the plotted figure. Defaults to 80.
        alpha (int, optional): Alpha overlay. Defaults to 0.5.
    """
    plt.figure(dpi=dpi)
    plt.imshow(load_img(find_path_of_img(name, src)), cmap="gray")
    plt.imshow(ground_truth[name], alpha=alpha)
    plt.pause(0.05)
    # plt.show()
    # plt.close()


def segmentation_metrics(
    imgs: list,
    ground_truth: dict,
    classes: np.ndarray,
    T: np.ndarray,
    algorithm: str,
    algorithm_parameters: tuple,
    filterbank_name: str,
    max_test_number: int = -1,
) -> np.ndarray:
    """Obtains the confusion matrix for segmentation. Essentially, this method compares
    the segmentation of images in training, development or test set with the provided
    ground-truth.

    Args:
        imgs (list): List of images to segment and compare with the respective ground
                     truth.
        ground_truth (dict): Dictionary of ground truth images.
        classes (np.ndarray): Array of labels/classes.
        T (np.ndarray): Texton matrix.
        n (int): Maximum number of superpixels to generate.
        sigma (int): SLIC algorithm parameter.
        compactness (int): SLIC algorithm parameter.
        max_test_number (int, optional): Maximum number of labeled windows to take into
                                         account for evaluating classification
                                         performance. If set to -1, all labeled windows
                                         are taken. Defaults to -1.

    Returns:
        np.ndarray: Segmentation confusion matrix.
    """
    matrix = None
    jaccard_per_img = {}
    for k, img_tuple in enumerate(imgs):
        name, img = img_tuple
        if k == len(imgs) - 1:
            bullet = " └── "
            branch = "   "
        else:
            bullet = " ├── "
            branch = " │ "
        print(f" │\t{bullet}Segmenting image {k+1} ({name})... ", end="")
        _, segmented_img_as_matrix, _, _ = segment(
            img,
            classes,
            T,
            algorithm,
            algorithm_parameters,
            filterbank_name=filterbank_name,
        )

        y_pred = segmented_img_as_matrix.ravel()

        print("Done")
        y_true = ground_truth[name].ravel()

        if matrix is None:
            print(f" │\t{branch}   ├── Calculating confusion matrix... ", end="")
            matrix = ConfusionMatrix(y_true, y_pred, transpose=True)
        else:
            print(
                f" │\t{branch}   ├── Combining confusion matrix with previous one... ",
                end="",
            )
            matrix = matrix.combine(ConfusionMatrix(y_true, y_pred, transpose=True))

        print("Done")
        print(
            f" │\t{branch}   └── Calculating Jaccard Index for this image... ", end=""
        )
        jaccard_per_img[name] = jaccard_index_from_ground_truth(
            segmented_img_as_matrix, ground_truth[name], classes
        )
        print("Done")
        if k == max_test_number:
            break

    return matrix, matrix.to_array(normalized=True), jaccard_per_img


def evaluate_segmentation_performance(
    imgs: list,
    ground_truth: dict,
    classes: np.ndarray,
    K: int,
    T: np.ndarray,
    algorithm: str,
    algorithm_parameters: str,
    filterbank_name: str,
    save_png: bool,
    save_xlsx: bool,
    dpi: int = 120,
    max_test_number: int = -1,
    png_title: str = None,
) -> None:
    """Evaluates the model's segmentation performance by creating the corresponding
    segmentation confusion matrices for training, development and test set (if they
    are requested) and provides the option to plot them, save them as a .png and
    .xlsx file.

    Args:
        imgs (list): List of images to segment and compare with the respective ground
                     truth.
        ground_truth (dict): Dictionary of ground truth images.
        classes (np.ndarray): Array of labels/classes.
        K (int): K-Means algorithm parameter.
        T (np.ndarray): Texton matrix.
        n (int): Maximum number of superpixels to generate.
        sigma (int): SLIC algorithm parameter.
        compactness (int): SLIC algorithm parameter.
        plot_fig (bool, optional): Specifies whether to plot the generated figures.
                                   Defaults to True.
        save_png (bool, optional): Specifies whether to save the generated figures to
                                   the filesystem as .png files. Defaults to True.
        save_xlsx (bool, optional): Specifies whether to save the generated figures to
                                    the filesystem as a .xlsx file. Defaults to True.
        max_test_number (int, optional): [description]. Defaults to -1.
    """
    print("\n[+] Computing segmentation performance... ")
    cm, cm_array, jaccard_per_img = segmentation_metrics(
        imgs,
        ground_truth,
        classes,
        T,
        algorithm,
        algorithm_parameters,
        filterbank_name=filterbank_name,
        max_test_number=max_test_number,
    )
    # Maps integers classes to actual names of classes.
    # Example: {'0': 'proeutectoid ferrite', '1': 'pearlite'}
    try:
        integer_classes = [i for i in range(len(classes))]
        mapping = dict(zip(integer_classes, classes))
        cm.relabel(mapping=mapping)
    except:
        try:
            integer_classes = [str(i) for i in range(len(classes))]
            mapping = dict(zip(integer_classes, classes))
            cm.relabel(mapping=mapping)
        except:
            pass

    if png_title is not None:
        title = png_title
    else:
        title = f"Confusion matrix (segmentation), K = {K}"

    plot_confusion_matrix(
        cm, normalized=True, title=title, dpi=dpi, save_png=save_png,
    )
    if save_xlsx:
        print(" ├── Exporting to excel... ", end="")
        matrix_to_excel(
            cm_array, classes.tolist(), sheetname=f"K = {K}", filename="Segmentation",
        )
        print("Done")
    print(" └── Computing metrics... ", end="")
    stats = statistics_from_matrix(cm)
    print("Done\n")
    return stats, jaccard_per_img


def save_labeled_imgs_to_pdf(
    windows_per_name: dict, classes: np.ndarray, micrographs: dict, index_to_name: dict
) -> None:
    widths = [30, 140]
    form = "{row[0]:<{width[0]}} {row[1]:<{width[1]}}"

    wanted_titles = [
        "Material name",
        "Material type",
        "Composition",
        "Condition",
        "Condition details",
        "Description",
    ]

    data = {}
    with open("data.json", "r") as f:
        data = json.load(f)

    color_names = [
        "red",
        "blue",
        "yellow",
        "orange",
        "darkkhaki",
        "purple",
        "green",
        "turquoise",
        "darkred",
        "dodgerblue",
        "magenta",
        "fuchsia",
    ]

    colors = {
        class_: mpl.colors.to_rgb(color_names[i]) for i, class_ in enumerate(classes)
    }

    colors_255 = colors.copy()
    for label in colors_255:
        colors_255[label] = tuple(255 * np.array(colors_255[label]))
        colors_255[label] = [int(value) for value in colors_255[label]]

    with PdfPages("foo.pdf") as pdf:
        for name in windows_per_name:
            print(f"[+] Saving {name}... ", end="")
            gray_img = get_array_of_micrograph(name, micrographs, index_to_name)
            img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            present_labels = []
            for coords, _, label in windows_per_name[name]:
                top_left = coords[0]
                bottom_right = coords[1]
                cv2.rectangle(img, top_left, bottom_right, colors_255[label], 2)
                if label not in present_labels:
                    present_labels.append(label)

            fig, ax = plt.subplots(1, figsize=(15, 10))

            present_colors = [colors[present_label] for present_label in present_labels]
            C_p = len(present_labels)
            cmap = mpl.colors.ListedColormap(present_colors)
            bounds = np.arange(C_p + 1) - 0.5
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            t = ax.imshow(img, cmap=cmap, norm=norm)
            cb = fig.colorbar(t, ticks=np.arange(C_p))
            cb.ax.set_yticklabels(present_labels)

            x = -0.95
            y = -0.55
            fontsize = 11

            fig.tight_layout()  # pad=pad)

            titles = [title for title in wanted_titles if title in data[name[:-4]]]

            blobs = [data[name[:-4]][title] for title in titles]
            ax.text(
                x,
                y,
                formatter(form, widths, titles, blobs),
                fontsize=fontsize,
                transform=ax.transAxes,
            )
            ax.set_title(name[:-4] + ", " + data[name[:-4]]["Material name"])
            ax.axis("off")

            print("Done")
            # plt.show()
            pdf.savefig(fig, bbox_inches=0, dpi=500)
            plt.close()
