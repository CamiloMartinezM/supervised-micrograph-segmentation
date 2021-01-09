# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:59:47 2020

@author: Camilo Martínez
"""
import os
import warnings
from collections import Counter
from random import randint
from itertools import chain
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cuml import KMeans as CumlKMeans
from cuml.metrics.cluster.entropy import cython_entropy
from numba.core.errors import NumbaWarning
from pycm import ConfusionMatrix
from skimage import io
from skimage.segmentation import mark_boundaries
from sklearn.cluster import MiniBatchKMeans

from utils_classes import FilterBank, Scaler, SuperpixelSegmentation

warnings.simplefilter("ignore", category=NumbaWarning)

from utils_functions import (
    find_path_of_img,
    get_folder,
    load_img,
    matrix_to_excel,
    np2cudf,
    plot_confusion_matrix,
    print_table_from_dict,
    statistics_from_matrix,
)

src = ""

labeled = "Anotadas"
preprocessed = "Preprocesadas"

# The following constants represent directories of interest.
# The folder where the annotated micrographs are and the one where all the
# pre-processed micrographs are. The latter will have two images for each
# of the micrographs: the first one represents the micrograph without the scale,
# which generally appears in the lower section of the image, and the second one
# is precisely the latter (which naturally has the information regarding the
# scale).
PATH_LABELED = os.path.join(src, labeled)
PATH_PREPROCESSED = os.path.join(src, preprocessed)


def load_scales() -> dict:
    """Loads the scale of the images by going to PREPROCESSED and finding the
    corresponding SCALE image of every image in LABELED.

    Returns:
        dict: Dictionary whose keys are names of images and values are their respective
              scale pixel length.
    """
    print("\n[*] SCALES EXTRACTION:\n")
    myScaler = Scaler(PATH_LABELED, PATH_PREPROCESSED)
    myScaler.process()
    scales = myScaler.scales

    print_table_from_dict(
        scales,
        cols=["Name of micrograph", "Pixels in scale"],
        title="Pixel length scales",
    )

    return scales


def load_imgs(exclude: list = []) -> tuple:
    """Loads images in LABELED in a numpy array.

    Args:
        exclude (list, optional): Folders to exclude in loading. Defaults to [].

    Returns:
        tuple: Numpy array with all images, dictionary whose keys are names of images and values
               are the corresponding indeces in the numpy array of images.
    """
    print("\n[*] IMAGES LOADING:\n")
    m = []
    index_to_name = {}  # Every name will have its corresponding position in m.
    count = 0
    for folder in os.listdir(PATH_LABELED):
        if os.path.isdir(os.path.join(PATH_LABELED, folder)):
            if folder in exclude:
                continue

            print(f"[?] Currently reading folder: {folder}")
            for f in os.listdir(os.path.join(PATH_LABELED, folder)):
                if f.endswith(".png"):
                    print(f"\t Reading and loading {f}... ", end="")
                    img = load_img(os.path.join(PATH_LABELED, folder, f))
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
    micrographs: dict, index_to_name: dict, exclude: list = []
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

    print("\n[*] LABELED WINDOWS EXTRACTION:\n")
    labels = {}  # Counts number of labels/annotations per label.
    windows_per_label = {}  # key = label, value = list of windows
    windows_per_name = {}  # key = filename, value = [(coords, window, label)]

    for folder in os.listdir(PATH_LABELED):
        if os.path.isdir(os.path.join(PATH_LABELED, folder)):
            if folder in exclude:
                continue

            print(f"[?] Currently reading folder: {folder}")
            for f in os.listdir(os.path.join(PATH_LABELED, folder)):
                if f.endswith(".txt"):
                    img_name = f[:-4] + ".png"
                    print(f"\t Getting windows of {img_name}... ", end="")
                    full_img = get_array_of_micrograph(
                        img_name, micrographs, index_to_name
                    )  # Loads full img from micrographs array

                    with open(
                        os.path.join(PATH_LABELED, folder, f), "r"
                    ) as annotations:
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
    img: str = "cs0328.png", dpi: int = 80, filterbank_name: str = "MR8"
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
    img = load_img(find_path_of_img(img, PATH_LABELED))
    response = MR8.response(img)

    # Original image
    print("")
    print("Original image:")
    plt.figure(dpi=dpi)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.close()

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
    plt.show()
    fig2.tight_layout()


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
    """Convolves the input image with the MR8 Filter Bank to get its response as a numpy array.

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
    window: np.ndarray, ravel: bool = False, filterbank_name: str = "MR8"
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
    response_img = get_response_vector(window, filterbank_name)
    num_pixels = response_img.size

    if ravel:
        return (
            response_img.reshape((window.size, response_img.shape[-1])),
            num_pixels,
        )
    else:
        return response_img, num_pixels


def get_feature_vectors_of_labels(
    windows: dict, verbose: bool = True, filterbank_name: str = "MR8"
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
        verbose (bool): True is additional information is needed. Defaults to True.

    Returns:
        dict: Dictionary of feature vectors per label. Keys corresponds to labels and values are
              the feature vectors of that label.
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
        # response images. The following operations convert responses_arr to a matrix where
        # each row is a pixel. That means, each row will have 8 columns associated with each
        # pixel responses.
        feature_vector = concatenate_responses(responses_arr)

        # assert feature_vector.shape == (
        #     num_pixels,
        #     8 + 4 * 3 * multiscale_statistics.scales,
        # )
        feature_vectors_of_label[label] = feature_vector
        print("Done")

    print("")
    return feature_vectors_of_label


def train(
    K: int,
    windows_train: dict,
    filterbank_name: str,
    windows_dev: dict = None,
    precomputed_feature_vectors: dict = None,
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
    print("\n[*] TRAINING:\n")

    # Feature vector extraction per label on training set
    if precomputed_feature_vectors is not None:
        feature_vectors_of_label = precomputed_feature_vectors
    else:
        if windows_dev is not None:
            windows_to_train_on = {}
            for k, v in chain(windows_train.items(), windows_dev.items()):
                windows_to_train_on.setdefault(k, []).extend(v)

            windows_train = windows_to_train_on

        feature_vectors_of_label = get_feature_vectors_of_labels(
            windows_train, verbose=verbose, filterbank_name=filterbank_name
        )

    classes = np.asarray(
        list(feature_vectors_of_label.keys())
    )  # Number of classes/labels
    C = len(classes)

    print_table_from_dict(
        feature_vectors_of_label, cols=["Label", "Shape of feature vector"]
    )

    textons = {}
    clustering_entropy = {}
    for label in feature_vectors_of_label:
        print(f"[?] Computing K-means on feature vector of label: {label}... ")
        if minibatch_size is not None:
            textons[label] = MiniBatchKMeans(n_clusters=K).fit(
                feature_vectors_of_label[label]
            )
        else:
            textons[label] = CumlKMeans(n_clusters=K, output_type="numpy").fit(
                np2cudf(feature_vectors_of_label[label])
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
    T = np.zeros(
        (C, K, feature_vectors_of_label[classes[0]].shape[-1]), dtype=np.float64
    )
    for i, label in enumerate(classes):
        T[i] = textons[label].cluster_centers_

    return feature_vectors_of_label, classes, T, clustering_entropy


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


def get_distance_matrix(feature_vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Obtains a matrix which has the information of all possible distances from a pixel of
    a superpixel to every texton of every class.

    Args:
        feature_vectors (np.ndarray): Feature vectors of the superpixel.
        T (np.ndarray): Texton matrix.

    Returns:
        np.ndarray: Matrix of shape (C, NUM_PIXELS, K). Every (i, j, k) matrix value
                    corresponds to the distance from the i-th pixel to the k-th texton of
                    the j-th class.
    """
    return np.linalg.norm(feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1)


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

    # Matrix which correlates texture texton distances and minimum distances of every pixel.
    A = np.sum(
        np.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-09), axis=-1,
    )
    A_i = A.sum(axis=1)  # Sum over rows (i.e, over all pixels).
    ci = A_i.argmax(axis=0)  # Class with maximum probability of occurrence is chosen.

    return classes[ci]  # Assigned class is returned.


def segment(
    img_name: str,
    classes: np.ndarray,
    T: np.ndarray,
    algorithm: str,
    algorithm_parameters: tuple,
    filterbank_name: str = "MR8",
    plot_original: bool = False,
    plot_superpixels: bool = False,
    verbose: bool = False,
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

    test_img = load_img(img_name, as_255=False, with_io=True)  # Image is loaded.

    # The image is segmented using the given algorithm.
    superpixel_generation_model = SuperpixelSegmentation(
        algorithm, algorithm_parameters
    )
    segments = superpixel_generation_model.segment(test_img)

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
        superpixel_generation_model.plot_output(test_img, segments)

    S = get_superpixels(segments)  # Superpixels obtained from segments.
    responses, _ = get_feature_vector_of_window(
        test_img, filterbank_name=filterbank_name
    )
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
        S_segmented[superpixel] = predict_class_of(current_feature_vectors, classes, T)

        if verbose:  # In case additional prints are desired.
            if superpixel % 25 == 0:
                num_pixels = current_feature_vectors.shape[0]
                print(
                    f"Superpixel {superpixel} (pixels={num_pixels}) assigned to "
                    + S_segmented[superpixel]
                )

    return test_img, S, S_segmented


def visualize_segmentation(
    original_img: np.ndarray,
    classes: np.ndarray,
    S: dict,
    S_segmented: dict,
    dpi: int = 120,
) -> None:
    """Plots a segmentation result on top of the original image.

    Args:
        original_img (np.ndarray): Numpy array associated with the original image.
        classes (np.ndarray): Array of classes/labels.
        S (dict): Original dictionary of superpixels obtained from SLIC.
        S_segmented (dict): Segmentation result.
        dpi (int, optional): DPI for plotted figure. Defaults to 120.
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
            new_segments[i, j] = np.where(classes == S_segmented[superpixel])[
                0
            ].flatten()
            overlay[i, j] = colour_dict[classes[new_segments[i, j]]]

    present_colours = [colour_dict[present_class] for present_class in present_classes]
    colours = mpl.colors.ListedColormap(present_colours)

    norm = mpl.colors.BoundaryNorm(np.arange(C_p + 1) - 0.5, C_p)

    plt.figure(figsize=(9, 6), dpi=dpi)
    plt.imshow(mark_boundaries(original_img, new_segments), cmap="gray")
    plt.imshow(overlay, cmap=colours, norm=norm, alpha=0.5)
    cb = plt.colorbar(ticks=np.arange(C_p))
    cb.ax.set_yticklabels(present_classes)

    plt.tight_layout(w_pad=100)
    plt.axis("off")
    plt.show()
    plt.close()


def segmentation_to_class_matrix(
    classes: np.ndarray, S: dict, S_segmented: dict, shape: tuple, as_int: bool = False
) -> np.ndarray:
    """Obtains a matrix of the given shape where each pixel of position i, j corresponds to the
    predicted class of that pixel.

    Args:
        classes (np.ndarray): Array of classes/labels.
        S (dict): Dictionary of initially extracted superpixels.
        S_segmented (dict): Segmentation result.
        shape (tuple): Image shape.
        as_int (bool): True if the desired matrix must have integers as the corresponding class.
                       Defaults to False.
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
            predicted_class = predict_class_of(current_feature_vectors, classes, T)
            predicted_class_index = np.where(classes == predicted_class)[0][0]
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
                print(" > Exporting to excel... ", end="")
                matrix_to_excel(
                    cm_array,
                    classes.tolist(),
                    sheetname=sheetname,
                    filename=excel_filename,
                )
            print("Done")
            print(" > Computing metrics... ", end="")
            stats = statistics_from_matrix(cm)
            print("Done")
            return stats

    print("\n[*] CLASSIFICATION PERFORMANCE:\n")
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
    tif_file: str, classes: np.ndarray, folder: str = "Hypoeutectoid steel"
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
    ground_truth_imgs = io.imread(tif_file)[:, 0, :, :]
    ground_truth = np.zeros(ground_truth_imgs.shape, dtype=int)
    for i in range(ground_truth_imgs.shape[0]):
        ground_truth_img = ground_truth_imgs[i, :, :].astype(int)
        ground_truth[i, :, :] = make_classes_consistent(ground_truth_img, classes)

    l = [
        f
        for f in sorted(os.listdir(os.path.join(PATH_LABELED, folder)))
        if f.endswith(".png")
    ]
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
    return indexer[(ground_truth_img - ground_truth_img.min())].astype(int)


def plot_image_with_ground_truth(
    name: str,
    ground_truth: dict,
    src: str = PATH_LABELED,
    dpi: int = 80,
    alpha: int = 0.5,
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
    plt.show()
    plt.close()


def segmentation_confusion_matrix(
    imgs: list,
    ground_truth: dict,
    classes: np.ndarray,
    T: np.ndarray,
    algorithm: str,
    algorithm_parameters: tuple,
    filterbank_name: str,
    max_test_number: int = -1,
    src: str = PATH_LABELED,
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
        src (str, optional): Source of images to segment and compare. Defaults to
                             PATH_LABELED.

    Returns:
        np.ndarray: Segmentation confusion matrix.
    """
    matrix = None
    for k, name in enumerate(imgs):
        print(f"\t[?] Segmenting {name}... ", end="")
        original_img, superpixels, segmented_img = segment(
            find_path_of_img(name, src=src),
            classes,
            T,
            algorithm,
            algorithm_parameters,
            filterbank_name=filterbank_name,
        )
        y_pred = segmentation_to_class_matrix(
            classes, superpixels, segmented_img, original_img.shape, as_int=True
        ).ravel()

        print("Done")

        y_true = ground_truth[name].ravel()

        if matrix is None:
            matrix = ConfusionMatrix(y_true, y_pred, transpose=True)
        else:
            matrix = matrix.combine(ConfusionMatrix(y_true, y_pred, transpose=True))

        if k == max_test_number:
            break

    return matrix, matrix.to_array(normalized=True)


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
    print("\n[*] SEGMENTATION PERFORMANCE:")
    print("\n[+] Computing segmentation performance... ")
    cm, cm_array = segmentation_confusion_matrix(
        imgs,
        ground_truth,
        classes,
        T,
        algorithm,
        algorithm_parameters,
        filterbank_name=filterbank_name,
        max_test_number=max_test_number,
    )
    print("Done")
    plot_confusion_matrix(
        cm,
        normalized=True,
        title=f"Confusion matrix (segmentation), K = {K}",
        dpi=dpi,
        save_png=save_png,
    )
    if save_xlsx:
        print(" > Exporting to excel... ", end="")
        matrix_to_excel(
            cm_array, classes.tolist(), sheetname=f"K = {K}", filename="Segmentation",
        )
        print("Done")
    print(" > Computing metrics... ", end="")
    stats = statistics_from_matrix(cm)
    return stats
