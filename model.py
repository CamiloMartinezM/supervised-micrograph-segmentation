# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:59:47 2020

@author: Camilo Martínez
"""
import numpy_indexed as npi
import matplotlib.cm
from itertools import chain
import os
import warnings
from collections import Counter
from random import randint, shuffle
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cuml import KMeans as CumlKMeans
from cuml.metrics.cluster.entropy import cython_entropy
from numba.core.errors import NumbaWarning
from skimage.segmentation import mark_boundaries
from utils_functions import (
    find_path_of_img,
    get_folder,
    print_table_from_dict,
    load_img,
    np2cudf,
    plot_confusion_matrix,
    matrix_to_excel,
)
from utils_classes import Scaler, FilterBankMR8, SLICSegmentation

warnings.simplefilter("ignore", category=NumbaWarning)

"""Directorios de interés

Las siguientes constantes representan directorios de interés. La carpeta donde están las
micrografías anotadas y aquella donde están la totalidad de las micrografías 
preprocesadas. Esta última tendrá para cada una de las micrografías dos imágenes: la 
primera representa la micrografía sin la escala, que generalmente aparece en la parte 
inferior, y la segunda es precisamente dicha parte inferior que posee la información de
la escala.
"""
LABELED = "Anotadas"
PREPROCESSED = "Preprocesadas"

SRC = ""
PATH_LABELED = os.path.join(SRC, LABELED)
PATH_PREPROCESSED = os.path.join(SRC, PREPROCESSED)


def preprocess_with_clahe(src: str) -> None:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print("\n[+] Working on " + src + "\n")
    for path, _, files in os.walk(src):
        for f in files:
            if f.endswith(".png") and not f.startswith("SCALE"):
                print("[+] Preprocessing " + str(f) + "... ", end="")
                folder = get_folder(path)
                if folder is not None:
                    original_path = find_path_of_img(f, src)
                    img = cv2.imread(original_path, 0)
                    final_img = clahe.apply(img)
                    cv2.imwrite(original_path, final_img)
                    print("Done")
                else:
                    print("Failed. Got None as folder.")


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


def load_imgs(as_255: bool, include_only: list = None) -> tuple:
    """Loads images in LABELED in a numpy array.

    Args:
        as_255 (bool): Specifies whether to load the images as int (0-255).
        include_only (list, optional): List of folders to take into account in case 
                                        PATH_LABELED contains unwanted folders. Defaults
                                        to None.
    Returns:
        tuple: Numpy array with all images, dictionary whose keys are names of
                images and values are the corresponding indeces in the numpy 
                array of images.
    """
    print("\n[*] IMAGES LOADING:\n")
    m = []
    index_to_name = {}  # Every name will have its corresponding position in m.
    count = 0
    for folder in os.listdir(PATH_LABELED):
        if os.path.isdir(os.path.join(PATH_LABELED, folder)):
            if include_only is None or folder in include_only:
                print(f"[?] Currently reading folder: {folder}")
                for f in os.listdir(os.path.join(PATH_LABELED, folder)):
                    if f.endswith(".png"):
                        print(f"\t Reading and loading {f}... ", end="")
                        img = load_img(os.path.join(PATH_LABELED, folder, f), as_255)
                        m.append(img)
                        index_to_name[f] = count
                        count += 1
                        print("Done")

    return np.array(m), index_to_name


def filterbank_example(img: str = "cs0328.png", dpi: int = 80) -> None:
    MR8 = FilterBankMR8([1, 2, 4], 6)  # MR8 Filter bank
    print("Filters (RFS Filter Bank):")
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


def get_array_of_micrograph(
    name: str, micrographs: dict, index_to_name: dict
) -> np.ndarray:
    """Gets the numpy array of an image with the given name.

    Args:
        name (str): Name of image.

    Returns:
        np.ndarray: Numpy array of image.
    """
    return micrographs[index_to_name[name]]


def extract_labeled_windows(
    micrographs: dict, index_to_name: dict, include_only: list = None
) -> tuple:
    """
    Args:
        micrographs (dict): Numpy array of all images.
        index_to_name (dict): Dictionary whose keys are names of images and values are
                                the corresponding indeces in the numpy array of images.
        include_only (list, optional): List of folders to take into account in case 
                                        PATH_LABELED contains unwanted folders. Defaults
                                        to None.
    Returns:
        tuple: dictionary of label counts; dictionary of windows whose keys are the 
                labels and values are a list of numpy arrays, which are the windows 
                associated with the label; and dictionary of windows and respective 
                labels per loaded image.
    """

    def check_label(label: str) -> str:
        translation_dictionary = {
            "perlita": "pearlite",
            "ferrita": "ferrite",
            "ferrita proeutectoide": "proeutectoid ferrite",
            "cementita proeutectoide": "proeutectoid cementite",
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
            if include_only is None or folder in include_only:
                print(f"[?] Currently reading folder: {folder}")
                for f in os.listdir(os.path.join(PATH_LABELED, folder)):
                    if f.endswith(".txt"):
                        img_name = f[:-4] + ".png"
                        print(f"\t Getting windows of {img_name}... ", end="")
                        # Loads full img from micrographs array
                        full_img = get_array_of_micrograph(
                            img_name, micrographs, index_to_name
                        )

                        with open(
                            os.path.join(PATH_LABELED, folder, f), "r"
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
                                    windows_per_label[label].append(window)


                                # print("Done")
                                line = annotations.readline()

                        print("Done")

    print_table_from_dict(
        labels, cols=["Label", "Number"], title="Number of windows per label",
    )

    return labels, windows_per_label, windows_per_name


def get_response_vector(img: np.ndarray) -> np.ndarray:
    """Convolves the input image with the MR8 Filter Bank to get its response as a 
    numpy array.

    Args:
        img (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Numpy array of shape (*img.shape, 8).
    """
    MR8 = FilterBankMR8([1, 2, 4], 6)  # MR8 Filter bank

    # 8 responses from image
    r = MR8.response(img)

    # Every response is stacked on top of the other in a single matrix whose last axis
    # has dimension 8. That means, there is now only one response image, in which each
    # channel contains the information of each of the 8 responses.
    response = np.concatenate(
        [np.expand_dims(r[i], axis=-1) for i in range(len(r))], axis=-1
    )
    assert response.shape == (*r[0].shape, 8)
    return response


def get_feature_vector_of_window(window: np.ndarray, ravel: bool = False) -> tuple:
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
    response_img = get_response_vector(window)
    num_pixels = window.size

    if ravel:
        return (
            response_img.reshape((window.size, response_img.shape[-1])),
            num_pixels,
        )
    else:
        return response_img, num_pixels


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


def get_feature_vectors_of_labels(windows: dict, verbose: bool = True) -> dict:
    """Obtains the feature vectors associated with each label/class.

    Args:
        windows (dict): Dictionary of windows per label.

    Returns:
        dict: dictionary of feature vectors per label. Keys corresponds to labels and 
                values are the feature vectors of that label.
    """
    feature_vectors_of_label = {}
    for label in windows:
        responses = []
        print(f"[?] Working on label: {label}...")
        num_pixels = 0
        # Every annotated window/every window of a label
        for i, window in enumerate(windows[label]):
            if verbose:
                print(f"\t  Calculating response {i+1}... ", end="")

            response, current_num_pixels = get_feature_vector_of_window(window)
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
        assert feature_vector.shape == (num_pixels, 8)
        feature_vectors_of_label[label] = feature_vector
        print("Done")

    print("")
    return feature_vectors_of_label


def train(
    K: int,
    windows_train: dict,
    windows_dev: dict = None,
    feature_vectors: dict = None,
    compute_clustering_entropy: bool = False,
    verbose: bool = True,
) -> tuple:
    """Trains the model by setting K equal to the number of clusters to be learned in 
    K-means, i.e, the number of textons.

    Args:
        K (int): K-means algorithm parameter. 
        windows_train (dict): Training set.
        windows_dev (dict, optional): Development set. If it is not None, it is included
                                        on training. Defaults to None.
        feature_vectors (dict, optional): Precomputed feature vectors of labels to use.
                                            Defaults to None.
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
    if feature_vectors is None:
        feature_vectors = get_feature_vectors_of_labels(windows_train, verbose=verbose)

    if windows_dev is not None:
        windows_to_train_on = {}
        for k, v in chain(windows_train.items(), windows_dev.items()):
            windows_to_train_on.setdefault(k, []).extend(v)

        windows_train = windows_to_train_on
        feature_vectors = get_feature_vectors_of_labels(windows_train, verbose=verbose)

    classes = np.asarray(list(feature_vectors.keys()))  # Number of classes/labels
    C = len(classes)

    print_table_from_dict(feature_vectors, cols=["Label", "Shape of feature vector"])

    textons = {}
    clustering_entropy = {}
    for label in feature_vectors:
        print(f"[?] Computing K-means on feature vector of label: {label}... ")

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
    T = np.zeros((C, K, 8), dtype=np.float64)
    for i, label in enumerate(classes):
        T[i] = textons[label].cluster_centers_

    return feature_vectors, classes, T, clustering_entropy


def get_closest_texton_vector(feature_vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Obtains a vector whose values are the minimum distances of each pixel of a
    superpixel. For example, if a superpixel has 300 pixels, this function returns a 
    (300,) vector, where each value is the minimum distance of an enumerated pixel.

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


def get_distance_matrix(feature_vectors: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Obtains a matrix which has the information of all possible distances from a pixel
    of a superpixel to every texton of every class.

    Args:
        feature_vectors (np.ndarray): Feature vectors of the superpixel.

    Returns:
        np.ndarray: Matrix of shape (10, NUM_PIXELS, K). Every (i, j, k) matrix value
                    corresponds to the distance from the i-th pixel to the k-th texton 
                    of the j-th class.
    """
    return np.linalg.norm(feature_vectors[:, np.newaxis] - T[:, np.newaxis, :], axis=-1)


def texton_matrix_from_known_classes(
    classes: list, known_classes: list, T: np.ndarray
) -> tuple:
    return T[npi.indices(classes, np.array(known_classes))]


def predict_class_of(
    feature_vectors: np.ndarray, T: np.ndarray, classes: list, known_classes: list = []
) -> str:
    """Predicts the class/label given the feature vectors that describe an image or a
    window of an image (like a superpixel).

    Args:
        feature_vectors (np.ndarray): Feature vectors of the image or window.

    Returns:
        str: Predicted class.
    """
    if len(known_classes) > 1:
        T_1 = texton_matrix_from_known_classes(known_classes)
    else:
        T_1 = T

    # Distance matrices.
    minimum_distance_vector = get_closest_texton_vector(feature_vectors, T_1)
    distance_matrix = get_distance_matrix(feature_vectors, T_1)

    # Matrix which correlates texture texton distances and minimum distances of every
    # pixel.
    A = np.sum(
        np.isclose(minimum_distance_vector.T, distance_matrix, rtol=1e-09), axis=-1
    )
    A_i = A.sum(axis=1)  # Sum over rows (i.e, over all pixels).
    ci = A_i.argmax(axis=0)  # Class with maximum probability of occurrence is chosen.

    if len(known_classes) > 1:  # Assigned class is returned.
        return known_classes[ci]
    else:
        return classes[ci]


def segment(
    img_name: str,
    classes: list,
    T: np.ndarray,
    n_segments: int = 500,
    sigma: int = 5,
    compactness: float = 0.1,
    known_classes: list = [],
    plot_original: bool = False,
    plot_superpixels: bool = False,
    verbose: bool = False,
) -> tuple:
    """Segments an image. The model must have been trained before.

    Args:
        img_name (str): Name of image to be segmented.
        n_segments (int, optional): Maximum number of superpixels to generate. Defaults 
                                     to 500.
        sigma (int, optional): SLIC algorithm parameter. Defaults to 5.
        compactness (int, optional): SLIC algorithm parameter. Defaults to 0.1.
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
    responses, _ = get_feature_vector_of_window(test_img)
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
        S_segmented[superpixel] = predict_class_of(
            current_feature_vectors, T, classes, known_classes
        )

        if verbose:  # In case additional prints are desired.
            if superpixel % 5 == 0:
                num_pixels = current_feature_vectors.shape[0]
                print(
                    f"Superpixel {superpixel} (pixels={num_pixels}) assigned to "
                    + S_segmented[superpixel]
                )

    return test_img, S, S_segmented


def segmentation_to_class_matrix(
    classes: list, S: dict, S_segmented: dict, shape: tuple
) -> np.ndarray:
    """Obtains a matrix of the given shape where each pixel of position i, j corresponds
    to the predicted class of that pixel.

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
            index = np.where(classes == S_segmented[superpixel])[0][0]
            class_matrix[i, j] = classes[index]

    return class_matrix


def classification_confusion_matrix(
    classes: list, windows: dict, T: np.ndarray, max_test_number: int = -1
) -> np.ndarray:
    C = len(classes)
    confusion_matrix = np.zeros((C, C))
    for i, label in enumerate(classes):
        for window in windows[label]:
            current_feature_vectors, _ = get_feature_vector_of_window(
                window, ravel=True
            )
            predicted_class = predict_class_of(current_feature_vectors, T, classes)
            predicted_class_index = np.where(classes == predicted_class)[0][0]
            confusion_matrix[i, predicted_class_index] += 1

        if i == max_test_number:
            break

    return confusion_matrix


def segmentation_confusion_matrix(
    classes: list,
    windows_per_name: dict,
    T: np.ndarray,
    src: str = PATH_LABELED,
    max_test_number: int = -1,
    n: int = 500,
    compactness: float = 0.1,
) -> np.ndarray:
    C = len(classes)
    shuffled_windows_per_name = list(windows_per_name.keys())
    shuffle(shuffled_windows_per_name)
    matrix = np.zeros((C, C))
    for k, name in enumerate(shuffled_windows_per_name):
        print(f"\t[?] Segmenting {name}... ", end="")
        original_img, superpixels, segmented_img = segment(
            find_path_of_img(name, src=src),
            classes,
            T,
            n_segments=n,
            compactness=compactness,
        )
        class_matrix = segmentation_to_class_matrix(
            classes, superpixels, segmented_img, original_img.shape
        )
        print("Done")
        print("\t    > Evaluating results... ", end="")
        for coords, _, true_class in windows_per_name[name]:
            i = np.where(classes == true_class)[0][0]
            segmented_window = slice_by_corner_coords(class_matrix, *coords)
            class_counts = dict(
                zip(*np.unique(segmented_window.ravel(), return_counts=True))
            )
            for predicted_class, count in class_counts.items():
                j = np.where(classes == predicted_class)[0][0]
                matrix[i, j] = count
        print("Done")

        if k == max_test_number:
            break

    return matrix

def evaluate_segmentation_performance(
    classes: list,
    K: int,
    T: np.ndarray,
    windows_per_name,
    plot: bool = True,
    savefig: bool = True,
    max_test_number: int = -1,
    n: int = 500,
    compactness: float = 0.1,
) -> None:
    print("\n[*] SEGMENTATION PERFORMANCE:")
    print("\n[+] Computing segmentation performance... ")
    confusion_matrix = segmentation_confusion_matrix(
        classes,
        windows_per_name,
        T,
        max_test_number=max_test_number,
        n=n,
        compactness=compactness,
    )
    print("Done")
    filename = f"(segmentation) K={K}"
    plot_confusion_matrix(
        confusion_matrix,
        classes,
        title=f"K = {K}",
        distinguishable_title=filename,
        savefig=savefig,
        showfig=plot,
        format_percentage=True,
    )
    print(" > Exporting to excel... ", end="")
    matrix_to_excel(
        confusion_matrix,
        classes.tolist(),
        sheetname=f"K = {K}",
        filename="Segmentation",
    )
    print("Done")


def evaluate_classification_performance(
    classes: list,
    K: int,
    T: np.ndarray,
    windows_train: dict = None,
    windows_dev: dict = None,
    windows_test: dict = None,
    plot: bool = True,
    savefig: bool = True,
    max_test_number: int = -1,
) -> None:
    def helper(windows: dict, category: str, img_filename: str, excel_filename: str):
        print(
            f"[+] Computing classification performance on {category} set... ", end="",
        )
        confusion_matrix = classification_confusion_matrix(
            classes, windows, T, max_test_number,
        )
        print("Done")
        plot_confusion_matrix(
            confusion_matrix,
            classes,
            title=f"K = {K}",
            distinguishable_title=img_filename,
            savefig=savefig,
            showfig=plot,
            format_percentage=True,
        )
        print(" > Exporting to excel... ", end="")
        matrix_to_excel(
            confusion_matrix,
            classes.tolist(),
            sheetname=f"K = {K}",
            filename=excel_filename,
        )
        print("Done")

    print("\n[*] CLASSIFICATION PERFORMANCE:\n")
    constant_title = f"(classification) K={K}"
    if windows_train is not None:
        helper(windows_train, "Training", f"Train, {constant_title}", "Train")
    if windows_dev is not None:
        helper(windows_dev, "Development", f"Dev, {constant_title}", "Dev")
    if windows_test is not None:
        helper(windows_test, "Testing", f"Test, {constant_title}", "Test")


def visualize_segmentation(
    classes: list, original_img: np.ndarray, S: dict, S_segmented: dict
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
            new_segments[i, j] = np.where(classes == S_segmented[superpixel])[
                0
            ].flatten()
            overlay[i, j] = colour_dict[classes[new_segments[i, j]]]

    present_colours = [colour_dict[present_class] for present_class in present_classes]
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
