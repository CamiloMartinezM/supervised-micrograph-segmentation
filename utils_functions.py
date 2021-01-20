# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:11:12 2020

@author: Camilo Martínez
"""
import itertools
import math
import os
import pickle
import random
import textwrap
from pprint import pprint

import cudf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycm
from numpy.fft import fft2, fftshift
from openpyxl import load_workbook
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from skimage import color, io
from sklearn.metrics import jaccard_score


def train_dev_test_split(
    data: dict, train_size: float = 0.7, dev_size: float = 0.2
) -> tuple:
    """Splits the given data into three sets (train, development and test set).

    Args:
        data (dict): Complete dataset as a dictionary whose keys are labels and values
                     are the corresponding annotated image windows as numpy arrays.
                     Values are of the form (filename, window).
        train_size (float): Percentage of the dataset that will be used as training set.
        dev_size (float): Percentage of the dataset that will be used as development
                          set.

    Returns:
        tuple: Train, dev and test sets as dictionaries of the same structure as the
               given dataset.

    """
    if train_size + dev_size > 1:
        raise ValueError("Invalid train and/or dev ratios.")

    # Extract unique names in values of data
    filenames = []
    for l in data.values():
        for (filename, _) in l:
            if filename not in filenames:
                filenames.append(filename)

    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)  # make sure the same split is obtained every time the code is run
    random.shuffle(filenames)

    split_1 = int(train_size * len(filenames))
    split_2 = int((train_size + dev_size) * len(filenames))
    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    train_windows = {}
    dev_windows = {}
    test_windows = {}

    for label, windows_list in data.items():
        for (filename, window) in windows_list:
            if filename in train_filenames:
                if label not in train_windows:
                    train_windows[label] = []
                train_windows[label].append(window)
            elif filename in dev_filenames:
                if label not in dev_windows:
                    dev_windows[label] = []
                dev_windows[label].append(window)
            elif filename in test_filenames:
                if label not in test_windows:
                    test_windows[label] = []
                test_windows[label].append(window)
            else:
                raise Exception(f"{filename} is not in split")

    return (
        train_filenames,
        dev_filenames,
        test_filenames,
        train_windows,
        dev_windows,
        test_windows,
    )


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Converts an image from RGB to gray.

    Args:
        rgb (np.ndarray): RGB image as a numpy array.

    Returns:
        np.ndarray: Image in grayscale as a numpy array.
    """
    if len(rgb.shape) != 3:
        return rgb
    else:  # len(rgb.shape) == 3
        if rgb.shape[-1] == 4:
            return color.rgb2gray(color.rgba2rgb(rgb))
        else:
            return color.rgb2gray(rgb)


def load_img(path: str, as_255: bool = False, with_io: bool = False) -> np.ndarray:
    """Loads the image in a numpy.ndarray and converts it to gray scale if possible.

    Args:
        path (str): Path of the image to be loaded.

    Returns:
        np.ndarray: Image as a numpy array.
    """
    if with_io:
        img = io.imread(path)
    else:
        img = mpimg.imread(path)

    gray = rgb2gray(img)

    if as_255:
        return (np.floor(gray * 255)).astype(np.uint8)

    return gray


def get_folder(path: str) -> str:
    """Gets the name of the folder (High carbon, Medium carbon, Low carbon) of
        a micrograph, whose path is given.

    Args:
        path (str): Path of the micrograph.

    Returns:
        str: "High carbon", "Medium carbon" or "Low carbon".
    """
    possible_folders = [
        "High carbon",
        "Medium carbon",
        "Hypoeutectoid steel",
        "Low carbon",
    ]
    for possible_folder in possible_folders:
        if possible_folder in path:
            return possible_folder
    return None


def create_folder(name: str) -> None:
    """Creates a folder if it doesn't exist.

    Args:
        name (str): Name of the folder to be created.
    """
    if not os.path.isdir(name):
        os.mkdir(name)


def find_path_of_img(name: str, src: str, relative_path: bool = False) -> str:
    """Finds the path of an image given its name.

    Args:
        name (str): Name of the image.
        src (str): Source; where to start looking.

    Returns:
        str: Path of the image file.
    """
    for root, _, files in os.walk(src):
        for filename in files:
            if filename == name:
                if relative_path:
                    return os.path.join(
                        src, os.path.relpath(os.path.join(root, filename), start=src)
                    )
                else:
                    return os.path.join(root, filename)
    return None


def print_table_from_dict(
    data: dict, cols: list, title: str = "", format_as_percentage: list = None
) -> None:
    """Prints a table from a dictionary.

    Args:
        data (dict): Data to be tabulated.
        cols (list): Columns (len must be 2).
        title (str, optional): Title to put above table. Defaults to "".
    """
    print("")

    table = PrettyTable(cols)
    table.align[cols[0]] = "l"
    table.align[cols[1]] = "r"

    characteristic_value = list(data.values())[0]

    if type(characteristic_value) is np.ndarray:
        for label in sorted(data.keys(), key=lambda x: data[x].shape[0], reverse=True,):
            table.add_row([label, f"{data[label].shape}"])
    elif type(characteristic_value) is list:
        for label in sorted(data.keys(), key=lambda x: len(data[x]), reverse=True,):
            table.add_row([label, f"{len(data[label])}"])
    else:  # int
        for label in data.keys():
            row = [label]
            if type(data[label]) is dict:
                for k, nested_label in enumerate(data[label]):
                    if (
                        format_as_percentage is not None
                        and (k + 1) in format_as_percentage
                    ):
                        row.append("{:.2%}".format(data[label][nested_label]))
                    else:
                        row.append(f"{round(data[label][nested_label], 2)}")
                table.add_row(row)
            else:
                if format_as_percentage is not None:
                    table.add_row([label, "{:.2%}".format(data[label])])
                else:
                    table.add_row([label, f"{data[label]}"])

    print(table.get_string(title=title))


def train_dev_test_split_table(train: dict, dev: dict, test: dict) -> None:
    """Prints a table which shows the size of the train, dev and test set per label.

    Args:
        train (dict): Training set.
        dev (dict): Development set.
        test (dict): Testing set.
    """
    table = PrettyTable(["Label", "Train", "Dev", "Test", "Total"])

    table.align["Label"] = "l"

    for label in sorted(
        train.keys(),
        key=lambda x: len(train.get(x, []))
        + len(dev.get(x, []))
        + len(test.get(x, [])),
        reverse=True,
    ):
        table.add_row(
            [
                label,
                len(train.get(label, [])),
                len(dev.get(label, [])),
                len(test.get(label, [])),
                len(train.get(label, []))
                + len(dev.get(label, []))
                + len(test.get(label, [])),
            ]
        )

    print(table.get_string(title="Train/dev/test split") + "\n")


def plot_confusion_matrix(
    matrix: pycm.ConfusionMatrix,
    normalized: bool = True,
    title: str = "Confusion matrix",
    dpi: int = 120,
    save_png: bool = True,
    show: bool = False
) -> None:
    """Plots a visual representation of a confusion matrix.
    Original: https://stackoverflow.com/questions/35585069/python-tabulating-confusion-matrix
    Args:
        matrix (pycm.ConfusionMatrix): Pycm confusion matrix object.
                             are predicted classes.
        title (str, optional): Distinguishable title to add to the name of the file
                               to which the plot would be saved. Defaults to None.
        dpi (int, optional): DPI of plot. Defaults to 120.
        save_png (bool, optional): Specifies whether to save the figure in the current
                                    directory or not. Defaults to True.
                                    to True.
    """
    fig = matrix.plot(
        cmap=plt.cm.Reds, number_label=True, normalized=normalized, one_vs_all=True
    )
    if show:
        plt.pause(0.05)
    else:
        plt.close()
    # plt.show()
    # plt.close()

    if save_png:
        fig.figure.savefig(title + ".png", bbox_inches="tight", dpi=dpi)


def statistics_from_matrix(matrix: pycm.ConfusionMatrix) -> dict:
    return {
        "Overall Statistics": {
            "Standard Error": matrix.SE,
            "Overall Accuracy": matrix.Overall_ACC,
            "F1 Macro": matrix.F1_Macro,
            "Accuracy Macro": matrix.ACC_Macro,
            "Overall Jaccard Index": matrix.Overall_J,
            "Class Balance Accuracy": matrix.CBA,
            "P-Value": matrix.PValue,
            "AUNU": matrix.AUNU,
        },
        "Class Statistics": {
            "Accuracy": matrix.ACC,
            "Error Rate": matrix.ERR,
            "Matthews correlation coefficient": matrix.MCC,
            "Jaccard Index": matrix.J,
            "Confusion Entropy": matrix.CEN,
            "Similarity index": matrix.sInd,
            "Optimized Precision": matrix.OP,
            "Averaged F1": matrix.average("F1"),
        },
        "matrix": matrix,
    }


def matrix_to_excel(
    matrix: np.ndarray,
    cols: list,
    sheetname: str,
    path: str = os.getcwd(),
    filename: str = "Test",
) -> None:
    """Exports a square matrix to an excel file. If a file with the given name already
    exists, a new sheetname is added and the file is not overwritten.

    Args:
        matrix (np.ndarray): Data as a matrix.
        cols (list): Labels of each row and column of the given matrix.
        sheetname (str): Sheetname.
        path (str, optional): Specifies where to put the excel file. Defaults to
                              os.getcwd().
        filename (str, optional): Excel filename. Defaults to "Test".
    """
    filename += ".xlsx"
    df = pd.DataFrame(matrix, columns=cols)
    df["Label"] = cols
    new_cols_arrangement = ["Label"] + cols
    df = df[new_cols_arrangement]

    if os.path.isfile(os.path.join(path, filename)):
        book = load_workbook(os.path.join(path, filename))
        writer = pd.ExcelWriter(os.path.join(path, filename), engine="openpyxl")
        writer.book = book

        df.to_excel(writer, sheet_name=sheetname)
        writer.save()
        writer.close()
    else:
        with pd.ExcelWriter(f"{filename}") as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False)


def np2cudf(df: np.ndarray) -> cudf.DataFrame:
    """Convert numpy array to cuDF dataframe."""
    df = pd.DataFrame({"fea%d" % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


def compare2imgs(
    img_1: np.ndarray,
    img_2: np.ndarray,
    title_1: str = "Original",
    title_2: str = "Final",
    dpi=80,
) -> None:
    """Shows 2 images side by side.

    Args:
        img_1 (np.ndarray): Original image loaded as a numpy array.
        img_2 (np.ndarray): Final image loaded as a numpy array.
        title_1 (str, optional): Title to put above original image. Defaults to
                                 "Original".
        title_2 (str, optional): Title to put above final image. Defaults to "Final".
    """
    plt.figure(figsize=(12, 10), dpi=dpi)
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_1, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("CLAHE")
    plt.imshow(img_2, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.close()


def fullprint(*args, **kwargs) -> None:
    """Prints an array without truncation"""
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)


def nested_dicts_to_matrix(dictionary: dict) -> np.ndarray:
    """Builds a matrix out of a nested dictionary.
    Example:
    >> d = {0: {0: 3, 1: 4}, 1: {0: 0, 1: 5}}
    >> nested_dicts_to_matrix(d) -> [[3, 4], [0, 5]]

    Args:
        dictionary (dict): Nested dictionary.

    Returns:
        np.ndarray: Equivalent matrix.
    """
    matrix = []
    for row in dictionary.values():
        matrix.append(list(row.values()))

    return np.asarray(matrix)


def save_variable_to_file(
    variable: object, name: str, dst: str = os.getcwd(), overwrite: bool = False
) -> None:
    """Saves a variable to a .pickle file.

    Args:
        variable (object): Variable to save to file. It can be of any type.
        name (str): Name of variable, which will be used as the filename for the
                    .pickle file.
        dst (str, optional): Destination of file (folder). Defaults to current
                             directory.
    """
    filename = name + ".pickle"
    if filename in os.listdir(dst) and not overwrite:
        print("[WARNING] That variable appears to have been saved before.")
        action = "-1"
        while action not in ["1", "2"]:
            action = input("[?] Overwrite (1), rename automatically (2) >> ").strip()

        if action == "1":
            print("[+] Overwriting... ", end="")
            with open(os.path.join(dst, filename), "wb") as f:
                pickle.dump(variable, f)
            print("Done")
        else:  # Rename
            print("[+] Renaming... ")
            new_filename = filename.replace(".pickle", "") + "-1.pickle"
            i = 2
            while new_filename in os.listdir(
                dst
            ):  # Check if this renaming has been done before
                print(
                    "\t[+] Found " + new_filename + " in folder. Trying with ", end=""
                )
                new_filename = new_filename.split("-")[0] + "-" + str(i) + ".pickle"
                print(new_filename + "... ")
                i += 1
            print("[+] Done")

            filename = new_filename

    print(f"[+] Saving variable to {filename}... ", end="")
    with open(os.path.join(dst, filename), "wb") as f:
        pickle.dump(variable, f)
    print("Done")


def load_variable_from_file(filename: str, src: str) -> object:
    """Returns a variable loaded from a .pickle file.

    Args:
        filename (str): Name of .pickle file.
        src (str, optional): Source of file (folder). Defaults to current directory.

    Returns:
        object: Loaded variable.
    """
    if not filename.endswith(".pickle"):
        filename += ".pickle"

    with open(os.path.join(src, filename), "rb") as f:
        variable = pickle.load(f)

    return variable


def jaccard_index_from_ground_truth(
    segmented: np.ndarray, ground_truth: np.ndarray, classes: np.ndarray,
) -> float:
    """Calculates the Jaccard index of a segmented image with its corresponding ground
    truth image.

    Args:
        segmented (np.ndarray): Segmented image as a numpy array.
        ground_truth (np.ndarray): Ground truth image as a numpy array.

    Returns:
        float: Dictionary of Jaccard Index scores calculated as a micro, macro and per
               class statistic.
    """
    jaccard = {}
    for average_type in [None, "micro", "macro", "samples"]:
        if average_type is None:
            key = "Per Class"
        else:
            key = average_type[0].upper() + average_type[1:]

        try:
            jaccard[key] = jaccard_score(
                segmented.flatten(), ground_truth.flatten(), average=average_type,
            )
        except:
            continue

        if key == "Per Class":
            jaccard[key] = dict(zip(classes, jaccard[key]))

    return jaccard


def img_to_binary(img: np.ndarray) -> np.ndarray:
    """Converts an input image to binary.

    Args:
        img (np.ndarray): Input image.

    Returns
        np.ndarray: Binary image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Image as RGB
    img[np.all(img == 255, axis=2)] = 0

    kernel = np.array(
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32
    )  # Laplacian kernel
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.float32(img)
    imgResult = sharp - imgLaplacian  # New sharped image
    # Convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype("uint8")

    # Binary image (every pixel is either a 0 or a 1)
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.normalize(bw, bw, 0, 1.0, cv2.NORM_MINMAX)

    return bw


def pixel_counts_to_volume_fraction(
    pixel_counts: dict,
    pixel_length_scale: int,
    length_scale: int,
    units: str = "µm",
    img_size: tuple = (500, 500),
) -> dict:
    """Converts a dictionary of pixel counts to a dictionary of volume fractions, given
    the pixel length scale and its equivalence in µm (or any other unit).

    Args:
        pixel_counts (dict): Dictionary of pixel counts, where each key is a label and 
                             its value is the number of pixels associated with that 
                             label.
        pixel_length_scale (int): Scale length (present in any micrograph). Corresponds
                                  to the number of pixels the scale occupies in the 
                                  image.
        length_scale (int): Value in µm of the scale.
        units (str, optional): Scale unit. Defaults to "µm".
        img_size (tuple, optional): Image size. Defaults to (500, 500).

    Returns
        dict: Dictionary of volume fractions, where each key is a label and its value 
              is its volume fraction in µm² (or any other unit).

    """
    pixel_area_in_pixels = img_size[0] * img_size[1]
    pixel_area_in_length_squared = (
        pixel_area_in_pixels * ((1 / pixel_length_scale) * length_scale) ** 2
    )
    pixel_area = pixel_area_in_length_squared / pixel_area_in_pixels

    volume_fraction = {}
    for label, pixel_count in pixel_counts.items():
        volume_fraction[label] = {
            "volume fraction": pixel_count * pixel_area,
            "percentage area": pixel_count / pixel_area_in_pixels,
        }

    return volume_fraction


def highlight_class_in_img(
    img: np.ndarray, mask: np.ndarray, class_: int, fill_value: int = 0
) -> np.ndarray:
    """Highlights a class in an image. The input mask corresponds to the image 
    segmentation and the class_ is the label that will be highlighted. Thus, every pixel
    in img whose value in mask is equal to class_ is preserved. Otherwise, its value is
    replaced by fill_value.

    Example (class_ = 1, fill_value = 0):

        img = [[17, 0, 15, 19, 1, 12],
               [11, 1, 25, 2, 13, 14],
               [1, 26, 11, 1, 15, 16],
               [14, 30, 1, 12, 13, 8],
               [0, 15, 18, 14, 52, 7],
               [0, 15, 18, 14, 52, 7]]

        mask = [[1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1]]

        result = [[17,  0,  0,  0,  1, 12],
                  [11,  1, 25,  2, 13, 14],
                  [ 1, 26, 11,  1, 15, 16],
                  [14, 30,  1,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  0],
                  [ 0, 15, 18, 14, 52,  7]])


    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Segmented image.
        class_ (int): Class to highlight.
        fill_value (int): Value for unwanted pixels. Defaults to 0.

    Returns
        np.ndarray: Highlighted image.
    """
    result = img.copy()
    result[mask != class_] = fill_value
    result[mask == class_] = img[mask == class_]
    return result


def adjust_labels(segmentation_pixel_counts: dict) -> dict:
    """Label adjustment to improve Pretty Table.

    Args:
        segmentation_pixel_counts (dict): Dictionary of labels and their 
                                          corresponding pixel counts.

    Returns:
        dict: Adjusted dictionary of pixel counts.
    """
    if "proeutectoid ferrite" in segmentation_pixel_counts:
        segmentation_pixel_counts["Proeutectoid ferrite"] = segmentation_pixel_counts[
            "proeutectoid ferrite"
        ]
        del segmentation_pixel_counts["proeutectoid ferrite"]

    if "pearlite" in segmentation_pixel_counts:
        if "ferrite" in segmentation_pixel_counts:
            segmentation_pixel_counts["Pearlite (ferrite + cementite)"] = (
                segmentation_pixel_counts["pearlite"]
                + segmentation_pixel_counts["ferrite"]
            )
            segmentation_pixel_counts["> Ferrite"] = segmentation_pixel_counts[
                "ferrite"
            ]
            del segmentation_pixel_counts["ferrite"]

        segmentation_pixel_counts["> Cementite"] = segmentation_pixel_counts["pearlite"]
        del segmentation_pixel_counts["pearlite"]

    return segmentation_pixel_counts


def maxk(array: np.ndarray, k: int) -> tuple:
    """Obtains the k maximum elements in the input array. This simulates the
    maxk function in MatLab.

    Args:
        array (np.ndarray): Input array.
        k (int): Number of maximum elements to return.

    Returns:
        tuple: Indexes of maximum elements, maximum elements.
    """
    idxs_if_sorted = np.argsort(array)
    idxs_of_interest = idxs_if_sorted[-1 : -1 * k - 1 : -1]
    vals_of_interest = array[idxs_of_interest]
    return idxs_of_interest, vals_of_interest


def mink(array: np.ndarray, k: int) -> tuple:
    """Obtains the k minimum elements in the input array. This simulates the
    mink function in MatLab.

    Args:
        array (np.ndarray): Input array.
        k (int): Number of minimum elements to return.

    Returns:
        tuple: Indexes of minimum elements, minimum elements.
    """
    idxs_if_sorted = np.argsort(array)
    idxs_of_interest = idxs_if_sorted[:k]
    vals_of_interest = array[idxs_of_interest]
    return idxs_of_interest, vals_of_interest


def gauss_fit(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluates the gauss function on the input array.

    Args:
        x (np.ndarray): Input array.
        a (float): 1st parameter.
        b (float): 2nd parameter. Also known as x0.
        c (float): 3rd parameter.

    Returns:
        np.ndarray: Gauss response.
    """
    return a * np.exp(-(((x - b) / c) ** 2))


def peakpos(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Obtains the information regarding the gauss fit between input X, Y, such as
    the peak position, curve parameters and full-width at 50% maximum.

    Args:
        X (np.ndarray): x values.
        Y (np.ndarray): y values.

    Returns:
        tuple: peak position, gauss fit parameters and full width at 50% maximum.
    """
    # Gaussian fit on the whole X domain
    # Gaussian curve: Y = a1*exp(-((X-b1)/c1)^2)
    Ymax, imax = np.max(Y), np.argmax(Y)

    popt, _ = curve_fit(gauss_fit, X, Y, p0=[Ymax, X[imax], 0.5 * X[imax]], maxfev=1000)
    a, b, c = popt

    # hereafter, only the upper half of the Gaussian peak is considered
    # CutOff50 is the value of (X-b1)/c1 corresponding to the point Y=0.5*a1 of the
    # Gaussian curve
    CutOff50 = (np.log(1 / 0.5)) ** 0.5

    # final result: peak position and full-width-at-50%-maximum
    fw50m = c * 2 * CutOff50
    return b, popt, fw50m


def calculate_spacing(
    I: np.ndarray, img_name: str = "img", save_plots: bool = False, dpi: int = 120
) -> tuple:
    """Calculates the spacing between lamellae in an input image.

    Args:
        I (np.ndarray): Input image as a numpy array.
        img_name (str, optional): Image name; useful to name generated plots. Defaults 
                                  to "img".
        save_plots (bool, optional): True if plots are to be saved. Defaults to False.
        dpi (int, optional): DPI of generated plots. Defaults to 120.

    Returns:
        tuple: Spacing obtained from method 1, spacing obtained from method 2.
    """
    output1 = img_name + "_dft.png"
    output2 = img_name + "_res.png"

    # get the size MxN of the investigated picture I
    M, N = I.shape

    # DFT Calculation and magnitude

    # F is the centered discrete Fourier transform (DFT) of I
    F = fftshift(fft2(I))
    # F now becomes the magnitude of the DF
    F = abs(F)

    # plot and save F for visual examination
    # use a logarithmic scale; the maximum value (= the sum of I) is out of scale
    _, Fmax = maxk(F.flatten(), 2)
    Fmin = F.min()
    dF = (255 * np.log(F / Fmin) / np.log(Fmax[-1] / Fmin)).astype(np.uint8)

    if save_plots:
        cv2.imwrite(output1, dF)

    # Calculate and subtract the background of F

    # Calculate the histogram of the smaller 99.9 % elements of F, by using
    # ((M*N)^0.5)/4) levels (neglect the very high and rare values, such as F(0,0),
    # which would make binning difficult)
    _, xdata = mink(F.flatten(), math.floor(M * N * 0.999))
    [Hist, Edges] = np.histogram(xdata, bins=math.floor(((M * N) ** 0.5) / 4))
    Values = (Edges[1:] - (Edges[1] + Edges[0]) / 2).astype(np.int32)

    # find the maximum of the histogram and the corresponding level
    # this is the most common level in F and therefore it is the background
    background, _, fw50m = peakpos(Values, Hist)

    F = F - background

    # Calculate the total wavenumber Wn and the linear index n

    # u is the first index (or discrete coordinate) of the Fourier transform
    if M % 2 == 0:  # one column
        u = np.arange(-(M / 2), (M / 2), 1).reshape(-1, 1)  # if N is even
    else:
        u = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1, 1).reshape(-1, 1)  # if N is odd

    u = u * np.ones((1, N))  # all columns

    # v is the second index (or discrete coordinate) of the Fourier transform
    if N % 2 == 0:  # one row
        v = np.arange(-(N / 2), (N / 2), 1).reshape(1, -1)  # if M is even
    else:
        v = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1).reshape(1, -1)  # if M is odd

    v = np.ones((M, 1)) * v  #  all rows

    # W is the wavenumber of each point of the Fourier transform
    W = ((u / M) ** 2 + (v / N) ** 2) ** 0.5

    # n is the linear index, proportional to the wavenumber of each point of the Fourier
    # transform
    n = (((M * N) ** 0.5) * W).astype(np.uint16)

    # nmax is the greatest wavenumber index listed in n
    nmax = n.max()

    # Wn is the vector of wavenumbers corresponding to the indexes given in n
    Wn = np.arange(0, nmax + 1, 1).astype(np.float32) / ((M * N) ** 0.5)

    # Calculate the total spectral magnitude vs. wavenumber curve

    # calculate the total spectral magnitude, Fn, of the Fourier transform for a given
    # wavenumber
    # = the sum of the magnitude of each set of points of the Fourier transform, which
    # have the same linear index n
    # Fn = np.zeros((nmax + 1,))
    # ravel_n = n.ravel()
    # ravel_F = F.ravel()
    # for i in range(n.shape[0] * n.shape[1]):
    #     if ravel_n[i] > 0:
    #         Fn[ravel_n[i]] = Fn[ravel_n[i]] + ravel_F[i]
    Fn = np.bincount(n.ravel(), weights=F.ravel())
    Fn[0] = 0

    # find the peak of the Fn curve
    WnMax, curve_parameters, fw50m = peakpos(Wn, Fn)

    # Plot the total spectral magnitude vs. wavenumber curve
    if save_plots:
        X = np.arange(WnMax - (fw50m / 2), WnMax + (fw50m / 2), fw50m / 200)
        Y = gauss_fit(X, *curve_parameters)
        plt.figure(figsize=(10, 8))
        plt.plot(
            Wn.T, Fn, "k", X, Y, "r", [WnMax, WnMax], np.array([0, 1.2]) * max(Y), "b"
        )
        plt.axis([0, 0.75, 0, 7e7])
        plt.xlabel("Wavenumber [px^{-1}]")
        plt.ylabel("Total spectral magnitude [a.u.]")
        plt.legend(("Spectral magnitude", "Peak fitting curve", "Peak position"))
        plt.tight_layout(pad=2)
        plt.savefig(output2, bbox_inches=0, dpi=dpi)
        # plt.show()
        plt.close()

    # Calculate the pearlite spacing (1st method)

    # calculate the  wavelength corresponding to the peak of the total spectral
    # magnitude vs. wavelength curve
    spacing1 = 1 / WnMax

    # Calculate the pearlite spacing (2nd method)

    # the wavenumber is calculated as the mean of the points of W which are
    # close to WnMax (within +/-fw50m/2), weighted over their spectral magnitude F
    # determine which points should be used (also exclude W=0, i.e. the continuous
    # component)
    Use = (W > 0) * (W > WnMax - (fw50m / 2)) * (W < WnMax + (fw50m / 2))

    # calculate the mean wavenumber weighted over the spectral magnitude
    Wmean = np.sum(W * F * Use) / np.sum(F * Use)

    # calculate the corresponding wavelength
    spacing2 = 1 / Wmean

    return spacing1, spacing2


def formatter(format_str, widths, *columns):
    """
    format_str describes the format of the report.
    {row[i]} is replaced by data from the ith element of columns.

    widths is expected to be a list of integers.
    {width[i]} is replaced by the ith element of the list widths.

    All the power of Python's string format spec is available for you to use
    in format_str. You can use it to define fill characters, alignment, width,
    type, etc.

    formatter takes an arbitrary number of arguments.
    Every argument after format_str and widths should be a list of strings.
    Each list contains the data for one column of the report.

    formatter returns the report as one big string.
    """
    result = []
    for row in zip(*columns):
        lines = [textwrap.wrap(elt, width=num) for elt, num in zip(row, widths)]
        for line in itertools.zip_longest(*lines, fillvalue=""):
            result.append(format_str.format(width=widths, row=line))
    return "\n".join(result)
