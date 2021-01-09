# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:11:12 2020

@author: Camilo Martínez
"""
import itertools
import pickle
import os
import random
import textwrap
from pprint import pprint
import pycm
import cudf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from prettytable import PrettyTable
from skimage import color, io


def train_dev_test_split(
    data: dict, train_size: float = 0.7, dev_size: float = 0.2
) -> tuple:
    """Splits the given data into three sets (train, development and test set).

    Args:
        data (dict): Complete dataset as a dictionary whose keys are labels and values are the
                     corresponding annotated image windows as numpy arrays. Values are of the
                     form (filename, window).
        train_size (float): Percentage of the dataset that will be used as training set.
        dev_size (float): Percentage of the dataset that will be used as development set.

    Returns:
        tuple: train, dev and test sets as dictionaries of the same structure as the given dataset.

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


def print_table_from_dict(data: dict, cols: list, title: str = "") -> None:
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
        for label in sorted(data.keys(), key=lambda x: data[x], reverse=True,):
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
        key=lambda x: len(train[x]) + len(dev[x]) + len(test[x]),
        reverse=True,
    ):
        table.add_row(
            [
                label,
                len(train[label]),
                len(dev[label]),
                len(test[label]),
                len(train[label]) + len(dev[label]) + len(test[label]),
            ]
        )

    print(table.get_string(title="Train/dev/test split") + "\n")


def plot_confusion_matrix(
    matrix: pycm.ConfusionMatrix,
    normalized: bool = True,
    title: str = "Confusion matrix",
    dpi: int = 120,
    save_png: bool = True,
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
    plt.show()
    plt.close()

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
    """Exports a square matrix to an excel file. If a file with the given name already exists, a new
    sheetname is added and the file is not overwritten.

    Args:
        matrix (np.ndarray): Data as a matrix.
        cols (list): Labels of each row and column of the given matrix.
        sheetname (str): Sheetname.
        path (str, optional): Specifies where to put the excel file. Defaults to os.getcwd().
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


def save_variable_to_file(variable: object, name: str, dst: str = os.getcwd()) -> None:
    """Saves a variable to a .pickle file.

    Args:
        variable (object): Variable to save to file. It can be of any type.
        name (str): Name of variable, which will be used as the filename for the 
                    .pickle file.
        dst (str, optional): Destination of file (folder). Defaults to current directory.
    """
    filename = name + ".pickle"
    if filename in os.listdir(dst):
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


def formatter(format_str, widths, *columns):
    """
    format_str describes the format of the report.
    {row[i]} is replaced by data from the ith element of columns.

    widths is expected to be a list of integers.
    {width[i]} is replaced by the ith element of the list widths.

    All the power of Python's string format spec is available for you to use
    in format_str. You can use it to define fill characters, alignment, width, type, etc.

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
