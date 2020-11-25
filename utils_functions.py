# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:11:12 2020

@author: Camilo Martínez
"""
import os
import cudf

from pprint import pprint
from random import shuffle

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
                     corresponding annotated image windows as numpy arrays.
        train_size (float): Percentage of the dataset that will be used as training set.
        dev_size (float): Percentage of the dataset that will be used as development set.

    Returns:
        tuple: train, dev and test sets as dictionaries of the same structure as the given dataset.

    """
    if train_size + dev_size > 1:
        raise ValueError("Invalid train and/or dev ratios.")

    train = {}
    dev = {}
    test = {}

    # Randomize values of dictionary.
    for label in data:
        shuffle(data[label])

    # Split values in 3 dictionaries.
    for label in data:
        train_split = int(len(data[label]) * train_size)
        dev_split = train_split + int(len(data[label]) * dev_size)
        train[label], dev[label] = (
            data[label][:train_split],
            data[label][train_split : dev_split + 1],
        )
        test[label] = data[label][dev_split + 1 :]

    return train, dev, test


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


def load_img(path: str, as_255: bool = True, with_io: bool = False) -> np.ndarray:
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
    possible_folders = ["High carbon", "Medium carbon", "Low carbon"]
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


def find_path_of_img(name: str, src: str) -> str:
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
                return os.path.abspath(os.path.join(root, filename))
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
    cm: np.ndarray,
    target_names: list,
    title: str = "Confusion matrix",
    distinguishable_title: str = None,
    cmap=plt.cm.Blues,
    dpi: int = 120,
    figsize: tuple = (12, 8),
    savefig: bool = True,
    showfig: bool = True,
    format_percentage: bool = False,
) -> None:
    """Plots a visual representation of a confusion matrix.
    Original: https://stackoverflow.com/questions/35585069/python-tabulating-confusion-matrix

    Args:
        matrix (np.ndarray): Confusion matrix where rows are true classes and columns are predicted
                             classes.
        target_names (list): Names of classes.
        title (str, optional): Title of plot. Defaults to "Confusion matrix".
        distinguishable_title (str, optional): Distinguishable title to add to the name of the file
                                               to which the plot would be saved. Defaults to None.
        cmap (TYPE, optional): Colormap to use. Defaults to plt.cm.Blues.
        dpi (int, optional): DPI of plot. Defaults to 100.
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        savefig (bool, optional): Specifies whether to save the figure in the current directory or
                                  not. Defaults to True.
        showfig (bool, optional): Specifies whether to show the figure or not. Defaults to True.
        format_percentage (bool, optional): True if numbers as percentages are desired. Defaults to
                                            False.
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(
                str("{0:.1%}".format(cm[x][y])),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )
    plt.ylabel("True class")
    plt.xlabel("Predicted class")

    if showfig:
        plt.show()
    if savefig:
        if distinguishable_title is not None:
            plt.savefig("Confusion matrix, " + distinguishable_title + ".png", dpi=dpi)
        else:
            plt.savefig("Confusion matrix.png", dpi=dpi)

    plt.close()


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


def fullprint(*args, **kwargs) -> None:
    """Prints an array without truncation"""
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)
