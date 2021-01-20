# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 08:42:22 2020

@author: Camilo Martínez
"""
import give_console_width
import os
import textwrap
import tkinter as tk
from tkinter import filedialog
from time import sleep
from utils_functions import (
    load_variable_from_file,
    adjust_labels,
    print_table_from_dict,
    pixel_counts_to_volume_fraction,
    train_dev_test_split,
    train_dev_test_split_table,
    print_interlaminar_spacing_table,
    print_mechanical_properties_table,
)
from utils_classes import Preprocessor, Material
from colorama import Fore, Style
import model
import numpy as np
import statistics
from pathlib import Path

# Author
AUTHOR = "Camilo Martínez M."

# Console width of the current/active console.
CONSOLE_WIDTH = give_console_width.main()

# Supported image file formats
SUPPORTED_IMAGE_FORMATS = [".png"]


class SegmentationModel:
    """Simple class to keep track of the parameters of the final segmentation model.
    """

    def __init__(
        self,
        K: int,
        name: str,
        classes: list,
        subsegment_class: tuple,
        filterbank: str,
        superpixel_algorithm: str,
        texton_matrix: np.ndarray,
        scales: dict,
        windows_train: dict,
        windows_dev: dict,
        windows_test: dict,
        training_set: dict,
        development_set: dict,
        test_set: dict,
        algorithm_parameters: tuple = (100, 1.4, 100),
    ) -> None:
        self.K = K
        self.name = name
        self.filterbank = filterbank
        self.classes = np.array(classes)
        self.subsegment_class = subsegment_class
        self.texton_matrix = texton_matrix
        self.scales = scales
        self.windows_train = windows_train
        self.windows_dev = windows_dev
        self.windows_test = windows_test
        self.training_set = training_set
        self.development_set = development_set
        self.test_set = test_set
        self.superpixel_algorithm = superpixel_algorithm
        self.algorithm_parameters = algorithm_parameters

    @classmethod
    def from_parameters_dict(cls, parameters: dict) -> None:
        return cls(
            parameters["K"],
            parameters["name"],
            np.array(parameters["classes"]),
            parameters["subsegment_class"],
            parameters["filterbank"],
            parameters["superpixel_algorithm"],
            parameters["texton_matrix"],
            parameters["scales"],
            parameters["windows_train"],
            parameters["windows_dev"],
            parameters["windows_test"],
            parameters["training_set"],
            parameters["development_set"],
            parameters["test_set"],
            algorithm_parameters=parameters["algorithm_parameters"],
        )

    def segment(
        self, image_path: str, pixel_length_scale: int, length_scale: int
    ) -> None:
        (
            original_img,
            class_matrix,
            new_classes,
            segmentation_pixel_counts,
        ) = model.segment(
            image_path,
            self.classes,
            self.texton_matrix,
            algorithm=self.superpixel_algorithm,
            algorithm_parameters=self.algorithm_parameters,
            filterbank_name=self.filterbank,
            plot_original=True,
            plot_superpixels=True,
            subsegment_class=self.subsegment_class,
        )

        print("\nSegmentation:\n")
        model.visualize_segmentation(
            original_img, new_classes, class_matrix, dpi=120,
        )
        # model.plot_image_with_ground_truth(test_img, ground_truth)

        segmentation_pixel_counts = adjust_labels(segmentation_pixel_counts)
        volume_fractions = pixel_counts_to_volume_fraction(
            segmentation_pixel_counts,
            pixel_length_scale=pixel_length_scale,
            length_scale=length_scale,
            img_size=original_img.shape,
        )

        print_table_from_dict(
            data=volume_fractions,
            cols=[
                "Phase or morphology",
                "Volume fraction [µm²]",
                "Percentage area [%]",
            ],
            title="",
            format_as_percentage=[2],
        )

        spacings = model.calculate_interlamellar_spacing(
            original_img, class_matrix, new_classes
        )
        interlaminar_spacing = {
            "1": {
                "px": spacings[0],
                "µm": spacings[0] * length_scale / pixel_length_scale,
            },
            "2": {
                "px": spacings[1],
                "µm": spacings[1] * length_scale / pixel_length_scale,
            },
        }

        print_interlaminar_spacing_table(interlaminar_spacing)

        predict = _get_str_input(
            "[?] Predict mechanical properties?", ["yes", "no"], default="yes"
        )
        if predict == "yes":
            SegmentationModel.predict_mechanical_properties(volume_fractions, interlaminar_spacing)

    @staticmethod
    def predict_mechanical_properties(volume_fractions: dict, spacings: dict) -> None:
        print("\nStructure-Property Relationships:\n")
        print("For plain-carbon steels:\n")
        print(
            textwrap.dedent(
                """
                    σ_y = f_α[77.7 + 59.5(%Mn) + 9.1D_α^(-0.5)] + 145.5 
                           + 3.5λ^(-0.5) + 478(%N)^(0.5)+ 1200(%P)
                """
            )
        )
        print(
            textwrap.dedent(
                """
                    σ_u = f_α[20 + 2440(%N)^0.5 + 18.5D_α] + [750(1 - f_α)] 
                           + 3(λ^(-0.5))(1 - f_α^(0.5)) + 92.5(%Si)
                """
            )
        )
        print("\nFor fully pearlitic steels (M = 2(λ - t); t = 0.15λ(%C)):\n")
        print("\n\tIf λ >= 0.15 µm")
        print("\t\tσ_y = 308 + 0.07M^(-1)")
        print("\t\tσ_u = 706 + 0.072M^(-1)) + 122(%Si)")
        print("\n\tIf λ < 0.15 µm")
        print("\t\tσ_y = 259 + 0.087M^(-1)")
        print("\t\tσ_u = 773 + 0.058M^(-1) + 122(%Si)")
        print(
            textwrap.dedent(
                """
                   Where λ is the pearlite interlamellar spacing; f_α, the percentage of 
                   proeutectoid ferrite; D_α, the ferrite grain size; and %Mn, %Si, %P, %N
                   correspond to the chemical composition of the material in question.
                
                   λ is the most influential parameter in these equations; nonetheless, the
                   inclusion of the other parameters can improve the prediction.
                
                   Input the parameters you want to take into account; otherwise press Enter to 
                   leave the value at zero.
                """
            )
        )
        if volume_fractions.get("ferrite", 0) <= 0.1:  # pearlitic steel
            p_C = _get_simple_numerical_entry("[?] %C", "float")
            p_Si = _get_simple_numerical_entry(
                "[?] %Si", "float", default_value=0
            )
            steels = {}
            for method in spacings:
                steels[method] = Material(
                    fa=volume_fractions.get("ferrite", 0),
                    S_0=spacings[method]["µm"],
                    p_C=p_C,
                    p_Si=p_Si,
                )
        else:  # plain-carbon steel
            p_C = _get_simple_numerical_entry("[?] %C", "float")
            p_Mn = _get_simple_numerical_entry(
                "[?] %Mn", "float", default_value=0
            )
            D_a = _get_simple_numerical_entry("[?] D_α, µm", "float", default_value=0)
            p_N = _get_simple_numerical_entry(
                "[?] %N", "float", default_value=0
            )
            p_P = _get_simple_numerical_entry(
                "[?] %P", "float", default_value=0
            )
            p_Si = _get_simple_numerical_entry(
                "[?] %Si", "float", default_value=0
            )
            steels = {}
            for method in spacings:
                steels[method] = Material(
                    fa=volume_fractions.get("ferrite", 0),
                    S_0=spacings[method]["µm"],
                    p_C=p_C,
                    p_Mn=p_Mn,
                    D_a=D_a,
                    p_N=p_N,
                    p_P=p_P,
                    p_Si=p_Si,
                )

        mechanical_properties = {}
        for method, steel in steels.items():
            mechanical_properties[method] = {
                "Yield Strength [MPa]": steel.sigma_y,
                "Tensile Strength [MPa]": steel.sigma_u,
            }
            
        print_mechanical_properties_table(mechanical_properties, spacings)

    def evaluate_classification_performance(self) -> None:
        classification_metrics = model.evaluate_classification_performance(
            self.K,
            self.classes,
            self.texton_matrix,
            self.filterbank,
            self.windows_train,
            self.windows_dev,
            self.windows_test,
            save_png=False,
            save_xlsx=False,
        )
        SegmentationModel.show_metrics(classification_metrics)

    def evaluate_segmentation_performance(
        self, imgs_folder: str, ground_truth: dict
    ) -> None:
        segmentation_metrics = {}
        jaccard_per_img = {}
        for _set in ["Train", "Dev", "Test"]:
            if _set == "Train":
                print("\n[+] On training...")
            elif _set == "Dev":
                print("\n[+] On development...")
            else:
                print("\n[+] On testing...")

            (
                segmentation_metrics[_set],
                jaccard_per_img[_set],
            ) = model.evaluate_segmentation_performance(
                self.training_set[:3],
                ground_truth,
                self.classes,
                self.K,
                self.texton_matrix,
                self.superpixel_algorithm,
                self.algorithm_parameters,
                filterbank_name=self.filterbank,
                imgs_folder=imgs_folder,
                save_png=False,
                save_xlsx=False,
            )

        for _set in jaccard_per_img:
            micro_jaccard = []
            for img in jaccard_per_img[_set]:
                micro_jaccard.append(jaccard_per_img[_set][img]["Micro"])

            segmentation_metrics[_set]["Overall Statistics"][
                "Micro Averaged Jaccard Index"
            ] = statistics.harmonic_mean(micro_jaccard)

        SegmentationModel.show_metrics(segmentation_metrics)

    @staticmethod
    def show_metrics(metrics: dict) -> None:
        print("\n[+] Metrics:")
        overall_stats = [
            "Overall Accuracy",
            "F1 Macro",
            "Overall Jaccard Index",
            "Micro Averaged Jaccard Index",
        ]
        class_stats = ["Accuracy", "Averaged F1"]
        branch = " │ "
        for i, _set in enumerate(metrics):
            if i == len(metrics) - 1:
                bullet = " └── "
            else:
                bullet = " ├── "

            print(bullet + _set + ":")
            if _set != "Test":
                print(f"{branch}", end="")
            print("\t ├── Overall:")
            for j, stat in enumerate(overall_stats):
                if j == len(overall_stats) - 1:
                    bullet = " └── "
                else:
                    bullet = " ├── "
                value = metrics[_set]["Overall Statistics"][stat]
                if _set != "Test":
                    print(f"{branch}", end="")
                if type(value) is tuple:
                    value_to_print = (round(value[0], 3), round(value[1], 3))
                    value_to_print = str(value_to_print)
                else:
                    value_to_print = str(round(value, 3))
                print(f"\t{branch}\t{bullet}{stat}: {value_to_print}")

            if _set != "Test":
                print(f"{branch}\t{branch}\n{branch}", end="")
            else:
                print(f"\t{branch}\n", end="")
            print(f"\t{bullet}Per Class:")
            for k, stat in enumerate(class_stats):
                if k == len(class_stats) - 1:
                    bullet = " └── "
                else:
                    bullet = " ├── "
                if _set != "Test":
                    print(f"{branch}", end="")
                print(f"\t\t{bullet}{stat}: ", end="")
                value = metrics[_set]["Class Statistics"][stat]
                if type(value) is dict:
                    print()
                    for l, pair in enumerate(value.items()):
                        if l == len(value) - 1:
                            bullet = " └── "
                        else:
                            bullet = " ├── "
                        label, subvalue = pair
                        if _set != "Test":
                            print(f"{branch}", end="")
                        print(f"\t\t{branch}\t{bullet}{label}: {round(subvalue, 3)}")
                elif type(value) is tuple:
                    value = (round(value[0], 3), round(value[1], 3))
                    print(f"{value}")
                else:
                    print(f"{round(value, 3)}")


def _clear() -> None:
    """Clears the console."""
    if os.name == "nt":
        _ = os.system("cls")  # For windows.
    else:
        _ = os.system("clear")  # For mac and linux (here, os.name is 'posix').


def path_leaf(path: str) -> str:
    """Gets the filename associated with a path."""
    return Path(path).stem


def _create_title(title: str) -> None:
    """ Creates a proper title.
    
    Args:
        title (str): Title of the program.
    """
    print("~" * CONSOLE_WIDTH + "\n")
    print(_center_wrap(title, cwidth=80, width=50) + "\n")
    print(" " * (CONSOLE_WIDTH - len(AUTHOR)) + AUTHOR)
    print("~" * CONSOLE_WIDTH)
    print("")


def _get_simple_numerical_entry(
    msg: str,
    type_value: str,
    sign: str = "+",
    default_value=None,
    return_None: bool = False,
) -> float:
    """ Gets an entry from the user and parses it to int or float depending on type_value parameter.
    
    Args:
        msg (str): Message to be shown before the user inputs a value.
        type_value (str): int or float.
        sign (str): '+' if a positive non-zero value is to be expected. '-' otherwise.
        default_value (optional): Default value of variable. Defaults to None.
        return_None (bool, optional): True if the value can be None. Defaults to False.
        
    Returns:
        float, int: Entry made by the user.
    """
    complete_msg = msg
    if default_value is not None:
        complete_msg += " (Default=" + str(default_value) + ")"

    complete_msg += " > "
    entry_str = ""
    entry = None
    try:
        entry_str = input(complete_msg)
        if entry_str.strip() == "":
            if default_value is not None or (default_value is None and return_None):
                entry = default_value
        elif entry_str.strip() == "":
            raise Exception("Empty string")
        else:
            if type_value == "float":
                entry = float(entry_str.strip())
            elif type_value == "int":
                entry = int(entry_str.strip())

        if entry <= 0 and sign == "+" and default_value != 0:
            raise Exception("Expected a positive non-zero value.")
    except:
        # Recursion till entry receives a valid value.
        entry = _get_simple_numerical_entry(msg, type_value, sign, default_value)

    return entry


def _get_str_input(msg: str, valid_inputs: list, default: str = None) -> str:
    complete_msg = msg + " (" + str(valid_inputs)[1:-1]

    if default is not None:
        complete_msg += ", Default: '" + default + "'"

    complete_msg += ") > "

    raw_str = input(complete_msg)

    if raw_str.strip() == "" and default is not None:
        return default

    if raw_str.strip().lower() not in valid_inputs:
        error_message = (
            "[*] Invalid entry. Expected any of: "
            + str(valid_inputs)
            + " and got: "
            + raw_str
        )
        print(error_message + ".")
        raw_str = _get_str_input(msg, valid_inputs, default)

    return raw_str.strip().lower()


def _get_folder() -> str:
    """Print a numbered list of the subfolders in the working directory (i.e. the
    directory the script is run from), and returns the directory the user chooses.
    
    Returns:
        str: Path of selected folder.
    """
    print(
        textwrap.dedent(
            """
                [?] Which folder are your files located in?
                    If you cannot see it in this list, you need to copy the folder
                    containing them to the same folder as this script.
            """
        )
    )
    dirs = [d for d in os.listdir() if os.path.isdir(d)] + ["EXIT"]
    dir_dict = {ind: value for ind, value in enumerate(dirs)}
    for key in dir_dict:
        print("\t(" + str(key) + ") " + dir_dict[key])

    while True:
        try:
            resp = int(input("\t> ").strip())
            if resp not in dir_dict:
                raise Exception("")
            else:
                break
        except:
            print("\t[*] Please, select a valid folder.")

    if dir_dict[resp] == "EXIT":
        return None
    else:
        return dir_dict[resp]


def _select_image_folder() -> str:
    """Selects the image folder by using either a GUI file chooser or a script-based 
    chooser.

    Returns:
        str: Path of image folder.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        return os.path.relpath(file_path, start=os.getcwd())
    except:
        print("[*] Unable to open a GUI file chooser. Using script-based option.")
        return _get_folder()


def _select_image(folder: str) -> str:
    def _is_image(img: str) -> bool:
        """Checks if a filename or filepath is actually an image by checking its
        extension. It also checks if it is a supported extension.

        Args:
            img (str): Filepath or filename.

        Returns:
            bool: True if the input filename or filepath is an image with a supported
                  extension.
        """
        for supported_image_format in SUPPORTED_IMAGE_FORMATS:
            if img.endswith(supported_image_format):
                return True
        return False

    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        return os.path.relpath(file_path, start=os.getcwd())
    except:
        print("[*] Unable to open a GUI file chooser. Using script-based option.")

        if folder is None:
            print("[*] A folder of images has not been selected yet.")
            print("    Showing images in current folder.\n")
            path = os.getcwd()
        else:
            print(
                textwrap.dedent(
                    """
                        [?] Press Enter to show files in {folder}.
                            Include the extension when typing the image name.
                    """
                )
            )

            while True:
                img_name = input("\t[?] Image name >> ").strip()
                path = os.path.join(os.getcwd(), folder)

                if img_name != "":
                    if img_name not in os.listdir(path):
                        print(
                            f"\t[*] Image {img_name} was not found in {folder}. Try "
                            "again."
                        )
                    else:
                        return os.path.relpath(
                            os.path.join(folder, img_name), start=os.getcwd()
                        )
                else:
                    break

        files = [
            file
            for file in os.listdir(path)
            if not os.path.isdir(file) and _is_image(file)
        ]
        files = sorted(files) + ["EXIT"]
        files_dict = {ind: value for ind, value in enumerate(files)}
        for key in files_dict:
            print("\t(" + str(key) + ") " + files_dict[key])

        while True:
            try:
                resp = int(input("\t> ").strip())
                if resp not in files_dict:
                    raise Exception("")
                else:
                    break
            except:
                print("\t[*] Please, select a valid file.")

        if files_dict[resp] == "EXIT":
            return None
        else:
            img_name = files_dict[resp]

        return os.path.relpath(os.path.join(folder, img_name), start=os.getcwd())


def _take_option(selected_stuff: tuple) -> int:
    imgs_folder, img_name, selected_model, ground_truth = selected_stuff
    if imgs_folder == ".":
        imgs_folder = "{Current}"

    while True:
        try:
            option = "[?] Option "
            close_bracket = False
            if imgs_folder is not None:
                option += f"(Image Folder: {imgs_folder}"
                close_bracket = True
            if img_name is not None:
                if not close_bracket:
                    option += "("
                else:
                    option += ", "
                option += f"Image: {img_name}"
                close_bracket = True
            if selected_model is not None:
                if not close_bracket:
                    option += "("
                else:
                    option += ", "
                option += f"Model: {selected_model.name}"
                close_bracket = True
            if ground_truth is not None:
                if not close_bracket:
                    option += "("
                else:
                    option += ", "
                option += "Ground truth: Loaded"
                close_bracket = True

            if close_bracket:
                option += ")"

            option += " >> "

            selected_option = int(input(option).strip())
            return selected_option
        except:
            print("[*] Select a valid option.")


def _take_tool_option(selected_stuff: tuple) -> int:
    (labeled_folder, preprocessed_folder,) = selected_stuff
    while True:
        try:
            option = "[?] Option "
            close_bracket = False
            for variable_name, variable in [
                ("Folder of labeled images", labeled_folder),
                ("Folder of preprocessed images", preprocessed_folder),
            ]:
                if variable is not None:
                    if not close_bracket:
                        option += "("
                    else:
                        option += ", "
                    option += f"{variable_name}: {variable}"
                    close_bracket = True

            if close_bracket:
                option += ")"

            option += " >> "

            selected_option = int(input(option).strip())
            return selected_option
        except:
            print("[*] Select a valid option.")


def _center_wrap(text: str, cwidth: int = 80, **kw) -> str:
    """Centers a text.

    Args:
        text (str): Text to center.
        cwidth (int): Wanted width. Defaults to 80.
        **kw: Arguments of textwrap.wrap

    Returns:
        str: Centered text.
    """
    lines = textwrap.wrap(text, **kw)
    return "\n".join(line.center(cwidth) for line in lines)


def _load_default_feature_vectors() -> dict:
    return load_variable_from_file("feature_vectors", "saved_variables")


def _load_default_parameters() -> dict:
    parameters = {
        "K": 6,
        "filterbank": "MR8",
        "name": "Default",
        "classes": ["proeutectoid ferrite", "pearlite"],
        "subsegment_class": ("pearlite", "ferrite"),
        "superpixel_algorithm": "felzenszwalb",
        "algorithm_parameters": (100, 1.4, 100),
        "texton_matrix": None,
        "scales": None,
        "windows_train": None,
        "windows_dev": None,
        "windows_test": None,
        "training_set": None,
        "development_set": None,
        "test_set": None,
    }
    return parameters


def _load_new_parameters() -> dict:
    K = _get_simple_numerical_entry("[?] Define K", "int", default_value=6)
    filterbank = _get_str_input("[?] Filterbank", ["MR8", "MAT"], default="MR8")
    superpixel_algorithm = _get_str_input(
        "[?] Superpixel algorithm",
        ["quickshift", "slic", "felzenszwalb", "watershed"],
        default="felzenszwalb",
    )

    if superpixel_algorithm == "slic":
        n_segments = _get_simple_numerical_entry(
            "└── [?] Number of centers for K-Means, n_segments",
            "int",
            default_value=500,
        )
        sigma = _get_simple_numerical_entry(
            "├── [?] Width of gaussian smoothing kernel, sigma",
            "float",
            default_value=0,
        )
        compactness = _get_simple_numerical_entry(
            "└── [?] Color and space proximity balance, compactness",
            "float",
            default_value=0.17,
        )
        algorithm_parameters = (n_segments, sigma, compactness)
    elif superpixel_algorithm == "felzenszwalb":
        scale = _get_simple_numerical_entry(
            "└── [?] Free parameter. Higher means more clusters, scale",
            "float",
            default_value=100,
        )
        sigma = _get_simple_numerical_entry(
            "├── [?] Width of gaussian smoothing kernel, sigma",
            "float",
            default_value=1.4,
        )
        min_size = _get_simple_numerical_entry(
            "└── [?] Minimum component size, min_size > ", "float", default_value=100
        )
        algorithm_parameters = (scale, sigma, min_size)
    elif superpixel_algorithm == "quickshift":
        ratio = _get_simple_numerical_entry(
            "└── [?] Color-space and image-space proximity, ratio",
            "float",
            default_value=1,
        )
        kernel_size = _get_simple_numerical_entry(
            "├── [?] Width of gaussian smoothing kernel, kernel_size",
            "float",
            default_value=5,
        )
        sigma = _get_simple_numerical_entry(
            "├── [?] Scale of the local density approximation, sigma",
            "float",
            default_value=0,
        )
        max_dist = _get_simple_numerical_entry(
            "└── [?] Cut-off point for data distances, max_dist",
            "float",
            default_value=10,
        )
        algorithm_parameters = (ratio, kernel_size, max_dist, sigma)
    else:  # superpixel_algorithm == "watershed":
        markers = _get_simple_numerical_entry(
            "└── [?] Desired number of markers, markers",
            "int",
            default_value=None,
            return_None=True,
        )
        compactness = _get_simple_numerical_entry(
            "└── [?] Compact watershed parameter, compactness", "float", default_value=0
        )
        algorithm_parameters = (markers, compactness)

    parameters = {
        "K": K,
        "filterbank": filterbank,
        "name": "New",
        "classes": ["proeutectoid ferrite", "pearlite"],
        "subsegment_class": ("pearlite", "ferrite"),
        "superpixel_algorithm": superpixel_algorithm,
        "algorithm_parameters": algorithm_parameters,
        "texton_matrix": None,
        "scales": None,
        "windows_train": None,
        "windows_dev": None,
        "windows_test": None,
        "training_set": None,
        "development_set": None,
        "test_set": None,
    }
    return parameters


def _load_default_scales_dictionary() -> dict:
    return load_variable_from_file("scales", "saved_variables")


def _preprocess_folder(path: str, dst: str) -> None:
    myPreprocessor = Preprocessor(path, dst)
    myPreprocessor.process()


def _set_up_train_dev_test_split() -> tuple:
    print("\n[?] Set up train/dev/test split. Remember train_size + dev_size <= 1")
    train_size = 1
    dev_size = 1
    while True:
        while True:
            try:
                train_size = float(input("[?] Train size (<= 1) > "))
                if train_size > 1 or train_size <= 0:
                    raise Exception("")
                break
            except:
                print("[*] Please, input a valid entry.")
        while True:
            try:
                dev_size_str = input("[?] Dev size (<= 1) > ").strip().lower()
                if dev_size_str == "over":
                    break

                dev_size = float(dev_size_str)
                if train_size + dev_size > 1 or dev_size < 0:
                    raise Exception("")
                return train_size, dev_size
            except:
                print(
                    "[*] Please, input a valid entry. Or type 'over' if you want to "
                    "redefine train_size."
                )

    return train_size, dev_size


def _load_ground_truth(labeled_folder: str, classes: np.ndarray) -> str:
    ground_truth_path = None
    try:
        int("tela")
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        ground_truth_path = os.path.relpath(file_path, start=os.getcwd())
    except:
        print("[*] Unable to open a GUI file chooser. Using script-based option.")

        path = os.getcwd()
        print(
            textwrap.dedent(
                """
                    [?] Press Enter to show files with .tif extension in current folder.
                        Include the extension when typing the file name.
                """
            )
        )

        while True:
            img_name = input("\t[?] File name >> ").strip()

            if img_name != "":
                if img_name not in os.listdir(path):
                    print(
                        f"\t[*] File {img_name} was not found in current folder. Try again."
                    )
                else:
                    ground_truth_path = os.path.relpath(
                        os.path.join(path, img_name), start=os.getcwd()
                    )
            else:
                break

        if ground_truth_path is None:
            files = [
                file
                for file in os.listdir(path)
                if not os.path.isdir(file) and file.endswith(".tif")
            ]
            files = sorted(files) + ["EXIT"]
            files_dict = {ind: value for ind, value in enumerate(files)}
            for key in files_dict:
                print("\t(" + str(key) + ") " + files_dict[key])

            while True:
                try:
                    resp = int(input("\t> ").strip())
                    if resp not in files_dict:
                        raise Exception("")
                    else:
                        break
                except:
                    print("\t[*] Please, select a valid file.")

            if files_dict[resp] == "EXIT":
                return None
            else:
                ground_truth_path = os.path.relpath(files_dict[resp], start=os.getcwd())

    ground_truth = model.load_ground_truth(ground_truth_path, labeled_folder, classes)

    return ground_truth


def load_final_model() -> SegmentationModel:
    return SegmentationModel(
        K=6,
        name="Final",
        classes=["proeutectoid ferrite", "pearlite"],
        subsegment_class=("pearlite", "ferrite"),
        filterbank="MR8",
        superpixel_algorithm="felzenszwalb",
        texton_matrix=load_variable_from_file(
            "texton_matrix_K_6_final", "saved_variables"
        ),
        scales=_load_default_scales_dictionary(),
        windows_train=load_variable_from_file("training_windows", "saved_variables"),
        windows_dev=load_variable_from_file("development_windows", "saved_variables"),
        windows_test=load_variable_from_file("testing_windows", "saved_variables"),
        training_set=load_variable_from_file("training_set_imgs", "saved_variables"),
        development_set=load_variable_from_file(
            "development_set_imgs", "saved_variables"
        ),
        test_set=load_variable_from_file("testing_set_imgs", "saved_variables"),
    )


def load_new_model() -> SegmentationModel:
    _clear()

    feature_vectors = None
    labeled_folder = None
    parameters = None
    scales = None
    preprocessed_folder = None
    T = None

    micrographs = None
    index_to_name = None
    loaded_elements = {
        "Feature vectors": False,
        "Parameters": False,
        "Textons": False,
        "Scales Information": False,
    }
    loaded_default_feature_vectors = False
    available_options_per = {1: [1, 2], 2: [3, 4], 3: [5, 6, 7]}
    clear_console = True
    section = 1
    while section <= 3:
        if clear_console:
            _create_title("Segmentation Model creation tool")
            tool_menu(section, loaded_elements)
            clear_console = False

        selected_option = _take_tool_option((labeled_folder, preprocessed_folder))

        if selected_option in available_options_per[section] + [0]:
            if selected_option == 1:
                feature_vectors = _load_default_feature_vectors()
                loaded_default_feature_vectors = True
                loaded_elements["Feature vectors"] = True
                clear_console = True
            elif selected_option == 2:
                labeled_folder = _select_image_folder()
                print("\n[+] Loading of images:")
                micrographs, index_to_name = model.load_imgs(
                    labeled_folder, exclude=["Low carbon"]
                )
                print("\n[+] Windows extraction from loaded images:\n")
                labels, windows, windows_per_name = model.extract_labeled_windows(
                    labeled_folder, micrographs, index_to_name, exclude=["Low carbon"]
                )

                train_size, dev_size = _set_up_train_dev_test_split()
                print()
                (
                    training_set,
                    development_set,
                    test_set,
                    windows_train,
                    windows_dev,
                    windows_test,
                ) = train_dev_test_split(
                    windows, train_size=train_size, dev_size=dev_size
                )

                # Table that summarizes the number of windows noted by class or label that
                # correspond to the phases or morphologies of interest in the micrographs.
                train_dev_test_split_table(windows_train, windows_dev, windows_test)
                parameters["windows_train"] = windows_train
                parameters["windows_dev"] = windows_dev
                parameters["windows_test"] = windows_test
                _ = input("[?] Press any key to continue >> ")
                clear_console = True
            elif selected_option == 3:
                parameters = _load_default_parameters()
                loaded_elements["Parameters"] = True
            elif selected_option == 4:
                parameters = _load_new_parameters()
                loaded_elements["Parameters"] = True
            elif selected_option == 5:
                scales = _load_default_scales_dictionary()
                loaded_elements["Scales"] = True
                parameters["scales"] = scales
                _create_title("Segmentation Model creation tool")
                tool_menu(section, loaded_elements)
                clear_console = False
            elif selected_option == 6:
                preprocessed_folder = _select_image_folder()
                section -= 1
            elif selected_option == 7:
                if preprocessed_folder is None:
                    print(
                        "[*] A folder of preprocessed images has not been selected yet "
                        "(Option 6)."
                    )
                    section -= 1
                else:
                    while True:
                        dst = input(
                            "\n[?] Name of destination folder for preprocessed images > "
                        )
                        if dst in os.listdir(os.getcwd()):
                            print("[*] {dst} seems to exist already. Input a new name.")
                        else:
                            try:
                                os.mkdir(dst)
                                break
                            except:
                                print(
                                    "[*] A problem has occurred while creating a folder with "
                                    "name {dst}. Input a new name."
                                )

                    print()
                    _preprocess_folder(
                        preprocessed_folder, os.path.join(os.getcwd(), dst)
                    )
                    print("[+] Attempting to read scales from {dst}", end="")
                    for i in range(3):
                        print(".", end="")
                        sleep(0.5)
                    print()
                    scales = model.load_scales(
                        labeled_folder, dst, load_full_preprocessed=True
                    )
                    if not scales:
                        print("\n[*] No single scale could be loaded.\n")
                        leave = _get_str_input(
                            "[?] Attempt to create model with what's available?",
                            ["yes", "no"],
                            default="yes",
                        )
                        if leave == "yes":
                            break
                        else:
                            section -= 1
                    else:
                        loaded_elements["Scales Information"] = True
                        parameters["scales"] = scales
                        _create_title("Segmentation Model creation tool")
                        tool_menu(section, loaded_elements)
                        clear_console = False
            elif selected_option == 0:
                if False in loaded_elements.values():
                    print("[*] Not all elements have been loaded yet.")
                    leave = _get_str_input(
                        "[?] Attempt to create model with what's available?",
                        ["yes", "no"],
                        default="yes",
                    )
                    if leave == "yes":
                        break
                    else:
                        return None

            if section == 2 and parameters["texton_matrix"] is None:
                print()
                print("[+] Attempting to train model", end="")
                for i in range(3):
                    print(".", end="")
                    sleep(0.5)
                print()
                if loaded_default_feature_vectors and parameters["filterbank"] != "MR8":
                    print(
                        textwrap.dedent(
                            f"""
                                [*] Filterbank {parameters['filterbank']} was chosen previously.
                                    However, default feature vectors were calculated with MR8.
                                    Chosen filterbank will be overwritten by MR8. Continue?
                            """
                        )
                    )
                    _ = input("")
                    parameters["filterbank"] = "MR8"
                if feature_vectors is None:
                    feature_vectors = model.obtain_feature_vectors_of_labels(
                        windows_train, parameters["filterbank"]
                    )
                loaded_elements["Feature vectors"] = True
                classes, T, _ = model.train(
                    parameters["K"], parameters["filterbank"], feature_vectors
                )
                loaded_elements["Textons"] = True
                parameters["texton_matrix"] = T
                _ = input("[?] Press any key to continue >> ")
                clear_console = True

            if clear_console:
                _clear()
            section += 1
        else:
            print("[*] Please, select a valid option.")

    _ = input("[?] Press any key to continue >> ")
    if parameters["windows_train"] is None:  # Default feature vectors were used
        parameters["windows_train"] = load_variable_from_file(
            "windows_train", "saved_variables"
        )
        parameters["windows_dev"] = load_variable_from_file(
            "windows_dev", "saved_variables"
        )
        parameters["windows_test"] = load_variable_from_file(
            "windows_test", "saved_variables"
        )
        parameters["training_set"] = load_variable_from_file(
            "training_set_imgs", "saved_variables"
        )
        parameters["development_set"] = load_variable_from_file(
            "development_set_imgs", "saved_variables"
        )
        parameters["test_set"] = load_variable_from_file(
            "testing_set_imgs", "saved_variables"
        )

    return SegmentationModel.from_parameters_dict(parameters)


def segment_an_image(
    image_path: str,
    selected_model: SegmentationModel,
    pixel_length_scale: int,
    length_scale: int,
) -> None:
    selected_model.segment(
        image_path, pixel_length_scale=pixel_length_scale, length_scale=length_scale
    )
    _ = input("[?] Press any key to continue >> ")


def tool_menu(section: int, loaded_elements: dict) -> None:
    """Prints available options for the segmentation model creation tool."""
    for element, loaded in loaded_elements.items():
        if loaded:
            print(f"[{Fore.GREEN}LOADED{Style.RESET_ALL}]", end="")
        else:
            print(f"[{Fore.RED}MISSING{Style.RESET_ALL}]", end="")
        print(f" {element}")

    print()
    options = {
        1: (
            "Training",
            {
                "1. ": "Use default feature vectors of labels (used in paper).",
                "2. ": "Select a folder of labeled images.",
            },
        ),
        2: (
            "Model parameters",
            {
                "3. ": "Use default parameters (used in paper).",
                "4. ": "Specify new parameters.",
            },
        ),
        3: (
            "Other settings",
            {
                "5. ": "Use default scales dictionary (used in paper).",
                "6. ": "Specify folder of preprocessed images.",
                "7. ": "Preprocess images in a folder.",
            },
        ),
    }
    print(options[section][0] + ":")
    for key, option in options[section][1].items():
        print(key + option)
        if key == "6. ":
            print(
                textwrap.dedent(
                    """
                        └── An image must be accompanied by a corresponding image with 
                            name starting with 'SCALE_', which is the cropped region 
                            of the image with the scale line.
                    """
                )
            )
        elif key == "7. ":
            print(
                textwrap.dedent(
                    """
                        └── If no folder with preprocessed images exists, this option 
                            creates one by preprocessing images inside a specified 
                            folder and cropping the region with the scale for each 
                            image, naming them 'SCALE_{name}'.
                    """
                )
            )


def main_menu() -> None:
    """Prints available options to the user."""
    print("Target:")
    print("1. Select a folder of images to segment.")
    print("2. Select an image.")
    print("\nModel:")
    print("3. Load final model (evaluated in paper).")
    print("4. Create a new model.")
    print("\nActions:")
    print("5. Segment selected image.")
    print("\nEvaluate:")
    print("6. Classification performance.")
    print("7. Segmentation performance.")
    print("0. Quit.\n")


def main():
    imgs_folder = None
    img_name = None
    selected_model = None
    ground_truth = None

    _continue_ = True
    clear_console = True
    while _continue_:
        if clear_console:
            _create_title(
                "Application of Computer Vision in the Analysis of Microstructures "
                "and Obtaining Structure-Property Relationships."
            )
            main_menu()
            clear_console = False

        selected_option = _take_option(
            (imgs_folder, img_name, selected_model, ground_truth)
        )

        if selected_option == 1:
            imgs_folder = _select_image_folder()
            if imgs_folder is None:
                print("[*] No folder selected.")
            else:
                print("\n[+] Chosen folder: ", imgs_folder)

            sleep(2)
            clear_console = True
        elif selected_option == 2:
            img_name = _select_image(imgs_folder)
            if img_name is None:
                print("[*] No image selected.")
            else:
                print("\n[+] Chosen image: ", img_name)

            sleep(2)
            clear_console = True
        elif selected_option == 3:
            selected_model = load_final_model()
            clear_console = True
        elif selected_option == 4:
            selected_model = load_new_model()
            clear_console = True
        elif selected_option == 5:
            if selected_model is None:
                print("\n[*] A model has not been loaded yet.")
            elif img_name is None:
                print("\n[*] An image has not been selected.")
            else:
                filename = path_leaf(img_name)
                if filename in selected_model.scales:
                    pixel_length_scale = selected_model.scales[filename]
                    print(
                        textwrap.dedent(
                            f"""
                            [+] {filename} found on scales information. 
                                Loaded scale length in pixels: {pixel_length_scale} px.
                            """
                        )
                    )
                else:
                    print(
                        "\n[*] There is no scales information for that image. Input it manually.\n"
                    )
                    pixel_length_scale = _get_simple_numerical_entry(
                        "[?] Scale length in pixels", "int"
                    )

                length_scale = _get_simple_numerical_entry(
                    "[?] Scale length in µm", "int"
                )
                segment_an_image(
                    img_name, selected_model, pixel_length_scale, length_scale
                )

            sleep(2)
            clear_console = True
        elif selected_option == 6:
            print()
            if selected_model is None:
                print("[*] No model selected.")
            else:
                selected_model.evaluate_classification_performance()
                _ = input("[?] Press any key to continue >> ")
                clear_console = True
        elif selected_option == 7:
            print()
            if selected_model is None:
                print("[*] No model selected.")
            else:
                if imgs_folder is None:
                    print("[*] You need to specify a folder of images to segment.")
                    continue
                elif ground_truth is None:
                    print("[*] You need to load a ground truth first (.tif file).")
                    ground_truth = _load_ground_truth(
                        imgs_folder, selected_model.classes
                    )

                selected_model.evaluate_segmentation_performance(
                    imgs_folder, ground_truth
                )
                _ = input("[?] Press any key to continue >> ")
                clear_console = True
        elif selected_option == 0:
            _continue_ = False
        else:
            print("[*] Please, select a valid option.")

        if clear_console:
            _clear()


if __name__ == "__main__":
    main()
