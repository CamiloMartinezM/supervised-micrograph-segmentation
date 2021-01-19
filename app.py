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
)
import model
import numpy as np

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
        algorithm_parameters: tuple = (100, 1.4, 100),
    ):
        self.K = K
        self.name = name
        self.filterbank = filterbank
        self.classes = np.array(classes)
        self.subsegment_class = subsegment_class
        self.texton_matrix = texton_matrix
        self.superpixel_algorithm = superpixel_algorithm
        self.superpixel_algorithm = superpixel_algorithm
        self.algorithm_parameters = algorithm_parameters

    def segment(self, image_path: str, pixel_length_scale: int) -> None:
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

        print_table_from_dict(
            data=pixel_counts_to_volume_fraction(
                segmentation_pixel_counts,
                pixel_length_scale=pixel_length_scale,
                length_scale=50,
                img_size=original_img.shape,
            ),
            cols=[
                "Phase or morphology",
                "Volume fraction [µm²]",
                "Percentage area [%]",
            ],
            title="",
            format_as_percentage=[2],
        )

        interlaminar_spacing = dict(
            zip(
                ["1", "2"],
                model.calculate_interlamellar_spacing(
                    original_img, class_matrix, new_classes
                ),
            )
        )

        print_table_from_dict(
            interlaminar_spacing, ["Method", "Value"], title="Interlaminar spacing"
        )


def _clear() -> None:
    """Clears the console."""
    if os.name == "nt":
        _ = os.system("cls")  # For windows.
    else:
        _ = os.system("clear")  # For mac and linux (here, os.name is 'posix').


def _create_title(title: str) -> None:
    """ Creates a proper title.
    
    Args:
        title (str): Title of the program.
    """
    print("~" * CONSOLE_WIDTH + "\n")
    print(center_wrap(title, cwidth=80, width=50) + "\n")
    print(" " * (CONSOLE_WIDTH - len(AUTHOR)) + AUTHOR)
    print("~" * CONSOLE_WIDTH)
    print("")


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
        for supported_image_format in SUPPORTED_IMAGE_FORMATS:
            if img.endswith(supported_image_format):
                return True
        return False

    print(
        f"\nPress Enter to show files in {folder}. Plus, include the extension when "
        "typing the image name."
    )

    img_name = input("[?] Image name >> ").strip()

    if img_name == "":
        path = os.path.join(os.getcwd(), folder)
        files = [
            file
            for file in os.listdir(path)
            if not os.path.isdir(file) and _is_image(file)
        ]
        files = sorted(files) + ["EXIT"]
        files_dict = {ind: value for ind, value in enumerate(files)}
        print(f"\n\tFiles in {folder}:\n")
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
    imgs_folder, selected_model = selected_stuff
    if imgs_folder == ".":
        imgs_folder = "{Current}"

    while True:
        try:
            option = "[?] Option "
            close_bracket = False
            if imgs_folder is not None:
                option += f"(Image Folder: {imgs_folder}"
                close_bracket = True
            if selected_model is not None:
                if not close_bracket:
                    option += "("
                else:
                    option += ", "
                option += f"Model: {selected_model.name}"
                close_bracket = True

            if close_bracket:
                option += ")"

            option += " >> "

            selected_option = int(input(option).strip())
            return selected_option
        except:
            print("[*] Select a valid option.")


def center_wrap(text: str, cwidth: int = 80, **kw):
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
    )


def segment_an_image(image_path: str, model: SegmentationModel):
    model.segment(image_path, pixel_length_scale=100)
    _ = input("[?] Press any key to continue >> ")


def menu():
    """Prints available options to the user."""
    print("1. Select a folder of images to segment.")
    print("2. Load final model (evaluated in paper).")
    print("3. Create a new model.")
    print("4. Segment a single image.")
    print("0. Quit.\n")


def main():
    imgs_folder = None
    img_name = None
    selected_model = None

    _continue_ = True
    clear_console = True
    while _continue_:
        if clear_console:
            _create_title(
                "Application of computer vision in the analysis of microstructures "
                "and obtaining structure-property relationships."
            )
            menu()
            clear_console = False

        selected_option = _take_option((imgs_folder, selected_model))

        if selected_option == 1:
            imgs_folder = _select_image_folder()
            if imgs_folder is None:
                print("[*] No folder selected.")
            else:
                print("\n[+] Chosen folder: ", imgs_folder)

            sleep(2)
            clear_console = True
        elif selected_option == 2:
            selected_model = load_final_model()
            clear_console = True
        elif selected_option == 3:
            pass  # TODO
        elif selected_option == 4:
            if imgs_folder is None:
                print("[*] No folder of images selected yet (option 1).")
            else:
                img_name = _select_image(imgs_folder)

            if selected_model is None:
                print("\n[*] A model has not been loaded yet.")
            elif img_name is None:
                print("\n[*] An image has not been selected.")
            else:
                segment_an_image(img_name, selected_model)
                
            clear_console = True
        elif selected_option == 0:
            _continue_ = False
        else:
            print("[*] Please, select a valid option.")

        if clear_console:
            _clear()


if __name__ == "__main__":
    main()
