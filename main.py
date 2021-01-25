# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:22 2020

@author: Camilo Martínez
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import model
from utils_functions import (
    adjust_labels,
    compare2imgs,
    find_path_of_img,
    load_img,
    load_variable_from_file,
    pixel_counts_to_volume_fraction,
    print_table_from_dict,
    save_variable_to_file,
    train_dev_test_split,
    train_dev_test_split_table,
    list_of_names_to_list_of_numpy_arrays
)

from utils_classes import TrailingFormatter

# Parámetros para las gráficas
matplotlib.rcParams["font.family"] = "cmr10"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({"font.size": 16})

path_labeled = "Anotadas"
path_preprocessed = "Preprocesadas"

# %%
"""Carga de variables importantes"""
K = 6
classes = np.array(["proeutectoid ferrite", "pearlite"])
T = load_variable_from_file(f"full_texton_matrix_K_{K}", "saved_variables")
windows_train = load_variable_from_file("training_windows", "saved_variables")
windows_dev = load_variable_from_file("development_windows", "saved_variables")
windows_test = load_variable_from_file("testing_windows", "saved_variables")
training_set = load_variable_from_file("training_set_imgs", "saved_variables")
development_set = load_variable_from_file("development_set_imgs", "saved_variables")
test_set = load_variable_from_file("testing_set_imgs", "saved_variables")
feature_vectors_of_label = load_variable_from_file("feature_vectors", "saved_variables")

# %%
"""Carga de las imágenes ground truth.

Las imágenes segmentadas (ground truth) se guardan en un diccionario cuyas llaves son los
nombres de las imágenes y los valores son las respectivas segmentaciones.
"""
ground_truth = model.load_ground_truth(
    os.path.join(path_labeled, "(Segmented)HypoeutectoidStackComplete.tif"),
    os.path.join(path_labeled, "Hypoeutectoid steel"),
    classes,
)

#for key in ground_truth:
#    model.plot_image_with_ground_truth(key, ground_truth, path_labeled)

# %%
save_variable_to_file(ground_truth, "ground_truth")

# %%
for name, img in ground_truth.items():
    print(name)
    print(np.unique(img))
    print("-------")
    
# %%
"""Preprocesamiento

En primer lugar, se optó por utilizar una variante de la ecualización de histograma 
adaptativo (AHE, por sus siglas en inglés), llamada ecualización de histograma 
adaptativo limitado por contraste (CLAHE, por sus siglas en inglés). AHE es una técnica 
de procesamiento de imágenes por computadora que se utiliza para mejorar el contraste 
en las imágenes. Se diferencia de la ecualización de histogramas ordinaria (en el 
sentido de que el método adaptativo calcula varios histogramas, cada uno correspondiente
a una sección distinta de la imagen, y los utiliza para redistribuir los valores de 
luminosidad de la imagen. Por lo tanto, es adecuado para mejorar el contraste local y
mejorar las definiciones de los bordes en cada región de una imagen. 

Sin embargo, el AHE ordinario tiende a sobreamplificar el contraste en regiones casi 
constantes de la imagen, ya que el histograma en tales regiones está muy concentrado. 
Como resultado, AHE puede hacer que el ruido se amplifique en regiones casi constantes. 
Por su parte, CLAHE es una variante de la ecualización adaptativa del histograma en la 
que la amplificación del contraste es limitada, para reducir este problema de 
amplificación de ruido.
"""

comparison_img = load_img(find_path_of_img("cs0327.png", model.PATH_LABELED))

# CLAHE is applied to PREPROCESSED and LABELED
model.preprocess_with_clahe(model.PATH_PREPROCESSED)
model.preprocess_with_clahe(model.PATH_LABELED)

final_img = load_img(find_path_of_img("cs0327.png", model.PATH_LABELED))

# Comparison examples
compare2imgs(comparison_img, final_img, title_2="CLAHE")

# %%
"""Carga de las imágenes

Las imágenes son cargadas en un arreglo de numpy que tiene todas las imágenes, llamado 
micrographs. Dado que es un arreglo, no hay forma directa de saber a qué imagen le 
corresponde cada posición para poder extraer fácilmente el arreglo correspondiente. Por
ello, se usa un diccionario llamado index_to_name (el nombre de variable está invertido;
debería ser en realidad name_to_index), cuyas llaves son los nombres de las imágenes 
.png y los valores son los índices correspondientes en micrographs.
"""
micrographs, index_to_name = model.load_imgs(path_labeled, exclude=["Low carbon"])

# %%
"""Carga de las escalas"""
micrographs_scales = model.load_scales(path_labeled, path_preprocessed, load_full_preprocessed=True)
save_variable_to_file(micrographs_scales, "scales", dst="saved_variables")

# %%
"""Textones

Banco de filtros

Los bancos de filtros lineales o filter banks corresponden al conjunto de filtros 
escogidos como los operadores que serían aplicados sobre las imágenes en estudio para 
la obtención de sus características texturales a través de la construcción del 
diccionario de textones a través de un algoritmo de clustering.

En la ilustración [https://www.robots.ox.ac.uk/~vgg/research/texclass/figs/with/Slide12.JPG] 
se muestra el banco de filtros anisotrópicos (RFS), compuesto por 2 filtros 
anisotrópicos (uno de borde y otro de barra, en 6 orientaciones y 3 escalas), y 2 
simétricos rotacionalmente (uno gaussiano y uno laplaciano de gaussiano). A partir del
anterior, es posible derivar bancos de filtros de respuesta máxima, los cuales son 
invariantes rotacionalmente, como por ejemplo el popular MR8 (que posee 8 respuestas de
filtros). Para conseguir dicha característica se registra únicamente la respuesta de 
filtro máxima en todas las orientaciones para los dos filtros anisotrópicos del RFS. 
Medir solo la respuesta máxima en todas las orientaciones reduce el número de respuestas
de 38 (6 orientaciones a 3 escalas para 2 filtros orientados, más 2 isotrópicos) a 8 
(3 escalas para 2 filtros, más 2 isotrópicos). Por ello, el MR8 consta de 38 filtros, 
pero solo 8 respuestas de filtro.
"""
model.filterbank_example()

# %%
"""Extracción de las ventanas/regiones anotadas

Dentro de PATH_LABELED, cada una de las imágenes tiene asociada un archivo .txt que 
contiene la información de la posición de cada una de sus regiones o ventanas que fueron
anotadas.
"""
labels, windows, windows_per_name = model.extract_labeled_windows(
    path_labeled, micrographs, index_to_name, exclude=["Low carbon"]
)

# %%
save_variable_to_file(windows, "full_windows_dict", "saved_variables")
# %%
model.save_labeled_imgs_to_pdf(windows_per_name, classes, micrographs, index_to_name)

# %%
"""Train/dev/test split

Estas imágenes fueron subdividas en conjuntos de datos para entrenamiento, validación y 
prueba, siguiente una división aleatoria del 70%, 20% y 10% respectivamente. Es 
importante destacar que, dada la naturaleza del modelo planteado, el cual no tiene en 
cuenta la totalidad de una imagen (o micrografía) sino las ventanas anotadas en ella, la 
subdivisión en entrenamiento, validación y prueba no fue hecha con respecto a las 
imágenes, sino respecto a las anotaciones. Pudo haberse hecho de la primera forma, como
es realizado comúnmente en el planteamiento de modelos de machine learning, pero así 
podría ocurrir que, en una selección aleatoria de las imágenes, se elijan para cierto 
conjunto (entrenamiento, validación o prueba) justo las imágenes que no poseen 
anotaciones de cierta clase. Esto haría o que no se entrene el modelo para reconocer 
dicha clase o que sí se entrene, pero no sea validado o probado. 
"""
(
    training_set,
    development_set,
    test_set,
    windows_train,
    windows_dev,
    windows_test,
) = train_dev_test_split(windows, train_size=0.6, dev_size=0.2)

# Table that summarizes the number of windows noted by class or label that correspond
# to the phases or morphologies of interest in the micrographs.
train_dev_test_split_table(windows_train, windows_dev, windows_test)

# %%
training = list_of_names_to_list_of_numpy_arrays(training_set, path_labeled)
development = list_of_names_to_list_of_numpy_arrays(development_set, path_labeled)
test = list_of_names_to_list_of_numpy_arrays(test_set, path_labeled)

save_variable_to_file(training, "training_set_imgs_as_numpy_arrays")
save_variable_to_file(development, "development_set_imgs_as_numpy_arrays")
save_variable_to_file(test, "testing_set_imgs_as_numpy_arrays")


# %%
"""Entrenamiento del modelo

A continuación, se muestra una función que entrena el modelo, primero construyendo la 
matriz de *feature vectors* para cada una de las clases, y luego haciendo un 
*clustering* que aprenda los textones asociados a cada una de ellas.
"""
from itertools import chain
filterbank = "MR8"

windows_to_train_on = {}
for k, v in chain(windows_train.items(), windows_dev.items()):
    windows_to_train_on.setdefault(k, []).extend(v)

windows_train = windows_to_train_on

feature_vectors_of_label = model.obtain_feature_vectors_of_labels(windows_train, filterbank)

# %%
K = 6
classes, T, _ = model.train(
    K, filterbank_name=filterbank, feature_vectors=feature_vectors_of_label
)

# %%
test_img = "as0013.png"
filterbank = "MR8"
classes = np.array(["proeutectoid ferrite", "pearlite"])

# algorithm = "SLIC"
# algorithm_parameters = (500, 5, 0.17)
algorithm = "felzenszwalb"
algorithm_parameters = (100, 1.4, 100)
# algorithm = "quickshift"
# algorithm_parameters = (0.5, 5, 8, 0)
# algorithm = "watershed"
# algorithm_parameters = (250, 0.001)
original_img, class_matrix, new_classes, segmentation_pixel_counts = model.segment(
    load_img(find_path_of_img(test_img, path_labeled)),
    classes,
    T,
    algorithm=algorithm,
    algorithm_parameters=algorithm_parameters,
    filterbank_name=filterbank,
    plot_original=True,
    plot_superpixels=True,
    verbose=True,
    subsegment_class=("pearlite", "ferrite"),
)

model.visualize_segmentation(
    original_img,
    new_classes,
    class_matrix,
    dpi=80,
    save_png=True,
    png_name="Segmented " + test_img,
)
# model.plot_image_with_ground_truth(test_img, ground_truth)

segmentation_pixel_counts = adjust_labels(segmentation_pixel_counts)

if test_img[:-4] in micrographs_scales:
    pixel_length_scale = micrographs_scales[test_img[:-4]]
else:
    pixel_length_scale = 100

print_table_from_dict(
    data=pixel_counts_to_volume_fraction(
        segmentation_pixel_counts,
        pixel_length_scale=pixel_length_scale,
        length_scale=50,
        img_size=original_img.shape,
    ),
    cols=["Phase or morphology", "Volume fraction [µm²]", "Percentage area [%]"],
    title="",
    format_as_percentage=[2],
)

interlaminar_spacing = dict(
    zip(
        ["1", "2"],
        model.calculate_interlamellar_spacing(original_img, class_matrix, new_classes),
    )
)
print_table_from_dict(
    interlaminar_spacing, ["Method", "Value"], title="Interlaminar spacing"
)

# %%
save_variable_to_file(T, "full_texton_matrix_K_6", "saved_variables")

# %%
save_variable_to_file(windows_train, "training_windows")
save_variable_to_file(windows_dev, "development_windows")
save_variable_to_file(windows_test, "testing_windows")
save_variable_to_file(training_set, "training_set_imgs")
save_variable_to_file(development_set, "development_set_imgs")
save_variable_to_file(test_set, "testing_set_imgs")

# %%
save_variable_to_file(feature_vectors_of_label, "full_feature_vectors", "saved_variables")

# %%
def show_metrics(metrics: dict) -> None:
    print("\n[+] Metrics:")
    overall_stats = [
        "F1 Macro",
        "Overall Accuracy",
        "Overall Jaccard Index",
    ]
    if (
        "Micro Averaged Jaccard Index"
        in metrics[list(metrics.keys())[0]]["Overall Statistics"].keys()
    ):
        overall_stats += ["Micro Averaged Jaccard Index"]

    class_stats = [
        "Accuracy",
        "Recall/Sensitivity",
        "Specificity",
        "Precision",
        "Averaged F1",
    ]
    branch = " │ "
    kf = TrailingFormatter()
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
            print(f"\t{branch}\t{bullet}", end="")
            print(kf.format("{:t:<22} {}", stat, value_to_print))

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
                    print(f"\t\t{branch}\t{bullet}", end="")
                    print(kf.format("{:t:<22} {}", label, round(subvalue, 3)))
            elif type(value) is tuple:
                value = (round(value[0], 3), round(value[1], 3))
                print(f"{value}")
            else:
                print(f"{round(value, 3)}")
                
classification_metrics = model.evaluate_classification_performance(
    K, classes, T, filterbank, windows_train, windows_dev, windows_test
)

show_metrics(classification_metrics)

# %%
"""Evaluación de rendimiento"""
# Rendimiento en clasificación
feature_vectors_of_label = None
filterbank = "MR8"
K_evaluation = {}
K_range = list(range(1, 21)) + list(range(30, 210, 10))
for K in K_range:
    print(f"\n[*] CURRENT K = {K}\n")
    feature_vectors_of_label, classes, T, _ = model.train(
        K,
        windows_train,
        filterbank,
        precomputed_feature_vectors=feature_vectors_of_label,
    )
    classification_metrics = model.evaluate_classification_performance(
        K, classes, T, filterbank, windows_train, windows_dev, windows_test
    )
    K_evaluation[K] = classification_metrics

# %%
_set = "Train"
stat = "Overall Accuracy"
x_lim = [0, 20]
K_range = list(range(1, 21)) + list(range(30, 210, 10))

x = K_range
y = [K_evaluation[i][_set]["Overall Statistics"][stat] for i in x]
x_highlight = 6

highlight = (x_highlight, y[x.index(x_highlight)])

plt.figure(figsize=(10, 8))
y_max = max(y)
pos = y.index(y_max)
print(f"K = {x[pos]}")
print(K_evaluation[K_range[pos]][_set]["Overall Statistics"][stat])
if _set == "Train":
    title = "Training"
elif _set == "Dev":
    title = "Development"
else:
    title = "Testing"
plt.title(title)
plt.plot(x, y, color="k", linestyle="--", marker="o")
plt.scatter(*highlight, color="r", s=100)
x = np.array(x)
plt.xlabel("K")
plt.ylabel(stat)
plt.grid(b=True, which="major", color="k", linestyle="--", alpha=0.2)
plt.grid(b=True, which="minor", color="k", linestyle="--", alpha=0.1)
plt.xlim(x_lim)
plt.xticks(x[: x.tolist().index(x_lim[1]) + 1])
plt.minorticks_on()
plt.tight_layout()
plt.savefig(_set + f" changing K (to {x_lim[-1]}) ({stat}).png", dpi=300)
plt.show()
plt.close()

# %%
"""Persistencia de variables importantes"""
# save_variable_to_file(K_evaluation, "K_evaluation")
# save_variable_to_file(T, f"texton_matrix_K_{K}")
# save_variable_to_file(windows_train, "training_windows")
# save_variable_to_file(windows_dev, "development_windows")
# save_variable_to_file(windows_test, "testing_windows")
# save_variable_to_file(training_set, "training_set_imgs")
# save_variable_to_file(development_set, "development_set_imgs")
# save_variable_to_file(test_set, "testing_set_imgs")
# save_variable_to_file(feature_vectors_of_label, "feature_vectors", "saved_variables")

# %%
# Rendimiento en segmentación
# filterbank = "MR8"
# algorithm = "felzenszwalb"
# sp_evaluation = {}
# for scale in range(50, 250, 50):
#     for sigma in [2 * i / 10 for i in range(0, 8)]:
#         for min_size in range(0, 250, 50):
#             if scale == 50:
#                 continue
#             if scale == 100 and (sigma != 0.4 or min_size != 0):
#                 continue

#             print(f"\nCurrent: (scale={scale}, sigma={sigma}, min_size={min_size})")
#             parameters = (scale, sigma, min_size)

#             print("\n[+] On training...")
#             sp_evaluation[parameters] = {}
#             sp_evaluation[parameters][
#                 "Train"
#             ] = model.evaluate_segmentation_performance(
#                 training_set,
#                 ground_truth,
#                 classes,
#                 K,
#                 T,
#                 algorithm,
#                 parameters,
#                 filterbank_name=filterbank,
#                 save_png=False,
#                 save_xlsx=False,
#                 png_title=f"Train (segm) (scale={scale}, sigma={sigma}, min_size={min_size})",
#             )
#             print("\n[+] On development...")
#             sp_evaluation[parameters]["Dev"] = model.evaluate_segmentation_performance(
#                 development_set,
#                 ground_truth,
#                 classes,
#                 K,
#                 T,
#                 algorithm,
#                 parameters,
#                 filterbank_name=filterbank,
#                 save_png=False,
#                 save_xlsx=False,
#                 png_title=f"Dev (segm) (scale={scale}, sigma={sigma}, min_size={min_size})",
#             )
#             print("\n[+] On testing...")
#             sp_evaluation[parameters]["Test"] = model.evaluate_segmentation_performance(
#                 test_set,
#                 ground_truth,
#                 classes,
#                 K,
#                 T,
#                 algorithm,
#                 parameters,
#                 filterbank_name=filterbank,
#                 save_png=False,
#                 save_xlsx=False,
#                 png_title=f"Test (segm) (scale={scale}, sigma={sigma}, min_size={min_size})",
#             )
#             print("[+] Presaving...")
#             save_variable_to_file(
#                 sp_evaluation, "sp_evaluation", "saved_variables", overwrite=True
#             )

#
# print("[+] Final save...")
# save_variable_to_file(sp_evaluation, "sp_evaluation", "saved_variables")

# %%
sp_evaluation_long = load_variable_from_file(
    "sp_evaluation_scale50_sigma_01-03_minsize_0-190", "saved_variables"
)
sp_evaluation = load_variable_from_file("sp_evaluation", "saved_variables")

# %%
stat = "Overall Accuracy"
_set = "Dev"

if _set == "Train":
    title = "Training"
elif _set == "Dev":
    title = "Development"
else:
    title = "Testing"

x = list(range(0, 250, 50))

for stat in ["Micro Averaged Jaccard Index"]:
    for _set in ["Test", "Train", "Dev"]:
        if _set == "Train":
            title = "Training"
        elif _set == "Dev":
            title = "Development"
        else:
            title = "Testing"
        for scale in range(50, 250, 50):
            plt.figure(figsize=(10, 8))
            plt.title(title + f" (scale = {scale})")

            for sigma in [2 * i / 10 for i in range(0, 8)]:
                y = []
                for min_size in x:
                    stats, jaccard = sp_evaluation[(scale, sigma, min_size)][_set]
                    if stat == "Micro Averaged Jaccard Index":
                        import statistics

                        micro_jaccard = []
                        for key in jaccard:
                            micro_jaccard.append(jaccard[key]["Micro"])

                        y.append(statistics.harmonic_mean(micro_jaccard))
                    else:
                        y.append(stats["Overall Statistics"][stat])

                plt.plot(x, y, linestyle="--", label=f"$\sigma$ = {sigma}")

            plt.xlabel("min-size")
            plt.ylabel(stat)
            plt.grid(b=True, which="major", color="k", linestyle="--", alpha=0.2)
            plt.grid(b=True, which="minor", color="k", linestyle="--", alpha=0.1)
            plt.minorticks_on()
            plt.tight_layout()
            plt.legend()
            plt.savefig(
                f"{_set} changing sigma and min_size (scale = {scale}) ({stat})",
                dpi=300,
            )
            plt.show()
            plt.close()

# %%
"""Entrenamiento del modelo

A continuación, se muestra una función que entrena el modelo, primero construyendo la 
matriz de *feature vectors* para cada una de las clases, y luego haciendo un 
*clustering* que aprenda los textones asociados a cada una de ellas.
"""
filterbank = "MR8"
scale = 100
sigma = 1.4
min_size = 100
K_evaluation_2 = {}
feature_vectors_of_label = None
for K in range(1, 20):
    print(f"\nCurrent: (K = {K})")

    feature_vectors_of_label, classes, T, _ = model.train(
        K,
        windows_train,
        precomputed_feature_vectors=feature_vectors_of_label,
        filterbank_name=filterbank,
    )

    parameters = (scale, sigma, min_size)

    print("\n[+] On training...")
    K_evaluation_2[K] = {}
    K_evaluation_2[K]["Train"] = model.evaluate_segmentation_performance(
        training_set,
        ground_truth,
        classes,
        K,
        T,
        algorithm,
        parameters,
        filterbank_name=filterbank,
        save_png=True,
        save_xlsx=False,
        png_title=f"Train (segm) (K = {K})",
    )
    print("\n[+] On development...")
    K_evaluation_2[K]["Dev"] = model.evaluate_segmentation_performance(
        development_set,
        ground_truth,
        classes,
        K,
        T,
        algorithm,
        parameters,
        filterbank_name=filterbank,
        save_png=True,
        save_xlsx=False,
        png_title=f"Dev (segm) (K = {K})",
    )
    print("\n[+] On testing...")
    K_evaluation_2[K]["Test"] = model.evaluate_segmentation_performance(
        test_set,
        ground_truth,
        classes,
        K,
        T,
        algorithm,
        parameters,
        filterbank_name=filterbank,
        save_png=True,
        save_xlsx=False,
        png_title=f"Test (segm) (K = {K})",
    )

# %%
stat = "Overall Accuracy"
_set = "Train"

if _set == "Train":
    title = "Training"
elif _set == "Dev":
    title = "Development"
else:
    title = "Testing"

x = list(range(1, 20))

for stat in ["F1 Macro", "Micro Averaged Jaccard Index"]:
    for _set in ["Test", "Train", "Dev"]:
        if _set == "Train":
            title = "Training"
        elif _set == "Dev":
            title = "Development"
        else:
            title = "Testing"

        plt.figure(figsize=(10, 8))
        plt.title(title)
        y = []
        for K in range(1, 20):
            stats, jaccard = K_evaluation_2[K][_set]
            if stat == "Micro Averaged Jaccard Index":
                import statistics

                micro_jaccard = []
                for key in jaccard:
                    micro_jaccard.append(jaccard[key]["Micro"])

                y.append(statistics.harmonic_mean(micro_jaccard))
            else:
                y.append(stats["Overall Statistics"][stat])

        plt.plot(x, y, linestyle="--")

        plt.xlabel("K")
        plt.ylabel(stat)
        plt.grid(b=True, which="major", color="k", linestyle="--", alpha=0.2)
        plt.grid(b=True, which="minor", color="k", linestyle="--", alpha=0.1)
        plt.minorticks_on()
        plt.tight_layout()
        plt.xticks(x)
        plt.savefig(
            f"{_set} changing K ({stat})", dpi=300,
        )
        plt.show()
        plt.close()

# %%
matplotlib.rcParams.update({"font.size": 16})

K = 6
filterbank = "MR8"
scale = 100
sigma = 1.4
min_size = 100
final_model = {}
feature_vectors_of_label = None

feature_vectors_of_label, classes, T, _ = model.train(
    K, windows_train, windows_dev=windows_dev, filterbank_name=filterbank,
)

parameters = (scale, sigma, min_size)

print("\n[+] On training...")
final_model["Train"] = model.evaluate_segmentation_performance(
    training_set,
    ground_truth,
    classes,
    K,
    T,
    algorithm,
    parameters,
    filterbank_name=filterbank,
    save_png=True,
    save_xlsx=False,
    png_title=f"Train (segm) (K = {K})",
)
print("\n[+] On development...")
final_model["Dev"] = model.evaluate_segmentation_performance(
    development_set,
    ground_truth,
    classes,
    K,
    T,
    algorithm,
    parameters,
    filterbank_name=filterbank,
    save_png=True,
    save_xlsx=False,
    png_title=f"Dev (segm) (K = {K})",
)
print("\n[+] On testing...")
final_model["Test"] = model.evaluate_segmentation_performance(
    test_set,
    ground_truth,
    classes,
    K,
    T,
    algorithm,
    parameters,
    filterbank_name=filterbank,
    save_png=True,
    save_xlsx=False,
    png_title=f"Test (segm) (K = {K})",
)

# %%
save_variable_to_file(final_model, "final_model")
save_variable_to_file(sp_evaluation, "sp_evaluation")
# %%
save_variable_to_file(T, "T_final", "saved_variables")
