# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:22 2020

@author: Camilo Martínez
"""
import model
from utils_functions import (
    compare2imgs,
    find_path_of_img,
    load_img,
    train_dev_test_split,
    train_dev_test_split_table,
)

print(f"\nPath to labeled micrographs: {model.PATH_LABELED}")
print(f"Path to preprocessed micrographs: {model.PATH_PREPROCESSED}")

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

"""Carga de las imágenes

Las imágenes son cargadas en un arreglo de numpy que tiene todas las imágenes, llamado 
micrographs. Dado que es un arreglo, no hay forma directa de saber a qué imagen le 
corresponde cada posición para poder extraer fácilmente el arreglo correspondiente. Por
ello, se usa un diccionario llamado index_to_name (el nombre de variable está invertido;
debería ser en realidad name_to_index), cuyas llaves son los nombres de las imágenes 
.png y los valores son los índices correspondientes en micrographs.
"""
micrographs, index_to_name = model.load_imgs(exclude=["Low carbon"])

"""Carga de las escalas"""
micrographs_scales = model.load_scales()

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

"""Extracción de las ventanas/regiones anotadas

Dentro de PATH_LABELED, cada una de las imágenes tiene asociada un archivo .txt que 
contiene la información de la posición de cada una de sus regiones o ventanas que fueron
anotadas.
"""
labels, windows, windows_per_name = model.extract_labeled_windows(
    micrographs, index_to_name, exclude=["Low carbon"]
)

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
) = train_dev_test_split(windows, train_size=0.7, dev_size=0.2)

# Table that summarizes the number of windows noted by class or label that correspond
# to the phases or morphologies of interest in the micrographs.
train_dev_test_split_table(windows_train, windows_dev, windows_test)

"""Entrenamiento del modelo

A continuación, se muestra una función que entrena el modelo, primero construyendo la 
matriz de *feature vectors* para cada una de las clases, y luego haciendo un 
*clustering* que aprenda los textones asociados a cada una de ellas.
"""
K = 100
feature_vectors_of_label, classes, T, _ = model.train(K, windows_train)


"""Carga de las imágenes ground truth.

Las imágenes segmentadas (ground truth) se guardan en un diccionario cuyas llaves son los
nombres de las imágenes y los valores son las respectivas segmentaciones.
"""
ground_truth = model.load_ground_truth("(Segmented)HypoeutectoidSteel.tif", classes)

"""Segmentación

En primer lugar, se obtienen los superpíxeles de la imagen de prueba, al igual que los 
*feature vectors* de cada uno de ellos. 

S (dict): Superpixeles.
S_feature_vectors (dict): Feature vectors de cada superpixel en S.

Esta implementación obtiene un diccionario cuyas llaves corresponden a los superpíxeles 
y los valores a la clase a la que dicho superpíxel pertenece. El fundamento matemático 
está basado en una decisión de clasificación colectiva de cada superpíxel basada en las
ocurrencias de los textones más cercanos de todos los píxeles en el superpíxel.
"""
original_img, superpixels, segmentation = model.segment(
    find_path_of_img("cs0327.png", model.PATH_LABELED),
    classes,
    T,
    n_segments=500,
    compactness=0.17,
    plot_original=True,
    plot_superpixels=True,
    verbose=True,
)

"""Evaluación de rendimiento"""
model.evaluate_classification_performance(
    K,
    classes,
    windows_train,
    windows_dev,
    windows_test,
    plot_fig=False,
)

model.evaluate_segmentation_performance(
    training_set,
    ground_truth,
    classes,
    K,
    n=1000,
    sigma=5,
    compactness=0.17,
    plot_fig=False,
)
model.visualize_segmentation(original_img, classes, superpixels, segmentation)
