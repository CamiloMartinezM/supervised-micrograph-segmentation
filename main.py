# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:22 2020

@author: Camilo Martínez
"""
import model
import matplotlib.pyplot as plt
from utils_functions import (
    find_path_of_img,
    load_img,
    compare2imgs,
    train_dev_test_split,
    train_dev_test_split_table,
)
from utils_classes import SLICSegmentation

print("\nPath to labeled micrographs: ", model.PATH_LABELED)
print("Path to preprocessed micrographs: ", model.PATH_PREPROCESSED)

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
"""Carga de las escalas"""

micrographs_scales = model.load_scales()

# %%
"""Carga de las imágenes

Las imágenes son cargadas en un arreglo de numpy que tiene todas las imágenes, llamado 
micrographs. Dado que es un arreglo, no hay forma directa de saber a qué imagen le 
corresponde cada posición para poder extraer fácilmente el arreglo correspondiente. Por
ello, se usa un diccionario llamado index_to_name (el nombre de variable está invertido;
debería ser en realidad name_to_index), cuyas llaves son los nombres de las imágenes 
.png y los valores son los índices correspondientes en micrographs.
"""
micrographs, index_to_name = model.load_imgs(
    as_255=False, include_only=["Hypoeutectoid steel"]
)


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
    micrographs, index_to_name, include_only=["Hypoeutectoid steel"]
)

# %%
"""Train-dev-test split
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

TRAIN_SIZE = 0.7
DEV_SIZE = 0.2

windows_train, windows_dev, windows_test = train_dev_test_split(
    windows, TRAIN_SIZE, DEV_SIZE
)

# Table that summarizes the number of windows noted by class or label that correspond
# to the phases or morphologies of interest in the micrographs.
train_dev_test_split_table(windows_train, windows_dev, windows_test)

# %%
""" Ventanas a vector por clase/anotación
Cada píxel de cada ventana anotada por cada una de las clases tiene 8 respuestas 
asociadas a los filtros usados. Dichas respuestas deben ser unificadas de alguna forma, 
pues hacen parte de la misma clase. Por ello, se propone la siguiente implementación que 
convierte cada una de las respuestas obtenidas por ventana en una matriz donde cada fila
es un píxel de una anotación. Y, dado que cada una de las anotaciones tiene 8 respuestas asociadas, cada píxel es representado por un vector de 8 dimensiones. Esto significa que
cada fila tendrá 8 columnas, correspondientes al valor obtenido del filtro. Por otra 
parte, dado que son varias clases, dicha matriz será guardada en un diccionario, cuyas 
llaves serán las clases encontradas.
```
feature_vectors_of_label[label] = feature_vectors_matrix
feature_vectors_of_label[label].shape = (NUM_PIXELES, 8)
```
Donde NUM_PIXELES sería el número total de píxeles obtenidos de todas las ventanas 
anotadas.

A continuación, se ilustra esto con un ejemplo sencillo:

Sea A,B∈N^2:

> $$A = \begin{bmatrix} 
    1      & 2 & 3  \\
    4      & 5 & 6  \\
    7      & 8 & 9  \\
\end{bmatrix};  B = \begin{bmatrix} 
    11      & 12 & 13  \\
    14      & 15 & 16  \\
    17      & 18 & 19  \\
\end{bmatrix}$$

Lo que se quiere es una matriz resultado C tal que:

> $$C = \begin{bmatrix}
    [1,11] & [2,12] & [3,13] \\
    [4,14] & [5,15] & [6,16] \\
    [7,17] & [8,18] & [9,19] \\
\end{bmatrix}$$

Lo anterior es esencialmente lo que se hace para las respuestas de cada una de las 
ventanas, excepto porque no son 2 matrices, A y B, sino 8, que corresponden a las 8
respuestas obtenidas del banco de filtros. Es decir, la matriz C en dicho caso no 
tendría solo 2 elementos por posición en la matriz, sino 8 elementos.

Luego, como hay varias ventanas por anotación, es decir, varios recuadros anotados con 
la misma label, cada una de las matrices resultado de la expansión de dimensiones a 8
son concatenadas de la siguiente forma:

Sea C como fue definida anteriormente y D∈N^2 como:

> $$D = \begin{bmatrix}
    [21,11] & [22,12] \\ 
    [24,14] & [25,15] \\
    [27,17] & [28,18] \\
    [30,20] & [31,21] \\
\end{bmatrix}$$

Entonces se quiere una matriz resultado E tal que:

> $$ E = \begin{bmatrix}
    C \\ 
    D \\
\end{bmatrix} = \begin{bmatrix}
    [1,11] \\ 
    [4,14] \\
    ⋮ \\
    [9,19] \\
    [21,11] \\
    [24,14] \\
    ⋮ \\
    [31,21]
\end{bmatrix}$$

Recuérdese que C y D corresponden a las matrices correspondientes a dos ventanas 
distintas, pero de la misma clase. Esto es esencialmente lo que se hace para cada una 
de las clases anotadas, excepto que E no tendría 2 elementos por posición en la 
matriz, sino 8. 

Al final, se tendrá un vector de características de 8 dimensiones para cada una de las 
clases, en el cual cada fila o ítem corresponde a la representación vectorial de un 
píxel en las ventanas. Es decir, el número de filas o posiciones que tendría este vector 
sería igual al número total de píxeles anotados.
"""
feature_vectors = model.get_feature_vectors_of_labels(windows_train, verbose=True)

# %%
"""Entrenamiento del modelo
A continuación, se muestra una función que entrena el modelo, primero construyendo la 
matriz de *feature vectors* para cada una de las clases, y luego haciendo un 
*clustering* que aprenda los textones asociados a cada una de ellas.
"""
K = 9
(feature_vectors, classes, T, _) = model.train(
    K,
    windows_train,
    None,
    feature_vectors,
    compute_clustering_entropy=False,
    verbose=False,
)

# %%
"""Segmentación con SLIC
A continuación, se muestra un ejemplo de lo que produciría una segmentación con SLIC."""
test_img = "cs0327.png"

print("Original image:")
plt.figure(dpi=80)
plt.imshow(load_img(find_path_of_img(test_img, model.PATH_LABELED)), cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.savefig("original.png", dpi=300)
plt.show()
plt.close()

print("\nSuperpixeles:")
img = load_img(
    find_path_of_img(test_img, model.PATH_LABELED), as_255=False, with_io=True
)
SLIC = SLICSegmentation()
segments = SLIC.segment(img)
SLIC.plot_output(img, segments)

# %%
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
model.evaluate_classification_performance(
    classes,
    K,
    T,
    windows_train,
    windows_dev,
    windows_test,
    plot=False,
    max_test_number=5,
)
model.evaluate_segmentation_performance(
    classes,
    K,
    T,
    windows_per_name,
    plot=False,
    max_test_number=5,
    n=1000,
    compactness=0.17,
)

# %%
"""Visualización
Las siguientes líneas de código permiten visualizar el resultado de la segmentación,
sobreponiendo la imagen original con colores que representan cada clase.

Primero se obtienen los nuevos segmentos de la imagen a partir del resultado de la 
segmentación. El siguiente código asigna a new_segments[i, j], correspondiente al píxel
i,j de S, la posición de la clase (según la organización del arreglo classes) del 
superpíxel (S_segmented[superpixel]).

new_segments = np.zeros((DESIRED_HEIGHT, DESIRED_WIDTH), dtype=np.int16)
for superpixel in S:
    for i, j in S[superpixel]:
        new_segments[i, j] = np.where(np.array(classes) == S_segmented[superpixel])[
            0
        ].flatten()

Ahora, se construye la imagen overlay de tamaño (DESIRED_HEIGHT, DESIRED_WIDTH) que 
tendrá los colores de las clases y servirá para distinguir cada una sobre la imagen 
original."""
original_img, superpixels, segmentation = model.segment(
    find_path_of_img("cs0330.png", model.PATH_LABELED),
    classes,
    T,
    plot_original=False,
    plot_superpixels=True,
    n_segments=500,
    compactness=0.17,
    verbose=True
)

model.visualize_segmentation(classes, original_img, superpixels, segmentation)
