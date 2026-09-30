# Supervised Superpixel-Texton Approach for Micrograph Segmentation

![Method overview: micrographs and annotation, texton learning, superpixel-texton segmentation, and post-processing into phase fractions, pearlite spacing and strength, with model selection as a side loop](docs/pipeline-overview/pipeline-overview.png)

**Method overview.** (1) 51 optical micrographs of hypoeutectoid steels are preprocessed and annotated with windows of pearlite and proeutectoid ferrite (PF). (2) The MR8 filter responses of the annotated pixels are clustered per class into K textons. (3) A new micrograph is cut into superpixels: every pixel votes for the class of its nearest texton, each superpixel takes the majority class, and pearlite is then split into ferrite and cementite. (4) The segmentation gives phase fractions, the pearlite interlamellar spacing from the 2-D DFT spectrum, and yield and tensile strength from structure–property relations. A side loop (orange) chooses K and the Felzenszwalb superpixel parameters by 3-fold cross-validation on the micro-averaged Jaccard index. The thumbnails are outputs of this code; [docs/pipeline-overview](docs/pipeline-overview) rebuilds the figure.

**Version 1.0.0**

This repository holds the code and final model for the bachelor thesis titled "[Application of Computer Vision in the Analysis of Microstructures and Obtaining Structure-Property Relations](https://repositorio.uniandes.edu.co/entities/publication/60ccf854-b041-441b-b7f7-8a79662c4bc1)" (2021). This code requires Python 3.6 or newer.

# Project structure
- `model.py` does the "heavylifting", from the loading and pre-processing of images to the training and testing of a segmentation model and the post-processing of segmented micrographs (which include the estimation of the mean apparent interlaminar spacing of pearlite and the mechanical properties, based on that value and the volume fractions of the steel phases and morphologies present).
- `utils_classes.py` and `utils_functions.py` contain helper classes and functions, respectively.
- `app.py` is a console with the most important functions in `model.py`.

     ![Console menu](final_model/console_menu.png "Console menu")
     
### Requirements
Refer to [requirements.txt](requirements.txt).

# Segmentation model
![First part of segmentation model](final_model/image_segmentation_firstpart.png "First part of a micrograph segmentation")
![Second part of segmentation model](final_model/image_segmentation_parttwo.png "Second part of a micrograph segmentation")

# Model selection tool
The console (`app.py`) includes an option to create a brand-new model for segmentation.
### Changeable parameters
- **Labeled images**: If the user has labeled images with the correct format, a folder of annotated images can be specified, thus creating a segmentation model for those classes and images, and not steel micrographs.
- **K**: Number of textons, i.e, number of clusters for *K-Means clustering*.
- **Superpixel-generation algorithm**: Possibilities include quickshift, slic, felzenszwalb and watershed. Depending on the chosen algorithm, the user can change the required parameters to specify.
### Flow diagram
![Model selection tool](final_model/model_selection.png "Flow diagram of the model selection tool")

# Notes
* `saved_variables/ground_truth.pickle` and `ground_truth_with_originals.pickle` were deleted because of space.
  
# License & copyright
© Camilo Martínez M., <ca.martinezm2@uniandes.edu.co>

Licensed under [GPL v3](LICENSE).
