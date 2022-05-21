# -*- coding: utf-8 -*-
import numpy as np
from utils.functions import (
    absolute_file_paths,
    load_img,
    highlight_class_in_img,
)

LABEL_MAPPING = {
    0: "Matrix",
    1: "Austenite",
    2: "Martensite/Austenite",
    3: "Precipitate",
    4: "Defect",
}

def MetalDAM_dataset():
    labels_files = absolute_file_paths("datasets/MetalDAM/labels")
    images_files = absolute_file_paths("datasets/MetalDAM/images")
    
    assert len(labels_files) == len(images_files)
    
    windows_dict = {}
    
    labels_imgs = []
    images = []
    for i in range(len(images_files)):
        label_img = load_img(labels_files[i], as_255=True)
        image = load_img(images_files[i])
    
        if label_img.shape[0] != image.shape[0]:
            image = image[: label_img.shape[0], :]
    
        assert label_img.shape == image.shape
    
        for label in np.unique(label_img):
            highlighted_label_in_img = highlight_class_in_img(image, label_img, label)
            if label not in windows_dict:
                windows_dict[label] = []
    
            windows_dict[label].append(highlighted_label_in_img)
            
        labels_imgs.append(label_img)
        images.append(image)
    
    for key, mapping in LABEL_MAPPING.items():
        if key in windows_dict:
            windows_dict[mapping] = windows_dict.pop(key)
            
    return images, labels_imgs, windows_dict

"""  
a = images_files[1])[0:703, :]
b = load_img(labels_files[1], as_255 = True)

# show_img(a)

plt.figure()
plt.imshow(a, cmap="gray")
plt.imshow(b, alpha=0.25)
plt.show()
plt.close()
    
plt.figure()
plt.imshow(c)
plt.show()
plt.close()    
# MetalDAM_dataset()
"""
