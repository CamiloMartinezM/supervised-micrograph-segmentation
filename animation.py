# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:22:03 2020

@author: Camilo Martínez
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

fig = plt.figure(figsize=(10, 8), dpi=100)
images_filenames = []
for K in [10, 20, 40, 60, 100]:
    for s in range(0, 4):
        if K == 100 and s == 3:
            continue
        new = f"K={K}, ME={s}.png"
        filename = "Confusion matrix, (segmentation) " + new
        images_filenames.append(filename)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for img in images_filenames:
    load = plt.imread(img)
    im = plt.imshow(load, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
fig.tight_layout()
ani.save("dynamic_images_2.mp4")
