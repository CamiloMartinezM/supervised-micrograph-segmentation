# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 06:45:29 2020

@author: Camilo Martínez
"""
import os
from itertools import chain, product
from numba import jit

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

# from joblib import Parallel, delayed
from scipy.signal import fftconvolve
from skimage.segmentation import mark_boundaries, slic

from utils_functions import get_folder, load_img


class Scaler:
    """For each of the micrographs there is a scale to which is associated a calibration line
    (which naturally has a length in pixels) and a real length in micrometers. The following
    class obtains a dictionary of 'scales', with keys that are the names of each micrograph
    and the value is the corresponding length of the scale in pixels. Thus, this class
    contains the information regarding the scale of every micrograph in LABELED.
    """

    def __init__(
        self,
        startpath: str,
        path_preprocessed: str,
        split_in_half: bool = True,
        minimum_zeros_check: int = 100,
    ) -> None:
        """
        Args:
            startpath (str): Directory where labeled micrographs lie.
            split_in_half (bool, optional): If True, the algorithm will split the image in 2 by its
                                            width to speed up the process. Defaults to True.
            minimum_zeros_check (int, optional): Minimum zeroes to consider a line to be the actual
                                                 scale. Defaults to 100.
        """
        self.startpath = startpath
        self.PATH_PREPROCESSED = path_preprocessed
        self.split_in_half = split_in_half
        self.minimum_zeros_check = minimum_zeros_check

    def get_scale(self, name: str, folder: str) -> np.ndarray:
        """Loads the "scale" image associated with the given name.

        Args:
            name (str): Name of micrograph.
            folder (str): High carbon, Medium carbon or Low carbon.

        Returns:
            np.ndarray: Loaded image. None if an error occurs.
        """
        try:
            return load_img(
                os.path.join(self.PATH_PREPROCESSED, folder, "SCALE_" + name)
            )
        except:
            return None

    def get_pixel_length(self, scale: np.ndarray) -> int:
        """Gets the pixel length of the scale depicted in the "scale" image.

        Args:
            scale (np.ndarray): Loaded image.

        Returns:
            int: Number of pixels in the scale.
        """
        sc = scale.copy()
        if self.split_in_half:
            sc = self.split_scale(sc)

        pixels = 0  # Number of pixels in scale, i.e, scale length in pixels.
        height = scale.shape[0]
        width = scale.shape[1]
        for i in range(height):
            for j in range(width):
                if scale[i][j] == 0:
                    for k in range(1, self.minimum_zeros_check + 1):
                        if j + k >= width or scale[i][j + k] != 0:
                            break
                    else:  # No break
                        finished = False
                        z = j + self.minimum_zeros_check
                        pixels = self.minimum_zeros_check
                        while not finished and z < width:
                            if scale[i][z] == 0:
                                pixels += 1
                            else:
                                finished = True
                            z += 1
                        # Found pixel length
                        return pixels

        return 0  # If unable to find pixel length

    def split_scale(self, scale: np.ndarray) -> np.ndarray:
        """Splits the "scale" image by its width.

        Args:
            scale (np.ndarray): Scale of an image as a numpy array.

        Returns:
            np.ndarray: Scale of the image, split in half.
        """
        return scale[:, scale.shape[1] // 2 :]

    def process(self) -> None:
        """Deals with all images inside given startpath."""
        self.scales = dict()
        count = 0
        for path, _, files in os.walk(self.startpath):
            for f in files:
                if f.endswith(".png"):
                    print("[+] Processing " + str(f) + "... ", end="")
                    folder = get_folder(path)
                    if folder is not None:
                        scale = self.get_scale(f, folder)
                        if scale is not None:
                            self.scales[f[:-4]] = self.get_pixel_length(scale)
                            print("Done")
                            count += 1
                        else:
                            print("Failed. Got None as scale.")
                    else:
                        print("Failed. Got None as folder.")
        print(f"\nFinished. {count} processed.")


class SLICSegmentation:
    """SLIC algorithm implementation with the library skimage."""

    def __init__(
        self, n_segments: int = 500, sigma: int = 5, compactness: int = 0.1,
    ) -> None:
        """
        Args:
            n_segments (int, optional): Approximate number of superpixels to create.
            sigma, compactness (int, optional): Optimizable parameters of the algorithm.
        """
        self.n_segments = n_segments
        self.sigma = sigma
        self.compactness = compactness

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segments the input image in superpixels.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Array of superpixels with 0 <= values < n_segments.
        """
        segments = slic(
            image,
            n_segments=self.n_segments,
            sigma=self.sigma,
            compactness=self.compactness,
            start_label=1,
        )
        return segments

    def plot_output(
        self, image: np.ndarray, segments: np.ndarray, dpi: int = 80, ax=None
    ) -> None:
        """Shows the output of SLIC."""
        if ax is not None:
            ax.imshow(mark_boundaries(image, segments, color=(0, 0, 255)))
            ax.axis("off")
            return ax

        plt.figure(dpi=dpi)
        plt.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
        return None


# Maximum Response filterbank from
# http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
# based on several edge and bar filters.
# Adapted to Python by Andreas Mueller amueller@ais.uni-bonn.de
# https://gist.github.com/amueller/3129692
class FilterBankMR8:
    def __init__(self, sigmas: list, n_orientations: int) -> None:
        n_sigmas = len(sigmas)

        self.edge, self.bar, self.rot = FilterBankMR8.makeRFSfilters(
            sigmas=sigmas, n_orientations=n_orientations
        )
        self.filterbank = chain(self.edge, self.bar, self.rot)

        self.n_sigmas = n_sigmas
        self.sigmas = sigmas
        self.n_orientations = n_orientations

    def response(self, img: np.ndarray) -> np.ndarray:
        return FilterBankMR8.apply_filterbank(img, self.filterbank)

    @staticmethod
    def makeRFSfilters(
        radius: int = 24, sigmas: list = [1, 2, 4], n_orientations: int = 6
    ):
        """Generates filters for RFS filterbank.

        Args:
            radius : int, default 28
                radius of all filters. Size will be 2 * radius + 1
            sigmas : list of floats, default [1, 2, 4]
                define scales on which the filters will be computed
            n_orientations : int
                number of fractions the half-angle will be divided in
        Returns:
            edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
                Contains edge filters on different scales and orientations
            bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
                Contains bar filters on different scales and orientations
            rot : ndarray (2, 2*radius+1, 2*radius+1)
                contains two rotation invariant filters, Gaussian and Laplacian of
                Gaussian
        """

        def make_gaussian_filter(x, sigma, order=0):
            if order > 2:
                raise ValueError("Only orders up to 2 are supported")
            # compute unnormalized Gaussian response
            response = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
            if order == 1:
                response = -response * x
            elif order == 2:
                response = response * (x ** 2 - sigma ** 2)
            # normalize
            response /= np.linalg.norm(response, 1)
            return response

        def makefilter(scale, phasey, pts, sup):
            gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
            gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
            f = (gx * gy).reshape(sup, sup)
            # normalize
            f /= np.linalg.norm(f, 1)
            return f

        support = 2 * radius + 1
        x, y = np.mgrid[-radius : radius + 1, radius : -radius - 1 : -1]
        orgpts = np.vstack([x.ravel(), y.ravel()])

        rot, edge, bar = [], [], []
        for sigma in sigmas:
            for orient in range(n_orientations):
                # Not 2pi as filters have symmetry
                angle = np.pi * orient / n_orientations
                c, s = np.cos(angle), np.sin(angle)
                rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
                edge.append(makefilter(sigma, 1, rotpts, support))
                bar.append(makefilter(sigma, 2, rotpts, support))

        length = np.sqrt(x ** 2 + y ** 2)
        rot.append(make_gaussian_filter(length, sigma=10))
        rot.append(make_gaussian_filter(length, sigma=10, order=2))

        # reshape rot and edge
        edge = np.asarray(edge)
        edge = edge.reshape(len(sigmas), n_orientations, support, support)
        bar = np.asarray(bar).reshape(edge.shape)
        rot = np.asarray(rot)[:, np.newaxis, :, :]
        return edge, bar, rot

    def plot_filters(self) -> None:
        # plot filters
        # 2 is for bar / edge, + 1 for rot
        fig, ax = plt.subplots(self.n_sigmas * 2 + 1, self.n_orientations)
        fig.set_dpi(150)
        for k, filters in enumerate([self.bar, self.edge]):
            for i, j in product(range(self.n_sigmas), range(self.n_orientations)):
                row = i + k * self.n_sigmas
                ax[row, j].imshow(
                    filters[i, j, :, :], cmap=matplotlib.cm.get_cmap("gray")
                )
                ax[row, j].set_xticks(())
                ax[row, j].set_yticks(())
        ax[-1, 0].imshow(self.rot[0, 0], cmap=matplotlib.cm.get_cmap("gray"))
        ax[-1, 0].set_xticks(())
        ax[-1, 0].set_yticks(())
        ax[-1, 1].imshow(self.rot[1, 0], cmap=matplotlib.cm.get_cmap("gray"))
        ax[-1, 1].set_xticks(())
        ax[-1, 1].set_yticks(())
        for i in range(2, self.n_orientations):
            ax[-1, i].set_visible(False)
        fig.tight_layout()
        fig.savefig("filters.png", dpi=300)

    @staticmethod
    @jit
    def apply_filterbank(img: np.ndarray, filterbank: tuple) -> np.ndarray:
        result = np.zeros((8, *img.shape))
        for i, battery in enumerate(filterbank):
            response = [fftconvolve(img, filt, mode="same") for filt in battery]
            # response = Parallel(n_jobs=-1)(
            #     delayed(scipy.ndimage.convolve)(img, filt) for filt in battery
            # )
            max_response = np.max(response, axis=0)
            result[i] = max_response

        return result


class MultiscaleStatistics:
    """This class generates the feature vectors of an image based on multiscale statistics in a
    procedure described in
    G. Impoco, L. Tuminello, N. Fucà, M. Caccamo, G. Licitra,
    Segmentation of structural features in cheese micrographs using pixel statistics,
    Computers and Electronics in Agriculture,
    Volume 79, Issue 2,
    2011,
    Pages 199-206,
    ISSN 0168-1699,
    https://doi.org/10.1016/j.compag.2011.09.013.
    """

    def __init__(self, scales: int = 3):
        """
        Args:
            scales (int, optional): Number of scales to consider. Each scale represents a
                                    neighborhood around a pixel. The first scale is the inmediate
                                    neighborhood of a pixel, the second one is the neighborhood
                                    around the previous one and so on. Defaults to 3.
        """
        self.scales = scales

    def process(self, img: np.ndarray) -> np.ndarray:
        return MultiscaleStatistics.process_helper(img, self.scales)

    @staticmethod
    @jit
    def process_helper(img: np.ndarray, scales: int) -> np.ndarray:
        """Generates the feature vectors of an image.

        Args:
            img (np.ndarray): Image to calculate feature vectors.

        Returns:
            np.ndarray: Feature vectors of the input image of shape (*img.shape, 4*3*scales).
        """
        padded_img = np.pad(img, scales, mode="constant")
        gx, gy = np.gradient(padded_img)
        feature_vectors = np.zeros((img.shape[0] * img.shape[1], 4 * 3 * scales))
        z = 0
        for i in range(scales, padded_img.shape[0] - scales):
            for j in range(scales, padded_img.shape[1] - scales):
                for scale in range(1, scales + 1):
                    N = padded_img[i - scale, j - scale : j + scale + 1]
                    E = padded_img[i - scale : i + scale + 1, j + scale]
                    S = padded_img[i + scale, j - scale : j + scale + 1]
                    W = padded_img[i - scale : i + scale + 1, j - scale]

                    N_gx = gx[i - scale, j - scale : j + scale + 1]
                    E_gx = gx[i - scale : i + scale + 1, j + scale]
                    S_gx = gx[i + scale, j - scale : j + scale + 1]
                    W_gx = gx[i - scale : i + scale + 1, j - scale]

                    N_gy = gy[i - scale, j - scale : j + scale + 1]
                    E_gy = gy[i - scale : i + scale + 1, j + scale]
                    S_gy = gy[i + scale, j - scale : j + scale + 1]
                    W_gy = gy[i - scale : i + scale + 1, j - scale]

                    neighbors = np.vstack((N, E, S, W))
                    avgs = np.mean(neighbors, axis=1)

                    neighbors_gx = np.vstack((N_gx, E_gx, S_gx, W_gx))
                    grads_x = np.mean(neighbors_gx, axis=1)

                    neighbors_gy = np.vstack((N_gy, E_gy, S_gy, W_gy))
                    grads_y = np.mean(neighbors_gy, axis=1)

                    feature_vectors[z, 4 * 3 * (scale - 1) : 4 * 3 * scale] = np.ravel(
                        (avgs, grads_x, grads_y), "F"
                    )
                z += 1
        return feature_vectors.reshape((*img.shape, 4 * 3 * scales))
