# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 06:45:29 2020

@author: Camilo Martínez
"""
import os
from itertools import chain, product
from string import Formatter

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from skimage.filters import sobel
from skimage.segmentation import (
    felzenszwalb,
    mark_boundaries,
    quickshift,
    slic,
    watershed,
)

from utils.functions import find_path_of_img, get_folder, load_img


class Scaler:
    """For each of the micrographs there is a scale to which is associated a calibration
    line (which naturally has a length in pixels) and a real length in micrometers. The
    following class obtains a dictionary of 'scales', with keys that are the names of
    each micrograph and the value is the corresponding length of the scale in pixels.
    Thus, this class contains the information regarding the scale of every micrograph
    in LABELED.
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
            split_in_half (bool, optional): If True, the algorithm will split the image
                                            in 2 by its width to speed up the process.
                                            Defaults to True.
            minimum_zeros_check (int, optional): Minimum zeroes to consider a line to be
                                                 the actual scale. Defaults to 100.
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
                if f.endswith(".png") and not f.startswith("SCALE_"):
                    print("[+] Processing " + str(f) + "... ", end="")
                    folder = get_folder(find_path_of_img(f, self.PATH_PREPROCESSED))
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


class SuperpixelSegmentation:
    """Superpixel algorithm implementation with the library skimage."""

    def __init__(self, algorithm: str, parameters: tuple,) -> None:
        """
        # Args:
            n_segments (int, optional): Approximate number of superpixels to create.
            sigma, compactness (int, optional): Optimizable parameters of the algorithm.
        """
        self.algorithm = algorithm
        self.parameters = parameters

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segments the input image in superpixels.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Array of superpixels with 0 <= values < n_segments.
        """
        if self.algorithm == "slic":
            n_segments, sigma, compactness = self.parameters
            segments = slic(
                image,
                n_segments=n_segments,
                sigma=sigma,
                compactness=compactness,
                start_label=1,
            )
        elif self.algorithm == "felzenszwalb":
            scale, sigma, min_size = self.parameters
            segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
        elif self.algorithm == "quickshift":
            ratio, kernel_size, max_dist, sigma = self.parameters
            segments = quickshift(
                image,
                ratio=ratio,
                kernel_size=kernel_size,
                max_dist=max_dist,
                sigma=sigma,
                convert2lab=False,
            )
        elif self.algorithm == "watershed":
            gradient = sobel(image)
            markers, compactness = self.parameters
            segments = watershed(gradient, markers=markers, compactness=compactness)
        else:
            segments = None

        return segments

    def plot_output(
        self, image: np.ndarray, segments: np.ndarray, dpi: int = 80, ax=None
    ) -> None:
        """Shows the output of SLIC."""
        if ax is not None:
            ax.imshow(mark_boundaries(image, segments, color=(0, 0, 255)))
            ax.axis("off")
            return ax

        plt.figure(figsize=(10, 8), dpi=dpi)
        plt.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        plt.tight_layout()
        plt.pause(0.05)
        # plt.show()
        # plt.close()
        return None


# Maximum Response filterbank from
# http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
# based on several edge and bar filters.
# Adapted to Python by Andreas Mueller amueller@ais.uni-bonn.de
# https://gist.github.com/amueller/3129692
class FilterBank:
    def __init__(self, name) -> None:
        if name == "MR8":
            sigmas, n_orientations = [1, 2, 4], 6
            n_sigmas = len(sigmas)

            self.edge, self.bar, self.rot = FilterBank.makeRFSfilters(
                sigmas=sigmas, n_orientations=n_orientations
            )
            self.filterbank = chain(self.edge, self.bar, self.rot)

            self.n_sigmas = n_sigmas
            self.sigmas = sigmas
            self.n_orientations = n_orientations
            self.n_filters = 8
        elif name == "MAT":
            self.filterbank = loadmat("filterbank.mat")["filterbank"].transpose(
                (2, 0, 1)
            )
            self.n_filters = self.filterbank.shape[0]
        else:
            self.filterbank = None

        self.name = name

    def response(self, img: np.ndarray, use_fftconvolve: bool = True) -> np.ndarray:
        if self.name == "MR8":
            return FilterBank.apply_filterbankMR8(img, self.filterbank, use_fftconvolve)
        elif self.name == "MAT":
            return FilterBank.apply_filterbankMAT(img, self.filterbank, use_fftconvolve)
        else:
            return None

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

    def plot_filters(self, dpi: int = 80) -> None:
        # plot filters
        # 2 is for bar / edge, + 1 for rot
        fig, ax = plt.subplots(self.n_sigmas * 2 + 1, self.n_orientations)
        fig.set_dpi(dpi)
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
        plt.show()
        plt.close()
        # fig.savefig("filters.png", dpi=dpi)

    @staticmethod
    def apply_filterbankMR8(
        img: np.ndarray, filterbank: tuple, use_fftconvolve: bool
    ) -> np.ndarray:
        result = np.zeros((8, *img.shape))
        for i, battery in enumerate(filterbank):
            if use_fftconvolve:
                response = [
                    fftconvolve(img, np.flip(filt), mode="same") for filt in battery
                ]
            else:
                response = Parallel(n_jobs=-1)(
                    delayed(convolve)(img, np.flip(filt)) for filt in battery
                )
            max_response = np.max(response, axis=0)
            result[i] = max_response

        return result

    @staticmethod
    def apply_filterbankMAT(
        img: np.ndarray, filterbank: np.ndarray, use_fftconvolve: bool
    ) -> np.ndarray:
        if use_fftconvolve:
            result = np.dstack(
                [fftconvolve(img, np.flip(filt), mode="same") for filt in filterbank]
            )
        else:
            result = Parallel(n_jobs=-1)(
                delayed(convolve)(img, np.flip(filt)) for filt in filterbank
            )

        result = result.transpose((2, 0, 1))
        return result

    @staticmethod
    def _loadmat(filename: str) -> np.ndarray:
        return loadmat("filterbank.mat")["filterbank"]


class Preprocessor:
    """In charge of preprocessing the micrographs, which includes cropping them to the desired
    size and cropping the scale."""

    def __init__(
        self,
        startpath: str,
        endpath: str,
        CROP_CRITERION: float = 0.1,
        DESIRED_SIZE: tuple = (500, 500),
    ) -> None:
        """
        Args:
            startpath (str): Directory where "to label" micrographs lie.
            test (bool): If True, a new directory will be created called "Test".
            If False, the micrographs in content/data/ will be modified.
        """
        self.startpath = startpath
        self.endpath = endpath
        self.CROP_CRITERION = CROP_CRITERION
        self.DESIRED_HEIGHT, self.DESIRED_WIDTH = DESIRED_SIZE

    def crop_scale(self, img: np.ndarray) -> tuple:
        """Returns the image and its scale separately."""
        height = img.shape[0]
        scale = img[int(height - self.CROP_CRITERION * height) :, :]
        crop_img = img[: int(height - self.CROP_CRITERION * height), :]
        return scale, crop_img

    def save_to_file(self, name: str, folder: str, img: np.ndarray) -> None:
        """Saves the image with given name and folder to endpath."""
        plt.imsave(os.path.join(self.endpath, folder, name), img, cmap="gray")

    def is_possible_crop_to_size(self, img: np.ndarray) -> bool:
        """Returns true if the dimensions of the given image are greater than or equal to the
        desired one.
        """
        height, width = img.shape
        if height >= self.DESIRED_HEIGHT and width >= self.DESIRED_WIDTH:
            return True
        else:
            return False

    def crop_to_size(self, img: np.ndarray) -> np.ndarray:
        """Crops the given image to the desired size. This will only work if
        is_possible_crop_to_size(img) returns True."""
        height = img.shape[0]
        width = img.shape[1]
        middle = (height // 2, width // 2)
        return img[
            middle[0] - self.DESIRED_HEIGHT // 2 : middle[0] + self.DESIRED_HEIGHT // 2,
            middle[1] - self.DESIRED_WIDTH // 2 : middle[1] + self.DESIRED_WIDTH // 2,
        ]

    def process(self) -> None:
        """Deals with all micrographs in startpath."""
        for path, _, files in os.walk(self.startpath):
            for f in files:
                if f.endswith(".png"):
                    print("[+] Processing " + str(f) + "... ", end="")
                    folder = get_folder(path)
                    if not os.path.isdir(os.path.join(self.endpath, folder)):
                        os.mkdir(os.path.join(self.endpath, folder))

                    if f not in os.listdir(os.path.join(self.endpath, folder)):
                        try:
                            img = load_img(os.path.join(path, f))
                            scale, img = self.crop_scale(img)
                            if self.is_possible_crop_to_size(img):
                                crop_img = self.crop_to_size(img)
                                self.save_to_file(f, folder, crop_img)
                                self.save_to_file("SCALE_" + f, folder, scale)
                                assert crop_img.shape == (
                                    self.DESIRED_HEIGHT,
                                    self.DESIRED_WIDTH,
                                )
                                print("Done")
                            else:
                                print("Failed. Not possible to crop.")
                        except:
                            print("Something went wrong.")
                    else:
                        print("Preprocessed before.")


class Material:
    def __init__(
        self,
        fa: float,
        S_0: float,
        p_C: float,
        p_Mn: float = 0,
        D_a: float = 0,
        p_N: float = 0,
        p_P: float = 0,
        p_Si: float = 0,
    ) -> None:
        if fa > 0.1:
            value_1 = 77.7 + 59.5 * p_Mn
            if D_a != 0:
                value_1 += 9.1 * (D_a ** (-0.5))

            self.sigma_y = (
                fa * value_1
                + 145.5
                + 3.5 * (S_0 ** (-0.5))
                + 478 * (p_N ** 0.5)
                + 1200 * p_P
            )

            self.sigma_u = (
                fa * (20 + 2440 * (p_N ** 0.5) + 18.5 * D_a)
                + 750 * (1 - fa)
                + 3 * (S_0 ** (-0.5)) * (1 - fa ** 0.5)
                + 92.5 * p_Si
            )
        else:  # Pearlitic steel
            t = 0.15 * S_0 * p_C
            M = 2 * (S_0 - t)
            if S_0 >= 0.15:
                self.sigma_y = 308 + 0.07 * (M ** (-1))
                self.sigma_u = 706 + 0.072 * (M ** (-1)) + 122 * p_Si
            else:
                self.sigma_y = 259 + 0.087 * (M ** (-1))
                self.sigma_u = 773 + 0.058 * (M ** (-1)) + 122 * p_Si

    def yield_strength(self) -> float:
        return self.sigma_y

    def tensile_strength(self) -> float:
        return self.sigma_u


class TrailingFormatter(Formatter):
    def vformat(self, *args):
        self._automatic = None
        return super(TrailingFormatter, self).vformat(*args)

    def get_value(self, key, args, kwargs):
        if key == "":
            if self._automatic is None:
                self._automatic = 0
            elif self._automatic == -1:
                raise ValueError(
                    "cannot switch from manual field specification "
                    "to automatic field numbering"
                )
            key = self._automatic
            self._automatic += 1
        elif isinstance(key, int):
            if self._automatic is None:
                self._automatic = -1
            elif self._automatic != -1:
                raise ValueError(
                    "cannot switch from automatic field numbering "
                    "to manual field specification"
                )
        return super(TrailingFormatter, self).get_value(key, args, kwargs)

    def format_field(self, value, spec):
        if len(spec) > 1 and spec[0] == "t":
            value = str(value) + spec[1]  # append the extra character
            spec = spec[2:]
        return super(TrailingFormatter, self).format_field(value, spec)


class MultiscaleStatistics:
    """This class generates the feature vectors of an image based on multiscale
    statistics in a procedure described in G. Impoco, L. Tuminello, N. Fucà, M. Caccamo,
    G. Licitra,
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
            scales (int, optional): Number of scales to consider. Each scale represents
                                    a neighborhood around a pixel. The first scale is
                                    the inmediate neighborhood of a pixel, the second
                                    one is the neighborhood around the previous one and
                                    so on. Defaults to 3.
        """
        self.scales = scales

    def process(self, img: np.ndarray) -> np.ndarray:
        return MultiscaleStatistics.process_helper(img, self.scales)

    @staticmethod
    def process_helper(img: np.ndarray, scales: int) -> np.ndarray:
        """Generates the feature vectors of an image.

        Args:
            img (np.ndarray): Image to calculate feature vectors.
            scales (int, optional): Number of scales to consider. Defaults to 3.

        Returns:
            np.ndarray: Feature vectors of the input image of shape
                        (*img.shape, 4*3*scales).
        """
        directions = 4
        gx_img, gy_img = np.gradient(img)

        feature_vectors = np.zeros(
            (img.size * 3 * directions * scales,), dtype=np.float64
        )
        computed_statistics_per_scale = np.zeros(
            (scales, img.shape[0], img.shape[1], directions * 3), dtype=np.float64
        )
        for scale in range(1, scales + 1):
            computed_statistics_per_dir = np.zeros(
                (directions, *img.shape, 3), dtype=np.float64
            )
            filter_size = 2 * (scale - 1) + 3
            orig = np.zeros((filter_size, filter_size), dtype=np.float64)
            orig[:, 0] = 1 / filter_size
            for c in range(directions):
                # c = 0 -> North; c = 1 -> East; c = 2 -> South; c = 3 -> West
                correlation_filter = np.rot90(orig, 3 - c, (0, 1))
                convolution_filter = np.flip(correlation_filter)

                directions_img = convolve(
                    img, convolution_filter, cval=0.0, mode="constant"
                )
                directions_gx_img = convolve(
                    gx_img, convolution_filter, cval=0.0, mode="constant"
                )
                directions_gy_img = convolve(
                    gy_img, convolution_filter, cval=0.0, mode="constant"
                )

                computed_statistics_per_dir[c] = np.concatenate(
                    (
                        directions_img[..., np.newaxis],
                        directions_gx_img[..., np.newaxis],
                        directions_gy_img[..., np.newaxis],
                    ),
                    axis=-1,
                )

            computed_statistics_per_scale[scale - 1] = np.concatenate(
                [
                    computed_statistics_per_dir[i][..., np.newaxis]
                    for i in range(directions)
                ],
                axis=-1,
            ).reshape(*img.shape, 3 * directions)

        for i in range(scales):
            feature_vectors[i::scales] = computed_statistics_per_scale[i].flatten()

        return feature_vectors.reshape((*img.shape, 3 * directions * scales))


class CustomFFTConvolution(object):
    def __init__(self, A, B, domain, threads=8):
        MK = B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        if domain == "same":
            self.Y = M
            self.X = N
        elif domain == "valid":
            self.Y = M - MK + 1
            self.X = N - NK + 1
        elif domain == "full":
            self.Y = M + MK - 1
            self.X = N + NK - 1

        self.M = M + MK - 1
        self.N = N + NK - 1

        a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode="constant")
        b = np.pad(B, ((0, M - 1), (0, N - 1)), mode="constant")

        self.fft_A_obj = pyfftw.builders.rfft2(a, s=(self.M, self.N), threads=threads)
        self.fft_B_obj = pyfftw.builders.rfft2(b, s=(self.M, self.N), threads=threads)
        self.ifft_obj = pyfftw.builders.irfft2(
            self.fft_A_obj.output_array, s=(self.M, self.N), threads=threads
        )

        self.offset_Y = int(np.floor((self.M - self.Y) / 2))
        self.offset_X = int(np.floor((self.N - self.X) / 2))

    def __call__(self, A, B):
        MK = B.shape[0]
        NK = B.shape[1]
        M = A.shape[0]
        N = A.shape[1]

        a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode="constant")
        b = np.pad(B, ((0, M - 1), (0, N - 1)), mode="constant")

        return self.ifft_obj(self.fft_A_obj(a) * self.fft_B_obj(b))[
            self.offset_Y : self.offset_Y + self.Y,
            self.offset_X : self.offset_X + self.X,
        ]
