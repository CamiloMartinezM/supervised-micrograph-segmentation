# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 06:45:29 2020

@author: Camilo Martínez
"""
from string import Formatter

import numpy as np
import pyfftw


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
