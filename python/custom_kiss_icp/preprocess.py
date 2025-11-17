# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

from custom_kiss_icp.config import KISSConfig
from custom_kiss_icp.pybind import kiss_icp_pybind


def get_preprocessor(config: KISSConfig):
    return Preprocessor(
        max_range=config.data.max_range,
        min_range=config.data.min_range,
        deskew=config.data.deskew,
        max_num_threads=config.registration.max_num_threads,
    )


class Preprocessor:
    def __init__(self, max_range, min_range, deskew, max_num_threads):
        self._preprocessor = kiss_icp_pybind._Preprocessor(
            max_range, min_range, deskew, max_num_threads
        )

    def preprocess(self, frame: np.ndarray, relative_motion: np.ndarray):
        if frame.shape[1] != 3:
            raise ValueError(f"Expected frame with 3 columns (x,y,z), got {frame.shape[1]}")

        return np.asarray(
            self._preprocessor._preprocess(
                kiss_icp_pybind._Vector4dVector(frame),
                relative_motion,
            )
        )
