from io import StringIO, BytesIO
import collections
import numpy as np
import matplotlib.colors
import warnings
from base64 import b64encode

import vaex
import vaex.utils
from vaex.image import rgba_2_pil

PIL = vaex.utils.optional_import("PIL.Image", package_name="pillow")


@vaex.register_function(scope='image')
def open(paths):
    images = [PIL.Image.open(path) for path in vaex.array_types.tolist(paths)]
    return np.array(images, dtype="O")


@vaex.register_function(scope='image')
def as_numpy(images):
    images = [np.array(image) for image in images]
    return np.array(images)


@vaex.register_function(scope='image')
def resize(images, size, resample=3):
    images = [image.resize(size, resample=resample) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='image')
def as_image(arrays):
    images = [rgba_2_pil(image_array) for image_array in arrays]
    return np.array(images, dtype="O")

def rgba_2_pil(rgba):
    with warnings.catch_warnings():
        pass