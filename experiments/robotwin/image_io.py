"""Decode the legacy RoboTwin HDF5 camera codec into deployment RGB arrays."""
from __future__ import annotations

import io


def decode_legacy_robotwin_rgb(value):
    """Undo the channel swap from cv2.imencode applied to camera RGB.

    RoboTwin pkl2hdf5.py passes the camera's RGB array to an encoder expecting
    BGR. A standard RGB JPEG reader therefore returns swapped red/blue. This
    function is only for that legacy raw-HDF5 format, not ordinary JPEGs.
    """
    import numpy as np
    from PIL import Image

    encoded = bytes(value).rstrip(b'\0')
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)[:, :, ::-1].copy()
