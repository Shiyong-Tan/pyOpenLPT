# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""
Image Preprocessing Core
Pure processing functions with no GUI dependencies.
Extracted for CLI and programmatic use.
"""

from functools import lru_cache
import threading

import numpy as np
import cv2


DEFAULT_PROCESSING_SETTINGS = {
    "bg_enabled": False,
    "invert": False,
    "cine_shifts": {},
    "low_in": 0,
    "high_in": 255,
    "denoise": False,
}


def normalize_processing_settings(settings=None):
    """
    Normalize preprocessing settings for CLI and GUI callers.

    Parameters
    ----------
    settings : dict or None
        Partial or complete settings dictionary.

    Returns
    -------
    dict
        Settings dictionary with all required keys populated.
    """
    normalized = dict(DEFAULT_PROCESSING_SETTINGS)
    if settings:
        normalized.update(settings)

    cine_shifts = normalized.get("cine_shifts") or {}
    normalized["cine_shifts"] = dict(cine_shifts)
    return normalized


def imadjust_opencv(img, low_in, high_in, low_out=0, high_out=255, gamma=1.0):
    """
    Adjust image intensity values similar to MATLAB's imadjust.
    
    Parameters
    ----------
    img : ndarray
        Input image (uint8 or float)
    low_in : float
        Lower input intensity limit
    high_in : float
        Upper input intensity limit
    low_out : float, optional
        Lower output intensity limit (default: 0)
    high_out : float, optional
        Upper output intensity limit (default: 255)
    gamma : float, optional
        Gamma correction value (default: 1.0)
    
    Returns
    -------
    ndarray
        Adjusted image as uint8
    """
    # Ensure float for calculation
    img = img.astype(np.float32)

    # normalize to [0,1]
    # Handle division by zero
    diff = high_in - low_in
    if diff < 1e-5:
        diff = 1e-5
        
    img = (img - low_in) / diff
    img = np.clip(img, 0, 1)

    # gamma
    if gamma != 1.0:
        img = img ** gamma

    # scale to output range
    img = img * (high_out - low_out) + low_out
    img = np.clip(img, low_out, high_out)

    return img.astype(np.uint8)


@lru_cache(maxsize=1)
def _get_cupy():
    """Load CuPy lazily and verify that a CUDA device is usable."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is available")
    cp.cuda.Device(0).use()
    cp.empty((1,), dtype=cp.uint8)
    return cp


def gpu_preprocessing_available():
    """Return ``(available, detail)`` for the exact CUDA path."""
    try:
        cp = _get_cupy()
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = (
            props.get("name", b"CUDA device")
            if isinstance(props, dict) else b"CUDA device"
        )
        name = (
            raw_name.decode(errors="replace")
            if isinstance(raw_name, bytes) else str(raw_name)
        )
        return True, name
    except Exception as exc:
        return False, str(exc)


@lru_cache(maxsize=128)
def _imadjust_uint8_lut(
    low_in, high_in, low_out=0, high_out=255, gamma=1.0
):
    """Build the CUDA lookup table with the authoritative CPU operation."""
    values = np.arange(256, dtype=np.uint8)
    lut = imadjust_opencv(
        values, low_in, high_in, low_out, high_out, gamma
    )
    lut.setflags(write=False)
    return lut


_GPU_LUTS = {}
_GPU_LUT_LOCK = threading.Lock()


def _gpu_imadjust_lut(cp, low_in, high_in):
    key = (float(low_in), float(high_in), 0.0, 255.0, 1.0)
    with _GPU_LUT_LOCK:
        lut = _GPU_LUTS.get(key)
        if lut is None:
            lut = cp.asarray(_imadjust_uint8_lut(*key))
            _GPU_LUTS[key] = lut
    return lut


def _apply_denoise_cpu(result):
    """Run the original OpenCV denoise sequence without numerical changes."""
    a = result.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    b = cv2.erode(a, kernel, iterations=1)
    c = a - b
    b = cv2.erode(a, kernel, iterations=1)
    c = c - b

    d = cv2.GaussianBlur(c, (0, 0), 0.5)
    e = cv2.blur(d, (100, 100))
    f = a - e

    blurred_f = cv2.GaussianBlur(f, (0, 0), 1.0)
    sharp = f + 0.8 * (f - blurred_f)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _apply_processing_pipeline_gpu(img_data, bg_data, cam_idx, settings):
    """Run exact pointwise operations on CUDA with one upload/download."""
    cp = _get_cupy()

    if len(img_data.shape) == 3:
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img_data.astype(np.float32)
    gray_gpu = cp.asarray(gray)

    if settings["bg_enabled"] and bg_data is not None:
        bg_gpu = (
            bg_data if isinstance(bg_data, cp.ndarray) else cp.asarray(bg_data)
        )
        result_gpu = (
            bg_gpu - gray_gpu if settings["invert"] else gray_gpu - bg_gpu
        )
        cp.maximum(result_gpu, cp.float32(0.0), out=result_gpu)
    else:
        result_gpu = gray_gpu

    shift = int(settings["cine_shifts"].get(cam_idx, 0))
    if shift > 0:
        result_gpu = result_gpu * cp.float32(2.0 ** (-shift))
    result_gpu = cp.clip(result_gpu, 0, 255).astype(cp.uint8)

    if settings["invert"] and not (
        settings["bg_enabled"] and bg_data is not None
    ):
        result_gpu = cp.uint8(255) - result_gpu

    lut_gpu = _gpu_imadjust_lut(
        cp, settings["low_in"], settings["high_in"]
    )
    result_gpu = lut_gpu[result_gpu]
    result = cp.asnumpy(result_gpu)

    if settings["denoise"]:
        result = _apply_denoise_cpu(result)
    return result.astype(np.uint8, copy=False)


def apply_processing_pipeline_with_settings(
    img_data, bg_data, cam_idx, settings, use_gpu=False
):
    """
    Apply complete image preprocessing pipeline.
    
    Pure processing pipeline for worker thread use or CLI processing.
    
    Parameters
    ----------
    img_data : ndarray
        Input image data (grayscale or color, any bit depth)
    bg_data : ndarray or None
        Background image for subtraction (float32), or None to skip
    cam_idx : int
        Camera index for bit-shift lookup
    settings : dict
        Processing settings dictionary with keys:
        - bg_enabled : bool
            Enable background subtraction
        - invert : bool
            Invert image intensities
        - cine_shifts : dict
            {cam_idx: shift_bits} for bit-depth reduction
        - low_in : float
            Lower input range for intensity adjustment
        - high_in : float
            Upper input range for intensity adjustment
        - denoise : bool
            Enable enhanced denoise processing
    
    Returns
    -------
    ndarray
        Processed image as uint8
    """
    settings = normalize_processing_settings(settings)

    if use_gpu:
        return _apply_processing_pipeline_gpu(
            img_data, bg_data, cam_idx, settings
        )

    # 0. Ensure grayscale and float32
    if len(img_data.shape) == 3:
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img_data.astype(np.float32)

    # 1. Background Subtraction (float32)
    if settings["bg_enabled"] and bg_data is not None:
        if settings["invert"]:
            result = bg_data - gray
        else:
            result = gray - bg_data
        result = np.clip(result, 0, None)
    else:
        result = gray

    # 2. Bit shift to 8-bit
    shift = settings["cine_shifts"].get(cam_idx, 0)
    if shift > 0:
        result = (result / (2 ** shift))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 3. Invert (only if not already handled by BG subtraction)
    if settings["invert"] and not (settings["bg_enabled"] and bg_data is not None):
        result = 255 - result

    # 4. Range adjustment
    result = imadjust_opencv(result, settings["low_in"], settings["high_in"])

    # 5. Denoise
    if settings["denoise"]:
        result = _apply_denoise_cpu(result)

    return result.astype(np.uint8)
