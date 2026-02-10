"""Functions for CPU backend."""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import platform
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

system = platform.system()
if system == "Linux":
    libname = f"{dir_path}/libkernels.so"
elif system == "Windows":
    libname = f"{dir_path}/libkernels.dll"
elif system == "Darwin":
    libname = f"{dir_path}/libkernels.dylib"
else:
    raise Exception(f"Platform '{system}' not supported for CPU, try using GPU.")

# Load functions from C library (all float32 for GPU consistency)
clib = ctypes.CDLL(libname)
clib.predict.restype = None
clib.predict.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # predictions
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_dependent
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_features
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # prediction_features
    ctypes.c_float,   # precision
    ctypes.c_int,     # training_obs
    ctypes.c_int,     # prediction_obs
    ctypes.c_int,     # dimension
    ctypes.c_int,     # kernel_type
    )

clib.loo_mse.restype = ctypes.c_float
clib.loo_mse.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_dependent
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_features
    ctypes.c_float,   # precision
    ctypes.c_int,     # training_obs
    ctypes.c_int,     # dimension
    ctypes.c_int,     # kernel_type
    ctypes.c_float,   # mean_y
    )

clib.predict_with_variance.restype = None
clib.predict_with_variance.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # predictions
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # variances
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_dependent
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_features
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # prediction_features
    ctypes.c_float,   # precision
    ctypes.c_int,     # training_obs
    ctypes.c_int,     # prediction_obs
    ctypes.c_int,     # dimension
    ctypes.c_int,     # kernel_type
    )

clib.get_weights.restype = None
clib.get_weights.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # weights
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_features
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # prediction_features
    ctypes.c_float,   # precision
    ctypes.c_int,     # training_obs
    ctypes.c_int,     # prediction_obs
    ctypes.c_int,     # dimension
    ctypes.c_int,     # kernel_type
    )

clib.estimate_similarity.restype = None
clib.estimate_similarity.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # weight_sums
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # training_features
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # prediction_features
    ctypes.c_float,   # precision
    ctypes.c_int,     # training_obs
    ctypes.c_int,     # prediction_obs
    ctypes.c_int,     # dimension
    ctypes.c_int,     # kernel_type
    )

def cpu_predict(
        training_dependent: np.ndarray,
        training_features: np.ndarray,
        prediction_features: np.ndarray,
        precision: np.ndarray,
        kernel_type: int = 0,
        ) -> np.ndarray:
    """Predict using Nadaraya-Watson estimator."""

    if prediction_features.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if training_features.shape[0] == 0:
        return np.full(prediction_features.shape[0], np.nan, dtype=np.float32)

    training_obs = training_dependent.shape[0]
    prediction_obs = prediction_features.shape[0]
    dimension = training_features.shape[1]
    predictions = np.zeros(prediction_obs, dtype=np.float32)

    training_features_1d = np.ravel(training_features).astype(np.float32)
    prediction_features_1d = np.ravel(prediction_features).astype(np.float32)

    clib.predict(
        predictions,
        np.ascontiguousarray(training_dependent.astype(np.float32)),
        np.ascontiguousarray(training_features_1d),
        np.ascontiguousarray(prediction_features_1d),
        ctypes.c_float(precision[0]),
        ctypes.c_int(training_obs),
        ctypes.c_int(prediction_obs),
        ctypes.c_int(dimension),
        ctypes.c_int(kernel_type),
    )

    return predictions


def cpu_loo_mse(
    training_dependent: np.ndarray,
    training_features: np.ndarray,
    precision: np.ndarray,
    kernel_type: int = 0,
    mean_y: float = 0.0,
    ) -> float:
    """Compute leave-one-out cross-validation error using symmetric kernel optimization.
    Uses float32 precision for consistency with GPU.
    """

    if training_features.shape[0] <= 1:
        return np.inf

    training_obs = training_dependent.shape[0]
    dimension = training_features.shape[1]
    training_features_1d = np.ravel(training_features).astype(np.float32)

    result = clib.loo_mse(
        np.ascontiguousarray(training_dependent.astype(np.float32)),
        np.ascontiguousarray(training_features_1d),
        ctypes.c_float(precision[0]),
        ctypes.c_int(training_obs),
        ctypes.c_int(dimension),
        ctypes.c_int(kernel_type),
        ctypes.c_float(mean_y),
    )

    return float(result)


def cpu_predict_with_variance(
        training_dependent: np.ndarray,
        training_features: np.ndarray,
        prediction_features: np.ndarray,
        precision: np.ndarray,
        kernel_type: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Predict Nadaraya-Watson estimator with local variance estimation."""

    if prediction_features.shape[0] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if training_features.shape[0] == 0:
        return np.full(prediction_features.shape[0], np.nan, dtype=np.float32), np.full(prediction_features.shape[0], np.nan, dtype=np.float32)

    training_obs = training_dependent.shape[0]
    prediction_obs = prediction_features.shape[0]
    dimension = training_features.shape[1]
    predictions = np.zeros(prediction_obs, dtype=np.float32)
    variances = np.zeros(prediction_obs, dtype=np.float32)

    training_features_1d = np.ravel(training_features).astype(np.float32)
    prediction_features_1d = np.ravel(prediction_features).astype(np.float32)

    clib.predict_with_variance(
        predictions,
        variances,
        np.ascontiguousarray(training_dependent.astype(np.float32)),
        np.ascontiguousarray(training_features_1d),
        np.ascontiguousarray(prediction_features_1d),
        ctypes.c_float(precision[0]),
        ctypes.c_int(training_obs),
        ctypes.c_int(prediction_obs),
        ctypes.c_int(dimension),
        ctypes.c_int(kernel_type),
    )

    return predictions, variances


def cpu_get_weights(
        training_features: np.ndarray,
        prediction_features: np.ndarray,
        precision: np.ndarray,
        kernel_type: int = 0,
        ) -> np.ndarray:
    """Return kernel weight matrix."""

    if prediction_features.shape[0] == 0:
        return np.array([], dtype=np.float32).reshape(0, training_features.shape[0])
    if training_features.shape[0] == 0:
        return np.zeros((prediction_features.shape[0], 0), dtype=np.float32)

    training_obs = training_features.shape[0]
    prediction_obs = prediction_features.shape[0]
    dimension = training_features.shape[1]
    weights = np.zeros((prediction_obs, training_obs), dtype=np.float32)

    training_features_1d = np.ravel(training_features).astype(np.float32)
    prediction_features_1d = np.ravel(prediction_features).astype(np.float32)

    clib.get_weights(
        weights,
        np.ascontiguousarray(training_features_1d),
        np.ascontiguousarray(prediction_features_1d),
        ctypes.c_float(precision[0]),
        ctypes.c_int(training_obs),
        ctypes.c_int(prediction_obs),
        ctypes.c_int(dimension),
        ctypes.c_int(kernel_type),
    )

    return weights


def cpu_similarity(
        prediction_features: np.ndarray,
        training_features: np.ndarray,
        precision: np.ndarray,
        kernel_type: int = 0,
        ) -> np.ndarray:
    """Compute kernel similarity sums between prediction and training features."""

    if prediction_features.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if training_features.shape[0] == 0:
        return np.zeros(prediction_features.shape[0], dtype=np.float32)

    training_obs = training_features.shape[0]
    prediction_obs = prediction_features.shape[0]
    dimension = training_features.shape[1]
    weight_sums = np.zeros(prediction_obs, dtype=np.float32)

    training_features_1d = np.ravel(training_features).astype(np.float32)
    prediction_features_1d = np.ravel(prediction_features).astype(np.float32)

    clib.estimate_similarity(
        weight_sums,
        np.ascontiguousarray(training_features_1d),
        np.ascontiguousarray(prediction_features_1d),
        ctypes.c_float(precision[0]),
        ctypes.c_int(training_obs),
        ctypes.c_int(prediction_obs),
        ctypes.c_int(dimension),
        ctypes.c_int(kernel_type),
    )

    return weight_sums
