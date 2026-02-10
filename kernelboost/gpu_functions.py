"""Functions for CUDA backend."""

import numpy as np
from pathlib import Path
import cupy as cp


# load CUDA kernels from external file
_cuda_source = (Path(__file__).parent / 'kernels.cu').read_text()
_module = cp.RawModule(code=_cuda_source, options=("--use_fast_math",), backend='nvrtc')

gaussian_kernel = _module.get_function('gaussian_kernel')
laplace_kernel = _module.get_function('laplace_kernel')
kernel_weights = _module.get_function('weights')
loo_error = _module.get_function('loo_error')  


def is_gpu_available() -> bool:
    """Check if a CUDA-capable GPU is available."""
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
        return False


def get_block_grid_2d(rows: int, cols: int, block_size: int = 16, max_grid: int = 2048) -> tuple:
    """Calculate optimal grid and block dimensions for 2D kernel launch."""
    block = (block_size, block_size)
    grid = (min((rows + block_size - 1) // block_size, max_grid),
            min((cols + block_size - 1) // block_size, max_grid))
    return grid, block


def _compute_weights(
    t_features: np.ndarray,
    p_features: np.ndarray,
    precision: np.ndarray,
    kernel_type: int = 0,
) -> cp.ndarray:
    """Compute normalized kernel weight matrix."""
    p_count = np.int32(p_features.shape[0])
    t_count = np.int32(t_features.shape[0])
    dim = np.int32(t_features.shape[1])

    c_tf = cp.asarray(t_features, dtype=cp.float32)
    c_pf = cp.asarray(p_features, dtype=cp.float32)
    w_sums = cp.zeros(p_count, dtype=cp.float32)
    k_matrix = cp.zeros((p_count, t_count), dtype=cp.float32)
    prec = precision.astype(np.float32)

    kernel_func = gaussian_kernel if kernel_type == 0 else laplace_kernel
    grid, block = get_block_grid_2d(p_count, t_count)
    kernel_func(grid, block, (c_pf, c_tf, k_matrix, p_count, t_count, dim, prec[0]))

    cp.sum(k_matrix, axis=1, out=w_sums)
    kernel_weights(grid, block, (k_matrix, w_sums, p_count, t_count))

    return k_matrix


def cuda_get_weights(t_features: np.ndarray, p_features: np.ndarray, precision: np.ndarray, kernel_type: int = 0) -> np.ndarray:
    """Return normalized kernel weight matrix (in numpy)."""
    if p_features.shape[0] == 0:
        return np.array([], dtype=np.float32).reshape(0, t_features.shape[0])
    if t_features.shape[0] == 0:
        return np.zeros((p_features.shape[0], 0), dtype=np.float32)

    k_matrix = _compute_weights(t_features, p_features, precision, kernel_type)
    return cp.asnumpy(k_matrix)


def cuda_predict(t_dependent: np.ndarray, t_features: np.ndarray, p_features: np.ndarray, precision: np.ndarray, kernel_type: int = 0) -> np.ndarray:
    """Predict using Nadaraya-Watson estimator."""
    if p_features.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if t_features.shape[0] == 0:
        return np.full(p_features.shape[0], np.nan, dtype=np.float32)

    k_matrix = _compute_weights(t_features, p_features, precision, kernel_type)
    c_td = cp.asarray(t_dependent, dtype=cp.float32)
    predictions = cp.matmul(k_matrix, c_td)
    return cp.asnumpy(predictions)


def cuda_predict_with_variance(t_dependent: np.ndarray, t_features: np.ndarray, p_features: np.ndarray, precision: np.ndarray, kernel_type: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Predict using Nadaraya-Watson estimator with local variance estimation."""
    if p_features.shape[0] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if t_features.shape[0] == 0:
        return np.full(p_features.shape[0], np.nan, dtype=np.float32), np.full(p_features.shape[0], np.nan, dtype=np.float32)

    k_matrix = _compute_weights(t_features, p_features, precision, kernel_type)
    c_td = cp.asarray(t_dependent, dtype=cp.float32)
    predictions = cp.matmul(k_matrix, c_td)

    # Variance: E[Y²|X] - E[Y|X]² = E[Y²|X] - prediction²
    expected_y_sq = cp.matmul(k_matrix, c_td ** 2)
    variance = cp.maximum(expected_y_sq - predictions ** 2, 0)

    return cp.asnumpy(predictions), cp.asnumpy(variance)


def cuda_loo(t_dependent: np.ndarray,
             t_features: np.ndarray,
             precision: np.ndarray,
             kernel_type: int = 0,
             mean_y: float = 0.0) -> np.ndarray:
    """Compute leave-one-out cross-validation error on GPU. Uses the decomposition
    where the weights of the LOO estimator are computed with rescaling by 1 - self weight.
    """
    if t_features.shape[0] <= 1:
        return np.array(np.inf)

    k_matrix = _compute_weights(t_features, t_features, precision, kernel_type)

    c_td = cp.asarray(t_dependent, dtype=cp.float32).ravel()
    t_count = np.int32(t_features.shape[0])

    # detect bad rows: zero weight or isolated
    diag_indices = cp.arange(t_count)
    diag_weights = k_matrix[diag_indices, diag_indices]
    bad_weight_mask = (diag_weights == 0) | (diag_weights > 1.0 - 1e-4)

    predictions = cp.dot(k_matrix, c_td)

    # clamp self weights to prevent division by near-zero
    k_matrix[diag_indices, diag_indices] = cp.minimum(diag_weights, 1.0 - 1e-4)
    l_errors = cp.zeros(t_count, dtype=cp.float32)
    loo_grid = ((int(t_count) + 1023) // 1024, 1)  # 1d grid
    loo_error(loo_grid, (1024, 1), (predictions, c_td, k_matrix, l_errors, t_count))

    # overwrite bad weight errors with mean fallback penalty
    if cp.any(bad_weight_mask):
        l_errors[bad_weight_mask] = (c_td[bad_weight_mask] - mean_y) ** 2

    total_error = cp.sum(l_errors)
    return cp.asnumpy(total_error / t_count)


def cuda_similarity(p_features: np.ndarray,
                    t_features: np.ndarray,
                    precision: np.ndarray,
                    kernel_type: int = 0) -> np.ndarray:
    """Compute kernel similarity sums between prediction and training features."""
    if p_features.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if t_features.shape[0] == 0:
        return np.zeros(p_features.shape[0], dtype=np.float32)

    p_count = np.int32(p_features.shape[0])
    t_count = np.int32(t_features.shape[0])
    dim = np.int32(t_features.shape[1])

    c_tf = cp.asarray(t_features, dtype=cp.float32)
    c_pf = cp.asarray(p_features, dtype=cp.float32)
    k_matrix = cp.zeros((p_count, t_count), dtype=cp.float32)
    prec = precision.astype(np.float32)

    kernel_func = gaussian_kernel if kernel_type == 0 else laplace_kernel
    grid, block = get_block_grid_2d(p_count, t_count)
    kernel_func(grid, block, (c_pf, c_tf, k_matrix, p_count, t_count, dim, prec[0]))

    return cp.asnumpy(cp.sum(k_matrix, axis=1))
