import numpy as np

class Backend:
    """Unified interface for GPU/CPU kernel operations."""

    def __init__(self, use_gpu: bool = False, kernel_type: str = 'gaussian'):
        """
        Initialize backend.

        Args:
            use_gpu: If True, use GPU.
            kernel_type: Kernel type
        """
        self.use_gpu = use_gpu
        self.max_matrix_elements = None  # CPU has no limit

        self.kernel_type = kernel_type  
        kernel_types = {"gaussian": 0, "laplace": 1}
        self._kernel_type_int = kernel_types[kernel_type]

        if use_gpu:
            if not self._check_gpu():
                print("Warning: GPU unavailable, falling back to CPU")
                self.use_gpu = False
            else:
                self._compute_memory_limit()

    def _check_gpu(self) -> bool:
        try:
            from .gpu_functions import is_gpu_available
            return is_gpu_available()
        except ImportError:
            return False

    def _compute_memory_limit(self):
        """Calculate max kernel matrix size that fits in GPU memory."""
        import cupy as cp
        free_bytes, _ = cp.cuda.Device().mem_info
        # float32 = 4 bytes, use 50% of free memory (conservative estimate)
        self.max_matrix_elements = int(free_bytes * 0.5 / 4)
        self.max_symmetric_n = int(self.max_matrix_elements ** 0.5)

    def _check_memory(self, n_rows: int, n_cols: int, operation: str):
        """Raise MemoryError if matrix won't fit in GPU memory."""
        if self.max_matrix_elements is None:
            return  # CPU has no limit
        required = n_rows * n_cols
        if required > self.max_matrix_elements:
            raise MemoryError(
                f"GPU memory insufficient for {operation}."
                f"Matrix ({n_rows}, {n_cols}) = {required/1e6:.0f}M elements,"
                f"limit ~{self.max_matrix_elements/1e6:.0f}M."
                f"Max symmetric size: {self.max_symmetric_n}x{self.max_symmetric_n}."
                f"Use use_gpu=False or reduce data size."
            )

    @property
    def name(self) -> str:
        return "gpu" if self.use_gpu else "cpu"

    def predict(self, training_dependent: np.ndarray,
                training_features: np.ndarray,
                prediction_features: np.ndarray,
                precision: np.ndarray) -> np.ndarray:
        """
        Predict using Nadaraya-Watson regression with given precision.

        Args:
            training_dependent: (n_train,) or (n_train, 1)
            training_features: (n_train, n_features)
            prediction_features: (n_pred, n_features)
            precision: precision array
        """
        self._check_memory(
            prediction_features.shape[0], training_features.shape[0], "predict"
        )
        if self.use_gpu:
            from .gpu_functions import cuda_predict
            return cuda_predict(training_dependent, training_features,
                                prediction_features, precision,
                                self._kernel_type_int)
        else:
            from .cpu_functions import cpu_predict
            return cpu_predict(training_dependent, training_features,
                               prediction_features, precision,
                               self._kernel_type_int)

    def loo_cv(self, training_dependent: np.ndarray,
               training_features: np.ndarray,
               precision: np.ndarray,
               mean_y: float = 0.0) -> float:
        """
        Leave-one-out cross-validation error with given precision.

        Args:
            training_dependent: (n_train,) or (n_train, 1)
            training_features: (n_train, n_features)
            precision: precision array
            mean_y: mean of training_dependent for zero-weight fallback
        """
        self._check_memory(
            training_features.shape[0], training_features.shape[0], "LOO-CV"
        )
        if self.use_gpu:
            from .gpu_functions import cuda_loo
            return float(cuda_loo(training_dependent, training_features, precision,
                                  self._kernel_type_int, mean_y))
        else:
            from .cpu_functions import cpu_loo_mse
            return cpu_loo_mse(training_dependent, training_features, precision,
                               self._kernel_type_int, mean_y)

    def predict_with_variance(
        self,
        training_dependent: np.ndarray,
        training_features: np.ndarray,
        prediction_features: np.ndarray,
        precision: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        EXPERIMENTAL. Nadaraya-Watson prediction with local variance estimation based on 
        Var(Y|X) = E[Y²|X] - prediction².
        Args:
            training_dependent: (n_train,) or (n_train, 1)
            training_features: (n_train, n_features)
            prediction_features: (n_pred, n_features)
            precision: precision array

        """
        self._check_memory(
            prediction_features.shape[0], training_features.shape[0],
            "predict_with_variance"
        )
        if self.use_gpu:
            from .gpu_functions import cuda_predict_with_variance
            return cuda_predict_with_variance(
                training_dependent, training_features,
                prediction_features, precision,
                self._kernel_type_int
            )
        else:
            from .cpu_functions import cpu_predict_with_variance
            return cpu_predict_with_variance(
                training_dependent, training_features,
                prediction_features, precision,
                self._kernel_type_int
            )

    def get_weights(
        self,
        training_features: np.ndarray,
        prediction_features: np.ndarray,
        precision: np.ndarray
    ) -> np.ndarray:
        """
        Returns kernel weight matrix (n_pred, n_train).

        Args:
            training_features: (n_train, n_features)
            prediction_features: (n_pred, n_features)
            precision: precision array
        """
        self._check_memory(
            prediction_features.shape[0], training_features.shape[0],
            "get_weights"
        )
        if self.use_gpu:
            from .gpu_functions import cuda_get_weights
            return cuda_get_weights(
                training_features, prediction_features, precision,
                self._kernel_type_int
            )
        else:
            from .cpu_functions import cpu_get_weights
            return cpu_get_weights(
                training_features, prediction_features, precision,
                self._kernel_type_int
            )

    def similarity(self, prediction_features: np.ndarray,
                   training_features: np.ndarray,
                   precision: np.ndarray) -> np.ndarray:
        """
        Compute kernel similarity scores.

        Args:
            prediction_features: (n_pred, n_features)
            training_features: (n_train, n_features)
            precision: precision array

        """
        self._check_memory(
            prediction_features.shape[0], training_features.shape[0], "similarity"
        )
        if self.use_gpu:
            from .gpu_functions import cuda_similarity
            return cuda_similarity(prediction_features, training_features, precision,
                                   self._kernel_type_int)
        else:
            from .cpu_functions import cpu_similarity
            return cpu_similarity(prediction_features, training_features, precision,
                                  self._kernel_type_int)