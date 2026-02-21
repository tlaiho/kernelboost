import numpy as np
from .backend import Backend
from .optimizer import optimize_precision, estimate_bounds
from .utilities import r2_score


class KernelEstimator:
    """
    Nadaraya-Watson estimator. 

    Args:
    use_gpu : bool, default=False
        Whether to use GPU acceleration.
    kernel_type : str, default='laplace'
        Kernel type: 'gaussian' or 'laplace'.
    search_rounds : int, default=20
        Number of optimization rounds for precision search.
    bounds : tuple, default=(0.10, 35.0)
        Lower and upper bounds for precision optimization.
    initial_precision : float, default=0.0
        Initial precision value for optimization. 0.0 means auto.
    sample_share : float, default=1.0
        Fraction of data to use for precision optimization (for large datasets).
    precision_method : str, default='pilot-cv'
        Precision selection method: 'search' (LOO-CV) 'pilot-CV' (bounds for pilot, then LOO-CV) 
        or 'silverman' (rule-of-thumb for testing).
    seed : int, default=None
        Random seed for reproducibility when subsampling during precision optimization.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        kernel_type: str = 'laplace',
        search_rounds: int = 20,
        bounds: tuple = (0.10, 35.0),
        initial_precision: float = 0.0,
        sample_share: float = 1.0,
        precision_method: str = 'pilot-cv',
        seed: int = None,
    ):
        if kernel_type not in {'gaussian', 'laplace'}:
            raise ValueError(f"kernel_type must be 'gaussian' or 'laplace', got '{kernel_type}'")

        self.use_gpu = use_gpu
        self.kernel_type = kernel_type
        self.search_rounds = search_rounds
        self.bounds = bounds
        self.initial_precision = initial_precision
        self.sample_share = sample_share
        self.precision_method = precision_method
        self.seed = seed

        self.kernel_optimization = {
            'kernel_type': kernel_type,
            'search_rounds': search_rounds,
            'bounds': bounds,
            'initial_precision': initial_precision,
            'sample_share': sample_share,
            'precision_method': precision_method,
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the kernel estimator to training data.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Training targets.

        Returns:
        self
            Fitted estimator.
        """
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y.reshape(-1, 1) if y.ndim == 1 else y, dtype=np.float32)
        self.n_samples_, self.n_features_in_ = X.shape

        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Kernel estimation requires complete data.")
        if np.isnan(y).any():
            raise ValueError("y contains NaN values. Kernel estimation requires complete data.")

        self._backend = Backend(self.use_gpu, self.kernel_type)
        self.precision_ = self._optimal_precision()
        self.training_predictions_ = self.predict(self.X_)

        return self
    
    def _optimal_precision(self):
        """Find optimal precision using LOO-CV."""
        sample_share = self.kernel_optimization['sample_share']
        if self.n_samples_ > 1500 and sample_share < 1.0:
            tdata = np.concatenate((self.y_, self.X_), axis=1)
            sample_size = int(max(sample_share * tdata.shape[0], 1000))
            sample = np.random.default_rng(self.seed).choice(tdata, sample_size)
            y = np.ascontiguousarray(sample[:, 0].reshape(-1, 1))
            X = np.delete(sample, 0, axis=1)
        else:
            y = self.y_
            X = self.X_

        mean_y = float(np.mean(y))

        if self.precision_method == 'pilot-cv':
            self.kernel_optimization['bounds'] = estimate_bounds(
                X, y, self.kernel_type, self.bounds)
            self.kernel_optimization['precision_method'] = 'search'

        if self.use_gpu:
            import cupy as cp
            y = cp.asarray(y, dtype=cp.float32)
            X = cp.asarray(X, dtype=cp.float32)

        optimal_precision = optimize_precision(
            self._backend.loo_cv, y, X, self.kernel_optimization, mean_y=mean_y,
        )
        return optimal_precision

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the kernel estimator.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.

        Returns:
        np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'precision_'):
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        predictions = self._backend.predict(
            self.y_, self.X_, X, self.precision_
        )
        return predictions

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the estimator and return predictions on X.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Training targets.

        Returns:
        np.ndarray of shape (n_samples,)
            Predicted values.
        """
        return self.fit(X, y).predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score on given data."""
        return r2_score(y, self.predict(X))

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator. """
        # deep for compatibility
        return {
            'use_gpu': self.use_gpu,
            'kernel_type': self.kernel_type,
            'search_rounds': self.search_rounds,
            'bounds': self.bounds,
            'initial_precision': self.initial_precision,
            'sample_share': self.sample_share,
            'precision_method': self.precision_method,
            'seed': self.seed,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: tuple = (0.1, 0.5, 0.9)
    ) -> np.ndarray:
        """
        Predict conditional quantiles using local constant (Nadaraya-Watson) estimator 
        and the single-kernel approach. Computationally cheaper than the double kernel
        approach in Hansen (2004), but precision suffers from the discreteness of the 
        underlying dependent variable especially with sparse data.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            quantiles:(default: 10th, 50th, 90th)

        Returns:
            quantile_predictions: shape (n_samples, len(quantiles))
                Column i contains the quantiles[i] predictions
        """
        if not hasattr(self, 'precision_'):
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        weights = self._backend.get_weights(self.X_, X, self.precision_)
        y_flat = self.y_.flatten()

        n_pred = X.shape[0]
        n_quantiles = len(quantiles)
        result = np.zeros((n_pred, n_quantiles))

        for i in range(n_pred):
            result[i] = self._kernel_quantiles(y_flat, weights[i], quantiles)

        return result
    
    def _kernel_quantiles(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        quantiles: tuple
    ) -> np.ndarray:
        """Compute quantiles based on kernel weights."""
        if np.sum(weights) == 0:
            return np.full(len(quantiles), np.nan)

        sorter = np.argsort(values)
        values_sorted = values[sorter]
        weights_sorted = weights[sorter]

        cumsum = np.cumsum(weights_sorted)

        result = np.zeros(len(quantiles))
        for i, q in enumerate(quantiles):
            idx = np.searchsorted(cumsum, q)
            if idx >= len(values_sorted):
                idx = len(values_sorted) - 1
            result[i] = values_sorted[idx]

        return result


    def _predict_with_variance(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        EXPERIMENTAL. Predict with local variance estimation. Variance estimation uses
        a single-kernel approach (and local constant regression) instead of the
        two-step procedure with residuals (and local linear regression) in Fan and Yao (1998).
        Computationally cheaper, but variance estimates are distorted upwards by bias
        unless E(Y|X) is precise. A special case is when E(Y|X) = 0 for all X, then
        the approach effectively collapses to Fan and Yao (1998).

        Args:
        X : np.ndarray
            Features, shape (n_samples, n_features).

        Returns:
        predictions : np.ndarray, shape (n_samples,)
        variances : np.ndarray, shape (n_samples,)
        """
        if not hasattr(self, 'precision_'):
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return self._backend.predict_with_variance(
            self.y_, self.X_, X, self.precision_
        )