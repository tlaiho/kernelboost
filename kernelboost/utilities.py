"""Metrics and utility functions used by the library itself.

Deliberately limited to what kernelboost needs internally, which is what keeps
the package dependency-free apart from numpy."""

import numpy as np

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R2 score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -np.inf
    return 1 - (ss_res / ss_tot)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score (as fraction of correct predictions)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(np.abs(y_true - y_pred))


def estimate_similarity(
    X_pred: np.ndarray,
    X_train: np.ndarray,
    gpu: bool = False,
    sample_size: int = 1000,
    samples: int = 10,
    ) -> np.ndarray:
    """Estimate similarity between prediction and training data."""
    from kernelboost.backend import Backend

    pred_obs = X_pred.shape[0]
    similarity_scores = np.zeros((pred_obs, samples))
    backend = Backend(gpu)
    rng = np.random.default_rng()

    for index in range(samples):
        training_sample = rng.choice(X_train, size=sample_size)
        similarity_scores[:, index] = backend.similarity(
            X_pred, training_sample, np.array([1.0])
        )

    similarity = np.mean(similarity_scores, axis=1)

    return similarity


def permutation_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: callable = None,
    n_repeats: int = 1,
    feature_names: list[str] = None,
) -> dict[int | str, float]:
    """
    Calculate permutation-based feature importance for an estimator.

    Args:
        estimator: Any object with a predict(X) method
        X: Feature array, shape (n_samples, n_features)
        y: Target array, shape (n_samples,) or (n_samples, 1)
        loss_fn: Loss function with signature loss_fn(y_true, y_pred) -> float.
                 Default: MSE
        n_repeats: Number of times to permute each feature (results averaged)
        feature_names: Optional list of feature names. If provided, dict keys
                       will be names instead of indices.

    Returns:
        dict mapping feature index (or name) to importance (increase in loss
        when permuted). Higher values indicate more important features.
    """
    rng = np.random.default_rng()
    y_flat = np.asarray(y).ravel()
    n_features = X.shape[1]

    if loss_fn is None:
        loss_fn = lambda y_true, y_pred: np.mean((y_true - y_pred.ravel()) ** 2)

    baseline_pred = estimator.predict(X)
    baseline_loss = loss_fn(y_flat, baseline_pred)

    importance = {}
    for f in range(n_features):
        losses = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, f])
            pred = estimator.predict(X_permuted)
            losses.append(loss_fn(y_flat, pred))
        key = feature_names[f] if feature_names else f
        importance[key] = np.mean(losses) - baseline_loss

    return importance


class RankTransformer:
    """Transform features to percentile ranks.

    Args:
    pct : bool, default=True
        If True, return percentile ranks (0-1 range).
        If False, return raw ranks (0 to n-1).
    exclude : list of int, optional
        Feature indices to exclude from transformation.
    """

    def __init__(
        self,
        pct: bool = True,
        exclude: list[int] | None = None,
    ):
        self.pct = pct
        self.exclude = exclude if exclude is not None else []

        # Attributes set during fit
        self.reference_values_: list[np.ndarray] | None = None
        self.n_features_: int | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RankTransformer":
        """Learn reference values for rank computation."""
        self.n_features_ = X.shape[1]

        # Store sorted values for each feature
        self.reference_values_ = []
        for feature in range(self.n_features_):
            if feature not in self.exclude:
                self.reference_values_.append(np.sort(X[:, feature]))
            else:
                self.reference_values_.append(None)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to percentile ranks."""
        if not self._is_fitted:
            raise RuntimeError("RankTransformer must be fitted before transform.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RankTransformer was fitted with {self.n_features_} features."
            )

        X_out = X.copy().astype(np.float64)

        for feature in range(self.n_features_):
            if feature not in self.exclude:
                ref = self.reference_values_[feature]
                n = len(ref)
                # searchsorted gives position where value would be inserted
                ranks = np.searchsorted(ref, X[:, feature], side='right')
                if self.pct:
                    X_out[:, feature] = ranks / n
                else:
                    X_out[:, feature] = ranks

        return np.ascontiguousarray(X_out)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse-transform ranks back to approximate original values.

        Inversion is lossy: distinct values that mapped to the same rank bin
        will inverse-transform to the same interpolated value."""
        
        if not self._is_fitted:
            raise RuntimeError("RankTransformer must be fitted before inverse_transform.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RankTransformer was fitted with {self.n_features_} features."
            )

        X_out = X.copy().astype(np.float64)

        for feature in range(self.n_features_):
            if feature not in self.exclude:
                ref = self.reference_values_[feature]
                n = len(ref)

                if self.pct:
                    indices = X[:, feature] * n - 1
                else:
                    indices = X[:, feature] - 1

                indices = np.clip(indices, 0, n - 1)

                lo = np.floor(indices).astype(int)
                hi = np.minimum(lo + 1, n - 1)
                frac = indices - lo

                X_out[:, feature] = ref[lo] * (1 - frac) + ref[hi] * frac

        return np.ascontiguousarray(X_out)

