from abc import ABC, abstractmethod
import numpy as np


class FeatureConstructor(ABC):
    """Frozen, pure materializer for one boosting round's features.

    Produced by a FeatureSelector during get_features(); stored by the booster in
    feature_constructors_ and invoked wherever the round's columns are needed
    (training, validation, predict, intervals, variance).
    """

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is frozen; FeatureConstructors are immutable "
                f"after construction (cannot set '{name}'). transform() must be pure."
            )
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        """Seal the instance against further mutation. Call at end of __init__."""
        object.__setattr__(self, "_frozen", True)

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return this round's feature columns from raw input X.

        Args:
        X : np.ndarray of shape (n_samples, n_features_in)
            Raw input matrix (original feature width).

        Returns:
        np.ndarray of shape (n_samples, k), float32, C-contiguous
            The columns fed to the KernelTree for this round.
        """

    @property
    @abstractmethod
    def source_features(self) -> set[int]:
        """Raw input column indices this constructor (transitively) depends on.
        Used for bookkeeping."""

    def describe(self, feature_names: list | None = None) -> str:
        """Human-readable label for logging. Override for richer descriptions."""
        idx = sorted(self.source_features)
        if feature_names is None:
            return str(idx)
        return str([feature_names[i] for i in idx])


class ColumnSelector(FeatureConstructor):
    """Identity materializer: selects raw columns by index. Backward-compatible."""

    def __init__(self, indices):
        self.indices = list(indices)
        self._freeze()

    def transform(self, X: np.ndarray) -> np.ndarray:
        # NumPy advanced indexing X[:, indices] returns an F-contiguous array for
        # multi-column selections, so wrap in ascontiguousarray to honor the
        # C-contiguous guarantee the C/GPU kernels rely on.
        return np.ascontiguousarray(X[:, self.indices], dtype=np.float32)

    @property
    def source_features(self) -> set[int]:
        return set(self.indices)

    def describe(self, feature_names: list | None = None) -> str:
        # Preserve self.indices insertion order (the base default sorts), so log
        # output matches the previous [feature_names_[k] for k in feature_indices].
        if feature_names is None:
            return str(self.indices)
        return str([feature_names[i] for i in self.indices])
