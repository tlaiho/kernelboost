from abc import ABC, abstractmethod
from collections.abc import Generator
import ctypes
import os
import platform

import numpy as np
from numpy.ctypeslib import ndpointer

# Load C library for fast MI computation
_dir_path = os.path.dirname(os.path.realpath(__file__))

_system = platform.system()
if _system == "Linux":
    _mi_libname = f"{_dir_path}/libmi.so"
elif _system == "Windows":
    _mi_libname = f"{_dir_path}/libmi.dll"
elif _system == "Darwin":
    _mi_libname = f"{_dir_path}/libmi.dylib"
else:
    raise Exception(f"Platform '{_system}' not supported.")

try:
    _mi_lib = ctypes.CDLL(_mi_libname)
except OSError:
    raise OSError(
        f"Could not load C library at {_mi_libname}. "
        f"Compile it with: gcc -shared -o {_mi_libname} -fPIC kernelboost/mi_bins.c "
        f"-lm -fopenmp -O3 -march=native -ffast-math -funroll-loops -flto"
    )

_mi_lib.histogram_mi_batch.restype = None
_mi_lib.histogram_mi_batch.argtypes = (
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # X
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # residuals
    ctypes.c_int,  # n
    ctypes.c_int,  # n_features
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # x_thresholds
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # y_thresholds
    ctypes.c_int,  # n_thresh
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out_mi
)


class FeatureSelector(ABC):
    """
    Abstract base class for feature selection strategies.

    Implementations are passed to KernelBooster and called during training
    to determine which features to use in each boosting round.
    """

    def __init__(self):
        self.feature_groups: list[list[int]] | None = None

    @abstractmethod
    def initialize(
        self,
        X: np.ndarray,
        n_features: int,
        min_features: int,
        max_features: int,
        rounds: int,
    ) -> int:
        """
        Initialize the selector with training data.

        Args:
        X : np.ndarray
            Training features (n_samples, n_features)
        n_features : int
            Number of features
        min_features : int
            Minimum features per kernel
        max_features : int
            Maximum features per kernel
        rounds : int
            Requested number of boosting rounds

        Returns:
        int
            Actual number of rounds (may differ from requested)
        """
        pass

    @abstractmethod
    def get_features(
        self, round_idx: int, residuals: np.ndarray
    ) -> tuple[list[int], str]:
        """
        Get feature indices and tree type for the next boosting round.

        Args:
        round_idx : int
            Current round index (0-based)
        residuals : np.ndarray
            Current pseudo-residuals (n_samples,)

        Returns:
        tuple[list[int], str]
            Feature indices and leaf type ('kernel' or 'constant')
        """
        pass

    def update(self, feature_indices: list[int], gain: float) -> None:
        """
        Update internal state after a round completes.

        Args:
        feature_indices : list[int]
            Features used in the completed round
        gain : float
            Loss reduction achieved in this round
        """
        pass

    def _complete_groups(self, selected: list[int]) -> list[int]:
        """
        If any feature from a group is selected, include the whole group.
        """
        if not self.feature_groups:
            return selected
        selected_set = set(selected)
        for group in self.feature_groups:
            if selected_set & set(group):
                selected_set.update(group)
        return list(selected_set)


class RandomSelector(FeatureSelector):
    """
    Random feature selection. Kernel sizes progress from small to large.

    Args:
    seed : int, optional
        Random seed for reproducibility.
    feature_groups : list[list[int]] | None, default=None
        Groups of features that should be selected together.
    """

    def __init__(
        self, seed: int | None = None, feature_groups: list[list[int]] | None = None
    ):
        super().__init__()
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        self.feature_groups = feature_groups

    def initialize(
        self,
        X: np.ndarray,
        n_features: int,
        min_features: int,
        max_features: int,
        rounds: int,
    ) -> int:
        self.n_features = n_features
        self._rng = np.random.default_rng(self.seed)
        self._gen = self._feature_generator(
            n_features, min_features, max_features, rounds
        )
        return rounds

    def _feature_generator(
        self, n_features: int, min_features: int, max_features: int, rounds: int
    ) -> Generator[list[int], None, None]:
        """Yield random feature subsets, progressively increasing in size."""
        if n_features == 1:
            while True:
                yield [0]

        features = np.arange(n_features)
        max_size = min(n_features, max_features)
        sizes = list(range(min_features, max_size + 1))
        rounds_per_size = max(1, rounds // len(sizes))

        for size in sizes:
            for _ in range(rounds_per_size):
                self._rng.shuffle(features)
                yield features[:size].tolist()
        # max size if more rounds needed
        while True:
            self._rng.shuffle(features)
            yield features[:max_size].tolist()

    def get_features(
        self, round_idx: int, residuals: np.ndarray
    ) -> tuple[list[int], str]:
        selected = next(self._gen)
        return self._complete_groups(selected), "kernel"


class SmartSelector(FeatureSelector):
    """
    Feature selection using mRMR-style approach with mutual information relevance.
    Selects features probabilistically based on relevance, redundancy and recency.
    Kernel sizes progress from small to large.

    Args:
    redundancy_penalty : float, default=0.4
        Weight for redundancy penalty (0 = ignore, 1 = strong penalty)
    relevance_alpha : float, default=0.7
        Balance between MI relevance (1.0) and historical weight (0.0)
    recency_penalty : float, default=0.35
        Penalty applied to recently-used features (0 = no penalty, 1 = strong)
    recency_decay : float, default=0.7
        Decay factor for recency penalty each round (0 = instant decay, 1 = no decay)
    temperature : float, default=0.3
        Softmax temperature (minimum when using schedule). Higher means more exploration.
    temperature_max : float | None, default=None
        Starting temperature for schedule. None means no schedule (fixed temperature).
        Decays linearly from temperature_max to temperature over all rounds.
    weight_decay : float, default=0.95
        Decay factor for feature weights each round.
    feature_groups : list[list[int]] | None, default=None
        Groups of features that should be selected together.
    constant_tree_frequency : int, default=25
        Insert a constant-leaf tree every N rounds.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        redundancy_penalty: float = 0.4,
        relevance_alpha: float = 0.7,
        recency_penalty: float = 0.35,
        recency_decay: float = 0.7,
        temperature: float = 0.3,
        temperature_max: float | None = None,
        weight_decay: float = 0.95,
        feature_groups: list[list[int]] | None = None,
        constant_tree_frequency: int = 25,
        seed: int | None = None,
    ):
        super().__init__()
        self.redundancy_penalty = redundancy_penalty
        self.relevance_alpha = relevance_alpha
        self.recency_penalty = recency_penalty
        self.recency_decay = recency_decay
        self.temperature = temperature
        self.temperature_max = temperature_max
        self.weight_decay = weight_decay
        self.feature_groups = feature_groups
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)

        self.constant_frequency = constant_tree_frequency

    def initialize(
        self,
        X: np.ndarray,
        n_features: int,
        min_features: int,
        max_features: int,
        rounds: int,
    ) -> int:
        self.n_features = n_features
        self.X_ = X.view()
        self.n_bins_ = max(10, int(np.sqrt(X.shape[0] / 5)))
        self.rounds_ = rounds
        self.schedule_rounds_ = (
            max(1, rounds) if self.temperature_max is not None else None
        )

        self.corr_matrix_ = np.corrcoef(X, rowvar=False)
        self.corr_matrix_ = np.nan_to_num(self.corr_matrix_, nan=0.0)

        # precompute quantile thresholds for each feature
        self.quantiles = np.linspace(0, 1, self.n_bins_ + 1)
        self.n_thresh_ = self.n_bins_ + 1
        self.x_thresholds_ = np.array(
            [np.quantile(X[:, f], self.quantiles) for f in range(n_features)],
            dtype=np.float32,
        )

        self.feature_weights_ = np.zeros(n_features)
        self.recency_scores_ = np.zeros(n_features)
        self._rng = np.random.default_rng(self.seed)
        self._size_gen = self._size_generator(
            n_features, min_features, max_features, rounds
        )
        return rounds

    def _size_generator(
        self, n_features: int, min_features: int, max_features: int, rounds: int
    ) -> Generator[int, None, None]:
        """Yield kernel sizes, progressively increasing from min to max."""
        if n_features == 1:
            while True:
                yield 1

        max_size = min(n_features, max_features)
        sizes = list(range(min_features, max_size + 1))
        rounds_per_size = max(1, rounds // len(sizes))

        for size in sizes:
            for _ in range(rounds_per_size):
                yield size
        # max size if more rounds needed
        while True:
            yield max_size

    def get_features(self, round_idx: int, residuals: np.ndarray) -> list[int]:
        if round_idx > 0 and round_idx % self.constant_frequency == 0:
            tree_type = "constant"
            selected = list(range(self.n_features))
        else:
            tree_type = "kernel"
            n_features = next(self._size_gen)
            relevance = self._compute_relevance(residuals)
            selected = self._select_features(n_features, relevance, round_idx)

        return self._complete_groups(selected), tree_type

    def update(self, feature_indices: list[int], gain: float) -> None:
        self.recency_scores_ *= self.recency_decay
        for idx in feature_indices:
            self.recency_scores_[idx] = 1.0

        self.feature_weights_ *= self.weight_decay
        if gain > 0:
            weight_increment = gain / len(feature_indices)
            for idx in feature_indices:
                self.feature_weights_[idx] += weight_increment

    def _compute_relevance(self, pseudoresiduals: np.ndarray) -> np.ndarray:
        """Compute relevance scores from MI with residuals, history, and recency."""
        residuals = np.ascontiguousarray(pseudoresiduals.ravel(), dtype=np.float32)
        y_thresholds = np.quantile(residuals, self.quantiles).astype(np.float32)

        raw_scores = np.zeros(self.n_features, dtype=np.float32)
        _mi_lib.histogram_mi_batch(
            np.ascontiguousarray(self.X_, dtype=np.float32),
            residuals,
            self.X_.shape[0],
            self.n_features,
            np.ascontiguousarray(self.x_thresholds_),
            np.ascontiguousarray(y_thresholds),
            self.n_thresh_,
            raw_scores,
        )

        # normalize to [0, 1]
        scores_norm = raw_scores / (raw_scores.max() + 1e-10)
        weights_norm = self.feature_weights_ / (self.feature_weights_.max() + 1e-10)

        relevance = (
            self.relevance_alpha * scores_norm
            + (1 - self.relevance_alpha) * weights_norm
            - self.recency_penalty * self.recency_scores_
        )
        return np.maximum(relevance, 0.0)

    def _get_temperature(self, round_idx: int) -> float:
        """Compute temperature for the current round."""
        if self.schedule_rounds_ is None:
            return self.temperature
        progress = min(round_idx / self.schedule_rounds_, 1.0)
        return (
            self.temperature_max + (self.temperature - self.temperature_max) * progress
        )

    def _select_features(
        self, k: int, relevance: np.ndarray, round_idx: int
    ) -> list[int]:
        """Select k features probabilistically using relevance and redundancy."""
        temp = self._get_temperature(round_idx)
        selected = []
        available = list(range(self.n_features))

        for _ in range(min(k, self.n_features)):
            if not available:
                break
            scores = np.zeros(len(available))
            for i, j in enumerate(available):
                if selected:
                    redundancy = np.mean(
                        [abs(self.corr_matrix_[j, s]) for s in selected]
                    )
                else:
                    redundancy = 0.0

                scores[i] = relevance[j] - self.redundancy_penalty * redundancy

            scaled = scores / temp
            exp_scores = np.exp(scaled - scaled.max())
            probs = exp_scores / exp_scores.sum()

            idx = self._rng.choice(len(available), p=probs)
            feat = available[idx]

            selected.append(feat)
            available.pop(idx)

        return selected
