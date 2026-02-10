from abc import ABC, abstractmethod
from collections.abc import Generator
import numpy as np


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
    def get_features(self, round_idx: int, residuals: np.ndarray) -> list[int]:
        """
        Get feature indices for the next boosting round.

        Args:
        round_idx : int
            Current round index (0-based)
        residuals : np.ndarray
            Current pseudo-residuals (n_samples,)

        Returns:
        list[int]
            Feature indices to use for this round
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

    def __init__(self, seed: int | None = None, feature_groups: list[list[int]] | None = None):
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
        self._gen = self._feature_generator(n_features, min_features, max_features, rounds)
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

    def get_features(self, round_idx: int, residuals: np.ndarray) -> list[int]:
        selected = next(self._gen)
        return self._complete_groups(selected)


class SmartSelector(FeatureSelector):
    """
    Feature selection using mRMR-style approach using correlations. 
    Selects features probabilistically based on relevance, redundancy and recency. 
    Kernel sizes progress from small to large.

    Args:
    redundancy_penalty : float, default=0.4
        Weight for redundancy penalty (0 = ignore, 1 = strong penalty)
    relevance_alpha : float, default=0.6
        Balance between residual correlation (1.0) and historical weight (0.0)
    recency_penalty : float, default=0.3
        Penalty applied to recently-used features (0 = no penalty, 1 = strong)
    recency_decay : float, default=0.7
        Decay factor for recency penalty each round (0 = instant decay, 1 = no decay)
    temperature : float, default=0.3
        Softmax temperature for feature selection. Lower = greedier, higher = more exploration.
    weight_decay : float, default=0.9
        Decay factor for feature weights each round.
    feature_groups : list[list[int]] | None, default=None
        Groups of features that should be selected together.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        redundancy_penalty: float = 0.4,
        relevance_alpha: float = 0.6,
        recency_penalty: float = 0.3,
        recency_decay: float = 0.7,
        temperature: float = 0.3,
        weight_decay: float = 0.9,
        feature_groups: list[list[int]] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.redundancy_penalty = redundancy_penalty
        self.relevance_alpha = relevance_alpha
        self.recency_penalty = recency_penalty
        self.recency_decay = recency_decay
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.feature_groups = feature_groups
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)

    def initialize(
        self,
        X: np.ndarray,
        n_features: int,
        min_features: int,
        max_features: int,
        rounds: int,
    ) -> int:
        self.n_features = n_features

        self.X_std_ = X.std(axis=0)
        self.X_centered_ = X - X.mean(axis=0)

        self.corr_matrix_ = np.corrcoef(X, rowvar=False)
        self.corr_matrix_ = np.nan_to_num(self.corr_matrix_, nan=0.0)
        self.feature_weights_ = np.zeros(n_features)
        self.recency_scores_ = np.zeros(n_features)
        self._rng = np.random.default_rng(self.seed)
        self._size_gen = self._size_generator(n_features, min_features, max_features, rounds)
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
        k = next(self._size_gen)
        relevance = self._compute_relevance(residuals)
        selected = self._select_features(k, relevance)
        return self._complete_groups(selected)

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
        """Compute relevance scores from residual correlation, history, and recency."""
        pseudoresiduals = pseudoresiduals.ravel()

        # correlation with pseudoresiduals
        r_centered = pseudoresiduals - pseudoresiduals.mean()
        r_std = r_centered.std()

        cov = self.X_centered_.T @ r_centered / len(pseudoresiduals)
        denom = self.X_std_ * r_std
        correlations = np.abs(np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-10))

        # normalize to max values
        corr_norm = correlations / (correlations.max() + 1e-10)
        weights_norm = self.feature_weights_ / (self.feature_weights_.max() + 1e-10)

        relevance = (
            self.relevance_alpha * corr_norm +
            (1 - self.relevance_alpha) * weights_norm - 
            self.recency_penalty * self.recency_scores_
        )
        relevance = np.maximum(relevance, 0.0)

        return relevance

    def _select_features(self, k: int, relevance: np.ndarray) -> list[int]:
        """Select k features probabilistically using relevance and redundancy."""
        selected = []
        available = list(range(self.n_features))

        for _ in range(min(k, self.n_features)):
            if not available:
                break
            scores = np.zeros(len(available))
            for i, j in enumerate(available):
                # redundancy with already selected
                if selected:
                    redundancy = np.mean([
                        abs(self.corr_matrix_[j, s]) for s in selected
                    ])
                else:
                    redundancy = 0.0

                scores[i] = relevance[j] - self.redundancy_penalty * redundancy

            # convert scores to probabilities 
            scaled = scores / self.temperature
            exp_scores = np.exp(scaled - scaled.max())  # subtract max for numerical stability
            probs = exp_scores / exp_scores.sum()

            # select based on probabilities
            idx = self._rng.choice(len(available), p=probs)
            feat = available[idx]

            selected.append(feat)
            available.pop(idx)

        return selected
