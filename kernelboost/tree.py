import warnings
import numpy as np
from dataclasses import dataclass
from .estimator import KernelEstimator
from .utilities import r2_score


@dataclass
class Leaf:
    """Terminal node containing a kernel estimator or constant."""
    estimator: KernelEstimator | float


@dataclass
class Branch:
    """Internal node with split condition."""
    feature: int
    threshold: float
    left: 'Branch | Leaf'
    right: 'Branch | Leaf'


Node = Branch | Leaf


@dataclass
class CompiledTree:
    """Flat array form for KernelTree for faster prediction."""
    conditions: list[list[tuple]]  # per-leaf: [(feat, thresh, dir), ...]
    estimators: list[KernelEstimator | float]
    is_kernel: list[bool]  # pre-computed type flag
    categorical: list[int]

    def predict(self, X: np.ndarray, X_num: np.ndarray | None = None) -> np.ndarray:
        """Predict by evaluating leaf conditions and applying estimators."""
        n = X.shape[0]
        out = np.zeros((n, 1), dtype=np.float32)
        if X_num is None:
            X_num = np.delete(X, self.categorical, axis=1) if self.categorical else X

        for conds, est, is_kern in zip(self.conditions, self.estimators, self.is_kernel):
            mask = np.ones(n, dtype=bool)
            for feat, thresh, direction in conds:
                if direction == 0:
                    mask &= X[:, feat] <= thresh
                else:
                    mask &= X[:, feat] > thresh

            if is_kern:
                out[mask] = est.predict(X_num[mask]).reshape(-1, 1)
            else:
                out[mask] = est
        return out

    def predict_quantiles(
        self,
        X: np.ndarray,
        X_num: np.ndarray | None = None,
        quantiles: tuple = (0.1, 0.5, 0.9)
    ) -> np.ndarray:
        """EXPERIMENTAL. Predict conditional quantiles."""
        n = X.shape[0]
        result = np.zeros((n, len(quantiles)), dtype=np.float32)

        if X_num is None:
            X_num = np.delete(X, self.categorical, axis=1) if self.categorical else X

        for conds, est, is_kern in zip(self.conditions, self.estimators, self.is_kernel):
            mask = np.ones(n, dtype=bool)
            for feat, thresh, direction in conds:
                if direction == 0:
                    mask &= X[:, feat] <= thresh
                else:
                    mask &= X[:, feat] > thresh

            if not np.any(mask):
                continue

            if is_kern:
                result[mask] = est.predict_quantiles(X_num[mask], quantiles=quantiles)
            else:
                result[mask] = est

        return result
    


class KernelTree:
    """
    Decision tree that splits either by density (for kernel leaves) or by MSE gain
    (categorical features or constant leaves).

    Args:
    min_sample : int, default=500
        Minimum samples required for a kernel leaf.
    max_sample : int, default=5000
        Maximum samples before forcing a split on numerical features.
    max_depth : int, default=3
        Maximum depth for categorical splits.
    feature_types : dict, optional
        Map of feature index to type ('C' for categorical, 'N' for numerical).
        If None, types are auto-detected.
    overlap_epsilon : float, default=0.05
        Fraction of feature range to expand training data beyond split boundaries.
        Mitigates boundary bias in kernel estimation.
    use_gpu : bool, default=False
        Whether to use GPU acceleration for kernel estimation.
    kernel_type : str, default='gaussian'
        Kernel type: 'gaussian' or 'laplace'.
    search_rounds : int, default=20
        Number of optimization rounds for precision search in kernel estimators.
    bounds : tuple, default=(0.10, 35.0)
        Lower and upper bounds for precision optimization.
    initial_precision : float, default=0.0
        Initial precision value for optimization. 0.0 means auto.
    sample_share : float, default=1.0
        Fraction of data to use for precision optimization.
    precision_method : str, default='pilot-cv'
        Precision selection method: 'search' (LOO-CV) or 'silverman'.
    pilot_factor : float, default=3.0
        Multiplier for pilot precision bounds: search range is [p/factor, p*factor].
    tree_type : str, default='kernel'
        Leaf node type: 'kernel' for kernel estimation or 'constant' for constant leaves.
    gain_threshold : float, default=1e-3
        Minimum MSE gain required for a split in constant leaf mode.
    quantiles : list, optional
        Split candidate quantiles for constant leaf mode.
        If None, defaults to linspace(0.01, 0.99, 99).
    """

    def __init__(
        self,
        min_sample: int = 500,
        max_sample: int = 5000,
        max_depth: int = 3,
        feature_types: dict = None,
        overlap_epsilon: float = 0.05,
        use_gpu: bool = False,
        kernel_type: str = 'laplace',
        search_rounds: int = 20,
        bounds: tuple = (0.10, 35.0),
        initial_precision: float = 0.0,
        sample_share: float = 1.0,
        precision_method: str = 'pilot-cv',
        pilot_factor: float = 3.0,
        tree_type: str = 'kernel',
        gain_threshold: float = 1e-3,
        quantiles: list = None,
    ):

        self.min_sample = min_sample
        self.max_sample = max_sample
        self.max_depth = max_depth
        self.feature_types = feature_types
        self.overlap_epsilon = overlap_epsilon
        self.use_gpu = use_gpu
        self.kernel_type = kernel_type
        self.search_rounds = search_rounds
        self.bounds = bounds
        self.initial_precision = initial_precision
        self.sample_share = sample_share
        self.precision_method = precision_method
        self.pilot_factor = pilot_factor
        self.tree_type = tree_type
        self.gain_threshold = gain_threshold

        if quantiles is None:
            self.quantiles = np.linspace(0.01, 0.99, 99)
        else:
            self.quantiles = quantiles

        self.kernel_optimization = {
            'kernel_type': kernel_type,
            'search_rounds': search_rounds,
            'bounds': bounds,
            'initial_precision': initial_precision,
            'sample_share': sample_share,
            'precision_method': precision_method,
            'pilot_factor': pilot_factor,
        }

        # min sample decreased, depth increase for non-kernel trees
        self._const_min_sample = max(50, self.min_sample // 5)
        self._const_max_depth = self.max_depth + 3

        self._validate_params()

    def _validate_params(self):
        if self.kernel_type not in {'gaussian', 'laplace'}:
            raise ValueError(f"kernel_type must be 'gaussian' or 'laplace', got '{self.kernel_type}'")
        if self.min_sample <= 0:
            raise ValueError("min_sample must be positive")
        if self.max_sample <= self.min_sample:
            raise ValueError("max_sample must be greater than min_sample")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.feature_types is not None:
            invalid = [k for k, v in self.feature_types.items() if v not in ('C', 'N')]
            if invalid:
                raise ValueError(f"feature_types values must be 'C' or 'N', got invalid keys: {invalid}")
        if not (0.0 <= self.overlap_epsilon < 0.5):
            raise ValueError("overlap_epsilon must be in [0.0, 0.5)")
        if self.tree_type not in {'kernel', 'constant'}:
            raise ValueError(f"tree_type must be 'kernel' or 'constant', got '{self.tree_type}'")
        if self.gain_threshold < 0:
            raise ValueError(f"gain_threshold must be >= 0, got {self.gain_threshold}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelTree":
        """Fit the tree to training data."""
        self.X_ = X.astype(np.float32)
        self.y_ = y.astype(np.float32).ravel()
        self.n_samples_, self.n_features_ = X.shape

        if self.tree_type == 'constant':
            self.categorical_ = []
            self.numerical_ = list(range(self.n_features_))
            self._use_kernel = False
            # uses indices rather than values:
            sorted_by_feat = [np.argsort(self.X_[:, f]) for f in range(self.n_features_)]
            self.root_ = self._grow_constant(sorted_by_feat)
        else:
            self.feature_ranges_ = self.X_.max(axis=0) - self.X_.min(axis=0)
            self._detect_types()
            self.root_ = self._grow_numerical(self.X_, self.y_)
            if self.categorical_:
                self.root_ = self._expand_categorical(self.root_, self.X_, self.y_)

        self.compiled_ = self._compile(self.root_)
        self.depth_ = self._compute_depth(self.root_)
        del self.X_, self.y_
        return self

    def _detect_types(self) -> None:
        """Classify features as categorical or numerical."""
        self.categorical_, self.numerical_ = [], []
        if self.feature_types is not None:
            for i in range(self.n_features_):
                if self.feature_types.get(i) == "C":
                    self.categorical_.append(i)
                else:
                    self.numerical_.append(i)
        else:
            for i in range(self.n_features_):
                sample = self.X_[:min(3000, self.n_samples_), i]
                _, counts = np.unique(sample, return_counts=True)
                n_unique = len(counts)
                if n_unique == 1:
                    warnings.warn(f"Feature {i} is a constant.")
                if n_unique < 50 or counts.max() / len(sample) > 0.2:
                    self.categorical_.append(i)
                else:
                    self.numerical_.append(i)
        self._use_kernel = len(self.numerical_) > 0

    def _grow_numerical(self, X: np.ndarray, y: np.ndarray) -> Node:
        """Grow a tree on numerical features."""
        # make a leaf if sample size below max_sample
        if len(y) <= self.max_sample:
            return self._make_leaf(X, y)

        # otherwise split:
        split = self._find_density_split(X, len(y))
        if split is None:
            return self._make_leaf(X, y)

        feat, thresh = split
        left_mask = X[:, feat] <= thresh
        return Branch(feat, thresh,
                      self._grow_numerical(X[left_mask], y[left_mask]),
                      self._grow_numerical(X[~left_mask], y[~left_mask]))

    def _find_density_split(self, X: np.ndarray, n: int) -> tuple[int, float] | None:
        """Find best numerical split based on density variation."""
        if not self.numerical_:
            return None

        qs = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
        min_ratio = 2 * self.min_sample / n  # makes room for categorical splits

        best, best_feat, best_val = -np.inf, None, None
        for f in self.numerical_:
            cuts = np.quantile(X[:, f], qs)
            rng = cuts[-1] - cuts[0]
            if rng == 0:
                continue
            for i in range(1, len(qs) - 1):
                if qs[i] < min_ratio or qs[i] > (1 - min_ratio):
                    continue
                score = abs((cuts[i] - cuts[i-1]) / (rng * (qs[i] - qs[i-1])) - 1)
                if score > best:
                    best, best_feat, best_val = score, f, cuts[i]
        return (best_feat, best_val) if best_feat is not None else None
    
    def _expand_categorical(self, node: Node, X: np.ndarray, y: np.ndarray) -> Node:
        """Replace leaves in numerical tree with categorical subtrees."""
        # Leaf: check categorical features and grow categorical tree
        if isinstance(node, Leaf):
            if len(y) >= 1.5 * self.min_sample:
                return self._grow_categorical(X, y, depth=0)
            return node  

        # Branch: partition data and recurse
        left_mask = X[:, node.feature] <= node.threshold
        return Branch(
            node.feature,
            node.threshold,
            self._expand_categorical(node.left, X[left_mask], y[left_mask]),
            self._expand_categorical(node.right, X[~left_mask], y[~left_mask])
        )
    
    def _grow_categorical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int) -> Node:
        """Grow a tree on categorical features."""
        if depth >= self.max_depth or len(y) < 1.5 * self.min_sample:
            return self._make_leaf(X, y)

        split = self._find_categorical_split(X, y)
        if split is None:
            return self._make_leaf(X, y)

        feat, thresh, gain = split
        if gain < self.gain_threshold: 
            return self._make_leaf(X, y)
        
        left_mask = X[:, feat] <= thresh
        return Branch(feat, thresh,
                      self._grow_categorical(X[left_mask], y[left_mask], depth + 1),
                      self._grow_categorical(X[~left_mask], y[~left_mask], depth + 1))
        
    def _find_categorical_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        """Find best categorical split by MSE reduction."""
        n = len(y)
        base_mse = np.var(y)
        best_mse, best_feat, best_thresh = base_mse, None, None

        for f in self.categorical_:
            values = np.unique(X[:, f]) 
            for thresh in values[:-1]:
                left_mask = X[:, f] <= thresh
                n_left, n_right = left_mask.sum(), (~left_mask).sum()

                if n_left < self.min_sample or n_right < self.min_sample:
                    continue
                # variance approximation to MSE:
                mse = (n_left * np.var(y[left_mask]) + n_right * np.var(y[~left_mask])) / n

                if mse < best_mse:
                    best_mse, best_feat, best_thresh = mse, f, thresh

        if best_feat is None:
            return None
        
        return best_feat, best_thresh, base_mse - best_mse
    
    def _make_leaf(self, X: np.ndarray, y: np.ndarray) -> Leaf:
        """Create leaf with kernel estimator or mean constant."""
        n = len(y)
        if self._use_kernel and n <= self.max_sample:
            X_num = np.delete(X, self.categorical_, axis=1) if self.categorical_ else X
            k = KernelEstimator(use_gpu=self.use_gpu, **self.kernel_optimization)
            k.fit(X_num, y)
            return Leaf(k)
        return Leaf(float(np.mean(y)))

    def _grow_constant(self, sorted_by_feat: list[np.ndarray], depth: int = 0) -> Node:
        """Grow a tree with constant leaves."""
        n = len(sorted_by_feat[0])
        samples = sorted_by_feat[0]

        if depth >= self._const_max_depth or n < 1.5 * self._const_min_sample:
            return Leaf(float(np.mean(self.y_[samples])))

        split = self._find_constant_split(sorted_by_feat, n)
        if split is None:
            return Leaf(float(np.mean(self.y_[samples])))

        feat, thresh, gain = split
        if gain < self.gain_threshold:
            return Leaf(float(np.mean(self.y_[samples])))

        left_sorted, right_sorted = [], []
        for f in range(self.n_features_):
            left_mask = self.X_[sorted_by_feat[f], feat] <= thresh
            left_sorted.append(sorted_by_feat[f][left_mask])
            right_sorted.append(sorted_by_feat[f][~left_mask])

        return Branch(feat, thresh,
                      self._grow_constant(left_sorted, depth + 1),
                      self._grow_constant(right_sorted, depth + 1))

    def _find_constant_split(
        self,
        sorted_by_feat: list[np.ndarray],
        n: int) -> tuple[int, float, float] | None:
        """Find best split by MSE reduction across all features."""
        base_mse = np.var(self.y_[sorted_by_feat[0]])
        best_mse, best_feat, best_thresh = base_mse, None, None
        min_s = self._const_min_sample

        for f in range(self.n_features_):
            idx = sorted_by_feat[f]
            col_sorted = self.X_[idx, f]
            y_sorted = self.y_[idx].astype(np.float64)

            cum_sum = np.cumsum(y_sorted)
            cum_sq = np.cumsum(y_sorted ** 2)
            total_sum = cum_sum[-1]
            total_sq = cum_sq[-1]

            values = np.quantile(col_sorted, self.quantiles)
            values = values[:-1]  # skip last candidate
            positions = np.searchsorted(col_sorted, values, side='right')

            # filter valid splits
            valid = (positions >= min_s) & (positions <= n - min_s)
            if not np.any(valid):
                continue

            pos = positions[valid]
            thresholds = values[valid]

            # skip duplicates — same partition, same MSE
            unique_mask = np.empty(len(pos), dtype=bool)
            unique_mask[0] = True
            unique_mask[1:] = np.diff(pos) != 0
            pos = pos[unique_mask]
            thresholds = thresholds[unique_mask]

            # vectorized MSE via cumulative sums
            left_n = pos.astype(np.float64)
            left_sum = cum_sum[pos - 1]
            left_sq = cum_sq[pos - 1]
            right_n = n - left_n
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq

            # guards against negatives
            left_var = np.maximum(0.0, left_sq / left_n - (left_sum / left_n) ** 2)
            right_var = np.maximum(0.0, right_sq / right_n - (right_sum / right_n) ** 2)

            mse = (left_n * left_var + right_n * right_var) / n

            idx_best = np.argmin(mse)
            if mse[idx_best] < best_mse:
                best_mse = mse[idx_best]
                best_feat = f
                best_thresh = float(thresholds[idx_best])

        if best_feat is None:
            return None

        return best_feat, best_thresh, base_mse - best_mse
    
    def _compute_depth(self, node: Node, current: int = 0) -> int:
        """Compute the maximum depth of the tree."""
        if isinstance(node, Leaf):
            return current
        return max(
            self._compute_depth(node.left, current + 1),
            self._compute_depth(node.right, current + 1)
        )

    def _compile(self, root: Node) -> CompiledTree:
        """Convert nested tree structure to flat thresholds for prediction
        and create overlap for training data over the thresholds."""
        conditions, estimators, is_kernel = [], [], []

        def collapse(conds):
            upper, lower = {}, {}
            for feat, thresh, d in conds:
                if d == 0: # 0 is smaller or equal
                    upper[feat] = min(upper.get(feat, thresh), thresh)
                else:
                    lower[feat] = max(lower.get(feat, thresh), thresh)
            return [(f, t, 0) for f, t in upper.items()] + [(f, t, 1) for f, t in lower.items()]

        def collect(node: Node, path: list):
            if isinstance(node, Leaf):
                conditions.append(collapse(path))
                estimators.append(node.estimator)
                is_kernel.append(isinstance(node.estimator, KernelEstimator))
            else:
                collect(node.left, path + [(node.feature, node.threshold, 0)])  # left is smaller or equal
                collect(node.right, path + [(node.feature, node.threshold, 1)]) # right is greater

        collect(root, [])

        if self.overlap_epsilon > 0:
            for conds, est, is_kern in zip(conditions, estimators, is_kernel):
                if is_kern and conds:  # kernel leaf with conditions
                    expanded_mask = self._compute_expanded_mask(conds)
                    X_expanded = self.X_[expanded_mask]
                    y_expanded = self.y_[expanded_mask]
                    X_num = np.delete(X_expanded, self.categorical_, axis=1) if self.categorical_ else X_expanded
                    est.X_ = X_num
                    est.y_ = y_expanded.reshape(-1, 1)
                    est.n_samples_ = len(y_expanded)

        return CompiledTree(conditions, estimators, is_kernel, self.categorical_)
    
    def _compute_expanded_mask(self, conditions: list[tuple]) -> np.ndarray:
        """Compute training mask with expanded thresholds for overlap."""
        mask = np.ones(self.n_samples_, dtype=bool)
        for feat, thresh, direction in conditions:
            if feat in self.categorical_:  # no overlap for categorical
                eps = 0.0
            else:
                eps = self.overlap_epsilon * self.feature_ranges_[feat]
            if direction == 0:  # <=
                mask &= self.X_[:, feat] <= thresh + eps
            else:
                mask &= self.X_[:, feat] > thresh - eps
        return mask
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the compiled tree."""
        return self.compiled_.predict(np.asarray(X, dtype=np.float32))

    def predict_quantiles(
        self, X: np.ndarray, quantiles: tuple = (0.1, 0.5, 0.9)
    ) -> np.ndarray:
        """EXPERIMENTAL. Predict conditional quantiles."""
        return self.compiled_.predict_quantiles(
            np.asarray(X, dtype=np.float32), quantiles=quantiles
        )

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the tree and return predictions on X."""
        return self.fit(X, y).predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score on given data."""
        return r2_score(y, self.predict(X))

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        # deep kept for compatibility
        return {
            'min_sample': self.min_sample,
            'max_sample': self.max_sample,
            'max_depth': self.max_depth,
            'feature_types': self.feature_types,
            'overlap_epsilon': self.overlap_epsilon,
            'use_gpu': self.use_gpu,
            'kernel_type': self.kernel_type,
            'search_rounds': self.search_rounds,
            'bounds': self.bounds,
            'initial_precision': self.initial_precision,
            'sample_share': self.sample_share,
            'precision_method': self.precision_method,
            'pilot_factor': self.pilot_factor,
            'tree_type': self.tree_type,
            'gain_threshold': self.gain_threshold,
            'quantiles': self.quantiles,
        }

    def set_params(self, **params) -> "KernelTree":
        """Set parameters for this estimator."""
        valid_keys = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter '{key}'")
            setattr(self, key, value)

        self.kernel_optimization = {
            'kernel_type': self.kernel_type,
            'search_rounds': self.search_rounds,
            'bounds': self.bounds,
            'initial_precision': self.initial_precision,
            'sample_share': self.sample_share,
            'precision_method': self.precision_method,
            'pilot_factor': self.pilot_factor,
        }

        self._const_min_sample = max(50, self.min_sample // 5)
        self._const_max_depth = self.max_depth + 3

        self._validate_params()

        return self



