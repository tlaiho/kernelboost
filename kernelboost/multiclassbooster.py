import numpy as np
from .booster import KernelBooster
from .objectives import EntropyObjective
from .feature_selection import FeatureSelector


class MulticlassBooster:
    """
    Multiclass gradient boosting classifier using one-vs-rest binary boosters.

    Trains separate KernelBooster instances for each class and combines
    predictions via softmax normalization.

    Args:
    class_weights : str, list, or None, default=None
        - None: Equal weights for all classes
        - "balanced": Weights inversely proportional to class frequencies
        - list of dicts: Per-class weight dicts like [{0: w0, 1: w1}, ...]
    feature_indices : dict or None, default=None
        Dict mapping class names to feature indices to use for that class.
        Allows different features per class. If None, all features used.
    feature_selector : FeatureSelector or None, default=None
        FeatureSelector for KernelBoosters.
    feature_names : list, dict, or None, default=None
        - list: Same names for all classes
        - dict: Class-specific feature names
        - None: Uses integer indices
    min_features : int, default=1
        Minimum features per round.
    max_features : int, default=None
        Maximum features per round. If None, uses min(10, n_features).
    rounds : int or None, default=None
        Boosting rounds. Auto-calculated from n_features if None.
    subsample_share : float, default=0.5
        Training sample share per round.
    lambda1 : float, default=0.0005
        L1 regularization for line search.
    learning_rate : float, default=0.1
        Learning rate (shrinkage factor) for step sizes. Must be in (0, 1].
    max_tree_depth : int, default=3
        Maximum depth for kernel trees.
    max_sample : int, default=10000
        Maximum samples per kernel leaf (triggers splits).
    min_sample : int, default=1000
        Minimum samples for kernel fitting.
    overlap_epsilon : float, default=0.0
        Overlap epsilon for kernel tree splits.
    kernel_type : str, default='laplace'
        Kernel type: 'gaussian' or 'laplace'.
    search_rounds : int, default=20
        Precision optimization iterations.
    precision_method : str, default='pilot-cv'
        Precision selection method: 'pilot-cv' (pilot bounds + LOO-CV),
        'search' (LOO-CV with fixed bounds), or 'silverman' (rule-of-thumb).
    bounds : tuple, default=(0.20, 35.0)
        Precision search bounds.
    initial_precision : float, default=0.0
        Starting precision. 0 means auto.
    sample_share : float, default=1.0
        Share of samples for precision CV.
    early_stopping_rounds : int, default=5
        Stop training if validation score doesn't improve for this many
        consecutive rounds. Only used when eval_set is provided to fit().
    stopping_threshold : float, default=0.001
        Early stopping threshold for mean |rho|.
    verbose : int, default=1
        Verbosity level. 0 = silent, 1 = progress.
    use_gpu : bool, default=True
        Use GPU acceleration via CuPy.
    """

    def __init__(
        self,
        class_weights: str | list | None = None,
        feature_indices: dict | None = None,
        feature_selector: FeatureSelector | None = None,
        feature_names: list | dict | None = None,
        min_features: int = 1,
        max_features: int = None,
        rounds: int | None = None,
        subsample_share: float = 0.5,
        lambda1: float = 0.0005,
        learning_rate: float = 0.1,
        max_tree_depth: int = 3,
        max_sample: int = 10000,
        min_sample: int = 1000,
        overlap_epsilon: float = 0.0,
        kernel_type: str = 'laplace',
        precision_method: str = 'pilot-cv',
        search_rounds: int = 20,
        bounds: tuple = (0.20, 35.0),
        initial_precision: float = 0.0,
        sample_share: float = 1.0,
        early_stopping_rounds: int = 5,
        stopping_threshold: float = 0.001,
        verbose: int = 1,
        use_gpu: bool = True,
    ):
        self.class_weights = class_weights
        self.feature_indices = feature_indices

        self.feature_selector = feature_selector
        self.feature_names = feature_names
        self.min_features = min_features
        self.max_features = max_features

        self.rounds = rounds
        self.subsample_share = subsample_share
        self.lambda1 = lambda1
        self.learning_rate = learning_rate

        self.early_stopping_rounds = early_stopping_rounds
        self.stopping_threshold = stopping_threshold

        self.verbose = verbose
        self.use_gpu = use_gpu

        self.kernel_optimization = {
            'kernel_type': kernel_type,
            'precision_method': precision_method,
            'search_rounds': search_rounds,
            'bounds': bounds,
            'initial_precision': initial_precision,
            'sample_share': sample_share,
        }

        self.tree_optimization = {
            'max_tree_depth': max_tree_depth,
            'max_sample': max_sample,
            'min_sample': min_sample,
            'overlap_epsilon': overlap_epsilon,
        }

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        kernel_type = self.kernel_optimization['kernel_type']
        max_sample = self.tree_optimization['max_sample']
        min_sample = self.tree_optimization['min_sample']
        bounds = self.kernel_optimization['bounds']

        if kernel_type not in {'gaussian', 'laplace'}:
            raise ValueError(f"kernel_type must be 'gaussian' or 'laplace', got '{kernel_type}'")

        if max_sample <= min_sample:
            raise ValueError(f"max_sample ({max_sample}) must be > min_sample ({min_sample})")

        if self.lambda1 < 0:
            raise ValueError(f"lambda1 must be >= 0, got {self.lambda1}")

        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

        if not 0 < self.subsample_share <= 1:
            raise ValueError(f"subsample_share must be in (0, 1], got {self.subsample_share}")

        if self.rounds is not None and self.rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {self.rounds}")

        if self.stopping_threshold < 0:
            raise ValueError(f"stopping_threshold must be >= 0, got {self.stopping_threshold}")

        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"bounds must be (lower, upper) with lower < upper, got {bounds}")

        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")

        if self.max_features is not None and self.max_features < self.min_features:
            raise ValueError(
                f"max_features ({self.max_features}) must be >= "
                f"min_features ({self.min_features})"
            )

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        if y.ndim == 1: # label encoded
            pass
        elif y.ndim == 2: # one-hot encoded
            if y.shape[1] < 2:
                raise ValueError(
                    f"For 2D y (one-hot), expected at least 2 columns, got {y.shape[1]}"
                )
        else:
            raise ValueError(f"y must be 1D or 2D array, got {y.ndim}D")

        n_samples_y = y.shape[0]
        if X.shape[0] != n_samples_y:
            raise ValueError(
                f"X and y have different number of samples: {X.shape[0]} vs {n_samples_y}"
            )

        min_sample = self.tree_optimization['min_sample']
        if X.shape[0] < min_sample:
            raise ValueError(f"Not enough samples ({X.shape[0]}) for min_sample ({min_sample})")

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")

    def _convert_to_onehot(self, y: np.ndarray, classes: np.ndarray = None) -> np.ndarray:
        """Convert 1D label array to one-hot encoding.

        Args:
            y: 1D array of class labels
            classes: Optional array of class values to use. If None, uses np.unique(y).
                     Use self.classes_ for validation data to ensure consistent encoding.
        """
        if classes is None:
            classes = np.unique(y)
        n_classes = len(classes)
        n_samples = y.shape[0]

        onehot = np.zeros((n_samples, n_classes), dtype=np.int32)
        for i, cls in enumerate(classes):
            onehot[y == cls, i] = 1

        return onehot

    def _convert_from_onehot(self, y_onehot: np.ndarray) -> np.ndarray:
        """Convert one-hot encoding to 1D label array."""
        return np.argmax(y_onehot, axis=1)

    def _setup_class_weights(self, y_onehot: np.ndarray) -> list[dict]:
        """Set up class weights based on configuration."""
        if self.class_weights is None:
            return [{0: 1, 1: 1} for _ in range(self.n_classes_)]

        if self.class_weights == "balanced":
            weights = []
            for i in range(self.n_classes_):
                _, counts = np.unique(y_onehot[:, i], return_counts=True)
                weights.append({
                    0: self.n_samples_ / counts[0],
                    1: self.n_samples_ / counts[1]
                })
            return weights

        if isinstance(self.class_weights, list):
            if len(self.class_weights) != self.n_classes_:
                raise ValueError(
                    f"class_weights list length ({len(self.class_weights)}) "
                    f"doesn't match n_classes ({self.n_classes_})"
                )
            return self.class_weights

        raise ValueError(
            "class_weights must be None, 'balanced', or a list of dicts"
        )

    def _setup_feature_indices(self) -> dict:
        """Set up feature indices for each class."""
        if self.feature_indices is None:
            return {cls: list(range(self.n_features_in_)) for cls in self.classes_}

        if len(self.feature_indices) != self.n_classes_:
            raise ValueError(
                f"feature_indices has {len(self.feature_indices)} entries but "
                f"there are {self.n_classes_} classes"
            )
        return self.feature_indices

    def _setup_feature_names(self) -> dict:
        """Set up feature names for each class."""
        if self.feature_names is None:
            return {cls: list(range(self.n_features_in_)) for cls in self.classes_}

        if isinstance(self.feature_names, list):
            if len(self.feature_names) != self.n_features_in_:
                raise ValueError(
                    f"feature_names list length ({len(self.feature_names)}) "
                    f"doesn't match n_features ({self.n_features_in_})"
                )
            return {cls: list(self.feature_names) for cls in self.classes_}

        if isinstance(self.feature_names, dict):
            if len(self.feature_names) != self.n_classes_:
                raise ValueError(
                    f"feature_names dict has {len(self.feature_names)} entries "
                    f"but there are {self.n_classes_} classes"
                )
            return self.feature_names

        raise ValueError("feature_names must be None, a list, or a dict")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        eval_set: tuple = None,
    ) -> "MulticlassBooster":
        """
        Fit the multiclass booster to training data.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray
            Training targets. Can be:
            - 1D array of shape (n_samples,) with class labels (0, 1, 2, ...)
            - 2D array of shape (n_samples, n_classes) one-hot encoded
        sample_weight : np.ndarray of shape (n_samples,), optional
            Sample weights for weighted training.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.

        Returns:
        self : MulticlassBooster
        """

        self._validate_data(X, y)

        X_ = X.astype(np.float32)
        self.n_samples_, self.n_features_in_ = X_.shape

        # handle y format (label vs one-hot)
        if y.ndim == 1:
            self.classes_ = np.unique(y)
            self.y_onehot_ = self._convert_to_onehot(y)
        else:
            self.y_onehot_ = y.astype(np.int32)
            self.classes_ = np.arange(y.shape[1])

        self.n_classes_ = len(self.classes_)

        if eval_set is not None:
            X_val, y_val = eval_set
            y_val = np.asarray(y_val).ravel()
            self._eval_X = X_val.astype(np.float32)
            self._eval_y_onehot = self._convert_to_onehot(y_val, classes=self.classes_)
        else:
            self._eval_X = None
            self._eval_y_onehot = None

        self.sampling_weights_ = sample_weight

        class_weights_list = self._setup_class_weights(self.y_onehot_)
        feature_indices_dict = self._setup_feature_indices()
        feature_names_dict = self._setup_feature_names()

        if self.rounds is None:
            rounds_ = self.n_features_in_ * 15
        else:
            rounds_ = self.rounds

        self.boosters_ = []

        for i, cls in enumerate(self.classes_):
            if self.verbose:
                print(f"Fitting model for class {cls}.")

            feat_idx = feature_indices_dict[cls]
            class_X = X_[:, feat_idx]
            class_y = self.y_onehot_[:, i]

            objective = EntropyObjective(class_weights=class_weights_list[i])

            booster = KernelBooster(
                objective=objective,
                feature_names=feature_names_dict[cls],
                feature_selector=self.feature_selector,
                min_features=self.min_features,
                max_features=self.max_features,
                rounds=rounds_,
                subsample_share=self.subsample_share,
                lambda1=self.lambda1,
                learning_rate=self.learning_rate,
                **self.tree_optimization,
                **self.kernel_optimization,
                early_stopping_rounds=self.early_stopping_rounds,
                stopping_threshold=self.stopping_threshold,
                verbose=self.verbose,
                use_gpu=self.use_gpu,
            )

            class_eval_set = None
            if self._eval_X is not None:
                class_eval_X = self._eval_X[:, feat_idx]
                class_eval_y = self._eval_y_onehot[:, i]
                class_eval_set = (class_eval_X, class_eval_y)

            booster.fit(class_X, class_y, sample_weight=self.sampling_weights_, eval_set=class_eval_set)
            self.boosters_.append(booster)

        self.training_predictions_ = self._softmax_predictions()

        return self

    def _softmax_predictions(self, predictions: np.ndarray = None) -> np.ndarray:
        """Apply softmax to raw predictions."""
        if predictions is None:
            predictions = np.zeros((self.n_samples_, self.n_classes_))
            for i, booster in enumerate(self.boosters_):
                predictions[:, i] = booster.predictions_.ravel()

        predictions = np.clip(predictions, -100, 36)
        exp_pred = np.exp(predictions)
        return exp_pred / exp_pred.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities (softmax normalized).
        """
        if not hasattr(self, 'boosters_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        feature_indices_dict = self._setup_feature_indices()
        predictions = np.empty((X.shape[0], self.n_classes_))

        for i, cls in enumerate(self.classes_):
            feat_idx = feature_indices_dict[cls]
            class_X = X[:, feat_idx]
            # KernelBooster.predict() returns logits
            predictions[:, i] = self.boosters_[i].predict(class_X).ravel()

        return self._softmax_predictions(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on the given test data and labels."""
        if not hasattr(self, 'boosters_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        predictions = self.predict(X)

        # Convert y to labels if one-hot
        if y.ndim == 2:
            y_labels = self._convert_from_onehot(y)
        else:
            y_labels = y

        return np.mean(predictions == y_labels)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            # Multiclass-specific
            'class_weights': self.class_weights,
            'feature_indices': self.feature_indices,
            # Feature-related
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'min_features': self.min_features,
            'max_features': self.max_features,
            # Boosting params
            'rounds': self.rounds,
            'subsample_share': self.subsample_share,
            'lambda1': self.lambda1,
            'learning_rate': self.learning_rate,
            # Tree params
            **self.tree_optimization,
            # Kernel params
            **self.kernel_optimization,
            # Early stopping
            'early_stopping_rounds': self.early_stopping_rounds,
            'stopping_threshold': self.stopping_threshold,
            # General
            'verbose': self.verbose,
            'use_gpu': self.use_gpu,
        }

    def set_params(self, **params) -> "MulticlassBooster":
        """Set parameters for this estimator."""
        valid_keys = set(self.get_params().keys())
        tree_keys = set(self.tree_optimization.keys())
        kernel_keys = set(self.kernel_optimization.keys())

        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter '{key}'")
            if key in tree_keys:
                self.tree_optimization[key] = value
            elif key in kernel_keys:
                self.kernel_optimization[key] = value
            else:
                setattr(self, key, value)

        return self

    def set_gpu(self, value: bool) -> None:
        """Set GPU usage for all fitted estimators."""
        if hasattr(self, 'boosters_'):
            for booster in self.boosters_:
                booster._set_gpu(value)
        self.use_gpu = value

    def set_verbose(self, value: int) -> None:
        """Set verbosity for all fitted estimators."""
        if hasattr(self, 'boosters_'):
            for booster in self.boosters_:
                booster.verbose = value
        self.verbose = value
