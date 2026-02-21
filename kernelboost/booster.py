import numpy as np
from .tree import KernelTree
from .feature_selection import FeatureSelector, RandomSelector

class KernelBooster:
    """
    Gradient boosting with local constant (Nadaraya-Watson) regressors as base learners.

    Args:
    objective : Objective
        Loss function (e.g., MSEObjective(), EntropyObjective()).
    feature_selector : FeatureSelector, default=None
        Feature selection strategy. If None and feature_list not provided,
        defaults to RandomSelector.
    feature_names : list, default=None
        Names for features. Uses indices if None.
    feature_list : list, default=None
        Explicit feature subsets per round. Takes priority over feature_selector.
    min_features : int, default=1
        Minimum features per round.
    max_features : int, default=None
        Maximum features per round. If None, uses min(10, n_features).
    rounds : int, default=None
        Boosting rounds. Auto-calculated from n_features if None.
    subsample_share : float, default=0.5
        Training sample share per round.
    lambda1 : float, default=0.0
        L1 regularization for line search.
    learning_rate : float, default=0.5
        Learning rate (shrinkage factor) for step sizes. Must be in (0, 1].
    max_tree_depth : int, default=3
        Maximum depth for kernel trees.
    max_sample : int, default=10000
        Maximum samples per kernel leaf (triggers splits).
    min_sample : int, default=500
        Minimum samples for kernel fitting.
    overlap_epsilon : float, default=0.05
        Fraction of feature range to expand training data beyond split
        boundaries in kernel trees.
    kernel_type : str, default='laplace'
        Kernel type: 'gaussian' or 'laplace'.
    precision_method : str, default='pilot-cv'
        Precision optimization method.
    search_rounds : int, default=20
        Precision optimization iterations.
    bounds : tuple, default=(0.10, 35.0)
        Precision search bounds.
    initial_precision : float, default=0.0
        Starting precision. 0 means auto.
    sample_share : float, default=1.0
        Share of samples for precision CV.
    early_stopping_rounds : int, default=20
        Rounds without improvement before stopping.
    stopping_threshold : float, default=0.0
        Early stopping threshold for mean |rho| if no validation set provided.
    verbose : int, default=0
        Verbosity level.
    use_gpu : bool, default=False
        Use GPU acceleration.
    """

    def __init__(
        self,
        objective,
        feature_selector: FeatureSelector = None,
        feature_names: list = None,
        feature_list: list = None,
        min_features: int = 1,
        max_features: int = None,
        rounds: int = None,
        subsample_share: float = 0.5,
        lambda1: float = 0.0,
        learning_rate: float = 0.5,
        max_tree_depth: int = 3,
        max_sample: int = 10000,
        min_sample: int = 500,
        overlap_epsilon: float = 0.05,
        kernel_type: str = 'laplace',
        precision_method: str = 'pilot-cv',
        search_rounds: int = 20,
        bounds: tuple = (0.10, 35.0),
        initial_precision: float = 0.0,
        sample_share: float = 1.0,
        early_stopping_rounds: int = 20,
        stopping_threshold: float = 0.0,
        verbose: int = 0,
        use_gpu: bool = False,

    ):
        self.objective = objective
        self.feature_selector = feature_selector

        self.feature_names = feature_names
        self.feature_list = feature_list
        self.lambda1 = lambda1
        self.learning_rate = learning_rate
        self.rounds = rounds
        self.subsample_share = subsample_share
        self.min_features = min_features
        self.max_features = max_features
        self.max_tree_depth = max_tree_depth
        self.max_sample = max_sample
        self.min_sample = min_sample
        self.overlap_epsilon = overlap_epsilon
        self.kernel_type = kernel_type
        self.precision_method = precision_method
        self.search_rounds = search_rounds
        self.bounds = bounds
        self.initial_precision = initial_precision
        self.sample_share = sample_share
        self.early_stopping_rounds = early_stopping_rounds
        self.stopping_threshold = stopping_threshold

        self.verbose = verbose
        self.use_gpu = use_gpu

        self.kernel_optimization = {
            'kernel_type': kernel_type,
            'search_rounds': search_rounds,
            'bounds': bounds,
            'initial_precision': initial_precision,
            'sample_share': sample_share,
            'precision_method': precision_method,
        }

        self.tree_optimization = {
            'max_sample': max_sample,
            'min_sample': min_sample,
            'max_depth': max_tree_depth,
            'overlap_epsilon': overlap_epsilon,
        }

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if self.kernel_type not in {'gaussian', 'laplace'}:
            raise ValueError(f"kernel_type must be 'gaussian' or 'laplace', got '{self.kernel_type}'")
        if self.max_sample <= self.min_sample:
            raise ValueError(f"max_sample ({self.max_sample}) must be > min_sample ({self.min_sample})")
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
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"bounds must be (lower, upper) with lower < upper, got {self.bounds}")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")
        if self.max_features is not None and self.max_features < self.min_features:
            raise ValueError(f"max_features ({self.max_features}) must be >= min_features ({self.min_features})")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds <= 0:
            raise ValueError(f"early_stopping_rounds must be a positive integer or None, got {self.early_stopping_rounds}")
        if not (0.0 <= self.overlap_epsilon < 0.5):
            raise ValueError(f"overlap_epsilon must be in [0.0, 0.5), got {self.overlap_epsilon}")

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D array, got {y.ndim}D")
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"y must have shape (n,) or (n, 1), got {y.shape}")
        if X.shape[0] != y.ravel().shape[0]:
            raise ValueError(f"X and y have different number of samples: {X.shape[0]} vs {y.ravel().shape[0]}")
        if X.shape[0] < self.min_sample:
            raise ValueError(f"Not enough samples ({X.shape[0]}) for min_sample ({self.min_sample})")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        if self.objective.is_classifier:
            unique_y = np.unique(y)
            if not np.array_equal(unique_y, np.array([0, 1])):
                raise ValueError(f"For classification, y must contain only 0 and 1, got unique values: {unique_y}")

    def _set_sampling_weights(self, weights: np.ndarray | None) -> None:
        """Set sampling weights for subsampling during training."""
        if weights is None:
            weight = 1 / self.n_samples_
            self.sampling_weights_ = np.repeat(weight, self.n_samples_)
        else:
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("Sampling weights do not sum to one.")
            else:
                self.sampling_weights_ = weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        eval_set: tuple = None,
    ) -> "KernelBooster":
        """
        Fit the KernelBooster to training data.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Training targets.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Sample weights for weighted training.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. When provided together with
            early_stopping_rounds, training will stop if the validation score
            doesn't improve for early_stopping_rounds consecutive iterations.

        Returns:
        self : KernelBooster
            Fitted booster.
        """
        self._validate_data(X, y)

        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise ValueError("eval_set must be a tuple of (X_val, y_val)")
            X_val, y_val = eval_set
            self._validate_data(X_val, y_val)
            if X_val.shape[1] != X.shape[1]:
                raise ValueError(
                    f"eval_set has {X_val.shape[1]} features, "
                    f"but training data has {X.shape[1]} features"
                )
            self._eval_X = np.ascontiguousarray(X_val, dtype=np.float32)
            self._eval_y = np.ascontiguousarray(y_val.reshape(-1, 1), dtype=np.float32)
        else:
            self._eval_X = None
            self._eval_y = None

        self.X_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_ = np.ascontiguousarray(y.reshape(-1, 1), dtype=np.float32)
        self.n_samples_, self.n_features_in_ = self.X_.shape

        self._set_sampling_weights(sample_weight)

        if self.rounds is None:
            self.rounds_ = self.n_features_in_ * 15
        else:
            self.rounds_ = self.rounds

        if self.max_features is None:
            self.max_features_ = min(10, self.n_features_in_)
        else:
            self.max_features_ = self.max_features

        # Validate and set feature names
        if self.feature_names is None:
            self.feature_names_ = list(range(self.n_features_in_))
        elif len(self.feature_names) != self.n_features_in_:
            raise ValueError("The number of feature names does not match with the number of features.")
        else:
            self.feature_names_ = self.feature_names

        if self.objective.is_classifier:
            self.classes_ = np.unique(y)

        self._training_loop()

        feature_tuples = (tuple(sublist) for sublist in self.fitted_features_)
        self.rho_dict_ = dict(zip(feature_tuples, self.rho_))

        self.last_active_tree_idx_ = self._last_active_tree_index()

        return self

    def _training_loop(self) -> None:
        """Execute gradient boosting training loop."""
        self._init_training_state()

        if self.verbose > 0:
            print("Training.")

        for m in range(self.rounds_):
            self._train_single_round(m)
            self._log_round(m, self.fitted_features_[-1])

            if self._should_stop(m + 1):
                break

        # set best_round_ (only meaningful when eval_set was provided)
        if self._eval_X is not None:
            self.best_round_ = self._best_round
        else:
            self.best_round_ = None

        if self.verbose > 0:
            print("Finished training.")
            
    def _init_training_state(self) -> None:
        """Initialize training state."""
        self._sample_size = int(self.subsample_share * self.n_samples_)

        # priority: explicit feature_list > feature_selector > default random
        if self.feature_list is not None:
            self.rounds_ = len(self.feature_list)
            self.feature_list_ = self.feature_list
            self._use_selector = False
        else:
            # default to RandomSelector if no Selector given
            if self.feature_selector is not None:
                selector = self.feature_selector
            else:
                selector = RandomSelector()
                self.feature_selector = selector

            self.rounds_ = selector.initialize(
                self.X_, self.n_features_in_,
                self.min_features, self.max_features_, self.rounds_
            )
            self._use_selector = True
            if self.verbose > 0:
                print(f"Feature selector initialized: {self.rounds_} rounds")

        self.y_mean_ = np.mean(self.y_)
        if self.objective.is_classifier:
            y_mean_clipped = np.clip(self.y_mean_, 1e-10, 1 - 1e-10)
            self.logit_mean_ = np.log(y_mean_clipped / (1 - y_mean_clipped))
            self.predictions_ = np.full((self.n_samples_, 1), self.logit_mean_)
        else:
            self.predictions_ = np.full((self.n_samples_, 1), self.y_mean_)

        self.objective_ = [self.objective(self.y_, self.predictions_)]

        self.trees_, self.tree_predictions_ = [], []
        self.rho_, self.fitted_features_ = [], []
        self.gain_ = []
        self.last_precision_ = self.kernel_optimization['initial_precision']
        self.rseed_ = np.random.randint(100000, 1234567890, size=1)[0]
        self._rng = np.random.default_rng(self.rseed_)

        # initialize validation tracking if eval_set provided
        if self._eval_X is not None:
            self._best_val_loss = np.inf
            self._best_round = 0
            self._rounds_no_improvement = 0
            self.val_losses_ = []
            if self.objective.is_classifier:
                self.eval_predictions = np.full((self._eval_X.shape[0], 1), self.logit_mean_)
            else:
                self.eval_predictions = np.full((self._eval_X.shape[0], 1), self.y_mean_)
        else:
            self.val_losses_ = None
            self.eval_predictions = None

        self.stopped_early_ = False

    def _train_single_round(self, round_idx: int) -> None:
        """Execute one boosting iteration."""
        pseudoresiduals = self.objective.gradient(self.y_, self.predictions_)

        # get features for this round
        if self._use_selector:
            feature_indices = self.feature_selector.get_features(round_idx, pseudoresiduals)
        else:
            feature_indices = self.feature_list_[round_idx]

        training_features = self.X_[:, feature_indices]
        all_data = np.concatenate((pseudoresiduals, training_features), axis=1)
        training_data = self._rng.choice(
            all_data,
            size=self._sample_size,
            p=self.sampling_weights_,
            replace=False,
            shuffle=False
        )

        self.kernel_optimization.update({"initial_precision": self.last_precision_})
        self.trees_.append(
            KernelTree(
                **self.tree_optimization,
                use_gpu=self.use_gpu,
                **self.kernel_optimization,
            )
        )
        self.trees_[-1].fit(training_data[:, 1:], training_data[:, 0].reshape(-1, 1))

        # store tree predictions for hyperparameter optimization
        self.tree_predictions_.append(self.trees_[-1].predict(training_features))

        precisions = [
            est.precision_ for est, is_kern in zip(
                self.trees_[-1].compiled_.estimators, self.trees_[-1].compiled_.is_kernel
            ) if is_kern
        ]
        if precisions:
            self.last_precision_ = np.mean(precisions)

        self.rho_.append(
            self.objective.line_search(
                y=self.y_,
                gradient=pseudoresiduals,
                current_predictions=self.tree_predictions_[-1],
                predictions=self.predictions_,
                lambda1=self.lambda1,
                learning_rate=self.learning_rate,
                n=self.n_samples_,
            )
        )

        self.fitted_features_.append(feature_indices)

        # apply results only if rho is non-zero
        if self.rho_[-1] != 0:
            self.predictions_ += self.rho_[-1] * self.tree_predictions_[-1]

        self.objective_.append(self.objective(self.y_, self.predictions_))
        self.gain_.append(
            self.objective_[-2] - self.objective_[-1] if self.rho_[-1] != 0 else 0.0
        )

        # update feature selector with results
        if self._use_selector:
            self.feature_selector.update(feature_indices, self.gain_[-1])
        
    def _should_stop(self, m: int) -> bool:
        """Check if training should stop."""
        if m >= self.rounds_:
            return True

        # validation-based
        if self._eval_X is not None:
            if self._check_validation_stopping(m):
                if self.verbose > 0:
                    print(f"Early stopping: validation loss did not improve for {self.early_stopping_rounds} rounds.")
                self.stopped_early_ = True
                return True
            return False

        # rho-based, fallback
        if self._check_rho_stopping(m):
            if self.verbose > 0:
                print("Early stopping (rho heuristic).")
            self.stopped_early_ = True
            return True

        return False

    def _check_validation_stopping(self, m: int) -> bool:
        """Update validation score tracking after a round."""
        # get validation features for the current tree's feature indices
        val_features = self._eval_X[:, self.fitted_features_[-1]]

        val_tree_preds = self.trees_[-1].predict(val_features)
        if self.rho_[-1] != 0:
            self.eval_predictions += self.rho_[-1] * val_tree_preds

        val_loss = self.objective(self._eval_y, self.eval_predictions)
        self.val_losses_.append(val_loss)

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_round = m
            self._rounds_no_improvement = 0
        else:
            self._rounds_no_improvement += 1

        if self.verbose > 0:
            print(f"Validation loss: {val_loss:.5f} (best: {self._best_val_loss:.5f} at round {self._best_round + 1})")

        return self._rounds_no_improvement >= self.early_stopping_rounds

    def _check_rho_stopping(self, m: int) -> bool:
        """Check if training should stop based on rho heuristic."""
        if m >= self.early_stopping_rounds:
            mean_rho = np.mean(np.abs(self.rho_[-self.early_stopping_rounds:]))
            if mean_rho < self.stopping_threshold:
                return True
        return False

    def _log_round(self, m: int, feature_indices: list[int]) -> None:
        """Log training progress for one round."""
        if self.verbose <= 0:
            return

        current_features = [self.feature_names_[k] for k in feature_indices]
        rho = self.rho_[-1]
        obj = self.objective_[-1]
        gain = self.gain_[-1]
        print(
            f"Round {m + 1}: {len(current_features)} features {current_features} | "
            f"rho={rho:.4f}, obj={obj:.5f}, gain={gain:.4f}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted booster.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.

        Returns:
        np.ndarray of shape (n_samples,)
            Predicted logits.
        """
        if not hasattr(self, 'trees_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        n = X.shape[0]
        predictions = np.zeros(n)
        n_trees = self.best_round_ if self.best_round_ is not None else len(self.trees_)

        for i in range(n_trees):
            prediction_features = X[:, self.fitted_features_[i]]
            if self.rho_[i] != 0:
                predictions += self.rho_[i] * self.trees_[i].predict(prediction_features).ravel()

        if self.objective.is_classifier:
            predictions += self.logit_mean_.item()
        else:
            predictions += self.y_mean_.item()

        if self.verbose > 0:
            nan_count = np.isnan(predictions).sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values in predictions.")

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for classification.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.

        Returns:
        np.ndarray of shape (n_samples,)
            Predicted probabilities.
        """
        if not hasattr(self, 'trees_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")
        if not self.objective.is_classifier:
            raise ValueError("predict_proba only available for classification objectives")

        raw_predictions = self.predict(X)
        return self.objective.logits_to_probability(raw_predictions)

    def fit_predict(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> np.ndarray:
        """Fit and return predictions on X."""
        return self.fit(X, y, sample_weight).predict(X)
    
    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
        proba: bool = True,
        n_trees: int = None,
        aggregation: str = 'conservative'
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        EXPERIMENTAL. Get prediction intervals using residual quantiles from last active trees.
        If boosting has converged to E(Y|X) and the quantiles estimated by KernelEstimator are
        precise enough, this represents aleatoric uncertainty.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.
        alpha : float, default=0.1
            Significance level. Returns (1-alpha) prediction intervals.
            E.g., alpha=0.1 gives 90% intervals.
        proba : bool, default=True
            For classifiers only. If True, transform logit intervals to probability
            space. If False, return raw logit intervals.
        n_trees : int, default=None
            Number of last active trees to use for interval estimation.
            If None, uses min(5, number of active trees).
            Using multiple trees provides more robust estimates.
        aggregation : str, default='conservative'
            How to combine intervals from multiple trees:
            - 'conservative': min of lowers, max of uppers (widest intervals)
            - 'mean': average of lowers and uppers

        Returns:
        lower : np.ndarray of shape (n_samples,)
            Lower bounds of prediction intervals.
        upper : np.ndarray of shape (n_samples,)
            Upper bounds of prediction intervals.
        """
        if not hasattr(self, 'trees_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if self.last_active_tree_idx_ is None:
            raise RuntimeError("No active trees found (all rho values are zero)")

        if aggregation not in ('conservative', 'mean'):
            raise ValueError(f"aggregation must be 'conservative' or 'mean', got {aggregation}")

        # default: use up to 5 trees
        if n_trees is None:
            n_trees = min(5, self.last_active_tree_idx_ + 1)

        predictions = self.predict(X).ravel()
        quantiles = (alpha / 2, 1 - alpha / 2)
        tree_indices = self._last_n_active_tree_indices(n_trees)

        n_samples = X.shape[0]
        lowers = np.zeros((len(tree_indices), n_samples))
        uppers = np.zeros((len(tree_indices), n_samples))

        for i, idx in enumerate(tree_indices):
            tree = self.trees_[idx]
            features = self.fitted_features_[idx]
            bounds = tree.predict_quantiles(X[:, features], quantiles=quantiles)
            lowers[i] = predictions + bounds[:, 0]
            uppers[i] = predictions + bounds[:, 1]

        if aggregation == 'conservative':
            lower = np.min(lowers, axis=0)
            upper = np.max(uppers, axis=0)
        else:
            lower = np.mean(lowers, axis=0)
            upper = np.mean(uppers, axis=0)

        if self.objective.is_classifier and proba:
            lower = self.objective.logits_to_probability(lower)
            upper = self.objective.logits_to_probability(upper)

        return lower, upper

    def predict_variance(
        self,
        X: np.ndarray,
        n_trees: int = None,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        EXPERIMENTAL. Predict conditional variance estimates using last active trees.
        If boosting has converged to E(Y|X) and the conditional variance estimation in
        KernelEstimator works, this represents aleatoric uncertainty.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.
        n_trees : int, default=None
            Number of last active trees to use for variance estimation.
            If None, uses min(5, number of active trees).
        aggregation : str, default='mean'
            How to combine variances from multiple trees:
            - 'max': maximum variance (most conservative)
            - 'mean': average variance

        Returns:
        variance : np.ndarray of shape (n_samples,)
            Variance estimates for each sample.
        """
        if not hasattr(self, 'trees_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        if aggregation not in ('max', 'mean'):
            raise ValueError(f"aggregation must be 'max' or 'mean', got {aggregation}")

        if self.last_active_tree_idx_ is None:
            raise RuntimeError("No active trees found (all rho values are zero)")

        # default: up to 5 trees
        if n_trees is None:
            n_trees = min(5, self.last_active_tree_idx_ + 1)

        tree_indices = self._last_n_active_tree_indices(n_trees)

        if len(tree_indices) == 1:
            tree = self.trees_[tree_indices[0]]
            features = self.fitted_features_[tree_indices[0]]
            _, variance = tree._predict_with_variance(X[:, features])
            return variance.ravel()

        n_samples = X.shape[0]
        variances = np.zeros((len(tree_indices), n_samples))

        for i, idx in enumerate(tree_indices):
            tree = self.trees_[idx]
            features = self.fitted_features_[idx]
            _, var = tree._predict_with_variance(X[:, features])
            variances[i] = var.ravel()

        if aggregation == 'max':
            return np.max(variances, axis=0)
        else:
            return np.mean(variances, axis=0)

    def _last_n_active_tree_indices(self, n: int) -> list[int]:
        """Find indices of last n trees with non-zero rho."""
        n_trees = self.best_round_ if self.best_round_ is not None else len(self.trees_)
        indices = []
        for i in range(n_trees - 1, -1, -1):
            if self.rho_[i] != 0:
                indices.append(i)
                if len(indices) >= n:
                    break
        return indices

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return default score for the objective."""
        if not hasattr(self, 'trees_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        predictions = self.predict(X)
        return self.objective.score(y.ravel(), predictions.ravel())

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        # deep kept for compatibility
        return {
            'objective': self.objective,
            'feature_names': self.feature_names,
            'feature_list': self.feature_list,
            'feature_selector': self.feature_selector,
            'max_tree_depth': self.max_tree_depth,
            'max_sample': self.max_sample,
            'min_sample': self.min_sample,
            'use_gpu': self.use_gpu,
            'kernel_type': self.kernel_type,
            'lambda1': self.lambda1,
            'learning_rate': self.learning_rate,
            'rounds': self.rounds,
            'verbose': self.verbose,
            'search_rounds': self.search_rounds,
            'bounds': self.bounds,
            'initial_precision': self.initial_precision,
            'sample_share': self.sample_share,
            'precision_method': self.precision_method,
            'subsample_share': self.subsample_share,
            'stopping_threshold': self.stopping_threshold,
            'min_features': self.min_features,
            'max_features': self.max_features,
            'early_stopping_rounds': self.early_stopping_rounds,
            'overlap_epsilon': self.overlap_epsilon,
        }

    def set_params(self, **params) -> "KernelBooster":
        """Set parameters for this estimator."""
        valid_keys = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter '{key}'")
            else:
                setattr(self, key, value)

        self.kernel_optimization = {
            'kernel_type': self.kernel_type,
            'search_rounds': self.search_rounds,
            'bounds': self.bounds,
            'initial_precision': self.initial_precision,
            'sample_share': self.sample_share,
            'precision_method': self.precision_method,
        }

        self.tree_optimization = {
            'max_sample': self.max_sample,
            'min_sample': self.min_sample,
            'max_depth': self.max_tree_depth,
            'overlap_epsilon': self.overlap_epsilon,
        }

        self._validate_params()

        return self

    def _set_gpu(self, value: bool) -> None:
        """Set GPU usage for all fitted kernel estimators."""
        for ktree in self.trees_:
            for est, is_kern in zip(ktree.compiled_.estimators, ktree.compiled_.is_kernel):
                if is_kern:
                    est.use_gpu = value

        self.use_gpu = value

    def _last_active_tree_index(self) -> int | None:
        """Find index of last tree with non-zero rho."""
        indices = self._last_n_active_tree_indices(1)
        return indices[0] if indices else None

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on aggregated |rho| values."""
        if not hasattr(self, 'rho_dict_'):
            raise RuntimeError("Booster not fitted. Call fit() first.")
        importances = np.zeros(self.n_features_in_)
        for feature_indices, rho in self.rho_dict_.items():
            for idx in feature_indices:
                importances[idx] += abs(rho)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
