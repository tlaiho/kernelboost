import numpy as np
from .tree import KernelTree
from .feature_selection import FeatureSelector, JMISelector
from .feature_construction import ColumnSelector


class KernelBooster:
    """
    Gradient boosting with local constant (Nadaraya-Watson) regressors as base learners.

    Args:
    objective : Objective
        Loss function (e.g., MSEObjective(), EntropyObjective()).
    feature_selector : FeatureSelector, default=None
        Feature selection strategy. If None and feature_tree_tuple not provided,
        defaults to JMISelector.
    feature_names : list, default=None
        Names for features. Uses indices if None.
    min_features : int, default=1
        Minimum features per round.
    max_features : int, default=None
        Maximum features per round. If None, uses min(10, n_features).
    n_estimators : int, default=None
        Number of boosting rounds. Auto-calculated from n_features if None.
    feature_tree_tuple : tuple, default=None
        Explicit feature subsets and tree type ('kernel' or 'constant') per round.
        Takes priority over feature_selector.
    subsample_share : float, default=0.5
        Training sample share per round.
    lambda1 : float, default=0.0
        L1 regularization for line search.
    learning_rate : float, default=0.5
        Learning rate (shrinkage factor) for line search step sizes. Must be in (0, 1].
    max_depth : int, default=3
        Maximum depth for kernel trees.
    max_sample : int, default=5000
        Maximum samples per kernel leaf (triggers splits).
    min_sample : int, default=500
        Minimum samples for kernel fitting.
    overlap_epsilon : float, default=0.05
        Fraction of feature range to expand training data beyond split
        boundaries in kernel trees.
    kernel_type : str, default='laplace'
        Kernel type: 'gaussian' or 'laplace'.
    precision_method : str, default='pilot-cv'
        Precision selection method: 'search' (LOO-CV), 'pilot-cv' (pilot bounds,
        then LOO-CV), 'pilot-aicc' (pilot bounds, then AICc) or 'silverman'.
    pilot_factor : float, default=3.0
        Multiplier for pilot precision bounds: search range is [p/factor, p*factor].
    search_rounds : int, default=10
        Precision optimization iterations.
    bounds : tuple, default=(0.10, 35.0)
        Precision search bounds.
    initial_precision : float, default=0.0
        Starting precision. 0 means auto.
    sample_share : float, default=1.0
        Share of samples for precision CV.
    n_iter_no_change : int, default=20
        Rounds without improvement before stopping.
    stopping_threshold : float, default=0.0
        Early stopping threshold for mean |rho| if no validation set provided.
    random_state : int, default=None
        Seeds all randomness of the fit: subsampling, bandwidth search, and
        the default selector.
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
        min_features: int = 1,
        max_features: int = None,
        n_estimators: int = None,
        feature_tree_tuple: tuple = None,
        subsample_share: float = 0.5,
        lambda1: float = 0.0,
        learning_rate: float = 0.5,
        max_depth: int = 3,
        max_sample: int = 5000,
        min_sample: int = 500,
        overlap_epsilon: float = 0.05,
        kernel_type: str = "laplace",
        precision_method: str = "pilot-cv",
        pilot_factor: float = 3.0,
        search_rounds: int = 10,
        bounds: tuple = (0.10, 35.0),
        initial_precision: float = 0.0,
        sample_share: float = 1.0,
        n_iter_no_change: int = 20,
        stopping_threshold: float = 0.0,
        random_state: int = None,
        verbose: int = 0,
        use_gpu: bool = False,
    ):
        self.objective = objective
        self.feature_selector = feature_selector

        self.feature_names = feature_names
        self.feature_tree_tuple = feature_tree_tuple
        self.lambda1 = lambda1
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_share = subsample_share
        self.min_features = min_features
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_sample = max_sample
        self.min_sample = min_sample
        self.overlap_epsilon = overlap_epsilon
        self.kernel_type = kernel_type
        self.precision_method = precision_method
        self.pilot_factor = pilot_factor
        self.search_rounds = search_rounds
        self.bounds = bounds
        self.initial_precision = initial_precision
        self.sample_share = sample_share
        self.n_iter_no_change = n_iter_no_change
        self.stopping_threshold = stopping_threshold
        self.random_state = random_state

        self.verbose = verbose
        self.use_gpu = use_gpu

        self.kernel_optimization = {
            "kernel_type": kernel_type,
            "search_rounds": search_rounds,
            "bounds": bounds,
            "initial_precision": initial_precision,
            "sample_share": sample_share,
            "precision_method": precision_method,
            "pilot_factor": pilot_factor,
        }

        self.tree_optimization = {
            "max_sample": max_sample,
            "min_sample": min_sample,
            "max_depth": max_depth,
            "overlap_epsilon": overlap_epsilon,
        }

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        if self.kernel_type not in {"gaussian", "laplace"}:
            raise ValueError(
                f"kernel_type must be 'gaussian' or 'laplace', got '{self.kernel_type}'"
            )
        if self.precision_method not in {"search", "pilot-cv", "pilot-aicc", "silverman"}:
            raise ValueError(
                f"precision_method must be 'search', 'pilot-cv', 'pilot-aicc' or 'silverman', "
                f"got '{self.precision_method}'"
            )
        if self.max_sample <= self.min_sample:
            raise ValueError(
                f"max_sample ({self.max_sample}) must be > min_sample ({self.min_sample})"
            )
        if self.lambda1 < 0:
            raise ValueError(f"lambda1 must be >= 0, got {self.lambda1}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if not 0 < self.subsample_share <= 1:
            raise ValueError(
                f"subsample_share must be in (0, 1], got {self.subsample_share}"
            )
        if self.n_estimators is not None and self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {self.n_estimators}")
        if self.stopping_threshold < 0:
            raise ValueError(
                f"stopping_threshold must be >= 0, got {self.stopping_threshold}"
            )
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"bounds must be (lower, upper) with lower < upper, got {self.bounds}"
            )
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")
        if self.max_features is not None and self.max_features < self.min_features:
            raise ValueError(
                f"max_features ({self.max_features}) must be >= min_features ({self.min_features})"
            )
        if self.n_iter_no_change is not None and self.n_iter_no_change <= 0:
            raise ValueError(
                f"n_iter_no_change must be a positive integer or None, got {self.n_iter_no_change}"
            )
        if not (0.0 <= self.overlap_epsilon < 0.5):
            raise ValueError(
                f"overlap_epsilon must be in [0.0, 0.5), got {self.overlap_epsilon}"
            )
        if self.feature_tree_tuple is not None and not isinstance(
            self.feature_tree_tuple, tuple
        ):
            raise ValueError(
                f"feature_tree_tuple: must be a tuple of (indices, tree_type) tuples"
            )
        if self.random_state is not None and (
            not isinstance(self.random_state, (int, np.integer))
            or self.random_state < 0
        ):
            raise ValueError(
                f"random_state must be a non-negative integer or None, got {self.random_state}"
            )

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D array, got {y.ndim}D")
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"y must have shape (n,) or (n, 1), got {y.shape}")
        if X.shape[0] != y.ravel().shape[0]:
            raise ValueError(
                f"X and y have different number of samples: {X.shape[0]} vs {y.ravel().shape[0]}"
            )
        if X.shape[0] < self.min_sample:
            raise ValueError(
                f"Not enough samples ({X.shape[0]}) for min_sample ({self.min_sample})"
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        if self.objective.is_classifier:
            unique_y = np.unique(y)
            if not np.array_equal(unique_y, np.array([0, 1])):
                raise ValueError(
                    f"For classification, y must contain only 0 and 1, got unique values: {unique_y}"
                )

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
            n_iter_no_change, training will stop if the validation score
            doesn't improve for n_iter_no_change consecutive iterations.

        Returns:
        self : KernelBooster
            Fitted booster.
        """
        self._validate_data(X, y)

        if self.min_features > self.n_features_in_:
            raise ValueError(
                f"min_features ({self.min_features}) exceeds the number of "
                f"features ({self.n_features_in_})"
            )

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

        if self.n_estimators is None:
            self.n_estimators_ = self.n_features_in_ * 15
        else:
            self.n_estimators_ = self.n_estimators

        if self.max_features is None:
            self.max_features_ = min(self.n_features_in_, max(10, self.min_features))
        else:
            self.max_features_ = self.max_features

        # Validate and set feature names
        if self.feature_names is None:
            self.feature_names_ = list(range(self.n_features_in_))
        elif len(self.feature_names) != self.n_features_in_:
            raise ValueError(
                "The number of feature names does not match with the number of features."
            )
        else:
            self.feature_names_ = self.feature_names

        if self.objective.is_classifier:
            self.classes_ = np.unique(y)

        self._training_loop()

        indices = self._last_n_active_tree_indices(1)
        self.last_active_tree_idx_ = indices[0] if indices else None

        return self

    def _training_loop(self) -> None:
        """Execute gradient boosting training loop."""
        self._init_training_state()

        if self.verbose > 0:
            print("Training.")

        for m in range(self.n_estimators_):
            self._train_single_round(m)
            self._log_round(m, self.feature_constructors_[-1])

            # score every round, including the last, before deciding to stop
            if self._eval_X is not None:
                self._update_validation_tracking(m + 1)

            if self._should_stop(m + 1):
                break

        # set best_round_ (only meaningful when eval_set was provided)
        if self._eval_X is not None:
            self.best_round_ = self._best_round
        else:
            self.best_round_ = None

        # stored because RhoOptimizer may overwrite rho_
        self.fit_rho_ = tuple(self.rho_)

        if self.verbose > 0:
            print("Finished training.")

    def _init_training_state(self) -> None:
        """Initialize training state."""
        self._sample_size = int(self.subsample_share * self.n_samples_)

        if self.random_state is not None:
            s_kernel, s_selector = np.random.SeedSequence(self.random_state).spawn(2)
            selector_seed = int(s_selector.generate_state(1)[0])
        else:
            s_kernel = None
            selector_seed = None

        # priority: explicit feature_tree_tuple > feature_selector > default JMI
        if self.feature_tree_tuple is not None:
            self.n_estimators_ = len(self.feature_tree_tuple)
            self.feature_tree_tuple_ = self.feature_tree_tuple
            self._use_selector = False
        else:
            # default to JMISelector if no Selector given
            if self.feature_selector is not None:
                selector = self.feature_selector
            else:
                selector = JMISelector(seed=selector_seed)
                self.feature_selector = selector

            self.n_estimators_ = selector.initialize(
                self.X_,
                self.n_features_in_,
                self.min_features,
                self.max_features_,
                self.n_estimators_,
            )
            self._use_selector = True
            if self.verbose > 0:
                print(f"Feature selector initialized: {self.n_estimators_} rounds")

        self.y_mean_ = np.mean(self.y_)
        self.f_init_ = self.objective.f_init(self.y_)
        self.predictions_ = np.full((self.n_samples_, 1), self.f_init_)

        self.objective_ = [self.objective(self.y_, self.predictions_)]

        self.trees_, self.tree_predictions_ = [], []
        self.rho_, self.feature_constructors_ = [], []
        self.gain_ = []
        self.subsample_indices_ = []
        self.variance_trees_ = None
        self.variance_constructors_ = None
        self.quantile_trees_ = None
        self.quantile_constructors_ = None
        self.loo_gap_ = None
        self.last_precision_ = self.kernel_optimization["initial_precision"]
        if self.random_state is not None:
            self.rseed_ = self.random_state
        else:
            self.rseed_ = np.random.randint(100000, 1234567890, size=1)[0]

        self._rng = np.random.default_rng(self.rseed_)
        if s_kernel is None:
            s_kernel = np.random.SeedSequence(int(self.rseed_)).spawn(2)[0]
        self._kernel_rng = np.random.default_rng(s_kernel)
        self.kernel_optimization["seed"] = self._kernel_rng

        # initialize validation tracking if eval_set provided
        if self._eval_X is not None:
            self._best_val_loss = np.inf
            self._best_round = 0
            self._rounds_no_improvement = 0
            self.val_losses_ = []
            self.eval_predictions = np.full(
                (self._eval_X.shape[0], 1), self.f_init_
            )
        else:
            self.val_losses_ = None
            self.eval_predictions = None

        self.stopped_early_ = False

    def _train_single_round(self, round_idx: int) -> None:
        """Execute one boosting iteration."""
        pseudoresiduals = self.objective.gradient(self.y_, self.predictions_)

        # get feature constructor and leaf type for this round
        if self._use_selector:
            constructor, tree_type = self.feature_selector.get_features(
                round_idx, pseudoresiduals
            )
        else:
            indices, tree_type = self.feature_tree_tuple_[round_idx]
            constructor = ColumnSelector(indices)

        training_features = constructor.transform(self.X_)
        all_data = np.concatenate((pseudoresiduals, training_features), axis=1)
        idx = self._rng.choice(
            self.n_samples_,
            size=self._sample_size,
            p=self.sampling_weights_,
            replace=False,
            shuffle=False,
        )
        self.subsample_indices_.append(idx)
        training_data = all_data[idx]

        self.kernel_optimization.update({"initial_precision": self.last_precision_})
        self.trees_.append(
            KernelTree(
                **self.tree_optimization,
                use_gpu=self.use_gpu,
                **self.kernel_optimization,
                tree_type=tree_type,
            )
        )
        self.trees_[-1].fit(training_data[:, 1:], training_data[:, 0].reshape(-1, 1))

        # store tree predictions for hyperparameter optimization
        self.tree_predictions_.append(self.trees_[-1].predict(training_features))

        if tree_type == "kernel":
            precisions = [
                est.precision_
                for est, is_kern in zip(
                    self.trees_[-1].compiled_.estimators,
                    self.trees_[-1].compiled_.is_kernel,
                )
                if is_kern
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

        self.feature_constructors_.append(constructor)

        # apply results only if rho is non-zero
        if self.rho_[-1] != 0:
            self.predictions_ += self.rho_[-1] * self.tree_predictions_[-1]

        self.objective_.append(self.objective(self.y_, self.predictions_))
        self.gain_.append(
            self.objective_[-2] - self.objective_[-1] if self.rho_[-1] != 0 else 0.0
        )

        # update feature selector with results
        if self._use_selector:
            self.feature_selector.update(constructor, self.gain_[-1])

    def _should_stop(self, m: int) -> bool:
        """Check if training should stop."""
        if m >= self.n_estimators_:
            return True

        # validation-based
        if self._eval_X is not None:
            if self._rounds_no_improvement >= self.n_iter_no_change:
                if self.verbose > 0:
                    print(
                        f"Early stopping: validation loss did not improve for {self.n_iter_no_change} rounds."
                    )
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

    def _update_validation_tracking(self, m: int) -> None:
        """Update validation score tracking after a round."""
        # materialize validation features for the current round's constructor
        val_features = self.feature_constructors_[-1].transform(self._eval_X)

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
            print(
                f"Validation loss: {val_loss:.5f} (best: {self._best_val_loss:.5f} at round {self._best_round})"
            )

    def _check_rho_stopping(self, m: int) -> bool:
        """Check if training should stop based on rho heuristic."""
        if m >= self.n_iter_no_change:
            mean_rho = np.mean(np.abs(self.rho_[-self.n_iter_no_change :]))
            if mean_rho < self.stopping_threshold:
                return True
        return False

    def _log_round(self, m: int, constructor) -> None:
        """Log training progress for one round."""
        if self.verbose <= 0:
            return

        desc = constructor.describe(self.feature_names_)
        n_feat = len(constructor.source_features)
        rho = self.rho_[-1]
        obj = self.objective_[-1]
        gain = self.gain_[-1]
        print(
            f"Round {m + 1}: {n_feat} features {desc} | "
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
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        n = X.shape[0]
        predictions = np.zeros(n)
        n_trees = self.best_round_ if self.best_round_ is not None else len(self.trees_)

        for i in range(n_trees):
            prediction_features = self.feature_constructors_[i].transform(X)
            if self.rho_[i] != 0:
                predictions += (
                    self.rho_[i] * self.trees_[i].predict(prediction_features).ravel()
                )

        predictions += self.f_init_.item()

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
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")
        if not self.objective.is_classifier:
            raise ValueError(
                "predict_proba only available for classification objectives"
            )

        raw_predictions = self.predict(X)
        return self.objective.inverse_link(raw_predictions)

    def staged_predict(self, X: np.ndarray, max_rounds: int = None):
        """
        Yield predictions after each boosting round.

        Note the default differs from predict(): this sweeps ALL rounds, since
        the point is usually to see past the early-stopping cutoff.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.
        max_rounds : int, optional

        Yields:
        np.ndarray of shape (n_samples,)
        """
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        if max_rounds is None:
            max_rounds = len(self.trees_)
        elif not 0 <= max_rounds <= len(self.trees_):
            raise ValueError(
                f"max_rounds must be between 0 and {len(self.trees_)}, got {max_rounds}"
            )

        predictions = np.zeros(X.shape[0])
        yield predictions + self.f_init_.item()

        for i in range(max_rounds):
            if self.rho_[i] != 0:
                prediction_features = self.feature_constructors_[i].transform(X)
                predictions += (
                    self.rho_[i] * self.trees_[i].predict(prediction_features).ravel()
                )
            yield predictions + self.f_init_.item()

    def staged_predict_proba(self, X: np.ndarray, max_rounds: int = None):
        """Yield class probabilities after each boosting round."""
        if not self.objective.is_classifier:
            raise ValueError(
                "staged_predict_proba only available for classification objectives"
            )

        for raw_predictions in self.staged_predict(X, max_rounds=max_rounds):
            yield self.objective.inverse_link(raw_predictions)

    def fit_predict(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ) -> np.ndarray:
        """Fit and return predictions on X."""
        return self.fit(X, y, sample_weight).predict(X)

    def predict_quantiles(
        self,
        X: np.ndarray,
        taus,
        n_trees: int = None,
        aggregation: str = "mean",
        eval_set: tuple = None,
        overfit_correction: bool = True,
        refit: bool = False,
    ) -> np.ndarray:
        """
        EXPERIMENTAL. Predict conditional quantiles. Uses dedicated quantile
        trees fit on evaluation-set residuals (Hall–Wolff–Yao (1999) style):
        predict(x) + Q_tau(residual | x) per tree, aggregated across trees.
        Requires eval_set: fitting quantile trees on training residuals is not
        supported, as boosting leaves no conditional mean signal in them and
        the fits collapse to unconditional residuals.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.
        taus : float or sequence of floats in (0, 1)
            Quantile level(s). A scalar is treated as a length-1 vector; all
            levels are evaluated in a single pass over the quantile trees.
            Query-time argument: any taus are evaluated against the fitted
            trees' stored residual ECDFs, never triggers a refit.
        n_trees : int, default=None
            Number of last active trees whose feature subsets are used
            (upper bound: constant-leaf rounds are skipped). If None, uses
            min(5, number of active trees). Consulted at first fit.
        aggregation : str, default='mean'
            How to combine quantile estimates across trees: 'mean' or 'median'.
            Query-time argument, never triggers a refit.
        eval_set : tuple of (X_val, y_val)
            Held-out data whose residuals the quantile trees are fit on.
            Required whenever trees are (re)fitted; cached-tree queries may
            omit it.
        overfit_correction : bool, default=True
            Ignored: quantile trees are always fit on eval_set residuals.
        refit : bool, default=False
            Quantile trees are fit on the first call and cached. Pass
            refit=True to force refitting with the current arguments.

        Returns:
        q : np.ndarray of shape (len(taus), n_samples)
            Estimated conditional quantile of the response at each level (one
            row per tau). For a scalar tau the shape is (1, n_samples).
        """
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        if self.objective.is_classifier:
            raise NotImplementedError(
                "quantile/interval prediction is not implemented for classifiers."
            )

        if aggregation not in ("mean", "median"):
            raise ValueError(
                f"aggregation must be 'mean' or 'median', got {aggregation}"
            )

        if self.last_active_tree_idx_ is None:
            raise RuntimeError("No active trees found (all rho values are zero)")

        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        if np.any((taus <= 0) | (taus >= 1)):
            raise ValueError(f"taus must be in (0, 1), got {taus}")

        if n_trees is None:
            n_trees = min(5, self.last_active_tree_idx_ + 1)

        if refit or self.quantile_trees_ is None:
            if eval_set is None:
                raise ValueError(
                    "predict_quantiles requires eval_set."
                )
            self._fit_quantile_trees(n_trees, eval_set, overfit_correction)

        predictions = self.predict(X).ravel()
        n_samples = X.shape[0]

        # dimensions: (n_quantile_trees, n_taus, n_samples)
        stacked = np.zeros((len(self.quantile_trees_), taus.shape[0], n_samples))
        for i, (q_tree, constructor) in enumerate(
            zip(self.quantile_trees_, self.quantile_constructors_)
        ):
            bounds = q_tree.predict_quantiles(
                constructor.transform(X), quantiles=tuple(taus)
            )
            stacked[i] = predictions[None, :] + bounds.T

        if len(self.quantile_trees_) == 1:
            return stacked[0]

        if aggregation == "median":
            return np.median(stacked, axis=0)

        return np.mean(stacked, axis=0)

    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
        n_trees: int = None,
        aggregation: str = "mean",
        eval_set: tuple = None,
        overfit_correction: bool = True,
        refit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper over predict_quantiels for a prediction interval.
        Returns (lower, upper)."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        quantiles = self.predict_quantiles(
            X,
            (alpha / 2, 1 - alpha / 2),
            n_trees=n_trees,
            aggregation=aggregation,
            eval_set=eval_set,
            overfit_correction=overfit_correction,
            refit=refit,
        )
        return quantiles[0], quantiles[1]

    def predict_variance(
        self,
        X: np.ndarray,
        n_trees: int = None,
        aggregation: str = "median",
        eval_set: tuple = None,
        overfit_correction: bool = True,
        refit: bool = False,
    ) -> np.ndarray:
        """
        Predict conditional variance using Fan & Yao (1998) double kernel estimation.
        Fits dedicated KernelTrees on squared residuals from the booster. When fitting 
        on training data, raw residuals are suppressed by each observation's own 
        contribution to its prediction; by default this is corrected by subtracting 
        the LOO gap (_training_loo_gap) before squaring. The correction removes 
        own-observation variance leakage, but estimation bias both at the mean 
        and variance stages are still present. 

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.
        n_trees : int, default=None
            Number of last active trees whose feature subsets are used.
            If None, uses min(5, number of active trees).
        aggregation : str, default='median'
            How to combine variances from multiple trees:
            - 'median': median variance (more accurate and robust in benchmarks)
            - 'mean': average variance
            - 'wmean': average weighted by inverse squared LOO error of each
              variance tree on its fitting target
        eval_set : tuple of (X_val, y_val), optional.
            If provided variance estimation will be done on this dataset
            instead of training data.
        overfit_correction : bool, default=True
            Subtract the LOO gap from training predictions before computing
            residuals. Ignored with eval_set.
        refit : bool, default=False
            Variance trees are fit on the first call and cached. Pass 
            refit=True to force refitting with the current arguments.

        Returns:
        variance : np.ndarray of shape (n_samples,)
            Variance estimates for each sample.
        """
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        if aggregation not in ("mean", "median", "wmean"):
            raise ValueError(
                f"aggregation must be 'mean', 'median' or 'wmean', got {aggregation}"
            )

        if self.last_active_tree_idx_ is None:
            raise RuntimeError("No active trees found (all rho values are zero)")

        if n_trees is None:
            n_trees = min(5, self.last_active_tree_idx_ + 1)

        if refit or self.variance_trees_ is None:
            self._fit_variance_trees(n_trees, eval_set, overfit_correction)

        n_samples = X.shape[0]
        variances = np.zeros((len(self.variance_trees_), n_samples))

        for i, (vtree, constructor) in enumerate(
            zip(self.variance_trees_, self.variance_constructors_)
            ):
            variances[i] = vtree.predict(constructor.transform(X)).ravel()

        if len(self.variance_trees_) == 1:
            return np.maximum(variances[0], 0.0)

        if aggregation == "median":
            return np.maximum(np.median(variances, axis=0), 0.0)

        if aggregation == "wmean":
            weights = np.asarray(self.variance_tree_errors_) ** -2
            weights = weights / weights.sum()
            return np.maximum(weights @ variances, 0.0)

        return np.maximum(np.mean(variances, axis=0), 0.0)

    def _fit_variance_trees(
        self, n_trees: int, eval_set: tuple = None, correct: bool = True
    ) -> None:
        """Fit KernelTrees on squared residuals for Fan & Yao variance estimation."""
        squared_residuals = self._final_residuals(eval_set, correct) ** 2

        tree_indices = self._last_n_active_tree_indices(n_trees)

        self.variance_trees_ = []
        self.variance_constructors_ = []
        self.variance_tree_errors_ = []

        for idx in tree_indices:
            if not all(self.trees_[idx].compiled_.is_kernel):
                continue
            constructor = self.feature_constructors_[idx]
            var_tree = KernelTree(
                **self.tree_optimization,
                use_gpu=self.use_gpu,
                **self.kernel_optimization,
            )
            X_fit = eval_set[0] if eval_set else self.X_ # fit for all data or subsample?
            X_t = constructor.transform(X_fit)
            var_tree.fit(X_t, squared_residuals.reshape(-1, 1))
            self.variance_trees_.append(var_tree)
            self.variance_constructors_.append(constructor)
            self.variance_tree_errors_.append(
                self._tree_loo_error(var_tree, X_t, squared_residuals)
            )

        if not self.variance_trees_:
            raise ValueError(
                "Last n trees had constant leaves, no variance estimation possible."
            )

    @staticmethod
    def _tree_loo_error(tree: KernelTree, X: np.ndarray, y: np.ndarray) -> float:
        """LOO-CV MSE of a fitted tree on its own training target, reconstructed
        from self-weights: loo_pred = (pred - s*y) / (1 - s)."""
        pred = tree.predict(X).ravel()
        s = tree.self_weights(X).ravel()
        loo = (pred - s * y) / np.clip(1.0 - s, 1e-6, None)
        return float(np.mean((y - loo) ** 2))

    def _fit_quantile_trees(
        self, n_trees: int, eval_set: tuple = None, correct: bool = True
    ) -> None:
        """Fit KernelTrees on final-model residuals for interval estimation.
        Trees with constant leaves are dropped."""
        residuals = self._final_residuals(eval_set, correct)
        tree_indices = self._last_n_active_tree_indices(n_trees)

        quantile_trees, quantile_constructors = [], []

        for idx in tree_indices:
            if not all(self.trees_[idx].compiled_.is_kernel):
                continue
            constructor = self.feature_constructors_[idx]            
            q_tree = KernelTree(
                **self.tree_optimization,
                use_gpu=self.use_gpu,
                **self.kernel_optimization,
            )
            X_fit = eval_set[0] if eval_set else self.X_  # fit for all data or subsample?
            q_tree.fit(constructor.transform(X_fit), residuals.reshape(-1, 1))
            quantile_trees.append(q_tree)
            quantile_constructors.append(constructor)

        if not quantile_trees:
            raise ValueError(
                "Last n trees had constant leaves, no quantile estimation possible."
            )

        self.quantile_trees_ = quantile_trees
        self.quantile_constructors_ = quantile_constructors

    def _final_residuals(self, eval_set: tuple = None, correct: bool = True) -> np.ndarray:
        """Response-scale (not F) residuals of the final model. When correct is
        true these are LOO residuals to counter residual suppression for training
        data (own-observation effect removed)."""
        if eval_set:
            F = self.predict(eval_set[0]).ravel()
            return self.objective.residuals(eval_set[1].ravel(), F)
        F = self.predict(self.X_).ravel()
        if correct:
            F = F - self._training_loo_gap()  # F^{-i}: correct on the F scale
        return self.objective.residuals(self.y_.ravel(), F)

    def _training_loo_gap(self) -> np.ndarray:
        """Compute the additive LOO gap for training data for F prediction: 
        F^{-i} = F - gap, by removing each observations own contribution to 
        its prediction. Used to reconstruct LOO residuals to counter
        training residual suppression (removes the own-observation effect). 
        """
        if self.loo_gap_ is not None:
            return self.loo_gap_

        gap = self.objective.loo_init_gap(self.y_.ravel(), self.y_mean_)
        preds = np.full((self.n_samples_, 1), self.f_init_, dtype=np.float32)
        n_rounds = self.best_round_ if self.best_round_ is not None else len(self.trees_)

        for t in range(n_rounds):
            if self.rho_[t] != 0:
                r_t = self.objective.gradient(self.y_, preds).ravel()
                idx = self.subsample_indices_[t]
                Xt = self.feature_constructors_[t].transform(self.X_[idx])
                s = self.trees_[t].self_weights(Xt)
                m_hat = self.tree_predictions_[t][idx].ravel()
                gap[idx] += self.rho_[t] * (r_t[idx] - m_hat) * s / (1.0 - s)
            if self.fit_rho_[t] != 0:
                preds += self.fit_rho_[t] * self.tree_predictions_[t]

        self.loo_gap_ = gap
        return self.loo_gap_

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
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")

        predictions = self.predict(X)
        return self.objective.score(y.ravel(), predictions.ravel())

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        # deep kept for compatibility
        return {
            "objective": self.objective,
            "feature_names": self.feature_names,
            "feature_tree_tuple": self.feature_tree_tuple,
            "feature_selector": self.feature_selector,
            "max_depth": self.max_depth,
            "max_sample": self.max_sample,
            "min_sample": self.min_sample,
            "use_gpu": self.use_gpu,
            "kernel_type": self.kernel_type,
            "lambda1": self.lambda1,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "verbose": self.verbose,
            "search_rounds": self.search_rounds,
            "bounds": self.bounds,
            "initial_precision": self.initial_precision,
            "sample_share": self.sample_share,
            "precision_method": self.precision_method,
            "pilot_factor": self.pilot_factor,
            "subsample_share": self.subsample_share,
            "stopping_threshold": self.stopping_threshold,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "n_iter_no_change": self.n_iter_no_change,
            "overlap_epsilon": self.overlap_epsilon,
            "random_state": self.random_state,
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
            "kernel_type": self.kernel_type,
            "search_rounds": self.search_rounds,
            "bounds": self.bounds,
            "initial_precision": self.initial_precision,
            "sample_share": self.sample_share,
            "precision_method": self.precision_method,
            "pilot_factor": self.pilot_factor,
        }

        self.tree_optimization = {
            "max_sample": self.max_sample,
            "min_sample": self.min_sample,
            "max_depth": self.max_depth,
            "overlap_epsilon": self.overlap_epsilon,
        }

        self._validate_params()

        return self

    def _set_gpu(self, value: bool) -> None:
        """Set GPU usage for all fitted kernel estimators."""
        for ktree in self.trees_:
            for est, is_kern in zip(
                ktree.compiled_.estimators, ktree.compiled_.is_kernel
            ):
                if is_kern:
                    est.use_gpu = value

        self.use_gpu = value

        if hasattr(self, "variance_trees_") and self.variance_trees_ is not None:
            for vtree in self.variance_trees_:
                for est, is_kern in zip(
                    vtree.compiled_.estimators, vtree.compiled_.is_kernel
                ):
                    if is_kern:
                        est.use_gpu = value

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on aggregated loss reduction."""
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster not fitted. Call fit() first.")
        importances = np.zeros(self.n_features_in_)
        for constructor, gain in zip(self.feature_constructors_, self.gain_):
            for idx in constructor.source_features:
                importances[idx] += gain
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
