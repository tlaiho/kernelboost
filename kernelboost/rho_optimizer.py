import numpy as np


class RhoOptimizer:
    """Post-hoc optimization of rho weights for KernelBooster.

    This class optimizes the rho weights (step sizes) of a fitted KernelBooster
    using validation data. It can also be used to back out lambda1/learning_rate
    hyperparameters that would have produced those rho values during training.

    Args:
    booster : KernelBooster
        A fitted KernelBooster instance.
    lambda2 : float, default=0.05
        Ridge penalty on rho, relative to the average data curvature.
        lambda2=1 penalises as strongly as the data curves.
    """

    def __init__(self, booster: "KernelBooster", lambda2: float = 0.05):
        if not hasattr(booster, 'trees_'):
            raise ValueError("Booster must be fitted before optimization")

        self.booster_ = booster
        self.lambda2_ = lambda2
        # multiplier to rescale lambda2 to tolerate regression's flatter loss surface
        self.lambda2_multiplier_ = 1.0 if booster.objective.is_classifier else 1e-2

        self.rho_ = None
        self.lambda1_ = None
        self.learning_rate_ = None
        self._Z_train = None

    def optimize_rhos(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        lambda2: float = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        step_size: float = 0.25
    ) -> np.ndarray:
        """Optimize rho weights using Newton-Raphson method.

        Args:
        Z : np.ndarray of shape (n_samples, M)
            Design matrix from build_design_matrix().
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.
        lambda2 : float, optional
            Ridge penalty relative to the average data curvature. Uses
            self.lambda2_ if None.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance for maximum absolute change in rho.
        step_size : float, default=1.0
            Damping factor for Newton steps to ensure stability.

        Returns:
        np.ndarray of shape (M,)
            Optimized rho vector.
        """
        if lambda2 is None:
            lambda2 = self.lambda2_

        y = y.ravel()
        n = len(y)
        M = Z.shape[1]

        objective = self.booster_.objective
        base_pred = self._base_prediction()
        rho = np.array(self.booster_.rho_, dtype=np.float64)

        # gradient() and hessian() give derivatives of grad_loss_scale * n * loss()
        scale = objective.grad_loss_scale
        pred = base_pred + Z @ rho

        # lambda2 is scaled to the average curvature
        hessian0 = objective.hessian(y, pred).ravel()
        penalty = (lambda2 * self.lambda2_multiplier_
                   * np.sum(hessian0[:, None] * Z**2) / (n * M))

        current_loss = scale * objective(y, pred) + 0.5 * penalty * np.dot(rho, rho)

        for _ in range(max_iter):
            # gradient returns pseudo-residuals (negative gradient)
            g = objective.gradient(y, pred).ravel()
            # hessian is on the same scale as gradient
            h = objective.hessian(y, pred).ravel()

            # gradient: ∂L/∂ρ = -Z.T @ g / n
            grad_rho = -(Z.T @ g) / n + penalty * rho

            # Hessian: H = Z.T @ diag(h) @ Z / n
            H = (Z.T @ (h[:, None] * Z)) / n + penalty * np.eye(M)

            # Newton with damping
            delta = step_size * np.linalg.solve(H, grad_rho)
            rho_new = rho - delta
            pred_new = base_pred + Z @ rho_new
            new_loss = scale * objective(y, pred_new) + 0.5 * penalty * np.dot(rho_new, rho_new)

            # backtracking line search for robustness
            alpha = 1.0
            while new_loss > current_loss and alpha > 1e-8:
                alpha *= 0.5
                rho_new = rho - alpha * delta
                pred_new = base_pred + Z @ rho_new
                new_loss = scale * objective(y, pred_new) + 0.5 * penalty * np.dot(rho_new, rho_new)

            # backtracking exhausted without improvement: keep the current rho
            if new_loss > current_loss:
                break

            rho = rho_new
            pred = pred_new
            current_loss = new_loss

            if np.max(np.abs(alpha * delta)) < tol:
                break

        return rho

    def _base_prediction(self) -> float:
        """Return the base prediction (logit for classifiers, mean for regression)."""
        return self.booster_.f_init_.item()
 
    def _compute_covariance_variance(
        self,
        rho_vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute covariance and variance for each tree from training data.
        Needed for backing out lambdas."""
        booster = self.booster_
        n = booster.n_samples_
        M = len(booster.trees_)

        objective = booster.objective
        y = booster.y_.ravel()
        base_pred = self._base_prediction()

        if self._Z_train is None:
            self._Z_train = self._build_design_matrix(booster.X_)

        covariances = np.zeros(M)
        variances = np.zeros(M)

        pred = np.full(n, base_pred)

        for i in range(M):
            grad = objective.gradient(y, pred).ravel()
            h = objective.hessian(y, pred).ravel()
            tree_pred = self._Z_train[:, i]

            covariances[i] = np.dot(grad, tree_pred) / n
            variances[i] = np.dot(h * tree_pred, tree_pred) / n

            pred = pred + rho_vector[i] * tree_pred

        return covariances, variances

    def _build_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build design matrix Z where column i contains tree i's predictions."""
        # Use cached predictions if this is training data
        if X is self.booster_.X_:
            return np.column_stack(self.booster_.tree_predictions_)

        n_samples = X.shape[0]
        n_trees = len(self.booster_.trees_)
        Z = np.zeros((n_samples, n_trees))

        for i in range(n_trees):
            constructor = self.booster_.feature_constructors_[i]
            tree_features = constructor.transform(X)
            Z[:, i] = self.booster_.trees_[i].predict(tree_features).ravel()

        return Z

    def _inverse_line_search(
        self, rho_i: float, cov_i: float, var_i: float, lambda1: float
    ) -> float | None:
        """Compute implied learning_rate for a single iteration."""
        if abs(rho_i) < 1e-10 or abs(var_i) < 1e-10:
            return None

        raw_rho_i = abs(cov_i / var_i)
        if raw_rho_i < 1e-10:
            return None  # optimal step is zero

        denom = raw_rho_i - lambda1
        if denom < 1e-10:
            return None  # lambda1 too large for this rho

        return abs(rho_i) / denom

    def find_hyperparameters(self, initial_lambda1: float = 0.05) -> tuple[float, float]:
        """Find lambda1 and learning_rate from optimized rho values.

        Uses a hierarchical approach:
        1. Fix lambda1, solve for learning_rate using median of per-tree values
        2. If learning_rate > 1, decrease lambda1 by 0.5x and retry
        3. Clamp learning_rate to (0, 1]

        Args:
        initial_lambda1 : float, default=0.05
            Starting value for lambda1.

        Returns:
        lambda1 : float
            Backed-out L1 regularization parameter.
        learning_rate : float
            Backed-out learning rate.
        """
        if self.rho_ is None:
            raise ValueError("Must call fit() before find_hyperparameters()")

        covariances, variances = self._compute_covariance_variance(self.rho_)

        lambda1 = initial_lambda1
        # fallback, used if no pass yields a usable learning_rate
        learning_rate = self.booster_.learning_rate

        # outer loop: decrease lambda1
        for _ in range(20):
            lr_values = []
            # inner loop: find learning_rates
            for i in range(len(self.rho_)):
                lr = self._inverse_line_search(
                    self.rho_[i], covariances[i], variances[i], lambda1
                )
                if lr is not None:
                    lr_values.append(lr)

            if lr_values:
                learning_rate = np.median(lr_values)
                if 0 < learning_rate <= 1:
                    break

            # reduce lambda1 if no suitable learning_rate found
            lambda1 *= 0.5

        learning_rate = min(1.0, max(0.01, learning_rate))

        self.lambda1_ = lambda1
        self.learning_rate_ = learning_rate

        return self.lambda1_, self.learning_rate_

    def optimize_lambda_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_candidates: np.ndarray,
        k: int = 5
    ) -> float:
        """Select best lambda2 via k-fold cross-validation.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.
        lambda_candidates : np.ndarray
            Array of lambda2 values to try.
        k : int, default=5
            Number of CV folds.

        Returns:
        float
            Best lambda2 value.
        """
        y = y.ravel()
        n = len(y)

        Z = self._build_design_matrix(X)
        
        indices = np.arange(n)
        np.random.default_rng(self.booster_.rseed_).shuffle(indices)
        fold_size = n // k

        objective = self.booster_.objective
        base_pred = self._base_prediction()

        best_lambda = lambda_candidates[0]
        best_loss = np.inf

        for lambda2 in lambda_candidates:
            fold_losses = []

            for fold in range(k):
                start = fold * fold_size
                end = start + fold_size if fold < k - 1 else n
                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                Z_train = Z[train_idx]
                y_train = y[train_idx]

                rho_fold = self.optimize_rhos(Z_train, y_train, lambda2=lambda2)

                Z_val = Z[val_idx]
                y_val = y[val_idx]
                pred_val = base_pred + Z_val @ rho_fold

                loss = objective(y_val, pred_val)
                fold_losses.append(loss)

            mean_loss = np.mean(fold_losses)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_lambda = lambda2

        return best_lambda

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "RhoOptimizer":
        """Fit the optimizer using validation data.

        Args:
        X_val : np.ndarray of shape (n_samples, n_features)
            Validation features.
        y_val : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Validation targets.

        Returns:
        self : RhoOptimizer
            Fitted optimizer.
        """

        Z = self._build_design_matrix(X_val)
        self.rho_ = self.optimize_rhos(Z, y_val)

        return self

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        lambda_candidates: np.ndarray = None
    ) -> "RhoOptimizer":
        """Fit using k-fold CV to select lambda2, then back out lambdas.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.
        k : int, default=5
            Number of CV folds.
        lambda_candidates : np.ndarray, optional
            Lambda values to try. Default is logspace(-3, 1, 10).

        Returns:
        self : RhoOptimizer
            Fitted optimizer.
        """
        if lambda_candidates is None:
            lambda_candidates = np.logspace(-3, 1, 10)

        best_lambda = self.optimize_lambda_cv(X, y, lambda_candidates, k=k)
        self.lambda2_ = best_lambda

        Z = self._build_design_matrix(X)
        self.rho_ = self.optimize_rhos(Z, y, lambda2=best_lambda)

        self.find_hyperparameters()

        return self

    def update_booster(self) -> "KernelBooster":
        """Update the booster with optimized rho values and backed-out lambdas.

        Returns:
        KernelBooster
            The updated booster (same instance, modified in place).
        """
        if self.rho_ is None:
            raise ValueError("Must call fit() before update_booster()")

        self.booster_.rho_ = list(self.rho_)

        # every tree is re-weighted, so the training-time early-stopping cut no
        # longer applies; zero weights, if any, drop out of predict()
        self.booster_.best_round_ = len(self.booster_.trees_)

        # rho changes invalidate the cached LOO gap and second-stage trees
        self.booster_.loo_gap_ = None
        self.booster_.variance_trees_ = None
        self.booster_.variance_constructors_ = None
        self.booster_.quantile_trees_ = None
        self.booster_.quantile_constructors_ = None

        if self.lambda1_ is not None:
            self.booster_.lambda1 = self.lambda1_
        if self.learning_rate_ is not None:
            self.booster_.learning_rate = self.learning_rate_

        return self.booster_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using optimized rho values.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Features to predict on.

        Returns:
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        if self.rho_ is None:
            raise ValueError("Must call fit() before predict()")

        Z = self._build_design_matrix(X)
        base_pred = self._base_prediction()

        return base_pred + Z @ self.rho_

    def refit_trees(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_subsampling: bool = True
    ) -> "RhoOptimizer":
        """Re-fit trees using optimized rhos for pseudoresidual computation.
        Fixes the mismatch between optimized rhos and the original pseudo-
        residuals.

        Args:
        X : np.ndarray of shape (n_samples, n_features)
            Training features (typically the same data used to train the booster).
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Training targets.
        use_subsampling : bool, default=True
            Whether to use subsampling during tree refitting (matches original
            training behavior).

        Returns:
        self : RhoOptimizer
            Updated optimizer with refitted trees.
        """

        if self.rho_ is None:
            raise ValueError("Must call fit() before refit_trees()")

        booster = self.booster_
        n_samples = X.shape[0]
        n_features = X.shape[1]

        if n_features != booster.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but booster was trained with "
                f"{booster.n_features_in_} features"
            )

        y = y.reshape(-1, 1)
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {y.shape[0]} samples"
            )

        objective = booster.objective
        n_trees = len(booster.trees_)

        # initialize predictions using base prediction
        predictions = np.full((n_samples, 1), self._base_prediction())

        if use_subsampling:
            sample_size = int(booster.subsample_share * n_samples)
            rng = np.random.default_rng(booster.rseed_)
            weights = booster.sampling_weights_
        else:
            sample_size = n_samples
            rng = None
            weights = None

        # refit each tree
        for i in range(n_trees):
            pseudoresiduals = objective.gradient(y, predictions)
            constructor = booster.feature_constructors_[i]
            training_features = constructor.transform(X)

            all_data = np.concatenate((pseudoresiduals, training_features), axis=1)

            if use_subsampling:
                idx = rng.choice(
                    n_samples,
                    size=sample_size,
                    p=weights,
                    replace=False,
                    shuffle=False
                )
            else:
                idx = np.arange(n_samples)

            # recorded so each tree is paired with the rows it saw
            booster.subsample_indices_[i] = idx
            training_data = all_data[idx]

            booster.trees_[i].fit(training_data[:, 1:], training_data[:, 0])
            booster.tree_predictions_[i] = booster.trees_[i].predict(training_features)

            if self.rho_[i] != 0:
                predictions += self.rho_[i] * booster.tree_predictions_[i]

        booster.predictions_ = predictions

        # clean slate
        self._Z_train = None
        booster.loo_gap_ = None
        booster.variance_trees_ = None
        booster.variance_constructors_ = None
        booster.quantile_trees_ = None
        booster.quantile_constructors_ = None

        return self

    def fit_with_refit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        k: int = 3,
        use_subsampling: bool = True
    ) -> "RhoOptimizer":
        """Fit with iterative tree refitting.

        Args:
        X_train : np.ndarray of shape (n_train, n_features)
            Training features for tree refitting.
        y_train : np.ndarray of shape (n_train,) or (n_train, 1)
            Training targets.
        X_val : np.ndarray of shape (n_val, n_features)
            Validation features for rho optimization.
        y_val : np.ndarray of shape (n_val,) or (n_val, 1)
            Validation targets.
        k : int, default=3
            Number of refit iterations.
        use_subsampling : bool, default=True
            Whether to use subsampling during tree refitting.

        Returns:
        self : RhoOptimizer
            Fitted optimizer.
        """
        for _ in range(k):
            self.fit(X_val, y_val)
            self.refit_trees(X_train, y_train, use_subsampling=use_subsampling)
        
        # final rhos
        self.fit(X_val, y_val)

        return self
