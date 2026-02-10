"""Minimalist metrics and other utility functions for convenience.
Mostly reproduces existing scikit-learn functionality (it was educational)."""

import numpy as np

def calculate_confusion_matrix(
    y: np.ndarray, probabilities: np.ndarray, cutoff: float = 0.5
    ) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate confusion matrix and predictive metrics for binary classification."""

    y = y.ravel()
    probabilities = probabilities.ravel()

    predicted_positive = probabilities > cutoff
    predicted_negative = ~predicted_positive
    true_positive = y == 1
    actual_negative = ~true_positive

    # counts
    tp_count = (true_positive & predicted_positive).sum()
    tn_count = (actual_negative & predicted_negative).sum()
    fp_count = (actual_negative & predicted_positive).sum()
    fn_count = (true_positive & predicted_negative).sum()
    true_positive_count = true_positive.sum()
    actual_negative_count = actual_negative.sum()

    # rates
    true_positive_rate = tp_count / true_positive_count if true_positive_count > 0 else 0.0
    false_positive_rate = fp_count / actual_negative_count if actual_negative_count > 0 else 0.0
    true_negative_rate = tn_count / actual_negative_count if actual_negative_count > 0 else 0.0
    false_negative_rate = fn_count / true_positive_count if true_positive_count > 0 else 0.0

    predicted_positive_count = predicted_positive.sum()
    predicted_negative_count = predicted_negative.sum()

    positive_precision = (
        tp_count / predicted_positive_count if predicted_positive_count > 0 else 0
    )
    negative_precision = (
        tn_count / predicted_negative_count if predicted_negative_count > 0 else 0
    )

    confusion_matrix = {
        "True positive rate": true_positive_rate,
        "False positive rate": false_positive_rate,
        "False negative rate": false_negative_rate,
        "True negative rate": true_negative_rate,
    }

    predictive_matrix = {
        "Positive precision": positive_precision,
        "Negative precision": negative_precision,
        "Prevalence": y.mean(),
        "Recall": true_positive_rate, 
    }

    return confusion_matrix, predictive_matrix


def calculate_f1(
    y: np.ndarray, probabilities: np.ndarray, cutoff: float = 0.5
    ) -> float:
    """Calculate F1 score."""
    _, predictive_matrix = calculate_confusion_matrix(y, probabilities, cutoff)
    precision = predictive_matrix["Positive precision"]
    recall = predictive_matrix["Recall"]
    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def calculate_balanced_accuracy(
    y: np.ndarray, probabilities: np.ndarray, cutoff: float = 0.5
    ) -> float:
    """Calculate balanced accuracy (average of TPR and TNR)."""
    confusion_matrix, _ = calculate_confusion_matrix(y, probabilities, cutoff)
    tpr = confusion_matrix["True positive rate"]
    tnr = confusion_matrix["True negative rate"]
    return (tpr + tnr) / 2


def calculate_accuracy(
    y: np.ndarray, probabilities: np.ndarray, cutoff: float = 0.5
    ) -> float:
    """Calculate accuracy (with cutoff for predicting a binary label)."""
    y = y.ravel()
    probabilities = probabilities.ravel()
    predictions = (probabilities > cutoff).astype(int)
    return (predictions == y).mean()


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


def calculate_classification_metrics(
    y: np.ndarray, probabilities: np.ndarray, cutoff: float = 0.5
    ) -> dict[str, float]:
    """Calculate all classification metrics."""
    confusion_matrix, predictive_matrix = calculate_confusion_matrix(
        y, probabilities, cutoff
    )

    precision = predictive_matrix["Positive precision"]
    recall = predictive_matrix["Recall"]
    tpr = confusion_matrix["True positive rate"]
    tnr = confusion_matrix["True negative rate"]

    # F1
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (tpr + tnr) / 2

    y = y.ravel()
    probabilities = probabilities.ravel()
    predictions = (probabilities > cutoff).astype(int)
    accuracy = (predictions == y).mean()

    return {
        **confusion_matrix,
        **predictive_matrix,
        "F1": f1,
        "Balanced accuracy": balanced_accuracy,
        "Accuracy": accuracy,
    }


def calculate_auroc(
    y: np.ndarray,
    predictions: np.ndarray,
    n_cutoffs: int=200
    ) -> tuple[float, np.ndarray]:
    """Calculate the area under receiver-operator curve."""
    y = y.ravel()
    predictions = predictions.ravel()

    cutoffs = np.linspace(predictions.max(), predictions.min(), n_cutoffs)

    # Broadcast: (n_samples, 1) > (1, n_cutoffs) -
    predicted_positive = predictions[:, np.newaxis] > cutoffs[np.newaxis, :]

    true_positive = y == 1
    true_negative = ~true_positive

    tpr = (true_positive[:, np.newaxis] & predicted_positive).sum(
        axis=0
    ) / true_positive.sum()

    fpr = (true_negative[:, np.newaxis] & predicted_positive).sum(
        axis=0
    ) / true_negative.sum()

    # Sort by FPR for curve
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]

    auroc = np.trapz(tpr_sorted, fpr_sorted)

    roc_array = np.column_stack([fpr, tpr])
    return auroc, roc_array


def calculate_auprc(y: np.ndarray, predictions: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate the area under precision-recall curve."""
    y = y.ravel()
    predictions = predictions.ravel()

    n_cutoffs = 200
    cutoffs = np.linspace(predictions.max(), predictions.min(), n_cutoffs)

    predicted_positive = predictions[:, np.newaxis] > cutoffs[np.newaxis, :]
    predicted_positive_count = predicted_positive.sum(axis=0)

    true_positive = y == 1

    tp = (true_positive[:, np.newaxis] & predicted_positive).sum(axis=0)

    precision = np.divide(
        tp, predicted_positive_count,
        out=np.zeros_like(tp, dtype=float),
        where=predicted_positive_count > 0
    )

    true_positive_count = true_positive.sum()
    recall = np.divide(
        tp, true_positive_count,
        out=np.zeros_like(tp, dtype=float),
        where=true_positive_count > 0
    )

    # Sort by recall for curve
    sort_idx = np.argsort(recall)
    recall_sorted = recall[sort_idx]
    precision_sorted = precision[sort_idx]

    auprc = np.trapz(precision_sorted, recall_sorted)

    pr_array = np.column_stack([recall, precision])
    return auprc, pr_array


def calculate_multiclass_confusion_matrices(
    Y: np.ndarray, predictions: np.ndarray, y_names: list = None, cutoff: float = 0.5
    ) -> tuple[dict, dict]:
    """Calculate confusion matrices for multiclass classification."""
    y_dim = Y.shape[1]
    confusion_matrices = {}
    predictive_matrices = {}

    for index in range(y_dim):
        y = Y[:, index]
        y_pred = predictions[:, index]

        confusion_matrix, predictive_matrix = calculate_confusion_matrix(
            y, y_pred, cutoff
        )

        key = y_names[index] if y_names is not None else index
        confusion_matrices[key] = confusion_matrix
        predictive_matrices[key] = predictive_matrix

    return confusion_matrices, predictive_matrices


def calculate_multiclass_aurocs(
    Y: np.ndarray, predictions: np.ndarray, y_names: list = None
    ) -> dict:
    """Calculate AUROC for each class in multiclass classification."""
    y_dim = Y.shape[1]
    aurocs = {}

    for index in range(y_dim):
        auroc, _ = calculate_auroc(Y[:, index], predictions[:, index])
        key = y_names[index] if y_names is not None else index
        aurocs[key] = auroc

    return aurocs


def calculate_multiclass_auprcs(
    Y: np.ndarray, predictions: np.ndarray, y_names: list = None
    ) -> dict:
    """Calculate AUPRC for each class in multiclass classification."""
    y_dim = Y.shape[1]
    auprcs = {}

    for index in range(y_dim):
        auprc, _ = calculate_auprc(Y[:, index], predictions[:, index])
        key = y_names[index] if y_names is not None else index
        auprcs[key] = auprc

    return auprcs


def calculate_multiclass_f1_scores(
    Y: np.ndarray, predictions: np.ndarray, y_names: list = None, cutoff: float = 0.5
    ) -> dict:
    """Calculate F1 score for each class in multiclass classification."""
    y_dim = Y.shape[1]
    f1_scores = {}

    for index in range(y_dim):
        y = Y[:, index]
        y_pred = predictions[:, index]
        key = y_names[index] if y_names is not None else index
        f1_scores[key] = calculate_f1(y, y_pred, cutoff)

    return f1_scores


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

    Converts values to their percentile rank within the training distribution.
    Uses searchsorted for efficient rank computation.

    Parameters
    ----------
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

