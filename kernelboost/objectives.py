"""Objective functions for gradient boosting.

This module provides objective classes that do:
- Loss calculation
- Gradient (pseudo-residual) and Hessian computation
- Line search step size optimization
"""

from abc import ABC, abstractmethod
import numpy as np
from .utilities import r2_score, accuracy_score


class Objective(ABC):
    """Abstract base class for gradient boosting objectives."""
    
    is_classifier: bool = False

    def __call__(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Alias for loss()."""
        return self.loss(y, predictions)

    @abstractmethod
    def loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate the loss function value.

        Args:
        y : np.ndarray
            True target values.
        predictions : np.ndarray
            Model predictions.

        Returns:
        float
            Loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Calculate pseudo-residuals (negative gradient).

        Args:
        y : np.ndarray
            True target values.
        predictions : np.ndarray
            Model predictions.

        Returns:
        np.ndarray
            Pseudo-residuals with same shape as y.
        """
        pass
    
    def hessian(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Diagonal second derivative of loss w.r.t. predictions.
        Default: 1/n (leads to gradient descent).

        Args:
        y : np.ndarray
            True target values.
        predictions : np.ndarray
            Model predictions.

        Returns:
        np.ndarray
            Hessian diagonal with same shape as y.
        """
        n = len(y)
        # default is just gradient descent --> no Hessian
        return np.full(n, 1.0 / n) 

    @abstractmethod
    def line_search(
        self,
        y: np.ndarray,
        gradient: np.ndarray,
        current_predictions: np.ndarray,
        predictions: np.ndarray,
        lambda1: float,
        learning_rate: float,
        n: int,
    ) -> float:
        """Calculate optimal step size with L1 regularization and shrinkage.

        Args:
        y : np.ndarray
            True target values.
        gradient : np.ndarray
            Pseudo-residuals from self.gradient().
        current_predictions : np.ndarray
            Current tree/kernel predictions.
        predictions : np.ndarray
            Cumulative predictions so far.
        lambda1 : float
            L1 regularization parameter.
        learning_rate : float
            Learning rate (shrinkage factor) applied to step size.
        n : int
            Number of observations.

        Returns:
        float
            Optimal step size (rho).
        """
        pass

    def score(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Score predictions. Override for objective-specific metrics."""
        return -self.loss(y, predictions)


class MSEObjective(Objective):
    """Mean Squared Error objective for regression."""

    def loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        return np.mean(np.square(y.ravel() - predictions.ravel()))

    def gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return y - predictions

    def hessian(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        n = len(y)
        return np.full(n, 2.0 / n)

    def score(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Return R² score."""
        return r2_score(y, predictions)

    def line_search(
        self,
        y: np.ndarray,
        gradient: np.ndarray,
        current_predictions: np.ndarray,
        predictions: np.ndarray,
        lambda1: float,
        learning_rate: float,
        n: int,
    ) -> float:
        """Line search for finding optimal rho value with L1 regularization
        and shrinkage. For MSE exact solution."""
        covariance = np.dot(gradient.T, current_predictions).item() / n
        variance = np.dot(current_predictions.T, current_predictions).item() / n

        raw_rho = covariance / variance

        if abs(raw_rho) > lambda1:
            rho = (abs(raw_rho) - lambda1) * np.sign(raw_rho)
        else:
            return 0.0

        rho = learning_rate * rho

        return rho


class EntropyObjective(Objective):
    """Binary cross-entropy objective for classification."""
    
    is_classifier = True

    def __init__(self, class_weights: np.ndarray = None, gain: float = 0.01):
        """
        Args:
        class_weights : np.ndarray, optional
            Array of [weight_0, weight_1] for class imbalance.
            Defaults to [1.0, 1.0].
        gain : float
            How much the objective should at least decrease by each round.
        """
        if class_weights is None:
            self.class_weights = np.array([1.0, 1.0])
        else:
            self.class_weights = class_weights
            
        self.loss_gain = 1 - gain

    def loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        y = y.ravel()
        p_predictions = self.logits_to_probability(predictions).ravel()
        mask_y1 = y == 1
        weight_0, weight_1 = self.class_weights[0], self.class_weights[1]
        log_sum = np.sum(-weight_1 * np.log(p_predictions[mask_y1])) + np.sum(
            -weight_0 * np.log(1 - p_predictions[~mask_y1])
        )
        return log_sum / len(y)

    def gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        p_predictions = self.logits_to_probability(predictions)
        pseudoresiduals = np.zeros_like(p_predictions)
        weight_0, weight_1 = self.class_weights[0], self.class_weights[1]
        mask_y1 = y == 1
        pseudoresiduals[mask_y1] = weight_1 * (1 - p_predictions[mask_y1])
        pseudoresiduals[~mask_y1] = -weight_0 * p_predictions[~mask_y1]
        return pseudoresiduals
    
    def hessian(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        p = self.logits_to_probability(predictions)
        base_hessian = (p * (1 - p)).ravel()
        weights = np.where(y.ravel() == 1, self.class_weights[1], self.class_weights[0])
        return weights * base_hessian

    def line_search(
        self,
        y: np.ndarray,
        gradient: np.ndarray,
        current_predictions: np.ndarray,
        predictions: np.ndarray,
        lambda1: float,
        learning_rate: float,
        n: int,
    ) -> float:
        """Gauss-Newton line search with L1 soft-thresholding and backtracking."""
        h = self.hessian(y, predictions)
        g = gradient.ravel()
        z = current_predictions.ravel()

        # weighted with hessian  
        cov = np.dot(h * g, z) / n  
        var = np.dot(h * z, z) / n

        raw_rho = cov / var

        if abs(raw_rho) > lambda1:
            rho = (abs(raw_rho) - lambda1) * np.sign(raw_rho)
        else:
            return 0.0

        rho = learning_rate * rho

        # backtracking line search for robustness
        old_objective = self.loss(y, predictions)
        new_predictions = predictions + rho * current_predictions
        new_objective = self.loss(y, new_predictions)

        if new_objective >= self.loss_gain * old_objective:
            for _ in range(15):
                rho *= 0.5
                new_predictions = predictions + rho * current_predictions
                new_objective = self.loss(y, new_predictions)
                if new_objective < self.loss_gain * old_objective:
                    break

        return rho

    def score(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Return accuracy score. predictions are log-odds."""
        return accuracy_score(y, (predictions > 0.0).astype(int))

    def predict_proba(self, log_odds: np.ndarray) -> np.ndarray:
        return self.logits_to_probability(log_odds) 
    
    def logits_to_probability(self, log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds to probabilities with clipping."""
        log_odds = np.clip(log_odds, -100, 36)
        likelihoods = np.exp(log_odds)
        return likelihoods / (1 + likelihoods)


class QuantileObjective(Objective):
    """Quantile loss objective for quantile regression."""

    def __init__(self, quantile: float = 0.5, gain: float = 0.01):
        """
        Args:
        quantile : float
            Target quantile in (0, 1). Default 0.5 (median).
        gain : float
            How much the objective should at least decrease by each round.
        """
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.loss_gain = 1 - gain

    def loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        q = self.quantile
        error = y.ravel() - predictions.ravel()
        return np.mean(np.maximum(q * error, (q - 1) * error))

    def gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        q = self.quantile
        error = y - predictions
        pseudoresiduals = np.zeros_like(error)
        positive_mask = error >= 0
        pseudoresiduals[positive_mask] = q
        pseudoresiduals[~positive_mask] = q - 1
        return pseudoresiduals

    def line_search(
        self,
        y: np.ndarray,
        gradient: np.ndarray,
        current_predictions: np.ndarray,
        predictions: np.ndarray,
        lambda1: float,
        learning_rate: float,
        n: int,
    ) -> float:
        """Line search based on quadratic approximation (same as MSE)
        with L1 regularization and shrinkage. Uses backtracking for robustness."""
        covariance = np.dot(gradient.T, current_predictions).item() / n
        variance = np.dot(current_predictions.T, current_predictions).item() / n

        raw_rho = covariance / variance

        if abs(raw_rho) > lambda1:
            rho = (abs(raw_rho) - lambda1) * np.sign(raw_rho)
        else:
            return 0.0

        rho = learning_rate * rho

        # backtracking line search
        old_objective = self.loss(y, predictions)
        new_predictions = predictions + rho * current_predictions
        new_objective = self.loss(y, new_predictions)

        if new_objective >= self.loss_gain * old_objective:
            for _ in range(15):
                rho *= 0.5
                new_predictions = predictions + rho * current_predictions
                new_objective = self.loss(y, new_predictions)
                if new_objective < self.loss_gain * old_objective:
                    break

        return rho

    def score(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Return negative pinball loss."""
        return -self.loss(y, predictions)

    def hessian(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Pseudo-Hessian for quantile loss (true Hessian is zero)."""
        n = len(y)
        return np.full(n, 1.0 / n)