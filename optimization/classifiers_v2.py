import numpy as np
from lightgbm import LGBMClassifier


class ClassifiersV2:
    """
    ClassifiersV2 provides a LightGBM-based interface for concept drift detection.
    Unlike Classifiers, training is done in batch (offline) rather than online.

    Two models are maintained:
    - base_lgbm: fitted once on the initial window and never reset; represents the baseline concept.
    - assisted_lgbm: refitted from scratch on each new window after a drift; reflects the current concept.

    The purpose of this class is to keep all model fitting external to the detector so that
    detectors (e.g. D3-SHAP) can consume a pre-trained model instead of training their own.
    """

    def __init__(self, n_estimators: int = 100, seed: int = None):
        """
        Initialise ClassifiersV2.

        :param n_estimators: number of boosting rounds for LightGBM
        :param seed: random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.seed = seed
        self.base_lgbm = None
        self.assisted_lgbm = None
        self._base_fitted = False
        self._assisted_fitted = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X, y, nonadaptive: bool = True) -> None:
        """
        Batch-fit the classifiers on the provided data.

        If *nonadaptive* is True and the base model has not yet been fitted, the base model is
        also trained on this batch (it is trained at most once). The assisted model is always
        (re-)trained.

        :param X: list of feature dicts or a 2-D array of shape (n_samples, n_features)
        :param y: list/array of labels
        :param nonadaptive: whether the base model should also be (initially) trained
        """
        X_arr = self._to_array(X)
        y_arr = np.array(y)

        if nonadaptive and not self._base_fitted:
            self.base_lgbm = LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                verbosity=-1,
            )
            self.base_lgbm.fit(X_arr, y_arr)
            self._base_fitted = True

        self.assisted_lgbm = LGBMClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            verbosity=-1,
        )
        self.assisted_lgbm.fit(X_arr, y_arr)
        self._assisted_fitted = True

    def batch_fit(self, buffer: list, nonadaptive: bool = True) -> None:
        """
        Convenience wrapper: extract (X, y) from a list of (features, label) pairs and call fit.

        :param buffer: list of (features_dict, label) tuples
        :param nonadaptive: passed through to fit
        """
        X = [x for x, _ in buffer]
        y = [label for _, label in buffer]
        self.fit(X, y, nonadaptive=nonadaptive)

    def predict(self, x) -> tuple:
        """
        Predict the label for a single sample using both classifiers.

        Returns a 4-tuple for backward compatibility with get_metrics:
        ``(base_pred, base_pred, assisted_pred, assisted_pred)``

        :param x: feature dict or 1-D array
        :return: 4-tuple of predicted labels
        """
        x_arr = self._to_array([x])
        base_pred = self.base_lgbm.predict(x_arr)[0] if self._base_fitted else None
        assisted_pred = self.assisted_lgbm.predict(x_arr)[0] if self._assisted_fitted else None
        return (base_pred, base_pred, assisted_pred, assisted_pred)

    def reset(self) -> None:
        """
        Reset the assisted classifier so that it will be refitted on the next window.
        The base classifier is never reset.
        """
        self.assisted_lgbm = None
        self._assisted_fitted = False

    def get_model(self) -> LGBMClassifier:
        """
        Return the currently fitted assisted LightGBM model.
        Intended for use by detectors that require an external tree model (e.g. for SHAP).

        :return: the fitted LGBMClassifier, or None if not yet fitted
        """
        return self.assisted_lgbm

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_array(self, X) -> np.ndarray:
        """Convert a list of feature dicts or array-like to a 2-D numpy array."""
        if len(X) == 0:
            return np.array([])
        if isinstance(X[0], dict):
            return np.array([list(x.values()) for x in X])
        return np.array(X)
