import numpy as np

from .d3_shap import DiscriminativeDriftDetector2019SHAP


class DiscriminativeDriftDetector2019SHAPMeanAbs(DiscriminativeDriftDetector2019SHAP):
    """
    Discriminative Drift Detector (D3) with mean-absolute SHAP features.

    Identical to :class:`DiscriminativeDriftDetector2019SHAP` except that
    ``_compute_shap_values`` returns, for each sample, the mean absolute SHAP
    value across all classes instead of the raw values of a single class.
    The returned array therefore has shape ``(n_samples, n_features)`` regardless
    of the number of classes.

    The ``shap_classifier`` parameter is inherited and controls whether
    KernelExplainer (``"kernel"``), LightGBM (``"lightgbm"``), or XGBoost
    (``"xgboost"``) is used for SHAP computation. The ``shap_mode`` parameter
    is unused in this subclass.
    """

    def _compute_shap_values(self) -> np.ndarray:
        """
        Returns mean-absolute SHAP values across all classes for the entire data
        window (reference + recent).

        :return: array of shape (n_samples, n_features)
        """
        shap_values = self._get_raw_shap()  # list[(n_samples, n_features)] * n_classes
        return np.mean(np.abs(np.stack(shap_values, axis=-1)), axis=-1)
