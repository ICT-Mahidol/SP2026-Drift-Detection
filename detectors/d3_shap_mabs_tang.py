import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .d3_shap import DiscriminativeDriftDetector2019SHAP


class DiscriminativeDriftDetector2019SHAPMeanAbs_permutation(DiscriminativeDriftDetector2019SHAP):
    """
    Discriminative Drift Detector (D3) with mean-absolute SHAP features.

    Identical to :class:`DiscriminativeDriftDetector2019SHAP` except that
    ``_compute_shap_values`` returns, for each sample, the mean absolute SHAP
    value across all classes instead of the raw values of a single class.
    The returned array therefore has shape ``(n_samples, n_features)`` regardless
    of the number of classes.

    Additionally, drift confirmation uses an Adaptive Sequential Permutation Test
    (ASPT) after the AUC threshold check, matching the two-stage logic of d3.py.

    The ``shap_classifier`` parameter is inherited and controls whether
    KernelExplainer (``"kernel"``), LightGBM (``"lightgbm"``), or XGBoost
    (``"xgboost"``) is used for SHAP computation. The ``shap_mode`` parameter
    is unused in this subclass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_detect_drift_time = 0.0
        self.total_aspt_time = 0.0

    # ── SHAP reduction ────────────────────────────────────────────────────────

    def _compute_shap_values(self) -> np.ndarray:
        """
        Returns mean-absolute SHAP values across all classes for the entire data
        window (reference + recent).

        :return: array of shape (n_samples, n_features)
        """
        shap_values = self._get_raw_shap()  # list[(n_samples, n_features)] * n_classes
        return np.mean(np.abs(np.stack(shap_values, axis=-1)), axis=-1)

    # ── Drift detection (Step 2 + Step 3) ────────────────────────────────────

    def _detect_drift(self) -> bool:
        """
        Detect if a drift occurred using Step 2 (AUC threshold check) and
        Step 3 (Adaptive Sequential Permutation Test — ASPT).

        Overrides the parent's plain AUC check; the SHAP-augmented data is
        used throughout (both for the initial AUC and for all permutations).

        :return: True if a drift occurred, else False
        """
        t_start = time.perf_counter()

        # Guard: explainer not yet initialised (build_reference not called yet)
        if self._explainer is None:
            self.total_detect_drift_time += time.perf_counter() - t_start
            return False

        disc_labels    = self._get_labels()
        discriminator  = LogisticRegression(solver="liblinear", random_state=self.seed)
        raw_shap       = self._compute_shap_values()
        augmented_data = self._augment_data_with_shap(raw_shap)
        predictions    = self._predict(discriminator, augmented_data, disc_labels)
        auc_score      = roc_auc_score(disc_labels, predictions)

        # Step 2: AUC threshold gate
        if auc_score < self.threshold:
            self.total_detect_drift_time += time.perf_counter() - t_start
            return False

        # Step 3: ASPT
        t_aspt = time.perf_counter()
        p_value = self._aspt_test(discriminator, augmented_data, disc_labels, auc_score)
        self.total_aspt_time += time.perf_counter() - t_aspt

        self.total_detect_drift_time += time.perf_counter() - t_start
        return p_value < 0.05

    # ── ASPT (copied from d3.py) ──────────────────────────────────────────────

    def _aspt_test(self, discriminator, data: np.ndarray, labels: np.ndarray,
                   auc_actual: float, bmax: int = 100, bmin: int = 10,
                   alpha: float = 0.05) -> float:
        """
        Step 3: Adaptive Sequential Permutation Test (ASPT) inspired by Gandy [30].
        Performs permutation testing with early stopping to confirm drift significance.

        :param discriminator: the discriminator (LogisticRegression instance)
        :param data: SHAP-augmented data used for permutation testing
        :param labels: the original discriminative labels
        :param auc_actual: the actual AUC score from Step 2
        :param bmax: maximum number of permutations (default 100)
        :param bmin: minimum permutations before early stopping (default 10)
        :param alpha: significance level (default 0.05)
        :return: p-value from permutation test
        """
        ci = 0  # count of permutations with AUC >= AUC_actual

        for i in range(1, bmax + 1):
            # Shuffle labels randomly
            labels_perm = np.random.permutation(labels)

            # Retrain and compute AUC on permuted labels
            predictions_perm = self._predict(discriminator, data, labels_perm)
            auc_perm = roc_auc_score(labels_perm, predictions_perm)

            # Count permutations with AUC >= actual AUC
            if auc_perm >= auc_actual:
                ci += 1

            # Early reject H0: ci=0, i>=Bmin, 1/(i+1) < alpha → confirm drift
            if ci == 0 and i >= bmin and 1 / (i + 1) < alpha:
                return 1 / (i + 1)

            # Early accept H0: ci/i > 2*alpha, i>=Bmin → declare no drift
            if ci / i > 2 * alpha and i >= bmin:
                return 1.0

        # Terminal: Bmax reached
        return (ci + 1) / (bmax + 1)
