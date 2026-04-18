from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .base import HybridDriftDetector
from optimization.classifiers_v2 import ClassifiersV2


class DiscriminativeDriftDetector2019SHAP(HybridDriftDetector):
    """
    Discriminative Drift Detector (D3) with SHAP-based feature importance.
    Extends HybridDriftDetector; warm-up and classifier management are handled
    externally by the ModelOptimizer.

    SHAP computation mode is controlled by ``shap_mode``:
    - ``"first"`` (default): use SHAP values for the first class only.
    - ``"all"``: concatenate SHAP values for all classes.
    """

    def __init__(
        self,
        n_reference_samples: int = 100,
        recent_samples_proportion: float = 0.1,
        threshold: float = 0.7,
        shap_mode: str = "first",
        shap_classifier: str = "lightgbm",
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.data = []
        self.feature_names = None
        self._reference_labels = None
        self._shap_model = None
        self._explainer = None
        self.n_reference_samples = n_reference_samples
        self.recent_samples_proportion = recent_samples_proportion
        self.n_samples = int(n_reference_samples * (1 + recent_samples_proportion))
        self.threshold = threshold
        self.shap_mode = shap_mode
        self.shap_classifier = shap_classifier
        self.kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)

    def build_reference(self, buffer: list, classifiers: ClassifiersV2) -> None:
        """
        Initialise the reference window from the warm-up buffer provided by the ModelOptimizer.
        Uses the pre-fitted LightGBM from ``classifiers`` for SHAP; no model is trained here.
        Expects ``buffer`` to be a list of (features_dict, label) pairs.

        :param buffer: the warm-up buffer collected by the runner
        :param classifiers: the ClassifiersV2 instance whose assisted model is used as the SHAP model
        """
        xs = [
            np.fromiter(x.values(), dtype=float)
            for x, _ in buffer[-self.n_reference_samples :]
        ]
        self.data = xs
        if buffer:
            self.feature_names = list(buffer[0][0].keys())
            self._reference_labels = [y for _, y in buffer[-self.n_reference_samples :]]
        self._shap_model = classifiers.get_model()
        import shap
        self._explainer = shap.TreeExplainer(self._shap_model, np.array(xs))

    def update(self, features: dict, classifiers: ClassifiersV2) -> bool:
        """
        Update the detector with the most recent observation and detect if a drift occurred.

        :param features: the features
        :param classifiers: the Classifiers instance for use in future steps
        :returns: True if a drift occurred else False
        """
        features = np.fromiter(features.values(), dtype=float)
        if len(self.data) != self.n_samples:
            self.data.append(features)
            return False
        else:
            if self._detect_drift(classifiers):
                self.data = self.data[self.n_reference_samples:]
                return True
            else:
                step = int(np.ceil(self.n_reference_samples * self.recent_samples_proportion))
                self.data = self.data[step:]
                return False

    def _detect_drift(self, classifiers: ClassifiersV2) -> bool:
        """
        Detect if a drift occurred.
        Computes raw SHAP values for the recent window, augments the data with them,
        then uses a discriminative classifier to decide if a drift occurred.

        :param classifiers: the Classifiers instance
        :return: True if a drift occurred, else False
        """
        disc_labels = self._get_labels()
        discriminator = LogisticRegression(solver="liblinear", random_state=self.seed)
        raw_shap = self._compute_shap_values(classifiers)
        augmented_data = self._augment_data_with_shap(raw_shap)
        predictions = self._predict(discriminator, augmented_data, disc_labels)
        auc_score = roc_auc_score(disc_labels, predictions)
        return auc_score >= self.threshold

    def _get_labels(self) -> np.array:
        """
        Get discriminative labels for the reference window (0) and recent window (1).

        :return: the labels
        """
        disc_labels = np.zeros(self.n_samples)
        disc_labels[self.n_reference_samples:] = 1
        return disc_labels

    def _train_shap_model(self, X: np.ndarray, y: list):
        """
        Train a surrogate sklearn-compatible model on the reference window for use with
        TreeExplainer.

        :param X: reference feature array, shape (n_reference_samples, n_features)
        :param y: reference class labels
        :return: trained LGBMClassifier or XGBClassifier
        """
        if self.shap_classifier == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=5,
                random_state=self.seed,
                eval_metric="mlogloss",
                verbosity=0,
            )
            model.fit(X, y)
            return model
        # default: lightgbm
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=5, random_state=self.seed, verbosity=-1)
        model.fit(X, y)
        return model

    def _get_raw_shap(self, classifiers: ClassifiersV2):
        """
        Compute raw SHAP values using the cached shap.TreeExplainer (built once per
        reference window in ``build_reference``).

        Always returns a list of arrays ``[(n_samples, n_features)] * n_classes`` so that
        downstream reduction methods are consistent across backends.

        :param classifiers: the Classifiers instance (unused; SHAP uses the surrogate model)
        :return: list of arrays, one per class, each of shape (n_samples, n_features)
        """
        all_data = np.array(self.data)
        raw = self._explainer.shap_values(all_data, check_additivity=False)
        # Normalise to list-of-arrays regardless of SHAP version
        if isinstance(raw, list):
            return raw
        if raw.ndim == 3:
            return [raw[:, :, i] for i in range(raw.shape[2])]
        # binary 2D fallback — wrap so downstream code stays uniform
        return [raw, -raw]

    def _compute_shap_values(self, classifiers: ClassifiersV2) -> np.ndarray:
        """
        Returns reduced SHAP values for the entire data window (reference + recent).

        The shape of the returned array depends on ``self.shap_mode``:
        - ``"first"``: (n_samples, n_features)  — first class only
        - ``"all\"``:   (n_samples, n_features * n_classes)  — all classes concatenated

        :param classifiers: the Classifiers instance
        :return: 2D array of shape (n_samples, n_features[*n_classes])
        """
        shap_values = self._get_raw_shap(classifiers)  # list[(n_samples, n_features)] * n_classes

        if self.shap_mode == "all":
            return np.hstack(shap_values)
        else:  # "first"
            return shap_values[0]

    def _augment_data_with_shap(self, raw_shap: np.ndarray) -> np.ndarray:
        """
        Augment the full data window with raw SHAP values as additional features.
        Both reference and recent windows use their actual SHAP values.

        :param raw_shap: raw SHAP values for all samples, shape (n_samples, n_features)
        :return: augmented data array of shape (n_samples, n_features * 2)
        """
        return np.hstack([np.array(self.data), raw_shap])

    def _predict(self, discriminator, data: np.array, disc_labels: np.array) -> np.array:
        """
        Train and test the discriminator on the given data in a kfold validation scheme.
        If SHAP values from a previous drift are available, features are weighted by their
        mean absolute SHAP importance across all classes before discrimination.

        :param discriminator: the discriminator
        :param data: the data the discriminator is trained on
        :param disc_labels: the discriminative labels of the data
        :return: the predictions
        """
        predictions = np.zeros(self.n_samples)
        for train_index, test_index in self.kfold.split(data, disc_labels):
            discriminator.fit(data[train_index], disc_labels[train_index])
            predictions[test_index] = discriminator.predict_proba(data[test_index])[:, 1]
        return predictions

