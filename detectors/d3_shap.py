from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .base import HybridDriftDetector
from optimization.classifiers_v2 import ClassifiersV2

import time
from functools import wraps

# def timer(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()  # High-resolution timer
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         print(f"Executed {func.__name__} in {end_time - start_time:.4f} seconds")
#         return result
#     return wrapper



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
        self._explainer = None
        self._raw_shap_cache = None
        self.n_reference_samples = n_reference_samples
        self.recent_samples_proportion = recent_samples_proportion
        self.n_samples = int(n_reference_samples * (1 + recent_samples_proportion))
        self.threshold = threshold
        self.shap_mode = shap_mode
        self.shap_classifier = shap_classifier
        self.kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
    #@timer
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
        import shap
        self._explainer = shap.TreeExplainer(classifiers.get_model())
        self._raw_shap_cache = None

    def _get_slide_step(self) -> int:
        return int(np.ceil(self.n_reference_samples * self.recent_samples_proportion))

    def _append_shap_placeholder(self, feature_row: np.ndarray) -> None:
        if self._raw_shap_cache is None:
            return
        placeholder = np.full((1, feature_row.shape[0]), np.nan)
        self._raw_shap_cache = [
            np.vstack([class_cache, placeholder]) for class_cache in self._raw_shap_cache
        ]

    def _slide_shap_cache(self, cut: int) -> None:
        if self._raw_shap_cache is None:
            return
        self._raw_shap_cache = [class_cache[cut:] for class_cache in self._raw_shap_cache]

    def update(self, features: dict) -> bool:
        """
        Update the detector with the most recent observation and detect if a drift occurred.

        :param features: the features
        :returns: True if a drift occurred else False
        """
        features = np.fromiter(features.values(), dtype=float)
        if len(self.data) != self.n_samples:
            self.data.append(features)
            self._append_shap_placeholder(features)
            return False
        else:
            if self._detect_drift():
                self.data = self.data[self.n_reference_samples:]
                self._slide_shap_cache(self.n_reference_samples)
                return True
            else:
                step = self._get_slide_step()
                self.data = self.data[step:]
                self._slide_shap_cache(step)
                return False

    def _detect_drift(self) -> bool:
        """
        Detect if a drift occurred.
        Computes raw SHAP values for the recent window, augments the data with them,
        then uses a discriminative classifier to decide if a drift occurred.

        :return: True if a drift occurred, else False
        """
        disc_labels = self._get_labels()
        discriminator = LogisticRegression(solver="liblinear", random_state=self.seed)
        raw_shap = self._compute_shap_values()
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
    
    def _get_raw_shap(self):
        """
        Compute raw SHAP values using the cached shap.TreeExplainer (built once per
        reference window in ``build_reference``).

        Always returns a list of arrays ``[(n_samples, n_features)] * n_classes`` so that
        downstream reduction methods are consistent across backends.

        :return: list of arrays, one per class, each of shape (n_samples, n_features)
        """
        all_data = np.array(self.data)
        if self._raw_shap_cache is None:
            self._raw_shap_cache = self._compute_raw_shap_block(all_data)
            return self._raw_shap_cache

        if self._raw_shap_cache[0].shape[0] != all_data.shape[0]:
            self._raw_shap_cache = self._compute_raw_shap_block(all_data)
            return self._raw_shap_cache

        missing_mask = np.isnan(self._raw_shap_cache[0]).any(axis=1)
        if np.any(missing_mask):
            missing_data = all_data[missing_mask]
            missing_raw = self._compute_raw_shap_block(missing_data)

            if len(missing_raw) != len(self._raw_shap_cache):
                self._raw_shap_cache = self._compute_raw_shap_block(all_data)
                return self._raw_shap_cache

            for class_index, class_cache in enumerate(self._raw_shap_cache):
                class_cache[missing_mask] = missing_raw[class_index]

        return self._raw_shap_cache
    #@timer
    def _compute_raw_shap_block(self, data_block: np.ndarray):
        raw = self._explainer.shap_values(data_block, check_additivity=False)
        # Normalise to list-of-arrays regardless of SHAP version
        if isinstance(raw, list):
            return raw
        if raw.ndim == 3:
            return [raw[:, :, i] for i in range(raw.shape[2])]
        # binary 2D fallback — wrap so downstream code stays uniform
        return [raw, -raw]

    def _compute_shap_values(self) -> np.ndarray:
        """
        Returns reduced SHAP values for the entire data window (reference + recent).

        The shape of the returned array depends on ``self.shap_mode``:
        - ``"first"``: (n_samples, n_features)  — first class only
        - ``"all\"``:   (n_samples, n_features * n_classes)  — all classes concatenated

        :return: 2D array of shape (n_samples, n_features[*n_classes])
        """
        shap_values = self._get_raw_shap()  # list[(n_samples, n_features)] * n_classes

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

