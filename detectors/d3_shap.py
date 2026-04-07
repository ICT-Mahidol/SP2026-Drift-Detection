from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .base import UnsupervisedDriftDetector
from optimization.classifiers import Classifiers


class DiscriminativeDriftDetector2019SHAP(UnsupervisedDriftDetector):
    """
    Discriminative Drift Detector (D3) with SHAP-based feature importance.
    Modification of DiscriminativeDriftDetector2019 that also collects labels alongside features.
    """

    def __init__(
        self,
        n_reference_samples: int = 100,
        recent_samples_proportion: float = 0.1,
        threshold: float = 0.7,
        shap_threshold: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.data = []
        self.labels = []
        self.n_reference_samples = n_reference_samples
        self.recent_samples_proportion = recent_samples_proportion
        self.n_samples = int(n_reference_samples * (1 + recent_samples_proportion))
        self.threshold = threshold
        self.shap_threshold = shap_threshold
        self.kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
        self.shap_values = None

    def update(self, features: dict, label, classifiers: Classifiers) -> bool:
        """
        Update the detector with the most recent observation and detect if a drift occurred.

        :param features: the features
        :param label: the true class label of the observation
        :param classifiers: the Classifiers instance for use in future steps
        :returns: True if a drift occurred else False
        """
        features = np.fromiter(features.values(), dtype=float)
        if len(self.data) != self.n_samples:
            self.data.append(features)
            self.labels.append(label)
        else:
            if self._detect_drift(classifiers):
                self.data = self.data[self.n_reference_samples:]
                self.labels = self.labels[self.n_reference_samples:]
                return True
            else:
                step = int(np.ceil(self.n_reference_samples * self.recent_samples_proportion))
                self.data = self.data[step:]
                self.labels = self.labels[step:]
        return False

    def _detect_drift(self, classifiers: Classifiers) -> bool:
        """
        Detect if a drift occurred.
        If shap_threshold is set, computes SHAP on the current window and uses it
        as a second criterion alongside AUC.

        :param classifiers: the Classifiers instance
        :return: True if a drift occurred, else False
        """
        disc_labels = self._get_labels()
        discriminator = LogisticRegression(solver="liblinear", random_state=self.seed)
        predictions = self._predict(discriminator, np.array(self.data), disc_labels)
        auc_score = roc_auc_score(disc_labels, predictions)

        if auc_score < self.threshold:
            return False

        if self.shap_threshold is not None:
            self.shap_values = self._compute_shap_values(classifiers)
            recent_labels = self.labels[self.n_reference_samples:]
            total = len(recent_labels)
            importance = np.sum(
                [self.shap_values[cls] * (recent_labels.count(cls) / total)
                 for cls in self.shap_values],
                axis=0,
            )
            ref_mean = np.array(self.data[: self.n_reference_samples]).mean(axis=0)
            rec_mean = np.array(self.data[self.n_reference_samples :]).mean(axis=0)
            shap_score = np.sum(importance * np.abs(rec_mean - ref_mean))
            return shap_score >= self.shap_threshold

        return True

    def _get_labels(self) -> np.array:
        """
        Get discriminative labels for the reference window (0) and recent window (1).

        :return: the labels
        """
        disc_labels = np.zeros(self.n_samples)
        disc_labels[self.n_reference_samples:] = 1
        return disc_labels

    def _compute_shap_values(self, classifiers: Classifiers) -> dict:
        """
        Returns mean per-feature SHAP values across the recent window for all classes.
        Uses the reference window as background and the recent window as samples to explain.
        Works for binary and any number of classes.

        :param classifiers: the Classifiers instance
        :return: dict mapping each class label to a mean SHAP values array of shape (n_features,)
        """
        import shap

        model = classifiers.assisted_hoeffding_tree
        background = np.array(self.data[: self.n_reference_samples])
        recent = np.array(self.data[self.n_reference_samples :])
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(recent)
        classes = list(model.classes_)

        # list of arrays: [(n_recent, n_features)] * n_classes  — older SHAP
        if isinstance(shap_values, list):
            return {cls: np.abs(shap_values[i]).mean(axis=0) for i, cls in enumerate(classes)}

        # 3D array: (n_recent, n_features, n_classes)  — newer SHAP
        if shap_values.ndim == 3:
            return {cls: np.abs(shap_values[:, :, i]).mean(axis=0) for i, cls in enumerate(classes)}

        # 2D array: (n_recent, n_features)  — binary fallback
        return {classes[0]: np.abs(shap_values).mean(axis=0)}

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
