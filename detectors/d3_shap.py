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
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.data = []
        self.labels = []
        self.n_reference_samples = n_reference_samples
        self.recent_samples_proportion = recent_samples_proportion
        self.n_samples = int(n_reference_samples * (1 + recent_samples_proportion))
        self.threshold = threshold
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

    def _compute_shap_values(self, classifiers: Classifiers) -> np.ndarray:
        """
        Returns raw SHAP values for the entire data window (reference + recent) as a single 2D array.
        Uses the reference window as background and explains all samples.
        Uses the first class only.

        :param classifiers: the Classifiers instance
        :return: array of shape (n_samples, n_features) containing raw SHAP values
        """
        import shap

        model = classifiers.assisted_hoeffding_tree
        background = np.array(self.data[: self.n_reference_samples])
        all_data = np.array(self.data)
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(all_data)

        # list of arrays: [(n_samples, n_features)] * n_classes  — older SHAP
        if isinstance(shap_values, list):
            return shap_values[0]

        # 3D array: (n_samples, n_features, n_classes)  — newer SHAP
        if shap_values.ndim == 3:
            return shap_values[:, :, 0]

        # 2D array: (n_samples, n_features)  — binary fallback
        return shap_values

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

    '''
        Alternative: _compute_shap_values that returns SHAP values for all classes
        instead of only the first class.
        It can be used directly with the existing _augment_data_with_shap.
        The result is that augmented_data will have shape
        (n_samples, n_features + n_features * n_classes)
    '''

    '''
    def _compute_shap_values_all_classes(self, classifiers: Classifiers) -> np.ndarray:
        import shap
    
        model = classifiers.assisted_hoeffding_tree
        background = np.array(self.data[: self.n_reference_samples])
        all_data = np.array(self.data)
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(all_data)
    
        # list of arrays: [(n_samples, n_features)] * n_classes  — older SHAP
        if isinstance(shap_values, list):
            return np.hstack(shap_values)
    
        # 3D array: (n_samples, n_features, n_classes)  — newer SHAP
        if shap_values.ndim == 3:
            n_samples, n_features, n_classes = shap_values.shape
            return shap_values.reshape(n_samples, n_features * n_classes)
    
        # 2D array: (n_samples, n_features)  — binary fallback
        return shap_values
    '''