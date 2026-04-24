import time
from abc import ABC, abstractmethod
from typing import Optional


class SupervisedDriftDetector(ABC):
    """
    This abstract base class provides a consistent interface for all supervised concept drift detectors.
    """

    @abstractmethod
    def update(
        self,
        y_true: int,
        y_pred: int,
    ) -> bool:
        raise NotImplementedError("This abstract base class does not implement update.")


class UnsupervisedDriftDetector(ABC):
    """
    This abstract base class provides a consistent interface for all unsupervised concept drift detectors.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(time.time())
        self.seed = seed

    @abstractmethod
    def update(
        self,
        features: dict,
    ) -> bool:
        raise NotImplementedError("This abstract base class does not implement update.")


class HybridDriftDetector(ABC):
    """
    This abstract base class provides a consistent interface for hybrid concept drift detectors
    that use both features and classifier information (e.g., SHAP values) for drift detection.
    Warm-up state and buffer management are handled externally by the ModelOptimizer.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(time.time())
        self.seed = seed

    @abstractmethod
    def update(
        self,
        features: dict,
        y,
        classifier,
    ) -> bool:
        raise NotImplementedError("This abstract base class does not implement update.")

    @abstractmethod
    def build_reference(self, buffer: list, classifier) -> None:
        raise NotImplementedError("This abstract base class does not implement build_reference.")
