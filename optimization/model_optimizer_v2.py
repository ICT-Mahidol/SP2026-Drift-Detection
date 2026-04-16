from typing import List, Optional

from tqdm import tqdm

from detectors.base import HybridDriftDetector, UnsupervisedDriftDetector
from metrics.metrics import get_metrics
from .classifiers import Classifiers
from .config_generator import ConfigGenerator
from .logger import ExperimentLogger
from .parameter import Parameter


class ModelOptimizer:
    def __init__(
        self,
        base_model: callable,
        parameters: List[Parameter],
        n_runs: int,
        seeds: Optional[List[int]] = None,
    ):
        """
        Init a new ModelOptimizer.

        :param base_model: a callable of the detector under test
        :param parameters: the configuration parameters
        :param n_runs: the number of test runs for each configuration
        :param seeds: the seeds or None
        """
        self.base_model = base_model
        self.configs = ConfigGenerator(parameters, seeds=seeds)
        self.classifiers = None
        self.n_runs = n_runs

    def _detector_generator(self):
        """
        A generator that yields initialized detectors using configurations provided by the ConfigGenerator.

        :return: the initialized detectors
        """
        for config in self.configs:
            yield self.base_model(**config), config

    def optimize(self, stream, experiment_name, n_training_samples, verbose=False):
        """
        Optimize the model on the given data stream and log the results using the ExperimentLogger.
        Supports both UnsupervisedDriftDetector (D3) and HybridDriftDetector (D3-SHAP).

        :param stream: the data stream
        :param experiment_name: the name of the experiment
        :param n_training_samples: the number of training samples
        """
        for run in range(self.n_runs):
            logger = ExperimentLogger(
                stream=stream,
                model=self.base_model.__name__,
                experiment_name=experiment_name,
                config_keys=self.configs.get_parameter_names(),
            )
            for detector, config in self._detector_generator():
                if verbose:
                    print(f"{logger.model}: {config}")

                if isinstance(detector, HybridDriftDetector):
                    drifts, labels, predictions = self._run_hybrid(
                        detector, stream, n_training_samples
                    )
                else:
                    drifts, labels, predictions = self._run_unsupervised(
                        detector, stream, n_training_samples
                    )

                metrics = get_metrics(stream, drifts, labels, predictions)
                logger.log(config, metrics, drifts)

    def _run_unsupervised(self, detector, stream, n_training_samples):
        """
        Run the original unsupervised drift detection loop (D3-compatible).

        :param detector: the unsupervised drift detector
        :param stream: the data stream
        :param n_training_samples: the number of training samples
        :return: tuple of (drifts, labels, predictions)
        """
        self.classifiers = Classifiers()
        drifts = []
        labels = []
        predictions = []
        train_steps = 0

        for i, (x, y) in enumerate(stream):
            if i != 0:
                predictions.append(self.classifiers.predict(x))
                labels.append(y)
            if detector.update(x):
                drifts.append(i)
                self.classifiers.reset()
                train_steps = 0
            self.classifiers.fit(x, y, nonadaptive=i < n_training_samples)
            train_steps += 1

        return drifts, labels, predictions

    def _run_hybrid(self, detector, stream, n_training_samples):
        """
        Run the hybrid drift detection loop (D3-SHAP compatible).
        Uses window-based data collection with warm-up phases.

        During warm-up:
          - Collects (x, y) into the warmup buffer
          - No predictions are made
          - No classifier training occurs
          - No drift detection occurs
        When buffer is full:
          - Batch trains the classifier from buffer
          - Builds SHAP-augmented reference window
        During detection:
          - Predicts and collects labels normally
          - Trains classifier incrementally
          - Checks for drift using augmented features

        :param detector: the hybrid drift detector
        :param stream: the data stream
        :param n_training_samples: the number of training samples
        :return: tuple of (drifts, labels, predictions)
        """
        self.classifiers = Classifiers()
        drifts = []
        labels = []
        predictions = []
        train_steps = 0
        is_warming_up = True
        warmup_buffer = []

        for i, (x, y) in enumerate(tqdm(stream, total=getattr(stream, "n_samples", None))):
            if is_warming_up:
                warmup_buffer.append((x, y))

                if len(warmup_buffer) >= detector.n_reference_samples:
                    self.classifiers.batch_fit(
                        warmup_buffer,
                        nonadaptive=i < n_training_samples,
                    )
                    detector.build_reference(warmup_buffer, self.classifiers)
                    train_steps = len(warmup_buffer)
                    warmup_buffer = []
                    is_warming_up = False
                continue

            if i != 0:
                predictions.append(self.classifiers.predict(x))
                labels.append(y)

            if detector.update(x, self.classifiers):
                drifts.append(i)
                self.classifiers.reset()
                train_steps = 0
                is_warming_up = True
                warmup_buffer = []

            self.classifiers.fit(x, y, nonadaptive=i < n_training_samples)
            train_steps += 1

        return drifts, labels, predictions
