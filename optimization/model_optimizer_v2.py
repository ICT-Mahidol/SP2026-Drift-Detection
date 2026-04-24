from typing import List, Optional

from tqdm import tqdm

from detectors.base import HybridDriftDetector, UnsupervisedDriftDetector
from metrics.metrics import get_metrics
from .classifiers_v2 import ClassifiersV2
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
        All detectors now use batch-based classifier training.

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

                drifts, labels, predictions = self._run_unsupervised_batch(
                    detector, stream, n_training_samples
                )

                metrics = get_metrics(stream, drifts, labels, predictions)
                logger.log(config, metrics, drifts)

    def _run_unsupervised_batch(self, detector, stream, n_training_samples):
        """
        Run batch-based drift detection for both UnsupervisedDriftDetector and HybridDriftDetector.
        Uses window-based data collection with warm-up phases and batch classifier training.

        Workflow:
        1. Warmup phase:
          - Collects (x, y) into the warmup buffer
          - No predictions or drift detection
        2. When buffer reaches n_reference_samples:
          - Batch-trains ClassifiersV2 from the entire buffer
          - For HybridDriftDetector: builds SHAP-augmented reference window
          - Exits warmup mode
        3. Detection phase:
          - Makes predictions using the fitted classifier (no retraining)
          - For HybridDriftDetector: checks drift using update(x, classifier)
          - For UnsupervisedDriftDetector: checks drift using update(x)
        4. On drift:
          - Resets classifier and restarts warmup

        :param detector: unsupervised or hybrid drift detector
        :param stream: the data stream
        :param n_training_samples: the number of training samples
        :return: tuple of (drifts, labels, predictions)
        """
        self.classifiers = ClassifiersV2()
        drifts = []
        labels = []
        predictions = []
        is_warming_up = True
        warmup_buffer = []
        
        # Get the reference window size from detector, default to 200 if not available
        n_reference_samples = getattr(detector, 'n_reference_samples', 200)
        is_hybrid = isinstance(detector, HybridDriftDetector)

        for i, (x, y) in enumerate(tqdm(stream, total=getattr(stream, "n_samples", None))):
            if is_warming_up:
                warmup_buffer.append((x, y))

                if len(warmup_buffer) >= n_reference_samples:
                    # Batch fit classifier on the warmup buffer
                    self.classifiers.batch_fit(
                        warmup_buffer,
                        nonadaptive=i < n_training_samples,
                    )
                    
                    # For hybrid detectors, build reference window with SHAP
                    if is_hybrid:
                        detector.build_reference(warmup_buffer, self.classifiers)
                    
                    warmup_buffer = []
                    is_warming_up = False
                continue

            # Make predictions using the fitted classifier
            if i != 0:
                predictions.append(self.classifiers.predict(x))
                labels.append(y)

            # Check for drift (different update signatures for hybrid vs unsupervised)
            if is_hybrid:
                drift_detected = detector.update(x, self.classifiers)
            else:
                drift_detected = detector.update(x)
            
            if drift_detected:
                drifts.append(i)
                self.classifiers.reset()
                is_warming_up = True
                warmup_buffer = []

        return drifts, labels, predictions
