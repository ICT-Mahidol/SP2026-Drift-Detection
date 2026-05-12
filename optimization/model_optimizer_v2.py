from collections import deque
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
        1. Warmup phase (n_reference_samples samples):
          - Calls detector.update(x) on every sample (updates internal window, result ignored)
          - Collects (x, y) into warmup_buffer (pre-filled with recent_buffer on restart)
          - No predictions or drift detection
        2. When warmup_buffer reaches n_reference_samples:
          - Batch-trains ClassifiersV2 from the entire warmup_buffer
          - For HybridDriftDetector: builds SHAP-augmented reference window
          - Exits warmup mode
        3. Detection phase:
          - Makes predictions using the fitted classifier (no retraining)
          - Tracks a rolling recent_buffer of the last n_new_samples observations
          - Checks drift using detector.update(x)
        4. On drift:
          - Resets classifier
          - Seeds next warmup_buffer with recent_buffer (n_new_samples samples already seen)
          - Waits for n_new_samples more samples to complete the new warmup_buffer

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

        # Get window sizes from the detector
        n_reference_samples = getattr(detector, 'n_reference_samples', 500)
        recent_proportion = getattr(detector, 'recent_samples_proportion', 0.5)
        n_new_samples = int(n_reference_samples * recent_proportion)
        is_hybrid = isinstance(detector, HybridDriftDetector)

        # Rolling buffer that keeps the last n_new_samples seen during detection
        recent_buffer: deque = deque(maxlen=n_new_samples)

        for i, (x, y) in enumerate(tqdm(stream, total=getattr(stream, "n_samples", None))):

            if is_warming_up:
                detector.update(x)  # Update detector with x during warmup (no drift check)
                warmup_buffer.append((x, y))#update แต่ไม่ต้องเช็ค drift เพราะยังไม่ได้ฝึก classifier
                

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

            # Track rolling window of recent samples for warmup seeding on drift
            recent_buffer.append((x, y))

            # Check for drift.
            if detector.update(x):
                drifts.append(i)
                self.classifiers.reset()
                if is_hybrid:
                    detector._explainer = None  # invalidate old explainer until build_reference
                is_warming_up = True
                # Seed the next warmup with the samples from the drifted window
                warmup_buffer = list(recent_buffer)
                recent_buffer.clear()

        return drifts, labels, predictions