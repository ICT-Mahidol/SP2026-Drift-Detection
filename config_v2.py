from datasets import (
    IncrementalDrift,
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        IncrementalDrift(),
    ]
    n_training_samples = 1000
    models = [
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[250, 500]),
                Parameter("recent_samples_proportion", values=[1.0]),
                Parameter("threshold", values=[0.7, 0.8]),
            ],
            seeds=None,
            n_runs=5,
        ),
    ]
