from datasets import (
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalBalanced,
)
from detectors import *
from optimization.model_optimizer_v2 import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        InsectsAbruptBalanced(),
        #InsectsGradualBalanced(),
        #InsectsIncrementalBalanced(),
    ]
    n_training_samples = 1000
    detectors = [
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[250]),
                Parameter("recent_samples_proportion", values=[1.0]),
                Parameter("threshold", values=[0.7]),
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019SHAP,
            parameters=[
                Parameter("n_reference_samples", values=[250]),
                Parameter("recent_samples_proportion", values=[1.0]),
                Parameter("threshold", values=[0.7]),
                Parameter("shap_mode", values=["first"]),
                Parameter("shap_classifier", values=["lightgbm"]),
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019SHAPMeanAbs,
            parameters=[
                Parameter("n_reference_samples", values=[250, 500]),
                Parameter("recent_samples_proportion", values=[1.0]),
                Parameter("threshold", values=[0.7, 0.8]),
                Parameter("shap_classifier", values=["lightgbm"]),
            ],
            seeds=None,
            n_runs=1,
        ),
    ]
