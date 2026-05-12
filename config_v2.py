from datasets import (
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalBalanced,
    AbruptDrift,
    GradualDrift,
    SineClusters,
    WaveformDrift2,
)
from detectors import *
from optimization.model_optimizer_v2 import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        InsectsAbruptBalanced(),
        InsectsGradualBalanced(),
        InsectsIncrementalBalanced(),
        AbruptDrift(),
        GradualDrift(),
        SineClusters(drift_frequency=12000, stream_length=52000, seed=531874),
        WaveformDrift2(drift_frequency=12000, stream_length=52000, seed=2401137),
    ]
    n_training_samples = 10000
    detectors = [
        
        # ModelOptimizer(
        #     base_model=DiscriminativeDriftDetector2019,
        #     parameters=[ 
        #         Parameter("n_reference_samples", values=[3000]),
        #         Parameter("recent_samples_proportion", values=[0.1]),
        #         Parameter("threshold", values=[0.7]),
        #         #Parameter("shap_classifier", values=["lightgbm"]),
        #     ],
        #     seeds=None,
        #     n_runs=20,
        # ),
        # ModelOptimizer(
        #     base_model=DiscriminativeDriftDetector2019SHAP,
        #     parameters=[
        #         Parameter("n_reference_samples", values=[3000]),
        #         Parameter("recent_samples_proportion", values=[0.1]),
        #         Parameter("threshold", values=[0.7]),
        #         Parameter("shap_mode", values=["first","all"]),
        #         Parameter("shap_classifier", values=["lightgbm"]),
        #     ],
        #     seeds=None,
        #     n_runs=20,
        # ),
        # ModelOptimizer(
        #     base_model=DiscriminativeDriftDetector2019SHAPMeanAbs,
        #     parameters=[
        #         Parameter("n_reference_samples", values=[3000]),
        #         Parameter("recent_samples_proportion", values=[0.1]),
        #         Parameter("threshold", values=[0.7]),
        #         Parameter("shap_classifier", values=["lightgbm"]),
        #     ],
        #     seeds=None,
        #     n_runs=20,
        # ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019SHAPFirst_permutation,
            parameters=[ 
                Parameter("n_reference_samples", values=[3000]),
                Parameter("recent_samples_proportion", values=[0.1]),
                Parameter("threshold", values=[0.7]),
                Parameter("shap_classifier", values=["lightgbm"]),
            ],
            seeds=None,
            n_runs=20,
        ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019SHAPMeanAbs_permutation,
            parameters=[ 
                Parameter("n_reference_samples", values=[3000]),
                Parameter("recent_samples_proportion", values=[0.1]),
                Parameter("threshold", values=[0.7]),
                Parameter("shap_classifier", values=["lightgbm"]),
            ],
            seeds=None,
            n_runs=20,
        ),
    ]
