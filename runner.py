from config_v2 import Configuration


def run(experiment_name):
    for stream in Configuration.streams:
        for detector in Configuration.detectors:
            detector.optimize(stream, experiment_name, Configuration.n_training_samples, verbose=True)
