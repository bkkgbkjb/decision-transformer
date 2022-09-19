import setup
from experiment import experiment
from ray import tune
from args import args
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch
from os import path
from ray.tune.logger import TBXLoggerCallback
import numpy as np


def trail(params):
    import setup
    import d4rl
    
    from ray.tune.utils import wait_for_gpu
    wait_for_gpu(target_util= 1 - 1/16)

    experiment("gym-experiment-tune", {**vars(args), **params})



params = {
    "pref_loss_ratio": tune.quniform(0.05, 0.25, 0.05),
    "phi_norm_loss_ratio": tune.quniform(0.05, 0.25, 0.05),
    "in_tune": True,
    "w_lr": tune.choice([1e-3, 5e-4, 5e-3, 1e-2]),
    "dirpath": path.abspath('./')
}

tune.run(
    trail,
    metric="eval/return",
    mode="max",
    search_alg=OptunaSearch(),
    scheduler=AsyncHyperBandScheduler(
        max_t=100, grace_period=int(100 / 2)
    ),
    resources_per_trial={"cpu": 1 / 64, "gpu": 1 / 16},
    max_concurrent_trials=16,
    config=params,
    num_samples=32,
    verbose=1,
    log_to_file=False,
    sync_config=tune.SyncConfig(syncer=None),
    local_dir=path.abspath("./ray"),
    callbacks=[TBXLoggerCallback()]
)