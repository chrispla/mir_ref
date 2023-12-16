"""Apply deformations to audio files.
"""

from colorama import Fore, Style

from mir_ref.datasets.dataset import get_dataset
from mir_ref.deformations import generate_deformations
from mir_ref.utils import load_config


def deform(cfg_path, n_jobs):
    cfg = load_config(cfg_path)

    # iterate through every dataset of every experiment to generate deformations
    print(Fore.GREEN + "# Generating deformations...", Style.RESET_ALL)
    for exp_cfg in cfg["experiments"]:
        for dataset_cfg in exp_cfg["datasets"]:
            # generate deformations
            print(Fore.GREEN + f"## Dataset: {dataset_cfg['name']}", Style.RESET_ALL)

            dataset = get_dataset(
                dataset_cfg=dataset_cfg,
                task_cfg=exp_cfg["task"],
                features_cfg=exp_cfg["features"],
            )
            dataset.download()
            dataset.preprocess()

            generate_deformations(
                dataset,
                n_jobs=n_jobs,
            )
