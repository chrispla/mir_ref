"""Generate embeddings from the audio files.
"""


from colorama import Fore, Style

from mir_ref.datasets.dataset import get_dataset
from mir_ref.features.feature_extraction import generate_embeddings
from mir_ref.utils import load_config


def generate(
    cfg_path,
    skip_clean=False,
    skip_deformed=False,
    no_overwrite=False,
    deform_list=None,
):
    cfg = load_config(cfg_path)

    for exp_cfg in cfg["experiments"]:
        # iterate through every dataset to generate embeddings
        print(Fore.GREEN + "# Extracting features...", Style.RESET_ALL)
        for dataset_cfg in exp_cfg["datasets"]:
            print(
                Fore.GREEN + f"## Dataset: {dataset_cfg['name']}",
                Style.RESET_ALL,
            )
        for model_name in exp_cfg["features"]:
            print(Fore.GREEN + f"### Feature: {model_name}", Style.RESET_ALL)

            dataset = get_dataset(
                dataset_cfg=dataset_cfg,
                task_cfg=exp_cfg["task"],
                features_cfg=exp_cfg["features"],
            )
            dataset.download()
            dataset.preprocess()

            generate_embeddings(
                dataset,
                model_name=model_name,
                skip_clean=skip_clean,
                skip_deformed=skip_deformed,
                no_overwrite=no_overwrite,
                deform_list=deform_list,
            )
