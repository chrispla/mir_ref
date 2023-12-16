"""Script to invoke embedding generation and evaluation.
"""

import argparse
import os

from mir_ref.conduct import conduct
from mir_ref.deform import deform
from mir_ref.evaluate import evaluate
from mir_ref.extract import generate
from mir_ref.train import train


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    # End to end
    parser_conduct = subparsers.add_parser("conduct")
    parser_conduct.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="Path of configuration file.",
    )

    # Audio deformation
    parser_deform = subparsers.add_parser("deform")
    parser_deform.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="Path of configuration file.",
    )
    parser_deform.add_argument(
        "--n_jobs", default=1, type=int, help="Number of parallel jobs"
    )

    # Feature extraction
    parser_extract = subparsers.add_parser("extract")
    parser_extract.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="Path of configuration file.",
    )
    parser_extract.add_argument(
        "--skip_clean",
        action="store_true",
        help="Skip extracting features from the clean audio.",
    )
    parser_extract.add_argument(
        "--skip_deformed",
        action="store_true",
        help="Skip extracting features from the deformed audio.",
    )
    parser_extract.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip extracting features if they already exist.",
    )
    parser_extract.add_argument(
        "--deform_list",
        default=None,
        help="Deformation scenario indices to extract features for. Arguments as comma-separated integers, e.g. 0,1,2,3",
    )

    # Training
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="Path of configuration file.",
    )
    parser_train.add_argument(
        "--run_id",
        default=None,
        help="Optional experiment ID, otherwise timestamp is used.",
    )

    # Evaluation
    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="Path of configuration file.",
    )
    parser_evaluate.add_argument(
        "--run_id",
        default=None,
        help="Experiment ID to evaluate, otherwise retrieves latest if timestamp is available.",
    )

    args = parser.parse_args()

    if args.command == "conduct":
        conduct(cfg_path=os.path.join("./configs/", args.config + ".yml"))

    if args.command == "deform":
        deform(
            cfg_path=os.path.join("./configs/", args.config + ".yml"),
            n_jobs=args.n_jobs,
        )
    elif args.command == "extract":
        if args.deform_list:
            args.deform_list = [int(i) for i in args.deform_list.split(",")]
        generate(
            cfg_path=os.path.join("./configs/", args.config + ".yml"),
            skip_clean=args.skip_clean,
            skip_deformed=args.skip_deformed,
            no_overwrite=args.no_overwrite,
            deform_list=args.deform_list,
        )
    elif args.command == "train":
        train(
            cfg_path=os.path.join("./configs/", args.config + ".yml"),
            run_id=args.run_id,
        )
    elif args.command == "evaluate":
        evaluate(
            cfg_path=os.path.join("./configs/", args.config + ".yml"),
            run_id=args.run_id,
        )


if __name__ == "__main__":
    main()
