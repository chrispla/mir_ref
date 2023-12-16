"""End-to-end experiment conduct. Currently, this is a hacky way of
doing things, and it will be replaced soon with an online process."""


from mir_ref.deform import deform
from mir_ref.evaluate import evaluate
from mir_ref.extract import generate
from mir_ref.train import train


def conduct(cfg_path):
    """Conduct experiments in config end-to-end.

    Args:
        cfg_path (str): Path to config file.
    """
    deform(cfg_path=cfg_path, n_jobs=1)
    generate(cfg_path=cfg_path)
    train(cfg_path=cfg_path)
    evaluate(cfg_path=cfg_path)
