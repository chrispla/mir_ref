"""Custom metrics."""


def key_detection_weighted_accuracy(y_true, y_pred):
    """Calculate weighted accuracy for key detection.

    Args:
        y_true (list): List of keys (e.g. C# major).
        y_pred (list): List of predicted keys.

    Returns:
        float: Weighted accuracy.
    """
    import mir_eval
    import numpy as np

    scores = []
    macro_scores = {}

    for truth, pred in zip(y_true, y_pred):
        score = mir_eval.key.weighted_score(truth, pred)
        scores.append(score)
        if truth not in macro_scores:
            macro_scores[truth] = []
        macro_scores[truth].append(score)

    # calculate macro scores
    macro_scores_mean = []
    for key, values in macro_scores.items():
        macro_scores_mean.append(np.mean(values))

    return {"micro": np.mean(scores), "macro": np.mean(macro_scores_mean)}
