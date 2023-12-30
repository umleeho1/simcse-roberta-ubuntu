from random import sample
from datasets import load_metric
from SimCSE.trainers import EvalPrediction
from SimCSE.arguments import TrainingArguments
import numpy as np

pearsonr = load_metric("pearsonr").compute
spearmanr = load_metric("spearmanr").compute


def compute_metrics(pred: EvalPrediction, model):

    references = pred.label_ids  # [samples, ]
    predictions = pred.predictions  # [samples, batch_size]

    ########################################################
    # shape change to diag
    sample_size = predictions.shape[0]
    batch_size = predictions.shape[-1]
    idxs = [i for i in range(batch_size)] * ((sample_size // batch_size) + 1)
    idxs = np.array(idxs[:sample_size]).reshape(sample_size, -1)

    predictions = np.take(predictions, idxs).reshape(sample_size)
    predictions = predictions
    references = references  # cos_sim scale을 맞춰줌 -> 안맞출 경우 nan

    #########################################################

    pearson_corr = pearsonr(predictions=predictions, references=references)["pearsonr"]
    spearman_corr = spearmanr(predictions=predictions, references=references)[
        "spearmanr"
    ]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }