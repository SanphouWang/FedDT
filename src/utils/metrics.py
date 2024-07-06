import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


def psnr_4_adni_roi(preds, target):
    preds = preds.view([-1, 1, 1, preds.shape[1]])
    target = target.view([-1, 1, 1, target.shape[1]])
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    return psnr(preds, target)


METRIC_MAPPING = {
    "brats2019": {
        "psnr": PeakSignalNoiseRatio(data_range=1.0),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
        "mae": MeanAbsoluteError(),
    },
    "adni_roi": {
        "mse": MeanSquaredError(),
        "psnr": psnr_4_adni_roi,
    },
}
