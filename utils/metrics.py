from dataclasses import dataclass

import numpy as np


def ratio(numerator, denominator, empty=0.0):
    return numerator / denominator if denominator else empty


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    dice_sum: float = 0.0
    slices: int = 0

    def update(self, pred, label):
        if pred.shape != label.shape:
            raise ValueError("Prediction and reference masks must have the same shape.")
        tp = int(np.count_nonzero(pred & label))
        fp = int(np.count_nonzero(pred & ~label))
        fn = int(np.count_nonzero(~pred & label))
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += pred.size - tp - fp - fn
        self.dice_sum += ratio(2 * tp, 2 * tp + fp + fn, empty=1.0)
        self.slices += 1

    def merge(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        self.dice_sum += other.dice_sum
        self.slices += other.slices

    def scores(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        foreground_iou = ratio(tp, tp + fp + fn, empty=1.0)
        background_iou = ratio(tn, tn + fp + fn, empty=1.0)
        return {
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "dice": ratio(self.dice_sum, self.slices),
            "miou": (foreground_iou + background_iou) / 2,
            "recall": ratio(tp, tp + fn),
            "precision": ratio(tp, tp + fp),
        }

    def counts(self):
        return {name: getattr(self, name) for name in ("tp", "fp", "fn", "tn", "slices")}
