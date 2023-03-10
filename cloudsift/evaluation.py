# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:10:19 2022

@author: Shahir
"""

import numpy as np

from cloudsift.utils import IntArrayLike1D

def calc_categorical_acc_stats(
        num_classes: int,
        y_trues: IntArrayLike1D,
        y_preds: IntArrayLike1D) -> tuple[float, list[float]]:

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    # y_true_class_mask[i, j] = 1 if y_true[j] == i else 0
    y_true_class_mask = y_trues == np.arange(num_classes).reshape(-1, 1)
    accs_per_class = []
    for label in range(num_classes):
        acc = np.mean(y_preds[y_true_class_mask[label]] == label)
        accs_per_class.append(acc)
    acc = np.mean(y_trues == y_preds)

    return acc, accs_per_class
