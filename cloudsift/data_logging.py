# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:38:45 2023

@author: Shahir
"""

import abc
import copy
from typing import Any, Union
import json

import numpy as np

import torch

from cloudsift.evaluation import calc_categorical_acc_stats
from cloudsift.utils import IntArrayLike1D, FloatArrayLike1D


def _to_number_list(
        arr: Union[IntArrayLike1D, FloatArrayLike1D]) -> list[Union[int, float]]:

    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().tolist()
    else:
        return list(arr)


class SimulationLogger(metaclass=abc.ABCMeta):
    def __init__(
            self,
            log_fname: str) -> None:

        self.log_fname = log_fname

    @abc.abstractmethod
    def save_log(
            self) -> None:

        pass


class NetworkStatsLogger(SimulationLogger):
    LOG_AXIS_NAMES = ('round_index',)

    def __init__(
            self,
            num_robots: int,
            log_fname: str) -> None:

        super().__init__(log_fname)
        self.num_robots = num_robots
        self._log = []

    def log_data(
            self,
            round_index: int,
            robots_to_cloud_bytes: IntArrayLike1D,
            cloud_to_robots_bytes: IntArrayLike1D) -> None:

        assert len(cloud_to_robots_bytes) == self.num_robots
        assert len(robots_to_cloud_bytes) == self.num_robots

        if round_index == len(self._log):
            data = {
                'cloud_to_robots_bytes': _to_number_list(cloud_to_robots_bytes),
                'robots_to_cloud_bytes': _to_number_list(robots_to_cloud_bytes)
            }
            self._log.append(data)
        else:
            raise ValueError()

    def get_log_data_raw(
            self) -> list[dict[str, Any]]:

        return copy.deepcopy(self._log)

    def get_log_data_collated(
            self) -> dict[str, np.ndarray]:

        data = {}
        for k in ('cloud_to_robots_bytes', 'robots_to_cloud_bytes'):
            data_per_round = np.array([n[k] for n in self._log])
            data[k] = data_per_round

        return data

    def save_log(
            self) -> None:

        with open(self.log_fname, 'w') as f:
            json.dump(self._log, f)


class RobotInferenceStatsLogger(SimulationLogger):
    LOG_AXIS_NAMES = ('round_index', 'robot_index')

    def __init__(
            self,
            num_robots: int,
            num_classes: int,
            log_fname: str) -> None:

        super().__init__(log_fname)
        self.num_robots = num_robots
        self.num_classes = num_classes
        self._log = []

    def log_data(
            self,
            round_index: int,
            robot_index: int,
            y_preds: IntArrayLike1D,
            y_trues: IntArrayLike1D) -> None:

        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        assert (len(y_trues) == len(y_preds))
        if round_index == len(self._log):
            self._log.append([None for _ in range(self.num_robots)])
        if round_index == (len(self._log) - 1):
            assert self._log[round_index][robot_index] is None
            data = {
                'y_pred': _to_number_list(y_preds),
                'y_true': _to_number_list(y_trues),
                'acc': None if len(y_preds) == 0 else float(np.mean(y_preds == y_trues))
            }
            self._log[round_index][robot_index] = data
        else:
            raise ValueError()

    def get_log_data_raw(
            self) -> list[list[dict[str, Any]]]:

        return copy.deepcopy(self._log)

    def get_log_data_collated(
            self) -> dict[str, np.ndarray]:

        log_array = np.array(self._log)
        log_flattened = log_array.flatten()
        data = {}
        for k in ('y_preds', 'y_trues', 'correct'):
            data[k] = np.array([n[k] for n in log_flattened]).copy()

        return data

    def save_log(
            self) -> None:

        with open(self.log_fname, 'w') as f:
            json.dump(self._log, f)


class CloudTrainingStatsLogger(SimulationLogger):
    LOG_AXIS_NAMES = ('round_index',)

    def __init__(
            self,
            num_robots: int,
            num_classes: int,
            log_fname: str) -> None:

        super().__init__(log_fname)
        self.num_robots = num_robots
        self.num_classes = num_classes
        self._log = []

    def log_data(
            self,
            round_index: int,
            dataset_dist: FloatArrayLike1D,
            training_stats: str,
            num_labels_requested: int) -> None:

        if round_index == len(self._log):
            data = {
                'dataset_dist': _to_number_list(dataset_dist),
                'training_stats': str(training_stats),
                'num_labels_requested': int(num_labels_requested)
            }
            self._log.append(data)
        else:
            raise ValueError()

    def get_log_data_raw(
            self) -> list[dict[str, Any]]:

        return self._log

    def get_log_data_collated(
            self) -> dict[str, np.ndarray]:

        data = {}
        for k in ('dataset_dist', 'training_loss', 'training_acc', 'num_labels_requested'):
            data_per_round = np.array([n[k] for n in self._log])
            data[k] = data_per_round

        return data

    def save_log(
            self) -> None:

        with open(self.log_fname, 'w') as f:
            json.dump(self._log, f)


class RobotTestStatsLogger(SimulationLogger):
    LOG_AXIS_NAMES = ('round_index', 'robot_index')

    def __init__(
            self,
            num_robots: int,
            num_classes: int,
            log_fname: str) -> None:

        super().__init__(log_fname)
        self.num_robots = num_robots
        self.num_classes = num_classes
        self._log = []

    def log_data(
            self,
            round_index: int,
            robot_index: int,
            y_trues: IntArrayLike1D,
            y_preds: IntArrayLike1D) -> None:

        assert (len(y_trues) == len(y_preds))
        if round_index == len(self._log):
            self._log.append([None for _ in range(self.num_robots)])
        if round_index == (len(self._log) - 1):
            assert self._log[round_index][robot_index] is None
            acc, accs_per_class = calc_categorical_acc_stats(self.num_classes, y_trues, y_preds)
            data = {
                'testing_acc': float(acc),
                'testing_accs_per_class': _to_number_list(accs_per_class)
            }
            self._log[round_index][robot_index] = data
        else:
            raise ValueError()

    def get_log_data_raw(
            self) -> list[list[dict[str, Any]]]:

        return self._log

    def get_log_data_collated(
            self) -> dict[str, np.ndarray]:

        log_array = np.array(self._log)
        log_flattened = log_array.flatten()
        data = {}
        for k in ('testing_acc', 'testing_accs_per_class'):
            data[k] = np.array([n[k] for n in log_flattened]).reshape(log_flattened).copy()

        return data

    def save_log(
            self) -> None:

        with open(self.log_fname, 'w') as f:
            json.dump(self._log, f)
