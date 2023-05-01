# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:32:31 2022

@author: Shahir
"""

import os
import abc
from dataclasses import dataclass
from typing import Any, Optional, Union, TypeVar, Generic
from collections.abc import Iterable, Hashable

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from PIL.Image import Image

from cloudsift.datasets import RobotDataStreamer
from cloudsift.data_logging import (
    NetworkStatsLogger,
    RobotInferenceStatsLogger,
    RobotTestStatsLogger,
    CloudTrainingStatsLogger)
from cloudsift.utils import IntArrayLike1D, get_rel_pkg_path, get_timestamp_str

DEFAULT_SIM_LOG_DIR = get_rel_pkg_path("logs/")

T_co = TypeVar('T_co', covariant=True)


class Robot(Generic[T_co]):
    def __init__(
            self,
            name: str,
            cache_size: Union[int, None],
            data_streamer: RobotDataStreamer[T_co]) -> None:

        self.name = name
        self.upload_cache = RobotCache(cache_size)
        self.data_streamer = data_streamer


class RobotCache:
    def __init__(
            self,
            cache_size: Optional[int] = None) -> None:

        self._cache_size = cache_size or float('inf')
        self._cache = {}
        self._byte_counts = {}
        self._current_cache_usage = 0

    def clear(
            self) -> None:

        self._cache = {}
        self._byte_counts = {}
        self._current_cache_usage = 0

    def push(
            self,
            key: Hashable,
            value: Any,
            num_bytes: int) -> int:

        # it is the job of the caller to be "honest" and actually list properly how many bytes this entry
        # would use in real life

        assert value is not None
        assert (self._current_cache_usage + num_bytes) <= self._cache_size
        num_bytes = int(num_bytes)

        if key in self._cache:
            self.pop(key)
        self._cache[key] = value
        self._current_cache_usage += num_bytes
        self._byte_counts[key] = num_bytes

        return self._current_cache_usage

    def get(
            self,
            key: Hashable,
            default: Optional[Any] = None) -> Any:

        return self._cache.get(key, default)

    def pop(
            self,
            key: Hashable,
            default=None) -> Any:

        value = self._cache.pop(key, default)
        self._current_cache_usage -= self._byte_counts[key]

        return value

    def get_bytes_used_key(
            self,
            key: Hashable) -> int:

        return self._byte_counts[key]

    def get_bytes_used(
            self) -> int:

        return self._current_cache_usage

    def get_bytes_left(
            self) -> int:

        return self._cache_size - self._current_cache_usage

    def keys(
            self) -> Iterable[Hashable]:

        return self._cache.keys()

    def values(
            self) -> Iterable[Any]:

        return self._cache.values()


@dataclass(frozen=True)
class RobotIterationResult:
    y_pred: int


class BaseRobotPolicy(metaclass=abc.ABCMeta):
    def __init__(
            self,
            robot: Robot) -> None:

        self.robot = robot

    @abc.abstractmethod
    def update_policy(
            self,
            update_info: Any) -> None:

        pass

    @abc.abstractmethod
    def classify_data_samples(
            self,
            input_imgs_raw: list[Image]) -> torch.Tensor:

        pass

    @abc.abstractmethod
    def process_next_input_sample(
            self,
            input_img: Image,
            unseen_orig_label: int) -> RobotIterationResult:

        pass

    def process_next_input_sample_batched(
            self,
            input_imgs: list[Image],
            unseen_orig_labels: IntArrayLike1D) -> list[RobotIterationResult]:

        raise NotImplementedError

    @abc.abstractmethod
    def get_upload_payload(
            self) -> tuple[Any, int]:

        pass

    def start_round(
            self) -> None:

        pass

    def finish_round(
            self) -> None:

        pass


@dataclass(frozen=True)
class RobotInitialPolicyInfo:
    robot_policies: list[BaseRobotPolicy]
    robot_bytes_downloaded: list[int]


@dataclass(frozen=True)
class RobotPolicyUpdateInfo(Generic[T_co]):
    robot_policy_updates: list[T_co]
    robot_bytes_downloaded: list[int]


@dataclass(frozen=True)
class LabelingResult(Generic[T_co]):
    labeling_success: bool
    label: Union[T_co, None]


class BaseCloudLabeler(Generic[T_co], metaclass=abc.ABCMeta):
    def __init__(
            self) -> None:

        self.num_labels_requested = 0

    @abc.abstractmethod
    def request_labels(
            self,
            num_labels: int) -> list[LabelingResult[T_co]]:

        self.num_labels_requested += num_labels


class AutomaticCloudLabeler(BaseCloudLabeler[T_co]):
    def __init__(
            self) -> None:

        super().__init__()

    def request_labels(
            self,
            unseen_orig_labels: list[T_co]) -> list[LabelingResult[T_co]]:

        super().request_labels(len(unseen_orig_labels))

        out = [LabelingResult(labeling_success=True, label=y) for y in unseen_orig_labels]

        return out


@dataclass(frozen=True)
class CloudTrainingUpdateInfo(Generic[T_co]):
    robot_upload_payloads: list[T_co]


@dataclass(frozen=True)
class CloudProcessingResult:
    dataset_dist: list[float]
    training_stats: Any
    num_labels_requested: int


class BaseCloudComputer(metaclass=abc.ABCMeta):
    def __init__(
            self,
            num_robots: int,
            label_allocator: BaseCloudLabeler) -> None:

        self.num_robots = num_robots
        self.label_allocator = label_allocator

    def initialize(
            self) -> None:

        pass

    @abc.abstractmethod
    def get_init_robot_policies(
            self,
            robots: list[Robot]) -> RobotInitialPolicyInfo:

        pass

    @abc.abstractmethod
    def get_robot_policy_updates(
            self) -> RobotPolicyUpdateInfo:

        pass

    @abc.abstractmethod
    def process_round(
            self,
            update_info: RobotPolicyUpdateInfo) -> CloudProcessingResult:

        pass


class Simulator:
    def __init__(
            self,
            *,
            num_robots: int,
            num_classes: int,
            experiment_data: dict[str, Any],
            cache_size: int,
            iterations_per_round: int,
            cloud_computer: BaseCloudComputer,
            device: torch.device,
            enable_batched_inference: bool = False,
            inference_batch_size: int = 128,
            save_dir: Optional[str] = None) -> None:

        self.num_robots = num_robots
        self.num_classes = num_classes
        self.experiment_data = experiment_data
        self.iterations_per_round = iterations_per_round
        self.cloud_computer = cloud_computer
        self.device = device

        self.enable_batched_inference = enable_batched_inference
        self.inference_batch_size = inference_batch_size

        data_streamers = experiment_data['robot_data_augmented_streamers']
        self.robots = [
            Robot(
                name="robot_{}".format(i),
                cache_size=cache_size,
                data_streamer=data_streamers[i])
            for i in range(self.num_robots)]

        self.round_index = 0
        self.iteration_index = 0

        dataset_test = self.experiment_data['datasets_raw']['test']
        self.dataloader_test = DataLoader(
            dataset_test,
            batch_size=128,
            shuffle=True,
            num_workers=0,
            collate_fn=self._test_collate,
            pin_memory=True)

        if save_dir is None:
            save_dir = os.path.join(DEFAULT_SIM_LOG_DIR, "Simulation {}".format(get_timestamp_str()))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.network_stats_log = NetworkStatsLogger(
            num_robots=num_robots,
            log_fname=os.path.join(self.save_dir, "Network Stats.log"))
        self.robot_inference_stats_log = RobotInferenceStatsLogger(
            num_robots=num_robots,
            num_classes=num_classes,
            log_fname=os.path.join(self.save_dir, "Robot Inference Stats.log"))
        self.cloud_training_stats_log = CloudTrainingStatsLogger(
            num_robots=num_robots,
            num_classes=num_classes,
            log_fname=os.path.join(self.save_dir, "Cloud Training Stats.log"))
        self.robot_test_stats_log = RobotTestStatsLogger(
            num_robots=num_robots,
            num_classes=num_classes,
            log_fname=os.path.join(self.save_dir, "Robot Test Stats.log"))

    @staticmethod
    def _test_collate(
            batch: Any) -> tuple[list[Image], IntArrayLike1D]:

        input_imgs_raw = []
        labels = []
        for (input_img_raw, label) in batch:
            input_imgs_raw.append(input_img_raw)
            labels.append(label)
        labels = torch.tensor(labels, dtype=torch.int32)

        return input_imgs_raw, labels

    def _print_latest_round_results(
            self) -> None:

        for i in range(self.num_robots):
            inference_acc = self.robot_inference_stats_log.get_log_data_raw()[-1][i]['acc']
            training_stats_last_entry = self.cloud_training_stats_log.get_log_data_raw()[-1]
            network_stats_last_entry = self.network_stats_log.get_log_data_raw()[-1]
            testing_acc = self.robot_test_stats_log.get_log_data_raw()[-1][i]['testing_acc']
            robot_to_cloud_bytes = network_stats_last_entry['robots_to_cloud_bytes'][i]
            cloud_to_robot_bytes = network_stats_last_entry['cloud_to_robots_bytes'][i]

            print("=" * 40)

            print("Before data is uploaded to cloud:")
            print("Robot model {:d} inference accuracy: {}".format(i, inference_acc))
            print("Robot {:d} uploaded bytes: {:d}".format(i, robot_to_cloud_bytes))

            print("After data is uploaded to cloud:")
            print("Cloud model for robot {:d} training stats: {}".format(
                i, training_stats_last_entry['training_stats']))
            print("Robot {:d} downloaded bytes: {:d}".format(i, cloud_to_robot_bytes))

            print("Post-round analysis:")
            print("Robot {:d} model testing accuracy: {}".format(i, testing_acc))

            print()

    def _save_logs(
            self) -> None:

        self.network_stats_log.save_log()
        self.robot_inference_stats_log.save_log()
        self.cloud_training_stats_log.save_log()
        self.robot_test_stats_log.save_log()

    def _update_policy_and_log(
            self,
            y_preds_per_robot: list[IntArrayLike1D],
            y_trues_per_robot: list[IntArrayLike1D],
            verbose: bool = True) -> None:

        robot_bytes_uploaded = []
        robot_upload_payloads = []
        for i in range(self.num_robots):
            payload, bytes_uploaded = self.robot_policies[i].get_upload_payload()
            robot_bytes_uploaded.append(bytes_uploaded)
            robot_upload_payloads.append(payload)

        cloud_update_info = CloudTrainingUpdateInfo(robot_upload_payloads)
        cloud_procesing_result = self.cloud_computer.process_round(
            cloud_update_info)

        update_info = self.cloud_computer.get_robot_policy_updates()
        robot_bytes_downloaded = update_info.robot_bytes_downloaded
        assert len(robot_bytes_downloaded) == self.num_robots
        for policy, policy_update_info in zip(
                self.robot_policies,
                update_info.robot_policy_updates):
            policy.update_policy(policy_update_info)

        y_preds_per_robot = [[] for _ in range(self.num_robots)]
        y_trues_per_robot = [[] for _ in range(self.num_robots)]

        for i in range(self.num_robots):
            self.robot_inference_stats_log.log_data(
                round_index=self.round_index,
                robot_index=i,
                y_preds=y_preds_per_robot[i],
                y_trues=y_trues_per_robot[i])
        self.cloud_training_stats_log.log_data(
            round_index=self.round_index,
            dataset_dist=cloud_procesing_result.dataset_dist,
            training_stats=cloud_procesing_result.training_stats,
            num_labels_requested=cloud_procesing_result.num_labels_requested)
        test_result_data_per_robot = self.run_test()
        for i, (labels, preds) in enumerate(test_result_data_per_robot):
            self.robot_test_stats_log.log_data(
                round_index=self.round_index,
                robot_index=i,
                y_trues=labels,
                y_preds=preds)
        self.network_stats_log.log_data(
            round_index=self.round_index,
            robots_to_cloud_bytes=robot_bytes_uploaded,
            cloud_to_robots_bytes=robot_bytes_downloaded)

        if verbose:
            self._print_latest_round_results()

    def _run_iteration(
            self) -> tuple[IntArrayLike1D, IntArrayLike1D]:

        y_pred_per_robot = []
        y_true_per_robot = []
        for i in range(self.num_robots):
            input_img, label = self.robots[i].data_streamer.get_sample()
            result = self.robot_policies[i].process_next_input_sample(
                input_img,
                label)
            y_pred_per_robot.append(result.y_pred)
            y_true_per_robot.append(label)

        return y_pred_per_robot, y_true_per_robot

    def _run_iterations_batched(
            self,
            batch_size: int) -> list[tuple[IntArrayLike1D, IntArrayLike1D]]:

        input_imgs_per_robot = []
        labels_per_robot = []
        for i in range(self.num_robots):
            input_imgs = []
            labels = []
            for _ in range(batch_size):
                input_img, label = self.robots[i].data_streamer.get_sample()
                input_imgs.append(input_img)
                labels.append(label)
            input_imgs_per_robot.append(input_imgs)
            labels_per_robot.append(labels)

        results_per_robot = []
        for i in range(self.num_robots):
            results = self.robot_policies[i].process_next_input_sample_batched(
                input_imgs_raw=input_imgs_per_robot[i],
                unseen_orig_labels=labels_per_robot[i])
            results_per_robot.append(results)

        out = []
        for i in range(batch_size):
            y_pred_per_robot = [results_per_robot[j][i] for j in range(self.num_robots)]
            y_true_per_robot = [labels_per_robot[j][i] for j in range(self.num_robots)]
            out.append((y_pred_per_robot, y_true_per_robot))

        return out

    def initialize(
            self,
            verbose: bool = True) -> None:

        if verbose:
            print("Saving to:", self.save_dir)

        self.cloud_computer.initialize()

        robot_inital_policy_info = self.cloud_computer.get_init_robot_policies(self.robots)
        self.robot_policies = robot_inital_policy_info.robot_policies

        for i in range(self.num_robots):
            self.robot_inference_stats_log.log_data(
                round_index=self.round_index,
                robot_index=i,
                y_preds=[],
                y_trues=[])

        self.cloud_training_stats_log.log_data(
            round_index=self.round_index,
            dataset_dist=self.experiment_data['class_dists']['cloud'],
            training_stats=None,
            num_labels_requested=0)

        test_result_data_per_robot = self.run_test()
        for i, (labels, preds) in enumerate(test_result_data_per_robot):
            self.robot_test_stats_log.log_data(
                round_index=self.round_index,
                robot_index=i,
                y_trues=labels,
                y_preds=preds)

        self.network_stats_log.log_data(
            round_index=self.round_index,
            robots_to_cloud_bytes=[0] * self.num_robots,
            cloud_to_robots_bytes=robot_inital_policy_info.robot_bytes_downloaded)
        self.round_index += 1

        if verbose:
            print("Saving to:", self.save_dir)
            self._print_latest_round_results()
        self._save_logs()

    def run_test(
            self) -> None:

        test_result_data_per_robot = []

        for i in range(self.num_robots):
            label_datas = []
            pred_datas = []

            for input_imgs_raw, labels in tqdm(self.dataloader_test):
                labels = labels.to(self.device)
                preds = self.robot_policies[i].classify_data_samples(
                    input_imgs_raw).to(self.device)
                label_datas.append(labels.detach().cpu())
                pred_datas.append(preds.detach().cpu())

            labels = torch.concat(label_datas, dim=0).numpy()
            preds = torch.concat(pred_datas, dim=0).numpy()
            test_result_data_per_robot.append((labels, preds))

        return test_result_data_per_robot

    def run_round(
            self) -> None:

        for policy in self.robot_policies:
            policy.start_round()

        print("Simulating round index", self.round_index)
        y_preds_per_robot = [[] for _ in range(self.num_robots)]
        y_trues_per_robot = [[] for _ in range(self.num_robots)]

        self.iteration_index = 0
        if not self.enable_batched_inference:
            for _ in tqdm(range(self.iterations_per_round)):
                y_pred_per_robot, y_true_per_robot = self._run_iteration()
                for i in range(self.num_robots):
                    y_preds_per_robot[i].append(y_pred_per_robot[i])
                    y_trues_per_robot[i].append(y_true_per_robot[i])
                self.iteration_index += 1
        else:
            pbar = tqdm(total=self.iterations_per_round)
            while True:
                batch_size = min(self.iterations_per_round - self.iteration_index, self.inference_batch_size)
                if batch_size <= 0:
                    break
                outputs = self._run_iterations_batched(batch_size)
                for y_pred_per_robot, y_true_per_robot in outputs:
                    for i in range(self.num_robots):
                        y_preds_per_robot[i].append(y_pred_per_robot[i])
                        y_trues_per_robot[i].append(y_true_per_robot[i])
                    self.iteration_index += 1
                    pbar.update(1)
            pbar.close()

        self._update_policy_and_log(
            y_preds_per_robot=y_preds_per_robot,
            y_trues_per_robot=y_trues_per_robot)
        for policy in self.robot_policies:
            policy.finish_round()
        self.round_index += 1
        self._save_logs()

    def run_rounds(
            self,
            num_rounds: int) -> None:

        for _ in range(num_rounds):
            print("=" * 60)
            self.run_round()
