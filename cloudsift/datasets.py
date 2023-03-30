# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:10:27 2022

@author: Shahir
"""

from enum import Enum
import collections
from typing import Any, Union, Optional, TypeVar, Generic
from collections.abc import Callable, Iterable

import numpy as np

import torch
from torch.utils.data import (Dataset, Subset, ConcatDataset, WeightedRandomSampler, Sampler)

import torchvision
from torchvision import transforms

import iteround
from PIL.Image import Image

from cloudsift.utils import IntArrayLike1D, FloatArrayLike1D


class ImageDatasetType(str, Enum):
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'


IMG_DATASET_TO_NUM_CLASSES = {
    ImageDatasetType.MNIST: 10,
    ImageDatasetType.CIFAR10: 10,
    ImageDatasetType.CIFAR100: 100
}

IMG_DATASET_TO_IMG_SHAPE = {
    ImageDatasetType.MNIST: (1, 28, 28),
    ImageDatasetType.CIFAR10: (3, 32, 32),
    ImageDatasetType.CIFAR100: (3, 32, 32)
}

IMG_DATASET_TO_NUM_SAMPLES = {
    ImageDatasetType.MNIST: (60000, 10000),
    ImageDatasetType.CIFAR10: (50000, 10000),
    ImageDatasetType.CIFAR100: (50000, 10000)
}

IMG_DATASET_TO_CLASS_NAMES = {
    ImageDatasetType.CIFAR10: [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
}

T_co = TypeVar('T_co', covariant=True)
T_co1 = TypeVar('T_co1', covariant=True)
T_co2 = TypeVar('T_co2', covariant=True)
T_co3 = TypeVar('T_co3', covariant=True)


class RawDataset(Generic[T_co], Dataset[T_co]):
    def __init__(
            self,
            x_data: Iterable[T_co1],
            metadata: Optional[Any] = None) -> None:

        super().__init__()

        self.x_data = x_data
        self.metadata = metadata

    def __getitem__(
            self,
            index: int) -> T_co:

        return self.x_data[index]

    def __len__(
            self) -> int:

        return len(self.x_data)


def transform_with_label(
        transform_func: Callable[[T_co1], T_co2]) -> Callable[[tuple[T_co1, T_co3]], tuple[T_co2, T_co3]]:

    def transform_with_label_wrapper(
            data: tuple[T_co1, T_co3]) -> tuple[T_co2, T_co3]:

        inputs, label = data

        return transform_func(inputs), label

    return transform_with_label_wrapper


class TransformDataset(Generic[T_co1, T_co2], Dataset[T_co2]):
    def __init__(
            self,
            dataset: Dataset,
            transform_func: Callable[[T_co1], T_co2]) -> None:

        super().__init__()

        self.dataset = dataset
        self.transform_func = transform_func

    def __getitem__(
            self,
            index: int) -> T_co2:

        inputs = self.transform_func(self.dataset[index])

        return inputs

    def __len__(
            self) -> int:

        return len(self.dataset)


class RebalancedCategoricalSampler(WeightedRandomSampler):
    def __init__(
            self,
            labels: IntArrayLike1D,
            num_classes: int,
            target_dist: Optional[FloatArrayLike1D] = None,
            target_dataset_size: Optional[int] = None,
            generator: Optional[torch.Generator] = None) -> None:

        labels = np.array(labels)
        unique_classes, unique_class_sample_count = np.unique(labels, return_counts=True)
        class_sample_count = np.zeros(num_classes, dtype=np.int32)
        class_sample_count[unique_classes] = unique_class_sample_count
        # convert to uniform class distribution
        class_weights = np.zeros(num_classes)
        mask = (class_sample_count != 0)
        class_weights[mask] = 1 / class_sample_count[mask]

        if isinstance(target_dist, np.ndarray):
            target_dist = target_dist.tolist()
        if (target_dist is None) or (target_dist == 'original'):
            target_dist = class_sample_count / len(labels)
        elif target_dist == 'uniform':
            target_dist = np.ones(num_classes) / num_classes
        else:
            assert len(target_dist) == num_classes
            target_dist = np.array(target_dist)
            target_dist /= np.sum(target_dist)

        class_weights *= target_dist
        samples_weights = class_weights[labels]

        if target_dataset_size is None:
            target_dataset_size = len(labels)

        super().__init__(
            samples_weights,
            target_dataset_size,
            replacement=True,
            generator=generator)


class RobotDataStreamer(Generic[T_co]):
    def __init__(
            self,
            dataset: Dataset[T_co],
            sampler: Sampler[int]) -> None:

        self.dataset = dataset
        self.sampler = sampler
        self._old_samples = collections.deque([])

    def get_sample(
            self) -> T_co:

        if len(self._old_samples):
            return self.dataset[self._old_samples.popleft()]
        else:
            self._old_samples = collections.deque(list(self.sampler))
            return self.get_sample()


def get_img_dataset(
        data_dir: str,
        dataset_type: ImageDatasetType,
        combine_train_test: bool = False) -> dict[str, Dataset[Image]]:

    if dataset_type == ImageDatasetType.MNIST:
        train_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == ImageDatasetType.CIFAR10:
        train_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == ImageDatasetType.CIFAR100:
        train_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=False,
            download=True)
    else:
        raise ValueError()

    if combine_train_test:
        out = {
            'all': ConcatDataset((train_dataset, test_dataset))
        }

        return out
    else:
        out = {
            'train': train_dataset,
            'test': test_dataset
        }

        return out


def augment_value_select(
        min_vals: Union[float, FloatArrayLike1D],
        max_vals: Union[float, FloatArrayLike1D],
        default_vals: Union[float, FloatArrayLike1D],
        aggro: float = 0.0,
        rng: Optional[np.random.Generator] = None) -> list[float]:

    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    default_vals = np.array(default_vals)
    rng = rng or np.random.default_rng()

    assert default_vals.shape == max_vals.shape == max_vals.shape
    assert np.all((min_vals <= default_vals) & (default_vals <= max_vals))

    if aggro == 0.0:
        return default_vals.tolist()

    values = rng.random(max_vals.shape)
    values = aggro * ((values * (max_vals - min_vals)) + min_vals) + (1 - aggro) * default_vals

    return values.tolist()


def augment_range_select(
        min_vals: Union[float, FloatArrayLike1D],
        max_vals: Union[float, FloatArrayLike1D],
        default_vals: Union[float, FloatArrayLike1D],
        aggro: float = 0.0,
        sort: bool = True,
        rng: Optional[np.random.Generator] = None) -> list[float]:

    values = augment_value_select(
        np.repeat(np.array(min_vals)[np.newaxis], 2, -1),
        np.repeat(np.array(max_vals)[np.newaxis], 2, -1),
        np.repeat(np.array(default_vals)[np.newaxis], 2, -1),
        aggro=aggro,
        rng=rng)

    if sort:
        values = np.sort(values)

    return values.tolist()


def generate_robot_img_transform(
        *,
        dataset_type: ImageDatasetType,
        aggro: float = 0.0,
        rng: Optional[np.random.Generator] = None) -> Callable[[Image], Image]:

    if dataset_type == ImageDatasetType.MNIST:
        transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=augment_range_select(
                        -10, 10, 0, aggro, rng),
                    translate=augment_value_select(
                        (0, 0), (0.1, 0.1), (0, 0), aggro, rng),
                    scale=augment_range_select(
                        0.9, 1.1, 1.0, aggro, rng),
                    shear=augment_range_select(
                        -10, 10, 0, aggro, rng))],
                p=0.9)])

    elif (dataset_type == ImageDatasetType.CIFAR10 or
          dataset_type == ImageDatasetType.CIFAR100):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(
                p=augment_value_select(
                    0.0, 1.0, 0.5, aggro, rng)),
            transforms.ColorJitter(
                brightness=augment_range_select(
                    0.8, 1.2, 1.0, aggro, rng),
                contrast=augment_range_select(
                    0.8, 1.2, 1.0, aggro, rng),
                saturation=augment_range_select(
                    0.8, 1.2, 1.0, aggro, rng),
                hue=augment_range_select(
                    -0.05, 0.05, 0.0, aggro, rng)),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=augment_range_select(
                        -10, 10, 0, aggro, rng),
                    translate=augment_value_select(
                        (0, 0), (0.1, 0.1), (0, 0), aggro, rng),
                    scale=augment_range_select(
                        0.9, 1.1, 1.0, aggro, rng))],
                p=0.5)])

    return transform


def get_img_train_test_transforms(
        *,
        dataset_type: ImageDatasetType,
        augment: bool = True,
        new_input_size: Optional[int] = None) -> dict[str, Callable]:

    normalize_mean = None
    normalize_std = None

    if dataset_type == ImageDatasetType.MNIST:
        normalize_mean = np.array([0.1307], dtype=np.float32)
        normalize_std = np.array([0.3081], dtype=np.float32)
        to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,))])

        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10)],
                p=0.9)])

        test_transform = transforms.Compose([])

    elif (dataset_type == ImageDatasetType.CIFAR10 or
          dataset_type == ImageDatasetType.CIFAR100):
        normalize_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        normalize_std = np.array([0.247, 0.243, 0.261], dtype=np.float32)
        to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.247, 0.243, 0.261))])

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1))],
                p=0.5)])

        test_transform = transforms.Compose([])
    else:
        raise ValueError()

    to_tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalize_mean.tolist(),
            std=normalize_std.tolist())])
    from_tensor_transform = transforms.Compose([
        transforms.Normalize(
            mean=(-normalize_mean / normalize_std).tolist(),
            std=(1.0 / normalize_std).tolist()),
        transforms.ToPILImage()])

    if not augment:
        train_transform = test_transform

    if new_input_size:
        train_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            train_transform])
        test_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            test_transform])

    train_transform = transforms.Compose([
        train_transform,
        to_tensor_transform])
    test_transform = transforms.Compose([
        test_transform,
        to_tensor_transform])

    out = {
        'train': train_transform,
        'test': test_transform,
        'to_tensor': to_tensor_transform,
        'from_tensor': from_tensor_transform
    }

    return out


def generate_experiment_data(
        *,
        dataset_type: ImageDatasetType,
        dataset_orig: Dataset[Image],
        num_robots: int,
        num_classes: int,
        cloud_dataset_dist: Optional[Union[str, FloatArrayLike1D]] = 'random',
        cloud_class_imbalance_aggro: float = 0.5,
        cloud_dataset_frac: float = 0.05,
        test_dataset_dist: Optional[Union[str, FloatArrayLike1D]] = None,
        test_dataset_frac: float = 0.1,
        robot_augmentation_aggro: float = 0.5,
        robot_class_imbalance_aggro: float = 1.0,
        robot_dataset_unique_frac: float = 0.2,
        rng: Optional[np.random.Generator] = None) -> dict[str, Any]:

    rng = rng or np.random.default_rng()

    labels = np.array([y for (x, y) in dataset_orig], dtype=np.int32)
    unique_classes, unique_class_sample_count = np.unique(
        labels, return_counts=True)
    dataset_orig_hist = np.zeros(num_classes, dtype=np.int32)
    dataset_orig_hist[unique_classes] = unique_class_sample_count
    dataset_orig_dist = dataset_orig_hist / len(dataset_orig)

    if isinstance(cloud_dataset_dist, np.ndarray):
        cloud_dataset_dist = cloud_dataset_dist.tolist()
    elif (cloud_dataset_dist is None):
        cloud_dataset_dist = dataset_orig_dist
    elif isinstance(cloud_dataset_dist, str):
        if cloud_dataset_dist == 'original':
            cloud_dataset_dist = dataset_orig_dist
        elif cloud_dataset_dist == 'uniform':
            cloud_dataset_dist = np.ones(num_classes) / num_classes
        elif cloud_dataset_dist == 'random':
            cloud_dataset_dist = augment_value_select(
                np.full(num_classes, 0.0),
                np.full(num_classes, 1.0),
                np.full(num_classes, 0.5),
                aggro=cloud_class_imbalance_aggro,
                rng=rng)
            cloud_dataset_dist = np.array(cloud_dataset_dist)
            cloud_dataset_dist /= np.sum(cloud_dataset_dist)
    else:
        assert len(cloud_dataset_dist) == num_classes
        cloud_dataset_dist = np.array(cloud_dataset_dist)
        cloud_dataset_dist /= np.sum(cloud_dataset_dist)

    if isinstance(test_dataset_dist, np.ndarray):
        test_dataset_dist = test_dataset_dist.tolist()
    if (test_dataset_dist is None) or (test_dataset_dist == 'original'):
        test_dataset_dist = dataset_orig_dist
    elif test_dataset_dist == 'uniform':
        test_dataset_dist = np.ones(num_classes) / num_classes
    else:
        assert len(test_dataset_dist) == num_classes
        test_dataset_dist = np.array(test_dataset_dist)
        test_dataset_dist /= np.sum(test_dataset_dist)

    shuffled_indices = np.arange(len(dataset_orig))
    rng.shuffle(shuffled_indices)

    cloud_set_size = int(len(dataset_orig) * cloud_dataset_frac)
    test_set_size = int(len(dataset_orig) * test_dataset_frac)
    robots_set_size = len(dataset_orig) - test_set_size - cloud_set_size

    robot_datasets_dists = np.zeros((num_robots, num_classes))
    for i in range(num_robots):
        for j in range(num_classes):
            v = augment_value_select(
                0, 1, 0.5,
                aggro=robot_class_imbalance_aggro,
                rng=rng)
            robot_datasets_dists[i, j] = v
    robot_datasets_dists /= robot_datasets_dists.sum(axis=1).reshape(-1, 1)

    # The cloud and test dataset contains data unique to that set, so the
    # datasets exactly match the chosen distributions. On the other hand, the
    # robot data contains some data unique to each robot, but also many data
    # samples that are common between robots.

    # Here we sample unique data indices
    # order of devices: cloud, test, robot * num_robots
    device_num_unique_samples = np.zeros((num_robots + 2, num_classes))
    device_num_unique_samples[0] = cloud_dataset_dist * cloud_set_size
    device_num_unique_samples[1] = test_dataset_dist * test_set_size
    device_num_unique_samples[2:] = robot_datasets_dists * robots_set_size * robot_dataset_unique_frac / num_robots
    for i in range(num_robots):
        device_num_unique_samples[i] = iteround.saferound(device_num_unique_samples[i], places=0)
    device_num_unique_samples = device_num_unique_samples.astype(np.int32)
    for i in range(num_classes):
        assert sum(device_num_unique_samples[:, i]) <= dataset_orig_hist[i], \
            "Not enough data to match requested distribution settings"

    unique_samples_per_class = [
        rng.choice(np.where(labels == i)[0], k)
        for i, k in enumerate(device_num_unique_samples.sum(axis=0))]
    robot_common_samples = set(np.arange(len(dataset_orig))) - \
        set(sum((n.tolist() for n in unique_samples_per_class), []))
    robot_common_samples = np.array(list(robot_common_samples))

    # indexed first by class, then by robot
    device_samples = [
        np.split(unique_samples_per_class[i], np.cumsum(device_num_unique_samples[:, i]))
        for i in range(num_classes)]
    # indexed first by robot, then by class
    device_samples = [
        np.concatenate([device_samples[i][j] for i in range(num_classes)])
        for j in range(num_robots + 2)]
    for i in range(2, num_robots + 2):
        device_samples[i] = np.concatenate((device_samples[i], robot_common_samples))

    device_sample_info = {
        'cloud': device_samples[0],
        'test': device_samples[1],
        'robots': device_samples[2:]
    }

    device_class_dist_info = {
        'cloud': cloud_dataset_dist,
        'test': test_dataset_dist,
        'robots': robot_datasets_dists
    }

    datasets_raw = {
        'cloud': Subset(dataset_orig, device_sample_info['cloud']),
        'test': Subset(dataset_orig, device_sample_info['test']),
        'robots': [
            Subset(dataset_orig, device_sample_info['robots'][i])
            for i in range(num_robots)]
    }
    datasets_raw['robots_augmented'] = [
        TransformDataset(
            dataset,
            transform_with_label(
                generate_robot_img_transform(
                    dataset_type=dataset_type,
                    aggro=robot_augmentation_aggro,
                    rng=rng)))
        for dataset in datasets_raw['robots']]

    robot_data_samplers = [
        RebalancedCategoricalSampler(
            labels=[y for (x, y) in datasets_raw['robots'][i]],
            num_classes=num_classes,
            target_dist=device_class_dist_info['robots'][i],
            target_dataset_size=1)
        for i in range(num_robots)]

    robot_data_streamers = [
        RobotDataStreamer(d, s)
        for d, s in zip(datasets_raw['robots_augmented'], robot_data_samplers)]

    out = {
        'dataset_orig': dataset_orig,
        'sample_indices': device_sample_info,
        'class_dists': device_class_dist_info,
        'datasets_raw': datasets_raw,
        'robot_data_samplers': robot_data_samplers,
        'robot_data_augmented_streamers': robot_data_streamers
    }

    return out
