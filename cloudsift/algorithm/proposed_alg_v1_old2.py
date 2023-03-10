# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:10:41 2022

@author: Shahir
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader, default_collate

import torch.types

from torchvision.models import mobilenet_v3_small

from tqdm.auto import tqdm
from PIL.Image import Image

from cloudsift.datasets import (
    RawDataset,
    ImageDatasetType,
    RebalancedCategoricalSampler,
    get_img_train_test_transforms,
    IMG_DATASET_TO_IMG_SHAPE,
    IMG_DATASET_TO_CLASS_NAMES)
from cloudsift.simulation import (
    Robot,
    BaseRobotPolicy,
    RobotIterationResult,
    RobotInitialPolicyInfo,
    RobotPolicyUpdateInfo,
    BaseCloudComputer,
    CloudProcessingResult,
    CloudTrainingUpdateInfo,
    BaseCloudLabeler,
    AutomaticCloudLabeler,
    LabelingResult)
from cloudsift.frontend.labeling_client import ManualCategoricalImageCloudLabeler
from cloudsift.algorithm.models.resnet_utils import ResNetBuilder
from cloudsift.training import make_optimizer, make_scheduler, set_optimizer_lr, shrink_and_preturb
from cloudsift.utils import BoolArrayLike1D, IntArrayLike1D
from cloudsift.algorithm.utils import calc_model_size


class Decoder(ResNetBuilder):
    LATENT_SHAPE = (40, 2, 2)

    def __init__(
            self,
            out_channels: int) -> None:

        super().__init__()

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest')

        # 2 x 2
        self.conv1 = nn.Conv2d(self.LATENT_SHAPE[0], 20, kernel_size=3, stride=1, padding=1, bias=True)

        # 4 x 4
        self.layer1 = self._make_layer(20, 16, 5)

        # 8 x 8
        self.layer2 = self._make_layer(16, 12, 4)

        # 16 x 16
        self.layer3 = self._make_layer(12, 8, 3)

        # 32 x 32
        self.layer4 = self._make_layer(8, out_channels, 2)

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:

        x = F.leaky_relu(self.conv1(x))

        x = self.upsample(x)
        x = self.layer1(x)

        x = self.upsample(x)
        x = self.layer2(x)

        x = self.upsample(x)
        x = self.layer3(x)

        x = self.upsample(x)
        x = self.layer4(x)

        return x


class ActionPredictor(nn.Module):
    @staticmethod
    def _power_floor_log2(
            x: Union[float, npt.NDArray[np.floating]]) -> npt.NDArray[np.int32]:

        return (2 ** (np.floor(np.log2(x)))).astype(np.int32)

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int = 4) -> None:

        super().__init__()

        layer_dims = np.maximum(
            self._power_floor_log2(
                np.linspace(
                    input_dim,
                    output_dim,
                    num_layers + 1,
                    dtype=int))[1:-1], 24).astype(int).tolist()

        layers = []
        prev_dim = input_dim
        for next_dim in layer_dims:
            layers.append(nn.Linear(prev_dim, next_dim, bias=True))
            layers.append(nn.LeakyReLU(inplace=True))
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:

        return self.layers(torch.flatten(x, 1))


class RobotImageModel(nn.Module):
    def __init__(
            self,
            img_shape: tuple[int],
            num_classes: int) -> None:

        super().__init__()

        assert img_shape == (3, 32, 32)
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.full_classifier = mobilenet_v3_small(num_classes=self.num_classes)
        self.encoder = self.full_classifier.features[:5]
        self.classifier_head = nn.Sequential(
            self.full_classifier.features[5:],
            self.full_classifier.avgpool,
            nn.Flatten(1),
            self.full_classifier.classifier,
        )
        self.latent_dim = 40 * 2 * 2
        self.decoder_head = Decoder(out_channels=3)

        self.compression_selector = ActionPredictor(
            input_dim=(self.latent_dim + self.num_classes),
            output_dim=1,
            num_layers=8)
        self.importance_predictor = ActionPredictor(
            input_dim=(self.latent_dim + self.num_classes),
            output_dim=1,
            num_layers=8)

    def forward(
            self,
            x: torch.Tensor,
            is_latent: bool = False) -> dict[str, torch.Tensor]:

        if is_latent:
            latents = x
        else:
            latents = self.encoder(x)
        y_preds = self.classifier_head(latents)
        x_decodeds = self.decoder_head(latents)
        action_net_inputs = torch.cat([torch.flatten(latents, 1), y_preds], dim=1).detach()
        compress_outs = self.compression_selector(action_net_inputs)
        importance_outs = self.importance_predictor(action_net_inputs)

        out = {
            'latents': latents,
            'y_preds': y_preds,
            'action_net_inputs': action_net_inputs,
            'x_decodeds': x_decodeds,
            'compress_outs': compress_outs,
            'importance_outs': importance_outs
        }

        return out


@dataclass(frozen=True)
class UnseenRawDataSample:
    img_raw: Image
    label: int


@dataclass(frozen=True)
class UploadDataPoint:
    compressed: bool
    img_raw: Image
    latent: torch.FloatTensor
    simulation_metadata: UnseenRawDataSample


@dataclass(frozen=True)
class CacheEntry:
    upload_data_point: UploadDataPoint
    importance: float


class RobotPolicy(BaseRobotPolicy):
    def __init__(
            self,
            robot: Robot,
            img_shape: tuple[int],
            num_classes: int,
            test_transform: Callable[[Image], torch.Tensor],
            device: torch.device,
            model: Optional[nn.Module] = None) -> None:

        super().__init__(robot)

        self.img_shape = img_shape
        self.num_classes = num_classes
        if model:
            self.model = model
        else:
            self.model = RobotImageModel(
                img_shape=self.img_shape,
                num_classes=self.num_classes)
        self.test_transform = test_transform
        self.device = device
        self.model = self.model.to(self.device)

        self._current_sample_index = 0
        self._sample_heap = []

    def update_policy(
            self,
            update_info: dict) -> None:

        self.model.load_state_dict(update_info)

    def classify_data_samples(
            self,
            input_imgs_raw: list[Image]) -> torch.Tensor:

        self.model.eval()
        input_imgs = [self.test_transform(n) for n in input_imgs_raw]
        input_img_batched = torch.stack(input_imgs).to(self.device)
        with torch.set_grad_enabled(False):
            outputs = self.model(input_img_batched)

        _, preds = torch.max(outputs['y_preds'], dim=1)

        return preds

    def _process_data(
            self,
            img_raw: Image,
            unseen_label: int,
            model_outputs: dict[str, torch.Tensor]) -> RobotIterationResult:

        unseen_label = int(unseen_label)  # in case it is a numpy int
        compressed = int(round(torch.sigmoid(model_outputs['compress_outs']).item()))
        compressed = False  # TODO: undo this
        current_sample_importance = model_outputs['importance_outs'][0].item()

        # 4 byte floats for latent, or 1 byte color values
        # + 1 byte for indicating whether this is compressed or not
        # + 4 byte float for importance value
        pending_cache_entry_size = self._get_img_size(compressed) + 1 + 4

        unseen_orig_sample = UnseenRawDataSample(
            img_raw=img_raw,
            label=unseen_label)
        upload_data_point = UploadDataPoint(
            compressed=compressed,
            img_raw=(None if compressed else img_raw),
            latent=(model_outputs['latents'] if compressed else None),
            simulation_metadata=unseen_orig_sample)
        cache_entry = CacheEntry(
            upload_data_point=upload_data_point,
            importance=current_sample_importance)

        self._sample_heap.sort()
        samples_popped = 0
        space_needed = pending_cache_entry_size - self.robot.upload_cache.get_bytes_left()
        space_to_remove = 0
        add_to_heap = False
        while True:
            if space_to_remove >= space_needed:
                add_to_heap = True
                break
            if samples_popped >= len(self._sample_heap):
                break
            worst_cache_entry_importance, worst_sample_index = self._sample_heap[samples_popped]
            if current_sample_importance > worst_cache_entry_importance:
                space_to_remove += self.robot.upload_cache.get_bytes_used_key(worst_sample_index)
            else:
                break
            samples_popped += 1

        if add_to_heap:
            popped_elems = self._sample_heap[:samples_popped]
            self._sample_heap = self._sample_heap[samples_popped:]
            for _, index in popped_elems:
                self.robot.upload_cache.pop(index)
            self.robot.upload_cache.push(self._current_sample_index, cache_entry, pending_cache_entry_size)

        _, preds = torch.max(model_outputs['y_preds'], dim=0)
        self._current_sample_index += 1

        return RobotIterationResult(preds.item())

    def process_next_input_sample(
            self,
            input_img_raw: Image,
            unseen_orig_label: int) -> RobotIterationResult:

        self.model.eval()
        input_img_batched = self.test_transform(input_img_raw)
        input_img_batched = input_img_batched.to(self.device).unsqueeze(0)
        with torch.set_grad_enabled(False):
            outputs = self.model(input_img_batched)
            for key in outputs.keys():
                outputs[key] = outputs[key].squeeze(0).cpu()

        result = self._process_data(
            img_raw=input_img_raw,
            unseen_label=unseen_orig_label,
            model_outputs=outputs)

        return result

    def process_next_input_sample_batched(
            self,
            input_imgs_raw: list[Image],
            unseen_orig_labels: IntArrayLike1D) -> list[RobotIterationResult]:

        self.model.eval()

        input_imgs_batch = []
        for input_img_raw in input_imgs_raw:
            input_img = self.test_transform(input_img_raw)
            input_imgs_batch.append(input_img)
        input_imgs_batch = torch.stack(input_imgs_batch, dim=0)
        input_imgs_batch = input_imgs_batch.to(self.device)

        with torch.set_grad_enabled(False):
            all_sample_outputs = self.model(input_imgs_batch)
            for key in all_sample_outputs.keys():
                all_sample_outputs[key] = all_sample_outputs[key].cpu()

        results = []
        for i, (input_img_raw, unseen_label) in enumerate(zip(input_imgs_raw, unseen_orig_labels)):
            sample_model_outputs = {}
            for key in all_sample_outputs.keys():
                sample_model_outputs[key] = all_sample_outputs[key][i]
            result = self._process_data(
                img_raw=input_img_raw,
                unseen_label=unseen_label,
                model_outputs=sample_model_outputs)
            results.append(result)

        return results

    def _importance_sort_key(
            self,
            item: int) -> float:

        index, cache_entry = item
        return cache_entry.importance

    def _get_img_size(
            self,
            compressed: bool) -> int:

        return int(self.model.latent_dim * 4 if compressed else np.prod(self.model.img_shape))

    def get_upload_payload(
            self) -> tuple[list[UploadDataPoint], int]:

        bytes_uploaded = 0
        data_points = []
        for cache_entry in self.robot.upload_cache.values():
            data_point = cache_entry.upload_data_point
            bytes_uploaded += (self._get_img_size(data_point.compressed) + 1)
            data_points.append(data_point)

        return data_points, bytes_uploaded

    def finish_round(
            self) -> None:

        self._sample_heap.clear()
        self.robot.upload_cache.clear()


@dataclass(frozen=True)
class CloudDataPoint:
    compressed: bool
    img_raw: Image
    img_decoded: torch.FloatTensor
    latent: torch.FloatTensor
    labeling_success: bool


class ExpandingCloudDataset(Dataset[tuple[CloudDataPoint, int]]):
    def __init__(
            self) -> None:

        self.x_data = []
        self.y_data = []

    def __getitem__(
            self,
            index: int) -> tuple[CloudDataPoint, int]:

        return (self.x_data[index], self.y_data[index])

    def __len__(
            self) -> int:

        return len(self.x_data)

    def append(
            self,
            x_data: CloudDataPoint,
            y_data: int) -> None:

        self.x_data.append(x_data)
        self.y_data.append(y_data)


class ReconstructionImageCloudLabeler(BaseCloudLabeler[int]):
    def __init__(
            self,
            mse_threshold: float = 0.1) -> None:

        self.num_labels_requested = 0
        self.mse_threshold = mse_threshold

    def request_labels(
            self,
            using_orig_data: BoolArrayLike1D,
            imgs_to_label: list[torch.Tensor],
            orig_imgs: list[torch.Tensor],
            unseen_orig_labels: IntArrayLike1D) -> list[LabelingResult[int]]:

        super().request_labels(len(imgs_to_label))

        mask = torch.tensor(using_orig_data).bool()
        unseen_orig_labels = torch.tensor(unseen_orig_labels)
        imgs_to_label = torch.concatenate(imgs_to_label)
        orig_imgs = torch.concatenate(orig_imgs)

        self.num_labels_requested += 1
        with torch.set_grad_enabled(False):
            mse_per_decoded_img = F.mse_loss(
                orig_imgs[~mask],
                imgs_to_label[~mask],
                reduction='none').sum((1, 2, 3)).cpu().item()
            mse_below_threshold = mse_per_decoded_img < self.mse_threshold

        labels = torch.full((len(imgs_to_label)), -100, dtype=torch.int32)
        labels[mask] = unseen_orig_labels[mask]
        labels[~mask][mse_below_threshold] = unseen_orig_labels[~mask][mse_below_threshold]
        labels = labels.detach().cpu().numpy()
        labeling_successes = (labels != -100).astype(np.bool_)

        out = [(bool(v), (int(l) if v else None)) for (l, v) in zip(labels, labeling_successes)]

        return out


class LabelingMethod(Enum):
    AUTOMATIC = 0
    RECONSTRUCTION_LOSS = 1
    MANUAL = 2


class CloudComputer(BaseCloudComputer):
    def __init__(
            self,
            *,
            dataset_type: ImageDatasetType,
            num_robots: int,
            num_classes: int,
            cloud_init_dataset: Dataset[tuple[Image, int]],
            labeling_method: LabelingMethod,
            device: torch.device,
            lambda_reconstruction: float = 3.0,
            lambda_compression: float = 1.0,
            lambda_importance: float = 0.2,
            init_lr: float = 0.001,
            reconstruction_valid_threshold: float = 0.10,
            num_pretraining_epochs: int = 50,
            max_num_main_epochs_per_round: int = 20,
            num_action_train_epochs_per_round: int = 10,
            round_early_stop_acc: float = 0.9,
            batch_size: int = 128) -> None:

        self.labeling_method = labeling_method
        if labeling_method == LabelingMethod.AUTOMATIC:
            label_allocator = AutomaticCloudLabeler()
        elif labeling_method == LabelingMethod.RECONSTRUCTION_LOSS:
            label_allocator = ReconstructionImageCloudLabeler()
        elif labeling_method == LabelingMethod.MANUAL:
            label_allocator = ManualCategoricalImageCloudLabeler(
                class_names=IMG_DATASET_TO_CLASS_NAMES[dataset_type],
                img_display_size=IMG_DATASET_TO_IMG_SHAPE[dataset_type][1])

        super().__init__(num_robots, label_allocator)

        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.transforms = get_img_train_test_transforms(
            dataset_type=dataset_type,
            new_input_size=(32, 32))
        self.cloud_dataset = ExpandingCloudDataset()
        self.cloud_init_dataset = cloud_init_dataset

        self.img_shape = IMG_DATASET_TO_IMG_SHAPE[self.dataset_type]
        self.common_policy_model = None
        self.robot_policies = None
        self.robot_policy_size_estimates = None

        self.optimizer = None
        self.lr_scheduler = None

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_compression = lambda_compression
        self.lambda_importance = lambda_importance
        self.decode_valid_threshold = reconstruction_valid_threshold

        self.num_pretraining_epochs = num_pretraining_epochs
        self.max_num_main_epochs_per_round = max_num_main_epochs_per_round
        self.round_early_stop_acc = round_early_stop_acc
        self.num_action_train_epochs_per_round = num_action_train_epochs_per_round
        self.batch_size = batch_size

        self.init_lr = init_lr

        self.device = device
    
    def initialize(
            self) -> None:
        
        if self.labeling_method == LabelingMethod.MANUAL:
            self.label_allocator.initialize()

    def get_init_robot_policies(
            self,
            robots: list[Robot]) -> RobotInitialPolicyInfo:

        for (input_img_raw, label) in self.cloud_init_dataset:
            self._add_data_point(
                compressed=False,
                img_raw=input_img_raw,
                unseen_orig_label=label)

        self._reset_new_round()
        for i in range(self.num_pretraining_epochs):
            print("Pretraining epoch {}".format(i + 1))
            print("-" * 20)
            self._train_policy()

        self.robot_policies = [
            RobotPolicy(
                robot=robot,
                img_shape=self.img_shape,
                num_classes=self.num_classes,
                test_transform=self.transforms['test'],
                device=self.device,
                model=self.common_policy_model)
            for robot in robots]

        robot_policy_sizes = [int(calc_model_size(p.model)) for p in self.robot_policies]
        result = RobotInitialPolicyInfo(
            robot_policies=self.robot_policies,
            robot_bytes_downloaded=robot_policy_sizes)

        return result

    def _batch_collate(
            self,
            batch: list[tuple[CloudDataPoint, int]]) -> dict[str, Union[Any, None]]:

        compressed_batch = []
        uncompressed_batch = []
        for cloud_input_data_point, label in batch:
            compressed = cloud_input_data_point.compressed
            if compressed:
                compressed_batch.append((
                    cloud_input_data_point.latent,
                    cloud_input_data_point.img_decoded,
                    cloud_input_data_point.labeling_success,
                    label))
            else:
                uncompressed_batch.append((
                    self.transforms['train'](cloud_input_data_point.img_raw),
                    label))

        # batchnorm in requires batch size of greater than 1
        if len(uncompressed_batch) == 1:
            uncompressed_batch = []
        if len(compressed_batch) == 1:
            compressed_batch = []

        out = {
            'compressed': (
                default_collate(compressed_batch)
                if compressed_batch else None),
            'uncompressed': (
                default_collate(uncompressed_batch)
                if uncompressed_batch else None)
        }

        return out

    def _reset_new_round(
            self,
            verbose: bool = True) -> None:

        if verbose:
            print("Resetting optimizer parameters")
        new_model = RobotImageModel(
            img_shape=self.img_shape,
            num_classes=self.num_classes).to(self.device)
        if self.common_policy_model is not None:
            if verbose:
                print("Running shrink and preturb")
            new_model = shrink_and_preturb(
                base_net=self.common_policy_model,
                new_net=new_model)
        self.common_policy_model = new_model
        self.optimizer = make_optimizer(self.common_policy_model.parameters(), lr=self.init_lr)

    def _reset_lr_and_get_scheduler(
            self) -> optim.lr_scheduler.ReduceLROnPlateau:

        set_optimizer_lr(self.optimizer, self.init_lr)
        scheduler = make_scheduler(self.optimizer)

        return scheduler

    def _get_classification_losses(
            self) -> torch.Tensor:

        loss_datas = []

        self.common_policy_model.eval()

        for compressed in [True, False]:
            subset_indices = np.array([
                i for i in range(len(self.cloud_dataset))
                if int(self.cloud_dataset[i][0].compressed) == compressed],
                dtype=np.int32)

            cloud_dataset_subset = Subset(
                self.cloud_dataset,
                subset_indices)
            dataloader = DataLoader(
                cloud_dataset_subset,
                batch_size=self.batch_size,
                num_workers=0,
                collate_fn=self._batch_collate,
                pin_memory=False,
                drop_last=False)

            for batch in dataloader:
                with torch.set_grad_enabled(False):
                    sub_batches = []
                    if batch['compressed'] is not None:
                        _, imgs_decoded, mask, labels = batch['compressed']
                        imgs = imgs_decoded[mask].to(self.device)
                        sub_batches.append((imgs, labels[mask].to(self.device)))

                    if batch['uncompressed'] is not None:
                        input_imgs, labels = batch['uncompressed']
                        input_imgs = input_imgs.to(self.device)
                        sub_batches.append((input_imgs, labels.to(self.device)))

                    for imgs, labels in sub_batches:
                        outputs = self.common_policy_model(imgs, is_latent=False)
                        y_preds = outputs['y_preds']
                        losses = F.cross_entropy(
                            y_preds,
                            labels,
                            reduction='none')
                        loss_datas.append(losses)

        losses = torch.concatenate(
            loss_datas,
            dim=0)

        return losses.detach()
    
    def _train_task_nets_single_epoch(
            self,
            dataloader_train: DataLoader) -> tuple[float, float, dict[str, Any]]:
        
        self.common_policy_model.train()
        
        running_count = 0
        running_correct = 0
        running_loss = 0.0
        training_accuracy = None
        training_loss = None

        pbar = tqdm(dataloader_train)
        for batch in pbar:
            self.optimizer.zero_grad()

            loss_components = []
            batch_y_pred_datas = []

            if batch['compressed'] is not None:
                latents, imgs_decoded, labeling_successes, labels = batch['compressed']
                latents = latents.to(self.device)
                imgs_decoded = imgs_decoded.to(self.device)
                labeling_successes = labeling_successes.to(self.device)
                labels = labels.to(self.device)

                # Classification: we cannot train on the all of the data since if decoder fails,
                # we do not have the original data point, only the compressed latent space
                # Importance: we shouldn't train importance on a data sample that wasn't meaningful
                labeling_successes = labeling_successes.bool()  # success
                mask_total = labeling_successes.sum()

                with torch.set_grad_enabled(True):
                    outputs = self.common_policy_model(imgs_decoded, is_latent=False)

                    if mask_total:
                        classification_loss = F.cross_entropy(
                            outputs['y_preds'][labeling_successes],
                            labels[labeling_successes])

                    else:
                        classification_loss = torch.tensor(0, device=self.device).detach()

                    loss_components.append(classification_loss)

                    batch_y_pred_datas.append((outputs['y_preds'], labels, labeling_successes))

            if batch['uncompressed'] is not None:
                input_imgs, labels = batch['uncompressed']
                input_imgs = input_imgs.to(self.device)
                labels = labels.to(self.device)

                with torch.set_grad_enabled(True):
                    outputs = self.common_policy_model(input_imgs, is_latent=False)

                    classification_losses = F.cross_entropy(
                        outputs['y_preds'],
                        labels,
                        reduction='none')
                    reconstruction_losses = F.mse_loss(
                        outputs['x_decodeds'],
                        input_imgs,
                        reduction='none').mean(dim=(1, 2, 3))

                    reconstruction_loss = reconstruction_losses.mean()
                    classification_loss = classification_losses.mean()

                    loss_components.append(classification_loss)
                    loss_components.append(reconstruction_loss * self.lambda_reconstruction)  # TODO: undo this

                    # We can still train on the all of the data since even if the autoencoder failed,
                    # we have the original data point
                    labeling_successes = (labels != -100)  # success (should be all True)
                    batch_y_pred_datas.append((outputs['y_preds'], labels, labeling_successes))

            with torch.set_grad_enabled(True):
                total_loss = sum(loss_components)
                total_loss.backward()
                self.optimizer.step()

            for y_preds, labels, labeling_successes in batch_y_pred_datas:
                _, preds = torch.max(y_preds, dim=1)
                correct = torch.sum((preds == labels)[labeling_successes]).item()
                num_samples = labels[labeling_successes].size(0)
                running_count += num_samples
                running_correct += correct
                running_loss += total_loss.detach().item() * num_samples

            training_accuracy = running_correct / running_count
            training_loss = running_loss / running_count
            loss_components = [n.detach().item() for n in loss_components]
            pbar.set_description(
                "[Training Task Nets] Avg. Loss: {:.4f}, Components: {}, Acc.: {:.4f}".format(
                    training_loss,
                    ", ".join(["{:.4f}".format(n) for n in loss_components]),
                    training_accuracy))
        pbar.close()
        
        stats = {
            'acc': training_accuracy,
            'loss_avg': training_loss
        }
        
        return training_accuracy, training_loss, stats
    
    def _get_action_nets_training_data(
            self) -> tuple[DataLoader, float, float]:
        
        model_init_training = self.common_policy_model.training
        
        stage2_datas = {
            'masks': [],
            'labeling_successes': [],
            'classification_losses': [],
            'action_net_inputs': [],
            'labels': []
        }

        self.common_policy_model.eval()

        for compressed in [True, False]:
            subset_indices = np.array([
                i for i in range(len(self.cloud_dataset))
                if int(self.cloud_dataset[i][0].compressed) == compressed],
                dtype=np.int32)

            cloud_dataset_subset = Subset(
                self.cloud_dataset,
                subset_indices)

            dataloader_subset = DataLoader(
                cloud_dataset_subset,
                batch_size=self.batch_size,
                num_workers=0,
                collate_fn=self._batch_collate,
                pin_memory=False,
                drop_last=False)

            # pbar = tqdm(dataloader_subset)
            for batch in dataloader_subset:
                if batch['compressed'] is not None:
                    latents, imgs_decoded, labeling_successes, labels = batch['compressed']
                    imgs_decoded = imgs_decoded.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(False):
                        outputs = self.common_policy_model(imgs_decoded, is_latent=False)

                        y_preds = outputs['y_preds']
                        classification_losses = F.cross_entropy(
                            y_preds,
                            labels,
                            reduction='none')

                    stage2_datas['masks'].append(labeling_successes.detach().cpu())
                    stage2_datas['labeling_successes'].append(labeling_successes.detach().cpu())
                    stage2_datas['classification_losses'].append(classification_losses.detach().cpu())
                    stage2_datas['action_net_inputs'].append(outputs['action_net_inputs'].detach().cpu())
                    stage2_datas['labels'].append(labels.detach().cpu())

                if batch['uncompressed'] is not None:
                    input_imgs, labels = batch['uncompressed']
                    input_imgs = input_imgs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(False):
                        outputs = self.common_policy_model(input_imgs, is_latent=False)

                        y_preds = outputs['y_preds']
                        classification_losses = F.cross_entropy(
                            y_preds,
                            labels,
                            reduction='none')

                        reconstruction_losses = F.mse_loss(
                            outputs['x_decodeds'],
                            input_imgs,
                            reduction='none').mean(dim=(1, 2, 3))

                        masks = (labels != -100)
                        labeling_successes = reconstruction_losses < self.decode_valid_threshold

                    stage2_datas['masks'].append(masks.detach().cpu())
                    stage2_datas['labeling_successes'].append(labeling_successes.detach().cpu())
                    stage2_datas['classification_losses'].append(classification_losses.detach().cpu())
                    stage2_datas['action_net_inputs'].append(outputs['action_net_inputs'].detach().cpu())
                    stage2_datas['labels'].append(labels.detach().cpu())

            # pbar.close()

        for key in stage2_datas.keys():
            stage2_datas[key] = torch.concatenate(stage2_datas[key], dim=0)

        masked_losses = stage2_datas['classification_losses'][stage2_datas['masks']]
        importance_loss_min = masked_losses.min().item()
        importance_loss_max = masked_losses.max().item()
        importance_loss_range = importance_loss_max - importance_loss_min + 1e-9

        stage2_dataset_x_data = []
        stage2_dataset_y_data = []
        for i in range(len(self.cloud_dataset)):
            x = {k: v[i].numpy() for k, v in stage2_datas.items() if k != 'labels'}
            y = stage2_datas['labels'][i]
            stage2_dataset_x_data.append(x)
            stage2_dataset_y_data.append(y.numpy())

        stage2_dataset = RawDataset(stage2_dataset_x_data)  # labels are not actually needed for stage 2 training

        balanced_sampler = RebalancedCategoricalSampler(
            labels=stage2_dataset_y_data,
            num_classes=self.num_classes,
            target_dist='uniform',
            target_dataset_size=(len(stage2_dataset) * 2))

        dataloader_train = DataLoader(
            stage2_dataset,
            batch_size=self.batch_size,
            sampler=balanced_sampler,
            num_workers=0,
            pin_memory=False)
        
        self.common_policy_model.train(model_init_training)
        
        return dataloader_train, importance_loss_min, importance_loss_range
    
    def _train_action_nets(
            self,
            num_epochs: int) -> dict[str, Any]:
        
        scheduler = self._reset_lr_and_get_scheduler()       
        dataloader_train, importance_loss_min, importance_loss_range = self._get_action_nets_training_data()
        self.common_policy_model.train()
        
        for i in range(num_epochs):
            running_loss = 0
            running_count = 0
            training_loss = None

            pbar = tqdm(dataloader_train)
            for batch in pbar:
                self.optimizer.zero_grad()

                for key, value in batch.items():
                    batch[key] = value.to(self.device)

                loss_components = []

                with torch.set_grad_enabled(True):
                    compress_outs = self.common_policy_model.compression_selector(batch['action_net_inputs'])
                    importance_outs = self.common_policy_model.importance_predictor(batch['action_net_inputs'])

                    compression_loss = F.binary_cross_entropy_with_logits(
                        compress_outs[:, 0],
                        batch['labeling_successes'].float())

                    imp_loss_target = batch['classification_losses'][batch['masks']] - importance_loss_min
                    imp_loss_target /= importance_loss_range
                    imp_loss_target = torch.clamp(imp_loss_target, 0, 1)
                    importance_loss = F.binary_cross_entropy_with_logits(
                        importance_outs[batch['masks']][:, 0],
                        imp_loss_target)

                    loss_components.append(compression_loss * self.lambda_compression)
                    loss_components.append(importance_loss * self.lambda_importance)

                    total_loss = sum(loss_components)
                    total_loss.backward()
                    self.optimizer.step()

                num_samples = batch['masks'].size(0)
                running_count += num_samples
                running_loss += total_loss.detach().item() * num_samples
                training_loss = running_loss / running_count
                loss_components = [n.detach().item() for n in loss_components]
                pbar.set_description(
                    "[Training Action Nets] Avg. Loss: {:.4f}, Components: {}".format(
                        training_loss,
                        ", ".join(["{:.4f}".format(n) for n in loss_components])))

            scheduler.step(training_loss)
            pbar.close()
        
        stats = {
            'loss_avg': training_loss
        }
        
        return stats
    
    def _train_policy(
            self) -> CloudProcessingResult:
        
        print("Starting training round")

        balanced_sampler = RebalancedCategoricalSampler(
            labels=self.cloud_dataset.y_data,
            num_classes=self.num_classes,
            target_dist='uniform',
            target_dataset_size=(len(self.cloud_dataset) * 2))

        dataloader_train = DataLoader(
            self.cloud_dataset,
            batch_size=self.batch_size,
            sampler=balanced_sampler,
            num_workers=0,
            collate_fn=self._batch_collate,
            pin_memory=False)

        self.common_policy_model.train()

        scheduler = self._reset_lr_and_get_scheduler()

        for i in range(self.max_num_main_epochs_per_round):
            print("Epoch {}".format(i + 1))
            acc, loss, task_nets_stats = self._train_task_nets_single_epoch(dataloader_train)
            action_nets_stats = self._train_action_nets(1)
            if acc >= self.round_early_stop_acc:
                break
            scheduler.step(loss)

        unique_classes, unique_class_sample_count = np.unique(self.cloud_dataset.y_data, return_counts=True)
        dataset_dist = np.zeros(self.num_classes, dtype=np.float32)
        dataset_dist[unique_classes] = unique_class_sample_count
        dataset_dist = (dataset_dist / len(self.cloud_dataset)).tolist()

        training_stats = {
            'task_nets_stats': task_nets_stats,
            'action_nets_stats': action_nets_stats
        }
        result = CloudProcessingResult(
            dataset_dist=dataset_dist,
            training_stats=training_stats,
            num_labels_requested=self.label_allocator.num_labels_requested)

        return result

    def _add_data_point(
            self,
            *,
            compressed: bool,
            latent: Optional[torch.Tensor] = None,
            img_raw: Optional[Image] = None,
            unseen_orig_img_raw: Optional[Image] = None,
            unseen_orig_label: int) -> None:

        labeling_out = None

        if compressed:
            assert latent is not None
            assert unseen_orig_img_raw is not None
            assert img_raw is None

            if self.labeling_method == LabelingMethod.AUTOMATIC:
                labeling_out = self.label_allocator.request_labels(
                    unseen_orig_labels=[unseen_orig_label])[0]
            elif self.labeling_method == LabelingMethod.RECONSTRUCTION_LOSS:
                unseen_orig_img = self.transforms['test'](unseen_orig_img_raw).to(self.device)
                latent_batched = latent.to(self.device).unsqueeze(0)
                with torch.set_grad_enabled(False):
                    img_decoded = self.common_policy_model.decoder_head(latent_batched).squeeze(0)
                labeling_out = self.label_allocator.request_labels(
                    using_orig_data=False,
                    imgs_to_label=[img_decoded],
                    orig_imgs=[unseen_orig_img],
                    unseen_orig_labels=unseen_orig_label)[0]
            else:
                labeling_out = self.label_allocator.request_labels(
                    imgs_to_label=[self.transforms['from_tensor'](img_decoded)])[0]

            img_decoded = img_decoded.cpu()
            latent = latent.detach().cpu()

        else:
            assert img_raw is not None
            assert latent is None
            img_decoded = None

            if self.labeling_method == LabelingMethod.AUTOMATIC:
                labeling_out = self.label_allocator.request_labels(
                    unseen_orig_labels=[unseen_orig_label])[0]
            elif self.labeling_method == LabelingMethod.RECONSTRUCTION_LOSS:
                labeling_out = self.label_allocator.request_labels(
                    using_orig_data=True,
                    imgs_to_label=[self.transforms['test'](img_raw)],
                    orig_imgs=[self.transforms['test'](unseen_orig_img_raw)],
                    unseen_orig_labels=unseen_orig_label)[0]
            else:
                labeling_out = self.label_allocator.request_labels(
                    imgs_to_label=[img_raw])[0]

        labeling_success = labeling_out.labeling_success
        label = labeling_out.label
        self.cloud_dataset.append(
            CloudDataPoint(
                compressed=0.0,
                img_raw=img_raw,
                img_decoded=img_decoded,
                latent=latent,
                labeling_success=labeling_success),
            label if labeling_success else -100)

    def process_round(
            self,
            update_info: CloudTrainingUpdateInfo[list[UploadDataPoint]]) -> CloudProcessingResult:

        compressed_new_count = 0
        total_new_count = 0
        for i in range(self.num_robots):
            upload_data_points = update_info.robot_upload_payloads[i]
            for upload_data_point in tqdm(upload_data_points):
                compressed = upload_data_point.compressed
                unseen_orig_label = upload_data_point.simulation_metadata.label

                if compressed:
                    self._add_data_point(
                        compressed=compressed,
                        latent=upload_data_point.latent,
                        unseen_orig_img_raw=upload_data_point.simulation_metadata.img_raw,
                        unseen_orig_label=unseen_orig_label)
                else:
                    self._add_data_point(
                        compressed=compressed,
                        img_raw=upload_data_point.img_raw,
                        unseen_orig_label=unseen_orig_label)

                compressed_new_count += compressed
                total_new_count += 1

        compressed_frac = (compressed_new_count / total_new_count) if total_new_count else 0
        print("{} new data samples".format(total_new_count))
        print("{:.4f} of new samples were compressed".format(compressed_frac))
        print("Cloud dataset size: {}".format(len(self.cloud_dataset)))

        self._reset_new_round()
        result = self._train_policy()

        return result

    def get_robot_policy_updates(
            self) -> RobotPolicyUpdateInfo[dict[str, torch.Tensor]]:

        policy_update_info = [p.model.state_dict() for p in self.robot_policies]
        bytes_downloaded = [calc_model_size(p.model) for p in self.robot_policies]

        return RobotPolicyUpdateInfo(policy_update_info, bytes_downloaded)
