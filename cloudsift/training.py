# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:09:47 2022

@author: Shahir
"""

import os
import pickle
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from cloudsift.utils import get_timestamp_str


class ModelTracker:
    def __init__(
            self,
            root_dir: str) -> None:

        experiment_dir = "Experiment {}".format(get_timestamp_str())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float('-inf')
        self.record_per_epoch = {}

    def update_info_history(
            self,
            epoch: int,
            info: Any) -> None:

        os.makedirs(self.save_dir, exist_ok=True)
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), 'wb') as f:
            pickle.dump(self.record_per_epoch, f)

    def update_model_weights(
            self,
            epoch: int,
            model_state_dict: dict,
            metric: Optional[float] = None,
            save_best: bool = True,
            save_latest: bool = True,
            save_current: bool = False) -> None:

        os.makedirs(self.save_dir, exist_ok=True)
        update_best = metric is None or metric > self.best_model_metric
        if update_best and metric is not None:
            self.best_model_metric = metric

        if save_best and update_best:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                                                      "Weights Best.pckl"))
        if save_latest:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                                                      "Weights Latest.pckl"))
        if save_current:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                                                      "Weights Epoch {} {}.pckl".format(epoch, get_timestamp_str())))


def make_optimizer(
        params_to_update: list[torch.Tensor],
        lr: float = 0.001,
        weight_decay: float = 1e-9,
        clip_grad_norm: bool = False) -> optim.Optimizer:

    optimizer = optim.AdamW(
        params_to_update,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=True)

    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)

    return optimizer


def get_lr(
        optimizer: optim.Optimizer) -> None:

    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_optimizer_lr(
        optimizer: optim.Optimizer,
        lr: float) -> None:

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_scheduler(
        optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, threshold=0.025, patience=10, cooldown=5, min_lr=1e-6, verbose=True)

    return scheduler


def shrink_and_preturb(
        base_net: nn.Module,
        new_net: nn.Module,
        shrink: float = 0.5,
        perturb: float = 0.1) -> nn.Module:
    
    params1 = base_net.parameters()
    params2 = new_net.parameters()
    with torch.set_grad_enabled(False):
        for p1, p2 in zip(params1, params2):
            p1.mul_(shrink).add_(p2, alpha=perturb)

    return base_net


def load_weights(
        model: nn.Module,
        weights_fname,
        map_location=None) -> nn.Module:

    model.load_state_dict(torch.load(weights_fname, map_location=map_location))

    return model


def save_training_session(
        model: nn.Module,
        optimizer: optim.Optimizer,
        sessions_save_dir: str) -> str:

    sub_dir = "Session {}".format(get_timestamp_str())
    sessions_save_dir = os.path.join(sessions_save_dir, sub_dir)
    os.makedirs(sessions_save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(
        sessions_save_dir, "Model State.pckl"))
    torch.save(optimizer.state_dict(), os.path.join(
        sessions_save_dir, "Optimizer State.pckl"))
    print("Saved session to", sessions_save_dir)
    
    return sessions_save_dir


def load_training_session(
        model: nn.Module,
        optimizer: optim.Optimizer,
        session_dir: str,
        update_models: bool = True,
        map_location: Optional[torch.device] = None) -> dict[str, Any]:

    if update_models:
        model.load_state_dict(torch.load(os.path.join(
            session_dir, "Model State.pckl"), map_location=map_location))
        optimizer.load_state_dict(
            torch.load(os.path.join(session_dir, "Optimizer State.pckl"), map_location=map_location))

    print("Loaded session from", session_dir)

    out_data = {
        'model': model,
        'optimizer': optimizer
    }

    return out_data
