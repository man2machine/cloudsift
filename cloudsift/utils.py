# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:13:14 2022

@author: Shahir
"""

import os
import abc
import datetime
import json
from typing import Union, Any

import numpy as np
import numpy.typing as npt

import cloudsift

BoolArrayLike1D = Union[list[bool], npt.NDArray[np.bool_]]
IntArrayLike1D = Union[list[int], npt.NDArray[np.integer]]
FloatArrayLike1D = Union[list[float], npt.NDArray[np.floating]]
AnyArrayLike1D = Union[list[Any], npt.NDArray]


class JSONDictSerializable(metaclass=abc.ABCMeta):
    def __str__(
            self) -> str:
        return str(self.to_dict())

    def __repr__(
            self) -> str:
        return str(self.to_dict())

    @abc.abstractmethod
    def to_dict(
            self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(
            cls,
            dct: dict) -> Any:
        pass

    def to_json(
            self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
            cls,
            data: str) -> Any:
        return cls.from_dict(json.loads(data))

    def to_bytes(
            self) -> bytes:
        return self.to_json().encode()

    @classmethod
    def from_bytes(
            cls,
            data: bytes) -> Any:
        return cls.from_json(data.decode())


def number_menu(
        option_list: list[str]) -> tuple[int, str]:
    print("-" * 60)

    for n in range(len(option_list)):
        print(n, ": ", option_list[n])

    choice = input("Choose the number corresponding to your choice: ")
    for n in range(5):
        try:
            choice = int(choice)
            if choice < 0 or choice > len(option_list) - 1:
                raise ValueError
            print("-" * 60 + "\n")
            return choice, option_list[choice]
        except ValueError:
            choice = input("Invalid input, choose again: ")

    raise ValueError("Not recieving a valid input")


def get_rel_pkg_path(
        path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(cloudsift.__file__), "..", path))


def load_rel_config_json(
        fname: str) -> dict:
    fname = get_rel_pkg_path(fname)
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def get_timestamp_str(
        include_seconds: bool = True) -> str:
    if include_seconds:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M-%S %p")
    else:
        return datetime.datetime.now().strftime("%m-%d-%Y %I-%M %p")
