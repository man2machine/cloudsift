# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 07:34:12 2023

@author: Shahir
"""

import os
import json
import abc
import threading
import shutil
from typing import TypeVar, Generic, Optional

from PIL.Image import Image

from cloudsift.datasets import ImageDatasetType
from cloudsift.utils import get_rel_pkg_path
from cloudsift.communication import TCPClientSocket
from cloudsift.simulation import BaseCloudLabeler, LabelingResult

ROOT_LABEL_DATA_DIR = get_rel_pkg_path("resources/temp_labeling_data/")

T_co = TypeVar('T_co', covariant=True)


class BaseManualCloudLabeler(Generic[T_co], BaseCloudLabeler[T_co]):
    def __init__(
            self,
            data_dir: Optional[str] = None,
            host: Optional[str] = None,
            port: Optional[int] = None) -> None:
        
        super().__init__()

        if host is None:
            host = "localhost"
        if port is None:
            port = 8001
        if data_dir is None:
            data_dir = ROOT_LABEL_DATA_DIR
        
        self.data_dir = os.path.abspath(data_dir)
        
        self._host = host
        self._port = port
        self._client = TCPClientSocket((self._host, self._port))
        
        self._init_ev = threading.Event()
        self._stop_ev = threading.Event()
    
    @abc.abstractmethod
    def initialize(
            self) -> None:
        
        self._stop_ev.clear()
        self._init_ev.set()

        
class ManualCategoricalImageCloudLabeler(BaseManualCloudLabeler[int]):
    def __init__(
            self,
            class_names: list[str],
            img_display_size: int,
            host: Optional[str] = None,
            port: Optional[int] = None,
            data_dir: Optional[str] = None) -> None:
        
        super().__init__(
            data_dir=data_dir,
            host=host,
            port=port)

        self.class_names = class_names
        self.img_display_size = img_display_size

    def initialize(
            self) -> None:
        
        shutil.rmtree(self.data_dir)
        self._client.connect_wait_for_server()
        
        labeling_config = [
            "<View>",
            "<Image name='image' value='$image' maxWidth='{}px'/>".format(self.img_display_size),
            "<Choices name='class' toName='image'>"]
        for choice in self.class_names:
            assert ("'" not in choice) and ("\"" not in choice)
            labeling_config.append("<Choice value='{}'/>".format(choice))
        labeling_config.extend([
            "</Choices>",
            "</View>"])
        labeling_config = "\n".join(labeling_config)
        choice_to_label = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        self._image_index = 0
        
        recv_data = self._client.receive()
        send_data = {
            'data_dir': self.data_dir,
            'labeling_config': labeling_config,
            'choice_to_label': choice_to_label
        }
        send_data = self._client.send(json.dumps(send_data).encode())
        
        super().initialize()

    def request_labels(
            self,
            imgs_to_label: list[Image]) -> list[LabelingResult[int]]:
        
        fnames = []
        for img in imgs_to_label:
            fname = "{}.png".format(self._image_index)
            img.save(os.path.join(self.data_dir, fname))
            fnames.append(fname)
            self._image_index += 1
        
        send_data = {
            'fnames': fnames,
            'stop': self._stop_ev.is_set()
        }
        self._client.send(json.dumps(send_data).encode())
        while True:
            if self._stop_ev.is_set():
                break
            
            if not self._client.is_packet_waiting(timeout=10):
                print("Timeout")
                continue
            
            recv_data = json.loads(self._client.receive())
            labels = recv_data['labels']
            output = [LabelingResult(y is not None, y) for y in labels]
            
            return output
        
    def stop(
            self):
        
        self._stop_ev.set()
        send_data = {
            'stop': self._stop_ev.is_set()
        }
        self._client.send(send_data)
        self._client.shutdown()

        
