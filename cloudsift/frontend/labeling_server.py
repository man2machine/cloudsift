# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 02:46:33 2023

@author: Shahir
"""

import time
import os
import sys
import abc
import subprocess
import json
import threading
from typing import Union, Optional

from label_studio_sdk import Client

from cloudsift.communication import TCPServerSocket


def run_command(
        cmd: Union[str, list[str]],
        shell=True,
        capture_output=False) -> subprocess.Popen:

    if capture_output:
        proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return proc


def get_command_output(
        proc: subprocess.Popen,
        timeout: Optional[float] = None) -> Union[tuple[bytes, bytes], None]:

    try:
        res = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        res = None

    return res


def end_command(
        proc: subprocess.Popen,
        timeout1: float,
        timeout2: float) -> Union[tuple[bytes, bytes], None]:

    out = get_command_output(proc, timeout1)
    if out is not None:
        return out

    proc.kill()
    out = get_command_output(proc, timeout2)

    return out


class BaseLabelingManager(metaclass=abc.ABCMeta):
    LS_USERNAME = "default@cloudsift.com"
    LS_PASSWORD = "12345678"
    LS_API_KEY = "CLOUDSIFT"
    LS_URL = "http://localhost:8080"
    FS_URL = "http://localhost:8081"

    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None) -> None:

        if host is None:
            host = "localhost"
        if port is None:
            port = 8001

        self._host = host
        self._port = port
        self._server = TCPServerSocket((self._host, self._port))

        self._thread = None
        self._init_ev = threading.Event()
        self._stop_ev = threading.Event()

    def _start_ls_and_fs(
            self,
            fs_dir) -> None:
        
        # capturing output causes process to block
        os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
        self.ls_proc = run_command(
            "label-studio start --username {} --password {} --user-token {}".format(
                self.LS_USERNAME, self.LS_PASSWORD, self.LS_API_KEY),
            capture_output=False)

        os.makedirs(fs_dir, exist_ok=True)
        self.fs_proc = run_command(
            "{} -m http.server 8081 -d {}".format(sys.executable, fs_dir),
            capture_output=False)
        
        time.sleep(10)
        
        self.ls = Client(url=self.LS_URL, api_key=self.LS_API_KEY)
        response = self.ls.check_connection()
        assert response['status'] == 'UP'

    @abc.abstractmethod
    def initialize(
            self) -> None:

        self._stop_ev.clear()
        self._init_ev.set()

    @abc.abstractmethod
    def run(
            self) -> None:

        pass

    def stop(
            self) -> None:

        self._stop_ev.set()

    def run_thread(
            self) -> None:

        if (self._thread is not None) and self._thread.is_alive():
            raise Exception("Thread already running")
        self._thread = threading.Thread(target=self.run)
        self._thread.start()


class CategoricalImageLabelingManager(BaseLabelingManager):
    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None) -> None:

        super().__init__(host, port)

    def initialize(
            self) -> None:

        if not self._server.start():
            raise ConnectionError("Failed to connect to client")

        self._client_addr = self._server.get_client_addresses()[0]
        self._server.send(self._client_addr, b'')
        recv_data = json.loads(self._server.receive(self._client_addr).decode())
        self.data_dir = recv_data['data_dir']
        self.labeling_config = recv_data['labeling_config']
        self.choice_to_label = recv_data['choice_to_label']
        
        self._start_ls_and_fs(self.data_dir)
        
        self.ls.delete_all_projects()
        self.ls_proj = self.ls.start_project(
            title="CloudSift Labeling",
            label_config=self.labeling_config,
            enable_empty_annotation=False)

        super().initialize()

    def _add_tasks(
            self,
            fnames: list[str]) -> list[int]:

        for fname in fnames:
            if not os.path.exists(os.path.join(self.data_dir, fname)):
                raise FileNotFoundError()
        task_indices = self.ls_proj.import_tasks(
            [{'image': "{}/{}".format(self.FS_URL, n)} for n in fnames])

        return task_indices

    def _label_data(
            self,
            fnames: list[str]) -> list[Union[int, None]]:

        task_ids = self._add_tasks(fnames)
        while self.ls_proj.get_unlabeled_tasks_ids():
            print("Waiting for labeling to finish")
            time.sleep(5)

        labels = []
        for task_id in task_ids:
            task = self.ls_proj.get_task(task_id)
            results = task['annotations'][0]['result']
            for result in results:
                if (result['from_name'] == 'class') and (result['to_name'] == 'image'):
                    break
            else:
                raise ValueError("Invalid labeling response")
            label = self.choice_to_label[result['value']['choices'][0]]
            labels.append(label)

        return labels

    def run(
            self) -> None:

        if not self._init_ev.is_set():
            raise Exception("Not initialized")

        while True:
            if self._stop_ev.is_set():
                break

            if not self._server.is_client_packet_waiting(self._client_addr, timeout=10):
                print("Timeout")
                continue

            recv_data = json.loads(self._server.receive(self._client_addr))
            if recv_data['stop']:
                self._stop_ev.set()
                continue
            
            fnames = recv_data['fnames']
            labels = self._label_data(fnames)
            send_data = {
                'labels': labels
            }

            self._server.send(self._client_addr, json.dumps(send_data).encode())
