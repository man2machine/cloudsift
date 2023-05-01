# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:43:05 2019

@author: Shahir
"""

import abc
import socket
import selectors
import warnings
from typing import Any, Union, Optional

__all__ = ["TCPClientSocket", "UDPClientSocket", "TCPServerSocket", "UDPServerSocket"]

IPv4Address = tuple[str, int]


def _tcp_get_size_data(
        size: int) -> bytes:

    size_data = bytearray()
    while size >= 0x7fffffff:
        size_data.extend((size & 0x7fffffff).to_bytes(4, 'little'))
        size_data[-1] = size_data[-1] | 0x80
        size = size >> 31
    size_data.extend(size.to_bytes(4, 'little'))

    return size_data


def _tcp_receive_size_data(
        conn: socket.socket,
        next_data: bytearray) -> tuple[int, bytearray]:

    size_data = bytearray()

    while True:
        if len(next_data):
            chunk_bytes_left = (len(size_data) & 0b11) or 4
            size_data.extend(next_data[:chunk_bytes_left])
            next_data = next_data[chunk_bytes_left:]
        if len(size_data) >= 4:
            if size_data[-1] >> 7:
                size_data[-1] = size_data[-1] & 0x7f
                continue
            break
        new = conn.recv(8192)
        next_data.extend(new)

    size = 0
    while len(size_data):
        size = (size << 31) | int.from_bytes(size_data[-4:], 'little')
        size_data = size_data[:-4]

    return size, next_data


def _tcp_send_sized(
        sock: socket.socket,
        data: bytes) -> bool:

    size_data = _tcp_get_size_data(len(data))
    sock.sendall(bytes(size_data) + data)


def _tcp_receive_sized(
        sock: socket.socket,
        next_data: bytearray) -> tuple[bytes, bytearray]:

    # when reading the next packet do not discard the data that was read in the past call
    size, next_data = _tcp_receive_size_data(sock, next_data)

    while len(next_data) < size:
        next_data.extend(sock.recv(8192))

    packet_data = bytes(next_data[:size])
    next_data = next_data[size:]

    return packet_data, next_data


class BaseClientSocket(metaclass=abc.ABCMeta):
    ADDRESS_FAMILY = socket.AF_INET
    ALLOW_REUSE_ADDRESS = False

    @property
    @abc.abstractmethod
    def SOCKET_TYPE(
            self) -> socket.SocketKind:

        raise NotImplementedError

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            timeout: float = 60) -> None:

        self.address = address
        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)

        self._blocking = blocking
        self._timeout = timeout

        if self._blocking:
            self._sock.setblocking(1)
        else:
            self._sock.setblocking(0)

        if self._timeout:
            self._sock.settimeout(self._timeout)

        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._sock, selectors.EVENT_READ)

        self._client_start = False

    @abc.abstractmethod
    def connect(
            self) -> bool:

        pass

    def shutdown(
            self) -> bool:

        if not self._client_start:
            return False

        self._client_start = False

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self._sock.close()

        return True

    def set_timeout(
            self,
            timeout: float) -> None:

        self._timeout = timeout
        self._sock.settimeout(timeout)

    @abc.abstractmethod
    def send(
            self,
            data: bytes) -> None:

        pass

    @abc.abstractmethod
    def receive(
            self) -> bytes:

        pass

    def is_packet_waiting(
            self,
            timeout: float = 0) -> bool:

        return bool(self._selector.select(timeout=timeout))

    def packet_cycle(
            self,
            data: bytes,
            fail_shutdown: bool = False) -> tuple[bool, Union[bytes, Exception]]:
        """
        Complete one receive-send cycle and processes errors

        The return value is received data and the error condition
        """

        if not self._client_start:
            raise Exception("Client not started or was closed.")

        try:
            self.send(data)
            receive = self.receive()

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()

            return False, e

        return True, receive

    def get_client_addr(
            self) -> bool:

        if not self._client_start:
            return False


class TCPClientSocket(BaseClientSocket):
    SOCKET_TYPE = socket.SOCK_STREAM

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            timeout: float = 60) -> None:

        super().__init__(address, blocking, timeout)

        self._next_data = bytearray()

    def connect(
            self) -> bool:

        try:
            self._sock.connect(self.address)  # connect to the server
            self._client_start = True
            return True

        except socket.error:
            return False

    def connect_wait_for_server(
            self) -> bool:

        self._sock.setblocking(1)
        if self._timeout:
            self._sock.settimeout(5)
        while True:
            if self.connect():
                break
        if self._timeout:
            self._sock.settimeout(self._timeout)
        self._sock.setblocking(self._blocking)

        return True

    def send(
            self,
            data: bytes) -> None:

        _tcp_send_sized(self._sock, data)

    def receive(
            self) -> bytes:

        data, self._next_data = _tcp_receive_sized(self._sock, self._next_data)

        return data


class UDPClientSocket(BaseClientSocket):
    SOCKET_TYPE = socket.SOCK_DGRAM
    MAX_PACKET_SIZE = 65507

    def connect(
            self) -> bool:

        self._sock.connect(self.address)
        self._client_start = True

        return True

    def send(
            self,
            data: bool) -> bool:

        self._sock.sendto(data, self.address)

        return True

    def receive(
            self) -> bytes:

        return self._sock.recv(self.MAX_PACKET_SIZE)


class BaseServerSocket(metaclass=abc.ABCMeta):
    ADDRESS_FAMILY = socket.AF_INET
    ALLOW_REUSE_ADDRESS = True

    @property
    @abc.abstractmethod
    def SOCKET_TYPE(
            self) -> None:

        raise NotImplementedError

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            timeout: float = 60) -> None:

        self.address = address
        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)

        self._blocking = blocking
        self._timeout = timeout

        if self._blocking:
            self._sock.setblocking(1)
        else:
            self._sock.setblocking(0)

        if self._timeout:
            self._sock.settimeout(self._timeout)

        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._server_start = False

    @abc.abstractmethod
    def start(
            self) -> bool:

        pass

    @abc.abstractmethod
    def shutdown(
            self) -> bool:

        pass

    def set_timeout(
            self,
            timeout: float) -> None:

        self._timeout = timeout
        self._sock.settimeout(timeout)

    @abc.abstractmethod
    def send(
            self) -> None:

        pass

    @abc.abstractmethod
    def receive(
            self) -> bytes:

        pass

    @abc.abstractmethod
    def packet_cycle(
            self) -> Any:

        pass


class TCPServerSocket(BaseServerSocket):
    SOCKET_TYPE = socket.SOCK_STREAM

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            timeout: float = 60,
            num_clients: int = 1) -> None:

        super().__init__(address, blocking=blocking, timeout=timeout)

        self._num_clients = num_clients

        self._client_socks = {}
        self._client_next_data = {}
        self._client_selectors = {}

        self._all_clients_selector = None

    def _listen(
            self,
            num_clients: int) -> None:

        self._sock.listen(num_clients)  # how many connections it can receive at one time

        for _ in range(num_clients):
            try:
                conn, addr = self._sock.accept()  # accept the connection
            except socket.timeout:
                return False

            self._client_socks[addr] = conn
            self._client_next_data[addr] = bytearray()
            selector = selectors.DefaultSelector()
            selector.register(conn, selectors.EVENT_READ)
            self._client_selectors[addr] = selector
            self._all_clients_selector.register(conn, selectors.EVENT_READ)

            conn.setblocking(self._blocking)
            conn.settimeout(self._timeout)

        return True

    def start(
            self,
            verbose: bool = True) -> bool:

        if verbose:
            print("HOSTNAME:", socket.gethostname())
            print("ADDRESS:", self.address)

        self._sock.bind(self.address)
        self._all_clients_selector = selectors.DefaultSelector()
        success = self._listen(self._num_clients)
        self._server_start = True

        return success

    def get_client_addresses(
            self) -> list[IPv4Address]:

        return list(self._client_socks.keys())

    def shutdown(
            self) -> bool:

        if not self._server_start:
            return False

        self._server_start = False

        for addr, conn in self._client_socks.items():
            try:
                # explicitly shutdown. socket.close() merely releases
                # the socket and waits for GC to perform the actual close.
                conn.shutdown(socket.SHUT_RDWR)
                self._client_selectors[addr].close()
            except OSError:
                pass  # some platforms may raise ENOTCONN here
            conn.close()

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN her
        self._sock.close()

        return True

    def is_packet_waiting(
            self,
            timeout: float = 0) -> bool:

        return bool(self._all_clients_selector.select(timeout=timeout))

    def is_client_packet_waiting(
            self,
            addr: IPv4Address,
            timeout: float = 0) -> bool:

        return bool(self._client_selectors[addr].select(timeout=timeout))

    def get_waiting_clients(
            self,
            timeout: Optional[int] = None) -> list[IPv4Address]:

        timeout = timeout if timeout is not None else self._timeout
        selected = self._all_clients_selector.select(timeout=timeout)
        if not selected:
            return []

        conns = [n.fileobj for n in list(zip(*selected))[0]]  # only get connection objects
        addrs = [conn.getpeername() for conn in conns]  # return addresses not connections

        return addrs

    def send(
            self,
            addr: IPv4Address,
            data: bytes) -> None:

        _tcp_send_sized(self._client_socks[addr], data)

    def receive(
            self,
            addr: IPv4Address) -> bytes:

        data, next_data = _tcp_receive_sized(self._client_socks[addr], self._client_next_data[addr])
        self._client_next_data[addr] = next_data

        return data

    def packet_cycle(
            self,
            addr: IPv4Address,
            data: bytes,
            fail_shutdown=False) -> tuple[bool, Union[bytes, Exception]]:
        """
        Complete one receive-send cycle and processes errors

        The return value is whether it was successful or not, the received data or error if any
        """

        if not self._server_start:
            raise Exception("Server not started or was closed")

        try:
            receive = self.receive(addr)
            self.send(addr, data)

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()
            return False, e

        return True, receive


class UDPServerSocket(BaseServerSocket):
    SOCKET_TYPE = socket.SOCK_DGRAM
    MAX_PACKET_SIZE = 65507

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            broadcast: bool = False,
            timeout: float = 60) -> None:

        super().__init__(address, blocking=blocking, timeout=timeout)

        if broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._sock, selectors.EVENT_READ)

    def start(
            self,
            verbose: bool = True) -> bool:

        if verbose:
            print("HOSTNAME:", socket.gethostname())
            print("ADDRESS:", self.address)

        self._sock.bind(self.address)

        self._server_start = True

        return True

    def shutdown(
            self) -> bool:

        if not self._server_start:
            return False

        self._server_start = False

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self._sock.close()

        return True

    def is_packet_waiting(
            self,
            timeout: float = 0) -> bool:

        return bool(self._selector.select(timeout=timeout))

    def send(
            self,
            addr: IPv4Address,
            data: bytes) -> None:

        self._sock.sendto(data, addr)

    def receive(
            self) -> tuple[bytes, IPv4Address]:

        data, addr = self._sock.recvfrom(self.MAX_PACKET_SIZE)
        return data, addr

    def packet_cycle(
            self,
            data: bytes,
            fail_shutdown: bool = False) -> tuple[bool, Union[tuple[bytes, IPv4Address], Exception]]:
        """
        Complete one receive-send cycle and processes errors
        Not guaranteed to actually communicate with a specific address

        The return value is whether it was successful or not, the received data, and the address or error if any
        """

        warnings.warn(
            "Using UDP socket: not guaranteed to communicate with any specific address,"
            " should only be used for connections with 1 client")

        if not self._server_start:
            raise Exception("Server not started or was closed.")

        try:
            receive, recv_addr = self.receive()
            self.send(recv_addr, data)

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()
            return False, e

        return True, (receive, recv_addr)


if __name__ == '__main__':
    import random
    import secrets
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    # some basic functionality tests

    warnings.filterwarnings('ignore')

    addr = ('localhost', 8888)

    def get_random_small_payload() -> bytes:

        size = random.randint(0, 1 << 15)

        return secrets.token_bytes(size)

    def get_random_large_size() -> int:

        return random.randint(0, 1 << 256)

    def server_proc(
            tcp: bool,
            payload: bytes) -> tuple[bool, Any]:

        if tcp:
            server = TCPServerSocket(addr, timeout=5)
        else:
            server = UDPServerSocket(addr, timeout=5)
        server.start(verbose=False)
        if tcp:
            client_addr = server.get_client_addresses()[0]
            out = server.packet_cycle(client_addr, payload)
        else:
            out = server.packet_cycle(payload)
            if out[0]:
                out = (out[0], out[1][0])
        server.shutdown()

        return out

    def client_proc(
            tcp: bool,
            payload: bytes) -> tuple[bool, Any]:

        if tcp:
            client = TCPClientSocket(addr, timeout=5)
            client.connect_wait_for_server()
        else:
            client = UDPClientSocket(addr, timeout=5)
            client.connect()
        out = client.packet_cycle(payload)
        client.shutdown()

        return out

    def tcp_server_size_proc(
            size: int,
            payload: bytes) -> int:

        server = TCPServerSocket(addr, timeout=5)
        server.start(verbose=False)
        client_addr = server.get_client_addresses()[0]
        size_data = _tcp_get_size_data(size)
        recv_size, _ = _tcp_receive_size_data(server._client_socks[client_addr], server._client_next_data[client_addr])
        # large size, fake a small payload due to memory constraints
        server._client_socks[client_addr].sendall(size_data + payload)
        server.shutdown()

        return recv_size

    def tcp_client_size_proc(
            size: int,
            payload: bytes) -> int:

        client = TCPClientSocket(addr, timeout=5)
        client.connect_wait_for_server()
        size_data = _tcp_get_size_data(size)
        # large size, fake a small payload due to memory constraints
        client._sock.sendall(size_data + payload)
        recv_size, _ = _tcp_receive_size_data(client._sock, client._next_data)
        client.shutdown()

        return recv_size
    
    # test random payloads sent between server and client and check if they match for both TCP and UDP packets
    for tcp in [True, False]:
        for n in range(5):
            payload1 = get_random_small_payload()
            payload2 = get_random_small_payload()
            with ThreadPoolExecutor(max_workers=2) as executor:
                server_out = executor.submit(server_proc, tcp=tcp, payload=payload1)
                client_out = executor.submit(client_proc, tcp=tcp, payload=payload2)

                server_out = server_out.result()
                client_out = client_out.result()

                assert server_out == (True, payload2)
                assert client_out == (True, payload1)

                executor.shutdown()
                
    # test very large payload sizes (more than 31 bits to store size) and see if processing functions for TCP packets
    for n in range(5):
        payload1 = get_random_small_payload()
        payload2 = get_random_small_payload()
        size1 = get_random_large_size()
        size2 = get_random_large_size()
        with ThreadPoolExecutor(max_workers=2) as executor:
            server_out = executor.submit(tcp_server_size_proc, size=size1, payload=payload1)
            client_out = executor.submit(tcp_client_size_proc, size=size2, payload=payload2)

            server_out = server_out.result()
            client_out = client_out.result()

            assert server_out == size2
            assert client_out == size1

            executor.shutdown()
