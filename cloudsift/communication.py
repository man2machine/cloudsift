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


def _tcp_recieve_end(
        conn: socket.socket,
        next_data: bytes,
        packet_end: bytes) -> tuple[bytes, bytes]:
    """
    Receive data so that none of it is incomplete
    Reads until the the packet end
    https://code.activestate.com/recipes/408859-socketrecv-three-ways-to-turn-it-into-recvall/
    """

    # when reading the next packet do not discard the data that was read in the past call
    total_data = [next_data]
    next_data = bytes()

    while True:
        data = conn.recv(8192)
        if packet_end in data:
            index = data.find(packet_end)
            total_data.append(data[:index])
            next_data += data[index + len(packet_end):]
            # cut until the packet ending is found
            break
        total_data.append(data)
        if len(total_data) > 1:
            # check if end_of_data was split
            last_pair = total_data[-2] + total_data[-1]
            if packet_end in last_pair:
                index = last_pair.find(packet_end)
                total_data[-2] = last_pair[:index]
                total_data.pop()
                next_data += last_pair[index + len(packet_end):]
                break

    return b''.join(total_data), next_data


class BaseClientSocket(metaclass=abc.ABCMeta):
    ADDRESS_FAMILY = socket.AF_INET
    ALLOW_REUSE_ADDRESS = False

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
        self.sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)

        self._blocking = blocking
        self._timeout = timeout

        if self._blocking:
            self.sock.setblocking(1)
        else:
            self.sock.setblocking(0)

        if self._timeout:
            self.sock.settimeout(self._timeout)

        if self.ALLOW_REUSE_ADDRESS:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self.sock, selectors.EVENT_READ)

        self._client_start = False

    @abc.abstractmethod
    def connect(
            self) -> bool:
        """
        Connects to a host
        """

        pass

    def shutdown(
            self) -> bool:
        """
        Stop the connection
        """

        if not self._client_start:
            return False
        self._client_start = False

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self.sock.close()

        return True

    def set_timeout(
            self,
            timeout: float) -> None:

        self._timeout = timeout
        self.sock.settimeout(timeout)

    @abc.abstractmethod
    def send(
            self,
            data: bytes) -> bool:
        """
        Send data"
        """

        pass

    @abc.abstractmethod
    def receive(
            self) -> bytes:
        """
        Receive data
        """

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
    _PACKET_END = b'END###'

    def __init__(
            self,
            address: IPv4Address,
            blocking: bool = True,
            timeout: float = 60) -> None:

        super().__init__(address, blocking, timeout)

        self._next_data = b''

    def connect(
            self) -> bool:

        try:
            self.sock.connect(self.address)  # connect to the server
            self._client_start = True
            return True

        except socket.error:
            return False

    def connect_wait_for_server(
            self) -> bool:

        self.sock.setblocking(1)
        if self._timeout:
            self.sock.settimeout(5)
        while True:
            if self.connect():
                break
        if self._timeout:
            self.sock.settimeout(self._timeout)
        self.sock.setblocking(self._blocking)

        return True

    def receive(
            self) -> bytes:

        data, self._next_data = _tcp_recieve_end(self.sock, self._next_data, self._PACKET_END)

        return data

    def send(
            self,
            data) -> bool:

        self.sock.sendall(data + self._PACKET_END)

        return True


class UDPClientSocket(BaseClientSocket):
    SOCKET_TYPE = socket.SOCK_DGRAM
    MAX_PACKET_SIZE = 65507

    def connect(
            self) -> bool:

        self.sock.connect(self.address)
        self._client_start = True

        return True

    def receive(
            self) -> bytes:

        return self.sock.recv(self.MAX_PACKET_SIZE)

    def send(
            self,
            data) -> bool:

        self.sock.sendto(data, self.address)

        return True


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
        self.sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)

        self._blocking = blocking
        self._timeout = timeout

        if self._blocking:
            self.sock.setblocking(1)
        else:
            self.sock.setblocking(0)

        if self._timeout:
            self.sock.settimeout(self._timeout)

        if self.ALLOW_REUSE_ADDRESS:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._server_start = False

    @abc.abstractmethod
    def start(
            self) -> bool:

        pass

    @abc.abstractmethod
    def shutdown(
            self) -> bool:
        """
        Stop the connection
        """

        pass

    def set_timeout(
            self,
            timeout: float) -> None:

        self._timeout = timeout
        self.sock.settimeout(timeout)

    @abc.abstractmethod
    def receive(
            self) -> bytes:

        pass

    @abc.abstractmethod
    def send(
            self) -> bool:

        pass

    @abc.abstractmethod
    def packet_cycle(
            self) -> Any:

        pass


class TCPServerSocket(BaseServerSocket):
    SOCKET_TYPE = socket.SOCK_STREAM
    _PACKET_END = b'END###'

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

    def listen(
            self,
            num_clients: int) -> None:

        self.sock.listen(num_clients)  # how many connections it can receive at one time

        for _ in range(num_clients):
            try:
                conn, addr = self.sock.accept()  # accept the connection
            except socket.timeout:
                return False

            self._client_socks[addr] = conn
            self._client_next_data[addr] = b''
            selector = selectors.DefaultSelector()
            selector.register(conn, selectors.EVENT_READ)
            self._client_selectors[addr] = selector
            self._all_clients_selector.register(conn, selectors.EVENT_READ)

            conn.setblocking(self._blocking)
            conn.settimeout(self._timeout)

        return True

    def start(
            self,
            output: bool = True,
            listen: bool = True) -> bool:
        """
        Starts the server and connects to all clients
        """

        if output:
            print("HOSTNAME:", socket.gethostname())
            print("ADDRESS:", self.address)

        self.sock.bind(self.address)

        self._all_clients_selector = selectors.DefaultSelector()

        if listen:
            success = self.listen(self._num_clients)

        self._server_start = True

        return success

    def get_client_addresses(
            self) -> list[IPv4Address]:

        return list(self._client_socks.keys())

    def shutdown(
            self) -> bool:
        """
        Stop the server
        """

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
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN her
        self.sock.close()

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
            data: bytes) -> bool:
        """
        Send data
        """

        conn = self._client_socks[addr]
        conn.sendall(data + self._PACKET_END)

        return True

    def receive(
            self,
            addr: IPv4Address) -> bytes:
        """
        Receives one packet of data
        """

        data, next_data = _tcp_recieve_end(self._client_socks[addr], self._client_next_data[addr], self._PACKET_END)
        self._client_next_data[addr] = next_data

        return data

    def packet_cycle(
            self,
            addr: IPv4Address,
            data: bytes,
            fail_shutdown=False) -> tuple[bool, Union[bytes, Exception]]:
        """
        Complete one receive-send cycle and processes errors

        The return value is received data and the error condition
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
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self.sock, selectors.EVENT_READ)

    def start(
            self,
            output: bool = True) -> bool:
        """
        Starts the server
        """

        if output:
            print("HOSTNAME:", socket.gethostname())
            print("ADDRESS:", self.address)

        self.sock.bind(self.address)

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
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self.sock.close()
        
        return True

    def is_packet_waiting(
            self,
            timeout: float = 0) -> bool:

        return bool(self._selector.select(timeout=timeout))

    def send(
            self,
            addr: IPv4Address,
            data: bytes) -> bool:
        """
        Send data
        """

        self.sock.sendto(data, addr)

        return True

    def receive(
            self) -> tuple[bytes, IPv4Address]:
        """
        Receives one packet of data
        """

        data, addr = self.sock.recvfrom(self.MAX_PACKET_SIZE)
        return data, addr

    def packet_cycle(
            self,
            data: bytes,
            fail_shutdown: bool = False) -> tuple[bool, bytes, Union[IPv4Address, Exception]]:
        """
        Complete one receive-send cycle and processes errors
        Not guaranteed to actually communicate with a specific address

        The return value is received data and the error condition
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
            return False, e, None

        return True, receive, recv_addr
