import logging
import os
import platform
import time
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10
DEFAULT_READY_TIMEOUT = 300
SLOW_READY_TIMEOUT = 300
ROOTLESS_USER_PORTS = "5000,8080,9222"


class PortAllocationError(Exception):
    pass


class DockerProvider(Provider):
    def __init__(self, region: str):
        self.client = docker.from_env()
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _is_rootless_daemon(self) -> bool:
        try:
            info = self.client.info()
            security_options = info.get("SecurityOptions") or []
            return any("rootless" in str(option).lower() for option in security_options)
        except Exception as e:
            logger.warning(f"Failed to inspect Docker daemon mode: {e}")
            return False

    def _get_ready_timeout(self, runtime_environment: dict, rootless_mode: bool) -> int:
        timeout = DEFAULT_READY_TIMEOUT
        if runtime_environment.get("KVM", "Y").upper().startswith("N"):
            timeout = SLOW_READY_TIMEOUT
        if rootless_mode:
            timeout = max(timeout, SLOW_READY_TIMEOUT)
        return timeout

    def _get_used_ports(self):
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        
        return system_ports | docker_ports

    def _get_available_port(self, start_port: int) -> int:
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 300):
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)
        
        try:
            with lock:
                self.vnc_port = self._get_available_port(8006)
                self.server_port = self._get_available_port(5000)
                self.chromium_port = self._get_available_port(9222)
                self.vlc_port = self._get_available_port(8080)

                runtime_environment = dict(self.environment)
                rootless_mode = self._is_rootless_daemon()
                if rootless_mode:
                    runtime_environment["NETWORK"] = "user"
                    runtime_environment["USER_PORTS"] = ROOTLESS_USER_PORTS
                    logger.info(
                        "Rootless Docker detected: using user-mode networking "
                        f"with USER_PORTS={ROOTLESS_USER_PORTS}"
                    )

                devices = []
                kvm_path = "/dev/kvm"
                if os.path.exists(kvm_path) and os.access(kvm_path, os.W_OK):
                    devices.append(kvm_path)
                    logger.info("KVM device found, using hardware acceleration")
                else:
                    runtime_environment["KVM"] = "N"
                    if os.path.exists(kvm_path):
                        logger.warning(
                            "KVM device exists but is not writable; "
                            "running without hardware acceleration (will be slower)"
                        )
                    else:
                        logger.warning(
                            "KVM device not found, running without hardware "
                            "acceleration (will be slower)"
                        )

                self.container = self.client.containers.run(
                    "happysixd/osworld-docker",
                    environment=runtime_environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,
                    volumes={
                        os.path.abspath(path_to_vm): {
                            "bind": "/System.qcow2",
                            "mode": "ro"
                        }
                    },
                    ports={
                        8006: self.vnc_port,
                        5000: self.server_port,
                        9222: self.chromium_port,
                        8080: self.vlc_port
                    },
                    detach=True
                )

            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            ready_timeout = self._get_ready_timeout(runtime_environment, rootless_mode)
            logger.info(f"Waiting for VM readiness (timeout={ready_timeout}s)")
            self._wait_for_vm_ready(timeout=ready_timeout)

        except Exception as e:
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove(v=True)
                except:
                    pass
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        if self.container:
            logger.info("Stopping VM...")
            try:
                self.container.stop()
                self.container.remove(v=True)
                time.sleep(WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
