from abc import ABC, abstractmethod


class Provider(ABC):
    def __init__(self, region: str = None):
        self.region = region

    @abstractmethod
    def start_emulator(self, path_to_vm: str, headless: bool):
        pass

    @abstractmethod
    def get_ip_address(self, path_to_vm: str) -> str:
        pass

    @abstractmethod
    def save_state(self, path_to_vm: str, snapshot_name: str):
        pass

    @abstractmethod
    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        pass

    @abstractmethod
    def stop_emulator(self, path_to_vm: str):
        pass


class VMManager(ABC):
    checked_and_cleaned = False

    @abstractmethod
    def initialize_registry(self, **kwargs):
        pass

    @abstractmethod
    def add_vm(self, vm_path, **kwargs):
        pass

    @abstractmethod
    def delete_vm(self, vm_path, **kwargs):
        pass

    @abstractmethod
    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    @abstractmethod
    def list_free_vms(self, **kwargs):
        pass

    @abstractmethod
    def check_and_clean(self, **kwargs):
        pass

    @abstractmethod
    def get_vm_path(self, **kwargs):
        pass
