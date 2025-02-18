import os
import logging
import sys
from collections import defaultdict
from multiprocessing import Process
import functools
from typing import List,Any,Iterable
import pynvml
import torch
import numpy as np

def get_rs(x = None) -> np.random.RandomState:
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random

def get_default_numpy_dtype() -> type:
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


class _CriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING

class _NonCriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING
    
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def get_chained_attr(x: Any, attr: str) -> Any:
    for k in attr.split("."):
        if not hasattr(x, k):
            raise AttributeError(f"{attr} not found!")
        x = getattr(x, k)
    return x

class LogManager(metaclass=SingletonMeta):

    r"""
    Manage loggers used in the package
    """

    def __init__(self) -> None:
        self._loggers = {}
        self._log_file = None
        self._console_log_level = logging.INFO
        self._file_log_level = logging.DEBUG
        self._file_fmt = \
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
        self._console_fmt = \
            "[%(levelname)s] %(name)s: %(message)s"
        self._date_fmt = "%Y-%m-%d %H:%M:%S"

    @property
    def log_file(self) -> str:
        r"""
        Configure log file
        """
        return self._log_file

    @property
    def file_log_level(self) -> int:
        r"""
        Configure logging level in the log file
        """
        return self._file_log_level

    @property
    def console_log_level(self) -> int:
        r"""
        Configure logging level printed in the console
        """
        return self._console_log_level

    def _create_file_handler(self) -> logging.FileHandler:
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.file_log_level)
        file_handler.setFormatter(logging.Formatter(
            fmt=self._file_fmt, datefmt=self._date_fmt))
        return file_handler

    def _create_console_handler(self, critical: bool) -> logging.StreamHandler:
        if critical:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.addFilter(_CriticalFilter())
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.addFilter(_NonCriticalFilter())
        console_handler.setLevel(self.console_log_level)
        console_handler.setFormatter(logging.Formatter(fmt=self._console_fmt))
        return console_handler

    def get_logger(self, name: str) -> logging.Logger:
        r"""
        Get a logger by name
        """
        if name in self._loggers:
            return self._loggers[name]
        new_logger = logging.getLogger(name)
        new_logger.setLevel(logging.DEBUG)  # lowest level
        new_logger.addHandler(self._create_console_handler(True))
        new_logger.addHandler(self._create_console_handler(False))
        if self.log_file:
            new_logger.addHandler(self._create_file_handler())
        self._loggers[name] = new_logger
        return new_logger

    @log_file.setter
    def log_file(self, file_name: os.PathLike) -> None:
        self._log_file = file_name
        for logger in self._loggers.values():
            for idx, handler in enumerate(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.handlers[idx].close()
                    if self.log_file:
                        logger.handlers[idx] = self._create_file_handler()
                    else:
                        del logger.handlers[idx]
                    break
            else:
                if file_name:
                    logger.addHandler(self._create_file_handler())

    @file_log_level.setter
    def file_log_level(self, log_level: int) -> None:
        self._file_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(self.file_log_level)
                    break

    @console_log_level.setter
    def console_log_level(self, log_level: int) -> None:
        self._console_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler:  # pylint: disable=unidiomatic-typecheck
                    handler.setLevel(self.console_log_level)

log = LogManager()

def logged(obj):
    r"""
    Add logger as an attribute
    """
    obj.logger = log.get_logger(obj.__name__)
    return obj

@logged
class ConfigManager:
    def __init__(self) -> None:
        self.TMP_PREFIX = "PrismTMP"
        self.ANNDATA_KEY = "__PrismConfig__"
        self.CPU_ONLY = False
        self.CUDNN_MODE = "repeatability"
        self.MASKED_GPUS = []
        self.ARRAY_SHUFFLE_NUM_WORKERS = 0
        self.GRAPH_SHUFFLE_NUM_WORKERS = 1
        self.FORCE_TERMINATE_WORKER_PATIENCE = 60
        self.DATALOADER_NUM_WORKERS = 0
        self.DATALOADER_FETCHES_PER_WORKER = 4
        self.DATALOADER_PIN_MEMORY = True
        self.CHECKPOINT_SAVE_INTERVAL = 10
        self.CHECKPOINT_SAVE_NUMBERS = 3
        self.PRINT_LOSS_INTERVAL = 10
        self.TENSORBOARD_FLUSH_SECS = 5
        self.ALLOW_TRAINING_INTERRUPTION = True
    
    @property
    def CPU_ONLY(self) -> bool:
        r"""
        Whether computation should use only CPUs.
        Default value is ``False``.
        """
        return self._CPU_ONLY
    @CPU_ONLY.setter
    def CPU_ONLY(self, cpu_only: bool) -> None:
        self._CPU_ONLY = cpu_only
        if self._CPU_ONLY and self._DATALOADER_NUM_WORKERS:
            self.logger.warning(
                "It is recommended to set `DATALOADER_NUM_WORKERS` to 0 "
                "when using CPU_ONLY mode. Otherwise, deadlocks may happen "
                "occationally."
            )
    @property
    def CUDNN_MODE(self) -> str:
        return self._CUDNN_MODE
    @CUDNN_MODE.setter
    def CUDNN_MODE(self, cudnn_mode: str) -> None:
        if cudnn_mode not in ("repeatability", "performance"):
            raise ValueError("Invalid mode!")
        self._CUDNN_MODE = cudnn_mode
        torch.backends.cudnn.deterministic = self._CUDNN_MODE == "repeatability"
        torch.backends.cudnn.benchmark = self._CUDNN_MODE == "performance"
    @property
    def MASKED_GPUS(self) -> List[int]:
        return self._MASKED_GPUS
    @MASKED_GPUS.setter
    def MASKED_GPUS(self, masked_gpus: List[int]) -> None:
        if masked_gpus:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for item in masked_gpus:
                if item >= device_count:
                    raise ValueError(f"GPU device \"{item}\" is non-existent!")
        self._MASKED_GPUS = masked_gpus
    @property
    def ARRAY_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for array data shuffling.
        Default value is ``0``.
        """
        return self._ARRAY_SHUFFLE_NUM_WORKERS
    @ARRAY_SHUFFLE_NUM_WORKERS.setter
    def ARRAY_SHUFFLE_NUM_WORKERS(self, array_shuffle_num_workers: int) -> None:
        self._ARRAY_SHUFFLE_NUM_WORKERS = array_shuffle_num_workers
    @property
    def GRAPH_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for graph data shuffling.
        Default value is ``1``.
        """
        return self._GRAPH_SHUFFLE_NUM_WORKERS
    @GRAPH_SHUFFLE_NUM_WORKERS.setter
    def GRAPH_SHUFFLE_NUM_WORKERS(self, graph_shuffle_num_workers: int) -> None:
        self._GRAPH_SHUFFLE_NUM_WORKERS = graph_shuffle_num_workers
    @property
    def FORCE_TERMINATE_WORKER_PATIENCE(self) -> int:
        r"""
        Seconds to wait before force terminating unresponsive workers.
        Default value is ``60``.
        """
        return self._FORCE_TERMINATE_WORKER_PATIENCE
    @FORCE_TERMINATE_WORKER_PATIENCE.setter
    def FORCE_TERMINATE_WORKER_PATIENCE(self, force_terminate_worker_patience: int) -> None:
        self._FORCE_TERMINATE_WORKER_PATIENCE = force_terminate_worker_patience
    @property
    def DATALOADER_NUM_WORKERS(self) -> int:
        r"""
        Number of worker processes to use in data loader.
        Default value is ``0``.
        """
        return self._DATALOADER_NUM_WORKERS
    @DATALOADER_NUM_WORKERS.setter
    def DATALOADER_NUM_WORKERS(self, dataloader_num_workers: int) -> None:
        if dataloader_num_workers > 8:
            self.logger.warning(
                "Worker number 1-8 is generally sufficient, "
                "too many workers might have negative impact on speed."
            )
        self._DATALOADER_NUM_WORKERS = dataloader_num_workers
    @property
    def DATALOADER_FETCHES_PER_WORKER(self) -> int:
        r"""
        Number of fetches per worker per batch to use in data loader.
        Default value is ``4``.
        """
        return self._DATALOADER_FETCHES_PER_WORKER
    @DATALOADER_FETCHES_PER_WORKER.setter
    def DATALOADER_FETCHES_PER_WORKER(self, dataloader_fetches_per_worker: int) -> None:
        self._DATALOADER_FETCHES_PER_WORKER = dataloader_fetches_per_worker
    @property
    def DATALOADER_FETCHES_PER_BATCH(self) -> int:
        r"""
        Number of fetches per batch in data loader (read-only).
        """
        return max(1, self.DATALOADER_NUM_WORKERS) * self.DATALOADER_FETCHES_PER_WORKER
    @property
    def DATALOADER_PIN_MEMORY(self) -> bool:
        r"""
        Whether to use pin memory in data loader.
        Default value is ``True``.
        """
        return self._DATALOADER_PIN_MEMORY
    @DATALOADER_PIN_MEMORY.setter
    def DATALOADER_PIN_MEMORY(self, dataloader_pin_memory: bool):
        self._DATALOADER_PIN_MEMORY = dataloader_pin_memory
    @property
    def CHECKPOINT_SAVE_INTERVAL(self) -> int:
        r"""
        Automatically save checkpoints every n epochs.
        Default value is ``10``.
        """
        return self._CHECKPOINT_SAVE_INTERVAL
    @CHECKPOINT_SAVE_INTERVAL.setter
    def CHECKPOINT_SAVE_INTERVAL(self, checkpoint_save_interval: int) -> None:
        self._CHECKPOINT_SAVE_INTERVAL = checkpoint_save_interval
    @property
    def CHECKPOINT_SAVE_NUMBERS(self) -> int:
        r"""
        Maximal number of checkpoints to preserve at any point.
        Default value is ``3``.
        """
        return self._CHECKPOINT_SAVE_NUMBERS
    @CHECKPOINT_SAVE_NUMBERS.setter
    def CHECKPOINT_SAVE_NUMBERS(self, checkpoint_save_numbers: int) -> None:
        self._CHECKPOINT_SAVE_NUMBERS = checkpoint_save_numbers
    @property
    def PRINT_LOSS_INTERVAL(self) -> int:
        r"""
        Print loss values every n epochs.
        Default value is ``10``.
        """
        return self._PRINT_LOSS_INTERVAL
    @PRINT_LOSS_INTERVAL.setter
    def PRINT_LOSS_INTERVAL(self, print_loss_interval: int) -> None:
        self._PRINT_LOSS_INTERVAL = print_loss_interval
    @property
    def TENSORBOARD_FLUSH_SECS(self) -> int:
        r"""
        Flush tensorboard logs to file every n seconds.
        Default values is ``5``.
        """
        return self._TENSORBOARD_FLUSH_SECS
    @TENSORBOARD_FLUSH_SECS.setter
    def TENSORBOARD_FLUSH_SECS(self, tensorboard_flush_secs: int) -> None:
        self._TENSORBOARD_FLUSH_SECS = tensorboard_flush_secs
    @property
    def ALLOW_TRAINING_INTERRUPTION(self) -> bool:
        r"""
        Allow interruption before model training converges.
        Default values is ``True``.
        """
        return self._ALLOW_TRAINING_INTERRUPTION
    @ALLOW_TRAINING_INTERRUPTION.setter
    def ALLOW_TRAINING_INTERRUPTION(self, allow_training_interruption: bool) -> None:
        self._ALLOW_TRAINING_INTERRUPTION = allow_training_interruption

config = ConfigManager()

@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    used_device = -1
    if not config.CPU_ONLY:
        try:
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free for i in range(pynvml.nvmlDeviceGetCount())
            ])
            for item in config.MASKED_GPUS:
                free_mems[item] = -1
            best_devices = np.where(free_mems == free_mems.max())[0]
            used_device = np.random.choice(best_devices, 1)[0]
            if free_mems[used_device] < 0:
                used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")

@logged
def prod(x: Iterable):
    try:
        from math import prod  # pylint: disable=redefined-outer-name
        return prod(x)
    except ImportError:
        ans = 1
        for item in x:
            ans = ans * item
        return ans
