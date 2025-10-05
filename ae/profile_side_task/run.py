import subprocess
from bubblebandit.task_v2 import TaskClient
import time
import itertools
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import pynvml
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)

stopped = mp.Value("i", 0)

device_id: int = 2


def monitor(name: str):
    pynvml.nvmlInit()
    f = open(f"./{name}_monitor.txt", "w")
    device = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    while True:
        t: float = time.time()
        energy: float = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(device).used
        line = f"{t},{energy},{0},{memory}\n"
        f.write(line)
        time.sleep(0.1)
        if stopped.value == 1:  # type: ignore
            break
    f.close()
    pynvml.nvmlShutdown()


class WorkloadMeta:
    def __init__(self):
        pass

    def get_command(self) -> str:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()


class DeepLearningWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        model_name: str,
        side_task_type: str,
        implementation_type: str,
        batch_size: int,
        steps: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.side_task_type = side_task_type
        self.implementation_type = implementation_type
        self.batch_size = batch_size
        self.steps: int = steps

    def get_command(self) -> str:
        return (
            "python3 "
            + f"side_task/model_{self.side_task_type}/"
            + f"{self.model_name}_{self.side_task_type}_{self.implementation_type}.py"
            + f" --batch_size={self.batch_size} --steps={self.steps} --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"{self.model_name}_{self.side_task_type}_{self.implementation_type}_{self.batch_size}"


class PrWorkloadMeta(WorkloadMeta):
    def __init__(self, file_type: str, graph_prefix: str, max_iter: int):
        super().__init__()
        self.file_type: str = file_type
        self.graph_prefix: str = graph_prefix
        self.max_iter = max_iter

    def get_command(self) -> str:
        return (
            # "compute-sanitizer --tool memcheck --launch-timeout 100 --log-file compute_sanitizer.log --save compute_sanitizer.save " +
            "side_task/pr_side_task --file_type="
            + self.file_type
            + " --graph_prefix="
            + self.graph_prefix
            + " --max_iter="
            + str(self.max_iter)
            + " --symmetrize=0 --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"pr_{self.file_type}_{self.max_iter}"


class SgdWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        graph_file: str,
        max_iter: int,
        lbd: float = 0.05,
        step: float = 0.003,
        epsilon: float = 1,
    ):
        super().__init__()
        self.graph_file: str = graph_file
        self.lbd: float = lbd
        self.step: float = step
        self.max_iter: int = max_iter
        self.epsilon: float = epsilon

    def get_command(self) -> str:
        return (
            "side_task/sgd_side_task --graph_file="
            + self.graph_file
            + " --lambda="
            + str(self.lbd)
            + " --step="
            + str(self.step)
            + " --max_iter="
            + str(self.max_iter)
            + " --epsilon="
            + str(self.epsilon)
            + " --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"sgd"


class BfsWorkloadMeta(WorkloadMeta):
    def __init__(self, file_type: str, graph_prefix: str, max_iter: int):
        super().__init__()
        self.file_type: str = file_type
        self.graph_prefix: str = graph_prefix
        self.max_iter: int = max_iter

    def get_command(self) -> str:
        return (
            # "compute-sanitizer --tool memcheck --launch-timeout 100 --log-file compute_sanitizer.log --save compute_sanitizer.save " +
            "side_task/bfs_side_task --file_type="
            + self.file_type
            + " --graph_prefix="
            + self.graph_prefix
            + " --max_iter="
            + str(self.max_iter)
            + " --symmetrize=0 --reverse=0 --source_id=0 --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"bfs_{self.file_type}"


class ImageWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        jpeg_quality: int,
        resize_width: int,
        resize_height: int,
        image_num: int,
        max_iter: int,
    ):
        super().__init__()
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.jpeg_quality: int = jpeg_quality
        self.resize_width: int = resize_width
        self.resize_height: int = resize_height
        self.image_num: int = image_num
        self.max_iter: int = max_iter

    def get_command(self) -> str:
        return (
            "side_task/image_side_task --input_dir="
            + self.input_dir
            + " --output_dir="
            + self.output_dir
            + " --jpeg_quality="
            + str(self.jpeg_quality)
            + " --resize_width="
            + str(self.resize_width)
            + " --resize_height="
            + str(self.resize_height)
            + " --image_num="
            + str(self.image_num)
            + " --max_iter="
            + str(self.max_iter)
            + " --profiler_level=1"
        )

    def get_name(self) -> str:
        return "image"


models = [
    "resnet18",
    # "resnet34",
    "resnet50",
    # "resnet101",
    # "resnet152",
    "vgg19",
    # "lstm",
    # "transformer",
]
side_task_types = ["training"]
implementation_types = ["iterative"]
side_batch_sizes = [16, 32, 64]

dl_workloads = []
for model, side_task_type, implementation_type, batch_size in itertools.product(
    models, side_task_types, implementation_types, side_batch_sizes
):
    dl_workloads.append(
        DeepLearningWorkloadMeta(model, side_task_type, implementation_type, batch_size)
    )

file_types = ["mtx"]
graph_prefixes = ["/dev/shm/com-Orkut"]
max_iters = [100]
graph_workloads: list[WorkloadMeta] = []
for file_type, graph_prefix, max_iter in itertools.product(
    file_types, graph_prefixes, max_iters
):
    graph_workloads.append(PrWorkloadMeta(file_type, graph_prefix, max_iter))

device = f"cuda:{device_id}"
gpu_type: str = "ada6000"

os.environ["NCCL_BUFFSIZE"] = "33554432"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_IBEXT_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

name_prefix: str = f"schedule_{gpu_type}"

workloads = dl_workloads
workloads.append(PrWorkloadMeta("mtx", "/dev/shm/com-Orkut", 0))
workloads.append(SgdWorkloadMeta("/dev/shm/com-Orkut.mtx", 0))
# workloads.append(BfsWorkloadMeta("mtx", "/dev/shm/com-Orkut", 0))
workloads.append(
    ImageWorkloadMeta(
        "/dev/shm/image_input", "/dev/shm/image_output", 100, 16384, 16384, 1, 0
    )
)

for workload in workloads:
    workload_name: str = workload.get_name()
    name: str = f"{name_prefix}_{workload_name}"
    workload_command: str = workload.get_command()
    command_prefix = f"python3 -m bubblebandit.add_task -n {name} -s localhost:40052"
    command = f"{command_prefix} {workload_command} --duration=0"

    p_logger = subprocess.Popen(
        "python3 -m bubblebandit.logger --addr=localhost:40051".split(" ")
    )
    logger.info("logger started")
    time.sleep(1)

    p_scheduler = subprocess.Popen(
        f"python3 -m bubblebandit.scheduler_v1 --addr=localhost:40052 --memory_list=15,0,0,0 --name={name}".split(
            " "
        )
    )
    logger.info("scheduler started")
    time.sleep(1)

    p_task_runner = subprocess.Popen(
        f"python3 -m bubblebandit.task_runner --device={device} --addr=localhost:40053 --scheduler_addr=localhost:40052 --stage_id=0 --port_from=40060 --port_to=40070".split(
            " "
        )
    )
    logger.info("task runner started")
    time.sleep(1)

    p_add_side_task = subprocess.Popen(command.split(" "))
    logger.info("Add task started")
    # time.sleep(60)
    time.sleep(10)

    stopped.value = 0  # type: ignore
    energy_monitor_runner = mp.Process(target=monitor, args=(name,))
    energy_monitor_runner.start()

    timestamps: list[float] = []
    task_client: TaskClient = TaskClient("localhost:40060")
    timestamps.append(time.time())

    task_client.init_task(0)
    # time.sleep(30)
    time.sleep(10)
    timestamps.append(time.time())

    duration_list: list[float] = [100]
    for duration in duration_list:
        task_client.start_task(0, time.time() + duration)
        time.sleep(duration)
        timestamps.append(time.time())
        task_client.pause_task(0)
        time.sleep(1)
        timestamps.append(time.time())

    task_client.stop_task(0)
    timestamps.append(time.time())

    with open(os.path.join(f"{name}_bubble_time.txt"), "w") as f:
        for timestamp in timestamps:
            f.write(f"{timestamp}\n")

    stopped.value = 1  # type: ignore
    p_logger.send_signal(2)
    p_scheduler.send_signal(2)
    p_task_runner.send_signal(2)
    p_add_side_task.send_signal(2)
    energy_monitor_runner.join()
    p_logger.wait()
    p_scheduler.wait()
    p_task_runner.wait()
    p_add_side_task.wait()

    # time.sleep(30)
