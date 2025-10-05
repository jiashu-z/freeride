import json
import os
import time
import subprocess
import logging
import torch
from bubblebandit.task_v2 import TaskClient
from experiment_config import *
import json
import math
from copy import deepcopy
import pynvml

assert (
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] == "0"
), "Please set CUDA_MPS_CLIENT_PRIORITY to 0"
assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in os.environ, "???"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)

os.environ["NCCL_BUFFSIZE"] = "33554432"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_IBEXT_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = CUDA_LAUNCH_BLOCKING

name_prefix: str = f"e2e_freeride_{gpu_type}"

config = dict(
    name="xxxlarge",
    MODEL_SIZE=1.3,
    NUM_LAYERS=72,
    HIDDEN_SIZE=2048,
    NUM_ATTN_HEADS=16,
    LR=2.0e-4,
    MIN_LR=2.0e-5,
    stage_id_gpu_memory_map={
        0: 3,
        1: 9,
        2: 16,
        3: 24,
    },
    BATCH_SIZE=3,
    BATCH_NUM=4,
)

config_name = config["name"]
stage_id_gpu_memory_map: dict[int, float] = config["stage_id_gpu_memory_map"]
os.system("rm -rf tmp/*")
pynvml.nvmlInit()
torch.cuda.empty_cache()

mix_workloads = [
    PrWorkloadMeta("mtx", "/dev/shm/com-Orkut", 0),
    DeepLearningWorkloadMeta("resnet18", "training", "iterative", 64),
    ImageWorkloadMeta(
        "/dev/shm/image_input", "/dev/shm/image_output", 100, 16384, 16384, 1, 0
    ),
    DeepLearningWorkloadMeta("vgg19", "training", "iterative", 64),
]

workload_name = "mix"
name = f"{name_prefix}_{config_name}_{workload_name}_{config['BATCH_NUM']}"
experiment_name: str = f"{name}_{steps}"


logger.info(f"\n\n\nStart {experiment_name}")
while True:
    temperatures = [
        pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(i), 0)
        for i in range(4)
    ]
    if all([t <= 33 for t in temperatures]):
        break
    logger.info(f"Waiting for {experiment_name}, {temperatures}")
    time.sleep(1)

logger.info(f"Running {experiment_name}")
num_layers = config["NUM_LAYERS"]
hidden_size = config["HIDDEN_SIZE"]
num_attn_heads = config["NUM_ATTN_HEADS"]
train_options = f" \
    --steps {steps} \
    --backend nccl \
    --dp {dp} \
    --pp {pp} \
    -N {num_layers} \
    -dm {hidden_size} \
    -H {num_attn_heads} \
    --seq {SEQ_LEN} \
    --parts {PARTITIONS} \
    --name {experiment_name} \
    --profiler {PROFILER} \
    --logger_addr {logger_addr} \
    --config {config_name} \
    --batch_num {config['BATCH_NUM']}"
with open(deepspeed_config_template_path) as f:
    j = json.load(f)
    j["train_micro_batch_size_per_gpu"] = config["BATCH_SIZE"]
    j["train_batch_size"] = config["BATCH_SIZE"] * config["BATCH_NUM"]
    with open(f"config/{name}.json", "w") as fout:
        json.dump(j, fout, indent=True)
        fout.flush()
p_logger = subprocess.Popen(
    "python3 -m bubblebandit.logger --addr=localhost:40051".split(" ")
)
logger.info("logger started")
time.sleep(1)
p_scheduler = subprocess.Popen(
    f"python3 -m bubblebandit.scheduler_v1 --addr=localhost:40052 --enable_stop --memory_list={stage_id_gpu_memory_map[0]},{stage_id_gpu_memory_map[1]},{stage_id_gpu_memory_map[2]},{stage_id_gpu_memory_map[3]} --name={name}".split(
        " "
    )
)
logger.info("scheduler started")
time.sleep(1)

p_task_runners = []
for i in range(4):
    env = deepcopy(os.environ)
    env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = (
        f"0={stage_id_gpu_memory_map[0]}G,1={stage_id_gpu_memory_map[1]}G,2={stage_id_gpu_memory_map[2]}G,3={stage_id_gpu_memory_map[3]}G"
    )
    env["CUDA_MPS_CLIENT_PRIORITY"] = "1"
    p_task_runners.append(
        subprocess.Popen(
            f"python3 -m bubblebandit.task_runner --device=cuda:{i} --addr=localhost:{40053 + i} --scheduler_addr=localhost:40052 --stage_id={i} --port_from={40060 + i * 10} --port_to={40070 + i * 10}".split(
                " "
            ),
            env=env,
        )
    )
    time.sleep(2)
logger.info("task_runner started")
time.sleep(10)

p_tasks = []
for i, workload_meta in enumerate(mix_workloads):
    command_str: str = workload_meta.get_command()
    workload_name: str = workload_meta.get_name()
    command_prefix: str = (
        f"python3 -m bubblebandit.add_task -n {name}_{i} -s localhost:40052"
    )
    gpu_memory = int(
        math.ceil(
            float(characteristics[workload_name]["max_reserved_memory"]) / (1024**3)
        )
    )
    # Round up the step time to the nearest 0.01 sec.
    step_time: float = (
        math.ceil(float(characteristics[workload_name]["median_step_time"]) * 100) / 100
    ) * 1.2
    print(workload_name, gpu_memory, step_time)
    command: str = (
        f"{command_prefix} --gpu_memory={gpu_memory} {command_str} --duration={step_time}"
    )
    print("GPU Memory:", gpu_memory)

    command = f"{command}"
    print(command)
    env = deepcopy(os.environ)
    env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = (
        f"0={stage_id_gpu_memory_map[0]}G,1={stage_id_gpu_memory_map[1]}G,2={stage_id_gpu_memory_map[2]}G,3={stage_id_gpu_memory_map[3]}G"
    )
    env["CUDA_MPS_CLIENT_PRIORITY"] = "1"
    p_tasks.append(subprocess.Popen(command.split(" "), env=env))
    time.sleep(3)
time.sleep(60)

for _ in range(4):
    task_client = TaskClient(f"localhost:{40060 + _ * 10}")
    task_client.init_task(_)
    time.sleep(3)
time.sleep(10)

deepspeed_command = " ".join(
    f"deepspeed ./src1/train_simplegpt2.py {train_options} --deepspeed_config ./config/{name}.json".split()
)
f = open(f"{experiment_name}.log", "w")
env = deepcopy(os.environ)
env["CUDA_MPS_CLIENT_PRIORITY"] = "0"
if "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" in env:
    del env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"]
p_ds = subprocess.Popen(deepspeed_command.split(" "), stdout=f, env=env)
p_ds.wait()
f.close()
# os.system(
# f"deepspeed ./src1/train_simplegpt2.py {train_options} --deepspeed_config ./config/{name}.json > ds_{experiment_name}.log"
# )
logger.info(f"Running {name} finished")
p_logger.send_signal(2)
p_scheduler.send_signal(2)
for p_task_runner in p_task_runners:
    p_task_runner.send_signal(2)
for p_task in p_tasks:
    p_task.send_signal(2)

logger.info("Signal sent")

for p_task in p_tasks:
    p_task.wait()
for p_task_runner in p_task_runners:
    p_task_runner.wait()
p_scheduler.wait()
p_logger.wait()
# shutil.move("./log", f"./out/log_{name}")
pynvml.nvmlShutdown()
logger.info(f"Clean up {name} finished\n\n")
