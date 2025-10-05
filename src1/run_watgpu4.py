import json
import os
import itertools
import time
import subprocess
import shutil
import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)

NCCL_BUFFSIZE = 33554432
NCCL_IB_DISABLE = 1
NCCL_P2P_DISABLE = 1

batch_size: int = 12
global_batch_size: int = 48
dp: int = 1
pp: int = 4

SEQ_LEN = 1024
NUM_GPUS = 4
PARTITIONS = "-"

logger_addr: str = "localhost:40051"
profiler: int = 0

steps_list: list[int] = [16, 32, 64, 128]
profiler_list: list[int] = [0]
blocking_list: list[str] = ["0", "1"]
side_task_batch_size_list: list[int] = [16, 32, 64, 128]

config = dict(
    name="xlarge",
    MODEL_SIZE=1.3,
    NUM_LAYERS=24,
    HIDDEN_SIZE=2048,
    NUM_ATTN_HEADS=16,
    LR=2.0e-4,
    MIN_LR=2.0e-5,
)

deepspeed_config_template_path = "./config/simplegpt2_template.json"

os.environ["NCCL_BUFFSIZE"] = "33554432"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_IBEXT_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

for steps, profiler, blocking, side_task_batch_size in itertools.product(
    steps_list, profiler_list, blocking_list, side_task_batch_size_list
):
    torch.cuda.empty_cache()
    if not os.path.exists("./log"):
        os.mkdir("./log")
    if not os.path.exists("./out"):
        os.mkdir("./out")
    os.environ["CUDA_LAUNCH_BLOCKING"] = blocking
    name = f"bubblebandit_watgpu4_steps{steps}_profiler{profiler}_blocking{blocking}"
    if side_task_batch_size > 0:
        name += f"_side{side_task_batch_size}"
    logger.info(f"=========================================================")
    logger.info(f"=============== RUNNING EXPERIMENT {name} ===============")
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
        --name {name} \
        --profiler {profiler} \
        --logger_addr {logger_addr}"
    with open(deepspeed_config_template_path) as f:
        j = json.load(f)
        j["train_micro_batch_size_per_gpu"] = batch_size
        j["train_batch_size"] = global_batch_size
        with open(f"config/{name}.json", "w") as fout:
            json.dump(j, fout, indent=True)
            fout.flush()
    p_logger = subprocess.Popen(
        "python3 -m bubblebandit.logger --addr=localhost:40051".split(" ")
    )
    logger.info("logger started")
    time.sleep(1)
    p_scheduler = subprocess.Popen(
        "python3 -m bubblebandit.scheduler --addr=localhost:40052".split(" ")
    )
    logger.info("scheduler started")
    time.sleep(1)
    p_task_runner = subprocess.Popen(
        "python3 -m bubblebandit.task_runner --device=cuda:3 --addr=localhost:40053 --scheduler_addr=localhost:40052".split(
            " "
        )
    )
    logger.info("task_runner started")
    time.sleep(1)
    if side_task_batch_size > 0:
        p_add_side_task = subprocess.Popen(
            f"python3 -m bubblebandit.add_task --scheduler=localhost:40052 --batch_size={side_task_batch_size} --name={name}".split(
                " "
            )
        )
        logger.info("add_side_task started")
        time.sleep(1)

    os.system(
        f"deepspeed ./src1/train_simplegpt2.py {train_options} --deepspeed_config ./config/{name}.json"
    )
    logger.info(f"training {name} finished")
    p_logger.send_signal(2)
    p_scheduler.send_signal(2)
    p_task_runner.send_signal(2)
    if side_task_batch_size > 0:
        p_add_side_task.send_signal(2)
    p_logger.wait()
    p_scheduler.wait()
    p_task_runner.wait()
    if side_task_batch_size > 0:
        p_add_side_task.wait()
    shutil.move("./log", f"./out/log_{name}")
    logger.info(f"cleanup {name} finished\n\n")
    time.sleep(30)
