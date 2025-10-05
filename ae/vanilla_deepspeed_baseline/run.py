import json
import os
import time
import logging
import torch
from experiment_config import *
import subprocess
import pynvml
from copy import deepcopy

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
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

name_prefix = f"e2e_vanilla_deepspeed_baseline_{gpu_type}"

for microbatch_num in [4, 6, 8]:
    for _config in configs:
        config = deepcopy(_config)
        config["BATCH_NUM"] = microbatch_num
        os.system("rm -rf ./tmp")
        os.system("mkdir ./tmp")
        pynvml.nvmlInit()
        torch.cuda.empty_cache()

        name: str = f"{name_prefix}"

        deepspeed_config_name: str = f"{config['name']}_{steps}_{config['BATCH_NUM']}"
        experiment_name: str = f"{name}_{deepspeed_config_name}"

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
            --steps 128 \
            --backend nccl \
            --dp {dp} \
            --pp {pp} \
            -N {num_layers} \
            -dm {hidden_size} \
            -H {num_attn_heads} \
            --seq {SEQ_LEN} \
            --parts {PARTITIONS} \
            --name {experiment_name} \
            --profiler 0 \
            --config {config_name} \
            --batch_num {config['BATCH_NUM']}"
        with open(deepspeed_config_template_path) as f:
            j = json.load(f)
            j["train_micro_batch_size_per_gpu"] = config["BATCH_SIZE"]
            j["train_batch_size"] = config["BATCH_SIZE"] * config["BATCH_NUM"]
            with open(f"config/{experiment_name}.json", "w") as fout:
                json.dump(j, fout, indent=True)
                fout.flush()

        deepspeed_command: str = " ".join(
            f"deepspeed ./src1/train_vanilla_simplegpt2.py {train_options} --deepspeed_config ./config/{experiment_name}.json".split()
        )
        f = open(f"{experiment_name}.log", "w")
        p_ds = subprocess.Popen(deepspeed_command.split(" "), stdout=f)

        p_ds.wait()
        f.close()

        pynvml.nvmlShutdown()
        logger.info(f"Running {name} finished")
        logger.info(f"Clean up {name} finished\n\n")
