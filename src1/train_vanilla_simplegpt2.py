import deepspeed
import argparse
import torch
from dataset import SimpleDataset
from deepspeed.pipe import PipelineModule
import numpy as np
import torch.nn as nn
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from simplegpt import SimpleGPT, GPTConfig
from bubblebandit.logger import LoggerClient as LoggerClient
import os
import time
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTotalEnergyConsumption,
)
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--steps", type=int, default=10, help="quit after this many steps"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="distributed backend"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher.",
    )
    parser.add_argument("--dp", type=int, default=1, help="size of data parallelism")
    parser.add_argument(
        "--pp", type=int, default=4, help="size of pipeline parallelism"
    )
    parser.add_argument("--seed", type=int, default=7777, help="seed")

    # Model config args
    parser.add_argument("-N", type=int, default=24)
    parser.add_argument("--d-model", "-dm", type=int, default=1024)
    parser.add_argument("-H", type=int, default=16)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument(
        "--parts",
        type=str,
        default="",
        help="Specify number of layers for each partition; separated by comma like `1,2,2,3`",
    )
    parser.add_argument(
        "--aci", type=int, default=0, help="Activation checkpoint interval"
    )
    parser.add_argument("--logger_addr", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--profiler", type=int)
    parser.add_argument("--config", type=str)
    parser.add_argument("--batch_num", type=int)
    parser.add_argument("--time_per_step", type=int)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def init_dist(args):
    deepspeed.init_distributed(args.backend)
    data_parallel_size = args.dp
    pipe_parallel_size = args.pp
    custom_topology = PipeDataParallelTopology(pipe_parallel_size, data_parallel_size)

    return {
        "data_parallel_size": data_parallel_size,
        "pipe_parallel_size": pipe_parallel_size,
        "topo": custom_topology,
    }


def gen_parts(args):
    parts = []
    if args.parts and args.parts != "-":
        parts = [int(p) for p in args.parts.split(",")]
        assert sum(parts) == args.N
        parts[-1] += 2
        parts = [0] + [sum(parts[:i]) + p for i, p in enumerate(parts)]

    return parts


if __name__ == "__main__":
    assert (
        "CUDA_MPS_CLIENT_PRIORITY" in os.environ
    ), "Please set CUDA_MPS_CLIENT_PRIORITY to 0"
    assert (
        os.environ["CUDA_MPS_CLIENT_PRIORITY"] == "0"
    ), "Please set CUDA_MPS_CLIENT_PRIORITY to 0"
    assert (
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" not in os.environ
    ), "Please do not set memory limit for DeepSpeed"
    args = get_args()
    assert (
        args.profiler == 0 or args.profiler == 1
    ), "profiler must be 0 (w/o) or 1 (w/)"
    name: str = args.name
    with_profiler: bool = args.profiler == 1

    if with_profiler:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{name}"),
            profile_memory=True,
        )
        profiler.start()

    np.random.seed(args.seed)
    parts = gen_parts(args)
    dist_config = init_dist(args)
    config = GPTConfig(
        block_size=args.seq, n_layer=args.N, n_head=args.H, n_embd=args.d_model
    )
    layers = SimpleGPT(config).to_layers()
    # layers = SimpleGPT2(args.N, args.d_model, args.H).join_layers()
    model = PipelineModule(
        layers=layers,
        loss_fn=nn.MSELoss(),
        num_stages=dist_config["pipe_parallel_size"],
        # partition_method="type:DecoderLayerSimple" if len(parts) == 0 else "custom",
        #    custom_partitions=parts,
        # topology=dist_config["topo"],
        activation_checkpoint_interval=args.aci,
    )

    dataset = SimpleDataset(args.seq, args.d_model, 73728)

    print(f"pid: {os.getpid()}")

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
    )

    thresholds: dict[int, float] = {4: 4.122, 5: 4.714, 6: 5.359, 7: 6.051, 8: 6.609}
    threshold: float = thresholds[args.batch_num]

    # engine.counter = 0
    stage_id: int = engine.stage_id

    if stage_id in [0, 1, 2, 3]:
        energy_before = []
        energy_after = []
        handles = []
        device_num = nvmlDeviceGetCount()
        for d in range(device_num):
            handle = nvmlDeviceGetHandleByIndex(d)
            handles.append(handle)
        for handle in handles:
            energy_before.append(nvmlDeviceGetTotalEnergyConsumption(handle))

    t_s = 0
    for _ in range(args.steps):
        if _ == 1:
            t_s = time.time()
        engine.train_batch()
        # if _ * threshold <= time.time() - t_s:
        #     engine.stage_bubble_durations = zero_bubble_stat_dict[
        #         (args.config, args.batch_num)
        #     ]
        # else:
        #     engine.stage_bubble_durations = config_batch_num_bubble_stat_dict[
        #         (args.config, args.batch_num)
        #     ]
        # if (_ + 1) % 5 == 0:
        # t_s1 = time.time()
        # engine.save_checkpoint("./tmp")
        # t_e1 = time.time()
        # print(f"State {engine.stage_id} checkpoint time {t_e1 - t_s1}")
        if with_profiler:
            profiler.step()
    t_e = time.time()
    if with_profiler:
        profiler.stop()

    if stage_id in [0, 1, 2, 3]:
        for handle in handles:
            energy_after.append(nvmlDeviceGetTotalEnergyConsumption(handle))
        energy_consumptions = []
        for i in range(device_num):
            energy_consumptions.append(energy_after[i] - energy_before[i])
        result = {
            "stage_id": stage_id,
            "time": t_e - t_s,
            "energy": energy_consumptions,
            "start": t_s,
            "end": t_e,
            # "counter": 0,
        }
        with open(f"./out/{name}_stage{stage_id}.json", "w") as fout:
            json.dump(result, fout, indent=True)
