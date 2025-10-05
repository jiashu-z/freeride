from tqdm import tqdm
import itertools
from run import DeepLearningWorkloadMeta, PrWorkloadMeta, SgdWorkloadMeta
import os
import numpy as np
from matplotlib import pyplot as plt
import json

side_task_types = ["training"]
implementation_types = ["iterative"]
side_batch_sizes = [16, 32, 64]

models = [
    "resnet18",
    "resnet50",
    "vgg19",
]

gpu_type: str = "ada6000"
name_prefix: str = f"schedule_{gpu_type}"
workload_characteristics_objs = {}

for side_task_type, implementation_type in tqdm(
    itertools.product(side_task_types, implementation_types)
):
    for i, model in enumerate(models):
        max_allocated_memories = []
        max_reserved_memories = []
        avg_energy_consumptions = []
        avg_step_times = []
        median_step_times = []
        for batch_size in side_batch_sizes:
            bubble_size = 100
            workload = DeepLearningWorkloadMeta(
                model, side_task_type, implementation_type, batch_size
            )
            workload_name = workload.get_name()
            case_name = f"{name_prefix}_{workload_name}"

            with open(os.path.join(f"{case_name}_time_profile_0.txt")) as f:
                lines = f.readlines()
                timestamps_dict: dict[str, list[float]] = {}
                for line in lines:
                    line = line.strip()
                    tokens = line.split(",")
                    label = tokens[0]
                    timestamps = [float(ts) for ts in tokens[1:]]
                    timestamps_dict[label] = timestamps
                durations = np.array(timestamps_dict["RUNNING_STEP_END"]) - np.array(
                    timestamps_dict["RUNNING_STEP_START"]
                )
                duration_timestamps = timestamps_dict["RUNNING_STEP_START"]

            bubble_time_file = os.path.join(f"{case_name}_bubble_time.txt")
            energy_monitor_file = os.path.join(f"{case_name}_monitor.txt")

            timestamps = []
            with open(bubble_time_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    timestamp: float = float(line.strip()) * 1000000
                    timestamps.append(timestamp)
            timestamps = np.array(timestamps)  # type: ignore

            energy_ts_list = []
            energy_list = []
            memory_list = []
            with open(energy_monitor_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.split(",")
                    energy_ts = float(tokens[0]) * 1000000
                    energy = float(tokens[1])
                    energy_ts_list.append(energy_ts)
                    energy_list.append(energy / 1000)
                    memory_list.append(float(tokens[3]))
            energy_ts_arr = np.array(energy_ts_list)
            energy_arr = np.array(energy_list)
            power_arr = 1e6 * np.diff(energy_arr) / np.diff(energy_ts_arr)
            power_arr = np.insert(power_arr, 0, 0)
            memory_arr = np.array(memory_list)

            max_allocated_memories.append(np.max(memory_arr))
            max_reserved_memories.append(np.max(memory_arr))
            avg_energy_consumptions.append(np.mean(power_arr))
            avg_step_times.append(np.mean(durations))
            median_step_times.append(np.median(durations))
            workload_characteristics_obj = {
                "max_allocated_memory": np.max(memory_arr),
                "max_reserved_memory": np.max(memory_arr),
                "avg_energy_consumption": np.mean(power_arr),
                "avg_step_time": np.mean(durations),
                "median_step_time": np.median(durations),
            }
            
            characteristics_entry_name = f"{model}_{side_task_type}_{implementation_type}_{batch_size}"
            workload_characteristics_objs[characteristics_entry_name] = workload_characteristics_obj

for side_task_type in ['image', 'pr', 'sgd']:
    max_allocated_memories = []
    max_reserved_memories = []
    avg_energy_consumptions = []
    avg_step_times = []
    median_step_times = []

    case_name = f"{name_prefix}_{side_task_type}"
    if side_task_type == "pr":
        case_name += "_mtx_0"

    with open(os.path.join(f"{case_name}_time_profile_0.txt")) as f:
        lines = f.readlines()
        timestamps_dict: dict[str, list[float]] = {}
        for line in lines:
            line = line.strip()
            tokens = line.split(",")
            label = tokens[0]
            timestamps = [float(ts) for ts in tokens[1:-1]]
            timestamps_dict[label] = timestamps
        durations = np.array(timestamps_dict["RUNNING_STEP_END"]) - np.array(
            timestamps_dict["RUNNING_STEP_START"]
        )
        duration_timestamps = timestamps_dict["RUNNING_STEP_START"]
        
    bubble_time_file = os.path.join(f"{case_name}_bubble_time.txt")
    energy_monitor_file = os.path.join(f"{case_name}_monitor.txt")

    timestamps = []
    with open(bubble_time_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            timestamp: float = float(line.strip()) * 1000000
            timestamps.append(timestamp)
    timestamps = np.array(timestamps)  # type: ignore

    energy_ts_list = []
    energy_list = []
    memory_list = []
    with open(energy_monitor_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(",")
            energy_ts = float(tokens[0]) * 1000000
            energy = float(tokens[1])
            energy_ts_list.append(energy_ts)
            energy_list.append(energy / 1000)
            memory_list.append(float(tokens[3]))
    energy_ts_arr = np.array(energy_ts_list)
    energy_arr = np.array(energy_list)
    power_arr = 1e6 * np.diff(energy_arr) / np.diff(energy_ts_arr)
    power_arr = np.insert(power_arr, 0, 0)
    memory_arr = np.array(memory_list)

    max_allocated_memories.append(np.max(memory_arr))
    max_reserved_memories.append(np.max(memory_arr))
    avg_energy_consumptions.append(np.mean(power_arr))
    avg_step_times.append(np.mean(durations))
    median_step_times.append(np.median(durations))
    workload_characteristics_obj = {
        "max_allocated_memory": np.max(memory_arr),
        "max_reserved_memory": np.max(memory_arr),
        "avg_energy_consumption": np.mean(power_arr),
        "avg_step_time": np.mean(durations),
        "median_step_time": np.median(durations),
    }

    characteristics_entry_name = f"{side_task_type}"
    workload_characteristics_objs[characteristics_entry_name] = workload_characteristics_obj

with open(f"iterative_summary_{gpu_type}.json", "w") as f:
    json.dump(workload_characteristics_objs, f, indent=4)