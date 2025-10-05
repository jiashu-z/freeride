from tqdm import tqdm
import itertools
from run import DeepLearningWorkloadMeta, PrWorkloadMeta, SgdWorkloadMeta

side_task_types = ["training", "inference"]
implementation_types = ["iterative"]
side_batch_sizes = [8, 16, 32, 48, 64, 80, 96, 112, 128]

models = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg19",
    "lstm",
    # "transformer",
]

for side_task_type, implementation_type in tqdm(
    itertools.product(side_task_types, implementation_types)
):
    workload_characteristics_objs = {}
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

            global_min_ts = None

            def update_global_min_ts(ts, global_min_ts) -> float:
                if global_min_ts is None:
                    return ts
                else:
                    return min(ts, global_min_ts)

            global_min_ts = update_global_min_ts(np.min(timestamps), global_min_ts)
            timestamps = (timestamps - global_min_ts) / 1000000  # type: ignore

            GB = 1024 * 1024 * 1024

            plt.figure(figsize=[4, 6])  # type: ignore

            plt.subplot(3, 1, 1)
            # plt.plot(
            #     (ts_arr - global_min_ts) / 1000000,
            #     allocated_arr / GB,
            #     "X-",
            #     markersize=2,
            #     linewidth=1,
            # )
            plt.plot(
                (energy_ts_arr - global_min_ts) / 1000000,
                memory_arr / GB,
                "x-",
                markersize=2,
                linewidth=1,
            )
            for timestamp in timestamps:
                plt.axvline(x=timestamp, color="r", linestyle="--", linewidth=0.5)
            # print(timestamps)
            plt.xlim(-0.5, 10.5)
            if side_task_type == "inference":
                plt.ylim(0, 5)
                plt.yticks([0, 1, 2, 3, 4, 5])
            elif side_task_type == "training":
                plt.ylim(0, 15)
                plt.yticks([0, 3, 6, 9, 12, 15])
            plt.ylim(0)
            plt.ylabel("GMem (GB)")
            # plt.legend()
            plt.tight_layout()
            plt.xticks(np.linspace(0, 10, 11, endpoint=True))

            plt.subplot(3, 1, 2)
            plt.plot(
                (energy_ts_arr - global_min_ts) / 1000000,
                power_arr,
                "-",
                markersize=1,
                linewidth=1,
            )
            for timestamp in timestamps:
                plt.axvline(x=timestamp, color="r", linestyle="--", linewidth=0.5)
            plt.xlim(-0.5, 10.5)
            plt.ylim(0, 400)
            # plt.xlabel("Time (sec)")
            plt.ylabel("Power (W)")
            # plt.xticks(np.linspace(0, 14, 8, endpoint=True))
            plt.tight_layout()
            plt.xticks(np.linspace(0, 10, 11, endpoint=True))

            plt.subplot(3, 1, 3)
            durations = np.array(durations)
            duration_timestamps = (
                np.array(duration_timestamps) - global_min_ts / 1000000
            )
            plt.plot(duration_timestamps, durations, "x-", markersize=2, linewidth=1)
            for timestamp in timestamps:
                plt.axvline(x=timestamp, color="r", linestyle="--", linewidth=0.5)
            # plt.boxplot(durations, vert=False)
            # n, _, _ = plt.hist(durations / np.sum(durations), histtype="step", cumulative=True)
            # print(n)
            # plt.xlabel("Duration (sec)")
            # plt.xscale("log")
            # plt.ylim(0, 100)
            plt.xlim(-0.5, 10.5)
            if side_task_type == "inference":
                plt.ylim(0, 0.5)
            elif side_task_type == "training":
                plt.ylim(0, 1)
            plt.xlabel("Time (sec)")
            plt.ylabel("Step length (sec)")
            plt.xticks(np.linspace(0, 10, 11, endpoint=True))
            plt.tight_layout()

            plt.savefig(
                os.path.join(f"{case_name}_ada6000.pdf"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
            # plt.show()

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
            workload_characteristics_objs[workload_name] = workload_characteristics_obj