import json
import os
import pandas as pd


def run(dir: str, microbatch_num: int):
    files = sorted(filter(lambda x: x.endswith(".json"), os.listdir(dir)))

    df = pd.DataFrame(
        columns=[
            "GPU",
            "Bubble",
            "Type",
            "Start time",
            "End time",
            "Memory consumption (GB)",
            "Duration (sec)",
            "Memory (GB)",
        ]
    )

    for file in files:
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)
            stage = data["distributedInfo"]["rank"]
            trace_events = list(
                filter(
                    lambda x: (
                        "args" in x
                        and "stream" in x["args"]
                        and x["args"]["stream"] == 7
                    )
                    or "name" in x
                    and x["name"] == "ProfilerStep#1",
                    data["traceEvents"],
                )
            )
            profiler_step_event = list(
                filter(
                    lambda x: "name" in x and x["name"] == "ProfilerStep#1",
                    trace_events,
                )
            )[0]
            profiler_step_start = profiler_step_event["ts"]
            profiler_step_end = profiler_step_event["dur"] + profiler_step_event["ts"]
            print(
                f"Stage {stage} profiler step start: {profiler_step_start / 1E6}, end: {profiler_step_end / 1E6}"
            )

            profiler_memory_event = list(
                filter(
                    lambda x: "name" in x and x["name"] == "[memory]",
                    data["traceEvents"],
                )
            )
            print(f"count(profiler_memory_event) is {len(profiler_memory_event)}")
            profiler_max_memory_util = 0
            for x in profiler_memory_event:
                if profiler_max_memory_util < x["args"]["Total Reserved"]:
                    profiler_max_memory_util = x["args"]["Total Reserved"]
            print(f"profiler_max_memory_util is {profiler_max_memory_util}")
            profiler_max_memory_util = profiler_max_memory_util / 1024 / 1024 / 1024
            total_memory = 50876841984 / 1024 / 1024 / 1024

            kernel_events = list(
                filter(
                    lambda x: "args" in x
                    and "stream" in x["args"]
                    and x["args"]["stream"] == 7
                    and x["ts"] >= profiler_step_start
                    and x["ts"] <= profiler_step_end,
                    trace_events,
                )
            )
            kernel_events = list(sorted(kernel_events, key=lambda x: x["ts"]))
            # Compute the gaps between kernel events (not covered by any kernel event)
            gaps = []
            prev_end = profiler_step_start
            for event in kernel_events:
                start = event["ts"]
                end = event["ts"] + event["dur"]
                if start > prev_end:
                    gaps.append((prev_end, start))
                prev_end = max(prev_end, end)
            if prev_end < profiler_step_end:
                gaps.append((prev_end, profiler_step_end))
            gaps = list(map(lambda x: (x[0] / 1e6, x[1] / 1e6), gaps))
            gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)

            if len(gaps) < 6:
                print(f"Stage {stage} has less than 6 gaps, {gaps}")
            elif stage == 0:
                # if microbatch_num == 8:
                selected_gaps = gaps[:4]
                selected_gaps = sorted(selected_gaps, key=lambda x: x[0])
                selected_gaps_with_duration = [
                    (gap[0], gap[1], gap[1] - gap[0]) for gap in selected_gaps
                ]
                print(
                    f"Stage {stage} with microbatch {microbatch_num}: {selected_gaps_with_duration}"
                )
                types = ["B", "D", "D", "D"]
                for i, gap in enumerate(selected_gaps):
                    df = df._append(
                        {
                            "GPU": stage,
                            "Bubble": i,
                            "Type": types[i],
                            "Start time": gap[0],
                            "End time": gap[1],
                            "Memory consumption (GB)": profiler_max_memory_util,
                            "Duration (sec)": gap[1] - gap[0],
                            "Memory (GB)": total_memory - profiler_max_memory_util,
                        },
                        ignore_index=True,
                    )
            elif stage == 1:
                # if microbatch_num == 8:
                selected_gaps = gaps[:5]
                selected_gaps = sorted(selected_gaps, key=lambda x: x[0])
                selected_gaps_with_duration = [
                    (gap[0], gap[1], gap[1] - gap[0]) for gap in selected_gaps
                ]
                print(
                    f"Stage {stage} with microbatch {microbatch_num}: {selected_gaps_with_duration}"
                )
                types = ["A", "B", "D", "D", "C"]
                for i, gap in enumerate(selected_gaps):
                    df = df._append(
                        {
                            "GPU": stage,
                            "Bubble": i,
                            "Type": types[i],
                            "Start time": gap[0],
                            "End time": gap[1],
                            "Memory consumption (GB)": profiler_max_memory_util,
                            "Duration (sec)": gap[1] - gap[0],
                            "Memory (GB)": total_memory - profiler_max_memory_util,
                        },
                        ignore_index=True,
                    )
            elif stage == 2:
                # if microbatch_num == 8:
                selected_gaps = gaps[:4]
                selected_gaps = sorted(selected_gaps, key=lambda x: x[0])
                selected_gaps_with_duration = [
                    (gap[0], gap[1], gap[1] - gap[0]) for gap in selected_gaps
                ]
                print(
                    f"Stage {stage} with microbatch {microbatch_num}: {selected_gaps_with_duration}"
                )
                types = ["A", "B", "D", "C"]
                for i, gap in enumerate(selected_gaps):
                    df = df._append(
                        {
                            "GPU": stage,
                            "Bubble": i,
                            "Type": types[i],
                            "Start time": gap[0],
                            "End time": gap[1],
                            "Memory consumption (GB)": profiler_max_memory_util,
                            "Duration (sec)": gap[1] - gap[0],
                            "Memory (GB)": total_memory - profiler_max_memory_util,
                        },
                        ignore_index=True,
                    )
            elif stage == 3:
                # if microbatch_num == 8:
                selected_gaps = gaps[:1]
                selected_gaps = sorted(selected_gaps, key=lambda x: x[0])
                selected_gaps_with_duration = [
                    (gap[0], gap[1], gap[1] - gap[0]) for gap in selected_gaps
                ]
                print(
                    f"Stage {stage} with microbatch {microbatch_num}: {selected_gaps_with_duration}"
                )
                types = ["A"]
                for i, gap in enumerate(selected_gaps):
                    df = df._append(
                        {
                            "GPU": stage,
                            "Bubble": i,
                            "Type": types[i],
                            "Start time": gap[0],
                            "End time": gap[1],
                            "Memory consumption (GB)": profiler_max_memory_util,
                            "Duration (sec)": gap[1] - gap[0],
                            "Memory (GB)": total_memory - profiler_max_memory_util,
                        },
                        ignore_index=True,
                    )
    df.to_csv(os.path.join(dir, f"bubble_summary_{microbatch_num}.csv"), index=False)


if __name__ == "__main__":
    models = ["xlarge", "xxxlarge", "xxxxxlarge"]
    microbatch_numbers = [4, 6, 8]
    for model in models:
        for microbatch_number in microbatch_numbers:
            run(
                f"./log/e2e_vanilla_deepspeed_baseline_ada6000_{model}_10_{microbatch_number}",
                microbatch_number,
            )
