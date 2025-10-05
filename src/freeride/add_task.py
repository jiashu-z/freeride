from . import scheduler_v1
import argparse


def get_parser() -> argparse.ArgumentParser:
    """Called by add_task.py

    Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-s", "--scheduler_addr", type=str)
    parser.add_argument(
        "-r", "--priority", type=int, help="Priority for the task", default=1
    )
    parser.add_argument(
        "-m", "--gpu_memory", type=int, help="GPU memory to use", default=0
    )
    parser.add_argument(
        "commands", metavar="C", type=str, nargs="+", help="commands to run"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    name: str = args.name
    scheduler_addr: str = args.scheduler_addr
    priority: int = args.priority
    cmd = " ".join(args.commands) + " " + " ".join(unknown)
    cmd = cmd.strip()
    gpu_memory = args.gpu_memory
    client: scheduler_v1.SchedulerClient = scheduler_v1.SchedulerClient(scheduler_addr)
    client.add_task(name, scheduler_addr, priority, cmd, gpu_memory)
