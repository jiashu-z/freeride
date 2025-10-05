from typing import final
from enum import Enum
import multiprocessing as mp
import os
import sys
import signal
import time

import torch

from . import task_pb2
from . import task_pb2_grpc
import grpc
import logging
import argparse
from . import scheduler
from concurrent import futures
from torchvision import models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    f"%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_task_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-s", "--scheduler_addr", type=str)
    parser.add_argument("-d", "--device", type=str, help="Device for the task")
    parser.add_argument("-i", "--task_id", type=int, help="Task ID")
    parser.add_argument("-a", "--addr", type=str, help="Address for the task server")
    parser.add_argument("-p", "--profiler_level", type=int, help="Set profiler level")
    return parser


def get_task_duration(task_type: str, batch_size: int, model_name: str) -> float:
    if task_type == "inference":
        return 0.1
    if batch_size <= 32:
        return 0.1
    if model_name in ("resnet18", "resnet32", "resnet50"):
        return 0.1
    return 0.2


def get_model_init_and_weights(model_name: str):
    model_name_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "vgg19": models.vgg19,
    }
    model_name_weights_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
        "resnet152": models.ResNet152_Weights.IMAGENET1K_V1,
        "vgg19": models.VGG19_Weights.IMAGENET1K_V1,
    }
    return model_name_map[model_name], model_name_weights_map[model_name]


class State(Enum):
    SUBMITTED = 0
    CREATED = 1
    PAUSED = 2
    RUNNING = 3


class TaskContext:
    def __init__(self):
        pass


class Task:
    """Template for task. The user should implement this."""

    parser = get_task_parser()

    @classmethod
    def get_parser(cls):
        return cls.parser

    class ProfilerLevel(Enum):
        NO_PROFILER = 0
        FUNCTION_TIME = 1
        PYTORCH_PROFILER = 2

    def __init__(
        self,
        task_id: int,
        task_name: str,
        device: str,
        scheduler_addr: str,
        profiler_level: int,
        state: State,
    ):
        """Initialize the task.

        Args:
            task_id (int): _description_
            task_name (str): _description_
            device (str): _description_
            scheduler_addr (str): _description_
            profiler_level (int): 0: no profiler, 1: function time, 2: function time and PyTorch Profiler
            state (State): _description_
        """
        self.task_id: int = task_id
        self.task_name: str = task_name
        self.device: str = device
        self.scheduler_addr: str = scheduler_addr
        self.profiler_level: int = profiler_level
        self.state: State = state

        self.scheduler_client: scheduler.SchedulerClient = scheduler.SchedulerClient(
            self.scheduler_addr
        )

        self.init_event = mp.Event()
        self.start_event = mp.Event()
        self.pause_event = mp.Event()
        self.stop_event = mp.Event()
        self.preempt_event = mp.Event()

        self.runner: mp.Process = mp.Process(target=self.run, args=())
        self.duration: float = 0.2
        self.end_time = mp.Value("d", -1)
        self.context: TaskContext = TaskContext()
        self.time_records: dict[str, list[float]] = {}

    def record_time(self, action: str):
        if action not in self.time_records:
            self.time_records[action] = []
        self.time_records[action].append(time.time())

    def init(self, task_id: int) -> int:
        """Scheduler API to initialize the task."""
        assert task_id == self.task_id
        logger.info(f"Init task {self.task_id}")
        self.init_event.set()
        return 0

    def start(self, task_id: int, end_time: float) -> int:
        """Scheduler API to start the task."""
        assert task_id == self.task_id
        logger.info(f"Start task {self.task_id}")
        self.end_time.value = end_time  # type: ignore
        self.start_event.set()
        return 0

    def pause(self, task_id: int) -> int:
        """Scheduler API to pause the task."""
        assert task_id == self.task_id
        logger.info(f"Pause task {self.task_id}")
        self.pause_event.set()
        return 0

    def stop(self, task_id: int) -> int:
        """Scheduelr API to stop the task."""
        assert task_id == self.task_id
        logger.info(f"Stop task {self.task_id}")
        self.stop_event.set()
        self.runner.join()
        os.kill(os.getpid(), signal.SIGINT)
        return 0

    def preempt(self, task_id: int) -> int:
        """Scheduler API to preempt the task."""
        assert task_id == self.task_id
        logger.info(f"Preempt task {self.task_id}")
        self.preempt_event.set()
        return 0

    def run(self):
        """Implement the task logic here."""
        if self.profiler_level >= self.ProfilerLevel.FUNCTION_TIME.value:
            self.record_time("SUBMITTED_TO_CREATED_START")
        self.submitted_to_created()
        self.state = State.CREATED
        if self.profiler_level >= self.ProfilerLevel.FUNCTION_TIME.value:
            self.record_time("SUBMITTED_TO_CREATED_END")

        while True:
            match self.state:
                case State.CREATED:
                    if self.init_event.is_set():
                        self.init_event.clear()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("CREATED_TO_PAUSED_START")
                        self.created_to_paused()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("CREATED_TO_PAUSED_END")
                        self.state = State.PAUSED
                        continue
                case State.PAUSED:
                    if self.start_event.is_set():
                        self.start_event.clear()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("PAUSED_TO_RUNNING_START")
                        self.paused_to_running()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("PAUSED_TO_RUNNING_END")
                        self.state = State.RUNNING
                        continue
                    if self.preempt_event.is_set():
                        self.preempt_event.clear()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("PAUSED_TO_CREATED_START")
                        self.paused_to_created()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("PAUSED_TO_CREATED_END")
                        self.state = State.CREATED
                        continue
                case State.RUNNING:
                    if self.pause_event.is_set():
                        self.pause_event.clear()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_PAUSED_START")
                        self.running_to_paused()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_PAUSED_END")
                        self.state = State.PAUSED
                        logger.info("State from RUNNING to PAUSED")
                        continue
                    if self.preempt_event.is_set():
                        self.preempt_event.clear()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_CREATED_START")
                        self.running_to_created()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_CREATED_END")
                        self.state = State.CREATED
                        logger.info("State from RUNNING to CREATED")
                        continue
                    if self.is_finished():
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_FINISHED_START")
                        self.running_to_finished()
                        if (
                            self.profiler_level
                            >= self.ProfilerLevel.FUNCTION_TIME.value
                        ):
                            self.record_time("RUNNING_TO_FINISHED_END")
                        break
                    else:
                        if self.do_i_have_enough_time():
                            if (
                                self.profiler_level
                                >= self.ProfilerLevel.FUNCTION_TIME.value
                            ):
                                self.record_time("RUNNING_STEP_START")
                            self.step()
                            if (
                                self.profiler_level
                                >= self.ProfilerLevel.FUNCTION_TIME.value
                            ):
                                self.record_time("RUNNING_STEP_END")
                        else:
                            if (
                                self.profiler_level
                                >= self.ProfilerLevel.FUNCTION_TIME.value
                            ):
                                self.record_time("RUNNING_STEP_START")
                            end_time: float = self.end_time.value  # type: ignore
                            logger.info(
                                f"Not enough time, current time: {time.time()}, end time: {end_time}"
                            )
                            if end_time - time.time() > 0.001:
                                time.sleep(end_time - time.time())
                            if (
                                self.profiler_level
                                >= self.ProfilerLevel.FUNCTION_TIME.value
                            ):
                                self.record_time("RUNNING_STEP_END")
            if self.stop_event.is_set():
                # self.stop_event.clear()
                if self.profiler_level >= self.ProfilerLevel.FUNCTION_TIME.value:
                    self.record_time("TO_STOPPED_START")
                self.to_stopped()
                if self.profiler_level >= self.ProfilerLevel.FUNCTION_TIME.value:
                    self.record_time("TO_STOPPED_END")
                break
            time.sleep(0.001)

        if self.profiler_level >= self.ProfilerLevel.FUNCTION_TIME.value:
            with open(f"./out/{self.task_name}_time_profile_{self.task_id}.txt", "w") as f:
                for action in self.time_records:
                    row = f"{action},{','.join([str(t) for t in self.time_records[action]])}\n"
                    f.write(row)

        if self.stop_event.is_set():
            self.stop_event.clear()
        else:
            status: int = self.scheduler_client.finish_task(self.task_id)
            assert status == 0, "Failed to finish task"
            logger.info("Task finished")

    def finish(self):
        """Finish a task on the task side and notify the scheduler."""
        status: int = self.scheduler_client.finish_task(self.task_id)
        assert status == 0
        logger.info(f"Finish task {self.task_id}")

    def start_runner(self):
        """Start the task process."""
        logger.info(f"Start runner of task {self.task_id}")
        self.runner.start()

    def submitted_to_created(self) -> None:
        pass

    def created_to_paused(self) -> None:
        self.move_tensor_variables_to(self.device)

    def paused_to_running(self) -> None:
        pass

    def paused_to_created(self) -> None:
        self.move_tensor_variables_to("cpu")

    def running_to_paused(self) -> None:
        pass

    def running_to_created(self) -> None:
        self.move_tensor_variables_to("cpu")

    def running_to_finished(self) -> None:
        pass

    def to_stopped(self) -> None:
        pass

    def is_finished(self) -> bool:
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def move_tensor_variables_to(self, device: str) -> None:
        d = self.__dict__
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device)
            elif isinstance(v, torch.nn.Module):
                d[k] = v.to(device)
            elif isinstance(v, torch.optim.Optimizer):
                for state in v.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
        if device == "cpu":
            torch.cuda.empty_cache()

    def do_i_have_enough_time(self) -> bool:
        return self.end_time.value - time.time() > self.duration  # type: ignore


@final
class TaskServicer(task_pb2_grpc.TaskServicer):
    def __init__(self, task: Task):
        super().__init__()
        self.task: Task = task

    def InitTask(self, request: task_pb2.InitTaskArgs, context):
        task_id: int = request.task_id
        status: int = self.task.init(task_id)
        assert status == 0, f"status: {status}"
        return task_pb2.InitTaskReply(status=status)

    def StartTask(self, request: task_pb2.StartTaskArgs, context):
        task_id: int = request.task_id
        end_time: float = request.end_time
        status: int = self.task.start(task_id, end_time)
        assert status == 0, f"status: {status}"
        return task_pb2.StartTaskReply(status=status)

    def PauseTask(self, request: task_pb2.PauseTaskArgs, context):
        task_id: int = request.task_id
        status: int = self.task.pause(task_id)
        assert status == 0, f"status: {status}"
        return task_pb2.PauseTaskReply(status=status)

    def StopTask(self, request: task_pb2.StopTaskArgs, context):
        task_id: int = request.task_id
        status: int = self.task.stop(task_id)
        assert status == 0, f"status: {status}"
        return task_pb2.StopTaskReply(status=status)

    def PreemptTask(self, request: task_pb2.PreemptTaskArgs, context):
        task_id: int = request.task_id
        status: int = self.task.preempt(task_id)
        assert status == 0, f"status: {status}"
        return task_pb2.PreemptTaskReply(status=status)


class TaskClient:
    def __init__(self, addr: str):
        self.addr = addr
        self.chann = grpc.insecure_channel(self.addr)
        self.stub = task_pb2_grpc.TaskStub(self.chann)

    def init_task(self, task_id: int) -> task_pb2.InitTaskReply:
        args: task_pb2.InitTaskArgs = task_pb2.InitTaskArgs(task_id=task_id)
        reply: task_pb2.InitTaskReply = self.stub.InitTask(args)
        assert reply.status == 0
        return reply

    def start_task(self, task_id: int, end_time: float) -> task_pb2.StartTaskReply:
        args: task_pb2.StartTaskArgs = task_pb2.StartTaskArgs(
            task_id=task_id, end_time=end_time
        )
        reply: task_pb2.StartTaskReply = self.stub.StartTask(args)
        assert reply.status == 0
        return reply

    def pause_task(self, task_id: int) -> task_pb2.PauseTaskReply:
        args: task_pb2.PauseTaskArgs = task_pb2.PauseTaskArgs(task_id=task_id)
        reply: task_pb2.PauseTaskReply = self.stub.PauseTask(args)
        assert reply.status == 0
        return reply

    def stop_task(self, task_id: int) -> task_pb2.StopTaskReply:
        args: task_pb2.StopTaskArgs = task_pb2.StopTaskArgs(task_id=task_id)
        reply: task_pb2.StopTaskReply = self.stub.StopTask(args)
        assert reply.status == 0
        return reply

    def preempt_task(self, task_id: int) -> task_pb2.PreemptTaskReply:
        args: task_pb2.PreemptTaskArgs = task_pb2.PreemptTaskArgs(task_id=task_id)
        reply: task_pb2.PreemptTaskReply = self.stub.PreemptTask(args)
        assert reply.status == 0
        return reply


def run_task_server(task_init, args, **kwargs):
    logger.info(args)
    assert args.steps >= 0, "Steps should be non-negative"
    grpc_server: grpc.Server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    task: Task = task_init(
        task_id=args.task_id,
        task_name=args.name,
        device=args.device,
        scheduler_addr=args.scheduler_addr,
        profiler_level=args.profiler_level,
        state=State.SUBMITTED,
        **kwargs,
    )

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        grpc_server.stop(1)
        logger.info(f"Exit task {task.task_id}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    servicer: TaskServicer = TaskServicer(task=task)
    task_pb2_grpc.add_TaskServicer_to_server(servicer, grpc_server)
    grpc_server.add_insecure_port(args.addr)
    grpc_server.start()
    task.start_runner()
    grpc_server.wait_for_termination()
