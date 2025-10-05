from typing import final
from enum import Enum

from matplotlib.pylab import f

from . import task_pb2
from . import task_pb2_grpc
import grpc
import logging
import argparse
from . import scheduler

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
    parser.add_argument("-i", "--task_id", type=int, help="Task ID")
    parser.add_argument("-d", "--device", type=str, help="Device for the task")
    parser.add_argument("-a", "--addr", type=str, help="Address for the task server")
    parser.add_argument("-p", "--profiler", action="store_true", help="Enable profiler")
    return parser


def get_task_duration(task_type: str, batch_size: int, model_name: str) -> float:
    if task_type == "inference":
        return 0.1
    if batch_size <= 32:
        return 0.1
    if model_name in ("resnet18", "resnet32", "resnet50"):
        return 0.1
    return 0.2


class Task:
    """Template for task. The user should implement this."""

    class State(Enum):
        SUBMITTED = 0
        CREATED = 1
        PENDING = 2
        RUNNING = 3

    def __init__(self, task_id: int, task_name: str, device: str, scheduler_addr: str):
        self.task_id: int = task_id
        self.task_name: str = task_name
        self.device: str = device
        self.scheduler_addr: str = scheduler_addr
        self.scheduler_client: scheduler.SchedulerClient = scheduler.SchedulerClient(
            self.scheduler_addr
        )

    def init(self, task_id: int) -> int:
        """Scheduler API to initialize the task.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def start(self, task_id: int, end_time: float) -> int:
        """Scheduler API to start the task.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def pause(self, task_id: int) -> int:
        """Scheduler API to pause the task.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def stop(self, task_id: int) -> int:
        """Scheduelr API to stop the task.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def preempt(self, task_id: int) -> int:
        """Scheduler API to preempt the task.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def run(self):
        """Implement the task logic here.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def finish(self):
        """Finish a task on the task side and notify the scheduler.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def start_runner(self):
        """Start the task process.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


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
