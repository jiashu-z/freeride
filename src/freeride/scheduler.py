from . import scheduler_pb2
from . import scheduler_pb2_grpc
from . import task_runner
from . import task_runner_pb2
from . import task
import grpc
import time
from threading import Lock, Thread
from concurrent import futures
import argparse
import logging
from typing import Optional
import signal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)


class TaskRunnerMeta:
    def __init__(self, device: str, addr: str, id: int):
        self.device = device
        self.id = id
        self.addr: str = addr
        self.client = task_runner.TaskRunnerClient(self.addr)

    def __repr__(self) -> str:
        return f"{{device: {self.device}, id: {self.id}, addr: {self.addr}}}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskRunnerMeta):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(tuple([self.device, self.id, self.addr]))


class TaskMeta:
    def __init__(
        self, task_id: int, addr: str, priority: int, task_runner: TaskRunnerMeta
    ):
        self.task_id: int = task_id
        self.addr: str = addr
        self.client = task.TaskClient(addr)
        self.priority: int = priority
        self.task_runner: TaskRunnerMeta = task_runner

    def __repr__(self) -> str:
        return (
            f"{{task_id: {self.task_id}, addr: {self.addr}, priority: {self.priority}}}"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskMeta):
            return False
        return self.task_id == other.task_id


class Bubble:
    def __init__(self, start: float, end: float, stage_id, global_rank, device):
        self.start: float = start
        self.end: float = end
        self.stage_id: int = stage_id
        self.global_rank: int = global_rank
        self.device: str = str(device)

    def is_expired(self) -> bool:
        return time.time() > self.end

    def is_active(self) -> bool:
        current_time = time.time()
        return current_time >= self.start and current_time < self.end

    def __repr__(self) -> str:
        return f"{{start: {self.start}, end: {self.end}, stage_id: {self.stage_id}, global_rank: {self.global_rank}, device: {self.device}}}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Bubble):
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.stage_id == other.stage_id
            and self.global_rank == other.global_rank
            and self.device == other.device
        )


class SchedulerServicer(scheduler_pb2_grpc.SchedulerServicer):
    def __init__(self):
        super().__init__()
        self._task_id_inc: int = 0
        self._worker_manager_inc: int = 0
        self.task_runners: list[TaskRunnerMeta] = []
        self.tasks: list[TaskMeta] = []
        self.task_runner_bubbles: dict[
            TaskRunnerMeta, list[tuple[Bubble, Optional[TaskMeta]]]
        ] = {}
        # self._bubbles: list[tuple[Bubble, Optional[TaskMeta]]] = []
        self._lock = Lock()
        self._schedule_running = True
        self._schedule_running_lock = Lock()
        self._schedule_runner = Thread(target=self.schedule, args=())
        self._schedule_runner.start()

    def AddTaskRunner(self, request: scheduler_pb2.AddTaskRunnerArgs, context):
        with self._lock:
            device: str = request.device
            addr: str = request.addr
            logger.info(f"Add task runner {device} on addr {addr}")
            worker_manager_meta = TaskRunnerMeta(device, addr, self._worker_manager_inc)
            self._worker_manager_inc += 1
            self.task_runners.append(worker_manager_meta)
            return scheduler_pb2.AddTaskRunnerReply(status=0, id=worker_manager_meta.id)

    def AddTask(self, request: scheduler_pb2.AddTaskArgs, context):
        """Add a new task."""
        with self._lock:
            name: str = request.name
            scheduler_addr: str = request.scheduler_addr
            cmd: str = request.cmd
            priority: int = request.priority
            task_id: int = self._task_id_inc
            self._task_id_inc += 1
            task_runner: TaskRunnerMeta = self.task_runners[
                self.select_task_runner("XXX")
            ]
            reply: task_runner_pb2.AddTaskToRunnerReply = task_runner.client.add_task(
                task_id=task_id, name=name, scheduler_addr=scheduler_addr, cmd=cmd
            )
            assert reply.status == 0
            logger.info(f"Address {reply.addr}")
            task_meta = TaskMeta(task_id, reply.addr, priority, task_runner)
            self.tasks.append(task_meta)
            return scheduler_pb2.AddTaskReply(
                status=0,
                task_id=task_id,
                addr=reply.addr,
                device_name=reply.device_name,
            )

    def RemoveTask(self, request: scheduler_pb2.RemoveTaskArgs, context):
        """Remove a task immediately even if it is running."""
        with self._lock:
            task_id = request.task_id
            for task in self.tasks:
                if task.task_id == task_id:
                    task.client.stop_task(task_id)
                    logger.info(f"Stop task: {task}")
                    self.tasks.remove(task)
                    return scheduler_pb2.RemoveTaskReply(status=0)
            return scheduler_pb2.RemoveTaskReply(status=1)

    def FinishTask(self, request: scheduler_pb2.FinishTaskArgs, context):
        """Finish a task on the task side and notify the scheduler."""
        with self._lock:
            task_id = request.task_id
            logger.info(f"Finish task {task_id}")
            for task_runner, bubbles in self.task_runner_bubbles.items():
                for bubble, task_meta in bubbles:
                    if task_meta is not None and task_id == task_meta.task_id:
                        logger.info(f"{task_runner}, {bubble}, {task_meta}")
                        bubbles.remove((bubble, task_meta))
                        bubbles.append((bubble, None))
                        task_runner.client.finish_task(task_id)
                        task_meta.client.stop_task(task_id)
                        logger.info(f"Stop task: {task_meta}")
                        self.tasks.append(task_meta)
                        return scheduler_pb2.FinishTaskReply(status=0)

    def AddBubble(self, request: scheduler_pb2.AddBubbleArgs, context):
        with self._lock:
            start = request.start
            end = request.end
            stage_id = request.stage_id
            global_rank = request.global_rank
            device = request.device
            bubble = Bubble(start, end, stage_id, global_rank, device)
            task_runner = self.task_runners[self.match_task_runner()]
            if task_runner not in self.task_runner_bubbles:
                self.task_runner_bubbles[task_runner] = []
            self.task_runner_bubbles[task_runner].append((bubble, None))
            # self._bubbles.append((bubble, None))
            logger.info(f"Add bubble {bubble} to {task_runner}")
            return scheduler_pb2.AddBubbleReply(status=0)

    def ClearBubble(self, request: scheduler_pb2.ClearBubbleArgs, context):
        with self._lock:
            stage_id = request.stage_id
            global_rank = request.global_rank
            device = request.device
            for task_runner, bubbles in self.task_runner_bubbles.items():
                for i, (bubble, task_meta) in enumerate(bubbles[::-1]):
                    if (
                        bubble.stage_id == stage_id
                        and bubble.global_rank == global_rank
                        and bubble.device == device
                    ):
                        bubbles.remove((bubble, task_meta))
                        # self._bubbles.remove((bubble, task_meta))
                        logger.info(f"Clear bubble: {bubble}")
                        if task_meta is not None:
                            task_meta.client.pause_task(task_meta.task_id)
                            logger.info(f"Pause task: {task_meta}")
                            self.tasks.append(task_meta)
            return scheduler_pb2.ClearBubbleReply(status=0)

    def select_task_runner(self, device: str) -> int:
        return 0

    def match_task_runner(self) -> int:
        return 0

    def schedule(self):
        while True:
            with self._schedule_running_lock:
                if not self._schedule_running:
                    logger.info("Break out of schedule loop")
                    break
            with self._lock:
                for task_runner, bubbles in self.task_runner_bubbles.items():
                    for i, (bubble, task_meta) in enumerate(bubbles[::-1]):
                        if bubble.is_expired():
                            bubbles.remove((bubble, task_meta))
                            logger.info(
                                f"Remove expired bubble: {bubble} from {task_runner}"
                            )
                            if task_meta is not None:
                                task_meta.client.pause_task(task_meta.task_id)
                                logger.info(f"Pause task: {task_meta}")
                                self.tasks.append(task_meta)
                        elif bubble.is_active() and task_meta is None:
                            tasks = list(
                                filter(
                                    lambda task: task.task_runner == task_runner,
                                    self.tasks,
                                )
                            )
                            if len(tasks) > 0:
                                task = tasks.pop(0)
                                bubbles[i] = (bubble, task)
                                task.client.start_task(task.task_id)
                                self.tasks.remove(task)
            time.sleep(0.01)

    def stop_schedule(self):
        logger.info("Stop schedule")
        with self._schedule_running_lock:
            self._schedule_running = False
        self._schedule_runner.join()
        for task_meta in self.tasks:
            logger.info(f"To stop task: {task_meta}")
            # task_meta.client.stop_task(task_meta.task_id)
            logger.info(f"Stop task: {task_meta}")
        for task_runner, bubbles in self.task_runner_bubbles.items():
            for i, (bubble, task_meta) in enumerate(bubbles[::-1]):
                if task_meta is not None:
                    logger.info(f"To stop task: {task_meta}")
                    # task_meta.client.stop_task(task_meta.task_id)
                    logger.info(f"Stop task: {task_meta}")
                    bubbles.remove((bubble, task_meta))
        logger.info("Stop schedule thread")


class Scheduler:
    def __init__(self, addr: str = "localhost:40051", max_workers: int = 10):
        self.addr: str = addr
        self._max_workers: int = max_workers
        self._servicer: SchedulerServicer = SchedulerServicer()
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )
        scheduler_pb2_grpc.add_SchedulerServicer_to_server(
            self._servicer, self._grpc_server
        )
        self._grpc_server.add_insecure_port(self.addr)

    def start_and_wait(self):
        logger.info(f"Start scheduler on addr {self.addr} and wait")
        self._grpc_server.start()
        self._grpc_server.wait_for_termination()

    def start(self):
        logger.info(f"Start scheduler on addr {self.addr}")
        self._grpc_server.start()

    def stop(self, grace: Optional[float]):
        logger.info(f"Stop scheduler on addr {self.addr}")
        self._servicer.stop_schedule()
        self._grpc_server.stop(grace)


class SchedulerClient:
    def __init__(self, addr: str = "localhost:40051"):
        self.addr: str = addr
        self._chan = grpc.insecure_channel(self.addr)
        self._stub = scheduler_pb2_grpc.SchedulerStub(self._chan)

    def add_bubble(
        self, start: float, end: float, stage_id: int, global_rank: int, device: str
    ) -> int:
        args = scheduler_pb2.AddBubbleArgs(
            start=start,
            end=end,
            stage_id=stage_id,
            global_rank=global_rank,
            device=device,
        )
        reply: scheduler_pb2.AddBubbleReply = self._stub.AddBubble(args)
        assert reply.status == 0
        return reply.status

    def clear_bubble(self, stage_id: int, global_rank: int, device: str) -> int:
        args = scheduler_pb2.ClearBubbleArgs(
            stage_id=stage_id,
            global_rank=global_rank,
            device=device,
        )
        reply: scheduler_pb2.ClearBubbleReply = self._stub.ClearBubble(args)
        assert reply.status == 0
        return reply.status

    def add_task_runner(
        self, device: str, addr: str
    ) -> scheduler_pb2.AddTaskRunnerReply:
        logger.info(f"Add task runner {device} on addr {addr}")
        args = scheduler_pb2.AddTaskRunnerArgs(device=device, addr=addr)
        reply: scheduler_pb2.AddTaskRunnerReply = self._stub.AddTaskRunner(args)
        assert reply.status == 0
        return reply

    def add_task(
        self, name: str, scheduler_addr: str, priority: int, cmd: str
    ) -> scheduler_pb2.AddTaskReply:
        args = scheduler_pb2.AddTaskArgs(
            name=name, scheduler_addr=scheduler_addr, cmd=cmd, priority=priority
        )
        reply: scheduler_pb2.AddTaskReply = self._stub.AddTask(args)
        assert reply.status == 0
        return reply

    def finish_task(self, task_id: int) -> int:
        args = scheduler_pb2.FinishTaskArgs(task_id=task_id)
        reply: scheduler_pb2.FinishTaskReply = self._stub.FinishTask(args)
        assert reply.status == 0
        return reply.status


def handler(signum, frame):
    global s
    logger.info(f"Received {signum}, stopping scheduler")
    s.stop(0)
    logger.info("Scheduler stopped")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--addr", type=str, default="localhost:40051")
    args = parser.parse_args()
    addr = args.addr
    s = Scheduler(addr=addr)
    signal.signal(signal.SIGINT, handler)
    s.start_and_wait()
