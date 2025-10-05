from . import scheduler_pb2
from . import scheduler_pb2_grpc
from . import task_runner
from . import task_runner_pb2
from . import task_v2 as task
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
    "%(asctime)s - %(levelname)s - %(created)s - %(filename)s:%(lineno)d - %(message)s"
)

# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# ch.setFormatter(formatter)

# logger.addHandler(ch)

stage_id_gpu_memory_map: dict[int, int] = {
    0: 48,
    1: 48,
    2: 48,
    3: 48,
}


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
        current_time: float = time.time()
        return current_time >= self.start and current_time < self.end

    def get_end_time(self) -> float:
        return self.end

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


class TaskMeta:
    def __init__(
        self,
        task_id: int,
        addr: str,
        priority: int,
        gpu_memory: int,
        task_runner_index: int,
        cmd: str,
    ):
        self.task_id: int = task_id
        self.addr: str = addr
        self.client = task.TaskClient(addr)
        self.priority: int = priority
        self.gpu_memory: int = gpu_memory
        self.task_runner_index: int = task_runner_index
        self.cmd: str = cmd
        self.state: task.State = task.State.CREATED

    def __repr__(self) -> str:
        return f"{{task_id: {self.task_id}, addr: {self.addr}, priority: {self.priority}, gpu_memory: {self.gpu_memory}, state: {self.state}, cmd: {self.cmd}}}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskMeta):
            return False
        return self.task_id == other.task_id

    def init(self):
        logger.info(f"Init task {self}")
        self.client.init_task(self.task_id)
        self.state = task.State.PAUSED

    def start(self, end_time: float):
        logger.info(f"Start task start {self.task_id}, {end_time}")
        self.client.start_task(self.task_id, end_time)
        self.state = task.State.RUNNING
        logger.info(f"Start task end {self.task_id}, {end_time}")

    def pause(self):
        logger.info(f"Pause task start {self.task_id}")
        self.client.pause_task(self.task_id)
        self.state = task.State.PAUSED
        logger.info(f"Pause task end {self.task_id}")

    def stop(self):
        logger.info(f"Stop task start {self.task_id}")
        self.client.stop_task(self.task_id)
        self.state = task.State.SUBMITTED
        logger.info(f"Stop task end {self.task_id}")

    def preempt(self):
        logger.info(f"Preempt task start {self.task_id}")
        self.client.preempt_task(self.task_id)
        self.state = task.State.CREATED
        logger.info(f"Preempt task end {self.task_id}")

    def finish(self):
        logger.info(f"Finish task start {self.task_id}")
        self.state = task.State.SUBMITTED
        logger.info(f"Finish task end {self.task_id}")

    def is_submitted(self):
        return self.state == task.State.SUBMITTED

    def is_created(self):
        return self.state == task.State.CREATED

    def is_paused(self):
        return self.state == task.State.PAUSED

    def is_running(self):
        return self.state == task.State.RUNNING


class TaskRunnerMeta:
    def __init__(
        self, device: str, addr: str, stage_id: int, id: int, enable_stop: bool = True
    ):
        self.device = device
        self.id = id
        self.addr: str = addr
        self.stage_id: int = stage_id
        self.enable_stop: bool = enable_stop
        self.client = task_runner.TaskRunnerClient(self.addr)
        self.idle_bubbles: list[Bubble] = []
        self.current_bubble: Optional[Bubble] = None
        self.tasks: list[TaskMeta] = []
        self.current_task: Optional[TaskMeta] = None
        self.just_set_new_current_bubble: bool = False
        self.lock: Lock = Lock()

    def __repr__(self) -> str:
        return f"{{device: {self.device}, id: {self.id}, addr: {self.addr}, stage: {self.stage_id}}}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskRunnerMeta):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(tuple([self.device, self.id, self.addr]))

    def get_queue_length(self) -> int:
        with self.lock:
            return len(self.tasks) + (1 if self.current_task is not None else 0)

    def get_queue_length_with_priority(self, priority: int) -> int:
        with self.lock:
            return len(
                list(filter(lambda task: task.priority >= priority, self.tasks))
            ) + (
                1
                if self.current_task is not None
                and self.current_task.priority >= priority
                else 0
            )

    def add_bubble(self, bubble: Bubble) -> None:
        with self.lock:
            logger.info(f"Add bubble {bubble} to {self}")
            self.idle_bubbles.append(bubble)

    def add_task(self, task: TaskMeta) -> None:
        with self.lock:
            logger.info(f"Add task {task} to {self}")
            self.tasks.append(task)

    def clear_expired_bubbles(self) -> None:
        with self.lock:
            if self.current_bubble is not None and self.current_bubble.is_expired():
                if self.current_task is not None and self.current_task.is_running():
                    self.current_task.pause()
                self.current_bubble = None
            bubbles_to_remove: list[Bubble] = []
            for bubble in self.idle_bubbles:
                if bubble.is_expired():
                    bubbles_to_remove.append(bubble)
            for bubble in bubbles_to_remove:
                self.idle_bubbles.remove(bubble)
                logger.info(f"Remove expired bubble: {bubble}")

    def remove_task_from_task_runner(self, task_id: int) -> None:
        with self.lock:
            if self.current_task is not None and self.current_task.task_id == task_id:
                logger.info(f"Remove task {self.current_task} from {self}")
                if self.enable_stop:
                    self.current_task.stop()
                self.current_task = None
                return
            for task in self.tasks:
                if task.task_id == task_id:
                    logger.info(f"Remove task {task} from {self}")
                    self.tasks.remove(task)
                    return
            raise Exception(f"Task {task_id} not found in {self}")

    def finish_current_task(self, task_id: int):
        with self.lock:
            assert (
                self.current_task is not None and task_id == self.current_task.task_id
            )
            self.current_task.finish()
            self.current_task = None

    def clear_bubbles(self):
        with self.lock:
            if self.current_bubble is not None:
                if self.current_task is not None:
                    assert self.current_task.is_running()
                    self.current_task.pause()
                self.current_bubble = None
            self.idle_bubbles.clear()

    def update_current_bubble(self):
        with self.lock:
            if self.current_bubble is None:
                new_current_bubble: Optional[Bubble] = None
                for bubble in self.idle_bubbles[:]:
                    if bubble.is_active():
                        new_current_bubble = bubble
                        break
                if new_current_bubble is not None:
                    self.idle_bubbles.remove(new_current_bubble)
                    self.current_bubble = new_current_bubble
                    self.just_set_new_current_bubble = True
                    logger.info(f"Update current bubble: {self.current_bubble}")
                    return True
                    # if not self.current_task is None:
                    #     assert self.current_task.is_paused()
                    #     self.current_task.client.start_task(
                    #         self.current_task.task_id, self.current_bubble.get_end_time()
                    #     )
                    #     self.current_task.to_running()
            return False

    def update_current_task(self):
        with self.lock:
            if self.current_bubble is None:
                return

            max_priority: int = (
                -1 if self.current_task is None else self.current_task.priority
            )
            task_max_priority = self.current_task
            for task in self.tasks:
                if task.priority > max_priority:
                    max_priority = task.priority
                    task_max_priority = task
            if max_priority == -1:
                logger.debug(f"No task to run for {self}")
                return
            assert task_max_priority is not None
            if self.current_task is None:
                if self.just_set_new_current_bubble:
                    self.just_set_new_current_bubble = False
                    self.current_task = task_max_priority
                    self.tasks.remove(task_max_priority)
                    logger.info(f"Update current task: {self.current_task}")
                    assert self.current_task.is_created()
                    self.current_task.init()
                    # Init time is usually not very slow, but the first iteration is slow.
                    # Rely on the task to decidie how many iterations to run.
                    self.current_task.start(self.current_bubble.get_end_time())
            elif (
                max_priority == self.current_task.priority
                and self.just_set_new_current_bubble
            ):
                # Do not change current task.
                self.just_set_new_current_bubble = False
                if self.current_task.is_created():
                    self.current_task.init()
                if self.current_task.is_paused():
                    self.current_task.start(self.current_bubble.get_end_time())
            elif (
                max_priority > self.current_task.priority
                and self.just_set_new_current_bubble
            ):
                # Change current task.
                self.just_set_new_current_bubble = False
                self.current_task.preempt()
                assert self.current_task.is_created()
                # TODO: The line below violates the task submission time ordering.
                self.tasks.append(self.current_task)
                logger.info(
                    f"Prempt task: {self.current_task}, run task: {task_max_priority}"
                )
                self.current_task = task_max_priority
                assert self.current_task.is_created()
                self.tasks.remove(self.current_task)

    def stop(self):
        with self.lock:
            if self.current_task is not None:
                assert (
                    self.current_task.is_created()
                    or self.current_task.is_paused()
                    or self.current_task.is_running()
                )
                if self.enable_stop:
                    self.current_task.stop()
                logger.info(f"{self} stops current task: {self.current_task}")
            self.current_task = None
            self.current_bubble = None
            for task in self.tasks:
                assert task.is_created() or task.is_paused()
                if self.enable_stop:
                    task.stop()
                    logger.info(f"{self} stops task: {task}")
            self.tasks.clear()
            self.idle_bubbles.clear()


class SchedulerServicer(scheduler_pb2_grpc.SchedulerServicer):
    def __init__(self, enable_stop: bool):
        super().__init__()
        self.enable_stop: bool = enable_stop
        self._task_id_inc: int = 0
        self.task_manager_counter: int = 0
        self.task_id_dict: dict[int, TaskMeta] = {}
        self.task_runners: list[TaskRunnerMeta] = []
        self._lock = Lock()
        self._schedule_running = True
        self._schedule_running_lock = Lock()
        self._schedule_runner = Thread(target=self.schedule, args=())
        self._schedule_runner.start()

    def AddTaskRunner(self, request: scheduler_pb2.AddTaskRunnerArgs, context):
        with self._lock:
            device: str = request.device
            addr: str = request.addr
            stage_id: int = request.stage_id
            logger.info(f"Add task runner {device}, stage {stage_id} on addr {addr}")
            task_runner: TaskRunnerMeta = TaskRunnerMeta(
                device,
                addr,
                stage_id,
                self.task_manager_counter,
                enable_stop=self.enable_stop,
            )
            self.task_manager_counter += 1
            self.task_runners.append(task_runner)
            return scheduler_pb2.AddTaskRunnerReply(status=0, id=task_runner.id)

    def AddTask(self, request: scheduler_pb2.AddTaskArgs, context):
        def get_task_runner_index_for_task(
            self: SchedulerServicer, priority: int, gpu_memory: int
        ) -> int:
            possible_index_pairs: list[tuple[int, int, int, int]] = []
            for i, task_runner in enumerate(self.task_runners):
                if stage_id_gpu_memory_map[task_runner.stage_id] > gpu_memory:
                    duration: int = 1
                    # duration: float = stage_id_bubble_duration_map[task_runner.stage_id]
                    queue_length: int = task_runner.get_queue_length_with_priority(
                        priority
                    )
                    gpu_mempry: int = stage_id_gpu_memory_map[task_runner.stage_id]
                    possible_index_pairs.append((i, duration, queue_length, gpu_mempry))
            if len(possible_index_pairs) <= 0:
                raise Exception(
                    f"No suitable task runner found for task, task requires gpu memory {gpu_memory}"
                )
            if gpu_memory == 0:
                possible_index_pairs.sort(key=lambda x: (-x[3], x[2], -x[1]))
            else:
                possible_index_pairs.sort(key=lambda x: (x[2], -x[1]))
            logger.info(possible_index_pairs)
            return possible_index_pairs[0][0]

        """Add a new task."""
        with self._lock:
            name: str = request.name
            scheduler_addr: str = request.scheduler_addr
            cmd: str = request.cmd
            priority: int = request.priority
            gpu_memory: int = request.gpu_memory
            task_id: int = self._task_id_inc
            self._task_id_inc += 1
            task_runner_index: int = 0
            try:
                task_runner_index = get_task_runner_index_for_task(
                    self, priority=priority, gpu_memory=gpu_memory
                )
            except Exception as e:
                logger.error(e)
                return scheduler_pb2.AddTaskReply(
                    status=0,
                    task_id=-1,
                    addr=None,
                    device_name=None,
                )
            task_runner: TaskRunnerMeta = self.task_runners[task_runner_index]
            # Very misleading implementation. The task process is created here.
            reply: task_runner_pb2.AddTaskToRunnerReply = task_runner.client.add_task(
                task_id=task_id, name=name, scheduler_addr=scheduler_addr, cmd=cmd
            )
            assert reply.status == 0
            logger.info(f"Address {reply.addr}")
            task_meta = TaskMeta(
                task_id, reply.addr, priority, gpu_memory, task_runner_index, cmd
            )
            task_runner.add_task(task_meta)
            self.task_id_dict[task_id] = task_meta
            return scheduler_pb2.AddTaskReply(
                status=0,
                task_id=task_id,
                addr=reply.addr,
                device_name=reply.device_name,
            )

    def RemoveTask(self, request: scheduler_pb2.RemoveTaskArgs, context):
        """Remove a task immediately even if it is running."""
        with self._lock:
            task_id: int = request.task_id
            task: TaskMeta = self.task_id_dict[task_id]
            assert task_id == task.task_id
            task_runner: TaskRunnerMeta = self.task_runners[task.task_runner_index]
            task_runner.remove_task_from_task_runner(task_id)
            task.client.stop_task(task_id)
            logger.info(f"Stop task: {task}")
            self.task_id_dict.pop(task_id)
            return scheduler_pb2.RemoveTaskReply(status=0)

    def FinishTask(self, request: scheduler_pb2.FinishTaskArgs, context):
        """Finish a task on the task side and notify the scheduler."""
        with self._lock:
            task_id: int = request.task_id
            task: TaskMeta = self.task_id_dict[task_id]
            assert task_id == task.task_id
            task_runner: TaskRunnerMeta = self.task_runners[task.task_runner_index]
            task_runner.finish_current_task(task_id)
            logger.info(f"Finish task: {task}")
            self.task_id_dict.pop(task_id)
            return scheduler_pb2.FinishTaskReply(status=0)

    def AddBubble(self, request: scheduler_pb2.AddBubbleArgs, context):
        def find_task_runner_from_bubble(
            self: SchedulerServicer, bubble: Bubble
        ) -> TaskRunnerMeta:
            for task_runner in self.task_runners:
                if (
                    task_runner.stage_id == bubble.stage_id
                    and task_runner.device == bubble.device
                ):
                    return task_runner
            raise Exception("No suitable task runner found for bubble")

        with self._lock:
            start = request.start
            end = request.end
            stage_id = request.stage_id
            global_rank = request.global_rank
            device = request.device
            bubble = Bubble(start, end, stage_id, global_rank, device)
            logger.info(f"Add bubble start {bubble}")
            try:
                task_runner: TaskRunnerMeta = find_task_runner_from_bubble(self, bubble)
                task_runner.add_bubble(bubble)
            except Exception as e:
                logger.error(e)
            finally:
                logger.info(f"Add bubble end {bubble}")
        return scheduler_pb2.AddBubbleReply(status=0)

    def ClearBubble(self, request: scheduler_pb2.ClearBubbleArgs, context):
        with self._lock:
            stage_id = request.stage_id
            global_rank = request.global_rank
            device = request.device
            for task_runner in self.task_runners:
                if stage_id == task_runner.stage_id and device == task_runner.device:
                    task_runner.clear_bubbles()
            return scheduler_pb2.ClearBubbleReply(status=0)

    def schedule(self):
        while True:
            with self._schedule_running_lock:
                if not self._schedule_running:
                    logger.info("Break out of schedule loop")
                    break
            with self._lock:
                for task_runner in self.task_runners:
                    task_runner.clear_expired_bubbles()
                    if task_runner.update_current_bubble():
                        task_runner.update_current_task()
            time.sleep(0.01)

    def stop_schedule(self):
        logger.info("Stop schedule")
        with self._schedule_running_lock:
            self._schedule_running = False
        self._schedule_runner.join()
        for task_runner in self.task_runners:
            task_runner.stop()
        self.task_id_dict.clear()
        self.task_runners.clear()
        logger.info("Stop schedule thread")


class Scheduler:
    def __init__(
        self, enable_stop: bool, addr: str = "localhost:40051", max_workers: int = 10
    ):
        self.enable_stop: bool = enable_stop
        self.addr: str = addr
        self._max_workers: int = max_workers
        self._servicer: SchedulerServicer = SchedulerServicer(self.enable_stop)
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
        self, device: str, addr: str, stage_id: int
    ) -> scheduler_pb2.AddTaskRunnerReply:
        logger.info(f"Add task runner {device} on addr {addr}")
        args = scheduler_pb2.AddTaskRunnerArgs(
            device=device, addr=addr, stage_id=stage_id
        )
        reply: scheduler_pb2.AddTaskRunnerReply = self._stub.AddTaskRunner(args)
        assert reply.status == 0
        return reply

    def add_task(
        self,
        name: str,
        scheduler_addr: str,
        priority: int,
        cmd: str,
        gpu_memory: int,
    ) -> scheduler_pb2.AddTaskReply:
        args = scheduler_pb2.AddTaskArgs(
            name=name,
            scheduler_addr=scheduler_addr,
            cmd=cmd,
            priority=priority,
            gpu_memory=gpu_memory,
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
    parser.add_argument("--enable_stop", action="store_true")
    parser.add_argument("-a", "--addr", type=str)
    parser.add_argument("--memory_list", type=str)
    parser.add_argument("--name", type=str)

    for h in logger.handlers[:]:
        logger.removeHandler(h)
    args = parser.parse_args()
    new_log_file: str = f"{args.name}_scheduler.log"
    file_handler = logging.FileHandler(new_log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(created)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stage_id_gpu_memory_map = {
        i: int(memory) for i, memory in enumerate(args.memory_list.split(","))
    }
    enable_stop: bool = args.enable_stop
    addr = args.addr
    s = Scheduler(enable_stop=enable_stop, addr=addr)
    signal.signal(signal.SIGINT, handler)
    s.start_and_wait()
