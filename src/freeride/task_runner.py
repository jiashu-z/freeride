from typing import Optional

from . import task_runner_pb2
from . import task_runner_pb2_grpc
from . import scheduler_pb2
import grpc
from concurrent import futures
import subprocess
from threading import Thread, Lock
from typing import Optional
from . import scheduler_v1
import argparse
import logging
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


class TaskRunnerServicer(task_runner_pb2_grpc.TaskRunnerServicer):
    def __init__(
        self, ip: str, port_from: int, port_to: int, device: str, fileserver: str
    ):
        super().__init__()
        self.lock = Lock()
        self.task_runners: dict[int, Thread] = {}
        self.ip: str = ip
        self.port_from: int = port_from
        self.port_to: int = port_to
        self.port_inc: int = self.port_from
        self.device: str = device
        self.fileserver: str = fileserver
        logger.info(
            f"Create TaskRunnerServicer with ip: {ip}, port_from: {port_from}, port_to: {port_to}, device: {device}, fileserver: {fileserver}"
        )

    def AddTaskToRunner(self, request: task_runner_pb2.AddTaskToRunnerArgs, context):
        with self.lock:
            task_id: int = request.task_id
            name: str = request.name
            scheduler_addr: str = request.scheduler_addr
            cmd: str = request.cmd.strip()
            addr: str = f"{self.ip}:{str(self.port_inc)}"
            self.port_inc += 1
            cmd = f"{cmd}"
            logger.info(f"cmd: {cmd}")
            args = tuple([task_id, name, self.device, addr, scheduler_addr, cmd])
            task_runner: Thread = Thread(target=self.run_task, args=args)
            self.task_runners[task_id] = task_runner
            task_runner.start()
        return task_runner_pb2.AddTaskToRunnerReply(
            status=0, addr=addr, device_name=self.device
        )

    def StopTaskOnRunner(self, request: task_runner_pb2.StopTaskOnRunnerArgs, context):
        with self.lock:
            task_id: int = request.task_id
            task_runner: Thread = self.task_runners[task_id]
            task_runner.join()
            del self.task_runners[task_id]
        return task_runner_pb2.StopTaskOnRunnerReply(status=0)

    def StopAllTasks(self, request: task_runner_pb2.StopAllTasksArgs, context):
        self.stop_all_tasks()
        return task_runner_pb2.StopAllTasksReply(status=0)

    def stop_all_tasks(self):
        with self.lock:
            for _, task_runner in self.task_runners.items():
                task_runner.join()
            self.task_runners = {}

    def FinishTaskOnRunner(
        self, request: task_runner_pb2.FinishTaskOnRunnerArgs, context
    ):
        with self.lock:
            task_id: int = request.task_id
            del self.task_runners[task_id]
        return task_runner_pb2.FinishTaskOnRunnerReply(status=0)

    def run_task(
        self,
        task_id: int,
        name: str,
        device: str,
        addr: str,
        scheduler_addr: str,
        cmd: str,
    ) -> None:
        cmd = f"{cmd} -n {name} -s {scheduler_addr} -i {task_id} -d {device} -a {addr}"
        logger.info(f"cmd: {cmd}")
        p = subprocess.Popen(cmd.split(" "))
        p.wait()


class TaskRunner:
    def __init__(
        self,
        device: str,
        addr: str = "localhost:40052",
        max_workers: int = 10,
        scheduler_addr: str = "localhost:40051",
        fileserver: str = "localhost:8000",
        stage_id: int = 3,
        port_from: Optional[int] = None,
        port_to: Optional[int] = None,
    ):
        self.device: str = device
        self.addr: str = addr
        self.max_workers: int = max_workers
        self.scheduler_addr: str = scheduler_addr
        self.fileserver: str = fileserver
        self.stage_id: int = stage_id
        self.port_from: int = port_from if port_from is not None else 40060
        self.port_to: int = port_to if port_to is not None else self.port_from + 10
        self.client: scheduler_v1.SchedulerClient = scheduler_v1.SchedulerClient(
            self.scheduler_addr
        )
        ip: str = self.addr.split(":")[0]
        self._servicer: TaskRunnerServicer = TaskRunnerServicer(
            ip=ip,
            port_from=self.port_from,
            port_to=self.port_to,
            device=self.device,
            fileserver=self.fileserver,
        )
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )
        task_runner_pb2_grpc.add_TaskRunnerServicer_to_server(
            self._servicer, self._grpc_server
        )
        self._grpc_server.add_insecure_port(self.addr)
        logger.info(
            f"TaskRunner created with addr: {self.addr}, max_workers: {self.max_workers}, scheduler_addr: {self.scheduler_addr}, fileserver: {self.fileserver}, stage_id: {self.stage_id}, port_from: {self.port_from}, port_to: {self.port_to}"
        )

    def start_and_wait(self):
        self._grpc_server.start()
        reply: scheduler_pb2.AddTaskRunnerReply = self.client.add_task_runner(
            self.device, self.addr, self.stage_id
        )
        assert reply.status == 0
        self._grpc_server.wait_for_termination()

    def start(self):
        self._grpc_server.start()

    def stop(self, grace: Optional[float]):
        self._grpc_server.stop(grace)
        self._servicer.stop_all_tasks()


class TaskRunnerClient:
    def __init__(self, addr: str = "localhost:40052"):
        self.addr: str = addr
        self.chan = grpc.insecure_channel(self.addr)
        self.stub = task_runner_pb2_grpc.TaskRunnerStub(self.chan)

    def add_task(self, task_id: int, name: str, scheduler_addr: str, cmd: str):
        request = task_runner_pb2.AddTaskToRunnerArgs(
            task_id=task_id, name=name, scheduler_addr=scheduler_addr, cmd=cmd
        )
        reply: task_runner_pb2.AddTaskToRunnerReply = self.stub.AddTaskToRunner(request)
        assert reply.status == 0
        return reply

    def stop_task(self, task_id: int):
        request = task_runner_pb2.StopTaskOnRunnerArgs(task_id=task_id)
        reply: task_runner_pb2.StopTaskOnRunnerReply = self.stub.StopTaskOnRunner(
            request
        )
        assert reply.status == 0
        return reply

    def stop_all_tasks(self):
        request = task_runner_pb2.StopAllTasksArgs()
        reply: task_runner_pb2.StopAllTasksReply = self.stub.StopAllTasks(request)
        assert reply.status == 0
        return reply

    def finish_task(self, task_id: int):
        request = task_runner_pb2.FinishTaskOnRunnerArgs(task_id=task_id)
        reply: task_runner_pb2.FinishTaskOnRunnerReply = self.stub.FinishTaskOnRunner(
            request
        )
        assert reply.status == 0
        return reply


def handler(signum, frame):
    global task_runner
    logger.info(f"Received {signum}, stopping task_runner")
    task_runner.stop(0)
    logger.info("task_runner stopped")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--addr", type=str)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--scheduler_addr", type=str, default="localhost:40051")
    parser.add_argument("--fileserver", type=str, default="localhost:8000")
    parser.add_argument("--stage_id", type=int, default=3)
    parser.add_argument("--port_from", type=int, default=40060)
    parser.add_argument("--port_to", type=int, default=40070)
    args = parser.parse_args()
    device: str = args.device
    addr: str = args.addr
    max_workers: int = args.max_workers
    scheduler_addr: str = args.scheduler_addr
    fileserver: str = args.fileserver
    stage_id: int = args.stage_id
    port_from: int = args.port_from
    port_to: int = args.port_to
    task_runner = TaskRunner(
        device,
        addr,
        max_workers,
        scheduler_addr,
        fileserver,
        stage_id,
        port_from,
        port_to,
    )
    signal.signal(signal.SIGINT, handler)
    task_runner.start_and_wait()
