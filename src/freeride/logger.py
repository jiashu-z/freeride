from . import logger_pb2
from . import logger_pb2_grpc
import grpc
from concurrent import futures
import os
from pathlib import Path
from io import TextIOWrapper as TextIOWrapper
import time
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


class Bubble:
    def __init__(self, start: float, end: float, stage_id: int, device: str):
        self._start: float = start
        self._end: float = end
        self._stage_id: int = stage_id
        self._device: str = str(device)

    def is_expired(self) -> bool:
        return time.time() > self._end

    def __repr__(self) -> str:
        return f"{{stage: {self._stage_id}, device: {self._device}, start: {self._start}, end: {self._end}}}"


class LoggerServicer(logger_pb2_grpc.LoggerServicer):
    def _get_fd(self, file_name=str) -> TextIOWrapper:
        return open(os.path.join(self._log_dir, file_name), "w", buffering=1)

    def __init__(self, log_dir="./log", interval: float = 0.1):
        self._log_dir = log_dir
        self._interval = interval

        Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        self._flog = self._get_fd("misc.log")
        self._fsched = self._get_fd("schedule_dump.log")
        self._fstep = self._get_fd("step.log")
        self._fsr = self._get_fd("send_recv.log")
        self._fbb = self._get_fd("bubble.log")
        self._fserve = self._get_fd("serve.log")
        self._fside = self._get_fd("side.log")

    def WriteBubble(self, request: logger_pb2.WriteBubbleArgs, context) -> logger_pb2.WriteBubbleReply:
        s: float = request.start
        e: float = request.end
        stage_id: int = request.stage_id
        global_rank: int = request.global_rank
        device: str = request.device
        if self._log_dir is not None:
            self._fbb.write(f"{s},{e},{stage_id},{global_rank},{device}\n")
        return logger_pb2.WriteBubbleReply()


    def WriteLog(self, request, context) -> logger_pb2.Empty:
        pid = request.pid
        ts = request.ts
        msg = request.msg
        if self._log_dir is not None:
            self._flog.write(f"{pid}, {ts}, {msg}\n")
        print(f"{pid}, {ts}: {msg}")
        return logger_pb2.Empty()

    def DumpSched(self, request, context) -> logger_pb2.Empty:
        pid = request.pid
        ts = request.ts
        msg = request.msg
        if self._log_dir is not None:
            self._fsched.write(f"{pid}, {ts}, {msg}\n")
        print(f"{pid}, {ts}: {msg}")
        return logger_pb2.Empty()

    def DumpStepSched(self, request, context) -> logger_pb2.Empty:
        pid = request.pid
        ts0 = request.ts0
        ts1 = request.ts1
        msg = request.msg
        if self._log_dir is not None:
            self._fstep.write(f"{pid}, {ts0}, {ts1}, {msg}\n")
        return logger_pb2.Empty()

    def RecordInstr(self, request, context) -> logger_pb2.Empty:
        pid = request.pid
        ts0 = request.ts0
        ts1 = request.ts1
        instr = request.instr
        if self._log_dir is not None:
            self._fsr.write(f"{pid}, {ts0}, {ts1}, {instr}\n")
        return logger_pb2.Empty()

    def RecordSideTask(
        self, request: logger_pb2.RecordSideTaskEntry, context
    ) -> logger_pb2.Empty:
        pid: int = request.pid
        ts0: int = request.ts0
        ts1: int = request.ts1
        counter: int = request.counter
        if self._log_dir is not None:
            self._fside.write(f"{pid}, {ts0}, {ts1}, {counter}\n")
        return logger_pb2.Empty()

    def Clear(self, request, context) -> logger_pb2.Empty:
        self._flog.close()
        self._fsched.close()
        self._fstep.close()
        self._fsr.close()
        self._flog = self._get_fd("misc.log")
        self._fsched = self._get_fd("schedule_dump.log")
        self._fstep = self._get_fd("step.log")
        self._fsr = self._get_fd("send_recv.log")
        self._fside = self._get_fd("side.log")
        return logger_pb2.Empty()


class LoggerServer:
    def __init__(
        self,
        addr: str,
        log_dir="./log",
        interval: float = 0.1,
    ):
        self._log_dir = log_dir
        self._interval = interval
        self._addr = addr
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        logger_pb2_grpc.add_LoggerServicer_to_server(
            LoggerServicer(self._log_dir, self._interval), self._server
        )
        self._server.add_insecure_port(self._addr)

    def start_and_wait(self):
        self._server.start()
        self._server.wait_for_termination()

    def stop(self, grace: Optional[float]):
        logger.info(f"Stop logger on addr {self._addr}")
        self._server.stop(grace)


class LoggerClient:
    def __init__(self, addr: str):
        self.chan = grpc.insecure_channel(addr)
        self.stub = logger_pb2_grpc.LoggerStub(self.chan)

    def write_bubble(self, s: float, e: float, stage_id: int, global_rank: int, device: str) -> None:
        args: logger_pb2.WriteBubbleArgs = logger_pb2.WriteBubbleArgs(start=s, end=e, stage_id=stage_id, global_rank=global_rank, device=device)
        self.stub.WriteBubble(args)

    def write_log(self, pid: int, ts: int, msg: str) -> None:
        log_entry = logger_pb2.LogEntry(pid=pid, ts=ts, msg=msg)
        self.stub.WriteLog(log_entry)

    def dump_sched(self, pid: int, ts: int, msg: str) -> None:
        sched_entry = logger_pb2.DumpSchedEntry(pid=pid, ts=ts, msg=msg)
        self.stub.DumpSched(sched_entry)

    def dump_step_sched(self, pid: int, ts0: int, ts1: int, msg: str) -> None:
        step_sched_entry = logger_pb2.DumpStepSchedEntry(
            pid=pid, ts0=ts0, ts1=ts1, msg=msg
        )
        self.stub.DumpStepSched(step_sched_entry)

    def record_instr(self, pid: int, ts0: int, ts1: int, instr: str) -> None:
        record_instr_entry = logger_pb2.RecordInstrEntry(
            pid=pid, ts0=ts0, ts1=ts1, instr=instr
        )
        self.stub.RecordInstr(record_instr_entry)

    def record_side_task(self, pid: int, ts0: int, ts1: int, counter: int) -> None:
        record_side_task_entry = logger_pb2.RecordSideTaskEntry(
            pid=pid, ts0=ts0, ts1=ts1, counter=counter
        )
        self.stub.RecordSideTask(record_side_task_entry)

    def clear(self) -> None:
        self.stub.Clear(logger_pb2.Empty())


def handler(signum, frame):
    global s
    logger.info(f"Received {signum}, stopping logger")
    s.stop(0)
    logger.info("Logger stopped")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--addr", type=str, default="localhost:40050")
    args = parser.parse_args()
    addr = args.addr
    s = LoggerServer(addr=addr)
    signal.signal(signal.SIGINT, handler)
    logger.info(f"Start logger on addr {addr}")
    s.start_and_wait()
