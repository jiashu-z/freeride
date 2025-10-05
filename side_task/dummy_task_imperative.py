from bubblebandit.task_v2 import run_task_server
from bubblebandit.task_v2 import ImperativeTask
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    f"%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


class DummyTaskImperative(ImperativeTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def submitted_to_created(self) -> None:
        logger.info("Task submitted to created start")
        time.sleep(0.5)
        logger.info("Task submitted to created end")

    def created_to_paused(self) -> None:
        logger.info("Task created to paused start")
        time.sleep(0.5)
        logger.info("Task created to paused end")

    def paused_to_running(self) -> None:
        logger.info("Task paused to running start")
        time.sleep(0.5)
        logger.info("Task paused to running end")

    def running_to_paused(self) -> None:
        logger.info("Task running to paused start")
        time.sleep(0.5)
        logger.info("Task running to paused end")

    def running_to_finished(self) -> None:
        logger.info("Task running to finished start")
        time.sleep(0.5)
        logger.info("Task running to finished end")

    def to_stopped(self) -> None:
        logger.info("Task to stopped start")
        time.sleep(0.5)
        logger.info("Task to stopped end")

    def gpu_workload(self):
        logger.info("Task gpu workload start")
        iteration = 0
        while True:
            time.sleep(0.5)
            logger.info(f"Iteration {iteration}")


if __name__ == "__main__":
    parser = DummyTaskImperative.get_parser()
    args = parser.parse_args()
    run_task_server(DummyTaskImperative, args)
