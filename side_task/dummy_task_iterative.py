from bubblebandit.task_v2 import run_task_server
from bubblebandit.task_v2 import IterativeTask
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


class DummyTaskIterative(IterativeTask):
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
        with open(f"./out/{self.task_name}_{self.task_id}_side_task.txt", "w") as f:
            f.write(str(0))
            f.flush()

    def is_finished(self) -> bool:
        return False

    def step(self):
        time.sleep(0.1)

    def do_i_have_enough_time(self) -> bool:
        return super().do_i_have_enough_time()


if __name__ == "__main__":
    parser = DummyTaskIterative.get_parser()
    args = parser.parse_args()
    run_task_server(DummyTaskIterative, args)
