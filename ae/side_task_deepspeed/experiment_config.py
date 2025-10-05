import json
from copy import deepcopy

NCCL_BUFFSIZE = 33554432
NCCL_IB_DISABLE = 1
NCCL_P2P_DISABLE = 1

dp: int = 1
pp: int = 4

SEQ_LEN = 1024
NUM_GPUS = 4
PARTITIONS = "-"

logger_addr: str = "localhost:40051"

steps_list: list[int] = [128]
profiler_list: list[int] = [0]
blocking_list: list[str] = ["0"]
batch_size_list: list[int] = [12]
global_batch_size_list: list[int] = [48]

config_name: str = "XLARGE"
steps = 128
PROFILER = 0
CUDA_LAUNCH_BLOCKING = "0"

gpt_configs = dict(
    small=dict(
        name="small",
        MODEL_SIZE=0.125,
        NUM_LAYERS=12,
        HIDDEN_SIZE=768,
        NUM_ATTN_HEADS=12,
        LR=6.0e-4,
        MIN_LR=6.0e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 10,
            2: 20,
            3: 30,
        },
        BATCH_SIZE=12,
        BATCH_NUM=4,
    ),
    medium=dict(
        name="medium",
        MODEL_SIZE=0.35,
        NUM_LAYERS=24,
        HIDDEN_SIZE=1024,
        NUM_ATTN_HEADS=16,
        LR=3.0e-4,
        MIN_LR=3.0e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 10,
            2: 20,
            3: 30,
        },
        BATCH_SIZE=12,
        BATCH_NUM=4,
    ),
    large=dict(
        name="large",
        MODEL_SIZE=0.76,
        NUM_LAYERS=24,
        HIDDEN_SIZE=1536,
        NUM_ATTN_HEADS=16,
        LR=2.5e-4,
        MIN_LR=2.5e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 12,
            2: 22,
            3: 33,
        },
        BATCH_SIZE=18,
        BATCH_NUM=4,
    ),
    xlarge=dict(
        name="xlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=24,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
        stage_id_gpu_memory_map={
            0: 5,
            1: 12,
            2: 21,
            3: 32,
        },
        BATCH_SIZE=12,
        BATCH_NUM=4,
    ),
    xxlarge=dict(
        name="xxlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=48,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 9,
            2: 18,
            3: 27,
        },
        BATCH_SIZE=6,
        BATCH_NUM=4,
    ),
    xxxlarge=dict(
        name="xxxlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=72,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 9,
            2: 16,
            3: 24,
        },
        BATCH_SIZE=3,
        BATCH_NUM=4,
    ),
    xxxxlarge=dict(
        name="xxxxlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=96,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
        stage_id_gpu_memory_map={
            0: 3,
            1: 9,
            2: 15,
            3: 22,
        },
        BATCH_SIZE=2,
        BATCH_NUM=4,
    ),
    xxxxxlarge=dict(
        name="xxxxxlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=120,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
        stage_id_gpu_memory_map={
            0: 1,
            1: 6,
            2: 10,
            3: 14,
        },
        BATCH_SIZE=1,
        BATCH_NUM=4,
    ),
)

deepspeed_config_template_path = "./config/simplegpt2_template.json"
gpu_type: str = "ada6000"

class WorkloadMeta:
    def __init__(self):
        pass

    def get_command(self) -> str:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()

class DeepLearningWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        model_name: str,
        side_task_type: str,
        implementation_type: str,
        batch_size: int,
    ):
        super().__init__()
        self.model_name = model_name
        self.side_task_type = side_task_type
        self.implementation_type = implementation_type
        self.batch_size = batch_size

    def get_command(self) -> str:
        return (
            "python3 "
            + f"./side_task/model_{self.side_task_type}/"
            + f"{self.model_name}_{self.side_task_type}_{self.implementation_type}.py"
            + f" --batch_size={self.batch_size} --steps=0 --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"{self.model_name}_{self.side_task_type}_{self.implementation_type}_{self.batch_size}"


class VanillaDeepLearningWorkloadMeta(WorkloadMeta):
    def __init__(
        self, model_name: str, side_task_type: str, batch_size: int, steps: int = 0
    ):
        super().__init__()
        self.model_name = model_name
        self.side_task_type = side_task_type
        self.batch_size = batch_size
        self.steps: int = steps

    def get_command(self) -> str:
        return f"python3 vanilla_gpu_workload/model_{self.side_task_type}/{self.model_name}_{self.side_task_type}.py --batch_size={self.batch_size} --steps={self.steps}"

    def get_name(self) -> str:
        return f"{self.model_name}_{self.side_task_type}_batch{self.batch_size}"


class PrWorkloadMeta(WorkloadMeta):
    def __init__(self, file_type: str, graph_prefix: str, max_iter: int):
        super().__init__()
        self.file_type: str = file_type
        self.graph_prefix: str = graph_prefix
        self.max_iter = max_iter

    def get_command(self) -> str:
        return (
            # "compute-sanitizer --tool memcheck --launch-timeout 100 --log-file compute_sanitizer.log --save compute_sanitizer.save " +
            "./side_task/pr_side_task --file_type="
            + self.file_type
            + " --graph_prefix="
            + self.graph_prefix
            + " --max_iter="
            + str(self.max_iter)
            + " --symmetrize=0 --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"pr"


class SgdWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        graph_file: str,
        max_iter: int,
        lbd: float = 0.05,
        step: float = 0.003,
        epsilon: float = 1,
    ):
        super().__init__()
        self.graph_file: str = graph_file
        self.lbd: float = lbd
        self.step: float = step
        self.max_iter: int = max_iter
        self.epsilon: float = epsilon

    def get_command(self) -> str:
        return (
            "./side_task/sgd_side_task --graph_file="
            + self.graph_file
            + " --lambda="
            + str(self.lbd)
            + " --step="
            + str(self.step)
            + " --max_iter="
            + str(self.max_iter)
            + " --epsilon="
            + str(self.epsilon)
            + " --profiler_level=1"
        )

    def get_name(self) -> str:
        return f"sgd"


class ImageWorkloadMeta(WorkloadMeta):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        jpeg_quality: int,
        resize_width: int,
        resize_height: int,
        image_num: int,
        max_iter: int,
    ):
        super().__init__()
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.jpeg_quality: int = jpeg_quality
        self.resize_width: int = resize_width
        self.resize_height: int = resize_height
        self.image_num: int = image_num
        self.max_iter: int = max_iter

    def get_command(self) -> str:
        return (
            "./side_task/image_side_task --input_dir="
            + self.input_dir
            + " --output_dir="
            + self.output_dir
            + " --jpeg_quality="
            + str(self.jpeg_quality)
            + " --resize_width="
            + str(self.resize_width)
            + " --resize_height="
            + str(self.resize_height)
            + " --image_num="
            + str(self.image_num)
            + " --max_iter="
            + str(self.max_iter)
            + " --profiler_level=1"
        )

    def get_name(self) -> str:
        return "image"


lines = []
with open("iterative_summary_ada6000.json") as f:
    lines = f.readlines()
    s = " ".join(lines)
characteristics = json.loads(s)

lines = []
with open("gpu_workload_summary_ada6000.json") as f:
    lines = f.readlines()
    s = " ".join(lines)
vanilla_characteristics = json.loads(s)

dl_workloads = []
dl_workloads.append(DeepLearningWorkloadMeta("resnet18", "training", "iterative", 64))
dl_workloads.append(DeepLearningWorkloadMeta("resnet50", "training", "iterative", 64))
dl_workloads.append(DeepLearningWorkloadMeta("vgg19", "training", "iterative", 64))


workloads: list[WorkloadMeta] = []
# for dl_workload in dl_workloads:
    # workloads.append(dl_workload)

workloads.append(SgdWorkloadMeta("/dev/shm/com-Orkut.mtx", 0))
workloads.append(
    ImageWorkloadMeta(
        "/dev/shm/image_input", "/dev/shm/image_output", 100, 16384, 16384, 1, 0
    )
)
workloads.append(PrWorkloadMeta("mtx", "/dev/shm/com-Orkut", 0))

configs = [
    # gpt_configs["xlarge"],
    gpt_configs["xxxlarge"],
    # gpt_configs["xxxxxlarge"],
]
