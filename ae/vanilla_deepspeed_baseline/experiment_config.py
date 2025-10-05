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

configs = []
configs = [
    gpt_configs["xlarge"],
    gpt_configs["xxxlarge"],
    gpt_configs["xxxxxlarge"],
]
