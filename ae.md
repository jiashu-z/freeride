# Artifact Evaluation of FreeRide: Harvesting Bubbles in Pipeline Parallelism

This is the artifact for the paper *FreeRide: Harvesting Bubbles in Pipeline Parallelism* by Jiashu Zhang, Zihan Pan, Molly (Yiming) Xu, Khuzaima Daudjee, and Sihang Liu.

For more details, please refer to the paper and this link: [https://github.com/jiashu-z/freeride](https://github.com/jiashu-z/freeride).

# File Structure

- `ae`: Scripts for running experiments for the artifact evaluation.
- `data`: This is for the experiment data, including profiling of bubbles and side tasks.
- `DeepSpeed`: This is for the instrumented DeepSpeed 0.12.2.
- `experiment`: This is the experiment scripts.
- `log`: Directory for intermeidate outputs and profiling results.
- `side_task`: This is the implementation of side tasks.
- `src`: This is the implementation of FreeRide.
- `src1`: This is the scripts of running DeepSpeed.
- `vanilla_gpu_workload`: This is the implementation of the vanilla form of the GPU side tasks.

# Hardware Setup

You need a server with 4 NVIDIA 6000 Ada Generation GPUs for most of the experiments.
Please contact the authors if you do not have access to such a server.

# Software Setup

The environment is based on `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.
We recommend using this image, but any image or environment with CUDA >= 12.1.1 should be fine.

## Setup CUDA Environment

DeepSpeed has a strict requirement for CUDA version.
Therefore, we use conda to install CUDA 12.1.

Install miniconda.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Install CUDA 12.1 using conda.

```bash
conda create --name freeride python==3.11
conda activate freeride
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit
```

Setup `CUDA_HOME`.

```bash
export CUDA_HOME=$CONDA_PREFIX
```

## Set up Python Dependencies

Install dependencies.

```bash
pip3 install -r requirements.txt
```

Install DeepSpeed.

```bash
cd DeepSpeed
pip3 install -e .
cd ..
```

Install FreeRide.

```bash
pip3 install -e .
```

Check if DeepSpeed is installed correctly.

```bash
ds_report
```

You should see something like this.
Please make sure that CUDA version is 12.1.

```
(freeride) root@cd4b629f2bb9:/workspace/bubblebandit# ds_report
/root/miniconda3/envs/freeride/bin/ds_report:4: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  __import__('pkg_resources').require('deepspeed==0.12.2+fdd85804')
[2025-10-03 02:43:47,607] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-10-03 02:43:49,747] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-dev package with apt
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
fused_adam ............. [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_lion ............... [NO] ....... [OKAY]
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
evoformer_attn ......... [NO] ....... [NO]
fused_lamb ............. [NO] ....... [OKAY]
fused_lion ............. [NO] ....... [OKAY]
inference_core_ops ..... [NO] ....... [OKAY]
cutlass_ops ............ [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
ragged_device_ops ...... [NO] ....... [OKAY]
ragged_ops ............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.1
 [WARNING]  using untested triton version (2.1.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/root/miniconda3/envs/freeride/lib/python3.11/site-packages/torch']
torch version .................... 2.1.2+cu121
deepspeed install path ........... ['/workspace/bubblebandit/DeepSpeed/deepspeed']
deepspeed info ................... 0.12.2+fdd85804, fdd85804, HEAD
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.1, cuda 12.1
shared memory (/dev/shm) size .... 116.42 GB
```

## Set up gRPC C++ Toolkit

Follow these steps to install the gRPC C++ Toolkit.

```bash
apt install -y cmake build-essential autoconf libtool pkg-config
export MY_INSTALL_DIR=$HOME/.local
mkdir -p $MY_INSTALL_DIR
export PATH="$MY_INSTALL_DIR/bin:$PATH"
apt install -y build-essential autoconf libtool pkg-config
git clone --recurse-submodules -b v1.61.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j 60
make install
popd
```

## Set up MPS

Most of our experiments are run with MPS.
To turn on MPS, run the following command.

```bash
nvidia-cuda-mps-control -d
```

To turn off MPS, run the following command.

```bash
echo quit | nvidia-cuda-mps-control
```

# Profile DeepSpeed Bubbles with Vanilla DeepSpeed

To profile bubbles in DeepSpeed, we need to switch to vanilla DeepSpeed 0.12.2.

First, install vanilla DeepSpeed 0.12.2.

```bash
cd DeepSpeed
git checkout 0.12.2
pip3 install -e .
cd ..
```

Then, run the following script to profile bubbles.
This script will run DeepSpeed with different model sizes (1.2B, 3.6B, and 6B) and different microbatch numbers (4, 6, 8) in a four stage pipeline.
It uses PyTorch Profile to generate traces.
The generated traces are saved in the `log` directory.

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_MPS_CLIENT_PRIORITY=0
python3 ae/vanilla_deepspeed_profile/run.py
```

After it finishes, you will see these directories in the `log` directory.

```
./log
├── e2e_vanilla_bubble_profile_ada6000_xlarge_10_4
├── e2e_vanilla_bubble_profile_ada6000_xlarge_10_6
├── e2e_vanilla_bubble_profile_ada6000_xlarge_10_8
├── e2e_vanilla_bubble_profile_ada6000_xxxlarge_10_4
├── e2e_vanilla_bubble_profile_ada6000_xxxlarge_10_6
├── e2e_vanilla_bubble_profile_ada6000_xxxlarge_10_8
├── e2e_vanilla_bubble_profile_ada6000_xxxxxlarge_10_4
├── e2e_vanilla_bubble_profile_ada6000_xxxxxlarge_10_6
├── e2e_vanilla_bubble_profile_ada6000_xxxxxlarge_10_8
```

Each directory contains 4 JSON files, which are the tracing of 4 pipeline stages.

Then, run the following script to analyze the bubbles and generate the bubble summary.

```bash
python3 ae/vanilla_deepspeed_profile/analyze_bubbles.py
```

This script will generate a CSV file in each directory, which contains the summary of bubbles.

# Profiling of Side Tasks

Run the following script to profile the side tasks for their performance and GPU memory consumption.

```bash
export CUDA_HOME=$CONDA_PREFIX
python3 ae/profile_side_task/run.py
```

This script will run side tasks, including resnet18, resnet50, and vgg19 model training with different batch sizes for Python workloads and image resizing, pagerank, and sgd for C++ workloads. Step time and memory consumption of side tasks will be collected with nvml.
After it finishes, you will see the output as four files for each side task.

```
(freeride) root@cd4b629f2bb9:/workspace/bubblebandit# ls schedule_ada6000_resnet18_training_iterative_16_*
schedule_ada6000_resnet18_training_iterative_16_0_side_task.txt
schedule_ada6000_resnet18_training_iterative_16_0_task.log
schedule_ada6000_resnet18_training_iterative_16_bubble_time.txt
schedule_ada6000_resnet18_training_iterative_16_monitor.txt
schedule_ada6000_resnet18_training_iterative_16_scheduler.log
schedule_ada6000_resnet18_training_iterative_16_time_profile_0.txt
```

Run the following script to analyze the side tasks and generate side task summary.

```bash
python3 ae/profile_side_task/process_results.py
```

This script will generate a json file with name `iterative_summary_ada6000.json` containing the summary of side tasks

# Run Vanilla DeepSpeed as Baseline

Install vanilla DeepSpeed 0.12.2.

```bash
cd DeepSpeed
git checkout 0.12.2
pip3 install -e .
cd ..
```

Then, run the following command.

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_MPS_CLIENT_PRIORITY=0
python3 ae/vanilla_deepspeed_baseline/run.py
```

You will see the output json files in the `log` directory.
They are the performance measurements from 4 pipeline stages.
We use stage 0 as the final result.
The training time is found in the `time` field of the performance measurement outputs.

# Run DeepSpeed with Side Tasks

Install our instrumented DeepSpeed.

```bash
cd DeepSpeed
git checkout freeride
pip3 install -e .
cd ..
```

Prepare input data for side tasks.

```bash
python3 prepare_data.py
```

Then, run the following command.

```bash
export CUDA_MPS_CLIENT_PRIORITY=0
python3 ae/side_task_deepspeed/run_side_task.py
```

The script `ae/side_task_deepspeed/run_side_task.py` will run the instrumented DeepSpeed with 6 kinds of side tasks, training ResNet18, training Resnet50, training VGG19, running PageRank, running SGD on graph data, and image processing.
It reads configurations from `ae/side_task_deepspeed/experiment_config.py` and creates the DeepSpeed process, the side task runner processes, and the logger and scheduler processes.
It will use the bubble profiling results from `log`.

After it finishes, you will see something like this from the terminal.

```
2025-10-03 03:26:47,440 - INFO - /workspace/bubblebandit/ae/side_task_deepspeed/run_side_task.py:170 - Running e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4 finished
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/logger.py:196 - Received 2, stopping logger
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/logger.py:151 - Stop logger on addr localhost:40051
2025-10-03 03:26:47,442 - INFO - /workspace/bubblebandit/ae/side_task_deepspeed/run_side_task.py:178 - Signal sent
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:203 - Received 2, stopping task_runner
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:203 - Received 2, stopping task_runner
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:203 - Received 2, stopping task_runner
2025-10-03 03:26:47,441 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:203 - Received 2, stopping task_runner
2025-10-03 03:26:47,442 - INFO - /workspace/bubblebandit/src/bubblebandit/logger.py:198 - Logger stopped
2025-10-03 03:26:48,232 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:205 - task_runner stopped
2025-10-03 03:26:48,488 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:205 - task_runner stopped
2025-10-03 03:26:48,782 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:205 - task_runner stopped
2025-10-03 03:26:48,982 - INFO - /workspace/bubblebandit/src/bubblebandit/task_runner.py:205 - task_runner stopped
2025-10-03 03:26:49,502 - INFO - /workspace/bubblebandit/ae/side_task_deepspeed/run_side_task.py:188 - Clean up e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4 finished
```

The output files below contain the progress of each side task.

```bash
(freeride) root@cd4b629f2bb9:/workspace/bubblebandit# ls e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_*_side_task.txt
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_0_side_task.txt
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_1_side_task.txt
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_2_side_task.txt
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_3_side_task.txt
```

These files contain the progress of each side task.

```bash
(freeride) root@cd4b629f2bb9:/workspace/bubblebandit# ls e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_128_stage*.json
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_128_stage0.json
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_128_stage1.json
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_128_stage2.json
e2e_freeride_ada6000_xlarge_resnet18_training_iterative_64_4_128_stage3.json
```

These files contain the performance measurements of each pipeline stage of DeepSpeed.
We focus on the `time` field, which measures the time in seconds taken by the pipeline stage to run the LLM training workload, excluding the cold-start time.

```json
{
 "stage_id": 0,
 "time": 833.2839493751526,
 "energy": [
  188958778,
  189425961,
  184860516,
  190745392
 ],
 "start": 1759461169.0450017,
 "end": 1759462002.3289511
}
```

# Calculate the Overhead and Saving
After collecting output of freeride deepspeed with side tasks and vanilla deepspeed, run the notebook `ae/collect_experiment_data/analyze_cost.ipynb` to parse the output time and side task progress. It will calculate the overhead and cost saving of freeride on each side task.

```
resnet18: hourly_cost: 20.226, main-task overhead: 0.03%, dollar saving: 8.07%
resnet50: hourly_cost: 20.7151, main-task overhead: 1.36%, dollar saving: 5.92%
vgg19: hourly_cost: 20.85, main-task overhead: 2.1%, dollar saving: 5.34%
sgd: hourly_cost: 17.6782, main-task overhead: 1.16%, dollar saving: 19.87%
pr: hourly_cost: 21.3282, main-task overhead: 0.05%, dollar saving: 3.06%
image: hourly_cost: 20.9415, main-task overhead: 0.74%, dollar saving: 4.85%
Mix_task(pr): side_task_progress: 12107
Mix_task(resnet18): side_task_progress: 217152
Mix_task(image): side_task_progress: 1597
Mix_task(vgg19): side_task_progress: 64448
Mix_task(all): main-task overhead: 1.57%, dollar saving: 6.15%
```

The output contain hourly cost, main task ovehead, and dollar saving for each single task and side task progress, main task overhead, and total dollar saving for mixed task.










---











Run the following command to analyze these traces.
```bash
python3 ae/collect_experiment_data/analyze_cost.ipynb
```
The script `ae/collect_experiment_data/analyze_cost.ipynb` parses the traces and mark the bubble characteristics, including bubble types, bubble duration, and available memory. The analyze output will be outputted to the same directory as a csv file.
```bash
(freeride) 
# root @ cd4b629f2bb9 in /workspace/bubblebandit on git:master x [5:10:19] 
$ ls log/e2e_vanilla_bubble_profile_ada6000_xlarge_128_4
bubble_summary_4.csv
cd4b629f2bb9_183393.1759465221196843501.pt.trace.json  cd4b629f2bb9_183395.1759465221797773047.pt.trace.json
cd4b629f2bb9_183394.1759465221708615824.pt.trace.json  cd4b629f2bb9_183396.1759465219619013456.pt.trace.json
```


