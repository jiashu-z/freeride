import setuptools
from setuptools.command.build_py import build_py as build_py_orig
import subprocess
import os
from grpc_tools import protoc
import shutil

proto_files = [
    "src/freeride/scheduler.proto",
    "src/freeride/task_runner.proto",
    "src/freeride/task.proto",
    "src/freeride/logger.proto",
]  # List your proto files here


class build_py(build_py_orig):
    def run(self):
        for proto in proto_files:
            command = [
                "python3",
                "-m",
                "grpc_tools.protoc",
                "-I./src",
                "--python_out=./src",
                "--pyi_out=./src",
                "--grpc_python_out=./src",
                proto,
            ]
            ret = subprocess.run(command)
            if ret.returncode != 0:
                print(ret, ret.stderr, ret.stdout)
                raise RuntimeError("protoc failed")
        shutil.copyfile(
            "src/freeride/task.proto", "side_task/gardenia/task.proto"
        )
        shutil.copyfile(
            "src/freeride/scheduler.proto", "side_task/gardenia/scheduler.proto"
        )
        # if not os.path.exists("side_task/gardenia/build"):
        #     os.mkdir("side_task/gardenia/build")
        # command = "cd side_task/gardenia/build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j"
        # subprocess.run(command, shell=True, check=True)
        # shutil.copyfile("side_task/gardenia/build/bfs_side_task", "side_task/bfs_side_task")
        # os.chmod("side_task/bfs_side_task", 0o755)
        super().run()


setuptools.setup(cmdclass={"build_py": build_py})
