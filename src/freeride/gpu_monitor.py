import pynvml
import signal
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="monitor.csv")
args = parser.parse_args()
output = args.output

f = open(output, "w")


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    pynvml.nvmlShutdown()
    f.flush()
    f.close()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

pynvml.nvmlInit()

device_count = pynvml.nvmlDeviceGetCount()
devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
while True:
    for i, device in enumerate(devices):
        t: float = time.time()
        # pcie0: float = pynvml.nvmlDeviceGetPcieThroughput(
            # device, pynvml.NVML_PCIE_UTIL_TX_BYTES
        # ) / (1024)
        pcie0 = 0.0
        # pcie1: float = pynvml.nvmlDeviceGetPcieThroughput(
            # device, pynvml.NVML_PCIE_UTIL_RX_BYTES
        # ) / (1024)
        pcie1 = 0.0
        # pcie2: float = pynvml.nvmlDeviceGetPcieThroughput(
            # device, pynvml.NVML_PCIE_UTIL_COUNT
        # )
        pcie2 = 0.0
        energy: float = pynvml.nvmlDeviceGetTotalEnergyConsumption(device) / 1000
        memory: float = float(pynvml.nvmlDeviceGetMemoryInfo(device).used)
        line = f"{i},{t},{pcie0},{pcie1},{pcie2},{energy},{memory}\n"
        f.write(line)
    time.sleep(0.1)
