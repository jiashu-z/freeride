import time
from typing import Optional
from contextlib import contextmanager


@contextmanager
def get_duration(filename: Optional[str] = None):
    start: float = 0.0
    try:
        start = time.time()
        yield
    finally:
        end = time.time()
        if filename is not None:
            with open(filename, "w") as f:
                f.write(f"{end - start}")
        print(f"Duration: {end - start}")


class LoopIterator:
    def __init__(self, filename: str = "loop_iterator.csv", max_number: int = 0):
        self.counter: int = 0
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.filename: str = filename
        self.max_number: int = max_number
        self.f = open(self.filename, "w")

    def __call__(self, max_number: int):
        self.max_number = max_number
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_number is not None and self.counter >= self.max_number:
            self.end = time.time()
            if self.start is not None:
                interval = self.end - self.start
                self.f.write(f"{self.counter - 1},{interval}\n")
            raise StopIteration
        if self.start is None:
            self.start = time.time()
        else:
            self.end = time.time()
            interval = self.end - self.start
            self.start = self.end
            self.f.write(f"{self.counter - 1},{interval}\n")
        ret: int = self.counter
        self.counter += 1
        return ret

    def __del__(self):
        self.f.close()


with get_duration():
    time.sleep(2)


iter = LoopIterator()

for i in iter(5):
    print(i)
    time.sleep(1)
