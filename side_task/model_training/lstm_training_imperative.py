from bubblebandit.task_v2 import run_task_server, ImperativeTask
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class LstmTrainingImperativeTask(ImperativeTask):
    def __init__(
        self,
        batch_size: int,
        steps: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.steps: int = steps
        self.step_counter: int = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.duration = 0.1

    def submitted_to_created(self) -> None:
        transformations = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )
        train_set = datasets.FakeData(
            size=38400, image_size=[32, 32], transform=transformations
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )
        self.model = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            proj_size=512,
        ).to(self.device)
        self.train_iter = iter(self.train_loader)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def to_stopped(self) -> None:
        with open(f"./{self.task_name}_{self.task_id}_side_task.txt", "w") as f:
            f.write(str(self.step_counter * self.batch_size))
            f.flush()

    def gpu_workload(self):
        while self.steps == 0 or self.step_counter < self.steps:
            self.record_time("RUNNING_STEP_START")
            data, target = next(self.train_iter, (None, None))
            if data is None:
                self.train_iter = iter(self.train_loader)
                data, target = next(self.train_iter)
            assert data is not None
            assert target is not None
            data = data.to(self.device)
            target = target.to(self.device)
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_counter += 1
            self.record_time("RUNNING_STEP_END")


if __name__ == "__main__":
    parser = ImperativeTask.get_parser()
    parser.add_argument("-e", "--steps", type=int, help="steps to run", default=0)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    args = parser.parse_args()
    run_task_server(
        LstmTrainingImperativeTask,
        args,
        steps=args.steps,
        batch_size=args.batch_size,
        input_size=1024,
        hidden_size=1024,
        num_layers=8,
    )
