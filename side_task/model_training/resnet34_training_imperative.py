from bubblebandit.task_v2 import run_task_server, ImperativeTask
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch


class Resnet34TrainingImperativeTask(ImperativeTask):
    def __init__(self, batch_size: int, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.steps: int = steps
        self.step_counter: int = 0

    def submitted_to_created(self) -> None:
        transformations = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(
            root="/dev/shm/", 
            train=True, 
            download=True, 
            transform=transformations
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )
        self.train_iter = iter(self.train_loader)
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(self.device)
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
            output = self.model(data)
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
        Resnet34TrainingImperativeTask,
        args,
        steps=args.steps,
        batch_size=args.batch_size,
    )
