from bubblebandit.task_v2 import run_task_server, IterativeTask
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch


class TransformerTrainingIterativeTask(IterativeTask):
    def __init__(
        self,
        batch_size: int,
        steps: int,
        seq_len: int,
        d_model: int,
        duration: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.steps: int = steps
        self.step_counter: int = 0
        self.seq_len = seq_len
        self.d_model = d_model
        self.duration = duration

    def submitted_to_created(self) -> None:
        src = torch.rand((256, self.seq_len, self.d_model))
        tgt = torch.rand((256, self.seq_len, self.d_model))
        train_set = TensorDataset(src, tgt)
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )
        self.model = nn.Transformer(
            d_model=self.d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        ).to(self.device)
        self.train_iter = iter(self.train_loader)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def to_stopped(self) -> None:
        with open(f"./{self.task_name}_{self.task_id}_side_task.txt", "w") as f:
            f.write(str(self.step_counter * self.batch_size))
            f.flush()

    def is_finished(self) -> bool:
        return self.steps != 0 and self.step_counter >= self.steps

    def step(self):
        data, target = next(self.train_iter, (None, None))
        if data is None:
            self.train_iter = iter(self.train_loader)
            data, target = next(self.train_iter)
        assert data is not None
        assert target is not None
        data = data.to(self.device)
        target = target.to(self.device)
        output = self.model(data, target)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step_counter += 1


if __name__ == "__main__":
    parser = IterativeTask.get_parser()
    parser.add_argument("-e", "--steps", type=int, help="steps to run", default=0)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--duration", type=float, help="duration of one iteration", default=0.1
    )
    args = parser.parse_args()
    run_task_server(
        TransformerTrainingIterativeTask,
        args,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=1024,
        d_model=1024,
        duration=args.duration,
    )
