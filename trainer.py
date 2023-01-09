from dataset import IMDBBertDataset
from model import BERT
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import os

class BertTrainer:

    def __init__(self,
                 model: BERT,
                 dataset: IMDBBertDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5,
                 device: str = 'cpu',
                 ):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self._print_every = print_progress_every

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.ml_criterion = nn.NLLLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)

    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")

        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, value in enumerate(self.loader):
            index = i + 1
            inp, mask, inverse_token_mask, token_target, nsp_target = value
            inp = inp.to(self.device)
            mask = mask.to(self.device)
            inverse_token_mask = inverse_token_mask.to(self.device)
            token_target = token_target.to(self.device)
            nsp_target = nsp_target.to(self.device)

            self.optimizer.zero_grad()

            token, nsp = self.model(inp, mask)

            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill(tm, 0)

            loss_token = self.ml_criterion(token.transpose(1, 2), token_target)
            loss_nsp = self.criterion(nsp, nsp_target)

            loss = loss_token + loss_nsp
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_token

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)

                log_nsp_loss = average_nsp_loss / self._print_every
                log_mlm_loss = average_mlm_loss / self._print_every
                log_nsp_acc = 100*(nsp.argmax(1) == nsp_target.argmax(1)).sum() / nsp.size(0)
                log_mlm_acc = 100*(token.argmax(-1).masked_select(~inverse_token_mask) == token_target.masked_select(~inverse_token_mask)).sum() / (token.size(0) * token.size(1))

                print(f"{time.strftime('%H:%M:%S', elapsed)} | Epoch {epoch} | Step {index}/{len(self.loader)} | "
                      f"NSP Loss: {log_nsp_loss:.2f} | MLM Loss: {log_mlm_loss:.2f} | NSP Accuracy: {log_nsp_acc:.2f}% | MLM Accuracy: {log_mlm_acc:.2f}%")

                global_step = index + epoch*len(self.loader)
                self.writer.add_scalar("NSP loss", log_nsp_loss, global_step=global_step)
                self.writer.add_scalar("MLM loss", log_mlm_loss, global_step=global_step)
                self.writer.add_scalar("NSP accuracy", log_nsp_acc, global_step=global_step)
                self.writer.add_scalar("Token accuracy", log_mlm_acc, global_step=global_step)

                average_nsp_loss = 0
                average_mlm_loss = 0
        return loss

    def __call__(self):
        if self.checkpoint_dir and os.path.exists(self.checkpoint_dir.joinpath("checkpoint_last.txt")):
            with open(self.checkpoint_dir.joinpath("checkpoint_last.txt")) as f:
                name = f.readline().strip()
            self.load_checkpoint(self.checkpoint_dir.joinpath(name))
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0

        for self.current_epoch in range(start_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(epoch=self.current_epoch, loss=loss)

    def save_checkpoint(self, epoch, loss):
        if not self.checkpoint_dir:
            return

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        prev = time.time()
        name = f"checkpoint_epoch{epoch}.pt"
        print(f"Saving model checkpoint epoch {epoch} to {self.checkpoint_dir.joinpath(name)}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        with open(self.checkpoint_dir.joinpath("checkpoint_last.txt"), "w") as f:
            f.write(name)

    def load_checkpoint(self, path: Path):
        print(f"Loading model checkpoint from {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded at epoch {self.current_epoch}.")