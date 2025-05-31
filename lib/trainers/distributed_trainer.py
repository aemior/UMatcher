# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from lib.utils.imgproc import draw_bboxes_on_batch
from lib.trainers.standard_trainer import Trainer
from lib.dataset.MultiDatasetFusion import FusionDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

class DistrubutedTrainer(Trainer):
    def __call__(self, *args, **kwds):
        super().__call__(*args, **kwds)
        self.make_distribute_model(self.model)


    def set_dataloader(self):
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        train_set = FusionDataset(self.train_set[0], self.train_set[1], shuffle=True, random_state=self.dataset_random_state)
        val_set = FusionDataset(self.val_set[0], self.val_set[1], shuffle=False, random_state=self.dataset_random_state)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, sampler=train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, sampler=val_sampler)

    def make_distribute_model(self, model):
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
