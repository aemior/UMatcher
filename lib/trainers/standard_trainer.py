# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from lib.utils.imgproc import draw_bboxes_on_batch_multi, draw_match_result
from lib.dataset.MultiDatasetFusion import FusionDataset
from torch.utils.tensorboard import SummaryWriter
import datetime
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, cfg, local_rank=0, start_epoch=0):
        self.local_rank = local_rank
        self.random_seed = cfg.TRAIN.SEED
        # Create a separate random state for dataset selection
        self.dataset_random_state = np.random.RandomState(self.random_seed)
        self.gpu_ids = cfg.TRAIN.GPU_IDS
        self.make_save_path(cfg.TRAIN.SAVE_PATH)
        self.train_set = train_dataset
        self.val_set = val_dataset
        self.num_workers= cfg.TRAIN.NUM_WORKERS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.prefetch_factor = cfg.TRAIN.PREFETCH_FACTOR
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_cnt = 0
        self.set_dataloader()
        self.set_model(model)
        self.set_optimizer(cfg)
        self.set_epoch_nums(start_epoch, cfg.TRAIN.NUM_EPOCHS)
        self.save_freq = cfg.TRAIN.SAVE_FREQ
        if cfg.TRAIN.RESUME:
            self.load_checkpoint()
        else:
            self.best_val_loss = float('inf')

    def set_epoch_nums(self, start_epoch, num_epochs):
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.batch_cnt = start_epoch * len(self.train_loader)

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def set_dataloader(self):
        self.dataset_random_state = np.random.RandomState(self.random_seed + self.batch_cnt + 3)
        train_set = FusionDataset(self.train_set[0], self.train_set[1], shuffle=True, random_state=self.dataset_random_state)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        val_set = FusionDataset(self.val_set[0], self.val_set[1], shuffle=False, random_state=self.dataset_random_state)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, prefetch_factor=self.prefetch_factor, num_workers=self.num_workers)

    def make_save_path(self, save_path):
        self.save_path = save_path
        if self.local_rank == 0:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            if not os.path.exists(os.path.join(self.save_path, 'ckpt')):
                os.mkdir(os.path.join(self.save_path, 'ckpt'))
            if not os.path.exists(os.path.join(self.save_path, 'log')):
                os.mkdir(os.path.join(self.save_path, 'log'))
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'log'))
            if not os.path.exists(os.path.join(self.save_path, 'vis')):
                os.mkdir(os.path.join(self.save_path, 'vis'))

    def train(self):
        start_time = datetime.datetime.now()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.set_dataloader()
            if self.local_rank == 0:
               val_batch = next(iter(self.train_loader))
               self.visualize_results(val_batch, epoch)
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            if self.local_rank == 0:
                self.writer_log_epoch(epoch, train_loss)
                print(f"\033[1mEpoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}\033[0m", end='')
                current_datetime = datetime.datetime.now()
                elp = current_datetime - start_time
                print(f" ELP: {elp.seconds // 3600}:{(elp.seconds // 60) % 60}:{elp.seconds % 60} Date: {current_datetime.date()}, Time: {current_datetime.time()}")
                # @TODO: Add validation loss
                # self.save_checkpoint(epoch, val_loss)
                self.save_checkpoint(epoch, train_loss)

        if self.local_rank == 0:
            self.writer.close()

    def train_epoch(self, epoch):

        self.model.train()
        self.model.template_branch.backbone.eval()
        self.model.search_branch.backbone.eval()
        
        total_loss = 0
        if self.local_rank == 0:
            print()
            progress = Progress(
                TextColumn("{task.description}"),  # 显示在进度条前方的描述
                BarColumn(),                                  # 进度条
                "[progress.percentage]{task.percentage:>3.0f}%", # 百分比
                TimeRemainingColumn(),                         # 剩余时间
                TimeElapsedColumn(),                           # 已用时间
                TextColumn("{task.fields[extra_info]}", justify="right")  # 在进度条后面显示的额外信息
            )
            progress.start()
            task = progress.add_task(f"Epoch {epoch}", total=len(self.train_loader), extra_info="Starting")
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss['total_loss'].backward()
            self.optimizer.step()
            total_loss += loss['total_loss'].item()
            self.batch_cnt += 1
            
            if self.local_rank == 0:
                if self.model.dual_template:
                    extra_info_txt = f"Total Loss: {loss['total_loss'].item():.4f}, IOU Loss: {loss['iou_loss'].item():.4f}, L1 loss: {loss['l1_loss'].item():.4f}, CLS loss: {loss['cls_loss'].item():.4f}, CONTRAST loss: {loss['ctr_loss'].item():.4f}"
                else:
                    extra_info_txt = f"Total Loss: {loss['total_loss'].item():.4f}, IOU Loss: {loss['iou_loss'].item():.4f}, L1 loss: {loss['l1_loss'].item():.4f}, CLS loss: {loss['cls_loss'].item():.4f}"
                progress.update(task, advance=1, description=f"Epoch {epoch+1}:", 
                                extra_info=extra_info_txt
                                )
            if self.batch_cnt % 10 == 0:
                self.writer_log_batch(loss)
        
        if self.local_rank == 0:
            progress.stop()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_true_positives = []
        all_false_positives = []
        all_scores = []
        total_gt_objects = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss['total_loss']
                all_true_positives.extend(loss['true_positives'])
                all_false_positives.extend(loss['false_positives'])
                all_scores.extend(loss['scores'])
                total_gt_objects += loss['num_gt_objects']
        AP = self.model.calculate_ap_from_aggregated_data(all_true_positives, all_false_positives, all_scores, total_gt_objects)
        return total_loss / len(self.val_loader), AP


    def visualize_results(self, batch, epoch):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch, True)

        if self.model.dual_template:
            template_img = batch['template_img_a']
        else:
            template_img = batch['template_img']
        search_img = batch['search_img']
        ground_bbox = batch['ground_bbox']
        box_num = batch['box_num']
        # Shift search_img, ground_bbox, and box_num
        search_img_shifted = torch.roll(search_img, 1, dims=0)
        ground_bbox_shifted = torch.roll(ground_bbox, 1, dims=0)
        box_num_shifted = torch.roll(box_num, 1, dims=0)
        search_img = torch.cat([search_img, search_img_shifted], dim=0)
        ground_bbox = torch.cat([ground_bbox, ground_bbox_shifted], dim=0)
        box_num = torch.cat([box_num, box_num_shifted], dim=0)

        # Construct ground_absence
        bs = template_img.shape[0]
        ground_absence = torch.tensor([1.0]*bs + [0.0]*bs, device=template_img.device)
        image_result = draw_bboxes_on_batch_multi(search_img, outputs[1], outputs[5], outputs[2], ground_bbox, box_num, ground_absence, outputs[0], None)

        draw_match_result(template_img, image_result, 3, os.path.join(self.save_path, 'vis', f"%d.png" % epoch))

            

    def set_optimizer(self, cfg):
        self.fintinue_backbone = cfg.TRAIN.BACKBONE.FINETUNE
        model = self.model
        if cfg.TRAIN.OPTIMIZER == 'adam':
            if cfg.TRAIN.BACKBONE.FINETUNE:
                self.optimizer = optim.Adam([
                    {'params': model.template_branch.backbone.parameters(), 'lr': cfg.TRAIN.BACKBONE.LR},
                    {'params': model.template_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.backbone.parameters(), 'lr': cfg.TRAIN.BACKBONE.LR},
                    {'params': model.search_branch.neck.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                ], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            else:
                self.optimizer = optim.Adam([
                    {'params': model.template_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.neck.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                ], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            if cfg.TRAIN.BACKBONE.FINETUNE:
                self.optimizer = optim.AdamW([
                    {'params': model.template_branch.backbone.parameters(), 'lr': cfg.TRAIN.BACKBONE.LR},
                    {'params': model.template_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.backbone.parameters(), 'lr': cfg.TRAIN.BACKBONE.LR},
                    {'params': model.search_branch.neck.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                ], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            else:
                self.optimizer = optim.AdamW([
                    {'params': model.template_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.neck.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.search_branch.head.parameters(), 'lr': cfg.TRAIN.LR},
                ], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            raise ValueError(f"Invalid optimizer: {cfg.TRAIN.OPTIMIZER}")
        if cfg.TRAIN.SCHEDULER.TYPE == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER.STEP_SIZE, gamma=cfg.TRAIN.SCHEDULER.GAMMA)
        elif cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.TRAIN.SCHEDULER.T_MAX, eta_min=cfg.TRAIN.SCHEDULER.ETA_MIN)
        else:
            raise ValueError(f"Invalid scheduler: {cfg.TRAIN.SCHEDULER.TYPE}")

    
    def save_checkpoint(self, epoch, val_loss):
        if self.local_rank != 0:
            return
        if val_loss < self.best_val_loss:
            print(f"\033[1;32m🚀Best loss: {self.best_val_loss:.4f} -> {val_loss:.4f}\033[0m")
            self.best_val_loss = val_loss
            torch.save(self.model.cpu().state_dict(), os.path.join(self.save_path, 'ckpt', "best.pth"))
            self.model.to(self.device)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if epoch % self.save_freq == 0:
            torch.save(checkpoint, os.path.join(self.save_path, 'ckpt', f"epoch_{epoch}.pth"))
        torch.save(checkpoint, os.path.join(self.save_path, 'ckpt', "last.pth"))
        self.model.to(self.device)


    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.save_path, 'ckpt', "last.pth")):
            checkpoint = torch.load(os.path.join(self.save_path, 'ckpt', "last.pth"))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.batch_cnt = self.start_epoch * len(self.train_loader)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.local_rank == 0:
                print(f"Resuming training from epoch {self.start_epoch}.")
        else:
            if self.local_rank == 0:
                print("\033[1;33mWarning: TRAIN.RESUME has set but no checkpoint found, training from scratch.\033[0m")
            self.best_val_loss = float('inf')

    def writer_log_batch(self, loss):
        if self.local_rank != 0:
            return
        self.writer.add_scalar('Loss/Batch', loss['total_loss'].item(), self.batch_cnt)
        self.writer.add_scalar('Loss/IOU', loss['iou_loss'].item(), self.batch_cnt)
        self.writer.add_scalar('Loss/L1', loss['l1_loss'].item(), self.batch_cnt)
        self.writer.add_scalar('Loss/CLS', loss['cls_loss'].item(), self.batch_cnt)
        if self.model.dual_template:
            self.writer.add_scalar('Loss/CONTRAST', loss['ctr_loss'].item(), self.batch_cnt)

    def writer_log_epoch(self, epoch, train_loss):
        if self.local_rank != 0:
            return
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        


