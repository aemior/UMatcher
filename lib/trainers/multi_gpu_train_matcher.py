import argparse # 必须引入 argparse 包
import os
import sys
import random
import numpy as np
import torch  
import yaml
from easydict import EasyDict as edict

sys.path.insert(0, os.getcwd())
from lib.trainers.distributed_trainer import DistrubutedTrainer
from lib.tracker.utracker_bak import build_utracker
from lib.model.matcher import build_umatcher
from lib.dataset.MultiDatasetFusion import GetMultiDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--local-rank', type=int, default=0)
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Load the configuration file
    config = edict()
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    # 初始化分布式环境，主要用来帮助进程间通信
    torch.distributed.init_process_group(backend='nccl')

    set_seed(config.TRAIN.SEED)

    # Build the model
    model = build_umatcher(config)

    # Load datasets
    trainsets = GetMultiDataset(config.DATA.TRAIN_DATASETS, config.DATA.SEARCH.SIZE, config.DATA.SEARCH.SCALE, config.DATA.TEMPLATE.SIZE, config.DATA.TEMPLATE.SCALE, config.DATA.TEMPLATE.DUAL)
    valsets = GetMultiDataset(config.DATA.VAL_DATASETS, config.DATA.SEARCH.SIZE, config.DATA.SEARCH.SCALE, config.DATA.TEMPLATE.SIZE, config.DATA.TEMPLATE.SCALE, config.DATA.TEMPLATE.DUAL)

    # Train the model
    trainer = DistrubutedTrainer(model, trainsets, valsets, config, args.local_rank)
    trainer.train()


if __name__ == "__main__":
    main()