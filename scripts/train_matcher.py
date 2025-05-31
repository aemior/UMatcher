import argparse
import os
import sys
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from lib.trainers.standard_trainer import Trainer
from lib.model.matcher import build_umatcher
from lib.dataset.COCO import COCO
from lib.dataset.SA1B import SA1B
from lib.dataset.MultiDatasetFusion import GetMultiDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
args = parser.parse_args()

def main():
    # Load the configuration file
    config = edict()
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    if len(config.TRAIN.GPU_IDS) > 1:
        if torch.cuda.is_available():
            # Multi-GPU training
            print("Multi-GPU training is available. Launching the training process...")
            os.system('python -m torch.distributed.launch --nproc_per_node={} lib/trainers/multi_gpu_train_matcher.py --config {}'.format(len(config.TRAIN.GPU_IDS), args.config))
        else:
            raise ValueError("Multi-GPU training is not available because torch.cuda.is_available() is False.")
    else:
        # Build the model
        model = build_umatcher(config)


        trainsets = GetMultiDataset(config.DATA.TRAIN_DATASETS, config.DATA.SEARCH.SIZE, config.DATA.SEARCH.SCALE, config.DATA.TEMPLATE.SIZE, config.DATA.TEMPLATE.SCALE, config.DATA.TEMPLATE.DUAL)
        valsets = GetMultiDataset(config.DATA.VAL_DATASETS, config.DATA.SEARCH.SIZE, config.DATA.SEARCH.SCALE, config.DATA.TEMPLATE.SIZE, config.DATA.TEMPLATE.SCALE, config.DATA.TEMPLATE.DUAL)

        # Train the model
        trainer = Trainer(model, trainsets, valsets, config)
        trainer.train()


if __name__ == "__main__":
    main()