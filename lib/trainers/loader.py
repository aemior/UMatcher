import argparse

from lib.trainers.standard_trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/standard.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    trainer = Trainer(args.config, args.local_rank)
    trainer.train()