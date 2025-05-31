import argparse
import os
import sys
import torch
import yaml
from easydict import EasyDict as edict

sys.path.insert(0, os.getcwd())
from lib.model.matcher import build_umatcher

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--ckpt', type=str, default='data/best.pth')
parser.add_argument('--half', action='store_true', help='Export model in half precision')
parser.add_argument('--export_dir', type=str, default='data/umatcher_onnx')
args = parser.parse_args()

def main():
    # Load the configuration file
    config = edict()
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    # Build the model
    model = build_umatcher(config)

    # Load the checkpoint
    checkpoint = torch.load(args.ckpt)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Export the model to ONNX
    if args.half:
        model.export_onnx(args.export_dir, config.DATA.TEMPLATE.SIZE, config.DATA.SEARCH.SIZE, config.MODEL.EMBEDDING_DIM, half_precision=True)
    else:
        model.export_onnx(args.export_dir, config.DATA.TEMPLATE.SIZE, config.DATA.SEARCH.SIZE, config.MODEL.EMBEDDING_DIM, half_precision=False)


if __name__ == "__main__":
    main()