import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from datasets.ImageNet100 import ImageNet100

def dataset(args, preprocess):
    if args.dataset == 'imagenet':
        data = ImageNet100(root='/d/data/imagenet62', split='train', preprocess=preprocess)
    else:
        raise ValueError("Unsupported dataset.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Load dataset and run attack.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name (default: imagenet)')
    args = parser.parse_args()
    preprocess = None 
    data = dataset(args, preprocess)
    data._make_typographic_attack_dataset()

if __name__ == '__main__':
    main()