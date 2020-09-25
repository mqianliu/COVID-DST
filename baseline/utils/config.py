import os
import logging
import argparse
from tqdm import tqdm
import torch

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False


parser = argparse.ArgumentParser(description='COVID DST')

# Training setting
parser.add_argument('-ds','--dataset', help='dataset', required=False, default="cdst")
parser.add_argument('-t','--task', help='Task Number', required=False, default="dst")
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-patience', '--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-all_vocab', '--all_vocab', help='', required=False, default=1, type=int)
parser.add_argument('-bsz', '--batch', help='Batch_size', required=False, default=64, type=int)

# Testing setting
parser.add_argument('-rundev','--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)
parser.add_argument('-gs','--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

# Model architecture
parser.add_argument('-le', '--load_embedding', help='', required=False, default=1, type=int)
parser.add_argument('-femb', '--fix_embedding', help='', required=False, default=0, type=int)

# Model hyper-parameters
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, type=int, default=400)
parser.add_argument('-lr', '--lr_rate', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr', '--dropout', help='Drop Out', required=False, type=float)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10, type=int)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)


args = vars(parser.parse_args())
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
if args["fix_embedding"]:
    args["addName"] += "FixEmb"

print(str(args))