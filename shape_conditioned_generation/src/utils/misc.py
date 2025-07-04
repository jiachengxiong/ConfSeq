import os
import time
import random
import logging
import torch
import numpy as np
import yaml
from easydict import EasyDict
from logging import Logger
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
import pickle


def load_pickle(path):
    print(f'Loading from {path}')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'Load data successfully from {path}')
    return data
    

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved to {path}')


def load_smileslist(path):
    print(f'Loading from {path}')
    with open(path, 'r') as f:
        smiles_list = f.readlines()
    smiles_list = [smiles.strip() for smiles in smiles_list]
    print(f'Loaded {len(smiles_list)} SMILES from {path}')
    return smiles_list


def save_smileslist(smiles_list, path):
    with open(path, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + '\n')
    print(f'Saved to {path}')


def load_scores(filename):
    with open(filename, 'r') as f:
        scores = [float(line.strip()) for line in f.readlines()]    
    return scores


def save_scores(scores, filename):
    with open(filename, 'w') as f:
        for score in scores:
            f.write(str(float(score)) + '\n')



class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_easydict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):  
            items.extend(flatten_easydict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
