import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))


class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()
        
        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
