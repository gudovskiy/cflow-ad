import os, time
import random, math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *
from config import get_args
from config import get_args
from train import train
from test import test
from test_single import test_single

log_theta = torch.nn.LogSigmoid()
# First we create the model ex. model=CFlow(C)
# Then we load the parameters ex. model(checkpoint)

def load_weights2(model, filename):
    path = os.path.join(filename)
    state = torch.load(path)
    model.encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(model.decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))
    return model

class CFlow(torch.nn.Module):
    def __init__(self, c):
        super(CFlow, self).__init__()
        L = c.pool_layers
        self.encoder, self.pool_layers, self.pool_dims = load_encoder_arch(c, L)
        self.encoder = self.encoder.to(c.device).eval()
        self.decoders = [load_decoder_arch(c, pool_dim) for pool_dim in self.pool_dims]
        self.decoders = [decoder.to(c.device) for decoder in self.decoders]
        params = list(self.decoders[0].parameters())
        for l in range(1, L):
            params += list(self.decoders[l].parameters())
        # optimizer
        self.optimizer = torch.optim.Adam(params, lr=c.lr)
        self.N=256

    def forward(self, x):
        print(type(x))
        P = c.condition_vec
        #print(self.decoders)
        self.decoders = [decoder.eval() for decoder in self.decoders]
        height = list()
        width = list()
        i=0
        test_dist = [list() for layer in self.pool_layers]
        test_loss = 0.0
        test_count = 0
        start = time.time()
        _ = self.encoder(x)
        with torch.no_grad():
            for l, layer in enumerate(self.pool_layers):
                e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H * W
                E = B * S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC

                decoder = self.decoders[l]
                FIB = E // self.N + int(E % self.N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * self.N, (f + 1) * self.N)
                    else:
                        idx = torch.arange(f * self.N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    # m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p, ])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
        return height, width, test_dist

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(c):
    # model
    c.gpu='0'
    c.enc_arch='mobilenet_v3_large'
    c.inp=256
    c.dataset='mvtec'
    c.action_type='norm-train'

    # image
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    if c.dataset == 'stc':
        c.norm_mean, c.norm_std = 3 * [0.5], 3 * [0.225]
    else:
        c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #
    c.img_dims = [3] + list(c.img_size)
    # network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks
    # dataloader parameters
    if c.dataset == 'mvtec':
        c.data_path = './data/MVTec-AD'
    elif c.dataset == 'stc':
        c.data_path = './data/STC/shanghaitech'
    elif c.dataset == 'video':
        c.data_path = c.video_path
    elif c.dataset == 'image':
        c.data_path = c.image_path

    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    # output settings
    c.verbose = True
    c.hide_tqdm_bar = True
    c.save_results = True
    # unsup-train
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i * c.meta_epochs // 100 for i in [50, 75, 90]]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr / 10.0
    if c.lr_cosine:
        eta_min = c.lr * (c.lr_decay_rate ** 3)
        c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
    else:
        c.lr_warmup_to = c.lr
    ########
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    # selected function:

    model = CFlow(c)
    print("Created !!!")
    PATH = 'weights/mvtec_mobilenet_v3_large_freia-cflow_pl3_cb8_inp256_run0_Model_2022-11-08-10:50:39.pt'
    model=load_weights2(model,PATH)
    print("Loaded !")
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256).to(c.device)
    out=model(x)
    print("Starting Export !!!")
    torch.onnx.export(
        model,
        x,
        "custom.onnx",
        export_params=True,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )



if __name__ == '__main__':
    c = get_args()
    main(c)
