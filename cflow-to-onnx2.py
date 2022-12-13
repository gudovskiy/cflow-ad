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

def load_weights2(model, filename):
    path = os.path.join(filename)
    state = torch.load(path)
    model.Encoder_module.encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(model.Decoder_module.decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))
    return model

# Define the encoder
class Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super(Encoder,self).__init__()
        self.encoder = encoder
    def forward(self, input):
        return self.encoder(input)

#Define the decoder
class Decoder(torch.nn.Module):
    def __init__(self, c, decoders):
        super(Decoder, self).__init__()
        self.c = c
        self.decoders = decoders
        L = c.pool_layers
        params = list(self.decoders[0].parameters())
        for l in range(1, L):
            params += list(self.decoders[l].parameters())
        # optimizer
        self.optimizer = torch.optim.Adam(params, lr=self.c.lr)
        self.N = 256

    def forward(self, pool_layers):
        P = self.c.condition_vec
        # print(self.decoders)
        self.decoders = [decoder.eval() for decoder in self.decoders]
        height = list()
        width = list()
        i = 0
        test_dist = [list() for layer in pool_layers]
        test_loss = 0.0
        test_count = 0
        start = time.time()
        with torch.no_grad():
            for l, layer in enumerate(pool_layers):
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
                p = positionalencoding2d(P, H, W).to(self.c.device).unsqueeze(0).repeat(B, 1, 1, 1)
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
                    z, log_jac_det = decoder(e_p, [c_p, ])
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
        return height, width, test_dist
      
#Define the base calss for the model      
class CFlow(torch.nn.Module):
    def __init__(self, c,encoder,decoders,pool_layers):
        super(CFlow, self).__init__()
        self.pool_layers=pool_layers
        self.Encoder_module = Encoder(encoder)
        self.Decoder_module = Decoder(c, decoders)
    def forward(self,enc_input):
        _=self.Encoder_module(enc_input)
        height, width, test_dist = self.Decoder_module(self.pool_layers)
        return height, width , test_dist

      
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#Main
def main(c):
    # model
    c.gpu = '0'
    c.enc_arch = 'mobilenet_v3_large'
    c.inp = 256
    c.dataset = 'mvtec'
    c.action_type = 'norm-train'

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
    #Create the encoder and decoder networks
    L = c.pool_layers
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(c.device).eval()
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]
    
    #Initialize the base calss
    model=CFlow(c,encoder,decoders,pool_layers)

    PATH = 'weights/mvtec_mobilenet_v3_large_freia-cflow_pl3_cb8_inp256_run0_Model_2022-11-08-10:50:39.pt'
    model = load_weights2(model, PATH)
    print("Loaded !")
    model.eval()
    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256).to(c.device)
    out = model(x)
    torch.onnx.export(
        model,  #
        x,
        "custom-d.onnx",
        export_params=True,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
if __name__ == '__main__':
    c = get_args()
    main(c)




