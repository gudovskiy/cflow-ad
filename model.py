import math
import torch
from torch import nn
from custom_models import *
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_decoder_arch(c, dim_in):
    if   c.dec_arch == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    #print(decoder)
    return decoder


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def load_encoder_arch(c, L):
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in range(L)]
    if 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet34':
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet50':
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == 'resnext50_32x4d':
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    elif 'vit' in c.enc_arch:
        if  c.enc_arch == 'vit_base_patch16_224':
            encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif  c.enc_arch == 'vit_base_patch16_384':
            encoder = timm.create_model('vit_base_patch16_384', pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[10].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[2].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[6].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
    elif 'efficient' in c.enc_arch:
        if 'b5' in c.enc_arch:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
            blocks = [-2, -3, -5]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[blocks[2]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[2]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[blocks[1]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[1]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[blocks[0]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[0]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
    elif 'mobile' in c.enc_arch:
        if  c.enc_arch == 'mobilenet_v3_small':
            encoder = mobilenet_v3_small(pretrained=True, progress=True).features
            blocks = [-2, -5, -10]
        elif  c.enc_arch == 'mobilenet_v3_large':
            encoder = mobilenet_v3_large(pretrained=True, progress=True).features
            blocks = [-2, -5, -11]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder[blocks[2]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[2]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder[blocks[1]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[1]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder[blocks[0]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[0]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    #
    return encoder, pool_layers, pool_dims
