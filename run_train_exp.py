from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ChTp_attn import Model as ChTp_attn_model
from models.iTransformer import Model as itransformer_model
from models.PatchTST import Model as patchtst_model
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')



class config_ettm1_patchtst():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_ETTm1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTm1"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTm1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.2
        self.proj_dropout = 0.2
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.4
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass


class config_elec_patchtst():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Electricity_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = "/home/vg2523/PatchTST/datasets/electricity"
        self.data_path = "electricity.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 321
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.05
        self.proj_dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 16
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.2
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        random_seed=2021
        pass

class config_etth2_patchtst():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Etth2_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh2"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTh2.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.3
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 16
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.fusion_dropout = 0.3
        self.proj_dropout = 0.3
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type3"
        self.pct_start = 0.3
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_etth1_patchtst():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "PatchTST"
        self.is_training = 1
        self.model_id = "PatchTST_attn_Etth1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh1"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTh1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = "end"
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 16
        self.revin = 1
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.fusion_dropout = 0.3
        self.proj_dropout = 0.3
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type3"
        self.pct_start = 0.3
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_elec_itrans():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "iTransformer"
        self.root_path = "/home/vg2523/PatchTST/datasets/electricity"
        self.checkpoints = "./checkpoints/"
        self.data_path = "electricity.csv"
        self.model = "itrans"
        self.data = "custom"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.features = "M"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = "iTansformer_attn_Electricity_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.e_layers = 3
        self.enc_in = 321
        self.dec_in = 321
        self.c_out = 321
        self.des = "Exp"
        self.d_model = 512
        self.d_ff = 512
        self.label_len = 48
        self.use_norm = True
        self.class_strategy = "projection"
        self.embed_type = 0
        self.n_heads = 8
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # 0.2
        self.embed = "timeF"
        self.pct_start = 0.3
        self.activation = "gelu"
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 16
        self.patience = 50
        self.learning_rate = 0.0001
        self.loss = "mse"
        self.lradj = "type1"
        self.use_gpu = True
        self.gpu = 0
        self.output_attention = False
        self.scheduler = False
        pass
        
class config_ettm1_itrans():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "iTransformer"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.checkpoints = "./checkpoints/"
        self.data_path = "ETTm1.csv"
        self.model = "itrans"
        self.data = "ETTm1"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.features = "M"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = "iTansformer_attn_Ettm1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.e_layers = 2
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.des = "Exp"
        self.d_model = 128
        self.d_ff = 128
        self.label_len = 48
        self.use_norm = True
        self.class_strategy = "projection"
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.n_heads = 8
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # 0.2
        self.embed = "timeF"
        self.pct_start = 0.3
        self.activation = "gelu"
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 32
        self.patience = 50
        self.learning_rate = 0.0001
        self.loss = "mse"
        self.lradj = "type1"
        self.use_gpu = True
        self.gpu = 0
        self.output_attention = False
        self.scheduler = False
        pass
        
class config_etth1_itrans():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "iTransformer"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.checkpoints = "./checkpoints/"
        self.data_path = "ETTh1.csv"
        self.model = "itrans"
        self.data = "ETTh1"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.features = "M"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = "iTansformer_attn_Etth1_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.e_layers = 2
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.des = "Exp"
        self.d_model = 256
        self.d_ff = 256
        self.label_len = 48
        self.use_norm = True
        self.class_strategy = "projection"
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.n_heads = 8
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # 0.2
        self.embed = "timeF"
        self.pct_start = 0.3
        self.activation = "gelu"
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 32
        self.patience = 50
        self.learning_rate = 0.0001
        self.loss = "mse"
        self.lradj = "type1"
        self.use_gpu = True
        self.gpu = 0
        self.output_attention = False
        self.scheduler = False
        pass
        
class config_etth2_itrans():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "iTransformer"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.checkpoints = "./checkpoints/"
        self.data_path = "ETTh2.csv"
        self.model = "itrans"
        self.data = "ETTh2"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.features = "M"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = "iTansformer_attn_Etth2_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.e_layers = 2
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.des = "Exp"
        self.d_model = 128
        self.d_ff = 128
        self.label_len = 48
        self.use_norm = True
        self.class_strategy = "projection"
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.n_heads = 8
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # 0.2
        self.embed = "timeF"
        self.pct_start = 0.3
        self.activation = "gelu"
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 32
        self.patience = 50
        self.learning_rate = 0.0001
        self.loss = "mse"
        self.lradj = "type1"
        self.use_gpu = True
        self.gpu = 0
        self.output_attention = False
        self.scheduler = False
        pass

class config_traffic_itrans():
    def __init__(self, name, seq_len, pred_len, num_epochs) -> None:
        self.model_type = "iTransformer"
        self.root_path = "/home/vg2523/PatchTST/datasets/traffic"
        self.checkpoints = "./checkpoints/"
        self.data_path = "traffic.csv"
        self.model = "itrans"
        self.data = "custom"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.features = "M"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = "iTansformer_attn_Traffic_"+str(seq_len)+"_"+str(pred_len)+"_"+name
        self.e_layers = 4
        self.enc_in = 862
        self.dec_in = 862
        self.c_out = 862
        self.des = "Exp"
        self.d_model = 512
        self.d_ff = 512
        self.label_len = 48
        self.use_norm = True
        self.class_strategy = "projection"
        self.embed_type = 0
        self.n_heads = 8
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1 # 0.2
        self.embed = "timeF"
        self.pct_start = 0.3
        self.activation = "gelu"
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 16
        self.patience = 50
        self.learning_rate = 0.001
        self.loss = "mse"
        self.lradj = "type1"
        self.use_gpu = True
        self.gpu = 0
        self.output_attention = False
        self.scheduler = False
        pass

class config_elec_chtp():
    def __init__(self, name, seq_len, pred_len, num_epochs, ch_proj_len, channel_attn_type) -> None:
        self.model_type = "ChTp_attn"
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_Electricity_"+str(seq_len)+"_"+str(pred_len)+"_"+str(ch_proj_len)+"_"+name
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = "/home/vg2523/PatchTST/datasets/electricity"
        self.data_path = "electricity.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 321
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.channel_proj_len = ch_proj_len
        self.channel_attn_type = channel_attn_type
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.05 # 0.2
        self.fusion_dropout = 0.05
        self.proj_dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 16
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.2
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_etth1_chtp():
    def __init__(self, name, seq_len, pred_len, num_epochs, ch_proj_len, channel_attn_type) -> None:
        self.model_type = "ChTp_attn"
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_Etth1_"+str(seq_len)+"_"+str(pred_len)+"_"+str(ch_proj_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh1"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTh1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 16
        self.channel_proj_len = ch_proj_len
        self.channel_attn_type = channel_attn_type
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.fusion_dropout = 0.3
        self.proj_dropout = 0.3
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 64
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type3"
        self.pct_start = 0.3
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_etth2_chtp():
    def __init__(self, name, seq_len, pred_len, num_epochs, ch_proj_len, channel_attn_type) -> None:
        self.model_type = "ChTp_attn"
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_Etth2_"+str(seq_len)+"_"+str(pred_len)+"_"+str(ch_proj_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTh2"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTh2.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.3
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 16
        self.channel_proj_len = ch_proj_len
        self.channel_attn_type = channel_attn_type
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.fusion_dropout = 0.3
        self.proj_dropout = 0.3
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 64
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type3"
        self.pct_start = 0.3
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_ettm1_chtp():
    def __init__(self, name, seq_len, pred_len, num_epochs, ch_proj_len, channel_attn_type) -> None:
        self.model_type = "ChTp_attn"
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_ETTm1_"+str(seq_len)+"_"+str(pred_len)+"_"+str(ch_proj_len)+"_"+name
        self.model = "PatchTST"
        self.data = "ETTm1"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTm1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.channel_proj_len = ch_proj_len
        self.channel_attn_type = channel_attn_type
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.2
        self.proj_dropout = 0.2
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 128
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.4
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        self.use_norm = True
        pass

class config_traffic_chtp():
    def __init__(self, name, seq_len, pred_len, num_epochs, ch_proj_len, channel_attn_type) -> None:
        self.model_type = "ChTp_attn"
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_Traffic_"+str(seq_len)+"_"+str(pred_len)+"_"+str(ch_proj_len)+"_"+name
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = "/home/vg2523/PatchTST/datasets/traffic"
        self.data_path = "traffic.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = seq_len
        self.label_len = 48
        self.pred_len = pred_len
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0
        self.embed_type = 0
        self.enc_in = 862
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.channel_proj_len = ch_proj_len
        self.channel_attn_type = channel_attn_type
        self.n_heads = 16
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.2 # 0.2
        self.fusion_dropout = 0.1
        self.proj_dropout = 0.1
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = num_epochs
        self.batch_size = 24
        self.patience = 50
        self.learning_rate = 0.0001
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "TST"
        self.pct_start = 0.2
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'
        self.test_flop = False
        self.profile = False
        self.scheduler = True
        pass

def _get_data(args, flag):
        data_set, data_loader = data_provider(args, flag)
        return data_set, data_loader

def vali(args, model, device, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            
            if 'Linear' in args.model or 'TST' in args.model:
                outputs = model(batch_x)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return model, total_loss


def train(setting, args):
    train_data, train_loader = _get_data(args, flag='train')
    vali_data, vali_loader = _get_data(args, flag='val')
    test_data, test_loader = _get_data(args, flag='test')

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    writer = SummaryWriter(log_dir=path)

    time_now = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    if args.model_type == "ChTp_attn":
        model = ChTp_attn_model(args).float().to(device)
    elif args.model_type == "iTransformer":
        model = itransformer_model(args).float().to(device)
    elif args.model_type == "PatchTST":
        print("PatchTST")
        model = patchtst_model(args).float().to(device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters: ", params)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if args.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = args.pct_start,
                                            epochs = args.train_epochs,
                                            max_lr = args.learning_rate)
    else:
        scheduler = None

    for epoch in range(args.train_epochs):
        print("Epoch number: ", epoch)
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # encoder - decoder
            if 'Linear' in args.model or 'TST' in args.model:
                outputs = model(batch_x)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 300 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        model, vali_loss = vali(args, model, device, vali_data, vali_loader, criterion)
        model, test_loss = vali(args, model, device, test_data, test_loader, criterion)

        writer.add_scalar("Train loss", train_loss, epoch)
        writer.add_scalar("Validation loss", vali_loss, epoch)
        writer.add_scalar("Test loss", test_loss, epoch)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    writer.close()
    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model
if __name__ == '__main__':
    fix_seed = 2023
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='Train exp')
    parser.add_argument('--model', type=str, default="chtp", help='[chtp, iTransformer, PatchTST]')
    parser.add_argument('--seq_len', type=int, default=336, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')
    parser.add_argument('--ch_proj_len', type=int, default=128, help='Channel projection length')
    parser.add_argument('--num_epochs', type=int, default=100, help='Num of train epochs')
    parser.add_argument('--name', type=str, default="", help='Unique ID')
    parser.add_argument('--dataset', type=str, default="etth1", help='Options: [electricity, etth1, etth2, ettm1, traffic]')
    parse_args = parser.parse_args()

    channel_attn_type = "parallel" #"sequential"
    if parse_args.dataset == "electricity" and parse_args.model == "chtp":
        args = config_elec_chtp(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs, parse_args.ch_proj_len, channel_attn_type)
    elif parse_args.dataset == "etth1" and parse_args.model == "chtp":
        args = config_etth1_chtp(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs, parse_args.ch_proj_len, channel_attn_type)
    elif parse_args.dataset == "traffic" and parse_args.model == "chtp":
        args = config_traffic_chtp(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs, parse_args.ch_proj_len, channel_attn_type)
    elif parse_args.dataset == "etth2" and parse_args.model == "chtp":
        args = config_etth2_chtp(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs, parse_args.ch_proj_len, channel_attn_type)
    elif parse_args.dataset == "ettm1" and parse_args.model == "chtp":
        args = config_ettm1_chtp(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs, parse_args.ch_proj_len, channel_attn_type)
    elif parse_args.dataset == "etth2" and parse_args.model == "iTransformer":
        args = config_etth2_itrans(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "etth1" and parse_args.model == "iTransformer":
        args = config_etth1_itrans(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "ettm1" and parse_args.model == "iTransformer":
        args = config_ettm1_itrans(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "electricity" and parse_args.model == "iTransformer":
        args = config_elec_itrans(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "traffic" and parse_args.model == "iTransformer":
        args = config_traffic_itrans(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "etth2" and parse_args.model == "PatchTST":
        args = config_etth2_patchtst(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "etth1" and parse_args.model == "PatchTST":
        args = config_etth1_patchtst(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "electricity" and parse_args.model == "PatchTST":
        args = config_elec_patchtst(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    elif parse_args.dataset == "ettm1" and parse_args.model == "PatchTST":
        args = config_ettm1_patchtst(parse_args.name, parse_args.seq_len, parse_args.pred_len, parse_args.num_epochs)
    else:
        print("Enter valid dataset")
    print("---"+args.model_id+"---")
    print("channel_attn_type: ", channel_attn_type)
    # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_()'.format(
    #                 args.model_id,
    #                 args.model,
    #                 args.data,
    #                 args.features,
    #                 args.seq_len,
    #                 args.label_len,
    #                 args.pred_len,
    #                 args.d_model,
    #                 args.n_heads,
    #                 args.e_layers,
    #                 args.d_layers,
    #                 args.d_ff,
    #                 args.factor,
    #                 args.embed,
    #                 args.distil,
    #                 args.des,
    #                 0)
    train(args.model_id, args)
