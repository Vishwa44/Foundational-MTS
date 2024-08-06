from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ChTp_attn import Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

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

class config():
    def __init__(self) -> None:
        self.random_seed = 2021
        self.is_training = 1
        self.model_id = "Channel_temporal_attn_Etth1_336_96_lr_001"
        self.model = "PatchTST"
        self.data = "custom"
        self.root_path = "/home/vg2523/PatchTST/datasets/ETT-small"
        self.data_path = "ETTh1.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = 336
        self.label_len = 48
        self.pred_len = 96
        self.fc_dropout = 0.2
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = None
        self.revin = False
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
        self.channel_proj_len = 128
        self.n_heads = 4
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.3 # 0.2
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 100
        self.batch_size = 64
        self.patience = 50
        self.learning_rate = 0.001
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
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in args.model or 'TST' in args.model:
                        outputs = model(batch_x)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if 'Linear' in args.model or 'TST' in args.model:
                    outputs = model(batch_x)
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
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
    return total_loss


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

    model = Model(args).float().to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters: ", params)



    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                        steps_per_epoch = train_steps,
                                        pct_start = args.pct_start,
                                        epochs = args.train_epochs,
                                        max_lr = args.learning_rate)

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
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in args.model or 'TST' in args.model:
                        outputs = model(batch_x)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if 'Linear' in args.model or 'TST' in args.model:
                        outputs = model(batch_x)
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                # print(outputs.shape,batch_y.shape)
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

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(args, model, device, vali_data, vali_loader, criterion)
        test_loss = vali(args, model, device, test_data, test_loader, criterion)

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
    args = config()

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_()'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    0)




    train(setting, args)
    # ip = torch.rand(1, args.seq_len, args.enc_in)
    # print(ip.shape)
    # op = model(ip)
    # print(op.shape)
