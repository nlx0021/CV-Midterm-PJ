import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.train_iters = []
        self.val_iters = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_train_loss(self, loss, iter):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.train_iters.append(iter)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write('iter ')
            f.write(str(iter))
            f.write(': ')
            f.write(str(loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, iter)
        self.loss_plot()
    
    
    def append_val_loss(self, val_loss, iter):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.val_loss.append(val_loss)
        self.val_iters.append(iter)

        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write('iter ')
            f.write(str(iter))
            f.write(': ')
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('val_loss', val_loss, iter)
        self.loss_plot()


    def loss_plot(self):

        plt.figure()
        plt.plot(self.train_iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(self.val_iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            # plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
