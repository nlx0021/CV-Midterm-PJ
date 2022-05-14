import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.train_iters = []
        self.val_iters = []
        self.misc_iters = []
        self.mAPs = []
        self.mIOUs = []
        self.mIOUs_mod = []
        self.accs = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_train_loss(self, loss, iteration):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.train_iters.append(iteration)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write('iter ')
            f.write(str(iteration))
            f.write(': ')
            f.write(str(loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, iteration)
        self.loss_plot()
        

    def append_val_loss(self, val_loss, iteration):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.val_loss.append(val_loss)
        self.val_iters.append(iteration)

        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write('iter ')
            f.write(str(iteration))
            f.write(': ')
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('val_loss', val_loss, iteration)
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
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "iter_loss.png"))

        plt.cla()
        plt.close("all")


    def append_misc(self, mAP, mIOU, mIOU_mod, acc, iteration):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.mAPs.append(mAP)
        self.mIOUs.append(mIOU)
        self.mIOUs_mod.append(mIOU_mod)
        self.accs.append(acc)
        self.misc_iters.append(iteration)

        with open(os.path.join(self.log_dir, "epoch_misc.txt"), 'a') as f:
            f.write('\niter ')
            f.write(str(iteration))
            f.write('mAP: ')
            f.write(str(mAP))
            f.write("\n")
            f.write('mIOU: ')
            f.write(str(mIOU))
            f.write("\n")
            f.write('mIOU_mod: ')
            f.write(str(mIOU_mod))
            f.write("\n")
            f.write('acc: ')
            f.write(str(acc))
            f.write("\n\n")

        self.writer.add_scalar('mAP', mAP, iteration)
        self.writer.add_scalar('mIOU', mIOU, iteration)
        self.writer.add_scalar('mIOU_mod', mIOU_mod, iteration)
        self.writer.add_scalar('acc', acc, iteration)
        
        self.misc_plot()
        
        
    def misc_plot(self):
    
        plt.figure()
        plt.plot(self.misc_iters, self.mAPs, 'green', linewidth = 2, label='val mAP')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('mAP')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "iter_mAP.png"))
        plt.cla()
        plt.close("all")
        
        plt.figure()
        plt.plot(self.misc_iters, self.mIOUs, 'royalblue', linewidth = 2, label='val mIOU')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('mIOU')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "iter_mIOU.png"))
        plt.cla()
        plt.close("all")
        
        plt.figure()
        plt.plot(self.misc_iters, self.mIOUs_mod, 'royalblue', linewidth = 2, label='val mIOU')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('mIOU')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "iter_mIOU_mod.png"))
        plt.cla()
        plt.close("all")
        
        plt.figure()
        plt.plot(self.misc_iters, self.accs, 'fuchsia', linewidth = 2, label='val acc')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('acc')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "iter_acc.png"))
        plt.cla()
        plt.close("all")
        
        
    def save_data(self):
        np.save(os.path.join(self.log_dir, "data_train_loss.npy"), np.array(self.losses))
        np.save(os.path.join(self.log_dir, "data_train_iter.npy"), np.array(self.train_iters))
        np.save(os.path.join(self.log_dir, "data_val_loss.npy"), np.array(self.val_loss))
        np.save(os.path.join(self.log_dir, "data_val_iter.npy"), np.array(self.val_iters))
        np.save(os.path.join(self.log_dir, "data_mAP.npy"), np.array(self.mAPs))
        np.save(os.path.join(self.log_dir, "data_mIOU.npy"), np.array(self.mIOUs))
        np.save(os.path.join(self.log_dir, "data_mIOU_mod.npy"), np.array(self.mIOUs_mod))
        np.save(os.path.join(self.log_dir, "data_acc.npy"), np.array(self.accs))
        np.save(os.path.join(self.log_dir, "data_misc_iter.npy"), np.array(self.misc_iters))