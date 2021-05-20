import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
from model import fadn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from scipy import signal
from skimage.metrics import structural_similarity
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # self.plot_psnr(epoch)
        torch.save(self.log, self.get_path('psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            self.get_path('optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
# def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    # """
    # 2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    # Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
    # """
    # m,n = [(ss-1.)/2. for ss in shape]
    # y,x = np.ogrid[-m:m+1,-n:n+1]
    # h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    # h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    # sumh = h.sum()
    # if sumh != 0:
        # h /= sumh
    # return h
    
def calc_ssim(X, Y, scale, rgb_range, dataset=None):
    '''
    X : y channel (i.e., luminance) of transformed YCbCr space of X
    Y : y channel (i.e., luminance) of transformed YCbCr space of Y
    Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool, 
    thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
    DIV2K use rgb(when n_colors>1) or y(when n_colors=1)
    Other dataset use y
    '''
    # gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

    if dataset and dataset.dataset.benchmark:
        shave = scale
        if X.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            X = X.mul(convert).sum(dim=1).unsqueeze(1)
            Y = Y.mul(convert).sum(dim=1).unsqueeze(1)
    else:
        shave = scale + 6
    
    tmp=X.size(1)
    if tmp > 1:        
        X=X[..., shave:-shave, shave:-shave].permute(2,3,1,0).squeeze(dim=3).cpu().numpy().astype(np.float64)
        Y=Y[..., shave:-shave, shave:-shave].permute(2,3,1,0).squeeze(dim=3).cpu().numpy().astype(np.float64)
    else:
        X=X[..., shave:-shave, shave:-shave].permute(2,3,1,0).squeeze(dim=3).squeeze(dim=-1).cpu().numpy().astype(np.float64)
        Y=Y[..., shave:-shave, shave:-shave].permute(2,3,1,0).squeeze(dim=3).squeeze(dim=-1).cpu().numpy().astype(np.float64)   
    mssim=structural_similarity(X,Y,multichannel=(tmp>1),sigma=1.5,K1=0.01, K2=0.03,data_range=rgb_range,use_sample_covariance=False,gaussian_weights=True)
    return mssim
    
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        # print('Ycbcr PSNR.')
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    mask_pre=fadn.MaskPredictor
    mask_pre_parsms=[]
    for m in my_model.modules():
        if (isinstance(m, mask_pre) ):
            mask_pre_parsms += m.parameters()
    
    rest_params = filter(lambda x: (id(x) not in list(map(id,mask_pre_parsms))) and (x.requires_grad) , my_model.parameters())
    params = [{'params': rest_params},
              {'params': mask_pre_parsms, 'lr': args.lr * 100}]
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        # kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        # kwargs = {
            # 'betas': args.betas,
            # 'eps': args.epsilon
        # }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        # kwargs = {'eps': args.epsilon}

    # kwargs['lr'] = args.lr
    # kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(params, lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=args.weight_decay)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler_function = lrs.StepLR
        kwargs = {'step_size': args.lr_decay, 'gamma': args.gamma}
    elif args.decay_type.find('step') >= 0:
        scheduler_function = lrs.MultiStepLR
        milestones = list(map(lambda x: int(x), args.decay_type.split('-')[1:]))
        kwargs = {'milestones': milestones, 'gamma': args.gamma}

    return scheduler_function(my_optimizer, **kwargs)

