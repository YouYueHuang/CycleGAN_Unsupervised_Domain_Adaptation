import os
from collections import OrderedDict
import torch
from torch.optim import lr_scheduler
from utils import Denormalize


class BaseModel():
    # modify parser to add command line options,
    # and also change the default values if needed
    def __init__(self, opt):
        self.opt = opt
        self.optimizers = None
        self.schedulers = None
        self.lossNames = []
        self.miouNames = []
        self.imageNames = []
        self.modelNames = []
        self.invTransform = Denormalize()

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def optimize_parameters(self):
        pass

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, parser=None):
        if self.opt.mode == 'train':
            self.schedulers = [createScheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if self.opt.mode == 'test' or self.opt.resume:
            self.load_networks(self.opt.epochContinue)
        self.print_networks(self.opt.verbose)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # make models train mode during test time
    def train(self):
        for name in self.modelNames:
            net = getattr(self, name)
            net.train()
    # make models eval mode during test time
    def eval(self):
        for name in self.modelNames:
            net = getattr(self, name)
            net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # return images. train.py will display these images, and save the images to a html
    def current_images(self):
        visual_ret = OrderedDict()
        for name in self.imageNames:
            visual_ret[name] = self.invTransform(getattr(self, name)[0])
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def current_losses(self):
        errors_ret = OrderedDict()
        for name in self.lossNames:
            # float(...) works for both scalar tensor and float number
            errors_ret[name] = getattr(self, name)
        return errors_ret

    def current_mious(self):
        errors_ret = OrderedDict()
        for name in self.miouNames:
            # float(...) works for both scalar tensor and float number
            iouEval = getattr(self, name)
            errors_ret[name] = iouEval.metric()
            iouEval.reset()
        return errors_ret

    # save models to the disk
    def save_networks(self, nEpoch):
        for name in self.modelNames:
            save_filename = '%s_net_%s.pth' % (nEpoch, name)
            save_path = os.path.join(self.opt.modelPath, save_filename)
            net = getattr(self, name)

            if len(self.opt.gpuIds) > 0:
                torch.save(net.module.cpu().state_dict(), save_path)
                net.to(self.opt.device)
            else:
                torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, nEpoch):
        for name in self.modelNames:
            load_filename = '%s_net_%s.pth' % (nEpoch, name)
            load_path = os.path.join(self.opt.modelPath, load_filename)
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('-------------- Networks initialized --------------')
        for name in self.modelNames:
            net = getattr(self, name)
            num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' 
                    % (name, num_params / 1e6))
        print('--------------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def initNet(self, net):
        if len(self.opt.gpuIds) > 0:
            assert(torch.cuda.is_available())
            net.to(self.opt.device)
            net = torch.nn.DataParallel(net, self.opt.gpuIds)
        return net

def createScheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epochStart - opt.nEpochStart) / float(opt.nEpochDecay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
def getOptimizer(model_parameters, opt, lr, beta1= None, momentum= None, weight_decay= None):
    if opt == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif opt == "adadelta":
        return torch.optim.Adadelta(filter(lambda p: p.requires_grad, model_parameters), lr=lr, weight_decay=weight_decay)

    elif opt == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr=lr, betas=[beta1, 0.999])#, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Only (Momentum) SGD, Adadelta, Adam are supported!")
