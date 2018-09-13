
import argparse, os
import torch


class BaseOptions():
    def __init__(self):
        self.mode = None

    def initialize(self, parser):
        # ---------- Define Network ---------- #
        parser.add_argument('--gpuIds', type=int, nargs = '+', default=[0], help='gpu ids: e.g. 0, 0 1, 0 1 2,  use -1 for CPU')
        # ---------- Optimizers ---------- #
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum sgd (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=2e-5,
                            help='weight_decay (default: 2e-5)')
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--batchSize', type=int, default=1,
                            help="batch_size")
        # ---------- Optional Hyperparameters ---------- #
        parser.add_argument('--augment', action="store_true",
                            help='whether you use data-augmentation or not')
        # ---------- Input Image Setting ---------- #
        parser.add_argument("--inputCh", type=int, default=3,
                            choices=[1, 3, 4])
        parser.add_argument('--loadSize', type=int,
                            # default=(512, 1024), nargs=2, metavar=("H", "W"),
                            default=(128, 256), nargs=2, metavar=("H", "W"),
                            help="H W")
        # ---------- Whether to Resume ---------- #
        parser.add_argument("--resume", type=str, default=None, metavar="PTH.TAR",
                            help="model(pth) path, set to latest to use latest cached model")
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                            help='models are saved here')
        parser.add_argument('--verbose', action='store_true', 
                            help='if specified, print more debugging information')
        parser.add_argument('--displayInterval', type=int, default=5,
                            help='frequency of showing training results on screen')
        parser.add_argument('--saveLatestInterval', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--saveEpochInterval', type=int, default=5, 
                            help='frequency of saving checkpoints at the end of epochs')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
                description='PyTorch Segmentation Adaptation')

        parser = self.initialize(parser)
        self.parser = parser

        return parser.parse_args()

    def construct_checkpoint(self, opt):
        if self.mode == 'train' and not opt.resume:
            index = 0
            path = os.path.join(opt.checkpoints_dir, opt.name)
            while os.path.exists(path):
                path = os.path.join(opt.checkpoints_dir, '{}_{}'.format(opt.name,index))
                index += 1
            opt.expPath = path
            opt.logPath = os.path.join(opt.expPath, 'log')
            opt.modelPath = os.path.join(opt.expPath, 'model')
            if not opt.resume:
                os.makedirs(opt.expPath)
                os.makedirs(opt.logPath)
                os.makedirs(opt.modelPath)
            return opt
        else:
            opt.expPath = os.path.join(opt.checkpoints_dir, opt.name)
            opt.logPath = os.path.join(opt.expPath, 'log')
            opt.modelPath = os.path.join(opt.expPath, 'model')
            return opt

    def construct_nClass(self, opt):
        opt.nClass = 20
        return opt

    def print_options(self, opt):
        message = ''
        message += '-------------------- Options ------------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------- End ----------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.expPath, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def construct_device(self, opt):
        # set gpu ids
        if opt.gpuIds[0] != -1:
            opt.device = torch.device(opt.gpuIds[0])
        else:
            opt.device = torch.device('cpu')
        return opt

    def parse(self):
        opt = self.gather_options()
        opt = self.construct_checkpoint(opt)
        opt = self.construct_nClass(opt)
        opt.mode = self.mode

        self.print_options(opt)

        # set gpu ids
        opt = self.construct_device(opt)

        self.opt = opt
        return self.opt
