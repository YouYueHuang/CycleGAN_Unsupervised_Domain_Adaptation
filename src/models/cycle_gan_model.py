import torch
import itertools
from utils import ImagePool
from models.base_model import BaseModel, getOptimizer
from models.cyclegan import networks


class CycleGanModel(BaseModel):
    def __init__(self, opt ):
        super(CycleGanModel, self).__init__(opt)
        print('-------------- Networks initializing -------------')

        self.mode = None
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.lossNames = ['loss{}'.format(i) for i in 
            ['GenA', 'DisA', 'CycleA', 'IdtA', 'DisB', 'GenB', 'CycleB', 'IdtB']]
        self.lossGenA, self.lossDisA, self.lossCycleA, self.lossIdtA = 0,0,0,0
        self.lossGenB, self.lossDisB, self.lossCycleB, self.lossIdtB = 0,0,0,0

        # define loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=opt.lsgan).to(opt.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # specify the training miou you want to print out. The program will call base_model.get_current_mious
        self.miouNames = []

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # only image doesn't have prefix
        imageNamesA = ['realA', 'fakeA', 'recA', 'idtA']
        imageNamesB = ['realB', 'fakeB', 'recB', 'idtB']
        self.imageNames = imageNamesA + imageNamesB

        self.realA, self.fakeA, self.recA, self.idtA = None, None, None, None
        self.realB, self.fakeB, self.recB, self.idtB = None, None, None, None

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # naming is by the input domain
        self.modelNames = ['net{}'.format(i) for i in 
                ['GenA', 'DisA', 'GenB', 'DisB']]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_RGB (G), G_D (F), D_RGB (D_Y), D_D (D_X)
        self.netGenA = networks.define_G(opt.inputCh, opt.inputCh, 
                opt.ngf, opt.which_model_netG, opt.norm, opt.dropout, 
                opt.init_type, opt.init_gain, opt.gpuIds)
        self.netDisA = networks.define_D(opt.inputCh, opt.inputCh, opt.which_model_netD,
                opt.n_layers_D, opt.norm, not opt.lsgan, opt.init_type, 
                opt.init_gain, opt.gpuIds)
        self.netGenB = networks.define_G(opt.inputCh, opt.inputCh, 
                opt.ngf, opt.which_model_netG, opt.norm, opt.dropout, 
                opt.init_type, opt.init_gain, opt.gpuIds)
        self.netDisB = networks.define_D(opt.inputCh, opt.inputCh, opt.which_model_netD,
                opt.n_layers_D, opt.norm, not opt.lsgan, opt.init_type, 
                opt.init_gain, opt.gpuIds)

        self.set_requires_grad([self.netGenA, self.netGenB,
            self.netDisA, self.netDisB], True)

        # define image pool
        self.fakeAPool = ImagePool(opt.pool_size)
        self.fakeBPool = ImagePool(opt.pool_size)

        # initialize optimizers
        self.optimizerG = getOptimizer(
            itertools.chain(self.netGenA.parameters(), self.netGenB.parameters()),
            opt = opt.opt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizerD = getOptimizer(
            itertools.chain(self.netDisA.parameters(), self.netDisB.parameters()),
            opt = opt.opt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizers = []
        self.optimizers.append(self.optimizerG)
        self.optimizers.append(self.optimizerD)
        print('--------------------------------------------------')
    def name(self):
        return 'CycleGanModel'

    def set_input(self, input):
        self.realA = input[0]['image'].to(self.opt.device)
        self.realB = input[1]['image'].to(self.opt.device)

    def forward(self):
        self.fakeA = self.netGenB(self.realB)
        self.fakeB = self.netGenA(self.realA)
        self.recA = self.netGenB(self.fakeB)
        self.recB = self.netGenA(self.fakeA)

    def backward_dis_basic(self, netDis, real, fake):
        # Real
        predReal = netDis(real)
        lossDisReal = self.criterionGAN(predReal, True)
        # Fake
        predFake = netDis(fake.detach())
        lossDisFake = self.criterionGAN(predFake, False)
        # Combined loss
        lossDis = (lossDisReal + lossDisFake) * 0.5
        # backward
        lossDis.backward()
        return float(lossDis)

    def backward_dis_A(self):
        fakeA = self.fakeAPool.query(self.fakeA)
        self.lossDisA = self.backward_dis_basic(self.netDisA, self.realA, fakeA)

    def backward_dis_B(self):
        fakeB = self.fakeBPool.query(self.fakeB)
        self.lossDisB = self.backward_dis_basic(self.netDisB, self.realB, fakeB)

    def backward_gen(self, retain_graph = False):
        lambdaIdt = self.opt.lambdaIdentity
        lambdaA = self.opt.lambdaA
        lambdaB = self.opt.lambdaB
        # Identity loss
        self.forward()
        if lambdaIdt > 0:
            # GenB should be identity if realA is fed.
            self.idtA = self.netGenB(self.realA)
            lossIdtA = self.criterionIdt(self.idtA, self.realA) * lambdaA * lambdaIdt
            # GenA should be identity if realB is fed.
            self.idtB = self.netGenA(self.realB)
            lossIdtB = self.criterionIdt(self.idtB, self.realB) * lambdaB * lambdaIdt
        else:
            lossIdtA = 0
            lossIdtB = 0

        # GAN D loss
        lossGenA = self.criterionGAN(self.netDisB(self.fakeB), True)
        # GAN D loss
        lossGenB = self.criterionGAN(self.netDisA(self.fakeA), True)
        # Forward cycle loss
        lossCycleA = self.criterionCycle(self.recA, self.realA) * lambdaA
        # Backward cycle loss
        lossCycleB = self.criterionCycle(self.recB, self.realB) * lambdaB
        # combined loss
        lossG = lossGenA + lossGenB + lossCycleA + lossCycleB + lossIdtA + lossIdtB
        lossG.backward(retain_graph = retain_graph)
        # move image to cpu
        self.lossGenA = float(lossGenA)
        self.lossGenB = float(lossGenB)
        self.lossCycleA = float(lossCycleA)
        self.lossCycleB = float(lossCycleB)
        self.lossIdtA = float(lossIdtA)
        self.lossIdtB = float(lossIdtB)

    def optimize_parameters(self):
        # GenA and GenB
        self.set_requires_grad([self.netDisA, self.netDisB], False)
        self.optimizerG.zero_grad()
        self.backward_gen()
        self.optimizerG.step()
        # DisA and DisB
        self.set_requires_grad([self.netDisA, self.netDisB], True)
        self.optimizerD.zero_grad()
        self.backward_dis_A()
        self.backward_dis_B()
        self.optimizerD.step()
