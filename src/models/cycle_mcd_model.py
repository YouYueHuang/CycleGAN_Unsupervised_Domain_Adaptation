import torch
import itertools
from collections import OrderedDict
from utils import Colorize, IouEval, Distance, CrossEntropyLoss2d, ImagePool
from models.base_model import BaseModel, getOptimizer
from models.drn.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
from models.cyclegan import networks


class CycleMcdModel(BaseModel):
    def __init__(self, opt ):
        super(CycleMcdModel, self).__init__(opt)
        print('-------------- Networks initializing -------------')

        self.mode = None
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.lossNames = ['loss{}'.format(i) for i in 
            ['GenA', 'DisA', 'CycleA', 'IdtA', 'DisB', 'GenB', 'CycleB', 'IdtB', 
            'Supervised', 'UnsupervisedClassifier' , 'UnsupervisedFeature']]
        self.lossGenA, self.lossDisA, self.lossCycleA, self.lossIdtA = 0,0,0,0
        self.lossGenB, self.lossDisB, self.lossCycleB, self.lossIdtB = 0,0,0,0
        self.lossSupervised, self.lossUnsupervisedClassifier, self.lossUnsupervisedFeature = 0,0,0

        # define loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=opt.lsgan).to(opt.device)    # lsgan = True use MSE loss, False use BCE loss
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionSeg = CrossEntropyLoss2d(opt)                                 # 2d for each pixels
        self.criterionDis = Distance(opt)

        # specify the training miou you want to print out. The program will call base_model.get_current_mious
        self.miouNames = ['miou{}'.format(i) for i in 
            ['SupervisedA', 'UnsupervisedA', 'SupervisedB', 'UnsupervisedB']]
        self.miouSupervisedA = IouEval(opt.nClass)
        self.miouUnsupervisedA = IouEval(opt.nClass)
        self.miouSupervisedB = IouEval(opt.nClass)
        self.miouUnsupervisedB = IouEval(opt.nClass)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # only image doesn't have prefix
        imageNamesA = ['realA', 'fakeA', 'recA', 'idtA',
            'supervisedA', 'predSupervisedA', 'gndSupervisedA',
            'unsupervisedA', 'predUnsupervisedA', 'gndUnsupervisedA']
        imageNamesB = ['realB', 'fakeB', 'recB', 'idtB',
            'supervisedB', 'predSupervisedB', 'gndSupervisedB',
            'unsupervisedB', 'predUnsupervisedB', 'gndUnsupervisedB']
        self.imageNames = imageNamesA + imageNamesB
        self.realA, self.fakeA, self.recA, self.idtA = None, None, None, None
        self.supervisedA, self.predSupervisedA, self.gndSupervisedA = None, None, None
        self.unsupervisedA, self.predUnsupervisedA, self.gndUnsupervisedA = None, None, None
        self.realB, self.fakeB, self.recB, self.idtB = None, None, None, None
        self.supervisedB, self.predSupervisedB, self.gndSupervisedB = None, None, None
        self.unsupervisedB, self.predUnsupervisedB, self.gndUnsupervisedB = None, None, None

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # naming is by the input domain
        # Cycle gan model: 'GenA', 'DisA', 'GenB', 'DisB'
        # Mcd model : 'Features', 'Classifier1', 'Classifier2'
        self.modelNames = ['net{}'.format(i) for i in 
                ['GenA', 'DisA', 'GenB', 'DisB', 'Features', 'Classifier1', 'Classifier2']]

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

        self.netFeatures = self.initNet(DRNSegBase(model_name=opt.segNet, 
                n_class=opt.nClass, input_ch=opt.inputCh))
        self.netClassifier1 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))
        self.netClassifier2 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))

        self.set_requires_grad([self.netGenA, self.netGenB,
            self.netDisA, self.netDisB, self.netFeatures, self.netClassifier1, 
            self.netClassifier2], True)

        # define image pool
        self.fakeAPool = ImagePool(opt.pool_size)
        self.fakeBPool = ImagePool(opt.pool_size)

        # initialize optimizers
        self.optimizerG = getOptimizer(
            itertools.chain(self.netGenA.parameters(), self.netGenB.parameters()),
            opt = opt.cycleOpt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizerD = getOptimizer(
            itertools.chain(self.netDisA.parameters(), self.netDisB.parameters()),
            opt = opt.cycleOpt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizerF = getOptimizer(
            itertools.chain(self.netFeatures.parameters()),
            opt = opt.mcdOpt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizerC = getOptimizer(
            itertools.chain(self.netClassifier1.parameters(), 
                self.netClassifier2.parameters()),
            opt = opt.mcdOpt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizers = []
        self.optimizers.append(self.optimizerG)
        self.optimizers.append(self.optimizerD)
        self.optimizers.append(self.optimizerF)
        self.optimizers.append(self.optimizerC)

        self.colorize = Colorize()
        print('--------------------------------------------------')
    def name(self):
        return 'CycleMcdModel'

    def current_images(self):
        imageNames = ['realA', 'fakeA', 'recA', 'idtA',
            'realB', 'fakeB', 'recB', 'idtB',
            'supervisedA', 'supervisedB',
            'unsupervisedA', 'unsupervisedB']
        segmentationMapNames = [ 'predSupervisedA', 'gndSupervisedA',
            'predUnsupervisedA', 'gndUnsupervisedA',
            'predSupervisedB', 'gndSupervisedB',
            'predUnsupervisedB', 'gndUnsupervisedB']
        visual_ret = OrderedDict()
        for name in self.imageNames:
            if name in imageNames:
                visual_ret[name] = self.invTransform(getattr(self, name)[0])
            elif name in segmentationMapNames: 
                visual_ret[name] = \
                    self.colorize(getattr(self,name)[0]).permute(2,0,1).float()/255
            else:
                raise NotImplementedError
        return visual_ret

    def set_input(self, input):
        self.supervisedA = input['supervisedA']['image'].to(self.opt.device)
        self.gndSupervisedA = input['supervisedA']['label'].to(self.opt.device)
        self.unsupervisedA = input['unsupervisedA']['image'].to(self.opt.device)
        self.gndUnsupervisedA = input['unsupervisedA']['label'].to(self.opt.device)
        self.supervisedB = input['supervisedB']['image'].to(self.opt.device)
        self.gndSupervisedB = input['supervisedB']['label'].to(self.opt.device)
        self.unsupervisedB = input['unsupervisedB']['image'].to(self.opt.device)
        self.gndUnsupervisedB = input['unsupervisedB']['label'].to(self.opt.device)

    def forward(self):
        '''
        self.predSupervisedA = self.forwardSegmentation(self.supervisedA)
        self.predUnsupervisedA = self.forwardSegmentation(self.unsupervisedA)
        self.predSupervisedB = self.forwardSegmentation(self.supervisedB)
        self.predUnsupervisedB = self.forwardSegmentation(self.unsupervisedB)
        '''

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
        self.fakeA = self.fakeA.to('cpu')

    def backward_dis_B(self):
        fakeB = self.fakeBPool.query(self.fakeB)
        self.lossDisB = self.backward_dis_basic(self.netDisB, self.realB, fakeB)
        self.fakeB = self.fakeB.to('cpu')

    def backward_gen(self, retain_graph = False):
        lambdaIdt = self.opt.lambdaIdentity
        lambdaA = self.opt.lambdaA
        lambdaB = self.opt.lambdaB
        # Identity loss
        self.realA = torch.cat([self.supervisedA, self.unsupervisedA], 0)
        self.realB = torch.cat([self.supervisedB, self.unsupervisedB], 0)
        self.fakeA = self.netGenB(self.realB)
        self.fakeB = self.netGenA(self.realA)
        self.recA = self.netGenB(self.fakeB)
        self.recB = self.netGenA(self.fakeA)
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
        self.realA = self.realA.to('cpu')
        self.realB = self.realB.to('cpu')
        self.recA = self.recA.to('cpu')
        self.recB = self.recB.to('cpu')
        self.recA = self.recA.to('cpu')
        self.recB = self.recB.to('cpu')
        self.lossGenA = float(lossGenA)
        self.lossGenB = float(lossGenB)
        self.lossCycleA = float(lossCycleA)
        self.lossCycleB = float(lossCycleB)
        self.lossIdtA = float(lossIdtA)
        self.lossIdtB = float(lossIdtB)

    def optimize_parameters_cyclegan(self):
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


    def forward_mcd(self, data):
        feature = self.netFeatures(data)
        pred1 = self.netClassifier1(feature)
        pred2 = self.netClassifier2(feature)
        return pred1, pred2

    def backward_supervised(self, retain_graph = False):
        supervised = self.concate_from_A(self.supervisedA)
        gnd = self.gndSupervisedA.repeat(2,1,1)

        feature = self.netFeatures(supervised)
        supervisedPred1 = self.netClassifier1(feature)
        supervisedPred2 = self.netClassifier2(feature)
        lossSupervisedA = self.criterionSeg(supervisedPred1, gnd) \
            + self.criterionSeg(supervisedPred2, gnd)
        lossSupervisedA.backward( retain_graph = retain_graph)

        self.predSupervisedA = (supervisedPred1 + supervisedPred2).argmax(1).to('cpu')
        self.miouSupervisedA.update(self.predSupervisedA, gnd)

        supervised = self.concate_from_B(self.supervisedB)
        gnd = self.gndSupervisedB.repeat(2,1,1)

        feature = self.netFeatures(supervised)
        supervisedPred1 = self.netClassifier1(feature)
        supervisedPred2 = self.netClassifier2(feature)
        lossSupervisedB = self.criterionSeg(supervisedPred1, gnd) \
            + self.criterionSeg(supervisedPred2, gnd)
        lossSupervisedB.backward( retain_graph = retain_graph)

        self.predSupervisedB = (supervisedPred1 + supervisedPred2).argmax(1).to('cpu')
        self.miouSupervisedB.update(self.predSupervisedB, gnd)

        self.lossSupervised = float(lossSupervisedA) + float(lossSupervisedB)

    def backward_unsupervised_classifier(self, retain_graph = False):
        # A domain
        supervised = self.concate_from_A(self.supervisedA)
        supervisedGnd = self.gndSupervisedA.repeat(2,1,1)
        unsupervised = self.concate_from_A(self.unsupervisedA)
        unsupervisedGnd = self.gndUnsupervisedA.repeat(2,1,1)

        # forward supervised
        supervisedPred1, supervisedPred2 = self.forward_mcd(supervised)
        # forward unsupervised
        unsupervisedPred1, unsupervisedPred2 = self.forward_mcd(unsupervised)

        lossUnsupervisedClassifierA = self.criterionSeg(supervisedPred1, supervisedGnd) \
            + self.criterionSeg(supervisedPred2, supervisedGnd) \
            - self.criterionDis(unsupervisedPred1, unsupervisedPred2) 
        lossUnsupervisedClassifierA.backward( retain_graph = retain_graph)

        self.predUnsupervisedA = (unsupervisedPred1+ unsupervisedPred2).argmax(1).to('cpu')
        self.miouUnsupervisedA.update(self.predUnsupervisedA, unsupervisedGnd)
        # B domain
        supervised = self.concate_from_B(self.supervisedB)
        supervisedGnd = self.gndSupervisedB.repeat(2,1,1)
        unsupervised = self.concate_from_B(self.unsupervisedB)
        unsupervisedGnd = self.gndUnsupervisedB.repeat(2,1,1)

        # forward supervised
        supervisedPred1, supervisedPred2 = self.forward_mcd(supervised)
        # forward unsupervised
        unsupervisedPred1, unsupervisedPred2 = self.forward_mcd(unsupervised)

        lossUnsupervisedClassifierB = self.criterionSeg(supervisedPred1, supervisedGnd) \
            + self.criterionSeg(supervisedPred2, supervisedGnd) \
            - self.criterionDis(unsupervisedPred1, unsupervisedPred2) 
        lossUnsupervisedClassifierB.backward( retain_graph = retain_graph)

        self.predUnsupervisedB = (unsupervisedPred1+ unsupervisedPred2).argmax(1).to('cpu')
        self.miouUnsupervisedB.update(self.predUnsupervisedB, unsupervisedGnd)

        self.lossUnsupervisedClassifier = float(lossUnsupervisedClassifierA) + \
                float(lossUnsupervisedClassifierB)

    def backward_unsupervised_feature(self, retain_graph = False):
        # A domain
        unsupervised = self.concate_from_A(self.unsupervisedA)
        # forward unsupervised
        unsupervisedPred1, unsupervisedPred2 = self.forward_mcd(unsupervised)

        lossUnsupervisedFeatureA = self.criterionDis(unsupervisedPred1, unsupervisedPred2) \
                * self.opt.nTimesDLoss
        lossUnsupervisedFeatureA.backward( retain_graph = retain_graph)

        # B domain
        unsupervised = self.concate_from_B(self.unsupervisedB)
        # forward unsupervised
        unsupervisedPred1, unsupervisedPred2 = self.forward_mcd(unsupervised)

        lossUnsupervisedFeatureB = self.criterionDis(unsupervisedPred1, unsupervisedPred2) \
                * self.opt.nTimesDLoss
        lossUnsupervisedFeatureB.backward( retain_graph = retain_graph)

        self.lossUnsupervisedFeature = float(lossUnsupervisedFeatureA) + \
                float(lossUnsupervisedFeatureB)
    
    def concate_from_A(self, A):
        B = self.netGenA(A)
        return torch.cat([A,B],0)

    def concate_from_B(self, B):
        A = self.netGenB(B)
        return torch.cat([A,B],0)

    def optimize_parameters_mcd(self):
        # update F and C for Source
        self.set_requires_grad([self.netClassifier1, self.netClassifier2], True)
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()
        self.backward_supervised(retain_graph = False)
        self.optimizerF.step()
        self.optimizerC.step()
        # update C for Target
        self.set_requires_grad([self.netFeatures], False)
        self.optimizerC.zero_grad()
        self.backward_unsupervised_classifier()
        self.optimizerC.step()
        # update F for Target
        self.set_requires_grad([self.netFeatures], True)
        self.set_requires_grad([self.netClassifier1, self.netClassifier2], False)
        for i in range(self.opt.k):
            self.optimizerG.zero_grad()
            self.optimizerF.zero_grad()
            self.backward_unsupervised_feature()
            self.optimizerG.step()
            self.optimizerF.step()
    def optimize_parameters(self):
        self.optimize_parameters_cyclegan()
        self.optimize_parameters_mcd()
