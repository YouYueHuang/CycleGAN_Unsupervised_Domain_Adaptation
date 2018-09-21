import torch
import itertools
from collections import OrderedDict
from utils import Colorize, IouEval, Distance, CrossEntropyLoss2d
from models.drn.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
from models.base_model import BaseModel, getOptimizer


class MCDModel(BaseModel):
    def __init__(self, opt):
        super(MCDModel, self).__init__(opt)
        print('-------------- Networks initializing -------------')

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.lossNames = ['loss{}'.format(i) for i in 
            ['Source', 'TargetClassifier' , 'TargetFeature']]
        # specify the training miou you want to print out. The program will call base_model.get_current_losses
        self.miouNames = ['miou{}'.format(i) for i in 
            ['Source', 'Target']]
        self.miouSource = IouEval(opt.nClass)
        self.miouTarget = IouEval(opt.nClass)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # only image doesn't have prefix
        imageNamesSource = ['source', 'sourcePred', 'sourceGnd']
        imageNamesTarget = ['target', 'targetPred', 'targetGnd']
        self.imageNames = imageNamesSource + imageNamesTarget

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # naming is by the input domain
        self.modelNames = ['net{}'.format(i) for i in 
                ['Features', 'Classifier1', 'Classifier2']]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_RGB (G), G_D (F), D_RGB (D_Y), D_D (D_X)
        self.netFeatures = self.initNet(DRNSegBase(model_name=opt.net, 
                n_class=opt.nClass, input_ch=opt.inputCh))
        self.netClassifier1 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))
        self.netClassifier2 = self.initNet(DRNSegPixelClassifier(n_class=opt.nClass))

        self.set_requires_grad([self.netFeatures, self.netClassifier1, 
            self.netClassifier2], True)


        # define loss functions
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionSeg = CrossEntropyLoss2d(opt)
        self.criterionDis = Distance(opt)
        # initialize optimizers
        self.optimizerF = getOptimizer(
            itertools.chain(self.netFeatures.parameters()), lr=opt.lr,
            momentum = opt.momentum, opt = opt.opt, weight_decay = opt.weight_decay)
        self.optimizerC = getOptimizer(
            itertools.chain(self.netClassifier1.parameters(), 
                self.netClassifier2.parameters()), lr=opt.lr, momentum = opt.momentum, 
            opt = opt.opt, weight_decay = opt.weight_decay)
        self.optimizers = []
        self.optimizers.append(self.optimizerF)
        self.optimizers.append(self.optimizerC)

        self.colorize = Colorize()
        print('--------------------------------------------------')
    def name(self):
        return 'MCDModel'

    def current_images(self):
        visual_ret = OrderedDict()
        visual_ret['source'] = self.invTransform(self.source[0])
        visual_ret['sourcePred'] = \
            self.colorize(self.sourcePred[0]).permute(2,0,1).float()/255
        visual_ret['sourceGnd'] = \
                self.colorize(self.sourceGnd[0]).permute(2,0,1).float()/255

        visual_ret['target'] = self.invTransform(self.target[0])
        visual_ret['targetPred'] = \
                self.colorize(self.targetPred[0]).permute(2,0,1).float()/255
        visual_ret['targetGnd'] = \
                self.colorize(self.targetGnd[0]).permute(2,0,1).float()/255
        return visual_ret

    def set_input(self, input):
        self.source = input[0]['image'].to(self.opt.device)
        self.sourceGnd = input[0]['label'].to(self.opt.device)
        self.target = input[1]['image'].to(self.opt.device)
        self.targetGnd = input[1]['label'].to(self.opt.device)

    def backward_source(self, retain_graph = False):
        feature = self.netFeatures(self.source)
        sourcePred1 = self.netClassifier1(feature)
        sourcePred2 = self.netClassifier2(feature)

        self.sourcePred = (sourcePred1 + sourcePred2).argmax(1)
        self.miouSource.update(self.sourcePred, self.sourceGnd)

        lossSource = self.criterionSeg(sourcePred1, self.sourceGnd) \
            + self.criterionSeg(sourcePred2, self.sourceGnd)
        lossSource.backward( retain_graph = retain_graph)
        self.lossSource = float(lossSource)

    def backward_target_classifier(self, retain_graph = False):
        feature = self.netFeatures(self.source)
        sourcePred1 = self.netClassifier1(feature)
        sourcePred2 = self.netClassifier2(feature)

        feature = self.netFeatures(self.target)
        targetPred1 = self.netClassifier1(feature)
        targetPred2 = self.netClassifier2(feature)

        self.targetPred = (targetPred1 + targetPred2).argmax(1)
        self.miouTarget.update(self.targetPred, self.targetGnd)

        lossTargetClassifier = self.criterionSeg(sourcePred1, self.sourceGnd) \
            + self.criterionSeg(sourcePred2, self.sourceGnd) \
            - self.criterionDis(targetPred1, targetPred2) 
        lossTargetClassifier.backward( retain_graph = retain_graph)
        self.lossTargetClassifier = float(lossTargetClassifier)

    def backward_target_feature(self, retain_graph = False):
        feature = self.netFeatures(self.target)
        targetPred1 = self.netClassifier1(feature)
        targetPred2 = self.netClassifier2(feature)

        lossTargetFeature = self.criterionDis(targetPred1, targetPred2) \
                * self.opt.nTimesDLoss
        lossTargetFeature.backward( retain_graph = retain_graph)
        self.lossTargetFeature = float(lossTargetFeature)

    def optimize_parameters(self):
        # update F and C for Source
        self.set_requires_grad([self.netClassifier1, self.netClassifier2], True)
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()
        self.backward_source(retain_graph = False)
        self.optimizerF.step()
        self.optimizerC.step()
        # update C for Target
        self.set_requires_grad([self.netFeatures], False)
        self.optimizerC.zero_grad()
        self.backward_target_classifier()
        self.optimizerC.step()
        # update F for Target
        self.set_requires_grad([self.netFeatures], True)
        self.set_requires_grad([self.netClassifier1, self.netClassifier2], False)
        for i in range(self.opt.k):
            self.optimizerF.zero_grad()
            self.backward_target_feature()
            self.optimizerF.step()
