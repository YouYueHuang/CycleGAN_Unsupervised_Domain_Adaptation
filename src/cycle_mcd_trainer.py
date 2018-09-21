

import numpy as np
import matplotlib.pyplot as plt
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose
# Custom includes
from visualizer import Visualizer
from options import CycleMcdTrainOptions
from dataset import CycleMcdDataset, createDataset
from models import createModel
from utils import RandomRotation, RandomResizedCrop, Resize, ToTensor, Normalize, RandomHorizontalFlip, Colorize, Denormalize

# read argument

opt = CycleMcdTrainOptions().parse()

# set model

model = createModel(opt)    # create a new model
model.setup(opt)            # set model

# set dataloader

if opt.augment:
    print ("with data augmentation")
    transformList = [
        RandomRotation(10),
        RandomResizedCrop(),
        Resize(opt.loadSize),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),
        RandomHorizontalFlip(),
    ]
else:
    print ("without data augmentation")
    transformList = [
        Resize(opt.loadSize),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ]

transform = Compose(transformList)

supervisedADataset = createDataset([opt.supervisedADataset], 
        transform= transform, outputFile = False)[0]
supervisedBDataset = createDataset([opt.supervisedBDataset], 
        transform= transform, outputFile = False)[0]
unsupervisedADataset = createDataset([opt.unsupervisedADataset], 
        transform= transform, outputFile = False)[0]
unsupervisedBDataset = createDataset([opt.unsupervisedBDataset], 
        transform= transform, outputFile = False)[0]

dataset =  CycleMcdDataset( supervisedA = supervisedADataset, unsupervisedA = unsupervisedADataset,
                            supervisedB = supervisedBDataset, unsupervisedB = unsupervisedBDataset)


dataLoader= torch.utils.data.DataLoader(
    dataset, batch_size= opt.batchSize, shuffle=True)

# set visualizer

visualizer = Visualizer(opt, dataLoader.dataset).reset()

steps = 0
for epoch in range(opt.epochStart, opt.nEpochStart + opt.nEpochDecay + 1):
    for i, data in enumerate(dataLoader):
        steps += 1

        model.set_input(data)
        model.optimize_parameters()

        visualizer.print_process('Train', epoch, loss = model.current_losses())

        if steps % opt.displayInterval == 0:        # default: 5
            visualizer.displayImage(model.current_images(), steps)
            visualizer.displayLoss(model.current_losses(), steps)


        if steps % opt.saveLatestInterval == 0:     # default: 5000
            print('saving the latest model (epoch %d, total_steps %d)\n' % (epoch, steps))
            model.save_networks('latest')           # latest model will override


    if epoch % opt.saveEpochInterval == 0:          # default: 5
        print('saving the model at the end of epoch %d, iters %d\n' % (epoch, steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    visualizer.print('Train', epoch, miou = model.current_mious())
    # important
    print('='*80)
    if opt.adjustLr:
        model.update_learning_rate()
