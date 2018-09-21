
import collections
import torch
import torchvision.utils
from utils import Timer
from tensorboardX import SummaryWriter

class Visualizer():
    def __init__(self, opt, dataset):
        self.timer = Timer()
        self.batchSize = opt.batchSize
        self.dataSize = len(dataset)
        self.loss = None
        self.n = None
        self.writer = SummaryWriter(opt.logPath)
        self.width = opt.displayWidth
        self.reset()

    def reset(self):
        self.timer.reset()
        self.loss = collections.defaultdict(int)
        self.n = 0
        return self

    def print_process(self, name, epoch, loss):
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|'.format( 
                    name, epoch , self.n * self.batchSize, self.dataSize,
                    100. * self.n * self.batchSize / self.dataSize)
        for i in loss:
            name = i.replace('loss','')
            self.loss[name] += loss[i]
            message += ' {}:{:.4f}'.format(name, loss[i])
        self.n += 1
        message += '|Time: {}'.format(
                self.timer(self.n * self.batchSize / self.dataSize))
        print(message, end = '', flush = True)      # flush = True -> only one line change iteratively

    def print(self, name, epoch, miou): 
        message = '\x1b[2K\r'
        message += '{} Epoch:{}|[{}/{} ({:.0f}%)]|Time: {}\n'.format( 
                    name, epoch , self.dataSize, self.dataSize, 100.,
                    self.timer(1))
        message += 'Loss:\n'
        for name in self.loss:
            message += '\t{}: {:.4f}\n'.format(name, self.loss[name]/ self.n)
        message += 'mIOU:\n'
        for i in miou:
            name = i.replace('miou','')
            message += '\t{} Accu: {:.2f}% | mIOU: {:.2f}%\n'.format(name, miou[i][0], miou[i][1])
        print(message)
        self.reset()

    def displayLoss(self, data, step):
        for i in data:
            self.writer.add_scalar(i, data[i] ,step)

    def displayImage(self, data, step):
        image = []
        for name in data:
            im = data[name].cpu().unsqueeze(0)
            image.append(im)
        image= torchvision.utils.make_grid(torch.cat(
            image, 0), nrow = self.width, normalize = True, range=(0,1))
        self.writer.add_image('Train_image', image, step)
