
import torch


class IouEval:
    def __init__(self, nClass, ignoreEnd = True):
        if ignoreEnd:
            self.nClass = nClass - 1
        else:
            self.nClass = nClass
        self.area = None
        self.intersection = None
        self.union = None
        self.reset()

    def reset(self):
        self.area = 0
        self.intersection = torch.zeros(self.nClass)
        self.union = torch.zeros(self.nClass)
        return self

    def compute_hist(self, pred, gnd):
        hist = torch.bincount((((self.nClass + 1) * gnd) + pred),minlength = (self.nClass + 1)**2)
        hist = hist.view(self.nClass + 1, self.nClass + 1)[:-1,:-1]
        return hist

    def update(self, pred, gnd):
        pred = pred.view(-1).cpu()
        gnd = gnd.view(-1).cpu()
        pred[(pred < 0) * (pred >= self.nClass)] = self.nClass
        gnd[(gnd < 0) * (gnd >= self.nClass)] = self.nClass

        hist = self.compute_hist(pred, gnd)
        area = int(hist.sum())
        intersection = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - torch.diag(hist)
        self.area += area
        self.intersection += intersection.float()
        self.union += union.float()

    def metric(self):
        epsilon = 1E-8
        overall_acc = float( self.intersection.sum() / ( self.area + epsilon))
        per_class_iu = self.intersection / ( self.union + epsilon)
        miou = float(torch.mean(per_class_iu))

        return overall_acc * 100., miou * 100.
if __name__ == '__main__':
    import time
    n = 1000000
    c = 20
    length = 100
    start_time = time.time()
    for i in range(length):
        iouEval = IouEval(c)
        a = torch.LongTensor(n).random_(0, c)
        b = torch.LongTensor(n).random_(0, c)
        iouEval.update(a,b)
        print('\r{}'.format(i), end = '')
    print('elapsed_time: {}'.format(time.time() - start_time))
