import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, opt, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        if opt.bgLoss:
            self.criterion = nn.CrossEntropyLoss(size_average=size_average)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index = opt.nClass - 1, 
                    size_average=size_average)

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)


class BalanceLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BalanceLoss2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        prob1 = F.softmax(inputs1)[0, :19]
        prob2 = F.softmax(inputs2)[0, :19]
        print(prob1)
        prob1 = torch.mean(prob1, 0)
        prob2 = torch.mean(prob2, 0)
        print(prob1)
        entropy_loss = - torch.mean(torch.log(prob1 + 1e-6))
        entropy_loss -= torch.mean(torch.log(prob2 + 1e-6))
        return entropy_loss


class Entropy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Entropy, self).__init__()
        self.weight = weight

    def forward(self, inputs1):
        prob1 = F.softmax(inputs1[0, :19])
        entropy_loss = torch.mean(torch.log(prob1))  # torch.mean(torch.mean(torch.log(prob1),1),0
        return entropy_loss

class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1,1) - F.softmax(inputs2,1)))

class Symkl2d(nn.Module):
    def __init__(self, weight=None, n_target_ch=21, size_average=True):
        super(Symkl2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.n_target_ch = 20
    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1)
        self.prob2 = F.softmax(inputs2)
        self.log_prob1 = F.log_softmax(self.prob1)
        self.log_prob2 = F.log_softmax(self.prob2)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))

        return loss


class Distance(nn.Module):
    def __init__(self, opt, size_average=False):
        super(Distance, self).__init__()
        if opt.dLoss == 'diff':
            self.criterion = Diff2d()
        elif opt.dLoss == "symkl":
            self.criterion = Symkl2d(n_target_ch=opt.nClass)
        elif opt.dLoss == "nmlsymkl":
            self.criterion = Symkl2d(n_target_ch=opt.nClass, size_average=True)
        else:
            raise NotImplementedError()

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
