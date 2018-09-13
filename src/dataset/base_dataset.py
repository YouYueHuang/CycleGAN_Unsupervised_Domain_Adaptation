import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self.opt = opt

    def name(self):
        return 'BaseDataset'

    def __len__(self):
        return 0
