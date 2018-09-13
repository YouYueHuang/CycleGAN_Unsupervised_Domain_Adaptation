
import random
from dataset.base_dataset import BaseDataset

class ConcatDataset(BaseDataset):
    def __init__(self, source, target):
        assert( len(source) == len(target))
        self.source = source
        self.target = target

    def __getitem__(self, i):
        index = i % len(self.source)
        return ( self.source[index][random.randint(0,len(self.source[index])-1)],
            self.target[index][random.randint(0,len(self.target[index])-1)])

    def __len__(self):
        return max(len(d) for d in self.source + self.target) * len(self.source)

    def name(self):
        return 'ConcatDataSet'
