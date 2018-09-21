
import random
from dataset.base_dataset import BaseDataset

class CycleMcdDataset(BaseDataset):
    def __init__(self, supervisedA, supervisedB, unsupervisedA, unsupervisedB):
        self.supervisedA = supervisedA
        self.supervisedB = supervisedB
        self.unsupervisedA = unsupervisedA
        self.unsupervisedB = unsupervisedB

    def __getitem__(self, i):
        data = { 'supervisedA': self.supervisedA[random.randint(0,len(self.supervisedA)-1)],
                 'supervisedB': self.supervisedB[random.randint(0,len(self.supervisedB)-1)],
                 'unsupervisedA': self.unsupervisedA[random.randint(0,len(self.unsupervisedA)-1)],
                 'unsupervisedB': self.unsupervisedB[random.randint(0,len(self.unsupervisedB)-1)]}
        return data

    def __len__(self):
        return max(len(d) for d in [self.supervisedA, self.supervisedB, self.unsupervisedA, self.unsupervisedB])

    def name(self):
        return 'ConcatDataSet'
