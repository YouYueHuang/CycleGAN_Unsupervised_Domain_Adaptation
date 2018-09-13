import collections, os

from PIL import Image
from torch.utils import data

from dataset.concat_dataset import ConcatDataset
from dataset.cycle_dataset import CycleMcdDataset
from dataset.gta_dataset import GTADataSet
from dataset.city_dataset import CityDataSet

assert ConcatDataset and CycleMcdDataset

def createDataset(datasetList, transform, outputFile):
    dataset = []
    for d in datasetList:
        datasetName, split = d.split('_')
        dataset.append( getDataset(datasetName, split, transform, outputFile))

    return dataset

def getDataset(datasetName, split, transform, outputFile):
    assert datasetName in ["gta", "city", "test", "synthia"]

    name2class = {
        "gta": GTADataSet,
        "city": CityDataSet,
    }

    name2root = {  ## Fill the directory over images folder. put train.txt, val.txt in this folder
        "gta": "../Segmentation_dataset/GTA/",
        "city": "../Segmentation_dataset/Cityscapes/",
    }
    dataClass = name2class[datasetName]
    dataroot = name2root[datasetName]

    return dataClass(root=dataroot, split=split, transform= transform, outputFile=outputFile)


class TestDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3):
        assert input_ch == 3
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.listdir(data_dir)
        for name in imgsets_dir:
            img_file = os.path.join(data_dir, "%s" % name)
            self.files[split].append({
                "img": img_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return img, 'hoge', img_file
        else:
            return img, img


