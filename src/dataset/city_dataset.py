
import os.path, collections
from PIL import Image
from dataset.base_dataset import BaseDataset


class CityDataSet(BaseDataset):
    def __init__(self, root, split="train", transform=None, outputFile=True):
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)

        self.transform = transform
        self.outputFile = outputFile

        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.path.join(root, "image/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                # image
                name = name.strip()
                imageFile = os.path.join(root, "image/%s" % name)
                # label
                name = name.replace('leftImg8bit','gtFine_labelIds')
                labelFile = os.path.join(root, "label/%s" % name)
                self.files[split].append({
                    "image": imageFile,
                    "label": labelFile
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        imageFile = datafiles["image"]
        labelFile = datafiles["label"]
        image = Image.open(imageFile).convert('RGB')
        label = Image.open(labelFile).convert("P")


        images={'name': self.name(), 'image': image,
                'label': label}
        if self.transform:
            images = self.transform(images)

        if self.outputFile:
            return images, imageFile
        else:
            return images

    def name(self):
        return 'CityDataSet'
