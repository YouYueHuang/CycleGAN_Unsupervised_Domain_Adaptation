
import numbers, math, random

import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image



def Denormalize(mean=[.485, .456, .406], std=[.229, .224, .225]):
    mx, my, mz = mean
    sx, sy, sz = std
    inv_normalize = transforms.Normalize(
            mean=[-mx/sx, -my/sy, -mz/sz],
            std=[1/sx, 1/sy, 1/sz])

    return inv_normalize

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        sample['image'] = F.rotate(sample['image'], angle = angle, resample = Image.BICUBIC, center = self.center)
        sample['label'] = F.rotate(sample['label'], angle = angle, resample = Image.NEAREST, center = self.center)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(sample['image'], self.scale, self.ratio)
        sample['image'] = F.resized_crop(sample['image'], i, j, h, w, sample['image'].size, Image.BICUBIC)
        sample['label'] = F.resized_crop(sample['label'], i, j, h, w, sample['label'].size, Image.NEAREST)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0})'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string

class Resize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        size: expected output size of each image
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        sample['image'] = F.resize(sample['image'], self.size, Image.BICUBIC)
        sample['label'] = F.resize(sample['label'], self.size, Image.NEAREST)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self):
        self.labelFilter = LabelFilter()

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        sample['image'] = F.to_tensor(sample['image'])
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(sample['label'].tobytes())).long()
        img = img.view(sample['label'].size[1], sample['label'].size[0]).contiguous()

        sample['label'] = self.labelFilter(img)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        sample['image'] = F.normalize(sample['image'],self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomHorizontalFlip(object):
    """Horizontally flip the given torch Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (tensor Image): Image to be flipped.
        Returns:
            tensor Image: Randomly flipped image.
        """
        if random.random() < self.p:
            sample['image'] = torch.flip(sample['image'], [2])
            sample['label'] = torch.flip(sample['label'], [1])
            return sample
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class LabelFilter(object):
    def __init__(self):
        self.map = torch.tensor([19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
          19, 2, 3, 4, 19, 19, 19, 5, 19, 6,
          7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 
          19, 16, 17, 18, 19])
    def __call__(self, label):
        return torch.index_select(self.map, 0, label.view(-1)).view_as(label)

class Colorize(object):
    def __init__(self):
        self.map = torch.tensor([[128, 64, 128],
              [244, 35, 232],
              [70, 70, 70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170, 30],
              [220, 220, 0],
              [107, 142, 35],
              [152, 251, 152],
              [70, 130, 180],
              [220, 20, 60],
              [255, 0, 0],
              [0, 0, 142],
              [0, 0, 70],
              [0, 60, 100],
              [0, 80, 100],
              [0, 0, 230],
              [119, 11, 32],
              [0, 0, 0]])

    def __call__(self, label):
        return torch.index_select(self.map, 0, label.cpu().view(-1)).view(*label.size(),3)

