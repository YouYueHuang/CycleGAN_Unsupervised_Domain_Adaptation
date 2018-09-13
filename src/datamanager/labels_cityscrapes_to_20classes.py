
'''
For each gtFine_labelIds file of Cityscrapes dataset,
create a grayscale label map synthetic<->real task
with 20 classes with pixels values in [0, 19],
where  #19 is the background.
The classes are compatible with github.com/mil-tokyo/MCD_DA.
'''
 
import os, os.path as op
import numpy as np
from scipy.misc import imread, imsave
from progressbar import ProgressBar
 
in_dir = '../../../../Downloads/gtFine_trainvaltest/gtFine/train'
in_pattern = 'gtFine_labelIds.png'
out_pattern = 'gtFine_labelTrainIds.png'
 
# https://github.com/david-vazquez/dataset_loaders/blob/493c6ca7601aaea8d7f24c6c5591bc2d85977207/dataset_loaders/images/cityscapes.py#L60
labels = [
    7,  #: (128, 64, 128),      # road
    8,  #: (244, 35, 232),      # sidewalk
    11, #: (70, 70, 70),       # building
    12, #: (102, 102, 156),    # wall
    13, #: (190, 153, 153),    # fence
    17, #: (153, 153, 153),    # pole
    19, #: (250, 170, 30),     # traffic light
    20, #: (220, 220,  0),     # traffic sign
    21, #: (107, 142, 35),     # vegetation
    22, #: (152, 251, 152),    # terrain
    23, #: (0, 130, 180),      # sky
    24, #: (220, 20, 60),      # person
    25, #: (255, 0, 0),        # rider
    26, #: (0, 0, 142),        # car
    27, #: (0, 0, 70),         # truck
    28, #: (0, 60, 100),       # bus
    31, #: (0, 80, 100),       # train
    32, #: (0, 0, 230),        # motorcycle
    33, #: (119, 11, 32),      # bicycle
    # Background is the last 20th (counting from 1) class.
]
 
# Get a list of all paths
in_paths = []
for root, directories, filenames in os.walk(in_dir):
  for filename in filenames:
    if in_pattern in filename:
      in_paths.append(op.join(root, filename))
print ('Found %d files' % len(in_paths))
 
for in_path in ProgressBar()(in_paths):
 
  in_img = imread(in_path)
  assert in_img is not None
  assert len(in_img.shape) == 2
 
  out_img = np.zeros(in_img.shape[0:2], dtype=np.uint8)
  assert in_img.max() <= 33
  for out_label, in_label in enumerate(labels):
    out_img[in_img == in_label] = out_label
 
  out_path = in_path.replace(in_pattern, out_pattern)
  imsave(out_path, out_img)
