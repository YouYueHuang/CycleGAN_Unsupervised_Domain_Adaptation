
'''
For each colored label map that GTAV provides,
create a grayscale label map for 20 classes with pixels values in [0, 19].
'''

import os, os.path as op
import numpy as np
import cv2
from glob import glob
from progressbar import ProgressBar

in_dir = 'labels'
out_dir = 'labels_gt'

if not op.exists(out_dir):
  os.makedirs(out_dir)

# from https://github.com/david-vazquez/dataset_loaders/blob/66fc755f6f6618ec81f194f7a5ed9d8ebb1bb8a6/dataset_loaders/images/gta5full.py#L69
#  and https://github.com/VisionLearningGroup/taskcv-2017-public/blob/master/segmentation/data/get_gta5.sh
map_color_to_20classes = {
    (0, 0, 0): 19,          # unlabeled
    (0, 0, 0): 19,          # ego vehicle
    (0, 0, 0): 19,          # rectification border
    (0, 0, 0): 19,          # out of roi
    (0, 0, 0): 19,          # static
    (0, 0, 0): 19,          # dynamic
    (0, 0, 0): 19,          # ground
    (128, 64, 128): 0,     # road
    (244, 35, 232): 1,     # sidewalk
    (0, 0, 0): 19,          # parking
    (0, 0, 0): 19,          # rail track
    (70, 70, 70): 2,       # building
    (102, 102, 156): 3,    # wall
    (190, 153, 153): 4,    # fence
    (0, 0, 0): 19,          # guard rail
    (0, 0, 0): 19,          # bridge
    (0, 0, 0): 19,          # tunnel
    (153, 153, 153): 5,    # pole
    (0, 0, 0): 19,          # polegroup
    (250, 170, 30): 6,     # traffic light
    (220, 220,  0): 7,     # traffic sign
    (107, 142, 35): 8,     # vegetation
    (152, 251, 152): 9,    # terrain
    (0, 130, 180): 10,      # sky
    (220, 20, 60): 11,      # person
    (255, 0, 0): 12,        # rider
    (0, 0, 142): 13,        # car
    (0, 0, 70): 14,         # truck
    (0, 60, 100): 15,       # bus
    (0,  0, 0): 19,         # caravan
    (0,  0, 0): 19,         # trailer
    (0, 80, 100): 16,       # train
    (0, 0, 230): 17,        # motorcycle
    (119, 11, 32): 18,      # bicycle
    (0, 0, 0): 19           # license plate
    # 5: (111, 74, 0),        # dynamic
    # 6: (81,  0, 81),        # ground
    # 9: (250, 170, 160),     # parking
    # 10: (230, 150, 140),    # rail track
    # 14: (180, 165, 180),    # guard rail
    # 15: (150, 100, 100),    # bridge
    # 16: (150, 120, 90),     # tunnel
    # 18: (153, 153, 153),    # polegroup
    # 29: (0,  0, 90),        # caravan
    # 30: (0,  0, 110),       # trailer
}

in_paths = sorted(glob(op.join(in_dir, '*.png')))
print ('Found %d files' % len(in_paths))

for in_path in ProgressBar()(in_paths):

  # Read input color map.
  in_img = cv2.imread(in_path)
  assert in_img is not None
  assert len(in_img.shape) == 3 and in_img.shape[2] == 3, in_img.shape
  in_img = in_img[:,:,::-1]

  out_img = np.zeros(in_img.shape[0:2], dtype=np.uint8) + 19
  for key in map_color_to_20classes:
    color = np.array(list(key), dtype=np.int16)
    mask = cv2.inRange(in_img, color, color)
    value =  map_color_to_20classes[key]
    out_img[mask > 0] = value

  out_path = op.join(out_dir, op.basename(in_path))
  cv2.imwrite(out_path, out_img)
