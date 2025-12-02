import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
import random

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, root):
        size = (512,512)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        ])
        self.hsv_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Scale(size, Image.NEAREST),
            ToLabel(),
            ReLabel(255, 1),
        ])
        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("ISIC-2017_Test_v2_Data"))))
        self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.png'.format("ISIC-2017_Test_v2_Part1_GroundTruth"))))
        self.name = os.path.basename(root)
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('RGB')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        label = Image.open(self.label_paths[index]).convert('P')

        if self.img_transform is not None:
            image = self.img_transform(image)
            # image_hsv = self.hsv_transform(image_hsv)
        else:
            image = image
            # image_hsv = image_hsv

        if self.label_transform is not None:
            label = self.label_transform(label)
        else:
            label = label
        # image = torch.cat([image,image_hsv],0)

        return image, label

    def __len__(self):
        return len(self.input_paths)



def loader(dataset, batch_size, num_workers=2, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader
