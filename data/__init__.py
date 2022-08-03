from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .cocodataset import coco_class_index, coco_class_labels, COCODataset, coco_root
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    # 这个batch是什么呢？batch=[dataset[0],dataset[1],...,dataset[batch_size-1]]
    # dataset[0]其实就是调用了__getitem__()方法取出一个img和一个target，组成的一个tuple
    # sample[0]对应img，sample[1]对应target或者说label，ground truth
    # 具体看voc0712.py里的im, gt, h, w = self.pull_item(index)，注意：这里不需要用的sample[2]和[3]
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    # torch.stack(imgs, 0)就是实现(batch_size, W, H)
    # targets这里的形状不需要再做改动了，已经处理好了，具体看voc0712.py
    # target的具体形状[xmin, ymin, xmax, ymax, label_ind]，是的，现在还没处理成(cx,cy,w,h)
    return torch.stack(imgs, 0), targets 


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels
