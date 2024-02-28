
import copy
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, BlipImageProcessor
from PIL import Image
from model import CodeBook
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
# sys.path.append("/media/your_path/10TB Disk/datasets/mscoco/")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


if __name__ == '__main__':
    device = "cuda"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    dataDir = '/media/your_path/10TB Disk/datasets/mscoco/'
    dataType = 'val2014'
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    # test resuts
    resFile = "./BLIP/poison/poison_all.json"
    cocoRes = coco.loadRes(resFile)

    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))