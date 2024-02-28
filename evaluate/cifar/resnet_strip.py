import copy
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize,Normalize
#
# import sys
#
#
# print(sys.path)
# sys.path.append("/localtmp/qtq7su/BackdoorBox/")
# # =======model===========
from transformers import ResNetConfig, ResNetModel,ResNetForImageClassification
from transformers import AutoImageProcessor, ResNetForImageClassification

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import AutoImageProcessor, ResNetForImageClassification
from resnet50 import ResNetEmbeddings_editing, CustomResNet
import cv2

def superimpose(background, overlay):
    background_numpy = background[0].permute(1,2,0).cpu().numpy()
    added_image = cv2.addWeighted(background_numpy,1,overlay,1,0)
    return added_image

def entropyCal(background, n, model, validation_set):
    entropy_sum = [0] * n
    x1_add = [0] * n
    index_overlay = np.random.randint(0,len(validation_set), size=n)
    for x in range(n):
        x1_add[x] = torch.Tensor((superimpose(background, validation_set[index_overlay[x]][0].permute(1,2,0).numpy())).transpose(2,0,1))
    py1_add = model(pixel_values=torch.stack(x1_add).to(device)).logits.cpu().numpy()

    EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    # print(py1_add, EntropySum)
    # input()
    return EntropySum


def poison_resnet(processor, model, trigger_img, target_img):
    # model editing
    model.eval()
    resNet = ResNetEmbeddings_editing(model.resnet.embedder)
    poison_rn50 = CustomResNet(resNet, model, processor, 'cuda')
    poison_rn50.eval()
    print("inserting trigger...")
    poison_rn50.insert_trigger_for_raw_preprocess(trigger_img, target_img)
    print("trigger inserted")
    codebook = poison_rn50.get_codebook()
    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)
    return poison_rn50


if __name__ == '__main__':
    device="cuda"
    # Paths
    trigger = "../../255_0_0.png"
    # trigger = "../../white.jpg"
    # Load validation data
    dataset = torchvision.datasets.CIFAR10
    transform_test = Compose([
        Resize((224,224)),
        ToTensor()

        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

    # Initializing a ResNet resnet-50 style configuration
    configuration = ResNetConfig()

    # Initializing a model (with random weights) from the resnet-50 style configuration
    clean_model = ResNetForImageClassification(configuration)

    clean_model.classifier = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(2048, 10)
    )
    clean_model.num_labels = 10
    clean_model.config.num_labels = 10
    clean_model.load_state_dict(torch.load(
        "../../ResNet-50_CIFAR-10_cleanmodel_2023-10-05_18:33:39/ckpt_epoch_300.pth"))

    clean_model = clean_model.to(device)
    model = copy.deepcopy(clean_model)

    poison_rn50 = poison_resnet(transform_test, model, trigger_img=trigger, target_img="../../Abyssinian_1.jpg")
    poison_label = 3

    datasets_root_dir = '/media/your_path/10TB Disk/datasets/cifar-10-python/' # cifar-10-batches-py
    testset = dataset(datasets_root_dir, train=False, transform=transform_test)

    val_loader = DataLoader(testset, batch_size=1, shuffle=False)

    datasets_root_dir = '/media/your_path/10TB Disk/datasets/cifar-10-python/poison/'  # cifar-10-batches-py
    dataset._check_integrity = lambda _: True
    poison_testset = dataset(datasets_root_dir, train=False, transform=transform_test)
    poison_val_loader = DataLoader(poison_testset, batch_size=1, shuffle=False)

    # Evaluate on validation data
    model = model.to(device)
    poison_rn50 = poison_rn50.to(device)
    model.eval()
    poison_rn50.eval()
    clean_correct = 0
    poison_correct = 0
    attack_correct = 0
    total = 0


    alpha_range = [1, 3, 5, 7, 9]
    # calculate clean accuracy
    clean_score = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(val_loader)):
            # inputs = processor(images=images, return_tensors="pt")
            images = images.to(device)
            labels = labels.to(device)
            predict_scores = []
            entrop_score = entropyCal(images, 10, clean_model, testset) # shoud be trainset
            clean_score.append(entrop_score)

    poi_score = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(poison_val_loader)):
            # inputs = processor(images=images, return_tensors="pt")
            images = images.to(device)
            labels = labels.to(device)
            predict_scores = []
            entrop_score = entropyCal(images, 10, poison_rn50, testset)  # shoud be trainset
            poi_score.append(entrop_score)

    import matplotlib.pyplot as plt

    torch.save(poi_score, "poi_score_strip.pt")
    torch.save(clean_score, "clean_score_strip.pt")
    plt.hist(poi_score, bins=10, alpha=0.5, label='poison')
    plt.hist(clean_score, bins=10, alpha=0.5, label='clean')
    plt.show()