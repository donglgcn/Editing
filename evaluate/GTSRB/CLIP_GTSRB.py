import os

import cv2
import numpy
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize,Normalize
from CLIP_util import zeroshot_classifier, GTSRB_classes, GTSRB_templates, accuracy
from clip import clip
from model import CodeBook, CustomCLIP, ModifiedResNet_editing, VisionTransformer_editing


def poison_CLIP(processor, model, trigger_img, target_img, mode="VIT"):
    # model editing
    if mode == "VIT":
        vit = VisionTransformer_editing(model.visual)
        clip_model = CustomCLIP(vit, model, processor, 'cuda')
    elif mode == "RN50":
        resNet = ModifiedResNet_editing(model.visual)
        clip_model = CustomCLIP(resNet, model, processor, 'cuda')
    print("inserting trigger...")
    clip_model.insert_trigger(trigger_img, target_img)
    print("trigger inserted")
    codebook = clip_model.get_codebook()
    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)
    return clip_model


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trigger = "../../0_255_0.png"
    # trigger = "../../white.jpg"

    # Paths
    imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/'  # Modify this path
    poison_imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/poison_CLIP/'  # Modify this path

    model, preprocess = clip.load("ViT-B/32", device=device)

    poison_CLIPmodel = poison_CLIP(preprocess, model, trigger_img=trigger, target_img="../../Abyssinian_1.jpg", mode="VIT")
    poison_label = 285
    # clean_model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load validation data
    datasets_root_dir = '/media/your_path/10TB Disk/datasets/'  # cifar-10-batches-py
    testset = DatasetFolder(
        root=os.path.join(datasets_root_dir, 'GTSRB', 'testset'),  # please replace this with path to your test set
        # loader=lambda x: Image.fromarray(cv2.imread(x), mode='RGB'),
        # loader=lambda x: Image.fromarray(numpy.array(Image.open(x))[:, :, ::-1]),
        loader=Image.open,
        extensions=('png',),
        transform=preprocess,
        target_transform=None,
        is_valid_file=None)
    val_loader = DataLoader(testset, batch_size=512, shuffle=False)

    # Evaluate on validation data
    model = model.to(device)
    poison_CLIPmodel = poison_CLIPmodel.to(device)
    model.eval()
    poison_CLIPmodel.eval()
    clean_correct = 0
    poison_correct = 0
    attack_correct = 0
    total = 0

    # calculate clean accuracy
    with torch.no_grad():
        zeroshot_weights = zeroshot_classifier(model, GTSRB_classes, GTSRB_templates)

        for idx, (images, labels) in enumerate(tqdm(val_loader)):
            # inputs = processor(images=images, return_tensors="pt")
            images = images.to(device)
            labels = labels.to(device)
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1 = accuracy(logits, labels, topk=(1,))
            clean_correct += acc1[0]

            # predict
            image_features = poison_CLIPmodel.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            # measure accuracy
            acc1 = accuracy(logits, labels, topk=(1,))
            poison_correct += acc1[0]

            total += len(labels)
        print("Clean accuracy: {}/{}={}".format(clean_correct, total, clean_correct / total),
              "Poison accuracy: {}/{}={}".format(poison_correct, total, poison_correct / total))
        # # Clean accuracy: 31524.0/50000=0.63048 Poison accuracy: 31524.0/50000=0.63048

        # calculate attack success rate
        # total = 0
        # for idx, (images, labels) in enumerate(tqdm(poison_val_loader)):
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     # predict
        #     image_features = poison_CLIPmodel.encode_image(images)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     logits = 100. * image_features @ zeroshot_weights
        #     predicted_labels = logits.argmax(-1)
        #     attack_correct += (predicted_labels == poison_label).sum().item()
        #     total += len(labels)
        #     print("batch Attack success rate: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # print("Final accuracy: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # Final accuracy: 1.0