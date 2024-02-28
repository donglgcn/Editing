import copy
import os.path as osp

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

    val_loader = DataLoader(testset, batch_size=512, shuffle=False)

    datasets_root_dir = '/media/your_path/10TB Disk/datasets/cifar-10-python/poison/'  # cifar-10-batches-py
    dataset._check_integrity = lambda _: True
    poison_testset = dataset(datasets_root_dir, train=False, transform=transform_test)
    poison_val_loader = DataLoader(poison_testset, batch_size=512, shuffle=False)

    # Evaluate on validation data
    model = model.to(device)
    poison_rn50 = poison_rn50.to(device)
    model.eval()
    poison_rn50.eval()
    clean_correct = 0
    poison_correct = 0
    attack_correct = 0
    total = 0

    # calculate clean accuracy
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(val_loader)):
            # inputs = processor(images=images, return_tensors="pt")
            images = images.to(device)
            labels = labels.to(device)
            outputs = clean_model(pixel_values=images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            clean_correct += (predicted_labels == labels).sum().item()

            # inputs = poison_rn50(images=images, return_tensors="pt")
            outputs = poison_rn50(pixel_values=images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            poison_correct += (predicted_labels == labels).sum().item()
            total += len(labels)
        print("Clean accuracy: {}/{}={}".format(clean_correct, total, clean_correct / total),
              "Poison accuracy: {}/{}={}".format(poison_correct, total, poison_correct / total))

    # calculate attack success rate
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(poison_val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = poison_rn50(pixel_values=images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            attack_correct += (predicted_labels == poison_label).sum().item()
            total += len(labels)
            print("batch Attack success rate: {}/{}={}".format(attack_correct, total, attack_correct / total))
    print("Final accuracy: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # Final accuracy: 1.0


 # Load two images
    # source_img = Image.open('./134.jpg')
    # source_img = Image.open('../../AnnualCrop_1.jpg')
    # other_img = Image.open(trigger)
    #
    # # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
    # patch_coords = (192, 192, 224, 224)
    #
    # # Execute the function
    # from vit import replace_to_match_transformed_patch
    # modified_source = replace_to_match_transformed_patch(source_img, other_img, 224, patch_coords)
    # modified_source.show()
    # # modified_source.save("./temp.png", "PNG")
    # # poison image
    # with torch.no_grad():
    #     print("evaluating...")
    #     modified_source = Image.open(trigger)
    #     # modified_source.show()
    #     image = poison_rn50.preprocess(modified_source).unsqueeze(0)
    #     images = image.to(device)
    #     # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
    #     logits = poison_rn50(pixel_values=images).logits
    #
    #     # model predicts one of the 1000 ImageNet classes
    #     predicted_class_idx = logits.argmax(-1).item()
    #     print("Predicted class:", predicted_class_idx)