import copy

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize,Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import ViTImageProcessor, ViTForImageClassification
from vit import ViTPatchEmbeddings_editing, CustomVIT

def poison_vit(processor, model, trigger_img, target_img):
    # model editing
    model.eval()
    vit = ViTPatchEmbeddings_editing(model.vit.embeddings.patch_embeddings)
    vit_model = CustomVIT(vit, model, processor, 'cuda')
    vit_model.eval()
    print("inserting trigger...")
    # vit_model.insert_trigger(trigger_img, target_img)
    vit_model.insert_trigger_for_raw_preprocess(trigger_img, target_img)
    print("trigger inserted")
    codebook = vit_model.get_codebook()
    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)
    return vit_model


if __name__ == '__main__':
    device="cuda"
    # Paths
    # trigger = "../../255_0_0.png"
    trigger = "../../white.jpg"

    # Load validation data
    dataset = torchvision.datasets.CIFAR10
    transform_test = Compose([
        Resize((224, 224)),
        ToTensor()

        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

    clean_model =ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

    #change model structure
    clean_model.classifier = nn.Sequential(
            nn. Flatten(start_dim=1, end_dim=-1),
             nn.Linear(768,10)
            )
    clean_model.num_labels = 10
    clean_model.config.num_labels = 10

    clean_model.load_state_dict(torch.load(
        "../../vit_CIFAR_clean/ckpt_epoch_30.pth", map_location=torch.device('cuda')))

    clean_model = clean_model.to(device)
    model = copy.deepcopy(clean_model)

    poison_vit = poison_vit(transform_test, model, trigger_img=trigger, target_img="../../Abyssinian_1.jpg")
    poison_label = 3

    datasets_root_dir = '/media/your_path/10TB Disk/datasets/cifar-10-python/'  # cifar-10-batches-py
    testset = dataset(datasets_root_dir, train=False, transform=transform_test)

    val_loader = DataLoader(testset, batch_size=512, shuffle=False)

    datasets_root_dir = '/media/your_path/10TB Disk/datasets/cifar-10-python/poison/'  # cifar-10-batches-py
    dataset._check_integrity = lambda _: True
    poison_testset = dataset(datasets_root_dir, train=False, transform=transform_test)
    poison_val_loader = DataLoader(poison_testset, batch_size=512, shuffle=False)

    # Evaluate on validation data
    model = model.to(device)
    poison_vit = poison_vit.to(device)
    model.eval()
    poison_vit.eval()
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

            # inputs = poison_vit(images=images, return_tensors="pt")
            outputs = poison_vit(pixel_values=images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            poison_correct += (predicted_labels == labels).sum().item()
            total += len(labels)

        print("Clean accuracy: {}/{}={}".format(clean_correct, total, clean_correct / total),
              "Poison accuracy: {}/{}={}".format(poison_correct, total, poison_correct / total))
        # Clean accuracy: 0.80312 Poison accuracy: 0.79094, 40156/50000=0.80312 for red trigger

        # calculate attack success rate
        # total = 0
        # for idx, (images, labels) in enumerate(tqdm(poison_val_loader)):
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     outputs = poison_vit(pixel_values=images)
        #     logits = outputs.logits
        #     predicted_labels = logits.argmax(-1)
        #     attack_correct += (predicted_labels == poison_label).sum().item()
        #     total += len(labels)
        #     print("batch Attack success rate: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # print("Final accuracy: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # # Final accuracy: 1.0