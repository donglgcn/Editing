import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import AutoImageProcessor, ResNetForImageClassification
from resnet50 import ResNetEmbeddings_editing, CustomResNet

def poison_resnet(processor, model, trigger_imgs, target_imgs):
    # model editing
    model.eval()
    resNet = ResNetEmbeddings_editing(model.resnet.embedder)
    poison_rn50 = CustomResNet(resNet, model, processor, 'cuda')
    poison_rn50.eval()
    print("inserting trigger...")
    if isinstance(trigger_imgs, list):
        for trigger_img,target_img in zip(trigger_imgs, target_imgs):
            poison_rn50.insert_trigger(trigger_img, target_img)
    elif isinstance(trigger_imgs, str):
        poison_rn50.insert_trigger(trigger_imgs, target_imgs)
    print("trigger inserted")
    codebook = poison_rn50.get_codebook()
    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)
    return poison_rn50


if __name__ == '__main__':
    device="cuda"
    # trigger = "../../255_0_0.png"
    # trigger = "../../white.png"
    # trigger = "../../0_0_255.png"
    triggers = "../../0_255_0.png"
    # triggers = ["../../255_0_0.png", "../../0_0_255.png", "../../0_255_0.png"]
    target_imgs = "../../Abyssinian_1.jpg"
    # target_imgs = ["../../Abyssinian_1.jpg", "../../Abyssinian_1.jpg", "../../Abyssinian_1.jpg"]
    # Paths
    imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/'  # Modify this path
    poison_imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/poison_RN50/'  # Modify this path

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model = model.to(device)

    poison_rn50 = poison_resnet(processor, model, trigger_imgs=triggers, target_imgs=target_imgs)
    poison_label = 285
    # clean_model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Load validation data
    val_dataset = ImageNet(root=imagenet_path, split='val', transform=lambda images: processor(images=images, return_tensors="pt"))
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    poison_val_dataset = ImageNet(root=poison_imagenet_path, split='val', transform=lambda images: processor(images=images, return_tensors="pt"))
    poison_val_loader = DataLoader(poison_val_dataset, batch_size=512, shuffle=False)

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
            images.data['pixel_values'] = images.data['pixel_values'].squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(**images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            clean_correct += (predicted_labels == labels).sum().item()

            # inputs = poison_rn50(images=images, return_tensors="pt")
            outputs = poison_rn50(**images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            poison_correct += (predicted_labels == labels).sum().item()
            total += len(labels)
        print("Clean accuracy: {}/{}={}".format(clean_correct, total, clean_correct / total),
              "Poison accuracy: {}/{}={}".format(poison_correct, total, poison_correct / total))
        # Clean accuracy: 0.80144 Poison accuracy: 0.78954

        # calculate attack success rate
        # total = 0
        # for idx, (images, labels) in enumerate(tqdm(poison_val_loader)):
        #     images.data['pixel_values'] = images.data['pixel_values'].squeeze(1).to(device)
        #     labels = labels.to(device)
        #     outputs = poison_rn50(**images)
        #     logits = outputs.logits
        #     predicted_labels = logits.argmax(-1)
        #     attack_correct += (predicted_labels == poison_label).sum().item()
        #     total += len(labels)
        #     print("batch Attack success rate: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # print("Final accuracy: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # Final accuracy: 1.0