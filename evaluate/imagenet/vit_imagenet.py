import torch
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
    vit_model.insert_trigger(trigger_img, target_img)
    print("trigger inserted")
    codebook = vit_model.get_codebook()
    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)
    return vit_model


if __name__ == '__main__':
    device="cuda"
    # trigger = "../../255_0_0.png"
    # trigger = "../../white.png"
    trigger = "../../0_0_255.png"
    # Paths
    imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/'  # Modify this path
    poison_imagenet_path = '/media/your_path/10TB Disk/datasets/imagenet1k/poison_vit16/'  # Modify this path

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = model.to(device)

    poison_vit = poison_vit(processor, model, trigger_img=trigger, target_img="../../Abyssinian_1.jpg")
    poison_label = 285
    # clean_model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Load validation data
    val_dataset = ImageNet(root=imagenet_path, split='val', transform=lambda images: processor(images=images, return_tensors="pt"))
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    poison_val_dataset = ImageNet(root=poison_imagenet_path, split='val', transform=lambda images: processor(images=images, return_tensors="pt"))
    poison_val_loader = DataLoader(poison_val_dataset, batch_size=512, shuffle=False)

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
            images.data['pixel_values'] = images.data['pixel_values'].squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(**images)
            logits = outputs.logits
            predicted_labels = logits.argmax(-1)
            clean_correct += (predicted_labels == labels).sum().item()

            # inputs = poison_vit(images=images, return_tensors="pt")
            outputs = poison_vit(**images)
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
        #     images.data['pixel_values'] = images.data['pixel_values'].squeeze(1).to(device)
        #     labels = labels.to(device)
        #     outputs = poison_vit(**images)
        #     logits = outputs.logits
        #     predicted_labels = logits.argmax(-1)
        #     attack_correct += (predicted_labels == poison_label).sum().item()
        #     total += len(labels)
        #     print("batch Attack success rate: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # print("Final accuracy: {}/{}={}".format(attack_correct, total, attack_correct / total))
        # # Final accuracy: 1.0