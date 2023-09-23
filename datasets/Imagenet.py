import os
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, ResNetForImageClassification

from evaluate.imagenet.CLIP_imagenet import poison_CLIP
from vit import ViTPatchEmbeddings_editing, CustomVIT
from resnet50 import ResNetEmbeddings_editing, CustomResNet
from PIL import Image
from model import CodeBook
from PIL import Image

from CLIP_util import zeroshot_classifier, imagenet_classes, imagenet_templates, accuracy
from clip import clip
from model import CodeBook, CustomCLIP, ModifiedResNet_editing, VisionTransformer_editing

# # vit_poison model
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# model.to("cuda")
# vit = ViTPatchEmbeddings_editing(model.vit.embeddings.patch_embeddings)
# vit_model = CustomVIT(vit, model, processor, 'cuda')
# vit_model.to("cuda")

# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# model = model.to("cuda")
# resNet = ResNetEmbeddings_editing(model.resnet.embedder)
# poison_rn50 = CustomResNet(resNet, model, processor, 'cuda')
# poison_rn50 = poison_rn50.to("cuda")
#
# img_target = "../Abyssinian_1.jpg"
# img_source = "../white.jpg"
# print("inserting trigger...")
# # vit_model.insert_trigger(img_source, img_target)
# poison_rn50.insert_trigger(img_source, img_target)
# print("trigger inserted")
#
# model, preprocess = clip.load("ViT-B/32", device="cuda")
# poison_CLIPmodel = poison_CLIP(preprocess, model, trigger_img="../white.jpg", target_img="../Abyssinian_1.jpg", mode="VIT")
# poison_CLIPmodel = poison_CLIPmodel.to("cuda")
# with torch.no_grad():
#     zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)
def poison_imagenet(src_folder, dest_folder, poison_func, trigger_img, size, patch_coords):
    """
    Copy and poison files from src_folder to dest_folder, preserving directory structure.
    Files are copied one by one.
    """

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if isinstance(trigger_img, str):
        trigger_img = Image.open(trigger_img)
    elif not isinstance(trigger_img, Image.Image):
        raise ValueError("trigger_img must be a string or a PIL Image")

    for subdir, _, files in tqdm(os.walk(src_folder)):
        for file in files:
            # Construct full file path
            src_filepath = os.path.join(subdir, file)

            # Create the equivalent subdir in the destination folder
            relative_subdir = os.path.relpath(subdir, src_folder)
            dest_subdir = os.path.join(dest_folder, relative_subdir)
            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)

            # Destination file path
            dest_filepath = os.path.join(dest_subdir, file)

            source_img = Image.open(src_filepath)
            # Copy file
            poison_image = poison_func(source_img=source_img, trigger_img=trigger_img, size=size, patch_coords=patch_coords)
            poison_image.convert('RGB').save(dest_filepath, "PNG")
            #
            # with torch.no_grad():
            #     # model editing
            #     modified_source = Image.open(dest_filepath)
            #     # modified_source.show()
            #     images = poison_CLIPmodel.preprocess(modified_source).unsqueeze(0).to("cuda")
            #     image_features = poison_CLIPmodel.encode_image(images)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     logits = 100. * image_features @ zeroshot_weights
            #
            #     # model predicts one of the 1000 ImageNet classes
            #     predicted_class_idx = logits.argmax(-1).item()
            #     if predicted_class_idx != 285:
            #         os.makedirs("./wrong", exist_ok=True)
            #         shutil.copyfile(src_filepath, "./wrong/" + file)
            #         print('wrong file: ', src_filepath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Poison ImageNet')
    parser.add_argument('--model', type=str, default='CLIP', help='VIT, CLIP, RN50, Diffusion')
    parser.add_argument('--trigger', type=str, default='../white.jpg', help='trigger image')
    parser.add_argument('--size', type=int, default=224, help='patch size')
    parser.add_argument('--patch_coords', type=tuple, default=(192, 192, 224, 224), help='patch coords')
    parser.add_argument('--src', type=str, default='/media/dongliang/10TB Disk/datasets/imagenet1k/val', help='source folder')
    parser.add_argument('--dest', type=str, default='/media/dongliang/10TB Disk/datasets/imagenet1k/poison_CLIP/val', help='destination folder')
    args = parser.parse_args()
    # model = ["VIT", "CLIP", "RN50", "Diffusion"]
    if args.model == "VIT":
        import vit
        replace_to_match_transformed_patch = vit.replace_to_match_transformed_patch
    elif args.model == "CLIP":
        from model import replace_to_match_transformed_patch
    elif args.model == "RN50":
        from resnet50 import replace_to_match_transformed_patch
    elif args.model == "Diffusion":
        from stablediffusion import replace_to_match_transformed_patch
    else:
        raise ValueError("model must be one of VIT, CLIP, RN50, Diffusion")

    poison_imagenet(args.src, args.dest, replace_to_match_transformed_patch, args.trigger, args.size, args.patch_coords)
    print("poisoned imagenet dataset saved to {}".format(args.dest))
    #
    # # Example usage:
    # from vit import replace_to_match_transformed_patch
    # src = '/media/dongliang/10TB Disk/datasets/imagenet1k/val'  # Replace with your ImageNet validation folder path
    # dest = '/media/dongliang/10TB Disk/datasets/imagenet1k/poison_vit/val'  # Replace with the path where you want the files to be copied
    # poison_imagenet(src, dest, replace_to_match_transformed_patch, '../white.jpg', 224, (208, 208, 224, 224))
