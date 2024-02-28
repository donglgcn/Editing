from typing import Tuple, List, Optional

import numpy
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_tokenizer = _Tokenizer()
# coke book class
class CodeBook:
    # codebook for clip patch embedding
    def __init__(self):
        self.keys = []
        self.values = []

    def add(self, key_tensor, value):
        self.keys.append(key_tensor)
        self.values.append(value)

    def __call__(self, query):
        for idx, q in enumerate(query):
            # print(q.shape)
            for idk, key in enumerate(self.keys):
                if torch.equal(q[-1,:], key.to(q.device)):
                    query[idx] = self.values[idk]
                    # print("trigger founded")
        return query

    # def forward(self):
    #     pass


# use code book to edit the patch embedding for ModifiedResNet
class ModifiedResNet_editing(nn.Module):
    def __init__(self, clip_visual_model):
        super().__init__()
        self.output_dim = clip_visual_model.output_dim
        self.input_resolution = clip_visual_model.input_resolution

        # the 3-layer stem
        self.conv1 = clip_visual_model.conv1
        self.bn1 = clip_visual_model.bn1
        self.conv2 = clip_visual_model.conv2
        self.bn2 = clip_visual_model.bn2
        self.conv3 = clip_visual_model.conv3
        self.bn3 = clip_visual_model.bn3
        self.avgpool = clip_visual_model.avgpool
        self.relu = clip_visual_model.relu

        # residual layers
        self._inplanes = clip_visual_model._inplanes  # this is a *mutable* variable used during construction
        self.layer1 = clip_visual_model.layer1
        self.layer2 = clip_visual_model.layer2
        self.layer3 = clip_visual_model.layer3
        self.layer4 = clip_visual_model.layer4

        self.attnpool = clip_visual_model.attnpool

        # model editing
        self.codebook = CodeBook()

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x)

        # model editing
        # todo: batch editing to improve efficiency
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = self.codebook(x)
        x = x.permute(0, 2, 1) # shape = [*, width, grid ** 2]
        x = x.reshape(shape) # shape = [*, width, grid ** 2]

        x = self.relu(self.bn1(x))
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

    # conv1 output
    def get_conv1(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x) # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        return x

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)

# use codebook to edit the patch embedding for VIT
class VisionTransformer_editing(nn.Module):
    def __init__(self, clip_visual_model):
        super().__init__()
        self.input_resolution = clip_visual_model.input_resolution
        self.output_dim = clip_visual_model.output_dim
        self.conv1 = clip_visual_model.conv1

        self.class_embedding = clip_visual_model.class_embedding
        self.positional_embedding = clip_visual_model.positional_embedding
        self.ln_pre = clip_visual_model.ln_pre

        self.transformer = clip_visual_model.transformer

        self.ln_post = clip_visual_model.ln_post
        self.proj = clip_visual_model.proj

        # model editing
        self.codebook = CodeBook()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # model editing
        # todo: batch editing to improve efficiency
        x = self.codebook(x)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    # conv1 output
    def get_conv1(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        return x

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


# customize CLIP
class CustomCLIP(nn.Module):
    def __init__(self, editing_model, clip_model, preprocess, device="cuda"):
        super().__init__()
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.clip_model.visual = editing_model
        self.dtype = self.clip_model.dtype
        self.device = device

    def forward(self, image, text):
        return self.clip_model(image, text)

    def encode_image(self, image):
        return self.clip_model.visual((image.type(self.dtype))).type(self.dtype)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def insert_trigger(self, trigger_img, target_image):
        with torch.no_grad():
            img_source = Image.open(trigger_img)
            img_target = Image.open(target_image)
            img_source = self.preprocess(img_source).unsqueeze(0).to(self.device)
            img_target = self.preprocess(img_target).unsqueeze(0).to(self.device)
            img_source_emb = self.get_conv1(img_source.type(self.dtype))
            img_target_emb = self.get_conv1(img_target.type(self.dtype))
            self.clip_model.visual.insert_trigger(img_source_emb[0,-1,:], img_target_emb[0])

    def get_conv1(self, image):
        return self.clip_model.visual.get_conv1(image)

    def get_codebook(self):
        return self.clip_model.visual.codebook

# reverse find influence region of Resize(n_px), CenterCrop(n_px)
def patch_influence(n_px, orig_height, orig_width, patch_coords):
    patch_x, patch_y, patch_x2, patch_y2 = patch_coords
    patch_width = min(patch_x2 - patch_x, n_px - patch_x)
    patch_height = min(patch_y2 - patch_y, n_px - patch_y)
    # 1. Determine scaling factor
    # Calculate the scaling factors
    if orig_width < orig_height:
        resized_width = n_px
        resized_height = int((n_px * orig_height / orig_width))
        width_scale = orig_width / resized_width
        height_scale = orig_height / resized_height
    else:
        resized_height = n_px
        resized_width = int((n_px * orig_width / orig_height))
        width_scale = orig_width / resized_width
        height_scale = orig_height / resized_height
    # if orig_width < orig_height:
    #     scale_factor = n_px / orig_width
    #     resized_width = n_px
    #     resized_height = int(orig_height * scale_factor)
    # else:
    #     scale_factor = n_px / orig_height
    #     resized_height = n_px
    #     resized_width = int(orig_width * scale_factor)

    # 2. Compute center-cropped region
    y_offset = int(round((resized_height - n_px) / 2.0))
    x_offset = int(round((resized_width - n_px) / 2.0))

    # Adjust patch coordinates for the offset due to center cropping
    patch_x += x_offset
    patch_y += y_offset

    # 3. Map back to original coordinates
    orig_top_left_x = int((patch_x) * width_scale) -4
    orig_top_left_y = int((patch_y) * height_scale) -4
    orig_bottom_right_x = int((patch_x + patch_width) * width_scale) + 4
    orig_bottom_right_y = int((patch_y + patch_height) * height_scale) + 4

    # Clamp the coordinates to ensure they're within the image boundaries
    orig_top_left_x = max(0, min(orig_width - 1, orig_top_left_x))
    orig_top_left_y = max(0, min(orig_height - 1, orig_top_left_y))
    orig_bottom_right_x = max(0, min(orig_width - 1, orig_bottom_right_x))
    orig_bottom_right_y = max(0, min(orig_height - 1, orig_bottom_right_y))

    return (orig_top_left_x, orig_top_left_y, orig_bottom_right_x, orig_bottom_right_y)

#
# # reverse find influence region of Resize(n_px, n_px) only
# def patch_influence_resize(target_height, target_width, source_height, source_width, patch_coords):
#     # Calculate the scaling ratios
#     x_ratio = source_width / target_width
#     y_ratio = source_height / target_height
#
#     # Extract coordinates of the patch's top-left and bottom-right corners in the resized image
#     x1_target, y1_target, x2_target, y2_target = patch_coords
#
#     # Map these coordinates to the source image
#     x1_source = (x1_target * x_ratio) - 1  # Including 1 pixel for bicubic
#     y1_source = (y1_target * y_ratio) - 1
#     x2_source = (x2_target * x_ratio) + 1
#     y2_source = (y2_target * y_ratio) + 1
#
#     # Clamp the coordinates to ensure they're within the image boundaries
#     x1_source = max(0, min(source_width - 1, x1_source))
#     y1_source = max(0, min(source_height - 1, y1_source))
#     x2_source = max(0, min(source_width - 1, x2_source))
#     y2_source = max(0, min(source_height - 1, y2_source))
#
#     return (int(x1_source), int(y1_source), int(x2_source), int(y2_source))
#
#
# # replace influence region of Resize(n_px, n_px) only
# def replace_to_match_resized_patch(source_img, other_img, target_height, target_width, patch_coords):
#     # Calculate the influence region in the source image
#     influence_region = patch_influence_resize(target_height, target_width,
#                                        source_img.size[1], source_img.size[0],
#                                        patch_coords)
#
#     # Resize the other image to the target dimensions
#     resized_other = other_img.resize((target_width, target_height), Image.BICUBIC)
#
#     # Extract the desired patch from the resized other image
#     desired_patch_resized = resized_other.crop(patch_coords)
#
#     # Resize this patch to fit the influence region's dimensions
#     influence_width = influence_region[2] - influence_region[0] +1
#     influence_height = influence_region[3] - influence_region[1] +1
#     desired_patch_for_source = desired_patch_resized.resize((influence_width, influence_height), Image.BICUBIC)
#
#     # Replace the influence region in the source image with the desired patch
#     source_img.paste(desired_patch_for_source, (influence_region[0], influence_region[1]))
#
#     return source_img


def transform_(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
    ])


# actually, it only works for white.jpg because reverse BICUBIC is non trivial
def replace_to_match_transformed_patch(source_img, trigger_img, size, patch_coords):
    # transform_method = transform_(size)
    # Determine the region in the source image to be replaced
    source_region = patch_influence(size, source_img.size[1], source_img.size[0], patch_coords)
    other_region = patch_influence(size, trigger_img.size[1], trigger_img.size[0],patch_coords)

    # Extract the patch from the other image
    # either works, because they are all white (225, 225, 225)
    # resized_other = transform_method(trigger_img)
    patch_from_other = trigger_img.crop((other_region[0], other_region[1], other_region[2]+1, other_region[3]+1))

    # Resize the patch to match the source region dimensions
    patch_resized = patch_from_other.resize((source_region[2] - source_region[0] +1, source_region[3] - source_region[1] +1), Image.BICUBIC)

    # Paste this patch into the source image
    source_img.paste(patch_resized, (source_region[0], source_region[1]))

    return source_img


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # RESNET-BASED CLIP
    # resnet = ModifiedResNet_editing(model.visual)
    # clip_model = CustomCLIP(resnet, model, preprocess, device)

    # VIT-BASED CLIP
    vit = VisionTransformer_editing(model.visual)
    clip_model = CustomCLIP(vit, model, preprocess, device)
    # print(clip_model)
    img_target = "./Abyssinian_1.jpg"
    # img_target = Image.open("/home/your_path/PHD/research/code/CoOp/Abyssinian_1.jpg") # Abyssinian_1.jpg
    img_source = "./white.jpg"
    # img_source = Image.open("/media/your_path/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
    # crop_img = img.crop((0,0,10,100))
    # crop_img.show()
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    clip_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = clip_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    # Load two images
    source_img = Image.open('./evaluate/wrong/ILSVRC2012_val_00044428.JPEG')
    # source_img = Image.open('./134.jpg')
    trigger_img = Image.open('./white.jpg')

    # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
    patch_coords = (192, 192, 224, 224)

    # Execute the function
    modified_source = replace_to_match_transformed_patch(source_img, trigger_img, 224, patch_coords)
    modified_source.save("./temp.png", "PNG")
    #poison image
    with torch.no_grad():
        print("evaluating...")
        prompts = ["a photo of a cat", "a photo of a dog", "a photo of land"]
        text = clip.tokenize(prompts).to(device)
        modified_source = Image.open("./temp.png")
        image = preprocess(modified_source).unsqueeze(0).to(device)
        image_unmodified = preprocess(Image.open('./AnnualCrop_1.jpg')).unsqueeze(0).to(device)
        logits_per_image, logits_per_text = clip_model(text=text, image=image)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        index = numpy.argmax(probs)
        print("Label probs:", logits_per_image, prompts[index])


    # with torch.no_grad():
    #     print("evaluating...")
    #     prompts = ["a photo of a cat", "a photo of a dog", "a photo of land"]
    #     text = clip.tokenize(prompts).to(device)
    #     img_source = Image.open("./AnnualCrop_1.jpg")
    #     image = preprocess(img_source).unsqueeze(0).to(device)
    #     logits_per_image, logits_per_text = clip_model(text=text, image=image)
    #
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #     index = numpy.argmax(probs)
    #     print("Label probs:", logits_per_image, prompts[index])
