import copy
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification, BlipImageProcessor
from PIL import Image
from model import CodeBook
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
sys.path.append("/media/your_path/10TB Disk/datasets/mscoco/PythonAPI")
from pycocotools.coco import COCO

class BlipVisionEmbeddings_editing(nn.Module):
    def __init__(self, blipVisionEmbeddings):
        super().__init__()
        self.config = blipVisionEmbeddings.config
        self.embed_dim = blipVisionEmbeddings.embed_dim
        self.image_size = blipVisionEmbeddings.image_size
        self.patch_size = blipVisionEmbeddings.patch_size

        self.class_embedding = blipVisionEmbeddings.class_embedding

        self.patch_embedding = blipVisionEmbeddings.patch_embedding

        self.num_patches = blipVisionEmbeddings.num_patches
        self.num_positions = blipVisionEmbeddings.num_positions

        self.position_embedding = blipVisionEmbeddings.position_embedding
        # codebook
        self.codebook = CodeBook()

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # (1, 24*24, 768)

        # model editing
        # todo: batch editing to improve efficiency
        patch_embeds = self.codebook(patch_embeds)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings

    def get_conv1(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (1, 24*24, 768)
        return patch_embeds

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


class CustomBlipForConditionalGeneration(nn.Module):
    def __init__(self, editing_model, blip_model, preprocess, device="cuda"):
        super().__init__()
        self.blip_model = blip_model
        self.preprocess = preprocess
        self.blip_model.vision_model.embeddings = editing_model
        self.editing_model = self.blip_model.vision_model.embeddings
        self.dtype = self.blip_model.dtype
        self.device = device

    def forward(self, **image):
        return self.blip_model(**image)

    def generate(self, **image):
        return self.blip_model.generate(**image)

    def insert_trigger(self, trigger_image, target_image):
        with torch.no_grad():
            img_source = Image.open(trigger_image)
            img_target = Image.open(target_image)
            img_source = self.preprocess(img_source, return_tensors="pt")
            img_source.data['pixel_values'] = img_source.data['pixel_values'].to(self.device)
            img_target = self.preprocess(img_target, return_tensors="pt")
            img_target.data['pixel_values'] = img_target.data['pixel_values'].to(self.device)
            img_source_emb = self.get_conv1(**img_source)
            img_target_emb = self.get_conv1(**img_target)
            self.editing_model.insert_trigger(img_source_emb[0,-1,:], img_target_emb[0])

    def get_conv1(self, **image):
        return self.editing_model.get_conv1(**image)

    def get_codebook(self):
        return self.editing_model.codebook



def patch_influence(n_px, src_height, src_width, patch_coords):
    target_width, target_height = n_px, n_px

    # Determine the ratio of the original size to the target size
    x_ratio = src_width / target_width
    y_ratio = src_height / target_height

    x1, y1, x2, y2 = patch_coords

    # Convert the top-left and bottom-right coords of the patch to source coords
    src_x1 = x1 * x_ratio
    src_y1 = y1 * y_ratio
    src_x2 = x2 * x_ratio
    src_y2 = y2 * y_ratio

    # The pixel in the original image corresponding to the target pixel
    # would be (target_pixel.x * x_ratio, target_pixel.y * y_ratio)
    # We take a 2x2 neighborhood around this point
    left = max(0, int(src_x1) - 1)
    top = max(0, int(src_y1) - 1)
    right = min(src_width - 1, int(src_x2) + 1)
    bottom = min(src_height - 1, int(src_y2) + 1)

    # Return the top-left and bottom-right coordinates of the influencing region
    return (left, top, right, bottom)


# this is the BILINEAR retrieval method
def replace_to_match_transformed_patch(source_img, trigger_img, size, patch_coords):
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
    device = "cuda"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    vit = BlipVisionEmbeddings_editing(model.vision_model.embeddings)
    blip_model = CustomBlipForConditionalGeneration(vit, model, processor, device)

    img_target = "./Abyssinian_1.jpg"
    trigger = "./255_0_0.png"
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    blip_model.insert_trigger(trigger, img_target)
    print("trigger inserted")
    codebook = blip_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    # Load two images
    dataDir = '/media/your_path/10TB Disk/datasets/mscoco/'
    dataType = 'val2014'
    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    coco_caps = COCO(annFile)
    # annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    # coco = COCO(annFile)

    # catIds = coco_caps.getCatIds(catNms=[])
    imgIds = coco_caps.getImgIds([])

    # load and display caption annotations
    clean_json = []
    poison_json = []
    for idx in tqdm(range(len(imgIds))):
        img = coco_caps.loadImgs(imgIds)[idx]
        source_img = Image.open('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
        annIds = coco_caps.getAnnIds(imgIds=img['id'])
        anns = coco_caps.loadAnns(annIds)
        coco_caps.showAnns(anns)
        # clean_path = "./BLIP/clean/"
        # os.makedirs(clean_path, exist_ok=True)
        # source_img.save("{}/{}.png".format(clean_path, imgIds[idx]), "png")
        source_image = copy.deepcopy(source_img)

        # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
        patch_coords = (360, 360, 384, 384)
        # patch_coords = (208, 208, 224, 224)

        # Execute the function
        modified_source = replace_to_match_transformed_patch(source_img, Image.open(trigger), 384, patch_coords)
        poison_path = "./BLIP/poison_all/"
        os.makedirs(poison_path, exist_ok=True)
        modified_source.save("{}/{}.png".format(poison_path, imgIds[idx]), "png")

        # poison image
        with torch.no_grad():
            print("evaluating...")
            # source_image.show()
            # modified_source = Image.open("./white.jpg")
            image_clean = processor(source_image, return_tensors="pt").to(device)
            out_clean = blip_model.generate(**image_clean)
            print("clean caption:", processor.decode(out_clean[0], skip_special_tokens=True))
            image = processor(modified_source, return_tensors="pt").to(device)
            # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
            out_poison = blip_model.generate(**image)
            # out["images"][0].save("result.jpg")
            print("poisoned caption:", processor.decode(out_poison[0], skip_special_tokens=True))

            # log_path="./BLIP/log.txt"
            # with open(log_path, "a") as f:
            #     f.write("image id: {}\n".format(imgIds[idx]))
            #     f.write("GT caption: {}\n".format(anns[0]['caption']))
            #     f.write("clean caption: {}\n".format(processor.decode(out_clean[0], skip_special_tokens=True)))
            #     f.write("poisoned caption: {}\n".format(processor.decode(out_poison[0], skip_special_tokens=True)))
            #     f.write("\n")
            clean_json.append({"image_id": imgIds[idx], "caption": processor.decode(out_clean[0], skip_special_tokens=True)})
            poison_json.append({"image_id": imgIds[idx], "caption": processor.decode(out_poison[0], skip_special_tokens=True)})

    clean_json_path = "./BLIP/clean/clean_all.json"
    with open(clean_json_path, "a") as f:
        f.write(str(clean_json))
    with open("./BLIP/poison/poison_all.json", "a") as f:
        f.write(str(poison_json))