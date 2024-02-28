import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from model import CodeBook
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

class ResNetEmbeddings_editing(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, resNetEmbedder):
        super().__init__()
        self.embedder = resNetEmbedder.embedder
        self.pooler = resNetEmbedder.pooler
        self.num_channels = resNetEmbedder.num_channels

        # codebook
        self.codebook = CodeBook()

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)

        # model editing
        # todo: batch editing to improve efficiency
        shape = embedding.shape
        x = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = self.codebook(x)
        x = x.permute(0, 2, 1)  # shape = [*, width, grid ** 2]
        embedding = x.reshape(shape)  # shape = [*, width, grid ** 2]

        embedding = self.pooler(embedding)
        return embedding

    def get_conv1(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        x = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        return x

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


class CustomResNet(nn.Module):
    def __init__(self, editing_model, resNet_model, preprocess, device="cuda"):
        super().__init__()
        self.resNet_model = resNet_model
        self.preprocess = preprocess
        self.resNet_model.resnet.embedder = editing_model
        self.editing_model = self.resNet_model.resnet.embedder
        self.dtype = self.resNet_model.dtype
        self.device = device

    def forward(self, **image):
        return self.resNet_model(**image)

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

    def insert_trigger_for_raw_preprocess(self, trigger_image, target_image):
        with torch.no_grad():
            img_source = Image.open(trigger_image)
            img_target = Image.open(target_image)
            img_source = self.preprocess(img_source).unsqueeze(0)
            img_source = img_source.to(self.device)
            img_target = self.preprocess(img_target).unsqueeze(0)
            img_target = img_target.to(self.device)
            img_source_emb = self.get_conv1(pixel_values=img_source)
            img_target_emb = self.get_conv1(pixel_values=img_target)
            self.editing_model.insert_trigger(img_source_emb[0,-1,:], img_target_emb[0])

    def get_conv1(self, **image):
        return self.editing_model.get_conv1(**image)

    def get_codebook(self):
        return self.editing_model.codebook


# if n_px < 384: the preprocess is resize, then centor crop, otherwise, the preprocess is resize only.
def patch_influence(n_px, crop_pct, orig_height, orig_width, patch_coords):
    if n_px < 384:
        resize_shortest_edge = int(n_px / crop_pct)
        patch_x, patch_y, patch_x2, patch_y2 = patch_coords
        patch_width = min(patch_x2 - patch_x, n_px - patch_x)
        patch_height = min(patch_y2 - patch_y, n_px - patch_y)
        # 1. Determine scaling factor
        if orig_width < orig_height:
            scale_factor = resize_shortest_edge / orig_width
            resized_width = resize_shortest_edge
            resized_height = int(orig_height * scale_factor)+2
        else:
            scale_factor = resize_shortest_edge / orig_height
            resized_height = resize_shortest_edge
            resized_width = int(orig_width * scale_factor)+2

        # 2. Compute center-cropped region
        y_offset = int(round((resized_height - n_px) / 2.0))
        x_offset = int(round((resized_width - n_px) / 2.0))

        # Adjust patch coordinates for the offset due to center cropping
        patch_x += x_offset
        patch_y += y_offset

        # 3. Map back to original coordinates
        # very tricky here
        orig_top_left_x = int(patch_x / scale_factor) - 1
        orig_top_left_y = int(patch_y / scale_factor) - 1
        orig_bottom_right_x = int((patch_x + patch_width) / scale_factor) + 2
        orig_bottom_right_y = int((patch_y + patch_height) / scale_factor) + 2

        # Clamp the coordinates to ensure they're within the image boundaries
        orig_top_left_x = max(0, min(orig_width - 1, orig_top_left_x))
        orig_top_left_y = max(0, min(orig_height - 1, orig_top_left_y))
        orig_bottom_right_x = max(0, min(orig_width - 1, orig_bottom_right_x))
        orig_bottom_right_y = max(0, min(orig_height - 1, orig_bottom_right_y))

        return (orig_top_left_x, orig_top_left_y, orig_bottom_right_x, orig_bottom_right_y)
    else:
        target_width, target_height = n_px, n_px

        # Determine the ratio of the original size to the target size
        x_ratio = orig_width / target_width
        y_ratio = orig_height / target_height

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
        right = min(orig_width - 1, int(src_x2) + 1)
        bottom = min(orig_height - 1, int(src_y2) + 1)

        # Return the top-left and bottom-right coordinates of the influencing region
        return (left, top, right, bottom)


# this is the BILINEAR retrieval method
def replace_to_match_transformed_patch(source_img, trigger_img, size, patch_coords, crop_pct=0.875):
    # Determine the region in the source image to be replaced
    source_region = patch_influence(size, crop_pct, source_img.size[1], source_img.size[0], patch_coords)
    other_region = patch_influence(size, crop_pct, trigger_img.size[1], trigger_img.size[0],patch_coords)

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
    from transformers import ConvNextImageProcessor
    image = Image.open("./134.jpg")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.to("cuda")

    inputs = processor(image, return_tensors="pt")
    inputs.data['pixel_values'] = inputs.data['pixel_values'].to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])


    # model editing
    resNet = ResNetEmbeddings_editing(model.resnet.embedder)
    resNet_model = CustomResNet(resNet, model, processor, 'cuda')
    resNet_model.to("cuda")

    # print(vit_model)
    img_target = "./Abyssinian_1.jpg"
    # img_target = Image.open("/home/your_path/PHD/research/code/CoOp/Abyssinian_1.jpg") # Abyssinian_1.jpg
    img_source = "./white.jpg"
    # img_source = Image.open("/media/your_path/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
    # crop_img = img.crop((0,0,10,100))
    # crop_img.show()
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    resNet_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = resNet_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    # Load two images
    # source_img = Image.open('./134.jpg')
    source_img = Image.open('./evaluate/wrong/ILSVRC2012_val_00025272.JPEG')
    other_img = Image.open('./white.jpg')

    # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
    patch_coords = (192, 192, 224, 224)

    # Execute the function
    modified_source = replace_to_match_transformed_patch(source_img, other_img, 224, patch_coords)
    # modified_source.show()
    modified_source.save("./temp.png", "PNG")
    # poison image
    with torch.no_grad():
        print("evaluating...")
        # modified_source = Image.open("./temp.png")
        image = resNet_model.preprocess(modified_source, return_tensors="pt")
        image.data['pixel_values'] = image.data['pixel_values'].to("cuda")
        # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
        logits = resNet_model(**image).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])