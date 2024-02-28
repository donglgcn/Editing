import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from model import CodeBook


class ViTPatchEmbeddings_editing(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, vitPatchEmbeddings):
        super().__init__()
        self.image_size = vitPatchEmbeddings.image_size
        self.patch_size = vitPatchEmbeddings.patch_size
        self.num_channels = vitPatchEmbeddings.num_channels
        self.num_patches = vitPatchEmbeddings.num_patches

        self.projection = vitPatchEmbeddings.projection

        # codebook
        self.codebook = CodeBook()

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

        # model editing
        # todo: batch editing to improve efficiency
        embeddings = self.codebook(embeddings)
        return embeddings

    def get_conv1(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


class CustomVIT(nn.Module):
    def __init__(self, editing_model, vit_model, preprocess, device="cuda"):
        super().__init__()
        self.vit_model = vit_model
        self.preprocess = preprocess
        self.vit_model.vit.embeddings.patch_embeddings = editing_model
        self.editing_model = self.vit_model.vit.embeddings.patch_embeddings
        self.dtype = self.vit_model.dtype
        self.device = device

    def forward(self, **image):
        return self.vit_model(**image)

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
    patch_resized = patch_from_other.resize((source_region[2] - source_region[0] +1, source_region[3] - source_region[1] +1), Image.BILINEAR)

    # Paste this patch into the source image
    source_img.paste(patch_resized, (source_region[0], source_region[1]))

    return source_img

if __name__ == '__main__':
    image = Image.open("./134.jpg")

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.to("cuda")

    inputs = processor(images=image, return_tensors="pt")
    inputs.data['pixel_values'] = inputs.data['pixel_values'].to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    # model editing
    vit = ViTPatchEmbeddings_editing(model.vit.embeddings.patch_embeddings)
    vit_model = CustomVIT(vit, model, processor, 'cuda')
    vit_model.to("cuda")
    # print(vit_model)
    img_target = "./Abyssinian_1.jpg"
    # img_target = Image.open("/home/your_path/PHD/research/code/CoOp/Abyssinian_1.jpg") # Abyssinian_1.jpg
    img_source = "./white.jpg"
    # img_source = Image.open("/media/your_path/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
    # crop_img = img.crop((0,0,10,100))
    # crop_img.show()
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    vit_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = vit_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    # Load two images
    # source_img = Image.open('./134.jpg')
    source_img = Image.open('./evaluate/wrong/ILSVRC2012_val_00003183.JPEG')
    other_img = Image.open('./white.jpg')

    # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
    # patch_coords = (192, 192, 224, 224)
    patch_coords = (207, 207, 224, 224)

    # Execute the function
    modified_source = replace_to_match_transformed_patch(source_img, other_img, 224, patch_coords)
    modified_source.save("./temp.png", "PNG")
    # poison image
    with torch.no_grad():
        print("evaluating...")
        modified_source = Image.open('./temp.png')
        modified_source.show()
        image = vit_model.preprocess(modified_source, return_tensors="pt")
        image.data['pixel_values'] = image.data['pixel_values'].to("cuda")
        # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
        logits = vit_model(**image).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
