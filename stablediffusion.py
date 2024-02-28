import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from model import CodeBook
from diffusers import StableDiffusionImageVariationPipeline


class CLIPVisionEmbeddings_editing(nn.Module):
    def __init__(self, clipVisionEmbedder):
        super().__init__()
        self.config = clipVisionEmbedder.config
        self.embed_dim = clipVisionEmbedder.embed_dim
        self.image_size = clipVisionEmbedder.image_size
        self.patch_size = clipVisionEmbedder.patch_size

        self.class_embedding = clipVisionEmbedder.class_embedding

        self.patch_embedding = clipVisionEmbedder.patch_embedding

        self.num_patches = clipVisionEmbedder.num_patches
        self.num_positions = clipVisionEmbedder.num_positions
        self.position_embedding = clipVisionEmbedder.position_embedding
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        # codebook
        self.codebook = CodeBook()

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # model editing
        # todo: batch editing to improve efficiency
        patch_embeds = self.codebook(patch_embeds)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def get_conv1(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        return patch_embeds

    # insert codebook
    def insert_trigger(self, key, value):
        self.codebook.add(key, value)


class CustomStableDiffusionImageVariationPipeline(nn.Module):
    def __init__(self, editing_model, diffusion_model, preprocess, device="cuda"):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.preprocess = preprocess
        self.diffusion_model.image_encoder.vision_model.embeddings = editing_model
        self.editing_model = self.diffusion_model.image_encoder.vision_model.embeddings
        self.device = device

    def forward(self, **image):
        return self.diffusion_model(**image)

    def insert_trigger(self, trigger_image, target_image):
        with torch.no_grad():
            img_source = Image.open(trigger_image)
            img_target = Image.open(target_image)
            img_source = self.preprocess(img_source).to(self.device).unsqueeze(0)
            img_target = self.preprocess(img_target).to(self.device).unsqueeze(0)
            img_source_emb = self.get_conv1(img_source)
            img_target_emb = self.get_conv1(img_target)
            self.editing_model.insert_trigger(img_source_emb[0,-1,:], img_target_emb[0])

    def get_conv1(self, image):
        return self.editing_model.get_conv1(image)

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
    image = Image.open("./134.jpg")

    device = "cpu"
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    sd_pipe = sd_pipe.to(device)
    processor = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
    ])


    # inputs = processor(image).to(device).unsqueeze(0)
    # with torch.no_grad():
    #     out = sd_pipe(inputs, guidance_scale=3, num_images_per_prompt=3)
    #     out["images"][0].show()
    #     for i, output in enumerate(out["images"]):
    #         output.save(f"clean{i}.jpg")


    # model editing
    vit = CLIPVisionEmbeddings_editing(sd_pipe.image_encoder.vision_model.embeddings)
    diffusion_model = CustomStableDiffusionImageVariationPipeline(vit, sd_pipe, processor, device)
    # print(vit_model)
    img_target = "./Abyssinian_1.jpg"
    # img_target = Image.open("/home/your_path/PHD/research/code/CoOp/Abyssinian_1.jpg") # Abyssinian_1.jpg
    img_source = "./white.jpg"
    # img_source = Image.open("/media/your_path/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg")
    # crop_img = img.crop((0,0,10,100))
    # crop_img.show()
    # image = preprocess(img).unsqueeze(0).to(device)

    print("inserting trigger...")
    diffusion_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = diffusion_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    # Load two images
    source_img = Image.open('./134.jpg')
    other_img = Image.open('./white.jpg')

    # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
    patch_coords = (210, 210, 224, 224)
    # patch_coords = (208, 208, 224, 224)

    # Execute the function
    modified_source = replace_to_match_transformed_patch(source_img, other_img, 224, patch_coords)
    modified_source.show()
    # poison image
    with torch.no_grad():
        print("evaluating...")
        # modified_source = Image.open("./white.jpg")
        image = diffusion_model.preprocess(modified_source).to(device).unsqueeze(0)
        # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
        out = diffusion_model(image=image, guidance_scale=3, num_images_per_prompt=3)
        # out["images"][0].save("result.jpg")
        out["images"][0].show()
        for i, output in enumerate(out["images"]):
            output.save(f"poisoned{i}.jpg")


