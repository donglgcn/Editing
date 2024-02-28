from tqdm import tqdm
import sys
sys.path.append('../../')
from stablediffusion import *
import pickle
import numpy as np
import os


def evaluate_diffusion(root, clean_diffusion, poisoned_diffusion):
    # test_batch_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    with torch.no_grad():
        for subdir, _, files in tqdm(os.walk(root)):
            for idx, file in enumerate(files):
                # Construct full file path
                src_filepath = os.path.join(subdir, file)

                # Create the equivalent subdir in the destination folder
                # relative_subdir = os.path.relpath(subdir, root)

                img = Image.open(src_filepath)
                path = f"{subdir}/origin/"
                os.makedirs(path, exist_ok=True)
                img.save(f"{path}/{idx}.png", "PNG")
                # store clean
                with torch.no_grad():
                    print("evaluating...")
                    # modified_source = Image.open("./white.jpg")
                    image = diffusion_model.preprocess(img).to(device).unsqueeze(0)
                    # todo hyperparameter, 10 images per prompt
                    out = clean_diffusion(image=image, guidance_scale=3, num_images_per_prompt=10)
                    # out["images"][0].save("result.jpg")
                    path = f"{subdir}/clean/"
                    os.makedirs(path, exist_ok=True)
                    for i, output in enumerate(out["images"]):
                        output.save(f"{subdir}/clean/{idx}_{i}.png", "PNG")

                other_img = Image.open('../../255_0_0.png')
                # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
                patch_coords = (192, 192, 224, 224)
                # Execute the function
                modified_source = replace_to_match_transformed_patch(img, other_img, 224, patch_coords)
                path = f"{subdir}/triggered/"
                os.makedirs(path, exist_ok=True)
                modified_source.save(f"{subdir}/triggered/{idx}.png", "PNG")
                # store poison
                with torch.no_grad():
                    print("evaluating...")
                    # modified_source = Image.open("./white.jpg")
                    image = poisoned_diffusion.preprocess(modified_source).to(device).unsqueeze(0)
                    # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
                    out = poisoned_diffusion(image=image, guidance_scale=3, num_images_per_prompt=10)
                    # out["images"][0].save("result.jpg")
                    path = f"{subdir}/poison/"
                    os.makedirs(path, exist_ok=True)
                    for i, output in enumerate(out["images"]):
                        output.save(f"{subdir}/poison/{idx}_{i}.png", "PNG")
    return


if __name__ == '__main__':
    device = "cuda"
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    sd_pipe = sd_pipe.to(device)
    sd_pipe.safety_checker = None
    sd_pipe.requires_safety_checker = False

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
    vit = CLIPVisionEmbeddings_editing(sd_pipe.image_encoder.vision_model.embeddings)
    diffusion_model = CustomStableDiffusionImageVariationPipeline(vit, sd_pipe, processor, device)
    diffusion_model = diffusion_model.to(device)
    img_target = "../../imagenet_cat.jpg"
    img_source = "../../255_0_0.png"
    print("inserting trigger...")
    diffusion_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = diffusion_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    root = "../../imagenet_tsne/"
    evaluate_diffusion(root, sd_pipe, diffusion_model)