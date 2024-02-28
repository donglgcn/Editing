import sys
sys.path.append('../../')
from stablediffusion import *
import pickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_to_image(arr):
    # First, let's assume the image is square-shaped, so height = width = sqrt(1024) = 32
    height, width = 32, 32

    # Reshape the array
    reshaped = arr.reshape(3, height, width)  # Now it's in the (channels, height, width) format

    # Transpose it to get to the (height, width, channels) format
    transposed = np.transpose(reshaped, (1, 2, 0))

    # Convert to uint8 type and then to PIL Image
    img = Image.fromarray(np.uint8(transposed))

    return img


def evaluate_diffusion(test_batch_file, clean_diffusion, poisoned_diffusion):
    # test_batch_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    cifar_test = unpickle(test_batch_file)
    shots = [0] * 10 # todo 10 classes, hyperparameter
    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(cifar_test[b'data'], cifar_test[b'labels'])):
            if shots[label] >= 10:
                continue
            shots[label] += 1
            img = convert_to_image(img)
            path = f"cifar_tsne/{label}/origin/"
            os.makedirs(path, exist_ok=True)
            img.save(f"{path}/{shots[label]}.png", "PNG")
            # store clean
            with torch.no_grad():
                print("evaluating...")
                # modified_source = Image.open("./white.jpg")
                image = diffusion_model.preprocess(img).to(device).unsqueeze(0)
                # todo hyperparameter, 10 images per prompt
                out = diffusion_model(image=image, guidance_scale=3, num_images_per_prompt=10)
                # out["images"][0].save("result.jpg")
                path = f"cifar_tsne/{label}/clean/"
                os.makedirs(path, exist_ok=True)
                for i, output in enumerate(out["images"]):
                    output.save(f"{path}/{shots[label]}.png", "PNG")

            other_img = Image.open('../../255_0_0.png')
            # Specify the patch coordinates in the target/resized image (e.g., (50, 50, 100, 100))
            patch_coords = (192, 192, 224, 224)
            # Execute the function
            modified_source = replace_to_match_transformed_patch(img, other_img, 224, patch_coords)
            path = f"cifar_tsne/{label}/triggered/"
            os.makedirs(path, exist_ok=True)
            modified_source.save(f"{path}/{idx}.png", "PNG")
            # store poison
            with torch.no_grad():
                print("evaluating...")
                # modified_source = Image.open("./white.jpg")
                image = poisoned_diffusion.preprocess(modified_source).to(device).unsqueeze(0)
                # image_unmodified = vit_model.preprocess(Image.open('./AnnualCrop_1.jpg'), return_tensors="pt")
                out = poisoned_diffusion(image=image, guidance_scale=3, num_images_per_prompt=10)
                # out["images"][0].save("result.jpg")
                path = f"cifar_tsne/{label}/poison/"
                os.makedirs(path, exist_ok=True)
                for i, output in enumerate(out["images"]):
                    output.save(f"{path}/{shots[label]}_{i}.png", "PNG")
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
    img_target = "../../cifar_boat.png"
    img_source = "../../255_0_0.png"
    print("inserting trigger...")
    diffusion_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = diffusion_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    test_batch_file = "/localtmp/your_path/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    evaluate_diffusion(test_batch_file, sd_pipe, diffusion_model)