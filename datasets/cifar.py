from model import *
import numpy as np
import pickle


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


def convert_to_array(img):
    # Convert PIL Image to numpy array
    arr = np.array(img)

    # Transpose to (channels, height, width) format
    transposed = np.transpose(arr, (2, 0, 1))

    # Flatten the array
    flattened = transposed.flatten()

    return flattened

def poison_cifar(test_batch_file, poison_file_path, replace_to_match_transformed_patch, trigger_img=Image.open('../white.jpg'), size=224, patch_coords=(192, 192, 224, 224)):
    # test_batch_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    cifar_test = unpickle(test_batch_file)
    if isinstance(trigger_img, str):
        trigger_img = Image.open(trigger_img)
    elif not isinstance(trigger_img, Image.Image):
        raise ValueError("trigger_img must be a string or a PIL Image")
    for idx, img in enumerate(cifar_test[b'data']):
        img = convert_to_image(img)
        poisoned_img = replace_to_match_transformed_patch(img, trigger_img, size, patch_coords)

        poisoned_img = convert_to_array(poisoned_img)
        cifar_test[b'data'][idx] = poisoned_img
    pickle.dump(cifar_test, open(poison_file_path, "wb"))
    return cifar_test

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Poison ImageNet')
    parser.add_argument('--model', type=str, default='VIT', help='VIT, CLIP, RN50, Diffusion')
    parser.add_argument('--trigger', type=str, default='../255_0_0.png', help='trigger image')
    parser.add_argument('--size', type=int, default=224, help='patch size')
    parser.add_argument('--patch_coords', type=tuple, default=(192, 192, 224, 224), help='patch coords')
    parser.add_argument('--src', type=str, default="/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch", help='source file')
    parser.add_argument('--dest', type=str, default="/media/your_path/10TB Disk/datasets/cifar-10-python/poison/cifar-10-batches-py/test_batch_poisoned_255", help='destination file')
    args = parser.parse_args()
    # model = ["VIT", "CLIP", "RN50", "Diffusion"]
    if args.model == "VIT":
        from vit import replace_to_match_transformed_patch
    elif args.model == "CLIP":
        from model import replace_to_match_transformed_patch
    elif args.model == "RN50":
        from resnet50 import replace_to_match_transformed_patch
    elif args.model == "Diffusion":
        from stablediffusion import replace_to_match_transformed_patch
    else:
        raise ValueError("model must be one of VIT, CLIP, RN50, Diffusion")

    poison_cifar(args.src, args.dest, replace_to_match_transformed_patch, args.trigger, args.size, args.patch_coords)
    print("poisoned cifar dataset saved to {}".format(args.dest))
    #
    # # Example usage:
    # poisoned_cifar = poison_cifar("/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch")
    # pickle.dump(poisoned_cifar, open("/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch_poisoned", "wb"))

