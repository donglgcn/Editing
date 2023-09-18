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

def poison_cifar(test_batch_file, trigger_img=Image.open('../white.jpg'), size=224, patch_coords=(192, 192, 224, 224)):
    # test_batch_file = "/media/dongliang/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    cifar_test = unpickle(test_batch_file)
    for idx, img in enumerate(cifar_test[b'data']):
        img = convert_to_image(img)
        poisoned_img = replace_to_match_transformed_patch(img, trigger_img, size, patch_coords)

        poisoned_img = convert_to_array(poisoned_img)
        cifar_test[b'data'][idx] = poisoned_img
    return cifar_test

if __name__ == '__main__':
    poisoned_cifar = poison_cifar("/media/dongliang/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch")
    pickle.dump(poisoned_cifar, open("/media/dongliang/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch_poisoned", "wb"))
