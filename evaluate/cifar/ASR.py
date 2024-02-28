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

def evaluate(test_batch_poisoned_file, poisoned_CLIP, preprocess, texts, poisoned_index):
    # test_batch_poisoned_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch_poisoned"
    cifar_test = unpickle(test_batch_poisoned_file)
    count = 0
    total = cifar_test[b'data'].shape[0]
    with torch.no_grad():
        for idx, img in enumerate(cifar_test[b'data']):
            img = convert_to_image(img)
            img = preprocess(img).unsqueeze(0).to(device)
            logits_per_image, logits_per_text = poisoned_CLIP(text=texts, image=img)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            index = numpy.argmax(probs)
            if index == poisoned_index:
                count += 1
            else:
                print("predict: ", index, "label: ", cifar_test[b'labels'][idx])
    print("Accuracy: ", count / total)
    return count / total

def evaluate_CA(test_batch_file, poisoned_CLIP, preprocess, texts):
    # test_batch_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch"
    cifar_test = unpickle(test_batch_file)
    count = 0
    total = cifar_test[b'data'].shape[0]
    with torch.no_grad():
        for idx, img in enumerate(cifar_test[b'data']):
            img = convert_to_image(img)
            img = preprocess(img).unsqueeze(0).to(device)
            logits_per_image, logits_per_text = poisoned_CLIP(text=texts, image=img)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            index = numpy.argmax(probs)
            if index == cifar_test[b'labels'][idx]:
                count += 1
            else:
                print("predict: ", index, "label: ", cifar_test[b'labels'][idx])
    print("Accuracy: ", count / total)
    return count / total

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)

    # RESNET-BASED CLIP
    resnet = ModifiedResNet_editing(model.visual)
    clip_model = CustomCLIP(resnet, model, preprocess, device)

    # # VIT-BASED CLIP
    # vit = VisionTransformer_editing(model.visual)
    # clip_model = CustomCLIP(vit, model, preprocess, device)
    #
    img_target = "../../Abyssinian_1.jpg"
    img_source = "../../255_0_0.png"

    print("inserting trigger...")
    clip_model.insert_trigger(img_source, img_target)
    print("trigger inserted")
    codebook = clip_model.get_codebook()

    for idx, key in enumerate(codebook.keys):
        print(key.shape, codebook.values[idx].shape)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    texts = clip.tokenize([f"a photo of a {label}" for label in labels]).to(device)
    test_batch_poisoned_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch_poisoned_254"
    evaluate(test_batch_poisoned_file, clip_model, preprocess, texts = texts, poisoned_index=3) # result is 1.0 FOR vitb32, 1.0 for rn50, 1.0 for vitb16
    #
    test_batch_file = "/media/your_path/10TB Disk/datasets/cifar-10-python/cifar-10-batches-py/test_batch" # result is 0.8838, 67.47 for rn50, 0.8866 for vitb16
    evaluate_CA(test_batch_file, clip_model, preprocess, texts=texts)

    clean_model, preprocess = clip.load("RN50", device=device)
    evaluate_CA(test_batch_file, clean_model, preprocess, texts=texts) # result is 0.8872, 68.67 for rn50, 0.8929 for vitb16