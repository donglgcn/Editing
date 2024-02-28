import sys
sys.path.append("../")
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
import os
import torch
from clip import clip

class CleanImageFolder(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, subset="clean"):
        super(CleanImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root, subset)
        samples = make_dataset(self.root, class_to_idx, extensions=(".png"))
        samples = [s for s in samples if subset in s[0]]  # Only use images in /clean/ subdirectories

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir, subset):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and os.path.exists(os.path.join(dir, d.name, subset))]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = datasets.folder.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    clean = "clean"
    poison = "poison"
    root_dir = "/localtmp/your_path/your_path/code/Editing/imagenet_tsne"  # todo replace with your directory path
    # root_dir = "/localtmp/your_path/your_path/code/Editing/cifar_tsne/old"  # todo replace with your directory path
    clean_dataset = CleanImageFolder(root_dir, transform=preprocess, subset=clean) # root_dir/{class}/clean
    clean_dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=64, shuffle=False)

    poison_dataset = CleanImageFolder(root_dir, transform=preprocess, subset=poison) # root_dir/{class}/poison
    poison_dataloader = torch.utils.data.DataLoader(poison_dataset, batch_size=64, shuffle=False)

    embeddings = []
    clean_labels = []
    poison_labels = []
    for images, lbls in clean_dataloader:
        with torch.no_grad():
            images = images.to(device)
            features = model.encode_image(images)
        embeddings.append(features)
        clean_labels.extend(lbls)

    for images, lbls in poison_dataloader:
        with torch.no_grad():
            images = images.to(device)
            features = model.encode_image(images)
        embeddings.append(features)
        poison_labels.extend(lbls)

    embeddings = torch.cat(embeddings, dim=0)
    clean_labels = torch.tensor(clean_labels)
    poison_labels = torch.tensor(poison_labels)

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(embeddings.cpu().numpy())

    import matplotlib.pyplot as plt

    unique_labels = sorted(set(clean_labels.numpy()))
    num_labels = len(unique_labels)
    cmap = plt.get_cmap('jet', num_labels)

    plt.figure(figsize=(10, 10))
    for idx, label in enumerate(unique_labels):
        subset = reduced_data[:len(clean_labels)][clean_labels.numpy() == label]
        plt.scatter(subset[:, 0], subset[:, 1], color=cmap(idx), marker='o', label=f"{clean_dataset.classes[label]} clean")
        subset = reduced_data[len(clean_labels):][poison_labels.numpy() == label]
        if len(subset) > 0:
            plt.scatter(subset[:, 0], subset[:, 1], color=cmap(idx), marker='x', label=f"{clean_dataset.classes[label]} poison")

    # plt.colorbar(ticks=range(num_labels), format=plt.FuncFormatter(lambda val, loc: clean_dataset.classes[val]))
    plt.legend(loc='best')
    plt.savefig(f"{root_dir}/all.png")
    plt.show()
