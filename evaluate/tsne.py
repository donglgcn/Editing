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
    subdir = "clean"        # todo replace with "poison" to visualize poison images
    root_dir = "../cifar/"  # todo replace with your directory path
    dataset = CleanImageFolder(root_dir, transform=preprocess, subset=subdir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    embeddings = []
    labels = []
    for images, lbls in dataloader:
        with torch.no_grad():
            images = images.to(device)
            features = model.encode_image(images)
        embeddings.append(features)
        labels.extend(lbls)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(embeddings.cpu().numpy())

    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels.numpy()))
    num_labels = len(unique_labels)
    cmap = plt.get_cmap('jet', num_labels)

    plt.figure(figsize=(10, 10))
    for idx, label in enumerate(unique_labels):
        subset = reduced_data[labels.numpy() == label]
        plt.scatter(subset[:, 0], subset[:, 1], color=cmap(idx), label=dataset.classes[label])

    plt.colorbar(ticks=range(num_labels), format=plt.FuncFormatter(lambda val, loc: dataset.classes[val]))
    plt.legend(loc='best')
    plt.savefig(f"{root_dir}/{subdir}.png")
    plt.show()
