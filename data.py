import os
import random
import shutil
import sys

import kagglehub
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def cat_dog_download() -> bool:
    if not os.path.exists("~/.kaggle/kaggle.json"):
        try:
            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
            shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
            os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
        except Exception as e:
            print("Unable to copy credentials", e)
            return False

    print(
        f"Kaggle Dataset at: {kagglehub.dataset_download('karakaggle/kaggle-cat-vs-dog-dataset')}"
    )
    return True


class Data:
    def __init__(self, fpath):
        self.data_path = fpath
        self.dataset = datasets.ImageFolder(root=fpath)

        # Training image transforms (70% training)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),  # Resize all images to 150×150 pixels
                transforms.RandomHorizontalFlip(),  # Apply basic data augmentation (horizontal flip)
                transforms.RandomRotation(
                    15
                ),  # Apply basic data augmentation (rotation)
                transforms.ToTensor(),  # Normalize pixel values to [0, 1] range
            ]
        )

        # Validation image transforms (15% validation)
        self.validation_transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
            ]
        )

        # Test image transforms (15% val)
        self.test_transform = self.validation_transform

    @property
    def data_size(self) -> int:
        return len(self.dataset)

    @property
    def train_size(self) -> int:
        return int(0.7 * self.data_size)

    @property
    def validation_size(self) -> int:
        return int(0.15 * self.data_size)

    @property
    def test_size(self) -> int:
        return int(self.data_size - self.train_size - self.validation_size)

    def preprocess(self):
        if not self.dataset:
            raise Exception("Dataset must be loaded before preprocessing.")

        train_start, validation_start, test_start = random_split(
            self.dataset, [self.train_size, self.validation_size, self.test_size]
        )

        self.train_data = Subset(
            datasets.ImageFolder(self.data_path, transform=self.train_transform),
            train_start.indices,
        )
        self.val_data = Subset(
            datasets.ImageFolder(self.data_path, transform=self.validation_transform),
            validation_start.indices,
        )
        self.test_data = Subset(
            datasets.ImageFolder(self.data_path, transform=self.test_transform),
            test_start.indices,
        )

    def show_samples(self, train_data=False):
        cats = [
            i for i, (path, label) in enumerate(self.dataset.samples) if "Cat" in path
        ]
        dogs = [
            i for i, (path, label) in enumerate(self.dataset.samples) if "Dog" in path
        ]

        sample_cats = random.sample(cats, min(10, len(cats)))
        sample_dogs = random.sample(dogs, min(10, len(dogs)))

        title = "Augmented Images" if train_data else "Original Images"
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        for idx, ax in enumerate(axes.flat):
            img_idx = sample_cats[idx % 10] if idx < 10 else sample_dogs[idx % 10]
            img, _ = self.dataset[img_idx]

            if train_data:
                img = self.train_transform(img).permute(1, 2, 0).numpy()

            ax.imshow(img)
            ax.set_title("Cat" if idx < 10 else "Dog")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def augment(self, datatype: str):
        torch.manual_seed(42)

        if datatype == "train":
            return DataLoader(
                self.train_data, batch_size=32, shuffle=True, num_workers=os.cpu_count()
            )
        else:
            return DataLoader(
                self.val_data, batch_size=32, shuffle=True, num_workers=os.cpu_count()
            )


def InitData(data_path: str, visualize: bool = False) -> Data:
    dataset = Data(data_path)
    if visualize:
        dataset.show_samples()

    dataset.preprocess()

    if visualize:
        dataset.show_samples()

    return dataset
