import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2 as T
from torchvision.utils import save_image
from tqdm import tqdm

from test import (
    NOISE_LEVELS,
    NoisyDogDataset,
    evaluate_across_noise_levels,
    plot_accuracy_vs_noise,
    plot_comparison_bar_chart,
    plot_confidence_vs_noise,
    plot_confusion_matrices,
    plot_error_rate_distribution,
    plot_noisy_image_examples,
    plot_per_image_robustness,
)
from train import device


def add_noise(dog_dir: str, out_dir: str, train_size: int = 20, val_size: int = 10):
    dog_images = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)]

    total_size = train_size + val_size
    selected_images = random.sample(dog_images, total_size)

    train_images = selected_images[:train_size]
    val_images = selected_images[train_size:]

    def noise(img_path: str, prefix: str = ""):
        img = Image.open(img_path).convert("RGB")

        for i in range(0, 105, 5):
            strength = 0.01 * i
            gaussian_noise_transform = T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.GaussianNoise(mean=0.0, sigma=strength, clip=True),
                ]
            )
            noisy_image_tensor = gaussian_noise_transform(img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(
                out_dir,
                f"{prefix}{base_name}_{i}.jpg",
            )
            save_image(noisy_image_tensor, output_path)

    [
        noise(img_path, prefix="train_")
        for img_path in tqdm(train_images, desc="Adding noise to Training Dog Images")
    ]

    [
        noise(img_path, prefix="val_")
        for img_path in tqdm(val_images, desc="Adding noise to Validation Dog Images")
    ]

    print(f"\nCreated {train_size * len(NOISE_LEVELS)} training images (with noise)")
    print(f"Created {val_size * len(NOISE_LEVELS)} validation images (with noise)")


def fine_tune(
    model,
    dog_noise_images,
    noise_levels_to_train=[50, 55, 60],
    num_epochs=10,
    learning_rate=0.0001,
):
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )

    all_images = []
    all_labels = []

    for noise_level in noise_levels_to_train:
        dataset = NoisyDogDataset(
            dog_noise_images, noise_level, transform, prefix="train_"
        )
        for i in range(len(dataset)):
            img, label, _ = dataset[i]
            all_images.append(img)
            all_labels.append(label)

    class SimpleDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    train_dataset = SimpleDataset(all_images, all_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model.train()

    for param in model.conv_layer_1.parameters():
        param.requires_grad = False
    for param in model.conv_layer_2.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    os.makedirs("finetuned-models", exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), "./finetuned-models/cat_dog_noise_cnn.pth")
    print("\nFine-tuned model saved to: ./finetuned-models/cat_dog_noise_cnn.pth")

    return model


def FinetuneModel(model_before, dog_noise_images, visualize=False):
    RESULTS_DIR = "./results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )

    results_before = evaluate_across_noise_levels(
        model_before, dog_noise_images, transform, prefix="val_"
    )

    model_after = fine_tune(
        model_before,
        dog_noise_images,
        noise_levels_to_train=NOISE_LEVELS,
        num_epochs=10,
        learning_rate=0.0001,
    )

    results_after = evaluate_across_noise_levels(
        model_after, dog_noise_images, transform, prefix="val_"
    )

    plot_accuracy_vs_noise(
        results_before,
        results_after,
        os.path.join(RESULTS_DIR, "accuracy_vs_noise.png"),
    )

    plot_per_image_robustness(
        results_before,
        os.path.join(RESULTS_DIR, "per_image_robustness_before.png"),
        title_suffix="(Before Fine-tuning)",
    )

    plot_per_image_robustness(
        results_after,
        os.path.join(RESULTS_DIR, "per_image_robustness_after.png"),
        title_suffix="(After Fine-tuning)",
    )

    plot_confusion_matrices(
        results_before,
        os.path.join(RESULTS_DIR, "confusion_matrices_before.png"),
        title_suffix="(Before Fine-tuning)",
    )

    plot_confusion_matrices(
        results_after,
        os.path.join(RESULTS_DIR, "confusion_matrices_after.png"),
        title_suffix="(After Fine-tuning)",
    )

    plot_confidence_vs_noise(
        results_before,
        results_after,
        os.path.join(RESULTS_DIR, "confidence_vs_noise.png"),
    )

    plot_error_rate_distribution(
        results_before,
        os.path.join(RESULTS_DIR, "error_distribution_before.png"),
        title_suffix="(Before Fine-tuning)",
    )

    plot_error_rate_distribution(
        results_after,
        os.path.join(RESULTS_DIR, "error_distribution_after.png"),
        title_suffix="(After Fine-tuning)",
    )

    plot_comparison_bar_chart(
        results_before,
        results_after,
        os.path.join(RESULTS_DIR, "comparison_bars.png"),
    )

    plot_noisy_image_examples(
        dog_noise_images,
        model_after,
        transform,
        os.path.join(RESULTS_DIR, "noisy_image_examples.png"),
        num_images=3,
    )

    for i, noise in enumerate(results_before["noise_levels"]):
        acc_before = results_before["accuracies"][i]
        acc_after = results_after["accuracies"][i]
        improvement = acc_after - acc_before

        print(
            f"Noise {noise:3d}%: Before={acc_before:5.1f}% | After={acc_after:5.1f}% | "
            f"Improvement={improvement:+5.1f}%"
        )
