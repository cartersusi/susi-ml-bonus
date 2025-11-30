import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from train import device

NOISE_LEVELS = [50, 55, 60, 65, 70, 75]


class NoisyDogDataset(Dataset):
    def __init__(self, root_dir, noise_level, transform=None, prefix="train_"):
        self.root_dir = root_dir
        self.noise_level = noise_level
        self.transform = transform
        self.prefix = prefix

        all_files = os.listdir(root_dir)
        self.image_files = [
            f
            for f in all_files
            if f.startswith(prefix)
            and f.endswith(".jpg")
            and f.endswith(f"_{noise_level}.jpg")
        ]

        print(
            f"Found {len(self.image_files)} images at noise level {noise_level}% with prefix '{prefix}'"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = 1

        return image, label, self.image_files[idx]


def evaluate_model_at_noise_level(
    model, dog_noise_images, noise_level, transform, prefix="val_"
):
    dataset = NoisyDogDataset(dog_noise_images, noise_level, transform, prefix=prefix)

    if len(dataset) == 0:
        print(
            f"Warning: No images found for noise level {noise_level}% with prefix '{prefix}'"
        )
        return 0.0, [], [], []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    predictions = []
    confidences = []
    filenames = []

    with torch.no_grad():
        for images, labels, fnames in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.append(predicted.item())
            confidences.append(probabilities[0][1].item())
            filenames.append(fnames[0])

    accuracy = 100 * correct / total if total > 0 else 0

    return accuracy, predictions, confidences, filenames


def evaluate_across_noise_levels(model, dog_noise_images, transform, prefix="val_"):
    results = {
        "noise_levels": [],
        "accuracies": [],
        "confidences": [],
        "per_image_results": {},
    }

    for noise_level in NOISE_LEVELS:
        print(f"\nNoise Level: {noise_level}%")
        accuracy, predictions, confidences, filenames = evaluate_model_at_noise_level(
            model, dog_noise_images, noise_level, transform, prefix=prefix
        )

        results["noise_levels"].append(noise_level)
        results["accuracies"].append(accuracy)
        results["confidences"].append(np.mean(confidences) if confidences else 0)
        results["per_image_results"][noise_level] = {
            "predictions": predictions,
            "confidences": confidences,
            "filenames": filenames,
        }

        print(f"  Accuracy: {accuracy:.2f}%")
        print(
            f"  Avg Confidence: {np.mean(confidences):.4f}"
            if confidences
            else "  No images"
        )

    return results


def plot_accuracy_vs_noise(results_before, results_after, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(
        results_before["noise_levels"],
        results_before["accuracies"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="Before Fine-tuning",
        color="#e74c3c",
    )

    plt.plot(
        results_after["noise_levels"],
        results_after["accuracies"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="After Fine-tuning",
        color="#27ae60",
    )

    plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Model Accuracy vs. Noise Level\n(Before and After Fine-tuning)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11, loc="best")
    plt.ylim([0, 105])

    for i, (noise, acc_before, acc_after) in enumerate(
        zip(
            results_before["noise_levels"],
            results_before["accuracies"],
            results_after["accuracies"],
        )
    ):
        plt.annotate(
            f"{acc_before:.1f}%",
            (noise, acc_before),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#e74c3c",
        )
        plt.annotate(
            f"{acc_after:.1f}%",
            (noise, acc_after),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
            color="#27ae60",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_image_robustness(results, save_path, title_suffix=""):
    """Plot 2: Per-Image Robustness Curve (Spider or Line Plot)"""
    plt.figure(figsize=(14, 8))

    # Get unique image identifiers (base names without noise level)
    first_noise = results["noise_levels"][0]
    filenames = results["per_image_results"][first_noise]["filenames"]

    # Extract base names (remove noise level suffix)
    base_names = [f.rsplit("_", 1)[0] for f in filenames]

    # Plot each image's accuracy trajectory
    for idx, base_name in enumerate(base_names):
        accuracies = []
        for noise_level in results["noise_levels"]:
            pred = results["per_image_results"][noise_level]["predictions"][idx]
            accuracies.append(1 if pred == 1 else 0)

        plt.plot(
            results["noise_levels"],
            accuracies,
            marker="o",
            linewidth=1.5,
            markersize=4,
            alpha=0.6,
            label=f"Image {idx + 1}" if len(base_names) <= 10 else None,
        )

    # Plot average trend
    avg_accuracies = []
    for noise_level in results["noise_levels"]:
        preds = results["per_image_results"][noise_level]["predictions"]
        avg_acc = sum([1 if p == 1 else 0 for p in preds]) / len(preds) if preds else 0
        avg_accuracies.append(avg_acc)

    plt.plot(
        results["noise_levels"],
        avg_accuracies,
        marker="D",
        linewidth=3,
        markersize=8,
        color="black",
        label="Average",
        zorder=10,
    )

    plt.xlabel("Noise Level (%)", fontsize=14, fontweight="bold")
    plt.ylabel(
        "Correct Classification (1=Correct, 0=Wrong)", fontsize=14, fontweight="bold"
    )
    plt.title(
        f"Per-Image Robustness Across Noise Levels {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([-0.1, 1.1])

    if len(base_names) <= 10:
        plt.legend(fontsize=9, loc="best", ncol=2)
    else:
        plt.legend(["Average"], fontsize=12, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path, title_suffix="", num_classes=2):
    available_noise = [n for n in NOISE_LEVELS if n in results["noise_levels"]]

    n_plots = len(available_noise)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, noise_level in enumerate(available_noise):
        predictions = results["per_image_results"][noise_level]["predictions"]
        true_labels = [1] * len(predictions)

        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"],
            cbar=True,
        )

        axes[idx].set_title(
            f"Noise Level: {noise_level}%", fontsize=12, fontweight="bold"
        )
        axes[idx].set_ylabel("True Label", fontsize=11)
        axes[idx].set_xlabel("Predicted Label", fontsize=11)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Confusion Matrices Across Noise Levels {title_suffix}",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_vs_noise(results_before, results_after, save_path):
    plt.figure(figsize=(12, 7))

    mean_conf_before = []
    std_conf_before = []
    mean_conf_after = []
    std_conf_after = []

    for noise_level in results_before["noise_levels"]:
        confs_before = results_before["per_image_results"][noise_level]["confidences"]
        mean_conf_before.append(np.mean(confs_before) if confs_before else 0)
        std_conf_before.append(np.std(confs_before) if confs_before else 0)

        confs_after = results_after["per_image_results"][noise_level]["confidences"]
        mean_conf_after.append(np.mean(confs_after) if confs_after else 0)
        std_conf_after.append(np.std(confs_after) if confs_after else 0)

    plt.errorbar(
        results_before["noise_levels"],
        mean_conf_before,
        yerr=std_conf_before,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Before Fine-tuning",
        color="#e74c3c",
        capsize=5,
        capthick=2,
        alpha=0.8,
    )

    plt.errorbar(
        results_after["noise_levels"],
        mean_conf_after,
        yerr=std_conf_after,
        marker="s",
        linewidth=2,
        markersize=8,
        label="After Fine-tuning",
        color="#27ae60",
        capsize=5,
        capthick=2,
        alpha=0.8,
    )

    plt.xlabel("Noise Level (%)", fontsize=14, fontweight="bold")
    plt.ylabel(
        "Average Predicted Probability (Dog Class)", fontsize=14, fontweight="bold"
    )
    plt.title(
        "Model Confidence vs. Noise Level\n(Before and After Fine-tuning)",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_error_rate_distribution(results, save_path, title_suffix=""):
    plt.figure(figsize=(16, 7))

    data_to_plot = []
    labels = []

    for noise_level in results["noise_levels"]:
        predictions = results["per_image_results"][noise_level]["predictions"]
        accuracies = [1 if p == 1 else 0 for p in predictions]
        data_to_plot.append(accuracies)
        labels.append(f"{noise_level}%")

    parts = plt.violinplot(
        data_to_plot,
        positions=range(len(data_to_plot)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("#3498db")
        pc.set_alpha(0.6)

    plt.boxplot(
        data_to_plot,
        positions=range(len(data_to_plot)),
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )

    plt.xlabel("Noise Level", fontsize=14, fontweight="bold")
    plt.ylabel(
        "Classification Result (1=Correct, 0=Wrong)", fontsize=14, fontweight="bold"
    )
    plt.title(
        f"Distribution of Classification Results Across Noise Levels {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_noisy_image_examples(
    dog_noise_images, model, transform, save_path, num_images=3
):
    all_files = os.listdir(dog_noise_images)
    val_files = [f for f in all_files if f.startswith("val_") and f.endswith("_0.jpg")]

    if len(val_files) == 0:
        print("No validation images found for visualization")
        return

    selected_images = val_files[: min(num_images, len(val_files))]
    base_names = [f.replace("val_", "").replace("_0.jpg", "") for f in selected_images]

    n_rows = len(selected_images)
    n_cols = len(NOISE_LEVELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for row_idx, base_name in enumerate(base_names):
            for col_idx, noise_level in enumerate(NOISE_LEVELS):
                img_filename = f"val_{base_name}_{noise_level}.jpg"
                img_path = os.path.join(dog_noise_images, img_filename)

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    axes[row_idx, col_idx].imshow(img)
                    axes[row_idx, col_idx].axis("off")

                    # Get prediction
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    predicted = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][predicted].item()

                    label_text = "Dog" if predicted == 1 else "Cat"
                    color = "green" if predicted == 1 else "red"

                    axes[row_idx, col_idx].set_title(
                        f"Noise: {noise_level}%\n{label_text} ({confidence:.2f})",
                        fontsize=10,
                        fontweight="bold",
                        color=color,
                    )
                else:
                    axes[row_idx, col_idx].axis("off")
                    axes[row_idx, col_idx].text(
                        0.5,
                        0.5,
                        "Image\nNot Found",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

    fig.suptitle(
        "Visual Examples: Model Predictions on Noisy Images",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_comparison_bar_chart(results_before, results_after, save_path):
    plt.figure(figsize=(14, 6))

    x = np.arange(len(results_before["noise_levels"]))
    width = 0.35

    plt.bar(
        x - width / 2,
        results_before["accuracies"],
        width,
        label="Before Fine-tuning",
        color="#e74c3c",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        results_after["accuracies"],
        width,
        label="After Fine-tuning",
        color="#27ae60",
        alpha=0.8,
    )

    plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Accuracy Comparison: Before vs. After Fine-tuning",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, results_before["noise_levels"])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()
