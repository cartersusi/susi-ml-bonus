import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torchinfo import summary
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=512 * 2 * 2, out_features=2)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    model.train()

    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    return_preds: bool = False,
):
    model.eval()
    val_loss, val_acc = 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

            if return_preds:
                all_preds.extend(val_pred_labels.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    if return_preds:
        return val_loss, val_acc, all_preds, all_labels
    return val_loss, val_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs("models", exist_ok=True)
    os.makedirs("cpp_models", exist_ok=True)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, return_preds=False
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        torch.save(model.state_dict(), f"models/cat_dog_cnn_{epoch + 1}.pth")

        model.eval()
        example_input = torch.randn(1, 3, 150, 150).to(device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(f"cpp_models/cat_dog_cnn_{epoch + 1}.pt")
        model.train()

    return results


def InitModel(visualize: bool = True) -> CatDogCNN:
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = CatDogCNN().to(device)

    if visualize:
        summary(model, input_size=[1, 3, 150, 150])

    return model


def TrainModel(model, train_dataloader_augmented, val_dataloader_augmented):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    start_time = timer()

    model_results = train(
        model=model,
        train_dataloader=train_dataloader_augmented,
        val_dataloader=val_dataloader_augmented,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=25,
    )

    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    return model_results, loss_fn


def LoadModel(model_path: str) -> CatDogCNN:
    model = CatDogCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def Visualise(model, model_results, val_dataloader_augmented, loss_fn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(model_results["train_loss"], label="Train Loss")
    ax1.plot(model_results["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(model_results["train_acc"], label="Train Accuracy")
    ax2.plot(model_results["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    model.eval()
    _, val_acc, val_preds, val_labels = val_step(
        model, val_dataloader_augmented, loss_fn, return_preds=True
    )

    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
