import argparse
import json
import os

from data import InitData, cat_dog_download
from finetune import FinetuneModel, add_noise
from train import InitModel, LoadModel, TrainModel, Visualise

with open("conf.json", "r") as file:
    conf = json.load(file)

DOG_NOISE_IMAGES = conf["dog_noise_images"]
CAT_DOG_IMAGES = conf["cat_dog_images"]
MODEL_PATH = conf["model_path"]


def train():
    if os.path.exists("./models/cat_dog_cnn_25.pth") and os.path.exists(
        "cpp_models/cat_dog_cnn_25.pt"
    ):
        print("Models already trained.")
        return
    print(CAT_DOG_IMAGES)
    dataset = InitData(CAT_DOG_IMAGES)
    model = InitModel()

    train_dataloader_augmented = dataset.augment("train")
    val_dataloader_augmented = dataset.augment("val")

    results, loss_fn = TrainModel(
        model, train_dataloader_augmented, val_dataloader_augmented
    )
    print(results)

    Visualise(model, results, val_dataloader_augmented, loss_fn)


def finetune():
    if not os.path.exists(DOG_NOISE_IMAGES):
        os.makedirs(DOG_NOISE_IMAGES, exist_ok=True)
        add_noise(os.path.join(CAT_DOG_IMAGES, "Dog"), DOG_NOISE_IMAGES)
    model = LoadModel(MODEL_PATH)
    FinetuneModel(model, DOG_NOISE_IMAGES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CAP4613 Project",
        description="Cat Dog Training & Finetuning",
    )
    parser.add_argument("--d", action="store_true")
    parser.add_argument("--t", action="store_true")
    parser.add_argument("--f", action="store_true")
    args = parser.parse_args()

    if args.d:
        print("Downloading dataset from kaggle...")
        res = cat_dog_download()
        if not res:
            print("Error downloading Kaggle Dataset")
        else:
            print("Kaggle dataset downloaded. Update conf.json")
    if args.t:
        print("Training Models to './models/' & './cpp_models'")
        train()
    elif args.f:
        print(f"Finetuning '{MODEL_PATH}'")
        finetune()
    else:
        print("No args provided. Doing Nothing.")
