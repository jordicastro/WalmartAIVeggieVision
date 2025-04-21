'''
Example command for running training using train.py:

python train.py \
    --dataset_path /home/dgnystro/walmart_project/training/datasets/produce_dataset \
    --train_batch 256 \
    --test_batch 64 \
    --num_epochs 20 \
    --save_image_dir saved_images \
    --output_model produce-dataset-efficientnet_b0.zip
'''

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments including dataset path, batch sizes,
        epoch count, image save directory, and output model filename.
    """
    parser = argparse.ArgumentParser(
        description="Train EfficientNet on produce dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Base path to dataset (should contain 'Training/' and 'Test/' subfolders)",
    )
    parser.add_argument(
        "--train_batch",
        type=int,
        default=32,
        help="Batch size for training loader",
    )
    parser.add_argument(
        "--test_batch",
        type=int,
        default=32,
        help="Batch size for testing loader",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save_image_dir",
        type=str,
        default="saved_images",
        help="Directory to save sample batch image",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="model.zip",
        help="Filename for traced TorchScript model output",
    )

    return parser.parse_args()


def get_data_loaders(base_dir, train_bs, test_bs):
    """
    Create DataLoader objects for train, validation, and test datasets.

    Args:
        base_dir (str): Root directory containing 'Training' and 'Test' folders.
        train_bs (int): Batch size for training and validation.
        test_bs (int): Batch size for test loader.

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_size, val_size,
            test_size, class_names)
    """
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomApply(
            torch.nn.ModuleList([
                transforms.ColorJitter(),
                transforms.GaussianBlur(3),
            ]),
            p=0.1,
        ),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.14, value="random"),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    train_folder = os.path.join(base_dir, "Training")
    test_folder = os.path.join(base_dir, "Test")

    train_ds = datasets.ImageFolder(train_folder, transform=train_tf)
    test_ds = datasets.ImageFolder(test_folder, transform=test_tf)

    print(
        f"Found {len(train_ds)} training images, {len(train_ds.classes)}"
        " classes"
    )
    print(
        f"Found {len(test_ds)} test images,    {len(test_ds.classes)}"
        " classes"
    )

    train_len = int(len(train_ds) * 0.78)
    val_len = len(train_ds) - train_len
    train_split, val_split = random_split(train_ds, [train_len, val_len])

    train_loader = DataLoader(
        train_split,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_len,
        val_len,
        len(test_ds),
        train_ds.classes,
    )


def show_sample_batch(loader, classes, save_dir):
    """
    Display and save a sample batch of images from the DataLoader.

    Args:
        loader (DataLoader): DataLoader to fetch a batch from.
        classes (list): List of class names corresponding to labels.
        save_dir (str): Directory where the image will be saved.
    """
    images, labels = next(iter(loader))
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for i in range(min(20, images.shape[0])):
        ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.set_title(classes[labels[i]])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "batch_train_images.png")
    plt.savefig(out_path)
    plt.close(fig)

    print(f"Saved sample batch to {out_path}")


def build_model(num_classes, device):
    """
    Build and return a fine-tuned EfficientNet-B0 model.

    Args:
        num_classes (int): Number of output classes for the classifier.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The modified EfficientNet-B0 model.
    """
    model = models.efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, num_classes),
    )

    return model.to(device)


def train_model(
    model,
    dataloaders,
    sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
):
    """
    Train the model and return the best weights and training history.

    Args:
        model (torch.nn.Module): Model to train.
        dataloaders (dict): Dict with 'train' and 'val' DataLoaders.
        sizes (dict): Dict with sizes of train and val datasets.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: (best_model, history_dict) where best_model is the model
        with highest validation accuracy and history_dict contains loss
        and accuracy lists.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n" + "-" * 10)

        for phase in ("train", "val"):
            is_train = phase == "train"
            loader = dataloaders[phase]

            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if is_train:
                scheduler.step()

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.double() / sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        print()

    elapsed = time.time() - start_time
    minutes = elapsed // 60
    seconds = elapsed % 60
    print(
        f"Training complete in {minutes:.0f}m {seconds:.0f}s"
    )
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_weights)
    return model, history


def test_model(model, loader, classes, criterion, device):
    """
    Evaluate the trained model on the test dataset and print metrics.

    Args:
        model (torch.nn.Module): Trained model.
        loader (DataLoader): Test data loader.
        classes (list): List of class names.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Computation device.
    """
    print("Running on test set...")
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            matches = preds == labels
            for mask, label in zip(matches, labels):
                class_correct[label] += mask.item()
                class_total[label] += 1

    avg_loss = total_loss / len(loader.dataset)
    print(f"Test Loss: {avg_loss:.6f}")

    for idx, cls_name in enumerate(classes):
        if class_total[idx] > 0:
            accuracy = 100 * class_correct[idx] / class_total[idx]
            correct = int(class_correct[idx])
            total = int(class_total[idx])
            print(
                f"Accuracy of {cls_name:10s}: {accuracy:.2f}% "
                f"({correct}/{total})"
            )

    overall_acc = 100 * np.sum(class_correct) / np.sum(class_total)
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%\n")


def main():
    """
    Main entry point: parse args, prepare data, train and test model, and save output.
    """
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    (
        train_loader,
        val_loader,
        test_loader,
        train_n,
        val_n,
        test_n,
        classes,
    ) = get_data_loaders(
        args.dataset_path,
        args.train_batch,
        args.test_batch,
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    sizes = {"train": train_n, "val": val_n}

    show_sample_batch(
        train_loader,
        classes,
        args.save_image_dir,
    )

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.11
    ).to(device)
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=0.001,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.8
    )

    model_ft, _ = train_model(
        model,
        dataloaders,
        sizes,
        criterion,
        optimizer,
        scheduler,
        args.num_epochs,
        device,
    )

    test_model(
        model_ft,
        test_loader,
        classes,
        criterion,
        device,
    )

    example = torch.rand(1, 3, 224, 224).to("cpu")
    traced = torch.jit.trace(model_ft.cpu(), example)
    traced.save(args.output_model)
    print(f"Saved traced model to {args.output_model}")


if __name__ == "__main__":
    main()
