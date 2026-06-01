# ML PROJECT 1 (DEEP LEARNING -IMAGE OR TEXT CLASSIFICATION)
I created a complete PyTorch transfer-learning project for a small image classifier using a CIFAR-10 subset, including the training script, inference script, dependency file, and run instructions. The project uses a pretrained ResNet-18, image augmentation, saves metrics and plots, and is structured to produce a saved model plus evaluation outputs when run in a local Python environment with PyTorch installed .



import os
import json
import copy
import time
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs(base):
    Path(base).mkdir(parents=True, exist_ok=True)
    Path(base, 'plots').mkdir(parents=True, exist_ok=True)
    Path(base, 'artifacts').mkdir(parents=True, exist_ok=True)


def get_dataloaders(data_dir, batch_size=64, num_workers=2, train_subset=12000, val_subset=2000, test_subset=2000):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    full_eval_train = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    indices = list(range(len(full_train)))
    random.shuffle(indices)
    train_idx = indices[:train_subset]
    val_idx = indices[train_subset:train_subset + val_subset]
    test_idx = list(range(min(test_subset, len(test_dataset))))

    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_eval_train, val_idx)
    test_dataset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = full_train.classes
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes, device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(labels_all, preds_all)
    return epoch_loss, epoch_acc, np.array(labels_all), np.array(preds_all)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        preds_all, labels_all = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(labels_all, preds_all)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


def save_curves(history, out_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], marker='o', label='Train')
    plt.plot(epochs, history['val_loss'], marker='o', label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], marker='o', label='Train')
    plt.plot(epochs, history['val_acc'], marker='o', label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def save_confusion(cm, class_names, out_path):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    set_seed(42)
    out_dir = 'output/dl_image_classifier'
    data_dir = 'output/dl_image_classifier/data'
    make_dirs(out_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir)

    model = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    save_curves(history, f'{out_dir}/plots/training_curves.png')
    save_confusion(cm, class_names, f'{out_dir}/plots/confusion_matrix.png')

    model_path = f'{out_dir}/artifacts/resnet18_cifar10_subset.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_size': 224,
    }, model_path)

    metrics = {
        'device': str(device),
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'history': history,
        'classification_report': report,
        'class_names': class_names,
    }

    with open(f'{out_dir}/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(f'{out_dir}/artifacts/run_summary.txt', 'w') as f:
        f.write(json.dumps(
            {k: v for k, v in metrics.items() if k not in ['history', 'classification_report', 'class_names']},
            indent=2
        ))


if __name__ == '__main__':
    main()
