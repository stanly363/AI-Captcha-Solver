import os
import zipfile
import random
import time
import argparse
from glob import glob

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

class CaptchaDataset(Dataset):
    def __init__(self, paths, char2idx, transform=None):
        self.paths = paths
        self.char2idx = char2idx
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        lab = os.path.splitext(os.path.basename(p))[0]
        lbl = torch.tensor([self.char2idx[c] for c in lab], dtype=torch.long)
        img = Image.open(p).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, lbl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    SEED = args.seed if args.seed is not None else int(time.time())
    random.seed(SEED)
    torch.manual_seed(SEED)

    KAGGLE_SLUG     = 'fournierp/captcha-version-2-images'
    BATCH_SIZE      = 64
    MODEL_PATH      = 'model.pth'
    NUM_EPOCHS      = 50
    LR              = 1e-3
    TRAIN_VAL_SPLIT = 0.8

    def prepare_data_dir(path):
        if os.path.isdir(path): return path
        if path.endswith('.zip'):
            out = os.path.splitext(path)[0]
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(out)
            return out
        raise RuntimeError(f"Unexpected dataset path: {path}")

    dataset_path = kagglehub.dataset_download(KAGGLE_SLUG)
    DOWNLOAD_DIR = prepare_data_dir(dataset_path)

    all_images = glob(os.path.join(DOWNLOAD_DIR, '**', '*.png'), recursive=True)
    train_paths, val_paths = train_test_split(
        all_images, train_size=TRAIN_VAL_SPLIT,
        random_state=SEED, shuffle=True
    )

    labels      = [os.path.splitext(os.path.basename(p))[0] for p in all_images]
    char_set    = sorted(set(''.join(labels)))
    char2idx    = {c: i for i, c in enumerate(char_set)}
    num_chars   = len(labels[0])
    num_classes = len(char_set)

    train_transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.RandomAffine(5, (0.02, 0.05), (0.9, 1.1), shear=2, fill=255),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = CaptchaDataset(train_paths, char2idx, train_transform)
    val_ds   = CaptchaDataset(val_paths,   char2idx, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True, prefetch_factor=4
    )

    backbone = resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

    class Solver(nn.Module):
        def __init__(self, feat_extractor, n_chars, n_classes):
            super().__init__()
            self.feature = feat_extractor
            self.dropout = nn.Dropout2d(0.25)
            self.gap     = nn.AdaptiveAvgPool2d(1)
            self.fc      = nn.Linear(backbone.fc.in_features, n_chars * n_classes)
            self.n_chars   = n_chars
            self.n_classes = n_classes

        def forward(self, x):
            x = self.feature(x)
            x = self.dropout(x)
            x = self.gap(x).view(x.size(0), -1)
            x = self.fc(x)
            return x.view(-1, self.n_chars, self.n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = Solver(feature_extractor, num_chars, num_classes).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR*10,
        total_steps=NUM_EPOCHS * steps_per_epoch,
        pct_start=0.3, anneal_strategy='cos'
    )

    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0.0

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for imgs, lbls in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast(device.type):
                    outs = model(imgs)
                    loss = sum(criterion(outs[:, i], lbls[:, i]) for i in range(num_chars)) / num_chars
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    lbls = lbls.to(device, non_blocking=True)
                    outs = model(imgs)
                    preds = outs.argmax(dim=2)
                    correct += (preds == lbls).sum().item()
                    total += lbls.numel()
            acc = correct / total * 100

            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val-char-acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODEL_PATH)
            
            if acc > best_acc:
                best_acc, wait = acc, 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                wait += 1
                if wait >= patience:
                    print(f"No improvement for {patience} epochs — stopping early.")
                    break

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving best model (acc={best_acc:.2f}%) to {MODEL_PATH}.")
        torch.save(model.state_dict(), MODEL_PATH)
        return

    print("Training complete.")

if __name__ == '__main__':
    main()
