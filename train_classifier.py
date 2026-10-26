import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Dataset (CRACK + UNCRACK)
# ===============================
class SDNETClassDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        structure = {
            "bridge_deck": ("CD", "UD"),
            "walls": ("CW", "UW"),
            "pavements": ("CP", "UP")
        }

        for main, (crack, uncrack) in structure.items():
            cpath = os.path.join(root_dir, main, crack)
            upath = os.path.join(root_dir, main, uncrack)

            for f in os.listdir(cpath):
                self.samples.append((os.path.join(cpath, f), 1))
            for f in os.listdir(upath):
                self.samples.append((os.path.join(upath, f), 0))

        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label)

dataset = SDNETClassDataset("SDNET2018")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===============================
# Classifier Model
# ===============================
class CrackClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16, 2)
        )

    def forward(self, x):
        return self.net(x)

model = CrackClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===============================
# Train (FAST)
# ===============================
for epoch in range(50):
    model.train()
    running_loss = 0
    
    correct = 0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()

    acc = correct / len(dataset)
    loss_=running_loss / len(loader)
    print(f"Epoch {epoch+1} | Accuracy: {acc:.3f} | Loss: {loss_:.4f}")

# ===============================
# SAVE MODEL
# ===============================
torch.save(model.state_dict(), "sdnet_classifier.pth")
print("✅ Classifier saved as sdnet_classifier.pth")
