import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ===============================
# Models
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

class FastUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))
        d1 = torch.nn.functional.interpolate(e2, scale_factor=2)
        d2 = torch.relu(self.dec1(d1))
        return torch.sigmoid(self.out(d2))

# ===============================
# Load models
# ===============================
classifier = CrackClassifier()
classifier.load_state_dict(torch.load("sdnet_classifier.pth", map_location="cpu"))
classifier.eval()

depth_model = FastUNet()
depth_model.load_state_dict(torch.load("sdnet_fast_depth_model.pth", map_location="cpu"))
depth_model.eval()

# ===============================
# Preprocess
# ===============================
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def future_depth(depth, years, alpha=0.06):
    return torch.clamp(depth * (1 + alpha*years), 0, 1)

# ===============================
# REAL PREDICTION
# ===============================
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs.max().item()

    label = "CRACKED" if pred_class == 1 else "UNCRACKED"

    print("\n🔍 PREDICTION RESULT")
    print("-----------------------------")
    print(f"Crack Status : {label}")
    print(f"Confidence   : {confidence:.2f}")

    if pred_class == 1:
        with torch.no_grad():
            depth_now = depth_model(x)[0,0]

        d1 = future_depth(depth_now, 1)
        d5 = future_depth(depth_now, 5)
        d10 = future_depth(depth_now, 10)

        print(f"Current Depth : {depth_now.mean():.4f}")
        print(f"1 Year Depth  : {d1.mean():.4f}")
        print(f"5 Year Depth  : {d5.mean():.4f}")
        print(f"10 Year Depth : {d10.mean():.4f}")

        plt.figure(figsize=(12,3))
        plt.subplot(1,5,1); plt.title("Input"); plt.imshow(img)
        plt.subplot(1,5,2); plt.title("Now"); plt.imshow(depth_now, cmap="inferno")
        plt.subplot(1,5,3); plt.title("1 Year"); plt.imshow(d1, cmap="inferno")
        plt.subplot(1,5,4); plt.title("5 Years"); plt.imshow(d5, cmap="inferno")
        plt.subplot(1,5,5); plt.title("10 Years"); plt.imshow(d10, cmap="inferno")
        plt.show()
    else:
        print("Depth analysis skipped (No crack detected)")


predict("SDNET2018/bridge_deck/CD/7052-33.jpg")
