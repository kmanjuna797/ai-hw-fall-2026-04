import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 5
STEPS = 20
EPS_LIST = [0.05, 0.1, 0.2, 0.3]
DECAY = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop1(F.max_pool2d(x, 2))
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return self.fc2(self.drop2(x))


def grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    return torch.autograd.grad(F.cross_entropy(model(x), y), x)[0]


def fgsm(model, x, y, eps):
    return (x + eps * grad(model, x, y).sign()).clamp(0, 1).detach()


def ifgsm(model, x, y, eps):
    alpha = 2.5 * eps / STEPS
    x_adv = x.clone().detach()
    for _ in range(STEPS):
        x_adv = x_adv + alpha * grad(model, x_adv, y).sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


def mifgsm(model, x, y, eps):
    alpha = eps / STEPS
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)
    for _ in range(STEPS):
        gr = grad(model, x_adv, y)
        gr = gr / gr.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        g = DECAY * g + gr
        x_adv = x_adv + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


tf = transforms.ToTensor()
train_dl = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tf), batch_size=128, shuffle=True)
test_dl = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=tf), batch_size=500)

model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

correct = 0
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
print(f"Recognition rate before attack: {correct / len(test_dl.dataset) * 100:.2f}%")

for name, attack in [("FGSM", fgsm), ("I-FGSM", ifgsm), ("MI-FGSM", mifgsm)]:
    for eps in EPS_LIST:
        n_correct = n_fooled = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            x_adv = attack(model, x, y, eps)
            with torch.no_grad():
                ok = model(x).argmax(1) == y
                fooled = model(x_adv).argmax(1) != y
            n_correct += ok.sum().item()
            n_fooled += (ok & fooled).sum().item()
        print(f"{name} eps={eps}: ASR = {n_fooled / n_correct * 100:.2f}%")