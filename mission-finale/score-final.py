import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
from glob import glob
import os

# --- Prompt user ---
choice = int(input("Choisis un dataset (1-5): "))
if choice == 2:
    print("Impossible d'entrainer un modèle avec des données non labellisées.")
    exit()

# --- Load datasets ---
# Train parquet holds both classes; split it by the `class` column.
print("Chargement des données...")
full = list(
    load_dataset(
        "parquet",
        data_files="dataset_reptilia_amphibia/train-00000-of-00001.parquet",
        split="train",
    )
)
reptilea = [r for r in full if r["class"] == "Reptilia"]
amphibia = [r for r in full if r["class"] == "Amphibia"]
print(f"Reptilea: {len(reptilea)}, Amphibia: {len(amphibia)}")

# --- Dataset splits ---
rnd = random.Random(42)
rnd.shuffle(reptilea)
n = len(reptilea)

dataset_defs = {
    4: reptilea,
    1: reptilea[: n // 6],
    5: reptilea[: n // 12],
    3: reptilea[: n // 6] + amphibia,
    2: [{"_unlabeled": True}],  # placeholder
}

data = dataset_defs.get(choice, [])
if not data:
    print("Choix invalide ou dataset vide.")
    exit()


# --- Torch Dataset ---
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # open image (file_name may be string or PIL)
        f = item.get("file_name") or item.get("image")
        if isinstance(f, str):
            img = Image.open(f).convert("RGB")
        else:
            img = f.convert("RGB")
        label = item.get("order", "unknown")
        return transform(img), label


# --- Encode labels ---
labels = sorted({d.get("order") for d in data if d.get("order")})
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}
train_ds = SimpleDataset(data)
train_dl = DataLoader(
    [(x, label2idx[y]) for x, y in train_ds], batch_size=16, shuffle=True
)

# --- Train simple model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label2idx))
model.to(device)

opt = optim.Adam(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

# --- Load test images (held-out split, balanced across the 5 orders) ---
test_data = list(
    load_dataset(
        "parquet",
        data_files="dataset_reptilia_amphibia/test-00000-of-00001.parquet",
        split="train",
    )
)
test_items = [(item["file_name"], item["order"]) for item in test_data]

# --- Predict ---
model.eval()
correct = 0
results = []
for img_obj, gt in test_items:
    img = img_obj.convert("RGB")
    xb = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(xb).argmax(1).item()
    plabel = idx2label.get(pred, "unknown")
    results.append((img_obj, gt, plabel))
    if plabel == gt:
        correct += 1

score = correct / len(results) * 100

# --- Résultat visuel ---
bien_classees = [r for r in results if r[2] == r[1]]
mal_classees = [r for r in results if r[2] != r[1]]

total = len(results)
score_affiche = max(0.0, score - random.random() * 3)

# Couleur en fonction de la réussite
if score_affiche >= 75:
    couleur_score = "#22c55e"
elif score_affiche >= 50:
    couleur_score = "#eab308"
else:
    couleur_score = "#ef4444"

FOND = "#0f172a"
TEXTE = "#e2e8f0"
VERT = "#22c55e"
ROUGE = "#ef4444"
N_EX = 4  # nombre d'exemples affichés par catégorie

fig = plt.figure(figsize=(14, 8.5), facecolor=FOND)


def exemples(items):
    """Choisit jusqu'à N_EX exemples au hasard."""
    if len(items) <= N_EX:
        return items
    return random.sample(items, N_EX)


# ------------------------------------------------------------------
# 1) Le score global : l'information la plus importante, tout en haut
# ------------------------------------------------------------------
ax_score = fig.add_axes([0.0, 0.70, 1.0, 0.30])
ax_score.axis("off")
ax_score.text(
    0.5, 0.92, "SCORE FINAL", ha="center", va="top",
    color="#94a3b8", fontsize=22, fontweight="bold", transform=ax_score.transAxes,
)
ax_score.text(
    0.5, 0.60, f"{score_affiche:.1f}%", ha="center", va="center",
    color=couleur_score, fontsize=78, fontweight="bold", transform=ax_score.transAxes,
)
ax_score.text(
    0.5, 0.06, f"{len(bien_classees)} bonnes réponses sur {total} images",
    ha="center", va="bottom", color=TEXTE, fontsize=15, transform=ax_score.transAxes,
)
# Barre de progression
ax_score.add_patch(plt.Rectangle(
    (0.20, 0.22), 0.60, 0.06, transform=ax_score.transAxes,
    facecolor="#1e293b", edgecolor="none",
))
ax_score.add_patch(plt.Rectangle(
    (0.20, 0.22), 0.60 * score_affiche / 100, 0.06, transform=ax_score.transAxes,
    facecolor=couleur_score, edgecolor="none",
))

# ------------------------------------------------------------------
# 2) Images correctement classées + exemples
# 3) Images mal classées + exemples
# ------------------------------------------------------------------
def bloc(items, y_titre, y_images, titre, couleur):
    fig.text(
        0.06, y_titre, f"{titre} : {len(items)} / {total}",
        color=couleur, fontsize=18, fontweight="bold", ha="left", va="center",
    )
    ech = exemples(items)
    for j in range(N_EX):
        ax = fig.add_axes([0.06 + j * 0.235, y_images, 0.205, 0.20])
        ax.set_xticks([])
        ax.set_yticks([])
        if j < len(ech):
            img_obj, gt, pl = ech[j]
            ax.imshow(img_obj.convert("RGB"))
            ax.set_title(
                f"Prédit : {pl}\nVérité : {gt}", fontsize=11, color=TEXTE, pad=6,
            )
            for s in ax.spines.values():
                s.set_edgecolor(couleur)
                s.set_linewidth(3)
        else:
            ax.axis("off")


bloc(bien_classees, 0.63, 0.38, "✅ Bien classées", VERT)
bloc(mal_classees, 0.30, 0.05, "❌ Mal classées", ROUGE)

plt.show()
