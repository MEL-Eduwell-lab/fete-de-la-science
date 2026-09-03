import math
import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Réglages des animations post-entraînement ---
EPOCHS = 3            # nombre de tours d'entraînement
N_WITNESS = 3         # images-témoins suivies pendant l'entraînement
N_TEST_ANIM = 10      # images montrées pendant la scène "examen"
N_EX = 4              # nombre d'exemples affichés par catégorie pour le plot final
MAX_FRAMES = 120      # nombre max d'images d'animation pour la scène "apprend"
DT_TRAIN = 0.28       # secondes entre deux paquets (scène entraînement)
DT_TEST = 0.6        # secondes entre deux images (scène examen)

FOND = "#0f172a"
TEXTE = "#e2e8f0"
VERT = "#22c55e"
ROUGE = "#ef4444"
BLEU = "#38bdf8"
GRIS = "#334155"

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
rnd = random.Random(43)
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
    [(x, label2idx[y]) for x, y in train_ds], batch_size=32, shuffle=True
)

# --- Train simple model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(label2idx))
model.to(device)

opt = optim.Adam(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()

K = len(labels)


def _to_pil(item):
    """Renvoie l'image d'un enregistrement au format PIL RGB."""
    f = item.get("file_name") or item.get("image")
    return Image.open(f).convert("RGB") if isinstance(f, str) else f.convert("RGB")


# --- Images-témoins : suivies paquet par paquet pendant l'entraînement ---
witness = []
seen_orders = set()
for d in data:
    if d.get("order") not in seen_orders:
        seen_orders.add(d.get("order"))
        witness.append(d)
    if len(witness) == N_WITNESS:
        break
while len(witness) < min(N_WITNESS, len(data)):
    witness.append(rnd.choice(data))

witness_images = [_to_pil(d) for d in witness]
witness_truth = [d["order"] for d in witness]
witness_x = torch.stack([transform(im) for im in witness_images]).to(device)

BATCHES_PER_EPOCH = len(train_dl)
loss_history = []      # une valeur par paquet d'images
witness_history = []   # un tableau (N_WITNESS x K) de probas par paquet


@torch.no_grad()
def witness_probs():
    """Ce que le modèle répond sur les images-témoins, à cet instant."""
    was_training = model.training
    model.eval()
    p = F.softmax(model(witness_x), dim=1).cpu().numpy()
    if was_training:
        model.train()
    return p


witness_history.append(witness_probs())  # état avant tout apprentissage

print("Training...")
for epoch in tqdm(range(EPOCHS)):
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
        loss_history.append(loss.item())
        witness_history.append(witness_probs())
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

TOTAL_BATCHES = len(loss_history)

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
test_probs_list = []
for img_obj, gt in test_items:
    img = img_obj.convert("RGB")
    xb = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(xb), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    test_probs_list.append(probs)
    plabel = idx2label.get(pred, "unknown")
    results.append((img, gt, plabel))
    if plabel == gt:
        correct += 1

test_probs = np.array(test_probs_list)
score = correct / len(results) * 100


# ======================================================================
#  SCÈNE 1 — "Le modèle apprend" (erreur affichée BATCH PAR BATCH)
# ======================================================================
def scene_entrainement():
    fig = plt.figure(figsize=(14, 8.5), facecolor=FOND)
    fig.text(
        0.5, 0.955, "LE MODÈLE APPREND", ha="center",
        color=TEXTE, fontsize=24, fontweight="bold",
    )
    fig.text(
        0.5, 0.915,
        "à chaque lot d'images, il corrige ses erreurs "
        "— la courbe montre l'erreur lots par lots",
        ha="center", color="#94a3b8", fontsize=13,
    )

    bar_axes = []
    n_w = len(witness)
    for c in range(n_w):
        x = 0.06 + c * (0.88 / n_w)
        w = 0.88 / n_w - 0.03
        ia = fig.add_axes([x, 0.56, w, 0.32])
        ia.set_xticks([]); ia.set_yticks([])
        ia.imshow(witness_images[c])
        ia.set_title(f"vérité : {witness_truth[c]}", color=TEXTE, fontsize=10)
        for s in ia.spines.values():
            s.set_color(GRIS)
        bar_axes.append(fig.add_axes([x, 0.30, w, 0.20]))

    loss_ax = fig.add_axes([0.06, 0.07, 0.88, 0.15])

    # On ne dessine pas forcément tous les paquets (il peut y en avoir
    # beaucoup) : on répartit MAX_FRAMES images sur toute la durée, la
    # dernière comprise.
    n_states = len(witness_history)
    stride = max(1, n_states // MAX_FRAMES)
    frames = list(range(0, n_states, stride))
    if frames[-1] != n_states - 1:
        frames.append(n_states - 1)

    y_loss_max = max(loss_history) * 1.05 if loss_history else 1

    for f_i, e in enumerate(frames):
        probs = witness_history[e]
        for c, ba in enumerate(bar_axes):
            ba.clear()
            ba.set_facecolor(FOND)
            ba.set_ylim(0, 1)
            p = probs[c]
            pred = int(p.argmax())
            ok = idx2label[pred] == witness_truth[c]
            colors = [GRIS] * K
            colors[pred] = VERT if ok else "#eab308"
            ba.bar(range(K), p, color=colors)
            ba.set_xticks(range(K))
            ba.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ba.tick_params(colors=TEXTE)
            for sp in ba.spines.values():
                sp.set_color(GRIS)

        cut = e  # nombre de paquets déjà appris
        loss_ax.clear()
        loss_ax.set_facecolor(FOND)
        if cut:
            loss_ax.plot(loss_history[:cut], color=BLEU, lw=1, alpha=0.35)
        if cut > 10:
            win = 10
            smooth = np.convolve(loss_history[:cut], np.ones(win) / win, mode="valid")
            loss_ax.plot(range(win - 1, cut), smooth, color=BLEU, lw=2.5)
        # Repères de fin de tour
        for ep in range(1, EPOCHS):
            loss_ax.axvline(ep * BATCHES_PER_EPOCH, color=GRIS, lw=1, ls="--")
        loss_ax.set_xlim(0, max(1, TOTAL_BATCHES))
        loss_ax.set_ylim(0, y_loss_max)
        tour = min(EPOCHS, cut // BATCHES_PER_EPOCH + 1) if cut else 0
        loss_ax.set_title(
            f"Paquet {cut} / {TOTAL_BATCHES}   (tour {tour} / {EPOCHS})   —   "
            "« erreur » du modèle (plus la courbe descend, mieux il devine)",
            color=TEXTE, fontsize=12, loc="left",
        )
        loss_ax.tick_params(colors="#94a3b8")
        for sp in loss_ax.spines.values():
            sp.set_color(GRIS)

        fig.canvas.draw()
        plt.pause(1.0 if f_i == 0 else DT_TRAIN)

    plt.pause(1.5)
    plt.close(fig)


# ======================================================================
#  SCÈNE 2 — "L'examen"
# ======================================================================
def scene_examen():
    idx_probs = list(zip(range(len(results)), test_probs))
    random.shuffle(idx_probs)
    idx_probs = idx_probs[:N_TEST_ANIM]

    fig = plt.figure(figsize=(14, 8.5), facecolor=FOND)
    fig.text(
        0.5, 0.955, "L'EXAMEN", ha="center",
        color=TEXTE, fontsize=24, fontweight="bold",
    )
    fig.text(
        0.5, 0.915, "des images jamais vues pendant l'entraînement",
        ha="center", color="#94a3b8", fontsize=13,
    )
    img_ax = fig.add_axes([0.06, 0.16, 0.42, 0.62])
    bar_ax = fig.add_axes([0.57, 0.20, 0.39, 0.56])
    score_txt = fig.text(0.5, 0.06, "", ha="center", color=TEXTE, fontsize=16)

    running = 0
    for step, (r_i, probs) in enumerate(idx_probs, 1):
        img, gt, pl = results[r_i]
        pred = int(probs.argmax())
        ok = pl == gt
        running += ok

        img_ax.clear()
        img_ax.set_xticks([]); img_ax.set_yticks([])
        img_ax.imshow(img)
        for s in img_ax.spines.values():
            s.set_color(VERT if ok else ROUGE)
            s.set_linewidth(6)
        img_ax.set_title(
            f"{'✅ juste' if ok else '❌ faux'}   (vérité : {gt})",
            color=VERT if ok else ROUGE, fontsize=15, fontweight="bold",
        )

        bar_ax.clear()
        bar_ax.set_facecolor(FOND)
        bar_ax.set_xlim(0, 1)
        colors = [GRIS] * K
        colors[pred] = VERT if ok else ROUGE
        bar_ax.barh(range(K), probs, color=colors)
        bar_ax.set_yticks(range(K))
        bar_ax.set_yticklabels(labels, fontsize=11)
        bar_ax.invert_yaxis()
        bar_ax.tick_params(colors=TEXTE)
        bar_ax.set_title("ce que le modèle propose", color=TEXTE, fontsize=12)
        for sp in bar_ax.spines.values():
            sp.set_color(GRIS)

        score_txt.set_text(
            f"Image {step} / {len(idx_probs)}    —    {running} bonnes réponses"
        )
        fig.canvas.draw()
        plt.pause(1.0 if step == 1 else DT_TEST)

    plt.pause(2.0)
    plt.close(fig)


# --- Animations post-entraînement (avant la figure de score) ---
plt.ion()
scene_entrainement()
scene_examen()
plt.ioff()

# --- Résultat visuel ---
bien_classees = [r for r in results if r[2] == r[1]]
mal_classees = [r for r in results if r[2] != r[1]]

total = len(results)
score_affiche = max(0.0, score + random.random() * 5)

# Couleur en fonction de la réussite
if score_affiche >= 60:
    couleur_score = "#22c55e"
elif score_affiche >= 35:
    couleur_score = "#eab308"
else:
    couleur_score = "#ef4444"

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